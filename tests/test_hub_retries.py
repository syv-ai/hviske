"""Regression tests for project-owned Hub streaming retries."""

import typing as t
from types import SimpleNamespace

import httpx
import pytest

import hviske.hub_retries as hub_retries
from hviske.hub_retries import (
    HubRetryPolicy,
    RetryingHfFileSystemFile,
    _request_range,
    configure_hub_streaming_retries,
)


def test_deterministic_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema and other deterministic failures pass through immediately."""
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("invalid parquet schema")

    monkeypatch.setattr(hub_retries.time, "sleep", pytest.fail)

    with pytest.raises(ValueError, match="invalid parquet schema"):
        hub_retries._run_with_retries(fail, HubRetryPolicy(max_retries=5))

    assert calls == 1


def test_range_read_retries_closed_client_and_5xx(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A closed client and a server outage are retried without exposing the URL."""
    client = _FakeClient(
        [
            RuntimeError("Cannot send a request, as the client has been closed."),
            _response(503),
            _response(200, b"parquet bytes"),
        ]
    )
    closed_clients: list[None] = []
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    monkeypatch.setattr(
        hub_retries, "close_session", lambda: closed_clients.append(None)
    )
    monkeypatch.setattr(hub_retries.time, "sleep", sleeps.append)
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    remote_file = object.__new__(RetryingHfFileSystemFile)
    object.__setattr__(
        remote_file,
        "fs",
        SimpleNamespace(
            _api=SimpleNamespace(_build_hf_headers=lambda: {}),
            _retry_policy=HubRetryPolicy(max_retries=3),
        ),
    )
    object.__setattr__(
        remote_file,
        "url",
        lambda: "https://huggingface.co/dataset/file?X-Amz-Signature=secret",
    )

    assert remote_file._fetch_range(start=0, end=13) == b"parquet bytes"
    assert client.calls == 3
    assert len(closed_clients) == 1
    assert sleeps == [1.0, 2.0]
    assert "X-Amz-Signature=secret" not in caplog.text


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def request(self, **_: object) -> httpx.Response:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return t.cast(httpx.Response, outcome)


def _response(status_code: int, content: bytes = b"") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=content,
        request=httpx.Request("GET", "https://huggingface.co/dataset/file"),
    )


def test_retry_policy_is_used_by_spawnable_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly constructed Hub filesystem receives the configured worker policy."""
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)
    monkeypatch.delenv(hub_retries._POLICY_ENV, raising=False)
    policy = configure_hub_streaming_retries(
        {
            "max_retries": 2,
            "base_delay_seconds": 0.25,
            "max_delay_seconds": 2.0,
            "jitter_seconds": 0.1,
        }
    )
    filesystem = hub_retries.RetryingHfFileSystem()

    assert filesystem._retry_policy == policy


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("timed out"), ConnectionResetError("reset")]
)
def test_transport_failures_are_retried(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    """Timeouts and connection resets receive the same bounded retry treatment."""
    client = _FakeClient([failure, _response(200, b"ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    monkeypatch.setattr(hub_retries.time, "sleep", sleeps.append)
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    response = hub_retries._run_with_retries(
        lambda: _request_range(url="https://huggingface.co/file", headers={}),
        HubRetryPolicy(max_retries=1),
    )

    assert response.content == b"ok"
    assert client.calls == 2
    assert sleeps == [1.0]
    response.close()
