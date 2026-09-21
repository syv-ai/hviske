"""Regression tests for project-owned Hub streaming retries."""

import contextlib
import email.utils
import multiprocessing as mp
import typing as t
from multiprocessing.connection import Connection
from types import SimpleNamespace

import fsspec
import httpx
import pytest

import hviske.hub_retries as hub_retries
from hviske.hub_retries import (
    HubRetryPolicy,
    RetryingHfFileSystemFile,
    _request_range,
    configure_hub_streaming_retries,
)


def _read_spawned_policy(connection: Connection) -> None:
    filesystem = fsspec.filesystem("hf")
    connection.send(filesystem._retry_policy.max_retries)
    connection.close()


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_auth_and_not_found_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    """Authentication and missing-file responses fail deterministically."""
    client = _FakeClient([_response(status_code)])
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    monkeypatch.setattr(hub_retries.time, "sleep", sleeps.append)

    remote_file = SimpleNamespace(
        fs=SimpleNamespace(
            _api=SimpleNamespace(_build_hf_headers=lambda: {}),
            _retry_policy=HubRetryPolicy(max_retries=5),
        ),
        url=lambda: "https://huggingface.co/file",
    )

    with pytest.raises(httpx.HTTPError):
        RetryingHfFileSystemFile._fetch_range(
            t.cast(RetryingHfFileSystemFile, remote_file), start=0, end=1
        )

    assert client.calls == 1
    assert sleeps == []


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


def _response(
    status_code: int, content: bytes = b"", headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=httpx.Request("GET", "https://huggingface.co/dataset/file"),
    )


@pytest.mark.parametrize(
    "failure",
    [ValueError("invalid parquet schema"), httpx.LocalProtocolError("invalid request")],
)
def test_deterministic_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    """Schema and local protocol failures pass through immediately."""
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(hub_retries.time, "sleep", pytest.fail)

    with pytest.raises(type(failure), match=str(failure)):
        hub_retries._run_with_retries(fail, HubRetryPolicy(max_retries=5))

    assert calls == 1


def test_range_read_retries_closed_client_and_remote_failures(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Transient remote failures are retried without exposing the URL."""
    client = _FakeClient(
        [
            RuntimeError("Cannot send a request, as the client has been closed."),
            _response(499),
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

    remote_file = SimpleNamespace(
        fs=SimpleNamespace(
            _api=SimpleNamespace(_build_hf_headers=lambda: {}),
            _retry_policy=HubRetryPolicy(max_retries=4),
        ),
        url=lambda: "https://huggingface.co/dataset/file?X-Amz-Signature=secret",
    )

    assert (
        RetryingHfFileSystemFile._fetch_range(
            t.cast(RetryingHfFileSystemFile, remote_file), start=0, end=13
        )
        == b"parquet bytes"
    )
    assert client.calls == 4
    assert len(closed_clients) == 1
    assert sleeps == [1.0, 2.0, 4.0]
    assert caplog.text.count("Transient Hugging Face Hub read failed") == 3
    assert "HTTP 499 from https://huggingface.co/dataset/file" in caplog.text
    assert "X-Amz-Signature=secret" not in caplog.text


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [("45", 45.0), (email.utils.formatdate(1_700_000_060, usegmt=True), 60.0)],
)
def test_rate_limit_retry_after_is_honoured(
    monkeypatch: pytest.MonkeyPatch, retry_after: str, expected_delay: float
) -> None:
    """429 retries use both supported Retry-After formats."""
    client = _FakeClient(
        [_response(429, headers={"Retry-After": retry_after}), _response(200, b"ok")]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)

    response = hub_retries._run_with_retries(
        lambda: _request_range(
            url="https://user:password@example.test/file?token=secret", headers={}
        ),
        policy=HubRetryPolicy(max_retries=1),
        sleep=sleeps.append,
        clock=lambda: 1_700_000_000,
        url="https://user:password@example.test/file?token=secret",
    )

    assert response.content == b"ok"
    assert sleeps == [expected_delay]
    response.close()


def test_rate_limit_retry_without_retry_after_uses_long_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed Retry-After uses a bounded delay longer than normal backoff."""
    client = _FakeClient(
        [_response(429, headers={"Retry-After": "not a delay"}), _response(200, b"ok")]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)

    response = hub_retries._run_with_retries(
        lambda: _request_range(url="https://huggingface.co/file", headers={}),
        policy=HubRetryPolicy(max_retries=1),
        sleep=sleeps.append,
    )

    assert response.content == b"ok"
    assert sleeps == [120.0]
    response.close()


def test_resumed_stream_416_is_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 416 while resuming a stream means the file is already exhausted."""
    client = _FakeStreamingClient(_response(416))
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)

    stream_file = SimpleNamespace(
        fs=SimpleNamespace(
            _api=SimpleNamespace(_build_hf_headers=lambda: {}),
            _retry_policy=HubRetryPolicy(max_retries=2),
        ),
        url=lambda: "https://huggingface.co/file",
        loc=17,
        response=None,
        _stream_iterator=None,
        _stream_buffer=bytearray(),
        _exit_stack=contextlib.ExitStack(),
    )
    stream_file._open_connection = lambda: (
        hub_retries.RetryingHfFileSystemStreamFile._open_connection(
            t.cast(hub_retries.RetryingHfFileSystemStreamFile, stream_file)
        )
    )

    assert (
        hub_retries.RetryingHfFileSystemStreamFile.read(
            t.cast(hub_retries.RetryingHfFileSystemStreamFile, stream_file)
        )
        == b""
    )

    assert stream_file.response is None
    assert stream_file._stream_iterator is None
    assert client.headers == {"Range": "bytes=17-"}
    assert client.context.closed


class _FakeStreamContext:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.closed = False

    def __enter__(self) -> httpx.Response:
        return self.response

    def __exit__(self, *_: object) -> None:
        self.closed = True


class _FakeStreamingClient:
    def __init__(self, response: httpx.Response) -> None:
        self.context = _FakeStreamContext(response)
        self.headers: dict[str, str] | None = None

    def stream(self, **kwargs: object) -> _FakeStreamContext:
        self.headers = t.cast(dict[str, str], kwargs["headers"])
        return self.context


def test_retry_exhaustion_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persistent server error stops after the configured number of retries."""
    client = _FakeClient([_response(503), _response(503), _response(503)])
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    monkeypatch.setattr(hub_retries.time, "sleep", sleeps.append)
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    with pytest.raises(hub_retries._RetryableStatus) as error:
        hub_retries._run_with_retries(
            lambda: _request_range(url="https://huggingface.co/file", headers={}),
            HubRetryPolicy(max_retries=2),
        )

    assert error.value.status_code == 503
    assert error.value.url == "https://huggingface.co/file"
    assert "HTTP 503 from https://huggingface.co/file" in str(error.value)
    assert client.calls == 3
    assert sleeps == [1.0, 2.0]


def test_retry_policy_is_available_in_a_spawned_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A focused spawned process picks up the policy without starting a DataLoader."""
    monkeypatch.delenv(hub_retries._POLICY_ENV, raising=False)
    policy = configure_hub_streaming_retries({"max_retries": 3})
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_read_spawned_policy, args=(child_connection,))
    process.start()
    child_connection.close()

    try:
        assert parent_connection.poll(10)
        assert parent_connection.recv() == policy.max_retries
    finally:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
        parent_connection.close()

    assert process.exitcode == 0


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


def test_retry_policy_reconfiguration_invalidates_cached_filesystems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated configuration changes the policy returned by fsspec."""
    monkeypatch.delenv(hub_retries._POLICY_ENV, raising=False)
    first_policy = configure_hub_streaming_retries({"max_retries": 1})
    first_filesystem = fsspec.filesystem("hf")

    second_policy = configure_hub_streaming_retries({"max_retries": 4})
    second_filesystem = fsspec.filesystem("hf")

    assert first_filesystem is not second_filesystem
    assert first_filesystem._retry_policy == first_policy
    assert second_filesystem._retry_policy == second_policy


def test_retryable_status_survives_context_manager_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status errors remain assignable when a generator context propagates them.

    Raises:
        _RetryableStatus:
            The status deliberately propagated through the context manager.
    """
    client = _FakeStreamingClient(_response(200))
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)

    with pytest.raises(hub_retries._RetryableStatus) as error:
        with hub_retries._stream_with_retries(
            url="https://user:password@example.test/file?token=secret",
            headers={},
            policy=HubRetryPolicy(max_retries=0),
        ):
            raise hub_retries._RetryableStatus(
                status_code=503,
                url="https://user:password@example.test/file?token=secret",
            )

    assert error.value.status_code == 503
    assert error.value.url == "https://example.test/file"
    assert "password" not in str(error.value)
    assert "token" not in str(error.value)


def test_stream_status_failure_preserves_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream opening errors retain status and sanitised URL after exhaustion."""
    client = _FakeStreamingClient(_response(503))
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)

    with pytest.raises(hub_retries._RetryableStatus) as error:
        with hub_retries._stream_with_retries(
            url="https://user:password@example.test/file?token=secret",
            headers={},
            policy=HubRetryPolicy(max_retries=0),
        ):
            pass

    assert error.value.status_code == 503
    assert error.value.url == "https://example.test/file"
    assert "password" not in str(error.value)
    assert "token" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("timed out"),
        httpx.RemoteProtocolError("peer closed the connection"),
        ConnectionResetError("reset"),
    ],
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
