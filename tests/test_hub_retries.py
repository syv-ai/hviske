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
    connection.send(filesystem._retry_policy.base_delay_seconds)
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
            _retry_policy=HubRetryPolicy(),
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
        hub_retries._run_with_retries(fail, HubRetryPolicy())

    assert calls == 1


def test_hub_metadata_errors_retry_past_previous_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub API metadata failures use the same indefinite retry policy."""
    calls = 0
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    def metadata_operation() -> str:
        nonlocal calls
        calls += 1
        if calls <= 8:
            raise hub_retries.HfHubHTTPError(
                "service unavailable", response=_response(503)
            )
        return "ready"

    assert (
        hub_retries._run_with_retries(
            operation=metadata_operation,
            policy=HubRetryPolicy(),
            sleep=sleeps.append,
            url="hf://datasets/example",
        )
        == "ready"
    )
    assert calls == 9
    assert len(sleeps) == 8


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
            _retry_policy=HubRetryPolicy(),
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
    assert caplog.text.count("Hugging Face Hub read failed") == 3
    assert "Hugging Face Hub read recovered after 3 retries" in caplog.text
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
        policy=HubRetryPolicy(),
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
        policy=HubRetryPolicy(),
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
            _retry_policy=HubRetryPolicy(),
        ),
        url=lambda: "https://huggingface.co/file",
        loc=17,
        response=None,
        _stream_iterator=None,
        _stream_buffer=bytearray(),
        _exit_stack=contextlib.ExitStack(),
        _close_retry_stream_context=lambda: None,
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
    def __init__(self, response: httpx.Response | list[httpx.Response]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.context = _FakeStreamContext(self.responses[0])
        self.contexts: list[_FakeStreamContext] = []
        self.headers: dict[str, str] | None = None

    def stream(self, **kwargs: object) -> _FakeStreamContext:
        self.headers = t.cast(dict[str, str], kwargs["headers"])
        response = (
            self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        )
        self.context = _FakeStreamContext(response)
        self.contexts.append(self.context)
        return self.context


def test_retry_continues_past_previous_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A persistent server error keeps retrying until Hub access recovers."""
    client = _FakeClient([*[_response(503) for _ in range(8)], _response(200)])
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    response = hub_retries._run_with_retries(
        lambda: _request_range(url="https://huggingface.co/file", headers={}),
        HubRetryPolicy(),
        sleep=sleeps.append,
        url="https://huggingface.co/file?token=secret",
    )

    assert response.status_code == 200
    assert client.calls == 9
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
    assert "retry attempt 8" in caplog.text
    assert "training will keep retrying until manually stopped" in caplog.text
    assert "token=secret" not in caplog.text


def test_retry_delay_saturates_for_arbitrarily_large_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indefinite retries cannot overflow exponential backoff arithmetic."""
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    assert HubRetryPolicy().delay(1_000_000) == 30.0


def test_retry_policy_is_available_in_a_spawned_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A focused spawned process picks up the policy without starting a DataLoader."""
    monkeypatch.delenv(hub_retries._POLICY_ENV, raising=False)
    policy = configure_hub_streaming_retries({"base_delay_seconds": 3.0})
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_read_spawned_policy, args=(child_connection,))
    process.start()
    child_connection.close()

    try:
        assert parent_connection.poll(10)
        assert parent_connection.recv() == policy.base_delay_seconds
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
        {"base_delay_seconds": 0.25, "max_delay_seconds": 2.0, "jitter_seconds": 0.1}
    )
    filesystem = hub_retries.RetryingHfFileSystem()

    assert filesystem._retry_policy == policy


def test_retry_policy_reconfiguration_invalidates_cached_filesystems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated configuration changes the policy returned by fsspec."""
    monkeypatch.delenv(hub_retries._POLICY_ENV, raising=False)
    first_policy = configure_hub_streaming_retries({"base_delay_seconds": 1.0})
    first_filesystem = fsspec.filesystem("hf")

    second_policy = configure_hub_streaming_retries({"base_delay_seconds": 4.0})
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
            policy=HubRetryPolicy(),
        ):
            raise hub_retries._RetryableStatus(
                status_code=503,
                url="https://user:password@example.test/file?token=secret",
            )

    assert error.value.status_code == 503
    assert error.value.url == "https://example.test/file"
    assert "password" not in str(error.value)
    assert "token" not in str(error.value)


def test_stream_read_continues_past_previous_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed stream keeps its offset through more than six interruptions."""
    outcomes: list[BaseException | bytes] = [
        *[httpx.ReadTimeout("interrupted") for _ in range(8)],
        b"abc",
    ]
    opens: list[None] = []
    closes: list[None] = []
    monkeypatch.setattr(hub_retries.time, "sleep", lambda _: None)
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    def read_from_stream(*_: object) -> bytes:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    stream_file = SimpleNamespace(
        fs=SimpleNamespace(
            _retry_policy=HubRetryPolicy(
                base_delay_seconds=0.0, max_delay_seconds=0.0, jitter_seconds=0.0
            )
        ),
        response=SimpleNamespace(close=lambda: closes.append(None)),
        _stream_iterator=object(),
        _read_from_stream=read_from_stream,
        _open_connection=lambda: opens.append(None),
        url=lambda: "https://huggingface.co/file?token=secret",
        loc=17,
    )

    result = hub_retries.RetryingHfFileSystemStreamFile.read(
        t.cast(hub_retries.RetryingHfFileSystemStreamFile, stream_file)
    )

    assert result == b"abc"
    assert stream_file.loc == 20
    assert len(opens) == 8
    assert len(closes) == 8


def test_stream_reconnect_replaces_context_without_accumulating_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indefinite reconnects retain one stream context and cleanup callback."""

    class TestStreamFile(hub_retries.RetryingHfFileSystemStreamFile):
        def url(self) -> str:
            return "https://huggingface.co/file"

    client = _FakeStreamingClient([_response(200) for _ in range(3)])
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    stream_file = TestStreamFile.__new__(TestStreamFile)
    object.__setattr__(
        stream_file,
        "fs",
        SimpleNamespace(
            _api=SimpleNamespace(_build_hf_headers=lambda: {}),
            _retry_policy=HubRetryPolicy(),
        ),
    )
    stream_file.loc = 0
    stream_file.response = None
    stream_file._stream_buffer = bytearray()
    stream_file._stream_iterator = None
    stream_file._exit_stack = contextlib.ExitStack()

    for _ in range(3):
        stream_file._open_connection()

    assert len(stream_file._exit_stack._exit_callbacks) == 1
    assert [context.closed for context in client.contexts] == [True, True, False]
    stream_file._exit_stack.close()
    assert all(context.closed for context in client.contexts)


def test_stream_status_failure_preserves_request_context(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Stream-opening retries log status and only sanitised request context."""
    client = _FakeStreamingClient([_response(503), _response(200)])
    monkeypatch.setattr(hub_retries, "get_session", lambda: client)
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    with hub_retries._stream_with_retries(
        url="https://user:password@example.test/file?token=secret",
        headers={},
        policy=HubRetryPolicy(),
    ):
        pass

    assert "HTTP 503 from https://example.test/file" in caplog.text
    assert "password" not in caplog.text
    assert "token" not in caplog.text


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
        HubRetryPolicy(),
    )

    assert response.content == b"ok"
    assert client.calls == 2
    assert sleeps == [1.0]
    response.close()


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_wrapped_deterministic_hub_errors_fail_immediately(status_code: int) -> None:
    """Transformers wrappers do not make deterministic Hub errors retryable."""
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise _wrapped_hub_error(status_code)

    with pytest.raises(OSError):
        hub_retries._run_with_retries(
            operation=fail,
            policy=HubRetryPolicy(),
            sleep=lambda _: pytest.fail("deterministic errors must not sleep"),
        )

    assert calls == 1


def _wrapped_hub_error(status_code: int) -> OSError:
    hub_error = hub_retries.HfHubHTTPError(
        "wrapped Hub failure", response=_response(status_code)
    )
    wrapper = OSError("Transformers could not load the Hub resource")
    wrapper.__cause__ = hub_error
    return wrapper


@pytest.mark.parametrize("status_code", [408, 429, 503])
def test_wrapped_transient_hub_errors_retry_until_recovery(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    """Transformers wrappers preserve indefinite transient Hub retries."""
    calls = 0
    sleeps: list[float] = []
    monkeypatch.setattr(hub_retries.random, "uniform", lambda *_: 0.0)

    def recover() -> str:
        nonlocal calls
        calls += 1
        if calls <= 8:
            raise _wrapped_hub_error(status_code)
        return "ready"

    assert (
        hub_retries._run_with_retries(
            operation=recover,
            policy=HubRetryPolicy(),
            sleep=sleeps.append,
            url="hf://models/example",
        )
        == "ready"
    )
    assert calls == 9
    assert len(sleeps) == 8
