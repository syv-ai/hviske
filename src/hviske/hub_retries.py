"""Interruptible retries for Hugging Face Hub streaming file reads."""

from __future__ import annotations

import contextlib
import dataclasses
import email.utils
import json
import logging
import math
import os
import random
import time
import typing as t
from collections.abc import Callable, Generator, Mapping
from urllib.parse import urlsplit, urlunsplit

import fsspec
import httpx
from huggingface_hub import HfFileSystem, constants
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_file_system import (
    HfFileSystemFile,
    HfFileSystemStreamFile,
    hf_raise_for_status,
)
from huggingface_hub.utils import close_session, get_session

from .dataloader_shutdown import (
    exit_worker_if_shutdown_requested,
    interruptible_retry_delay,
    raise_or_exit_worker,
    shutdown_requested,
    start_worker_shutdown_watcher,
)
from .hub_access_health import clear_hub_access_retrying, mark_hub_access_retrying

Result = t.TypeVar("Result")

logger = logging.getLogger(__name__)

_POLICY_ENV = "HVISKE_HUB_STREAMING_RETRY_POLICY"
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 499})


@dataclasses.dataclass(frozen=True)
class HubRetryPolicy:
    """Delays for indefinite remote Hub file-read retries.

    Args:
        base_delay_seconds:
            Initial exponential backoff delay. Defaults to 1.0.
        max_delay_seconds:
            Upper bound for the exponential backoff delay. Defaults to 30.0.
        jitter_seconds:
            Maximum additional random delay. Defaults to 0.5.
        rate_limit_fallback_seconds:
            Delay for a 429 response without a valid ``Retry-After`` header.
            Defaults to 120.0.
        rate_limit_max_delay_seconds:
            Upper bound for a 429 delay, including a ``Retry-After`` value.
            Defaults to 300.0.
    """

    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_seconds: float = 0.5
    rate_limit_fallback_seconds: float = 120.0
    rate_limit_max_delay_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate the retry policy.

        Raises:
            ValueError:
                If a retry delay is invalid.
        """
        if not all(
            math.isfinite(value)
            for value in (
                self.base_delay_seconds,
                self.max_delay_seconds,
                self.jitter_seconds,
                self.rate_limit_fallback_seconds,
                self.rate_limit_max_delay_seconds,
            )
        ):
            raise ValueError("hub retry delays must be finite")
        if self.base_delay_seconds < 0:
            raise ValueError("hub retry base_delay_seconds must be non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                "hub retry max_delay_seconds must be at least base_delay_seconds"
            )
        if self.jitter_seconds < 0:
            raise ValueError("hub retry jitter_seconds must be non-negative")
        if self.rate_limit_fallback_seconds < 0:
            raise ValueError(
                "hub retry rate_limit_fallback_seconds must be non-negative"
            )
        if self.rate_limit_max_delay_seconds < self.rate_limit_fallback_seconds:
            raise ValueError(
                "hub retry rate_limit_max_delay_seconds must be at least "
                "rate_limit_fallback_seconds"
            )

    def delay(self, retry_number: int) -> float:
        """Return a bounded exponential delay with additive jitter.

        Args:
            retry_number:
                Zero-based retry number, where zero is the first retry.
        """
        if self.base_delay_seconds == 0:
            exponential = 0.0
        elif self.max_delay_seconds == self.base_delay_seconds:
            exponential = self.max_delay_seconds
        else:
            maximum_exponent = math.ceil(
                math.log2(self.max_delay_seconds / self.base_delay_seconds)
            )
            exponent = min(retry_number, maximum_exponent)
            exponential = min(
                self.max_delay_seconds, self.base_delay_seconds * (2**exponent)
            )
        return min(
            self.max_delay_seconds,
            exponential + random.uniform(0.0, self.jitter_seconds),
        )


class _RetryableStatus(Exception):
    """An HTTP status that is safe to retry, with safe request context."""

    def __init__(
        self, status_code: int, url: str, retry_after: str | None = None
    ) -> None:
        self.status_code = status_code
        self.url = _sanitise_url(url)
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code} from {self.url}")


def _sanitise_url(url: str) -> str:
    """Remove URL credentials, query parameters, and fragments.

    Returns:
        A URL containing only its scheme, host, port, and path.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "<invalid URL>"
    return urlunsplit(
        (parsed.scheme, parsed.netloc.rsplit("@", maxsplit=1)[-1], parsed.path, "", "")
    )


_DEFAULT_POLICY = HubRetryPolicy()


class RetryingHfFileSystem(HfFileSystem):
    """Hugging Face filesystem whose reads use the project retry policy."""

    def __init__(
        self,
        endpoint: str | None = None,
        token: bool | str | None = None,
        block_size: int | None = None,
        expand_info: bool | None = None,
        **storage_options: object,
    ) -> None:
        """Create a filesystem using the process-local retry policy."""
        super().__init__(
            endpoint=endpoint,
            token=token,
            block_size=block_size,
            expand_info=expand_info,
            **storage_options,
        )
        self._retry_policy = _policy_from_environment()

    def _get_instance_state(self) -> dict[str, object]:
        state = super()._get_instance_state()
        state["_retry_policy"] = self._retry_policy
        return state

    def _open(  # type: ignore[override]
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        revision: str | None = None,
        **kwargs: object,
    ) -> HfFileSystemFile | HfFileSystemStreamFile:
        effective_block_size = block_size if block_size is not None else self.block_size

        def create_file() -> HfFileSystemFile | HfFileSystemStreamFile:
            if effective_block_size == 0:
                stream_block_size = t.cast(int, kwargs.pop("block_size", 0))
                cache_type = t.cast(str, kwargs.pop("cache_type", "none"))
                return RetryingHfFileSystemStreamFile(
                    self,
                    path,
                    mode=mode,
                    revision=revision,
                    block_size=stream_block_size,
                    cache_type=cache_type,
                    **kwargs,
                )
            return RetryingHfFileSystemFile(
                self,
                path,
                mode=mode,
                block_size=effective_block_size,
                revision=revision,
                **kwargs,
            )

        return _run_with_retries(
            operation=create_file, policy=self._retry_policy, url=f"hf://{path}"
        )


class RetryingHfFileSystemFile(HfFileSystemFile):
    """Range-reading Hub file using the configured retry policy."""

    def _fetch_range(self, start: int, end: int) -> bytes:
        exit_worker_if_shutdown_requested()
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        request_url = _run_with_retries(
            operation=self.url, policy=self.fs._retry_policy, url="hf://resolved-file"
        )
        response = _run_with_retries(
            lambda: _request_range(url=request_url, headers=headers),
            policy=self.fs._retry_policy,
            url=request_url,
        )
        try:
            hf_raise_for_status(response)
            return response.content
        finally:
            response.close()


def _request_range(url: str, headers: Mapping[str, str]) -> httpx.Response:
    response = get_session().request(
        method="GET",
        url=url,
        headers=dict(headers),
        timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
    )
    if _is_retryable_status(response.status_code):
        status_code = response.status_code
        retry_after = response.headers.get("Retry-After")
        response.close()
        raise _RetryableStatus(
            status_code=status_code, url=url, retry_after=retry_after
        )
    return response


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code <= 599


def _run_with_retries(
    operation: Callable[[], Result],
    policy: HubRetryPolicy,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    url: str | None = None,
) -> Result:
    sleeper = time.sleep if sleep is None else sleep
    now = time.time if clock is None else clock
    request_url = None if url is None else _sanitise_url(url)
    retry_number = 0
    retrying = False
    start_worker_shutdown_watcher()
    try:
        while True:
            exit_worker_if_shutdown_requested()
            try:
                result = operation()
            except _RetryableStatus as error:
                caught_error: BaseException = error
            except BaseException as error:
                if not _is_retryable_error(error):
                    raise
                caught_error = error
                if "client has been closed" in str(error).lower():
                    close_session()
            else:
                if retrying:
                    logger.info(
                        "Hugging Face Hub read recovered after %d retries (%s)",
                        retry_number,
                        request_url or "URL unavailable",
                    )
                return result

            if shutdown_requested():
                raise_or_exit_worker(error=caught_error)
            if not retrying:
                mark_hub_access_retrying()
                retrying = True
            delay = _retry_delay(
                error=caught_error, retry_number=retry_number, policy=policy, now=now
            )
            logger.warning(
                "Hugging Face Hub read failed (%s); retry attempt %d in %.2fs; "
                "training will keep retrying until manually stopped",
                _retry_context(error=caught_error, fallback_url=request_url),
                retry_number + 1,
                delay,
            )
            interruptible_retry_delay(delay=delay, error=caught_error, sleep=sleeper)
            if shutdown_requested():
                raise_or_exit_worker(error=caught_error)
            retry_number += 1
    finally:
        if retrying:
            clear_hub_access_retrying()


def _is_retryable_error(error: BaseException) -> bool:
    """Return whether an exception chain contains a transient Hub failure."""
    for candidate in _exception_chain(error):
        if isinstance(candidate, HfHubHTTPError):
            response = getattr(candidate, "response", None)
            return response is not None and _is_retryable_status(response.status_code)
        if isinstance(candidate, httpx.RemoteProtocolError):
            # A peer closing an HTTP stream is transient, unlike local protocol errors.
            return True
        if isinstance(
            candidate,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                ConnectionError,
                ConnectionResetError,
            ),
        ):
            return True
        if (
            isinstance(candidate, RuntimeError)
            and "client has been closed" in str(candidate).lower()
        ):
            return True
    return False


def _exception_chain(error: BaseException) -> Generator[BaseException, None, None]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _retry_context(error: BaseException, fallback_url: str | None) -> str:
    if isinstance(error, _RetryableStatus):
        return f"HTTP {error.status_code} from {error.url}"
    hub_error = _hub_http_error(error)
    if hub_error is not None:
        response = hub_error.response
        try:
            request_url = _sanitise_url(str(response.request.url))
        except RuntimeError:
            request_url = fallback_url or "URL unavailable"
        return f"HTTP {response.status_code} from {request_url}"
    category = type(error).__name__
    if fallback_url is None:
        return category
    return f"{category} from {fallback_url}"


def _hub_http_error(error: BaseException) -> HfHubHTTPError | None:
    return next(
        (
            candidate
            for candidate in _exception_chain(error)
            if isinstance(candidate, HfHubHTTPError)
        ),
        None,
    )


def _retry_delay(
    error: BaseException,
    retry_number: int,
    policy: HubRetryPolicy,
    now: Callable[[], float],
) -> float:
    if isinstance(error, _RetryableStatus):
        status_code = error.status_code
        retry_after_header = error.retry_after
    else:
        hub_error = _hub_http_error(error)
        if hub_error is None:
            return policy.delay(retry_number)
        status_code = hub_error.response.status_code
        retry_after_header = hub_error.response.headers.get("Retry-After")
    if status_code != 429:
        return policy.delay(retry_number)
    retry_after = _parse_retry_after(retry_after_header, now=now())
    if retry_after is None:
        retry_after = policy.rate_limit_fallback_seconds
    return min(retry_after, policy.rate_limit_max_delay_seconds)


def _parse_retry_after(value: str | None, now: float) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
        if delay < 0:
            return None
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        delay = retry_at - now
    if not math.isfinite(delay):
        return None
    return max(0.0, delay)


class RetryingHfFileSystemStreamFile(HfFileSystemStreamFile):
    """Streaming Hub file with retries for opening and resuming connections."""

    def read(self, length: int = -1) -> bytes:
        """Read from the stream, reopening it after transient interruptions.

        Returns:
            Bytes read from the remote stream.
        """
        exit_worker_if_shutdown_requested()
        start_worker_shutdown_watcher()
        if self.response is None:
            self._open_connection()

        retry_number = 0
        retrying = False
        request_url = _sanitise_url(
            _run_with_retries(
                operation=self.url,
                policy=self.fs._retry_policy,
                url="hf://resolved-stream",
            )
        )
        try:
            while True:
                try:
                    if self.response is None or self._stream_iterator is None:
                        return b""
                    out = self._read_from_stream(self._stream_iterator, length)
                    self.loc += len(out)
                    if retrying:
                        logger.info(
                            "Hugging Face Hub stream recovered after %d retries (%s)",
                            retry_number,
                            request_url,
                        )
                    return out
                except BaseException as error:
                    if not _is_retryable_error(error):
                        raise
                    if self.response is not None:
                        self.response.close()
                    if shutdown_requested():
                        raise_or_exit_worker(error=error)
                    if "client has been closed" in str(error).lower():
                        close_session()
                    if not retrying:
                        mark_hub_access_retrying()
                        retrying = True
                    delay = self.fs._retry_policy.delay(retry_number)
                    logger.warning(
                        "Hugging Face Hub stream failed (%s); retry attempt %d in "
                        "%.2fs; training will keep retrying until manually stopped",
                        _retry_context(error=error, fallback_url=request_url),
                        retry_number + 1,
                        delay,
                    )
                    interruptible_retry_delay(
                        delay=delay, error=error, sleep=time.sleep
                    )
                    if shutdown_requested():
                        raise_or_exit_worker(error=error)
                    retry_number += 1
                    self._open_connection()
        finally:
            if retrying:
                clear_hub_access_retrying()

    def _open_connection(self) -> None:
        self._close_retry_stream_context()
        self._stream_buffer.clear()
        self._stream_iterator = None
        headers = self.fs._api._build_hf_headers()
        if self.loc > 0:
            headers["Range"] = f"bytes={self.loc}-"
        request_url = _run_with_retries(
            operation=self.url, policy=self.fs._retry_policy, url="hf://resolved-stream"
        )
        try:
            context = _stream_with_retries(
                url=request_url, headers=headers, policy=self.fs._retry_policy
            )
            self.response = context.__enter__()
            self._retry_stream_context = context
            if not getattr(self, "_retry_stream_cleanup_registered", False):
                self._exit_stack.callback(self._close_retry_stream_context)
                self._retry_stream_cleanup_registered = True
        except HfHubHTTPError as error:
            if self.loc > 0 and error.response.status_code == 416:
                # Match HfFileSystemStreamFile: an exhausted resumed range is EOF.
                self.response = None
                return
            raise
        self._stream_iterator = self.response.iter_bytes()

    def _close_retry_stream_context(self) -> None:
        context = getattr(self, "_retry_stream_context", None)
        if context is None:
            return
        self._retry_stream_context = None
        context.__exit__(None, None, None)


@contextlib.contextmanager
def _stream_with_retries(
    url: str, headers: Mapping[str, str], policy: HubRetryPolicy
) -> Generator[httpx.Response, None, None]:
    context, response = _run_with_retries(
        lambda: _open_stream(url=url, headers=headers), policy=policy, url=url
    )
    try:
        yield response
    finally:
        context.__exit__(None, None, None)


def _open_stream(
    url: str, headers: Mapping[str, str]
) -> tuple[contextlib.AbstractContextManager[httpx.Response], httpx.Response]:
    context = get_session().stream(
        method="GET",
        url=url,
        headers=dict(headers),
        timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
    )
    try:
        response = context.__enter__()
        if _is_retryable_status(response.status_code):
            raise _RetryableStatus(
                status_code=response.status_code,
                url=url,
                retry_after=response.headers.get("Retry-After"),
            )
        hf_raise_for_status(response)
    except BaseException:
        context.__exit__(None, None, None)
        raise
    return context, response


def _policy_from_environment() -> HubRetryPolicy:
    encoded = os.getenv(_POLICY_ENV)
    if not encoded:
        return _DEFAULT_POLICY
    try:
        values = json.loads(encoded)
        if not isinstance(values, dict):
            return _DEFAULT_POLICY
        return _policy_from_config(t.cast(dict[str, object], values))
    except (TypeError, ValueError, json.JSONDecodeError):
        return _DEFAULT_POLICY


def _policy_from_config(retry_config: Mapping[str, object] | None) -> HubRetryPolicy:
    if retry_config is None:
        return _policy_from_environment()
    base_delay_seconds = t.cast(
        float | int,
        retry_config.get("base_delay_seconds", _DEFAULT_POLICY.base_delay_seconds),
    )
    max_delay_seconds = t.cast(
        float | int,
        retry_config.get("max_delay_seconds", _DEFAULT_POLICY.max_delay_seconds),
    )
    jitter_seconds = t.cast(
        float | int, retry_config.get("jitter_seconds", _DEFAULT_POLICY.jitter_seconds)
    )
    rate_limit_fallback_seconds = t.cast(
        float | int,
        retry_config.get(
            "rate_limit_fallback_seconds", _DEFAULT_POLICY.rate_limit_fallback_seconds
        ),
    )
    rate_limit_max_delay_seconds = t.cast(
        float | int,
        retry_config.get(
            "rate_limit_max_delay_seconds", _DEFAULT_POLICY.rate_limit_max_delay_seconds
        ),
    )
    return HubRetryPolicy(
        base_delay_seconds=float(base_delay_seconds),
        max_delay_seconds=float(max_delay_seconds),
        jitter_seconds=float(jitter_seconds),
        rate_limit_fallback_seconds=float(rate_limit_fallback_seconds),
        rate_limit_max_delay_seconds=float(rate_limit_max_delay_seconds),
    )


def configure_hub_streaming_retries(
    retry_config: Mapping[str, object] | None = None,
) -> HubRetryPolicy:
    """Install the configured Hub filesystem for this process and its children.

    The policy is copied to an environment variable so a freshly spawned DataLoader
    worker can construct the same filesystem without inheriting an HTTP client. Only
    the ``hf`` fsspec protocol is changed; local paths are untouched.

    Args:
        retry_config (optional):
            Mapping containing fields accepted by :class:`HubRetryPolicy`.
            Defaults to the conservative project defaults.

    Returns:
        The validated policy used for subsequent Hub filesystem instances.
    """
    policy = _policy_from_config(retry_config)
    os.environ[_POLICY_ENV] = json.dumps(dataclasses.asdict(policy), sort_keys=True)
    RetryingHfFileSystem.clear_instance_cache()
    fsspec.register_implementation("hf", RetryingHfFileSystem, clobber=True)
    return policy


def retry_hub_access(operation: Callable[[], Result], url: str | None = None) -> Result:
    """Run a Hub operation until it succeeds or manual shutdown is requested.

    Deterministic errors, including authentication and missing-resource responses,
    still fail immediately.

    Args:
        operation:
            Hub operation to execute.
        url (optional):
            Credential-free request context for retry logs. Defaults to unavailable.

    Returns:
        The operation result.
    """
    return _run_with_retries(
        operation=operation, policy=_policy_from_environment(), url=url
    )


# Register at import time so a fresh spawn worker resolves ``hf://`` paths to the
# project-owned implementation before it starts reading a pickled streaming dataset.
configure_hub_streaming_retries()
