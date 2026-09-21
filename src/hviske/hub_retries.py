"""Bounded retries for Hugging Face Hub streaming file reads."""

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
)

Result = t.TypeVar("Result")

logger = logging.getLogger(__name__)

_POLICY_ENV = "HVISKE_HUB_STREAMING_RETRY_POLICY"
_RETRYABLE_STATUS_CODES = frozenset({429, 499})


@dataclasses.dataclass(frozen=True)
class HubRetryPolicy:
    """Retry limits and delays for remote Hub file reads.

    Args:
        max_retries:
            Number of retries after the initial request. Defaults to 6.
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

    max_retries: int = 6
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_seconds: float = 0.5
    rate_limit_fallback_seconds: float = 120.0
    rate_limit_max_delay_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate the retry policy.

        Raises:
            ValueError:
                If a retry bound is invalid.
        """
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("hub retry max_retries must be a non-negative integer")
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
        exponential = min(
            self.max_delay_seconds, self.base_delay_seconds * (2**retry_number)
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


class RetryingHfFileSystemFile(HfFileSystemFile):
    """Range-reading Hub file using the configured retry policy."""

    def _fetch_range(self, start: int, end: int) -> bytes:
        exit_worker_if_shutdown_requested()
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        response = _run_with_retries(
            lambda: _request_range(url=self.url(), headers=headers),
            policy=self.fs._retry_policy,
            url=self.url(),
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
    for retry_number in range(policy.max_retries + 1):
        exit_worker_if_shutdown_requested()
        caught_error: BaseException | None = None
        try:
            return operation()
        except _RetryableStatus as error:
            caught_error = error
        except BaseException as error:
            if not _is_retryable_error(error):
                raise
            caught_error = error
            if shutdown_requested():
                raise_or_exit_worker(error=caught_error)
            if "client has been closed" in str(error).lower():
                close_session()
        if caught_error is None:
            raise AssertionError("retry loop did not capture its transient error")
        if shutdown_requested():
            raise_or_exit_worker(error=caught_error)
        if retry_number == policy.max_retries:
            raise caught_error
        delay = _retry_delay(
            error=caught_error, retry_number=retry_number, policy=policy, now=now
        )
        if isinstance(caught_error, _RetryableStatus):
            logger.warning(
                "Transient Hugging Face Hub read failed (HTTP %d from %s); "
                "retrying in %.2fs (%d/%d)",
                caught_error.status_code,
                caught_error.url,
                delay,
                retry_number + 1,
                policy.max_retries,
            )
        elif request_url is None:
            logger.warning(
                "Transient Hugging Face Hub read failed; retrying in %.2fs (%d/%d)",
                delay,
                retry_number + 1,
                policy.max_retries,
            )
        else:
            logger.warning(
                "Transient Hugging Face Hub read failed (%s); retrying in %.2fs "
                "(%d/%d)",
                request_url,
                delay,
                retry_number + 1,
                policy.max_retries,
            )
        interruptible_retry_delay(delay=delay, error=caught_error, sleep=sleeper)
        if shutdown_requested():
            raise_or_exit_worker(error=caught_error)
    raise AssertionError("retry loop did not return or raise")


def _is_retryable_error(error: BaseException) -> bool:
    """Return whether an exception represents a transient transport failure."""
    if isinstance(error, httpx.RemoteProtocolError):
        # A peer closing an HTTP stream is transient, unlike local protocol errors.
        return True
    if isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            ConnectionError,
            ConnectionResetError,
        ),
    ):
        return True
    return (
        isinstance(error, RuntimeError)
        and "client has been closed" in str(error).lower()
    )


def _retry_delay(
    error: BaseException,
    retry_number: int,
    policy: HubRetryPolicy,
    now: Callable[[], float],
) -> float:
    if not isinstance(error, _RetryableStatus) or error.status_code != 429:
        return policy.delay(retry_number)
    retry_after = _parse_retry_after(error.retry_after, now=now())
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

        Raises:
            AssertionError:
                If the bounded retry loop reaches an unreachable state.
        """
        exit_worker_if_shutdown_requested()
        if self.response is None:
            self._open_connection()

        for retry_number in range(self.fs._retry_policy.max_retries + 1):
            try:
                if self.response is None or self._stream_iterator is None:
                    return b""
                out = self._read_from_stream(self._stream_iterator, length)
                self.loc += len(out)
                return out
            except BaseException as error:
                if not _is_retryable_error(error):
                    raise
                if self.response is not None:
                    self.response.close()
                if shutdown_requested():
                    raise_or_exit_worker(error=error)
                if retry_number == self.fs._retry_policy.max_retries:
                    raise
                if "client has been closed" in str(error).lower():
                    close_session()
                delay = self.fs._retry_policy.delay(retry_number)
                logger.warning(
                    "Transient Hugging Face Hub stream interruption; retrying in "
                    "%.2fs (%d/%d)",
                    delay,
                    retry_number + 1,
                    self.fs._retry_policy.max_retries,
                )
                interruptible_retry_delay(delay=delay, error=error, sleep=time.sleep)
                if shutdown_requested():
                    raise_or_exit_worker(error=error)
                self._open_connection()
        raise AssertionError("stream retry loop did not return or raise")

    def _open_connection(self) -> None:
        self._stream_buffer.clear()
        self._stream_iterator = None
        headers = self.fs._api._build_hf_headers()
        if self.loc > 0:
            headers["Range"] = f"bytes={self.loc}-"
        try:
            context = _stream_with_retries(
                url=self.url(), headers=headers, policy=self.fs._retry_policy
            )
            self.response = self._exit_stack.enter_context(context)
        except HfHubHTTPError as error:
            if self.loc > 0 and error.response.status_code == 416:
                # Match HfFileSystemStreamFile: an exhausted resumed range is EOF.
                self.response = None
                return
            raise
        self._stream_iterator = self.response.iter_bytes()


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
    max_retries = t.cast(
        int, retry_config.get("max_retries", _DEFAULT_POLICY.max_retries)
    )
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
        max_retries=max_retries,
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


# Register at import time so a fresh spawn worker resolves ``hf://`` paths to the
# project-owned implementation before it starts reading a pickled streaming dataset.
configure_hub_streaming_retries()
