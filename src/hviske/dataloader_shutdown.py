"""Cooperative shutdown for spawned training DataLoader workers."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch.utils.data._utils as data_utils
from huggingface_hub import constants

SHUTDOWN_SENTINEL_ENV = "HVISKE_DATALOADER_SHUTDOWN_SENTINEL"
_FINALISATION_MARGIN_SECONDS = 5.0


class DataLoaderShutdownController:
    """Signal spawned workers before the parent joins them.

    A filesystem sentinel is used rather than a multiprocessing primitive because the
    dataset and its retry code are reconstructed in spawned workers. The sentinel path
    is placed in the environment before any worker can be created.
    """

    def __init__(self, enabled: bool) -> None:
        """Create a controller, optionally allocating per-run state.

        Args:
            enabled:
                Whether this run has spawned DataLoader workers.
        """
        self.enabled = enabled
        self._directory: Path | None = None
        self._sentinel: Path | None = None
        self._previous_env = os.environ.get(SHUTDOWN_SENTINEL_ENV)
        self._previous_join_grace: int | float | None = None
        self._shutdown_requested = False
        self._started = False

        if enabled:
            self._directory = Path(tempfile.mkdtemp(prefix="hviske-dataloader-"))
            self._sentinel = self._directory / "shutdown"

    def request_shutdown(self) -> None:
        """Atomically signal workers and extend only the parent's join grace."""
        if not self.enabled or not self._started:
            return
        assert self._sentinel is not None
        if not self._shutdown_requested:
            try:
                file_descriptor = os.open(
                    self._sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                pass
            else:
                os.close(file_descriptor)
            self._shutdown_requested = True

        if self._previous_join_grace is None:
            self._previous_join_grace = data_utils.MP_STATUS_CHECK_INTERVAL
            required_grace = (
                float(constants.HF_HUB_DOWNLOAD_TIMEOUT) + _FINALISATION_MARGIN_SECONDS
            )
            setattr(
                data_utils,
                "MP_STATUS_CHECK_INTERVAL",
                max(float(self._previous_join_grace), required_grace),
            )

    def reset(self) -> None:
        """Restore process state and remove this run's temporary sentinel."""
        if not self.enabled:
            return
        if self._previous_join_grace is not None:
            setattr(data_utils, "MP_STATUS_CHECK_INTERVAL", self._previous_join_grace)
            self._previous_join_grace = None
        if self._started:
            if self._previous_env is None:
                os.environ.pop(SHUTDOWN_SENTINEL_ENV, None)
            else:
                os.environ[SHUTDOWN_SENTINEL_ENV] = self._previous_env
            self._started = False
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
            self._directory = None
            self._sentinel = None
        self._shutdown_requested = False

    def start(self) -> None:
        """Publish the sentinel path for this run's current and future workers."""
        if not self.enabled or self._started:
            return
        assert self._sentinel is not None
        os.environ[SHUTDOWN_SENTINEL_ENV] = str(self._sentinel)
        self._started = True

    close = reset

    def __enter__(self) -> DataLoaderShutdownController:
        """Start publishing the per-run worker state.

        Returns:
            This started controller.
        """
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Signal workers on scope exit and restore the parent process."""
        del exc_type, exc_value, traceback
        self.request_shutdown()
        self.reset()


def interruptible_retry_delay(
    delay: float, error: BaseException, sleep: Callable[[float], None] | None = None
) -> None:
    """Wait for a retry while allowing a shutdown sentinel to interrupt it.

    Args:
        delay:
            The normal retry delay.
        error:
            The transient error that should be re-raised if shutdown is requested.
        sleep (optional):
            Sleep function used by tests and callers that need a controlled clock.
            Defaults to :func:`time.sleep`.

    """
    sleeper = sleep or time.sleep
    if not os.getenv(SHUTDOWN_SENTINEL_ENV):
        sleeper(delay)
        return

    deadline = time.monotonic() + delay
    while True:
        if shutdown_requested():
            raise error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        sleeper(min(0.1, remaining))


def shutdown_requested() -> bool:
    """Return whether the inherited worker sentinel has been created."""
    sentinel = os.getenv(SHUTDOWN_SENTINEL_ENV)
    return bool(sentinel) and Path(sentinel).is_file()
