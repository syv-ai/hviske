"""Cross-process health signalling for Hugging Face Hub access retries."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

HUB_ACCESS_HEALTH_DIRECTORY_ENV = "HVISKE_HUB_ACCESS_HEALTH_DIRECTORY"
_DEFAULT_HEARTBEAT_SECONDS = 30.0
_POLL_SECONDS = 1.0

_marker_depth = 0
_marker_lock = threading.Lock()


class HubAccessHealthMonitor:
    """Report Hub retry health from a main-process background thread."""

    def __init__(
        self,
        report: Callable[[Mapping[str, float]], None],
        heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create a stopped monitor.

        Args:
            report:
                Best-effort callback receiving W&B-compatible numeric metrics.
            heartbeat_seconds (optional):
                Interval between reports while Hub access is blocked. Defaults to 30.
            clock (optional):
                Wall-clock provider. Defaults to :func:`time.time`.

        Raises:
            ValueError:
                If the heartbeat interval is not positive.
        """
        if heartbeat_seconds <= 0:
            raise ValueError("Hub access heartbeat interval must be positive")
        self._report_callback = report
        self._heartbeat_seconds = heartbeat_seconds
        self._clock = clock
        self._directory: Path | None = None
        self._previous_env: str | None = None
        self._owns_directory = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._blocked_since: float | None = None
        self._last_reported_at: float | None = None
        self._recoveries = 0
        self._report_error_active = False

    def start(self) -> None:
        """Publish worker state and start background reporting."""
        if self._thread is not None:
            return
        self._previous_env = os.environ.get(HUB_ACCESS_HEALTH_DIRECTORY_ENV)
        if self._previous_env is None:
            self._directory = Path(tempfile.mkdtemp(prefix="hviske-hub-health-"))
            os.environ[HUB_ACCESS_HEALTH_DIRECTORY_ENV] = str(self._directory)
            self._owns_directory = True
        else:
            self._directory = Path(self._previous_env)
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        thread = threading.Thread(
            target=self._run, name="hviske-hub-access-health", daemon=True
        )
        try:
            thread.start()
        except BaseException:
            self._cleanup_state()
            raise
        self._thread = thread

    def stop(self) -> None:
        """Stop reporting, restore process state, and remove worker markers."""
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=max(_POLL_SECONDS * 2, 2.0))
            self._thread = None
        self._cleanup_state()

    close = stop

    def _cleanup_state(self) -> None:
        if self._previous_env is None:
            os.environ.pop(HUB_ACCESS_HEALTH_DIRECTORY_ENV, None)
        else:
            os.environ[HUB_ACCESS_HEALTH_DIRECTORY_ENV] = self._previous_env
        self._previous_env = None
        if self._owns_directory and self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
        self._directory = None
        self._owns_directory = False

    def _run(self) -> None:
        blocked = False
        self._report(worker_count=0, now=self._clock())
        while not self._stop_event.wait(_POLL_SECONDS):
            now = self._clock()
            worker_count = self._retrying_process_count()
            if worker_count > 0:
                if not blocked:
                    blocked = True
                    self._blocked_since = now
                    self._report(worker_count=worker_count, now=now)
                elif (
                    self._last_reported_at is None
                    or now - self._last_reported_at >= self._heartbeat_seconds
                ):
                    self._report(worker_count=worker_count, now=now)
            elif blocked:
                blocked = False
                self._recoveries += 1
                self._report(worker_count=0, now=now)
                self._blocked_since = None

    def _report(self, worker_count: int, now: float) -> None:
        blocked_seconds = (
            0.0 if self._blocked_since is None else max(0.0, now - self._blocked_since)
        )
        metrics = {
            "health/hub_access_blocked": float(worker_count > 0),
            "health/hub_access_retrying_workers": float(worker_count),
            "health/hub_access_blocked_seconds": blocked_seconds,
            "health/hub_access_heartbeat_unix_seconds": now,
            "health/hub_access_recoveries_total": float(self._recoveries),
        }
        try:
            self._report_callback(metrics)
        except BaseException:
            if not self._report_error_active:
                logger.exception(
                    "Hugging Face Hub health telemetry failed; training will continue"
                )
            self._report_error_active = True
        else:
            self._report_error_active = False
            self._last_reported_at = now

    def _retrying_process_count(self) -> int:
        directory = self._directory
        if directory is None:
            return 0
        count = 0
        for marker in directory.glob("retrying-*"):
            try:
                process_id = int(marker.name.removeprefix("retrying-"))
            except ValueError:
                marker.unlink(missing_ok=True)
                continue
            if _process_exists(process_id):
                count += 1
            else:
                marker.unlink(missing_ok=True)
        return count


def _process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class HubAccessHealthScope:
    """Publish one deterministic health directory to this training rank."""

    def __init__(self, run_key: str) -> None:
        """Create a stopped health scope.

        Args:
            run_key:
                Stable identifier shared by every local distributed rank.
        """
        digest = hashlib.sha256(run_key.encode()).hexdigest()[:16]
        self.directory = Path(tempfile.gettempdir()) / f"hviske-hub-health-{digest}"
        self._previous_env = os.environ.get(HUB_ACCESS_HEALTH_DIRECTORY_ENV)
        self._started = False

    def start(self) -> None:
        """Create and publish the shared health directory."""
        if self._started:
            return
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.environ[HUB_ACCESS_HEALTH_DIRECTORY_ENV] = str(self.directory)
        self.directory.joinpath(f"retrying-{os.getpid()}").unlink(missing_ok=True)
        self._started = True

    def stop(self) -> None:
        """Clear this rank's marker and restore its previous environment."""
        if not self._started:
            return
        clear_hub_access_retrying()
        if self._previous_env is None:
            os.environ.pop(HUB_ACCESS_HEALTH_DIRECTORY_ENV, None)
        else:
            os.environ[HUB_ACCESS_HEALTH_DIRECTORY_ENV] = self._previous_env
        self._started = False

    close = stop

    def __enter__(self) -> HubAccessHealthScope:
        """Start and return this scope.

        Returns:
            This started scope.
        """
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Restore the process environment on scope exit."""
        del exc_type, exc_value, traceback
        self.stop()


def clear_hub_access_retrying() -> None:
    """Clear this process's Hub retry marker after recovery or shutdown."""
    global _marker_depth

    marker = _marker_path()
    if marker is None:
        return
    with _marker_lock:
        if _marker_depth == 0:
            return
        _marker_depth -= 1
        if _marker_depth != 0:
            return
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not clear Hugging Face Hub retry health")


def _marker_path() -> Path | None:
    directory = os.environ.get(HUB_ACCESS_HEALTH_DIRECTORY_ENV)
    if directory is None:
        return None
    return Path(directory) / f"retrying-{os.getpid()}"


def mark_hub_access_retrying() -> None:
    """Mark this process as blocked on retryable Hub access."""
    global _marker_depth

    marker = _marker_path()
    if marker is None:
        return
    with _marker_lock:
        _marker_depth += 1
        if _marker_depth != 1:
            return
        try:
            file_descriptor = os.open(marker, os.O_CREAT | os.O_WRONLY, 0o600)
        except OSError:
            _marker_depth -= 1
            logger.warning("Could not publish Hugging Face Hub retry health")
        else:
            os.close(file_descriptor)
