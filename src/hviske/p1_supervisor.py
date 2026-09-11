"""Spawn and supervise one isolated process for every P1 partition.

The supervisor is deliberately a narrow parent process.  It resolves partition
configuration, starts children with the ``spawn`` context, and receives only bounded
status records.  Source data, pipeline objects, ledgers, and exception instances never
cross the process boundary.
"""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import json
import logging
import multiprocessing
import queue
import re
import signal
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .p1_pipeline import PipelineSettings, _safe_exception_category, run_pipeline

logger = logging.getLogger(__name__)

_DEFAULT_PROCESS_COUNT = 8
_DEFAULT_RETRY_DELAY_SECONDS = 5.0
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 30.0
_MAX_RETRY_DELAY_SECONDS = 300.0
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class StatusPayload(t.TypedDict):
    """The complete, bounded payload allowed on the child status queue."""

    partition_index: int
    attempt: int
    outcome: str
    category: str
    exception_class: str
    status: str
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class SupervisorSettings:
    """Runtime-only controls for the P1 process supervisor."""

    process_count: int = _DEFAULT_PROCESS_COUNT
    retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    shutdown_grace_seconds: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS
    run_root: Path | None = None

    @classmethod
    def from_config(cls, config: DictConfig | dict[str, object]) -> SupervisorSettings:
        """Resolve supervisor controls without reading any pipeline state.

        Args:
            config:
                Hydra configuration containing the optional ``supervisor`` section.

        Returns:
            Runtime controls for one supervised production run.

        """
        raw = OmegaConf.to_container(config, resolve=True)
        root = t.cast(dict[str, object], raw)
        supervisor = root.get("supervisor", {})
        values = (
            t.cast(dict[str, object], supervisor)
            if isinstance(supervisor, dict)
            else {}
        )
        runtime = root.get("runtime", {})
        runtime_values = (
            t.cast(dict[str, object], runtime) if isinstance(runtime, dict) else {}
        )

        process_count = _positive_int(
            values.get(
                "process_count",
                runtime_values.get("process_count", _DEFAULT_PROCESS_COUNT),
            ),
            "process_count",
        )
        max_attempts = _positive_int(
            values.get(
                "max_attempts",
                values.get(
                    "retry_max_attempts",
                    runtime_values.get("max_attempts", _DEFAULT_MAX_ATTEMPTS),
                ),
            ),
            "max_attempts",
        )
        retry_delay = _nonnegative_float(
            values.get(
                "retry_delay_seconds",
                values.get(
                    "retry_delay",
                    runtime_values.get(
                        "retry_delay_seconds", _DEFAULT_RETRY_DELAY_SECONDS
                    ),
                ),
            ),
            "retry_delay_seconds",
        )
        shutdown_grace = _nonnegative_float(
            values.get(
                "shutdown_grace_seconds",
                runtime_values.get(
                    "shutdown_grace_seconds", _DEFAULT_SHUTDOWN_GRACE_SECONDS
                ),
            ),
            "shutdown_grace_seconds",
        )
        run_root_value = values.get("run_root", runtime_values.get("run_root"))
        run_root = (
            None if run_root_value is None else Path(str(run_root_value)).expanduser()
        )
        return cls(
            process_count=process_count,
            retry_delay_seconds=retry_delay,
            max_attempts=max_attempts,
            shutdown_grace_seconds=shutdown_grace,
            run_root=run_root,
        )


def _child_entry(
    config: DictConfig, partition_index: int, attempt: int, status_queue: object
) -> None:
    """Execute one partition and send only bounded status records."""
    started_at = _timestamp()
    _send_status(
        status_queue=status_queue,
        partition_index=partition_index,
        attempt=attempt,
        outcome="STARTED",
        category="none",
        exception_class="none",
        status="running",
        started_at=started_at,
        finished_at=started_at,
    )
    try:
        # Child logs are kept out of the parent's terminal stream.  The pipeline's
        # metadata log remains in the partition scratch directory.
        logging.disable(logging.CRITICAL)
        run_pipeline(config=config)
    except BaseException as error:
        _send_status(
            status_queue=status_queue,
            partition_index=partition_index,
            attempt=attempt,
            outcome="FAILED",
            category=_safe_category(error),
            exception_class=_safe_class(error),
            status="failed",
            started_at=started_at,
            finished_at=_timestamp(),
        )
        return
    _send_status(
        status_queue=status_queue,
        partition_index=partition_index,
        attempt=attempt,
        outcome="DONE",
        category="none",
        exception_class="none",
        status="done",
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _send_status(
    *,
    status_queue: object,
    partition_index: int,
    attempt: int,
    outcome: str,
    category: str,
    exception_class: str,
    status: str,
    started_at: str,
    finished_at: str,
) -> None:
    payload: StatusPayload = {
        "partition_index": partition_index,
        "attempt": attempt,
        "outcome": outcome,
        "category": _safe_name(category),
        "exception_class": _safe_name(exception_class),
        "status": _safe_name(status),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    try:
        status_queue.put(payload, timeout=2.0)
    except (OSError, queue.Full):
        # A missing terminal status is deliberately treated as a child crash by the
        # parent.  Never block a worker indefinitely on operational telemetry.
        return


@dataclass(frozen=True)
class SupervisorResult:
    """Sanitised result of a supervised run."""

    completed: tuple[int, ...]
    failed: tuple[int, ...]
    interrupted: bool
    exit_code: int


def run_supervisor(
    *, config: DictConfig, settings: SupervisorSettings | None = None
) -> SupervisorResult:
    """Run all P1 partitions under one spawn-based parent.

    Args:
        config:
            Fully resolved Hydra P1 configuration.
        settings (optional):
            Runtime controls.  When omitted, they are read from ``config``.

    Returns:
        A metadata-only result.  ``exit_code`` is non-zero when a partition exhausts
        retries or the parent receives a shutdown signal.

    """
    controls = settings or SupervisorSettings.from_config(config)
    partition_configs = derive_partition_configs(config=config, settings=controls)
    validate_invariants(configs=partition_configs, settings=controls)
    run_root = _resolved_run_root(config=config, settings=controls)
    supervisor_root = run_root / "supervisor"
    marker_root = supervisor_root / "markers"
    supervisor_root.mkdir(parents=True, exist_ok=True)
    marker_root.mkdir(parents=True, exist_ok=True)
    _remove_old_markers(supervisor_root=supervisor_root, marker_root=marker_root)

    states = [_PartitionState(config=item) for item in partition_configs]
    logs = _StatusLogs(root=supervisor_root, process_count=controls.process_count)
    context = multiprocessing.get_context("spawn")
    status_queue = context.Queue(maxsize=max(controls.process_count * 2, 2))
    active: dict[int, multiprocessing.Process] = {}
    retry_at: dict[int, float] = {}
    terminal: dict[tuple[int, int], StatusPayload] = {}
    completed: set[int] = set()
    failed: set[int] = set()
    interrupted = False
    stop_requested = False
    previous_handlers = _install_signal_handlers(
        callback=lambda _signum, _frame: _request_stop()
    )

    try:
        if _stop_requested():
            stop_requested = True
            interrupted = True
        for partition_index in range(controls.process_count):
            if stop_requested or _stop_requested():
                stop_requested = True
                break
            _start_partition(
                context=context,
                partition_index=partition_index,
                states=states,
                active=active,
                status_queue=status_queue,
            )

        while active or retry_at:
            if _stop_requested():
                stop_requested = True
                interrupted = True
                retry_at.clear()
                _terminate_active(
                    active=active, grace_seconds=controls.shutdown_grace_seconds
                )

            _drain_statuses(status_queue=status_queue, logs=logs, terminal=terminal)
            _reap_partitions(
                active=active,
                states=states,
                terminal=terminal,
                logs=logs,
                retry_at=retry_at,
                completed=completed,
                failed=failed,
                max_attempts=controls.max_attempts,
                retry_delay_seconds=controls.retry_delay_seconds,
                stop_requested=stop_requested,
            )
            if stop_requested:
                break
            now = time.monotonic()
            for partition_index, deadline in list(retry_at.items()):
                if deadline <= now:
                    del retry_at[partition_index]
                    _start_partition(
                        context=context,
                        partition_index=partition_index,
                        states=states,
                        active=active,
                        status_queue=status_queue,
                    )
            if active or retry_at:
                time.sleep(0.05)

        _drain_statuses(status_queue=status_queue, logs=logs, terminal=terminal)
        if stop_requested:
            failed.update(
                index
                for index in range(controls.process_count)
                if index not in completed
            )
        for partition_index in range(controls.process_count):
            if partition_index not in completed:
                failed.add(partition_index)

        _write_markers(
            supervisor_root=supervisor_root,
            marker_root=marker_root,
            completed=completed,
            failed=failed,
            states=states,
        )
        return SupervisorResult(
            completed=tuple(sorted(completed)),
            failed=tuple(sorted(failed)),
            interrupted=interrupted,
            exit_code=1 if failed or interrupted else 0,
        )
    finally:
        _restore_signal_handlers(previous_handlers)
        logs.close()
        status_queue.close()
        status_queue.join_thread()


@dataclass
class _PartitionState:
    """Parent-only bookkeeping for one partition."""

    config: DictConfig
    attempt: int = 0
    done: bool = False
    failed: bool = False


class _StatusLogs:
    """Parent-owned, non-interleaved JSONL status logs."""

    def __init__(self, *, root: Path, process_count: int) -> None:
        self._streams = {
            index: (root / f"partition-{index}.jsonl").open("a", encoding="utf-8")
            for index in range(process_count)
        }

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()

    def write(self, *, status: StatusPayload) -> None:
        stream = self._streams.get(status["partition_index"])
        if stream is None:
            return
        stream.write(json.dumps(status, sort_keys=True) + "\n")
        stream.flush()


def _drain_statuses(
    *,
    status_queue: object,
    logs: _StatusLogs,
    terminal: dict[tuple[int, int], StatusPayload],
) -> None:
    while True:
        try:
            raw = t.cast(object, status_queue.get_nowait())
        except queue.Empty:
            return
        status = _sanitise_status(raw)
        if status is None:
            continue
        logs.write(status=status)
        if status["outcome"] in {"DONE", "FAILED"}:
            terminal[(status["partition_index"], status["attempt"])] = status


def _sanitise_status(value: object) -> StatusPayload | None:
    if not isinstance(value, dict):
        return None
    integer_fields = ("partition_index", "attempt")
    if any(not isinstance(value.get(key), int) for key in integer_fields):
        return None
    partition_index = t.cast(int, value["partition_index"])
    attempt = t.cast(int, value["attempt"])
    if partition_index < 0 or attempt < 1:
        return None
    strings = (
        "outcome",
        "category",
        "exception_class",
        "status",
        "started_at",
        "finished_at",
    )
    if any(not isinstance(value.get(key), str) for key in strings):
        return None
    return {
        "partition_index": partition_index,
        "attempt": attempt,
        "outcome": _safe_name(t.cast(str, value["outcome"])),
        "category": _safe_name(t.cast(str, value["category"])),
        "exception_class": _safe_name(t.cast(str, value["exception_class"])),
        "status": _safe_name(t.cast(str, value["status"])),
        "started_at": t.cast(str, value["started_at"])[:40],
        "finished_at": t.cast(str, value["finished_at"])[:40],
    }


def _install_signal_handlers(
    *, callback: c.Callable[[int, object], None]
) -> dict[int, object]:
    previous: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, callback)
    return previous


def _reap_partitions(
    *,
    active: dict[int, multiprocessing.Process],
    states: list[_PartitionState],
    terminal: dict[tuple[int, int], StatusPayload],
    logs: _StatusLogs,
    retry_at: dict[int, float],
    completed: set[int],
    failed: set[int],
    max_attempts: int,
    retry_delay_seconds: float,
    stop_requested: bool,
) -> None:
    for partition_index, process in list(active.items()):
        if process.is_alive():
            continue
        process.join()
        del active[partition_index]
        state = states[partition_index]
        status = terminal.pop((partition_index, state.attempt), None)
        succeeded = (
            process.exitcode == 0 and status is not None and status["outcome"] == "DONE"
        )
        if not succeeded and (status is None or status["outcome"] == "DONE"):
            logs.write(
                status=_crash_status(
                    partition_index=partition_index, attempt=state.attempt
                )
            )
        if succeeded:
            state.done = True
            completed.add(partition_index)
            continue
        if stop_requested or state.attempt >= max_attempts:
            state.failed = True
            failed.add(partition_index)
            continue
        retry_at[partition_index] = time.monotonic() + min(
            retry_delay_seconds * (2 ** (state.attempt - 1)), _MAX_RETRY_DELAY_SECONDS
        )


def _crash_status(*, partition_index: int, attempt: int) -> StatusPayload:
    timestamp = _timestamp()
    return {
        "partition_index": partition_index,
        "attempt": attempt,
        "outcome": "FAILED",
        "category": "child_crash",
        "exception_class": "ProcessExit",
        "status": "crashed",
        "started_at": timestamp,
        "finished_at": timestamp,
    }


def _remove_old_markers(*, supervisor_root: Path, marker_root: Path) -> None:
    for path in (supervisor_root / "DONE", supervisor_root / "FAILED"):
        path.unlink(missing_ok=True)
    for path in marker_root.glob("partition-*.DONE"):
        path.unlink()
    for path in marker_root.glob("partition-*.FAILED"):
        path.unlink()


def _resolved_run_root(*, config: DictConfig, settings: SupervisorSettings) -> Path:
    if settings.run_root is not None:
        return settings.run_root.resolve()
    raw = OmegaConf.to_container(config, resolve=True)
    root = t.cast(dict[str, object], raw)
    runtime = t.cast(dict[str, object], root["runtime"])
    return Path(str(runtime["scratch_root"])).expanduser().resolve()


def _start_partition(
    *,
    context: multiprocessing.context.BaseContext,
    partition_index: int,
    states: list[_PartitionState],
    active: dict[int, multiprocessing.Process],
    status_queue: object,
) -> None:
    state = states[partition_index]
    state.attempt += 1
    process = context.Process(
        target=_child_entry,
        args=(state.config, partition_index, state.attempt, status_queue),
        name=f"p1-partition-{partition_index}",
    )
    process.start()
    active[partition_index] = process


def _terminate_active(
    *, active: dict[int, multiprocessing.Process], grace_seconds: float
) -> None:
    for process in active.values():
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in active.values():
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in active.values():
        if process.is_alive():
            process.kill()
        process.join()
    active.clear()


def _write_markers(
    *,
    supervisor_root: Path,
    marker_root: Path,
    completed: set[int],
    failed: set[int],
    states: list[_PartitionState],
) -> None:
    done_lines: list[str] = []
    failed_lines: list[str] = []
    for index, state in enumerate(states):
        if index in completed:
            target = marker_root / f"partition-{index}.DONE"
            done_lines.append(f"partition={index} attempts={state.attempt}")
        else:
            target = marker_root / f"partition-{index}.FAILED"
            failed_lines.append(f"partition={index} attempts={state.attempt}")
        target.write_text(
            f"partition={index}\nattempts={state.attempt}\n", encoding="utf-8"
        )
    if done_lines:
        (supervisor_root / "DONE").write_text(
            "\n".join(done_lines) + "\n", encoding="utf-8"
        )
    if failed_lines:
        (supervisor_root / "FAILED").write_text(
            "\n".join(failed_lines) + "\n", encoding="utf-8"
        )


def derive_partition_configs(
    *, config: DictConfig, settings: SupervisorSettings | None = None
) -> tuple[DictConfig, ...]:
    """Derive deterministic, runtime-isolated configuration for every partition.

    The common root is intentionally retained in each child config: ``PipelineSettings``
    appends the partition namespace and therefore gives each child its own scratch,
    ledger, audit reservoir, and metadata log.

    Returns:
        One Hydra configuration for each deterministic partition.

    Raises:
        ValueError:
            If the runtime section is missing.
    """
    controls = settings or SupervisorSettings.from_config(config)
    root = t.cast(dict[str, object], OmegaConf.to_container(config, resolve=False))
    runtime = root.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime configuration is required")
    common_root = _resolved_run_root(config=config, settings=controls)
    lock_value = runtime.get("publish_lock_path")
    lock_path = (
        common_root / "publish.lock"
        if lock_value is None
        else Path(str(lock_value)).expanduser().resolve()
    )
    result: list[DictConfig] = []
    for partition_index in range(controls.process_count):
        child = OmegaConf.create(root)
        child.runtime.partition_count = controls.process_count
        child.runtime.partition_index = partition_index
        child.runtime.workers = 1
        child.runtime.scratch_root = str(common_root)
        child.runtime.publish_lock_path = str(lock_path)
        result.append(child)
    return tuple(result)


def validate_invariants(
    *, configs: c.Sequence[DictConfig], settings: SupervisorSettings
) -> None:
    """Validate all cross-process invariants before any child is spawned.

    Args:
        configs:
            One derived configuration per partition.
        settings:
            Supervisor controls used to derive the configurations.

    Raises:
        ValueError:
            If identity, partition, scratch, worker, or lock invariants fail.
    """
    if len(configs) != settings.process_count:
        raise ValueError("one configuration is required for every partition")
    resolved = [PipelineSettings.from_config(config) for config in configs]
    digests = {item.pipeline_digest for item in resolved}
    scratch_roots = {item.scratch_root for item in resolved}
    locks = {item.publish_lock_path for item in resolved}
    if len(digests) != 1:
        raise ValueError("all partitions must use one canonical pipeline digest")
    if len(scratch_roots) != settings.process_count:
        raise ValueError("partitions must have unique scratch roots")
    if len(locks) != 1:
        raise ValueError("all partitions must use one common publication lock")
    for index, item in enumerate(resolved):
        if (
            item.partition_count != settings.process_count
            or item.partition_index != index
        ):
            raise ValueError("partition assignment is not deterministic")
        if item.workers != 1:
            raise ValueError("each partition must have exactly one runtime worker")
        if item.alignment_method != "timestamp-native:p1-transcripts.words":
            raise ValueError("production requires timestamp-native alignment")


_STOP_REQUESTED = False


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _request_stop() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    global _STOP_REQUESTED
    for signum, handler in previous.items():
        safe_handler = t.cast(int | None | c.Callable[[int, object], object], handler)
        signal.signal(signum, safe_handler)
    _STOP_REQUESTED = False


def _safe_category(error: BaseException) -> str:
    return (
        _safe_exception_category(error)
        if isinstance(error, Exception)
        else "child_crash"
    )


def _safe_class(error: BaseException) -> str:
    return _safe_name(type(error).__name__)


def _safe_name(value: str) -> str:
    return value if _SAFE_NAME.fullmatch(value) else "redacted"


def _stop_requested() -> bool:
    return _STOP_REQUESTED


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "StatusPayload",
    "SupervisorResult",
    "SupervisorSettings",
    "derive_partition_configs",
    "run_supervisor",
    "validate_invariants",
]
