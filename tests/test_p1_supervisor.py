"""Focused tests for the spawn-based P1 supervisor."""

from __future__ import annotations

import collections.abc as c
import multiprocessing
import queue
import threading
import time
import typing as t
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import BadRequestError
from omegaconf import DictConfig, OmegaConf

import p1_dataset.supervisor as supervisor
from p1_dataset.pipeline import PipelineSettings


def test_aggregate_scratch_capacity_is_checked_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent reserves every partition's conservative scratch budget."""
    config = _config(tmp_path)
    configs = supervisor.derive_partition_configs(
        config=config,
        settings=supervisor.SupervisorSettings(process_count=2, run_root=tmp_path),
    )
    monkeypatch.setattr(
        supervisor, "calculate_scratch_requirement", lambda **kwargs: 100
    )
    monkeypatch.setattr(
        supervisor.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 150})()
    )

    with pytest.raises(ValueError, match="aggregate free space"):
        supervisor._validate_aggregate_scratch_capacity(
            configs=configs, run_root=tmp_path
        )


def _config(tmp_path: Path) -> DictConfig:
    config = OmegaConf.load("config/p1_segments.yaml")
    config.runtime.scratch_root = str(tmp_path / "run")
    config.supervisor.process_count = 3
    return t.cast(DictConfig, config)


def test_child_ipc_is_bounded_status_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Child IPC contains bounded status fields and no source payloads."""
    messages: list[dict[str, object]] = []

    class StatusQueue:
        def put(self, payload: dict[str, object], *, timeout: float) -> None:
            del timeout
            messages.append(payload)

    monkeypatch.setattr(supervisor, "run_pipeline", lambda *, config: None)
    monkeypatch.setattr(supervisor.logging, "disable", lambda level: None)
    supervisor._child_entry(
        config=OmegaConf.create({"runtime": {"workers": 1}}),
        partition_index=2,
        attempt=1,
        status_queue=StatusQueue(),
    )

    assert [message["outcome"] for message in messages] == ["STARTED", "DONE"]
    assert all(
        set(message)
        == {
            "partition_index",
            "attempt",
            "outcome",
            "category",
            "exception_class",
            "http_status_code",
            "hub_phase",
            "hub_reason",
            "hub_retryable",
            "status",
            "started_at",
            "finished_at",
        }
        for message in messages
    )
    assert not any(
        forbidden in repr(message).casefold()
        for message in messages
        for forbidden in ("audio", "transcript", "payload", "source_id", "ledger")
    )


def test_failed_child_diagnostic_cannot_leak_transport_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor diagnostics expose only allowlisted classifications."""
    messages: list[dict[str, object]] = []

    class StatusQueue:
        def put(self, payload: dict[str, object], *, timeout: float) -> None:
            del timeout
            messages.append(payload)

    def fail(*, config: object) -> None:
        del config
        request = httpx.Request(
            "POST",
            "https://hub.test/api/datasets/private/source-id/preupload/main"
            "?token=hf_secret&signature=signed",
        )
        response = httpx.Response(
            400,
            request=request,
            headers={"X-Error-Message": "private/path source-id hf_secret"},
            json={"error": "private/path source-id hf_secret"},
        )
        raise BadRequestError("signed URL private record", response=response)

    monkeypatch.setattr(supervisor, "run_pipeline", fail)
    monkeypatch.setattr(supervisor.logging, "disable", lambda level: None)
    supervisor._child_entry(
        config=OmegaConf.create({"runtime": {"workers": 1}}),
        partition_index=2,
        attempt=1,
        status_queue=StatusQueue(),
    )

    failure = messages[-1]
    assert failure["http_status_code"] == 400
    assert failure["hub_phase"] == "preupload"
    assert failure["hub_reason"] == "unknown"
    assert failure["hub_retryable"] is False
    encoded = repr(failure).casefold()
    assert not any(
        forbidden in encoded
        for forbidden in (
            "hub.test",
            "private/path",
            "source-id",
            "hf_secret",
            "signature",
            "signed url",
        )
    )


def test_kill_fallback_is_used_after_graceful_termination() -> None:
    """A stubborn child is killed after the shutdown grace period."""
    process = _StubbornProcess()
    active = {0: process}

    supervisor._terminate_active(
        active=t.cast(dict[int, multiprocessing.Process], active), grace_seconds=0
    )

    assert process.terminated
    assert process.killed
    assert not active


class _StubbornProcess:
    terminated = False
    killed = False

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True


def test_mixed_partition_outcome_removes_stale_aggregate_done(tmp_path: Path) -> None:
    """Mixed outcomes publish only FAILED while retaining partition markers."""
    supervisor_root = tmp_path / "supervisor"
    marker_root = supervisor_root / "markers"
    marker_root.mkdir(parents=True)
    (supervisor_root / "DONE").write_text("stale", encoding="utf-8")
    states = [
        supervisor._PartitionState(config=OmegaConf.create({}), attempt=1),
        supervisor._PartitionState(config=OmegaConf.create({}), attempt=2),
    ]

    supervisor._write_markers(
        supervisor_root=supervisor_root,
        marker_root=marker_root,
        completed={0},
        failed={1},
        states=states,
    )

    assert not (supervisor_root / "DONE").exists()
    assert (supervisor_root / "FAILED").exists()
    assert (marker_root / "partition-0.DONE").exists()
    assert (marker_root / "partition-1.FAILED").exists()


def test_parent_exception_terminates_children_before_closing_ipc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected parent errors still reap every child before queue teardown."""
    context = _LiveContext()
    monkeypatch.setattr(supervisor.multiprocessing, "get_context", lambda name: context)

    def explode(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("parent failure")

    monkeypatch.setattr(supervisor, "_drain_statuses", explode)
    with pytest.raises(RuntimeError, match="parent failure"):
        supervisor.run_supervisor(
            config=_config(tmp_path),
            settings=supervisor.SupervisorSettings(
                process_count=1,
                max_attempts=1,
                shutdown_grace_seconds=0,
                run_root=tmp_path / "run",
            ),
        )

    assert context.process.terminated
    assert context.process.joined


class _FakeQueue:
    def __init__(self) -> None:
        self.values: list[object] = []

    def close(self) -> None:
        pass

    def get_nowait(self) -> object:
        if not self.values:
            raise queue.Empty
        return self.values.pop(0)

    def join_thread(self) -> None:
        pass

    def put(self, value: object, *, timeout: float) -> None:
        del timeout
        self.values.append(value)


class _LiveProcess:
    exitcode = None

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False
        self.joined = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.joined = True

    def kill(self) -> None:
        self.alive = False

    def start(self) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


class _LiveContext:
    def __init__(self) -> None:
        self.process = _LiveProcess()

    def Process(
        self, *, target: c.Callable[..., None], args: tuple[object, ...], name: str
    ) -> _LiveProcess:
        del target, args, name
        return self.process

    def Queue(self, *, maxsize: int) -> _FakeQueue:
        del maxsize
        return _FakeQueue()


def test_partition_configs_have_one_digest_unique_scratch_and_common_lock(
    tmp_path: Path,
) -> None:
    """Partition configs share identity and lock but isolate scratch roots."""
    config = _config(tmp_path)
    settings = supervisor.SupervisorSettings.from_config(config)
    configs = supervisor.derive_partition_configs(config=config, settings=settings)

    supervisor.validate_invariants(configs=configs, settings=settings)
    resolved = [PipelineSettings.from_config(item) for item in configs]
    assert {item.pipeline_digest for item in resolved} == {
        PipelineSettings.from_config(config).pipeline_digest
    }
    assert len({item.scratch_root for item in resolved}) == 3
    assert len({item.publish_lock_path for item in resolved}) == 1
    assert {item.workers for item in resolved} == {1}


def test_reap_waits_for_asynchronous_terminal_status(tmp_path: Path) -> None:
    """A feeder-delayed DONE status is observed after the child has exited."""
    context = multiprocessing.get_context("spawn")
    status_queue = context.Queue(maxsize=2)
    status: supervisor.StatusPayload = {
        "partition_index": 0,
        "attempt": 1,
        "outcome": "DONE",
        "category": "none",
        "exception_class": "none",
        "http_status_code": None,
        "hub_phase": "unknown",
        "hub_reason": "unknown",
        "hub_retryable": None,
        "status": "done",
        "started_at": "now",
        "finished_at": "now",
    }

    def publish_later() -> None:
        time.sleep(0.03)
        status_queue.put(status)

    publisher = threading.Thread(target=publish_later)
    publisher.start()
    process = _AlreadyExitedProcess()
    logs = supervisor._StatusLogs(root=tmp_path, process_count=1)
    states = [supervisor._PartitionState(config=OmegaConf.create({}), attempt=1)]
    active = {0: t.cast(multiprocessing.Process, process)}
    completed: set[int] = set()
    failed: set[int] = set()
    try:
        supervisor._reap_partitions(
            active=active,
            states=states,
            terminal={},
            logs=logs,
            status_queue=status_queue,
            retry_at={},
            completed=completed,
            failed=failed,
            max_attempts=1,
            retry_delay_seconds=0,
            stop_requested=False,
        )
    finally:
        publisher.join()
        logs.close()
        status_queue.close()
        status_queue.join_thread()

    assert completed == {0}
    assert not failed
    assert process.joined


class _AlreadyExitedProcess:
    exitcode = 0
    joined = False

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.joined = True


def test_supervisor_exhausts_retries_and_writes_failed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausted child retries produce a non-zero result and FAILED marker."""
    config = _config(tmp_path)

    def fail(*, config: DictConfig) -> None:
        del config
        raise RuntimeError("expected test failure")

    monkeypatch.setattr(supervisor, "run_pipeline", fail)
    monkeypatch.setattr(supervisor.logging, "disable", lambda level: None)
    monkeypatch.setattr(
        supervisor.multiprocessing, "get_context", lambda name: _FakeContext()
    )
    result = supervisor.run_supervisor(
        config=config,
        settings=supervisor.SupervisorSettings(
            process_count=1,
            retry_delay_seconds=0,
            max_attempts=2,
            shutdown_grace_seconds=0,
            run_root=tmp_path / "run",
        ),
    )

    assert result.exit_code == 1
    assert result.failed == (0,)
    assert (tmp_path / "run/supervisor/FAILED").exists()


class _FakeProcess:
    def __init__(
        self, *, target: c.Callable[..., None], args: tuple[object, ...], name: str
    ) -> None:
        del name
        self._target = target
        self._args = args
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def start(self) -> None:
        self._target(*self._args)
        self.exitcode = 0


class _FakeContext:
    def Process(
        self, *, target: c.Callable[..., None], args: tuple[object, ...], name: str
    ) -> _FakeProcess:
        return _FakeProcess(target=target, args=args, name=name)

    def Queue(self, *, maxsize: int) -> _FakeQueue:
        del maxsize
        return _FakeQueue()


def test_supervisor_records_abnormal_exit_without_child_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that exits before IPC is classified as a crash."""
    config = _config(tmp_path)
    monkeypatch.setattr(
        supervisor.multiprocessing, "get_context", lambda name: _CrashContext()
    )
    result = supervisor.run_supervisor(
        config=config,
        settings=supervisor.SupervisorSettings(
            process_count=1,
            retry_delay_seconds=0,
            max_attempts=1,
            shutdown_grace_seconds=0,
            run_root=tmp_path / "run",
        ),
    )

    log = (tmp_path / "run/supervisor/partition-0.jsonl").read_text()
    assert result.exit_code == 1
    assert '"category": "child_crash"' in log


class _CrashProcess:
    exitcode = 1

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def start(self) -> None:
        pass


class _CrashContext:
    def Process(
        self, *, target: c.Callable[..., None], args: tuple[object, ...], name: str
    ) -> _CrashProcess:
        del target, args, name
        return _CrashProcess()

    def Queue(self, *, maxsize: int) -> _FakeQueue:
        del maxsize
        return _FakeQueue()


def test_supervisor_retries_a_failed_child_and_writes_done_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrying child eventually creates the aggregate DONE marker."""
    config = _config(tmp_path)
    attempts: list[int] = []

    def run(config: DictConfig) -> None:
        attempts.append(int(config.runtime.partition_index))
        if len(attempts) == 1:
            raise RuntimeError("not sent through IPC")

    monkeypatch.setattr(supervisor, "run_pipeline", run)
    monkeypatch.setattr(supervisor.logging, "disable", lambda level: None)
    context = _FakeContext()
    monkeypatch.setattr(supervisor.multiprocessing, "get_context", lambda name: context)
    result = supervisor.run_supervisor(
        config=config,
        settings=supervisor.SupervisorSettings(
            process_count=1,
            retry_delay_seconds=0,
            max_attempts=2,
            shutdown_grace_seconds=0,
            run_root=tmp_path / "run",
        ),
    )

    assert result.exit_code == 0
    assert result.completed == (0,)
    assert attempts == [0, 0]
    assert (tmp_path / "run/supervisor/DONE").exists()
    assert not (tmp_path / "run/supervisor/FAILED").exists()


def test_supervisor_retries_known_missing_object_hub_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded missing-object retry remains eligible for supervisor retry."""
    config = _config(tmp_path)
    attempts: list[int] = []

    def run(config: DictConfig) -> None:
        attempts.append(int(config.runtime.partition_index))
        if len(attempts) == 1:
            raise BadRequestError(
                "private source-id",
                response=httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST", "https://huggingface.co/api/datasets/org/repo/commit"
                    ),
                    content=b"LFS pointer pointed to a file that does not exist",
                ),
            )

    monkeypatch.setattr(supervisor, "run_pipeline", run)
    monkeypatch.setattr(supervisor.logging, "disable", lambda level: None)
    monkeypatch.setattr(
        supervisor.multiprocessing, "get_context", lambda name: _FakeContext()
    )
    result = supervisor.run_supervisor(
        config=config,
        settings=supervisor.SupervisorSettings(
            process_count=1,
            retry_delay_seconds=0,
            max_attempts=2,
            shutdown_grace_seconds=0,
            run_root=tmp_path / "run",
        ),
    )

    assert attempts == [0, 0]
    assert result.completed == (0,)
    assert result.exit_code == 0
    log = (tmp_path / "run/supervisor/partition-0.jsonl").read_text()
    assert '"hub_retryable": true' in log


def test_supervisor_stops_non_retryable_hub_failure_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic Hub failure does not consume supervisor attempts."""
    config = _config(tmp_path)
    attempts = 0

    def fail(*, config: DictConfig) -> None:
        nonlocal attempts
        del config
        attempts += 1
        raise BadRequestError(
            "private source-id",
            response=httpx.Response(
                400,
                request=httpx.Request(
                    "POST", "https://huggingface.co/api/datasets/org/repo/commit"
                ),
                json={"error": "unrelated validation failure"},
            ),
        )

    monkeypatch.setattr(supervisor, "run_pipeline", fail)
    monkeypatch.setattr(supervisor.logging, "disable", lambda level: None)
    monkeypatch.setattr(
        supervisor.multiprocessing, "get_context", lambda name: _FakeContext()
    )
    result = supervisor.run_supervisor(
        config=config,
        settings=supervisor.SupervisorSettings(
            process_count=1,
            retry_delay_seconds=0,
            max_attempts=3,
            shutdown_grace_seconds=0,
            run_root=tmp_path / "run",
        ),
    )

    assert attempts == 1
    assert result.failed == (0,)
    assert (tmp_path / "run/supervisor/partition-0.jsonl").read_text().count(
        '"attempt": 1'
    ) == 2
    assert (
        '"hub_retryable": false'
        in (tmp_path / "run/supervisor/partition-0.jsonl").read_text()
    )


def test_validation_rejects_nested_workers_before_spawn(tmp_path: Path) -> None:
    """Validation rejects nested process workers before spawning children."""
    config = _config(tmp_path)
    settings = supervisor.SupervisorSettings.from_config(config)
    configs = list(
        supervisor.derive_partition_configs(config=config, settings=settings)
    )
    configs[0].runtime.workers = 2

    with pytest.raises(ValueError, match="runtime worker"):
        supervisor.validate_invariants(configs=configs, settings=settings)
