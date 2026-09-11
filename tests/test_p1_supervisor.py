"""Focused tests for the spawn-based P1 supervisor."""

from __future__ import annotations

import collections.abc as c
import multiprocessing
import queue
import typing as t
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

import hviske.p1_supervisor as supervisor
from hviske.p1_pipeline import PipelineSettings


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


def _config(tmp_path: Path) -> DictConfig:
    config = OmegaConf.load("config/p1_segments.yaml")
    config.runtime.scratch_root = str(tmp_path / "run")
    config.supervisor.process_count = 3
    return t.cast(DictConfig, config)


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
