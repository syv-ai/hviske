"""Tests for cooperative spawned DataLoader shutdown."""

from __future__ import annotations

import gc
import os
import time
import typing as t
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import hviske.finetune as finetune_module
from hviske import dataloader_shutdown, hub_retries
from hviske.dataloader_shutdown import (
    SHUTDOWN_SENTINEL_ENV,
    DataLoaderShutdownController,
    start_worker_shutdown_watcher,
)
from hviske.finetune import DataLoaderShutdownCallback


def _read_worker_sentinel(_: list[int]) -> str:
    """Return the inherited sentinel path from a spawned worker."""
    sentinel = os.getenv(SHUTDOWN_SENTINEL_ENV)
    assert sentinel is not None
    return sentinel


def test_controller_restores_state_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown changes are temporary and repeated cleanup is harmless."""
    previous_path = "/tmp/previous-hviske-sentinel"
    previous_grace = dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL
    monkeypatch.setenv(SHUTDOWN_SENTINEL_ENV, previous_path)
    controller = DataLoaderShutdownController(enabled=True)
    controller.start()
    current_path = os.environ[SHUTDOWN_SENTINEL_ENV]
    assert current_path != previous_path
    assert not Path(current_path).exists()

    controller.request_shutdown()
    controller.request_shutdown()
    assert Path(current_path).is_file()
    assert dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL >= previous_grace

    controller.reset()
    controller.reset()
    assert os.environ[SHUTDOWN_SENTINEL_ENV] == previous_path
    assert dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL == previous_grace
    assert not Path(current_path).parent.exists()


def test_controller_sentinel_is_visible_to_spawned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker spawned after controller startup inherits the sentinel path."""
    controller = DataLoaderShutdownController(enabled=True)
    controller.start()
    try:
        loader = DataLoader(
            _SentinelDataset(),
            num_workers=1,
            multiprocessing_context="spawn",
            collate_fn=_read_worker_sentinel,
        )
        assert next(iter(loader)) == os.environ[SHUTDOWN_SENTINEL_ENV]
    finally:
        controller.request_shutdown()
        controller.reset()
        monkeypatch.delenv(SHUTDOWN_SENTINEL_ENV, raising=False)


class _SentinelDataset(Dataset[str]):
    """Dataset used to inspect the worker environment."""

    def __getitem__(self, index: int) -> str:
        del index
        sentinel = os.getenv(SHUTDOWN_SENTINEL_ENV)
        assert sentinel is not None
        return sentinel

    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize(
    ("tracking_value", "expected_error"),
    [
        pytest.param("???", MissingMandatoryValue, id="config-access"),
        pytest.param(True, RuntimeError, id="tracking-load"),
    ],
)
def test_setup_failures_restore_controller_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tracking_value: object,
    expected_error: type[Exception],
) -> None:
    """Config and tracking setup failures restore all controller process state."""
    previous_path = "/tmp/previous-hviske-sentinel"
    previous_grace = dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL
    controller_directory = tmp_path / "controller"
    config = OmegaConf.create(
        {"dataloader_num_workers": 3, "enable_experiment_tracking": tracking_value}
    )

    def create_controller_directory(prefix: str) -> str:
        assert prefix == "hviske-dataloader-"
        controller_directory.mkdir()
        return str(controller_directory)

    def fail_tracking_setup(config: object) -> None:
        del config
        sentinel = Path(os.environ[SHUTDOWN_SENTINEL_ENV])
        assert sentinel.parent == controller_directory
        assert controller_directory.is_dir()
        raise RuntimeError("tracking setup failed")

    monkeypatch.setenv(SHUTDOWN_SENTINEL_ENV, previous_path)
    monkeypatch.setattr(
        dataloader_shutdown.tempfile, "mkdtemp", create_controller_directory
    )
    monkeypatch.setattr(
        finetune_module, "_configure_dataloader_multiprocessing", lambda config: None
    )
    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(finetune_module, "load_extracking_setup", fail_tracking_setup)

    with pytest.raises(expected_error):
        finetune_module.finetune(config=config)

    assert os.environ[SHUTDOWN_SENTINEL_ENV] == previous_path
    assert dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL == previous_grace
    assert not controller_directory.exists()


def test_shutdown_callback_requests_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The terminal callback signals after all normal callbacks have run."""
    calls: list[str] = []
    controller = DataLoaderShutdownController(enabled=False)
    monkeypatch.setattr(
        controller, "request_shutdown", lambda: calls.append("shutdown")
    )
    callback = DataLoaderShutdownCallback(controller=controller)
    callback.on_train_end(
        args=TrainingArguments(output_dir="/tmp/hviske-test"),
        state=TrainerState(),
        control=TrainerControl(),
    )
    assert calls == ["shutdown"]


def test_shutdown_callback_waits_for_terminal_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal step shutdown waits until any in-loop evaluation completes."""
    calls: list[str] = []
    controller = DataLoaderShutdownController(enabled=False)
    monkeypatch.setattr(
        controller, "request_shutdown", lambda: calls.append("shutdown")
    )
    callback = DataLoaderShutdownCallback(controller=controller)
    args = TrainingArguments(output_dir="/tmp/hviske-test")
    state = TrainerState(global_step=4)

    pending_evaluation = TrainerControl(should_training_stop=True, should_evaluate=True)
    callback.on_step_end(args=args, state=state, control=pending_evaluation)
    assert calls == []
    callback.on_evaluate(args=args, state=state, control=pending_evaluation)
    assert calls == ["shutdown"]

    calls.clear()
    intermediate_evaluation = TrainerControl(should_evaluate=True)
    callback.on_step_end(args=args, state=state, control=intermediate_evaluation)
    callback.on_evaluate(args=args, state=state, control=intermediate_evaluation)
    assert calls == []

    terminal_epoch = TrainerControl(should_training_stop=True)
    callback.on_epoch_end(args=args, state=state, control=terminal_epoch)
    assert calls == ["shutdown"]


def test_shutdown_interrupts_range_and_stream_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Range and stream retry loops re-raise without a second attempt."""
    sentinel = tmp_path / "shutdown"
    sentinel.touch()
    monkeypatch.setenv(SHUTDOWN_SENTINEL_ENV, str(sentinel))
    error = httpx.ReadTimeout("transient")
    calls = 0

    def fail() -> object:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(httpx.ReadTimeout) as range_error:
        hub_retries._run_with_retries(
            operation=fail, policy=hub_retries.HubRetryPolicy(max_retries=3)
        )
    assert range_error.value is error

    response = SimpleNamespace(close=lambda: None)
    stream_file = SimpleNamespace(
        fs=SimpleNamespace(_retry_policy=hub_retries.HubRetryPolicy(max_retries=3)),
        response=response,
        _stream_iterator=object(),
        _read_from_stream=lambda *_: (_ for _ in ()).throw(error),
    )
    with pytest.raises(httpx.ReadTimeout) as stream_error:
        hub_retries.RetryingHfFileSystemStreamFile.read(
            t.cast(hub_retries.RetryingHfFileSystemStreamFile, stream_file)
        )
    assert stream_error.value is error
    assert calls == 1


def test_three_idle_workers_exit_from_watcher(tmp_path: Path) -> None:
    """Idle prefetched workers observe terminal shutdown without retry backoff.

    Raises:
        AssertionError:
            If all workers do not start their watcher within the startup timeout.
    """
    ready_directory = tmp_path / "ready-watcher"
    ready_directory.mkdir()
    controller = DataLoaderShutdownController(enabled=True)
    controller.start()
    iterator = None
    workers: list[torch.multiprocessing.Process] = []
    try:
        loader = DataLoader(
            _WatchedPrefetchDataset(ready_directory),
            num_workers=3,
            multiprocessing_context="spawn",
            prefetch_factor=1,
        )
        iterator = iter(loader)
        workers = list(iterator._workers)  # type: ignore[attr-defined]
        deadline = time.monotonic() + 10.0
        while len(list(ready_directory.iterdir())) < 3:
            if time.monotonic() >= deadline:
                raise AssertionError("workers did not start their shutdown watchers")
            time.sleep(0.05)
        assert all(worker.is_alive() for worker in workers)

        controller.request_shutdown()
        for worker in workers:
            worker.join(timeout=2.0)
        assert [worker.exitcode for worker in workers] == [0, 0, 0]
    finally:
        del iterator
        controller.request_shutdown()
        controller.reset()


class _WatchedPrefetchDataset(Dataset[int]):
    """Dataset whose workers become idle after starting the shutdown watcher."""

    def __init__(self, ready_directory: Path) -> None:
        self.ready_directory = ready_directory

    def __getitem__(self, index: int) -> int:
        start_worker_shutdown_watcher()
        self.ready_directory.joinpath(f"watcher-{os.getpid()}").touch()
        return index

    def __len__(self) -> int:
        return 3


def test_three_workers_exit_from_long_backoff_without_termination(
    tmp_path: Path,
) -> None:
    """Three spawned workers leave an interruptible retry during teardown.

    Raises:
        AssertionError:
            If workers do not enter backoff within the startup timeout.
    """
    ready_directory = tmp_path / "ready"
    ready_directory.mkdir()
    controller = DataLoaderShutdownController(enabled=True)
    controller.start()
    iterator = None
    workers: list[torch.multiprocessing.Process] = []
    try:
        loader = DataLoader(
            _BackoffDataset(ready_directory),
            num_workers=3,
            multiprocessing_context="spawn",
        )
        iterator = iter(loader)
        workers = list(iterator._workers)  # type: ignore[attr-defined]
        deadline = time.monotonic() + 10.0
        while len(list(ready_directory.iterdir())) < 3:
            if time.monotonic() >= deadline:
                raise AssertionError("workers did not enter retry backoff")
            time.sleep(0.05)
        controller.request_shutdown()
        del iterator
        iterator = None
        gc.collect()
        for worker in workers:
            worker.join(timeout=2.0)
        assert [worker.exitcode for worker in workers] == [0, 0, 0], (
            "workers must exit normally without terminate or SIGABRT"
        )
    finally:
        del iterator
        controller.request_shutdown()
        controller.reset()


class _BackoffDataset(Dataset[int]):
    """Dataset whose workers wait in an interruptible transient retry."""

    def __init__(self, ready_directory: Path) -> None:
        self.ready_directory = ready_directory

    def __getitem__(self, index: int) -> int:
        def mark_backoff_and_sleep(delay: float) -> None:
            self.ready_directory.joinpath(f"backoff-{os.getpid()}").touch()
            time.sleep(delay)

        hub_retries._run_with_retries(
            operation=lambda: (_ for _ in ()).throw(
                httpx.ReadTimeout("worker shutdown test")
            ),
            policy=hub_retries.HubRetryPolicy(
                max_retries=3,
                base_delay_seconds=30.0,
                max_delay_seconds=30.0,
                jitter_seconds=0.0,
            ),
            sleep=mark_backoff_and_sleep,
        )
        raise AssertionError("worker retry unexpectedly completed")

    def __len__(self) -> int:
        return 3


def test_training_exception_releases_three_workers_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exception unwinding retains sentinel and join grace until workers exit."""
    ready_directory = tmp_path / "ready"
    ready_directory.mkdir()
    workers: list[torch.multiprocessing.Process] = []
    reset_observations: list[tuple[float, list[int | None]]] = []
    previous_grace = float(dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL)
    original_reset = DataLoaderShutdownController.reset

    def recording_reset(controller: DataLoaderShutdownController) -> None:
        current_grace = float(dataloader_shutdown.data_utils.MP_STATUS_CHECK_INTERVAL)
        if workers and current_grace > previous_grace:
            reset_observations.append(
                (current_grace, [worker.exitcode for worker in workers])
            )
        original_reset(controller)

    class Processor:
        tokenizer = object()

        def save_pretrained(self, save_directory: str) -> None:
            del save_directory

    class Model:
        def save_pretrained(self, save_directory: str) -> None:
            del save_directory

    class FailingTrainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def train(self, resume_from_checkpoint: object) -> None:
            del resume_from_checkpoint
            loader = DataLoader(
                _BackoffDataset(ready_directory),
                num_workers=3,
                multiprocessing_context="spawn",
            )
            iterator = iter(loader)
            workers.extend(iterator._workers)  # type: ignore[attr-defined]
            deadline = time.monotonic() + 10.0
            while len(list(ready_directory.iterdir())) < 3:
                if time.monotonic() >= deadline:
                    raise AssertionError("workers did not enter retry backoff")
                time.sleep(0.05)
            raise RuntimeError("training failed")

    class ModelSetup:
        def load_compute_metrics(self) -> None:
            return None

        def load_data_collator(self) -> None:
            return None

        def load_model(self) -> Model:
            return Model()

        def load_processor(self) -> Processor:
            return Processor()

        def load_trainer_class(self) -> type[FailingTrainer]:
            return FailingTrainer

        def load_training_arguments(self) -> None:
            return None

    config = OmegaConf.create(
        {
            "dataloader_num_workers": 3,
            "enable_experiment_tracking": False,
            "model_dir": str(tmp_path / "model"),
            "resume_from_checkpoint": False,
            "early_stopping": False,
            "push_to_hub": False,
            "model": {"use_decoder": False},
        }
    )
    monkeypatch.setattr(DataLoaderShutdownController, "reset", recording_reset)
    monkeypatch.setattr(
        finetune_module, "_configure_dataloader_multiprocessing", lambda config: None
    )
    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(finetune_module, "download_background_noises", lambda: None)
    monkeypatch.setattr(
        finetune_module, "load_model_setup", lambda config: ModelSetup()
    )
    monkeypatch.setattr(
        finetune_module,
        "load_data_for_finetuning",
        lambda config, processor: {"train": object()},
    )
    monkeypatch.setattr(finetune_module, "block_terminal_output", lambda: None)
    monkeypatch.setattr(finetune_module, "disable_tqdm", nullcontext)

    with pytest.raises(RuntimeError, match="training failed"):
        finetune_module.finetune(config=config)

    for worker in workers:
        worker.join(timeout=2.0)
    assert len(list(ready_directory.iterdir())) == 3
    assert len(reset_observations) == 1
    grace_at_reset, exitcodes_at_reset = reset_observations[0]
    assert grace_at_reset > previous_grace
    assert exitcodes_at_reset == [0, 0, 0], (
        "workers must exit before sentinel and join grace are restored"
    )
    assert [worker.exitcode for worker in workers] == [0, 0, 0], (
        "workers must exit normally without terminate or SIGABRT"
    )
