"""Tests for cooperative spawned DataLoader shutdown."""

from __future__ import annotations

import gc
import os
import time
import typing as t
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from hviske import dataloader_shutdown, hub_retries
from hviske.dataloader_shutdown import (
    SHUTDOWN_SENTINEL_ENV,
    DataLoaderShutdownController,
    interruptible_retry_delay,
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
        assert all(not worker.is_alive() for worker in workers)
    finally:
        del iterator
        controller.request_shutdown()
        controller.reset()


class _BackoffDataset(Dataset[int]):
    """Dataset whose workers wait in an interruptible transient retry."""

    def __init__(self, ready_directory: Path) -> None:
        self.ready_directory = ready_directory

    def __getitem__(self, index: int) -> int:
        self.ready_directory.joinpath(str(os.getpid())).touch()
        try:
            interruptible_retry_delay(
                delay=30.0, error=RuntimeError("worker shutdown test")
            )
        except RuntimeError:
            return index
        raise AssertionError("worker retry unexpectedly completed")

    def __len__(self) -> int:
        return 3
