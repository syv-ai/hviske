"""Tests for cross-process Hugging Face Hub access health reporting."""

import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

import hviske.hub_access_health as hub_health
from hviske.hub_access_health import (
    HUB_ACCESS_HEALTH_DIRECTORY_ENV,
    HubAccessHealthMonitor,
    HubAccessHealthScope,
    clear_hub_access_retrying,
    mark_hub_access_retrying,
)


def _publish_sibling_rank_marker(run_key: str, connection: Connection) -> None:
    with HubAccessHealthScope(run_key=run_key):
        mark_hub_access_retrying()
        connection.send(True)
        connection.recv()
        clear_hub_access_retrying()
    connection.close()


def test_initial_telemetry_cannot_block_training_start() -> None:
    """The initial tracking callback runs only on the reporter thread."""
    entered = threading.Event()
    release = threading.Event()

    def block(_: object) -> None:
        entered.set()
        release.wait(timeout=2)

    monitor = HubAccessHealthMonitor(report=block)
    started_at = time.monotonic()
    monitor.start()
    try:
        assert time.monotonic() - started_at < 0.5
        assert entered.wait(timeout=1)
    finally:
        release.set()
        monitor.stop()


def test_monitor_observes_independently_configured_sibling_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank zero observes a sibling rank that did not inherit its environment."""
    run_key = f"distributed-{os.getpid()}"
    reports: list[dict[str, float]] = []
    monkeypatch.delenv(HUB_ACCESS_HEALTH_DIRECTORY_ENV, raising=False)
    monkeypatch.setattr(hub_health, "_POLL_SECONDS", 0.01)
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_publish_sibling_rank_marker, args=(run_key, child_connection)
    )
    process.start()
    child_connection.close()
    assert parent_connection.poll(10)
    assert parent_connection.recv() is True

    with HubAccessHealthScope(run_key=run_key):
        monitor = HubAccessHealthMonitor(
            report=lambda metrics: reports.append(dict(metrics)), heartbeat_seconds=0.02
        )
        monitor.start()
        try:
            _wait_until(lambda: any(r["health/hub_access_blocked"] for r in reports))
            parent_connection.send(True)
            process.join(timeout=10)
            assert process.exitcode == 0
            _wait_until(lambda: reports[-1]["health/hub_access_recoveries_total"] == 1)
        finally:
            monitor.stop()
            if process.is_alive():
                process.terminate()
                process.join()
            parent_connection.close()


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not met before timeout")
        time.sleep(0.01)


def test_monitor_prunes_dead_worker_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markers from terminated workers cannot leave W&B permanently blocked."""
    reports: list[dict[str, float]] = []
    monkeypatch.setattr(hub_health, "_POLL_SECONDS", 0.01)
    monitor = HubAccessHealthMonitor(
        report=lambda metrics: reports.append(dict(metrics)), heartbeat_seconds=0.02
    )
    monitor.start()
    try:
        assert monitor._directory is not None
        stale_marker = monitor._directory / "retrying-99999999"
        stale_marker.touch()
        _wait_until(lambda: not stale_marker.exists())
    finally:
        monitor.stop()

    assert all(report["health/hub_access_blocked"] == 0 for report in reports)


def test_monitor_reports_blocked_heartbeats_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main-process monitor reports fresh blocked state until recovery."""
    reports: list[dict[str, float]] = []
    monkeypatch.setattr(hub_health, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(hub_health, "_marker_depth", 0)
    monitor = HubAccessHealthMonitor(
        report=lambda metrics: reports.append(dict(metrics)), heartbeat_seconds=0.03
    )
    monitor.start()
    try:
        mark_hub_access_retrying()
        _wait_until(
            lambda: sum(report["health/hub_access_blocked"] for report in reports) >= 2
        )
        clear_hub_access_retrying()
        _wait_until(
            lambda: (
                reports[-1]["health/hub_access_blocked"] == 0
                and reports[-1]["health/hub_access_recoveries_total"] == 1
            )
        )
    finally:
        clear_hub_access_retrying()
        monitor.stop()

    assert reports[0]["health/hub_access_blocked"] == 0
    blocked = [report for report in reports if report["health/hub_access_blocked"]]
    assert len(blocked) >= 2
    assert all(report["health/hub_access_retrying_workers"] == 1 for report in blocked)
    assert blocked[-1]["health/hub_access_blocked_seconds"] > 0
    assert HUB_ACCESS_HEALTH_DIRECTORY_ENV not in os.environ


def test_telemetry_failure_does_not_escape(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken tracking callback neither aborts start nor the reporter thread."""
    calls = 0

    def fail(_: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(hub_health, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(hub_health, "_marker_depth", 0)
    monitor = HubAccessHealthMonitor(report=fail, heartbeat_seconds=0.02)
    monitor.start()
    try:
        mark_hub_access_retrying()
        _wait_until(lambda: calls >= 3)
    finally:
        clear_hub_access_retrying()
        monitor.stop()

    assert caplog.text.count("health telemetry failed") == 1


def test_worker_marker_is_nested_and_credential_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nested retry scopes share one process marker and clear it at depth zero."""
    monkeypatch.setenv(HUB_ACCESS_HEALTH_DIRECTORY_ENV, str(tmp_path))
    monkeypatch.setattr(hub_health, "_marker_depth", 0)

    mark_hub_access_retrying()
    mark_hub_access_retrying()

    markers = list(tmp_path.iterdir())
    assert [marker.name for marker in markers] == [f"retrying-{os.getpid()}"]
    assert markers[0].read_bytes() == b""

    clear_hub_access_retrying()
    assert markers[0].exists()
    clear_hub_access_retrying()
    assert not markers[0].exists()
