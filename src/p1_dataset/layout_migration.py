"""Offline migration of pending P1 publication paths.

This module never imports a Hub client and never opens a remote repository.  It only
changes metadata in local ledgers and manifests after complete evidence checks.
"""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import fcntl
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .ledger import Ledger
from .publication_layout import new_shard_path

_LEGACY_NESTED = re.compile(
    r"^data/train/([A-Za-z0-9][A-Za-z0-9_.-]*)/part-([0-9]{5})\.parquet$"
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class LayoutMigrationReport:
    """Aggregate non-sensitive result of one offline migration."""

    ledger_count: int
    shard_count: int
    manifest_count: int
    backup_count: int
    applied: bool


def migrate_layout(
    *,
    run_root: Path,
    expected_digest: str,
    apply: bool,
    publication_lock: Path | None = None,
) -> LayoutMigrationReport:
    """Validate or apply all eight closed partition ledgers under ``run_root``.

    The complete expected digest and a process lock are required in both modes.
    Applying is restart-safe: an already migrated ledger has no legacy records and
    contributes zero changes.

    Returns:
        Aggregate counts without source identifiers or paths.

    Raises:
        ValueError:
            If the run root, digest, ledger set, or evidence is unsafe.
    """
    if not _DIGEST.fullmatch(expected_digest):
        raise ValueError("expected digest must be a complete SHA-256")
    root = run_root.expanduser().resolve()
    ledgers = tuple(sorted(root.glob("partition-*/ledger.sqlite")))
    if len(ledgers) != 8:
        raise ValueError("exactly eight partition ledgers are required")
    lock_path = (publication_lock or root / "publish.lock").expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        manifests = _manifest_paths(root=root, ledgers=ledgers)
        plans = tuple(_plan_ledger(path, expected_digest) for path in ledgers)
        mappings = {old: new for plan in plans for old, new in plan.items()}
        targets = tuple(mappings.values())
        if len(targets) != len(set(targets)):
            raise ValueError("path mapping contains a collision")
        if not apply:
            return LayoutMigrationReport(
                ledger_count=8,
                shard_count=len(mappings),
                manifest_count=len(_manifest_rewrites(manifests, mappings)),
                backup_count=0,
                applied=False,
            )
        backup_root = _make_backup_root(root)
        backups = _backup_files(backup_root, (*ledgers, *manifests))
        try:
            changed = 0
            for path, mapping in zip(ledgers, plans, strict=True):
                with Ledger(
                    path, pipeline_digest=expected_digest, reset_processing=False
                ) as ledger:
                    changed += ledger.remap_remote_paths(mapping)
            manifest_rewrites = _manifest_rewrites(manifests, mappings)
            for path, payload in manifest_rewrites.items():
                _write_durable(path, payload)
        except BaseException:
            _restore_backups(backups)
            raise
        return LayoutMigrationReport(
            ledger_count=8,
            shard_count=changed,
            manifest_count=len(manifest_rewrites),
            backup_count=len(backups),
            applied=True,
        )


def _backup_files(root: Path, paths: c.Sequence[Path]) -> dict[Path, Path]:
    backups: dict[Path, Path] = {}
    for index, source in enumerate(paths):
        if not source.is_file() or source.is_symlink():
            continue
        target = root / f"{index:04d}.backup"
        shutil.copyfile(source, target)
        descriptor = os.open(target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        backups[source] = target
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return backups


def _make_backup_root(root: Path) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / "publication-layout-backups" / stamp
    target.mkdir(parents=True, exist_ok=False)
    return target


def _manifest_paths(*, root: Path, ledgers: c.Sequence[Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for ledger in ledgers:
        paths.update(
            item
            for item in ledger.parent.rglob("manifests/*.json")
            if item.is_file() and not item.is_symlink()
        )
    paths.update(
        item
        for item in root.glob("manifests/*.json")
        if item.is_file() and not item.is_symlink()
    )
    return tuple(sorted(paths))


def _manifest_rewrites(
    manifests: c.Sequence[Path], mapping: c.Mapping[str, str]
) -> dict[Path, bytes]:
    rewrites: dict[Path, bytes] = {}
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        rewritten = _replace_paths(payload, mapping)
        if rewritten != payload:
            rewrites[path] = json.dumps(
                rewritten, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
    return rewrites


def _replace_paths(value: object, mapping: c.Mapping[str, str]) -> object:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_paths(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, mapping) for key, item in value.items()}
    return value


def _plan_ledger(path: Path, expected_digest: str) -> dict[str, str]:
    with Ledger(
        path, pipeline_digest=expected_digest, reset_processing=False
    ) as ledger:
        records = ledger.pending_shards()
        mapping: dict[str, str] = {}
        for record in records:
            if record.path.startswith("data-shards/train/"):
                continue
            if not record.path.startswith("data/train/"):
                raise ValueError("ledger contains a noncanonical shard root")
            if record.local_path is None:
                raise ValueError("legacy shard has no local evidence")
            old, deterministic_id, ordinal = _legacy_parts(record.path)
            target = new_shard_path(deterministic_id, ordinal)
            if old in mapping and mapping[old] != target:
                raise ValueError("legacy path has inconsistent deterministic mapping")
            mapping[old] = target
        return mapping


def _legacy_parts(path: str) -> tuple[str, str, int]:
    match = _LEGACY_NESTED.fullmatch(path)
    if match is not None:
        return path, match.group(1), int(match.group(2))
    raise ValueError("legacy shard path is not canonical")


def _restore_backups(backups: c.Mapping[Path, Path]) -> None:
    for destination, source in backups.items():
        shutil.copyfile(source, destination)
        descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_durable(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.layout.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["LayoutMigrationReport", "migrate_layout"]
