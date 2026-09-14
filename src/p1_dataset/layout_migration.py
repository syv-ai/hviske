"""Offline migration of pending P1 publication paths.

This module never imports a Hub client and never opens a remote repository. It only
changes metadata in local ledgers and manifests after complete evidence checks.
"""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .ledger import Ledger
from .publication_layout import is_allowed_shard_path, new_shard_path

_LEGACY_FLAT = re.compile(
    r"^data/train/([A-Za-z0-9][A-Za-z0-9_.-]*)-([0-9]{5})\.parquet$"
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PARTITIONS = tuple(range(8))


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _assert_snapshot(*, path: Path, snapshots: c.Mapping[Path, bytes | None]) -> None:
    """Ensure opening a writable ledger did not mutate its database artefacts.

    Raises:
        ValueError:
            If opening the ledger changed its primary file or a SQLite sidecar.
    """
    for artifact in _ledger_artifacts(path):
        expected = snapshots[artifact]
        if artifact.is_symlink() or (artifact.exists() and not artifact.is_file()):
            raise ValueError("writable ledger open changed the original database")
        actual = artifact.read_bytes() if artifact.is_file() else None
        if actual != expected:
            raise ValueError("writable ledger open changed the original database")


def _ledger_artifacts(path: Path) -> tuple[Path, ...]:
    """Return a ledger and all SQLite sidecars that must be restorable."""
    return (path, Path(f"{path}-journal"), Path(f"{path}-wal"), Path(f"{path}-shm"))


def _backup_files(
    root: Path, snapshots: c.Mapping[Path, bytes | None]
) -> dict[Path, Path]:
    backups: dict[Path, Path] = {}
    for index, (source, payload) in enumerate(snapshots.items()):
        if payload is None:
            continue
        target = root / f"{index:04d}.backup"
        _write_durable(target, payload)
        backups[source] = target
    _fsync_directory(root)
    return backups


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.layout.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _complete_migration_marker(root: Path) -> None:
    supervisor = root / "supervisor"
    failed = supervisor / "FAILED"
    done = supervisor / "DONE"
    os.replace(failed, done)
    _fsync_directory(supervisor)


def _make_backup_root(root: Path) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / "publication-layout-backups" / stamp
    target.mkdir(parents=True, exist_ok=False)
    _fsync_directory(target.parent)
    return target


def _marker_paths(root: Path) -> tuple[Path, ...]:
    marker_root = root / "supervisor" / "markers"
    paths: list[Path] = [root / "supervisor" / "FAILED", root / "supervisor" / "DONE"]
    for index in _PARTITIONS:
        paths.extend(
            (
                marker_root / f"partition-{index}.FAILED",
                marker_root / f"partition-{index}.DONE",
            )
        )
    return tuple(paths)


def _partition_ledgers(root: Path) -> tuple[Path, ...]:
    ledgers = tuple(
        root / f"partition-{index}" / "ledger.sqlite" for index in _PARTITIONS
    )
    if any(not path.is_file() or path.is_symlink() for path in ledgers):
        raise ValueError("exactly partition-0 through partition-7 ledgers are required")
    extras = tuple(root.glob("partition-*/ledger.sqlite"))
    if len(extras) != len(_PARTITIONS) or set(extras) != set(ledgers):
        raise ValueError("exactly partition-0 through partition-7 ledgers are required")
    return ledgers


@dataclass(frozen=True)
class _BatchPlan:
    """Selected ledger rows belonging to one uncommitted publication group."""

    batch_id: str
    sealed: bool
    values: dict[str, object]
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _Occurrence:
    """One selected legacy occurrence and its destination."""

    source: str
    target: str
    batch_id: str
    local_path: str


@dataclass(frozen=True)
class _LedgerPlan:
    """All migration evidence collected from one partition ledger."""

    path: Path
    occurrences: tuple[_Occurrence, ...]
    batches: tuple[_BatchPlan, ...]
    all_paths: frozenset[str]


def _plan_ledger(path: Path, expected_digest: str) -> _LedgerPlan:
    connection = _open_read_only_ledger(path=path)
    try:
        _validate_read_only_schema(
            connection=connection, expected_digest=expected_digest
        )
        rows = tuple(
            dict(row)
            for row in connection.execute(
                """SELECT s.shard_id, s.batch_id, s.state, s.path, s.byte_size,
                s.row_count, s.sha256, s.local_path,
                b.state AS batch_state, b.commit_id AS batch_commit_id,
                b.sealed AS batch_sealed, b.programme_count AS batch_programme_count,
                b.row_count AS batch_row_count, b.rejection_counts AS batch_rejections
                FROM shards AS s LEFT JOIN batches AS b ON b.batch_id = s.batch_id
                ORDER BY s.shard_id"""
            ).fetchall()
        )
    except sqlite3.Error as error:
        raise ValueError("ledger schema is unreadable") from error
    finally:
        connection.close()
    all_paths = frozenset(str(row["path"]) for row in rows)
    selected = tuple(
        row
        for row in rows
        if row["state"] == "sharded"
        and row["batch_state"] == "sharded"
        and row["batch_commit_id"] is None
        and row["local_path"] is not None
        and isinstance(row["batch_id"], str)
    )
    occurrences: list[_Occurrence] = []
    batches: dict[str, _BatchPlan] = {}
    for row in selected:
        path_value = str(row["path"])
        if path_value.startswith("data/train/"):
            source, deterministic_id, ordinal = _legacy_parts(path_value)
            target = new_shard_path(deterministic_id, ordinal)
            local_path = str(row["local_path"])
            _validate_local_evidence(row=row, local_path=Path(local_path))
            occurrences.append(
                _Occurrence(
                    source=source,
                    target=target,
                    batch_id=str(row["batch_id"]),
                    local_path=local_path,
                )
            )
        elif not is_allowed_shard_path(path_value):
            raise ValueError("selected shard path is not canonical")
        batch_id = str(row["batch_id"])
        batch_rows = tuple(
            item
            for item in rows
            if item["batch_id"] == batch_id
            and item["batch_state"] == "sharded"
            and item["batch_commit_id"] is None
        )
        batches[batch_id] = _BatchPlan(
            batch_id=batch_id,
            sealed=bool(row["batch_sealed"]),
            values=row,
            rows=batch_rows,
        )
    return _LedgerPlan(
        path=path,
        occurrences=tuple(occurrences),
        batches=tuple(batches.values()),
        all_paths=all_paths,
    )


def _legacy_parts(path: str) -> tuple[str, str, int]:
    match = _LEGACY_FLAT.fullmatch(path)
    if match is None:
        raise ValueError("legacy shard path is not canonical")
    return path, match.group(1), int(match.group(2))


def _open_read_only_ledger(*, path: Path) -> sqlite3.Connection:
    """Open a ledger without permitting SQLite to write any database artefact.

    Returns:
        An immutable, query-only SQLite connection.

    Raises:
        ValueError:
            If the path is not a readable SQLite database or cannot be configured
            for read-only access.
    """
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro&immutable=1",
            isolation_level=None,
            uri=True,
        )
    except sqlite3.Error as error:
        raise ValueError("ledger is not a readable SQLite database") from error
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
    except sqlite3.Error as error:
        connection.close()
        raise ValueError("ledger read-only configuration failed") from error
    return connection


def _validate_local_evidence(*, row: dict[str, object], local_path: Path) -> None:
    if local_path.is_symlink() or not local_path.is_file():
        raise ValueError("legacy shard local evidence is unavailable")
    digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if digest != row["sha256"] or local_path.stat().st_size != row["byte_size"]:
        raise ValueError("legacy shard local evidence changed")


def _validate_read_only_schema(
    *, connection: sqlite3.Connection, expected_digest: str
) -> None:
    """Reject ledgers that a writable ``Ledger`` open would have to migrate.

    Raises:
        ValueError:
            If the schema, version, or pipeline digest is incompatible.
    """
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != 3:
        raise ValueError("ledger schema version is not current")
    required_columns = {
        "programmes": {"pipeline_digest"},
        "batches": {
            "batch_id",
            "state",
            "pipeline_digest",
            "commit_id",
            "programme_count",
            "row_count",
            "rejection_counts",
            "sealed",
        },
        "shards": {
            "shard_id",
            "batch_id",
            "state",
            "path",
            "byte_size",
            "row_count",
            "sha256",
            "local_path",
        },
        "ledger_metadata": {"key", "value"},
    }
    for table, required in required_columns.items():
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not required.issubset(columns):
            raise ValueError("ledger schema is missing required columns")
    metadata = connection.execute(
        "SELECT value FROM ledger_metadata WHERE key = 'pipeline_digest'"
    ).fetchone()
    if metadata is None or str(metadata[0]) != expected_digest:
        raise ValueError("ledger pipeline digest is not bound to the expected digest")
    stored = {
        str(row[0])
        for row in connection.execute(
            """SELECT pipeline_digest FROM programmes
            UNION SELECT pipeline_digest FROM batches"""
        )
    }
    if stored != {expected_digest} and stored:
        raise ValueError("ledger records use a different pipeline digest")


@dataclass(frozen=True)
class _ManifestPlan:
    """A validated owned manifest and its optional rewritten bytes."""

    path: Path
    rewritten: bytes | None


def _plan_manifests(
    *, root: Path, plans: c.Sequence[_LedgerPlan], occurrences: c.Sequence[_Occurrence]
) -> tuple[_ManifestPlan, ...]:
    mapping = {item.source: item.target for item in occurrences}
    output: list[_ManifestPlan] = []
    for plan in plans:
        for batch in plan.batches:
            candidates = {root / "manifests" / f"{batch.batch_id}.json"}
            candidates.update(
                Path(str(row["local_path"])).parent
                / "manifests"
                / f"{batch.batch_id}.json"
                for row in batch.rows
                if row["local_path"] is not None
            )
            existing = tuple(
                path
                for path in sorted(candidates)
                if path.exists() or path.is_symlink()
            )
            if batch.sealed and not existing:
                raise ValueError("sealed pending group has no manifest")
            for path in existing:
                payload = _read_manifest(path)
                _validate_manifest(payload=payload, batch=batch)
                rewritten_payload = _replace_paths(payload, mapping)
                rewritten = None
                if rewritten_payload != payload:
                    rewritten = _json_bytes(rewritten_payload)
                output.append(_ManifestPlan(path=path, rewritten=rewritten))
    return tuple(output)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_manifest(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError("pending manifest is not a regular file")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("pending manifest is unreadable or malformed") from error


def _replace_paths(value: object, mapping: c.Mapping[str, str]) -> object:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_paths(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, mapping) for key, item in value.items()}
    return value


def _validate_manifest(*, payload: object, batch: _BatchPlan) -> None:
    if not isinstance(payload, dict):
        raise ValueError("pending manifest must be a JSON object")
    expected_keys = {
        "batch_id",
        "programme_count",
        "row_count",
        "rejection_counts",
        "shards",
    }
    if set(payload) != expected_keys or payload["batch_id"] != batch.batch_id:
        raise ValueError("pending manifest has the wrong schema or batch")
    if not _nonnegative_int(payload["programme_count"]):
        raise ValueError("pending manifest has invalid programme_count")
    if not _nonnegative_int(payload["row_count"]):
        raise ValueError("pending manifest has invalid row_count")
    if payload["programme_count"] != batch.values["batch_programme_count"]:
        raise ValueError("pending manifest programme count differs from ledger")
    if payload["row_count"] != batch.values["batch_row_count"]:
        raise ValueError("pending manifest row count differs from ledger")
    rejections = payload["rejection_counts"]
    if not isinstance(rejections, dict) or any(
        not isinstance(key, str) or not _nonnegative_int(value)
        for key, value in rejections.items()
    ):
        raise ValueError("pending manifest has invalid rejection counts")
    try:
        ledger_rejections = json.loads(str(batch.values["batch_rejections"]))
    except json.JSONDecodeError as error:
        raise ValueError("ledger has invalid rejection counts") from error
    if rejections != ledger_rejections:
        raise ValueError("pending manifest rejection counts differ from ledger")
    shards = payload["shards"]
    if not isinstance(shards, list) or len(shards) != len(batch.rows):
        raise ValueError("pending manifest does not contain a complete shard mapping")
    for item, row in zip(shards, batch.rows, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "byte_size",
            "row_count",
            "sha256",
        }:
            raise ValueError("pending manifest has an invalid shard entry")
        if not isinstance(item["path"], str) or not is_allowed_shard_path(item["path"]):
            raise ValueError("pending manifest has a noncanonical shard path")
        if any(
            item[key] != row[key]
            for key in ("path", "byte_size", "row_count", "sha256")
        ):
            raise ValueError("pending manifest shard evidence differs from ledger")


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _reject_duplicate_mappings(occurrences: c.Sequence[_Occurrence]) -> None:
    sources = [item.source for item in occurrences]
    targets = [item.target for item in occurrences]
    if len(sources) != len(set(sources)):
        raise ValueError("a legacy source occurs more than once")
    if len(targets) != len(set(targets)):
        raise ValueError("path mapping contains a collision")


def _reject_existing_targets(
    *, plans: c.Sequence[_LedgerPlan], occurrences: c.Sequence[_Occurrence]
) -> None:
    existing = set().union(*(set(plan.all_paths) for plan in plans))
    if existing.intersection(item.target for item in occurrences):
        raise ValueError("path mapping collides with existing ledger evidence")


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

    Only local shards in uncommitted, sharded publication groups are selected. A
    stopped supervisor's aggregate FAILED marker is required before the first run;
    a successful migration changes it to DONE. No Parquet file is written.

    Args:
        run_root:
            Root containing ``partition-0`` through ``partition-7`` and supervisor
            evidence.
        expected_digest:
            Complete pipeline SHA-256 expected by every ledger.
        apply:
            Whether to perform the metadata migration rather than only validate it.
        publication_lock (optional):
            Shared lock used to exclude publishers while inspecting and applying.

    Returns:
        Aggregate counts without source identifiers or paths.

    Raises:
        ValueError:
            If the run root, digest, ledger set, run evidence, or manifests is unsafe.
    """
    if not _DIGEST.fullmatch(expected_digest):
        raise ValueError("expected digest must be a complete SHA-256")
    root = run_root.expanduser().resolve()
    ledgers = _partition_ledgers(root)
    lock_path = (publication_lock or root / "publish.lock").expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        plans = tuple(_plan_ledger(path, expected_digest) for path in ledgers)
        occurrences = tuple(item for plan in plans for item in plan.occurrences)
        _reject_duplicate_mappings(occurrences)
        _reject_existing_targets(plans=plans, occurrences=occurrences)
        manifests = _plan_manifests(root=root, plans=plans, occurrences=occurrences)
        has_legacy = bool(occurrences)
        marker_state = _validate_run_markers(root=root)

        # A completed migration is the only permitted markerless-looking rerun. It
        # still scans every ledger above, so a newly reintroduced legacy path cannot
        # silently be treated as already migrated.
        if marker_state == "done":
            if has_legacy:
                raise ValueError("completed migration still has eligible legacy paths")
            return _report(apply=apply, manifest_count=0, shard_count=0, backup_count=0)

        if not apply:
            return _report(
                apply=False,
                manifest_count=sum(item.rewritten is not None for item in manifests),
                shard_count=len(occurrences),
                backup_count=0,
            )

        touched = (
            *(
                database_path
                for ledger_path in ledgers
                for database_path in _ledger_artifacts(ledger_path)
            ),
            *(item.path for item in manifests),
            *_marker_paths(root),
        )
        snapshots = _snapshot_files(touched)
        backup_root = _make_backup_root(root)
        backups = _backup_files(backup_root, snapshots)
        try:
            changed = 0
            for plan in plans:
                mapping = {item.source: item.target for item in plan.occurrences}
                if mapping:
                    with Ledger(
                        plan.path,
                        pipeline_digest=expected_digest,
                        reset_processing=False,
                    ) as ledger:
                        _assert_snapshot(path=plan.path, snapshots=snapshots)
                        changed += ledger.remap_remote_paths(mapping)
            for item in manifests:
                if item.rewritten is not None:
                    _write_durable(item.path, item.rewritten)
            _complete_migration_marker(root)
        except BaseException:
            _restore_snapshots(snapshots)
            raise
        return _report(
            apply=True,
            manifest_count=sum(item.rewritten is not None for item in manifests),
            shard_count=changed,
            backup_count=len(backups),
        )


def _report(
    *, apply: bool, manifest_count: int, shard_count: int, backup_count: int
) -> LayoutMigrationReport:
    return LayoutMigrationReport(
        ledger_count=8,
        shard_count=shard_count,
        manifest_count=manifest_count,
        backup_count=backup_count,
        applied=apply,
    )


def _restore_snapshots(snapshots: c.Mapping[Path, bytes | None]) -> None:
    for path, payload in snapshots.items():
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            _write_durable(path, payload)
    for path in snapshots:
        _fsync_directory(path.parent)


def _snapshot_files(paths: c.Sequence[Path]) -> dict[Path, bytes | None]:
    snapshots: dict[Path, bytes | None] = {}
    for path in paths:
        if path in snapshots:
            continue
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("migration evidence is not a regular file")
        snapshots[path] = path.read_bytes() if path.is_file() else None
    return snapshots


def _validate_run_markers(*, root: Path) -> str:
    supervisor = root / "supervisor"
    failed = supervisor / "FAILED"
    done = supervisor / "DONE"
    if done.exists() and failed.exists():
        raise ValueError("supervisor has inconsistent aggregate markers")
    if done.exists():
        if done.is_symlink() or not done.is_file():
            raise ValueError("supervisor DONE marker is unsafe")
        return "done"
    if failed.is_symlink() or not failed.is_file():
        raise ValueError("stopped run requires an aggregate FAILED marker")
    aggregate = _read_marker_text(failed)
    aggregate_lines = aggregate.splitlines()
    aggregate_matches = [
        re.fullmatch(r"partition=([0-9]+) attempts=[0-9]+", line)
        for line in aggregate_lines
    ]
    if not aggregate_lines or any(match is None for match in aggregate_matches):
        raise ValueError("aggregate FAILED marker is malformed")
    aggregate_indexes = [int(match.group(1)) for match in aggregate_matches if match]
    if (
        not aggregate_indexes
        or len(aggregate_indexes) != len(set(aggregate_indexes))
        or not set(aggregate_indexes).issubset(_PARTITIONS)
    ):
        raise ValueError("aggregate FAILED marker is inconsistent")

    # The aggregate marker is the library's closed-run proof. Partition markers are
    # optional because older supervisors only emitted the aggregate file; when they
    # are present, however, every one must agree with it.
    marker_root = supervisor / "markers"
    individual_paths = (
        tuple(marker_root.glob("partition-*.DONE"))
        + tuple(marker_root.glob("partition-*.FAILED"))
        if marker_root.is_dir() and not marker_root.is_symlink()
        else ()
    )
    if individual_paths:
        individual: dict[int, str] = {}
        for path in individual_paths:
            match = re.fullmatch(r"partition-([0-9]+)\.(DONE|FAILED)", path.name)
            if match is None or path.is_symlink() or not path.is_file():
                raise ValueError("partition completion markers are unsafe")
            index = int(match.group(1))
            if index not in _PARTITIONS or index in individual:
                raise ValueError("partition completion markers are inconsistent")
            individual[index] = match.group(2)
            _validate_partition_marker(path=path, index=index)
        if set(individual) != set(_PARTITIONS):
            raise ValueError(
                "exactly partition-0 through partition-7 markers are required"
            )
        actual_failed = {
            index for index, state in individual.items() if state == "FAILED"
        }
        if set(aggregate_indexes) != actual_failed:
            raise ValueError("aggregate FAILED marker is inconsistent")
    return "failed"


def _read_marker_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("completion marker is unreadable") from error


def _validate_partition_marker(*, path: Path, index: int) -> None:
    lines = _read_marker_text(path).splitlines()
    if (
        len(lines) < 2
        or lines[0] != f"partition={index}"
        or not re.fullmatch(r"attempts=[0-9]+", lines[1])
    ):
        raise ValueError("partition completion marker is malformed")


__all__ = ["LayoutMigrationReport", "migrate_layout"]
