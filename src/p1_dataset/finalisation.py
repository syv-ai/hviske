"""Metadata-only finalisation of the public P1 v8 release."""

from __future__ import annotations

import collections.abc as c
import json
import os
import re
import sqlite3
import tempfile
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from .contracts import OUTPUT_SCHEMA
from .publication_layout import is_allowed_shard_path
from .publish import _EXPECTED_ARROW_SCHEMA

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PARTITIONS = tuple(range(8))
_REPOSITORY = "syvai/p1-segments"
_METADATA_COLUMNS = tuple(
    field.name for field in OUTPUT_SCHEMA.fields if field.name != "audio"
)
_TERMINAL_PROGRAMME_STATES = {"purged", "rejected"}


@dataclass(frozen=True)
class _Shard:
    path: str
    byte_size: int
    row_count: int
    sha256: str


class _LedgerSummary(t.TypedDict):
    shards: tuple[_Shard, ...]
    rows: int
    programme_states: dict[str, int]
    batches: dict[str, int]


@dataclass
class _Stats:
    """Bounded scalar statistics for one numeric column."""

    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    sample: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.total += value
        if len(self.sample) < 2048:
            self.sample.append(value)
        else:
            slot = (self.count * 1103515245 + 12345) % self.count
            if slot < len(self.sample):
                self.sample[slot] = value

    def as_dict(self) -> dict[str, int | float | None]:
        ordered = sorted(self.sample)

        def percentile(position: float) -> float | None:
            if not ordered:
                return None
            return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * position))]

        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.total / self.count if self.count else None,
            "median": percentile(0.5),
            "p90": percentile(0.9),
        }


class _ScanStats(t.TypedDict):
    rows: int
    programmes: int
    duration_ms: int
    duration: _Stats
    words: _Stats
    speakers: dict[str, int]


def finalise_public_corpus(
    *,
    hub: object,
    run_root: Path | str,
    revision: str,
    expected_pipeline_config_sha256: str,
    repository: str = "syvai/p1-segments",
    report_path: Path | str | None = None,
    update_card: bool = False,
) -> dict[str, object]:
    """Validate the completed run and optionally CAS-update its public card.

    The scan opens Parquet footers and streams only non-audio columns.  It never
    calls ``load_dataset`` and therefore cannot decode embedded audio.

    Args:
        hub:
            Hub adapter exposing repository, inventory, metadata and file methods.
        run_root:
            Completed eight-partition production run root.
        revision:
            Complete production commit SHA.
        expected_pipeline_config_sha256:
            Pipeline digest recorded by every ledger and row.
        repository (optional):
            Dataset repository identifier. Defaults to ``syvai/p1-segments``.
        report_path (optional):
            Atomically written aggregate JSON destination.
        update_card (optional):
            Whether to perform the separate card CAS operation. Defaults to False.

    Returns:
        A sanitised aggregate report. The ``final_head`` value is only present after
        a successful card update.

    Raises:
        FinalisationError:
            If any release invariant is not proven.
    """
    _validate_identity(revision, expected_pipeline_config_sha256)
    if repository != _REPOSITORY:
        raise FinalisationError("the P1 finaliser has a fixed repository")
    ledger: _LedgerSummary = _read_run_root(
        Path(run_root), expected_pipeline_config_sha256, revision
    )
    remote_files = _remote_inventory(hub, repository, revision)
    remote_shards = {path for path in remote_files if path.endswith(".parquet")}
    expected_shards = {item.path for item in ledger["shards"]}
    if remote_shards != expected_shards:
        raise FinalisationError("remote Parquet inventory differs from the ledgers")
    if len(remote_shards) != len(expected_shards):
        raise FinalisationError("remote Parquet paths are not one-to-one")
    _check_remote_metadata(hub, repository, revision, ledger["shards"])
    _check_manifests(Path(run_root), ledger["shards"])
    stats = _scan_parquet(
        hub, repository, revision, ledger["shards"], expected_pipeline_config_sha256
    )
    if stats["rows"] != ledger["rows"]:
        raise FinalisationError("remote row count differs from the ledgers")

    report: dict[str, object] = {
        "report_type": "p1-v8-public-finalisation",
        "revision": revision,
        "pipeline_version": "p1-segmentation-8",
        "pipeline_config_sha256": expected_pipeline_config_sha256,
        "checks": {
            "public_immutable_revision": True,
            "completed_run": True,
            "exact_eight_ledgers": True,
            "remote_inventory": True,
            "exact_schema": True,
            "metadata_only_scan": True,
            "structural_rows": True,
            "no_duplicate_segment_ids": True,
            "no_overlapping_intervals": True,
            "immutable_remote_metadata": True,
        },
        "counts": {
            "shards": len(expected_shards),
            "rows": stats["rows"],
            "programmes": stats["programmes"],
            "programme_states": ledger["programme_states"],
            "batches": ledger["batches"],
        },
        "audio": {
            "duration_ms": stats["duration_ms"],
            "hours": stats["duration_ms"] / 3_600_000,
        },
        "totals": {
            "rows": stats["rows"],
            "shards": len(expected_shards),
            "programmes": stats["programmes"],
            "audio_duration_ms": stats["duration_ms"],
            "parquet_bytes": sum(item.byte_size for item in ledger["shards"]),
        },
        "distributions": {
            "duration_ms": stats["duration"].as_dict(),
            "word_count": stats["words"].as_dict(),
        },
        "speaker_counts": dict(sorted(stats["speakers"].items())),
        "pass": True,
    }
    if update_card:
        final_head = _cas_card_update(
            hub=hub,
            repository=repository,
            revision=revision,
            report=report,
            remote_files=remote_files,
        )
        report["final_head"] = final_head
        report["checks"]["card_cas"] = True  # type: ignore[index]
    _write_report(report_path, report)
    return report


class FinalisationError(ValueError):
    """Raised when a reproducible P1 release cannot be proven."""


# American spelling is intentionally available for integrations that use it.
finalize_public_corpus = finalise_public_corpus


def _cas_card_update(
    *,
    hub: object,
    repository: str,
    revision: str,
    report: dict[str, object],
    remote_files: c.Sequence[str],
) -> str:
    reader = getattr(hub, "stream_file", None)
    committer = getattr(hub, "create_commit", None)
    if not callable(reader) or not callable(committer):
        raise FinalisationError("card CAS is unavailable")
    try:
        before = b"".join(
            reader(repository, "README.md", repo_type="dataset", revision=revision)
        )
        license_before = _remote_digest(hub, repository, "LICENSE", revision)
        current_files = tuple(_remote_inventory(hub, repository, revision))
    except FinalisationError:
        raise
    except Exception:
        raise FinalisationError("card metadata is unavailable") from None
    if tuple(current_files) != tuple(remote_files) or not _four_sections(before):
        raise FinalisationError("card or inventory changed during CAS preparation")
    updated = _add_card_statistics(before.decode("utf-8"), report).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="p1-card-") as temporary:
        path = Path(temporary) / "README.md"
        path.write_bytes(updated)
        from .publish import UploadOperation

        try:
            response = committer(
                repository,
                (UploadOperation("README.md", path),),
                repo_type="dataset",
                commit_message="docs: add final P1 statistics",
                parent_commit=revision,
            )
        except Exception:
            raise FinalisationError("card CAS failed") from None
    new_head = (
        _object_value(response, "commit_id")
        or _object_value(response, "oid")
        or _object_value(response, "sha")
    )
    if not isinstance(new_head, str) or _COMMIT_SHA.fullmatch(new_head) is None:
        raise FinalisationError("card CAS returned no immutable SHA")
    _require_public_head(hub, repository, new_head)
    if _remote_digest(hub, repository, "LICENSE", new_head) != license_before:
        raise FinalisationError("card CAS changed the licence")
    if set(_remote_inventory(hub, repository, new_head)) != set(remote_files):
        raise FinalisationError("card CAS changed the remote inventory")
    try:
        after = b"".join(
            reader(repository, "README.md", repo_type="dataset", revision=new_head)
        )
    except Exception:
        raise FinalisationError("card CAS could not verify README bytes") from None
    if after != updated:
        raise FinalisationError("card CAS returned unexpected README bytes")
    return new_head


def _add_card_statistics(card: str, report: dict[str, object]) -> str:
    marker = "\n\n## Final structural statistics\n"
    if marker in card:
        raise FinalisationError("card already contains final statistics")
    counts = t.cast(dict[str, int], report["counts"])
    audio = t.cast(dict[str, float | int], report["audio"])
    block = (
        f"{marker}\n"
        f"- Segments: {counts['rows']:,}\n"
        f"- Shards: {counts['shards']:,}\n"
        f"- Audio duration: {audio['hours']:.2f} hours\n"
    )
    return card.rstrip("\n") + block


def _four_sections(card: bytes) -> bool:
    try:
        text = card.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(
        f"## {name}" in text for name in ("Dataset", "Source", "Access", "Licence")
    )


def _object_value(value: object, name: str) -> object | None:
    if isinstance(value, c.Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _remote_digest(hub: object, repository: str, path: str, revision: str) -> str:
    getter = getattr(hub, "get_paths_info", None)
    if not callable(getter):
        raise FinalisationError("remote metadata is unavailable")
    try:
        values = tuple(
            getter(repository, [path], repo_type="dataset", revision=revision)
        )
    except Exception:
        raise FinalisationError("remote metadata is unavailable") from None
    if len(values) != 1:
        raise FinalisationError("remote metadata is incomplete")
    digest = _object_value(values[0], "sha256") or _object_value(values[0], "oid")
    if not isinstance(digest, str):
        raise FinalisationError("remote metadata has no immutable digest")
    return digest


def _remote_inventory(hub: object, repository: str, revision: str) -> tuple[str, ...]:
    _require_public_head(hub, repository, revision)
    getter = getattr(hub, "list_repo_files", None)
    if not callable(getter):
        raise FinalisationError("Hub inventory is unavailable")
    try:
        values = tuple(getter(repository, repo_type="dataset", revision=revision))
    except Exception:
        raise FinalisationError("Hub inventory is unavailable") from None
    paths_list: list[str] = []
    for item in values:
        path = _object_value(item, "path")
        if path is None:
            path = item
        if not isinstance(path, str) or "\x00" in path:
            raise FinalisationError("Hub inventory is malformed")
        paths_list.append(path)
    paths = tuple(paths_list)
    if len(paths) != len(set(paths)):
        raise FinalisationError("Hub inventory contains duplicate paths")
    return paths


def _require_public_head(hub: object, repository: str, revision: str) -> None:
    getter = getattr(hub, "repo_info", None)
    if not callable(getter):
        raise FinalisationError("Hub visibility is unverifiable")
    try:
        info = getter(repository, repo_type="dataset", revision=revision)
    except Exception:
        raise FinalisationError("Hub visibility is unverifiable") from None
    private = _object_value(info, "private")
    resolved = (
        _object_value(info, "sha")
        or _object_value(info, "oid")
        or _object_value(info, "commit_id")
    )
    if private is not False or resolved != revision:
        raise FinalisationError(
            "the pinned dataset is not public at the requested revision"
        )


def _check_manifests(root: Path, shards: c.Sequence[_Shard]) -> None:
    """Cross-check retained publication manifests without payload access.

    Raises:
        FinalisationError:
            If a retained manifest cannot be reconciled with the shard ledger.
    """
    expected = {item.path: item for item in shards}
    manifest_paths: set[str] = set()
    found_manifest = False
    for path in root.rglob("*"):
        if not path.is_file() or "manifest" not in path.name.lower():
            continue
        found_manifest = True
        if path.stat().st_size > 16 * 1024 * 1024:
            raise FinalisationError("a publication manifest exceeds the metadata bound")
        try:
            values = _manifest_values(path)
            for value in values:
                if not isinstance(value, dict):
                    raise FinalisationError("a publication manifest is malformed")
                entries = value.get("shards") or [value]
                if not isinstance(entries, list):
                    raise FinalisationError("a publication manifest is malformed")
                for entry in entries:
                    if not isinstance(entry, dict):
                        raise FinalisationError("a publication manifest is malformed")
                    shard_path = entry.get("path")
                    if shard_path is None:
                        shard_path = entry.get(
                            "parquet_path", entry.get("remote_parquet_path")
                        )
                    if shard_path is None:
                        continue
                    if not isinstance(shard_path, str):
                        raise FinalisationError(
                            "a publication manifest has an invalid path"
                        )
                    manifest_paths.add(shard_path)
                    evidence = expected.get(shard_path)
                    if evidence is None or entry.get("row_count") != evidence.row_count:
                        raise FinalisationError(
                            "manifest inventory differs from the ledgers"
                        )
                    if entry.get("sha256") not in (None, evidence.sha256):
                        raise FinalisationError(
                            "manifest checksum differs from the ledgers"
                        )
                    if entry.get("byte_size") not in (None, evidence.byte_size):
                        raise FinalisationError(
                            "manifest size differs from the ledgers"
                        )
        except FinalisationError:
            raise
        except (OSError, ValueError, UnicodeError):
            raise FinalisationError("a publication manifest is unreadable") from None
    if found_manifest and manifest_paths != set(expected):
        raise FinalisationError("manifest inventory differs from the ledgers")


def _manifest_values(path: Path) -> c.Iterator[object]:
    """Yield one JSON or JSONL object at a time."""
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    yield json.loads(path.read_text(encoding="utf-8"))


def _check_remote_metadata(
    hub: object, repository: str, revision: str, shards: c.Sequence[_Shard]
) -> None:
    getter = getattr(hub, "get_paths_info", None)
    if not callable(getter):
        raise FinalisationError("remote immutable metadata is unavailable")
    expected = {item.path: item for item in shards}
    paths = tuple(expected)
    for start in range(0, len(paths), 128):
        batch = list(paths[start : start + 128])
        try:
            metadata = tuple(
                getter(repository, batch, repo_type="dataset", revision=revision)
            )
        except Exception:
            raise FinalisationError(
                "remote immutable metadata is unavailable"
            ) from None
        if len(metadata) != len(batch):
            raise FinalisationError("remote immutable metadata is incomplete")
        seen_metadata: set[str] = set()
        for item in metadata:
            path = _object_value(item, "path")
            if not isinstance(path, str) or path in seen_metadata:
                raise FinalisationError("remote immutable metadata is duplicated")
            seen_metadata.add(path)
            evidence = expected.get(path)
            size = _object_value(item, "size") or _object_value(item, "size_bytes")
            digest = _object_value(item, "sha256") or _object_value(item, "oid")
            if (
                evidence is None
                or size != evidence.byte_size
                or digest != evidence.sha256
            ):
                raise FinalisationError(
                    "remote immutable metadata differs from the ledgers"
                )
        if seen_metadata != set(batch):
            raise FinalisationError("remote immutable metadata is incomplete")


def _read_run_root(root: Path, digest: str, revision: str) -> _LedgerSummary:
    if not root.is_dir() or root.is_symlink():
        raise FinalisationError("completed run root is unavailable")
    supervisor = root / "supervisor"
    if not (supervisor / "DONE").is_file() or (supervisor / "FAILED").exists():
        raise FinalisationError("supervisor completion markers are invalid")
    shard_rows: list[_Shard] = []
    accepted_total = 0
    state_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}
    for index in _PARTITIONS:
        partition = root / f"partition-{index}"
        database = partition / "ledger.sqlite"
        if not database.is_file() or database.is_symlink():
            raise FinalisationError("the eight production ledgers are required")
        if not (supervisor / "markers" / f"partition-{index}.DONE").is_file():
            raise FinalisationError("a production partition is not complete")
        if (supervisor / "markers" / f"partition-{index}.FAILED").exists():
            raise FinalisationError("a production partition has failed")
        connection = _open_readonly(database)
        try:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            required = {"programmes", "batches", "shards", "ledger_metadata"}
            if not required.issubset(tables):
                raise FinalisationError(
                    "a production ledger is structurally incomplete"
                )
            metadata = dict(
                connection.execute("SELECT key, value FROM ledger_metadata")
            )
            if metadata.get("pipeline_digest") != digest:
                raise FinalisationError("a ledger has a different pipeline digest")
            for row in connection.execute(
                "SELECT state, COUNT(*), COALESCE(SUM(accepted_count), 0) "
                "FROM programmes GROUP BY state"
            ):
                state = str(row[0])
                state_counts[state] = state_counts.get(state, 0) + int(row[1])
                accepted_total += int(row[2])
                if state not in _TERMINAL_PROGRAMME_STATES:
                    raise FinalisationError("a programme is not in a terminal state")
            for row in connection.execute(
                "SELECT state, publication_artifact_purged_at, COUNT(*) "
                "FROM batches GROUP BY state, publication_artifact_purged_at"
            ):
                state = str(row[0])
                batch_counts[state] = batch_counts.get(state, 0) + int(row[2])
                if state != "purged" or row[1] is None:
                    raise FinalisationError("a publication batch is not fully closed")
            for row in connection.execute(
                "SELECT s.path, s.byte_size, s.row_count, s.sha256, s.state, "
                "b.commit_id, s.local_path FROM shards AS s "
                "JOIN batches AS b ON b.batch_id = s.batch_id"
            ):
                path, size, count, sha256, state, commit_id, local_path = row
                if (
                    not isinstance(path, str)
                    or not is_allowed_shard_path(path)
                    or not isinstance(size, int)
                    or size < 1
                    or not isinstance(count, int)
                    or count < 1
                    or not isinstance(sha256, str)
                    or _SHA256.fullmatch(sha256) is None
                    or state != "purged"
                    or commit_id != revision
                ):
                    raise FinalisationError("a shard has invalid immutable evidence")
                if local_path and Path(str(local_path)).exists():
                    raise FinalisationError("a publication artefact remains staged")
                shard_rows.append(_Shard(path, size, count, sha256))
        finally:
            connection.close()
    extras = tuple(root.glob("partition-*/ledger.sqlite"))
    if len(extras) != 8 or any(
        path.parent.name not in {f"partition-{i}" for i in _PARTITIONS}
        for path in extras
    ):
        raise FinalisationError("the run root contains an unexpected ledger set")
    if not shard_rows:
        raise FinalisationError("the completed corpus contains no shards")
    if accepted_total != sum(item.row_count for item in shard_rows):
        raise FinalisationError("programme accepted counts differ from shard rows")
    if len({item.path for item in shard_rows}) != len(shard_rows):
        raise FinalisationError("a remote shard path occurs more than once")
    for path in root.rglob("*"):
        if path.is_file() and (
            path.suffix in {".parquet", ".tmp", ".partial"}
            or path.name.endswith(("-wal", "-shm"))
            or ".staging" in path.name
        ):
            raise FinalisationError("a staged publication artefact remains")
    return {
        "shards": tuple(shard_rows),
        "rows": sum(item.row_count for item in shard_rows),
        "programme_states": dict(sorted(state_counts.items())),
        "batches": dict(sorted(batch_counts.items())),
    }


def _open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error:
        raise FinalisationError("a production ledger cannot be read") from None


def _scan_parquet(
    hub: object, repository: str, revision: str, shards: c.Sequence[_Shard], digest: str
) -> _ScanStats:
    temporary = tempfile.TemporaryDirectory(prefix="p1-finalisation-")
    seen_db = sqlite3.connect(str(Path(temporary.name) / "metadata.sqlite"))
    seen_db.execute(
        "CREATE TABLE rows ("
        "segment_id TEXT PRIMARY KEY, source TEXT, start INTEGER, end INTEGER)"
    )
    stats: _ScanStats = {
        "rows": 0,
        "programmes": 0,
        "duration_ms": 0,
        "duration": _Stats(),
        "words": _Stats(),
        "speakers": {},
    }
    try:
        for shard in shards:
            handle = _open_remote_file(hub, repository, shard.path, revision)
            try:
                parquet = pq.ParquetFile(handle)
                if parquet.schema_arrow != _EXPECTED_ARROW_SCHEMA:
                    raise FinalisationError(
                        "a Parquet shard has the wrong Arrow schema"
                    )
                batches = parquet.iter_batches(
                    columns=list(_METADATA_COLUMNS), batch_size=1024
                )
                for batch in batches:
                    for row in batch.to_pylist():
                        _check_row(row, digest, seen_db, stats)
            except FinalisationError:
                raise
            except Exception:
                raise FinalisationError("a Parquet shard cannot be scanned") from None
            finally:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()
    finally:
        stats["programmes"] = int(
            seen_db.execute("SELECT COUNT(DISTINCT source) FROM rows").fetchone()[0]
        )
        seen_db.close()
        temporary.cleanup()
    return stats


def _check_row(
    row: object, digest: str, seen_db: sqlite3.Connection, stats: _ScanStats
) -> None:
    if not isinstance(row, dict) or set(row) != set(_METADATA_COLUMNS):
        raise FinalisationError("a row does not have the exact metadata schema")
    strings = (
        "audio_sha256",
        "text",
        "alignment_text",
        "language",
        "segment_id",
        "source_file_id",
        "alignment_score_type",
        "alignment_backend",
        "alignment_method",
        "pipeline_version",
        "pipeline_config_sha256",
    )
    if any(not isinstance(row.get(name), str) or not row[name] for name in strings):
        raise FinalisationError("a row has invalid structural metadata")
    if (
        row["language"] != "da"
        or row["pipeline_version"] != "p1-segmentation-8"
        or row["pipeline_config_sha256"] != digest
        or row["alignment_backend"] != "timestamp-native"
        or row["alignment_method"] != "timestamp-native:p1-transcripts.words"
        or row["alignment_score_type"] != "not_applicable:source_timestamps"
    ):
        raise FinalisationError("a row has a different v8 identity")
    if (
        _SHA256.fullmatch(row["audio_sha256"]) is None
        or _SHA256.fullmatch(row["segment_id"]) is None
    ):
        raise FinalisationError("a row has an invalid digest")
    ints = (
        "source_start_ms",
        "source_end_ms",
        "source_duration_ms",
        "duration_ms",
        "proposal_start_ms",
        "proposal_end_ms",
    )
    if any(type(row.get(name)) is not int for name in ints):
        raise FinalisationError("a row has invalid timing metadata")
    start, end, source_duration = (
        row[name] for name in ("source_start_ms", "source_end_ms", "source_duration_ms")
    )
    if not (0 <= start < end <= source_duration and row["duration_ms"] == end - start):
        raise FinalisationError("a row has invalid source boundaries")
    if not (row["proposal_start_ms"] < row["proposal_end_ms"]):
        raise FinalisationError("a row has invalid proposal boundaries")
    if row["proposal_start_ms"] != start or row["proposal_end_ms"] != end:
        raise FinalisationError("a v8 row has non-native boundaries")
    if any(
        row.get(name) is not None
        for name in (
            "alignment_score",
            "start_drift_ms",
            "end_drift_ms",
            "vad_speech_ratio",
        )
    ):
        raise FinalisationError("a v8 row contains acoustic evidence")
    for name in ("alignment_word_map", "speaker_ids"):
        value = row.get(name)
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise FinalisationError("a row has invalid list metadata")
    if not row["text"].strip() or not row["alignment_text"].strip():
        raise FinalisationError("a row has empty text")
    words = len(row["text"].split())
    try:
        seen_db.execute(
            "INSERT INTO rows VALUES (?, ?, ?, ?)",
            (row["segment_id"], row["source_file_id"], start, end),
        )
    except sqlite3.IntegrityError:
        raise FinalisationError("duplicate segment IDs were found") from None
    overlap = seen_db.execute(
        "SELECT 1 FROM rows WHERE source = ? AND start < ? AND end > ? "
        "AND segment_id != ? LIMIT 1",
        (row["source_file_id"], end, start, row["segment_id"]),
    ).fetchone()
    if overlap is not None:
        raise FinalisationError("overlapping source intervals were found")
    seen_db.commit()
    stats["rows"] = int(stats["rows"]) + 1
    stats["duration_ms"] = int(stats["duration_ms"]) + row["duration_ms"]
    stats["duration"].add(row["duration_ms"])
    stats["words"].add(words)
    speaker_counts = stats["speakers"]
    speaker_counts[str(len(row["speaker_ids"]))] = (
        speaker_counts.get(str(len(row["speaker_ids"])), 0) + 1
    )


def _open_remote_file(hub: object, repository: str, path: str, revision: str) -> object:
    opener = getattr(hub, "open_file", None)
    if callable(opener):
        try:
            return opener(repository, path, repo_type="dataset", revision=revision)
        except Exception:
            raise FinalisationError("remote Parquet cannot be opened") from None
    filesystem = getattr(hub, "filesystem", None)
    if filesystem is not None and callable(getattr(filesystem, "open", None)):
        try:
            return filesystem.open(
                f"datasets/{repository}/{path}", mode="rb", revision=revision
            )
        except Exception:
            raise FinalisationError("remote Parquet cannot be opened") from None
    raise FinalisationError("Hub adapter does not support bounded Parquet access")


def _validate_identity(revision: str, digest: str) -> None:
    if not isinstance(revision, str) or _COMMIT_SHA.fullmatch(revision) is None:
        raise FinalisationError("revision must be a complete 40-character commit SHA")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise FinalisationError("pipeline digest must be a complete SHA-256")


def _write_report(path: Path | str | None, report: dict[str, object]) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


__all__ = ["FinalisationError", "finalise_public_corpus", "finalize_public_corpus"]
