"""Focused tests for the metadata-only P1 release finaliser."""

from __future__ import annotations

import collections.abc as c
import hashlib
import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import p1_dataset.finalisation as finalisation
from p1_dataset.finalisation import FinalisationError, finalise_p1_corpus
from p1_dataset.publish import UploadOperation, build_dataset_card
from p1_dataset.validation import PinnedHubClipRetriever

_REVISION = "a" * 40
_DIGEST = "b" * 64
_SHARD = finalisation._Shard("data/train/part-00000.parquet", 10, 2, "c" * 64)


def test_batch_history_must_contain_each_recorded_commit() -> None:
    """A complete Hub history proves ancestor status without equal SHAs."""
    batch = finalisation._Batch("batch-001", "c" * 40, 1, 2, {}, (_SHARD,))

    class HistoryHub:
        def list_repo_commits(
            self, repo_id: str, *, repo_type: str, revision: str
        ) -> tuple[object, ...]:
            del repo_id, repo_type, revision
            return (
                SimpleNamespace(commit_id=_REVISION),
                SimpleNamespace(commit_id=batch.commit_id),
            )

    finalisation._check_batch_ancestry(
        HistoryHub(), "syvai/p1-segments", _REVISION, (batch,)
    )

    class UnrelatedHistoryHub(HistoryHub):
        def list_repo_commits(
            self, repo_id: str, *, repo_type: str, revision: str
        ) -> tuple[object, ...]:
            del repo_id, repo_type, revision
            return (SimpleNamespace(commit_id=_REVISION),)

    with pytest.raises(FinalisationError, match="ancestor"):
        finalisation._check_batch_ancestry(
            UnrelatedHistoryHub(), "syvai/p1-segments", _REVISION, (batch,)
        )


def test_card_update_uses_cas_and_preserves_licence_and_inventory() -> None:
    """Card mode commits only README bytes against the pinned parent."""
    old_head = _REVISION
    new_head = "d" * 40
    readme = build_dataset_card(
        source_provenance="Pinned source programmes",
        permitted_use="Internal ASR research",
        private_access_terms="Access is limited to the project organisation",
        alignment_method=f"pipeline_config_sha256: {_DIGEST}",
        field_schema="audio, text and deterministic metadata",
        known_limitations="Danish speech only",
        rejection_policy="Reject undecodable or poorly aligned material",
        source_revisions="{}",
    ).encode("utf-8")
    licence = b"unchanged licence"

    class CardHub(_Hub):
        def __init__(self) -> None:
            super().__init__(("README.md", "LICENSE", _SHARD.path))
            self.head = old_head
            self.files: dict[str, bytes] = {
                "README.md": readme,
                "LICENSE": licence,
                _SHARD.path: b"parquet",
            }
            self.operations: tuple[str, ...] = ()

        def create_commit(
            self,
            repo_id: str,
            operations: c.Iterable[UploadOperation],
            **kwargs: object,
        ) -> object:
            del repo_id
            assert kwargs["parent_commit"] == old_head
            self.operations = tuple(item.path_in_repo for item in operations)
            self.files["README.md"] = next(
                item.path.read_bytes() for item in operations
            )
            self.head = new_head
            return SimpleNamespace(commit_id=new_head)

        def get_paths_info(
            self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
        ) -> tuple[object, ...]:
            del repo_id, repo_type, revision
            return tuple({"path": path, "sha256": "e" * 64} for path in paths)

        def list_repo_files(
            self, repo_id: str, *, repo_type: str, revision: str
        ) -> tuple[str, ...]:
            del repo_id, repo_type, revision
            return tuple(self.files)

        def repo_info(self, repo_id: str, *, repo_type: str, revision: str) -> object:
            del repo_id, repo_type, revision
            return SimpleNamespace(private=True, sha=self.head)

        def stream_file(
            self, repo_id: str, path: str, *, repo_type: str, revision: str
        ) -> tuple[bytes, ...]:
            del repo_id, repo_type, revision
            return (self.files[path],)

    hub = CardHub()
    head = finalisation._cas_card_update(
        hub=hub,
        repository="syvai/p1-segments",
        revision=old_head,
        report={"counts": {"rows": 2, "shards": 1}, "audio": {"hours": 1.0}},
        remote_files=("README.md", "LICENSE", _SHARD.path),
    )
    assert head == new_head
    assert hub.operations == ("README.md",)
    assert hub.files["LICENSE"] == licence


def test_finalisation_dry_run_is_metadata_only_and_writes_sanitised_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful dry run uses no dataset loader or Hub mutation."""
    hub = _Hub(("README.md", "LICENSE", _SHARD.path))
    monkeypatch.setattr(finalisation, "_read_run_root", lambda *args: _summary())
    monkeypatch.setattr(finalisation, "_check_remote_metadata", lambda *args: None)
    monkeypatch.setattr(finalisation, "_check_manifests", lambda *args: None)
    monkeypatch.setattr(finalisation, "_scan_parquet", lambda *args: _stats())

    report_path = tmp_path / "report.json"
    report = finalise_p1_corpus(
        hub=hub,
        run_root=tmp_path,
        revision=_REVISION,
        expected_pipeline_config_sha256=_DIGEST,
        report_path=report_path,
    )

    assert report["pass"] is True
    assert report_path.is_file()
    saved = report_path.read_text(encoding="utf-8")
    assert _SHARD.path not in saved
    assert "e" * 64 not in saved
    assert hub.commits == 0


class _Hub:
    """Small Hub fake used to prove metadata-only orchestration."""

    def __init__(self, files: tuple[str, ...]) -> None:
        self.files = files
        self.commits = 0

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> tuple[str, ...]:
        del repo_id, repo_type, revision
        return self.files

    def repo_info(self, repo_id: str, *, repo_type: str, revision: str) -> object:
        del repo_id, repo_type
        return SimpleNamespace(private=True, sha=revision)


def _stats() -> dict[str, object]:
    duration = finalisation._Stats()
    duration.add(1_000)
    words = finalisation._Stats()
    words.add(2)
    return {
        "rows": 2,
        "programmes": 1,
        "duration_ms": 2_000,
        "duration": duration,
        "words": words,
        "speakers": {"1": 2},
    }


def _summary() -> dict[str, object]:
    return {
        "shards": (_SHARD,),
        "rows": 2,
        "programme_states": {"purged": 1},
        "batches": {"purged": 1},
    }


def test_finalisation_fails_closed_on_remote_inventory_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path not represented by the ledgers prevents the report."""
    hub = _Hub(("README.md", "LICENSE", "data/train/other.parquet"))
    monkeypatch.setattr(finalisation, "_read_run_root", lambda *args: _summary())
    with pytest.raises(FinalisationError, match="inventory"):
        finalise_p1_corpus(
            hub=hub,
            run_root=tmp_path,
            revision=_REVISION,
            expected_pipeline_config_sha256=_DIGEST,
        )


def test_pinned_retriever_requires_private_visibility() -> None:
    """The default private policy accepts a private repository."""
    hub = _Hub(("README.md",))
    retriever = PinnedHubClipRetriever(
        hub, repository="syvai/p1-segments", revision=_REVISION
    )
    retriever.verify_repository()


def test_remote_manifest_is_checked_against_batch_and_shards() -> None:
    """Finalisation validates remote manifests after local publication purge."""
    shard = finalisation._Shard(
        "data/train/part-00000.parquet", 3, 2, hashlib.sha256(b"abc").hexdigest()
    )
    batch = finalisation._Batch("batch-001", "c" * 40, 1, 2, {}, (shard,))
    payload = finalisation._manifest_bytes(batch)

    class ManifestHub:
        def stream_file(
            self, repo_id: str, path: str, *, repo_type: str, revision: str
        ) -> tuple[bytes, ...]:
            del repo_id, repo_type, revision
            assert path == "manifests/batch-001.json"
            return (payload[:2], payload[2:])

    finalisation._check_remote_manifests(
        ManifestHub(),
        "syvai/p1-segments",
        _REVISION,
        (batch,),
        remote_files=("README.md", "LICENSE", shard.path, "manifests/batch-001.json"),
    )
    with pytest.raises(FinalisationError, match="manifest inventory"):
        finalisation._check_remote_manifests(
            ManifestHub(),
            "syvai/p1-segments",
            _REVISION,
            (batch,),
            remote_files=(
                "README.md",
                "LICENSE",
                shard.path,
                "manifests/batch-001.json",
                "manifests/extra.json",
            ),
        )


def test_repo_file_git_metadata_uses_bounded_content_hashing() -> None:
    """Git blob IDs trigger streaming rather than a false SHA-256 match."""
    content = b"0123456789"
    shard = finalisation._Shard(
        "data/train/part-00000.parquet",
        len(content),
        2,
        hashlib.sha256(content).hexdigest(),
    )

    class GitHub:
        def get_paths_info(
            self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
        ) -> tuple[object, ...]:
            del repo_id, repo_type, revision
            return tuple(
                SimpleNamespace(path=path, size=len(content), blob_id="a" * 40)
                for path in paths
            )

        def stream_file(
            self, repo_id: str, path: str, *, repo_type: str, revision: str
        ) -> tuple[bytes, ...]:
            del repo_id, path, repo_type, revision
            return (content[:3], content[3:])

    finalisation._check_remote_metadata(
        GitHub(), "syvai/p1-segments", _REVISION, (shard,)
    )


def test_scan_batches_metadata_and_checks_overlaps_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk scanning uses one transaction and one corpus-wide overlap query."""
    rows = [_metadata_row(index=index, start=index * 1_000) for index in range(3_000)]
    traces: list[str] = []
    real_connect = finalisation.sqlite3.connect

    def traced_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(traces.append)
        return connection

    class Batch:
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = values
            self.num_rows = len(values)

        def to_pylist(self) -> list[dict[str, object]]:
            return self.values

    class Parquet:
        schema_arrow = finalisation._EXPECTED_ARROW_SCHEMA
        metadata = SimpleNamespace(num_rows=len(rows))

        def iter_batches(
            self, *, columns: list[str], batch_size: int
        ) -> c.Iterator[Batch]:
            assert columns == list(finalisation._METADATA_COLUMNS)
            assert batch_size == finalisation._ARROW_BATCH_SIZE
            for offset in range(0, len(rows), batch_size):
                yield Batch(rows[offset : offset + batch_size])

    class ScanHub:
        def open_file(
            self, repo_id: str, path: str, *, repo_type: str, revision: str
        ) -> io.BytesIO:
            del repo_id, path, repo_type, revision
            return io.BytesIO()

    monkeypatch.setattr(finalisation.sqlite3, "connect", traced_connect)
    monkeypatch.setattr(finalisation.pq, "ParquetFile", lambda handle: Parquet())

    stats = finalisation._scan_parquet(
        ScanHub(),
        "syvai/p1-segments",
        _REVISION,
        (finalisation._Shard("part.parquet", 3_000, 3_000, "c" * 64),),
        _DIGEST,
    )

    assert stats["rows"] == 3_000
    assert stats["programmes"] == 1
    assert stats["duration_ms"] == 3_000_000
    assert stats["speakers"] == {"1": 3_000}
    assert sum(trace == "COMMIT" for trace in traces) == 1
    assert sum(trace.startswith("WITH ordered") for trace in traces) == 1


def _metadata_row(
    *, index: int, start: int, end: int | None = None
) -> dict[str, object]:
    """Build one valid metadata-only row for scanner tests.

    Returns:
        A metadata-only row satisfying the v8 scanner contract.
    """
    interval_end = end if end is not None else start + 1_000
    return {
        "audio_sha256": "a" * 64,
        "text": "et eksempel",
        "alignment_text": "et eksempel",
        "alignment_word_map": ["et", " eksempel"],
        "language": "da",
        "segment_id": f"{index:064x}",
        "source_file_id": "source",
        "source_start_ms": start,
        "source_end_ms": interval_end,
        "source_duration_ms": 3_000_000,
        "duration_ms": interval_end - start,
        "speaker_ids": ["speaker"],
        "proposal_start_ms": start,
        "proposal_end_ms": interval_end,
        "alignment_score": None,
        "alignment_score_type": "not_applicable:source_timestamps",
        "start_drift_ms": None,
        "end_drift_ms": None,
        "vad_speech_ratio": None,
        "alignment_backend": "timestamp-native",
        "alignment_method": "timestamp-native:p1-transcripts.words",
        "pipeline_version": "p1-segmentation-8",
        "pipeline_config_sha256": _DIGEST,
    }


def test_scan_rejects_nested_source_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The interval query catches an earlier long interval around later rows."""
    rows = [
        _metadata_row(index=0, start=0, end=5_000),
        _metadata_row(index=1, start=1_000, end=2_000),
    ]

    class Batch:
        num_rows = len(rows)

        def to_pylist(self) -> list[dict[str, object]]:
            return rows

    class Parquet:
        schema_arrow = finalisation._EXPECTED_ARROW_SCHEMA
        metadata = SimpleNamespace(num_rows=len(rows))

        def iter_batches(
            self, *, columns: list[str], batch_size: int
        ) -> c.Iterator[Batch]:
            del columns, batch_size
            yield Batch()

    class ScanHub:
        def open_file(
            self, repo_id: str, path: str, *, repo_type: str, revision: str
        ) -> io.BytesIO:
            del repo_id, path, repo_type, revision
            return io.BytesIO()

    monkeypatch.setattr(finalisation.pq, "ParquetFile", lambda handle: Parquet())
    with pytest.raises(FinalisationError, match="overlapping"):
        finalisation._scan_parquet(
            ScanHub(),
            "syvai/p1-segments",
            _REVISION,
            (finalisation._Shard("part.parquet", 3_000, 2, "c" * 64),),
            _DIGEST,
        )
