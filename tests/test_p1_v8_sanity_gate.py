"""Tests for the privacy-safe post-pilot v8 sanity gate."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import typing as t
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from hviske.p1_contracts import OutputRow
from hviske.p1_segments import _rows_table
from hviske.p1_v8_sanity_gate import run_v8_sanity_gate
from hviske.p1_validation import (
    AuditReservoir,
    PinnedHubClipRetriever,
    _metadata_digest,
)

_REVISION = "a" * 40


def _flac() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.full(16_000, 0.1, dtype=np.float32), 16_000, format="FLAC")
    return stream.getvalue()


def test_active_p1_imports_and_sanity_help_are_model_free() -> None:
    """P1 imports and CLI help do not load model runtimes."""
    modules = "import hviske.p1_pipeline, hviske.p1_v8_sanity_gate"
    check = (
        f"{modules}; import sys; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    imported = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True, check=False
    )
    assert imported.returncode == 0
    assert imported.stdout.strip() == "False False"

    help_check = """
import runpy
import sys
sys.argv = ["run_p1_v8_sanity_gate.py", "--help"]
try:
    runpy.run_path("src/scripts/run_p1_v8_sanity_gate.py", run_name="__main__")
except SystemExit:
    pass
print("torch" in sys.modules, "transformers" in sys.modules)
"""
    helped = subprocess.run(
        [sys.executable, "-c", help_check], capture_output=True, text=True, check=False
    )
    assert helped.returncode == 0
    assert "usage:" in helped.stdout
    assert helped.stdout.rstrip().endswith("False False")


_AUDIO = _flac()
_AUDIO_SHA256 = hashlib.sha256(_AUDIO).hexdigest()


def test_audit_manifest_retrieves_published_parquet_rows(tmp_path: Path) -> None:
    """The final audit manifest passes validation against its published shard."""
    repository = "syvai/p1-segments"
    parquet_path = tmp_path / "part.parquet"
    revision = _REVISION
    rows = []
    for index in range(12):
        raw_row = _row(index)
        raw_row["audio"] = _AUDIO
        raw_row["alignment_word_map"] = ("hej", "verden")
        raw_row["speaker_ids"] = ("speaker",)
        rows.append(OutputRow.model_validate(raw_row))
    parquet_buffer = io.BytesIO()
    pq.write_table(_rows_table(rows), parquet_buffer)
    parquet_bytes = parquet_buffer.getvalue()
    parquet_sha256 = hashlib.sha256(parquet_bytes).hexdigest()

    reservoir = AuditReservoir(
        tmp_path / "reservoir.json",
        accepted_quota=12,
        rejected_quota=0,
        borderline_quota=0,
        seed="e2e",
    )
    candidates = []
    for index, row in enumerate(rows):
        published = row.model_dump(mode="python")
        digest = _metadata_digest(published)
        published.update(
            {
                "_p1_metadata_sha256": digest,
                "metadata_sha256": digest,
                "repository": repository,
                "revision": revision,
                "parquet_path": "data/part.parquet",
                "row_locator": index,
                "parquet_sha256": parquet_sha256,
            }
        )
        candidates.append(published)
    reservoir.add(candidates)
    manifest = reservoir.finalise(tmp_path / "manifest.jsonl")
    parquet_path.write_bytes(parquet_bytes)

    class _PublishedHub:
        def get_paths_info(
            self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
        ) -> list[dict[str, object]]:
            assert (repo_id, paths, repo_type, revision) == (
                repository,
                ["data/part.parquet"],
                "dataset",
                _REVISION,
            )
            return [{"path": paths[0], "sha256": parquet_sha256}]

        def load_dataset(
            self, repo_id: str, *, shard_path: str, revision: str, streaming: bool
        ) -> list[dict[str, object]]:
            assert (repo_id, shard_path, revision, streaming) == (
                repository,
                "data/part.parquet",
                _REVISION,
                True,
            )
            return pq.read_table(parquet_path).to_pylist()

        def repo_info(
            self, repo_id: str, *, repo_type: str, revision: str
        ) -> dict[str, object]:
            assert (repo_id, repo_type, revision) == (repository, "dataset", _REVISION)
            return {"private": True, "sha": _REVISION}

    retriever = PinnedHubClipRetriever(
        _PublishedHub(),
        repository=repository,
        revision=revision,
        expected_pipeline_version="p1-segmentation-8",
        expected_pipeline_config_sha256="c" * 64,
    )
    report = run_v8_sanity_gate(
        manifest,
        retriever=retriever,
        pilot_head=revision,
        report_path=tmp_path / "sanity.json",
    )

    assert report["pass"] is True
    assert report["counts"] == {
        "accepted": 12,
        "selected": 12,
        "retrieved": 12,
        "structural_failures": 0,
        "pipeline_digests": 1,
    }
    assert all("_p1_metadata_sha256" not in candidate for candidate in manifest)


def _row(index: int) -> dict[str, object]:
    return {
        "audio": {"bytes": _AUDIO},
        "audio_sha256": _AUDIO_SHA256,
        "text": "hej verden",
        "alignment_text": "hej verden",
        "alignment_word_map": ["hej", "verden"],
        "language": "da",
        "segment_id": hashlib.sha256(f"segment-{index}".encode()).hexdigest(),
        "source_file_id": f"programme-{index % 3}",
        "source_start_ms": 0,
        "source_end_ms": 1_000,
        "source_duration_ms": 2_000,
        "duration_ms": 1_000,
        "speaker_ids": ["speaker"],
        "proposal_start_ms": 0,
        "proposal_end_ms": 1_000,
        "alignment_score": None,
        "alignment_score_type": "not_applicable:source_timestamps",
        "start_drift_ms": None,
        "end_drift_ms": None,
        "vad_speech_ratio": None,
        "alignment_backend": "timestamp-native",
        "alignment_method": "timestamp-native:p1-transcripts.words",
        "pipeline_version": "p1-segmentation-8",
        "pipeline_config_sha256": "c" * 64,
    }


def test_gate_fails_without_a_full_dozen(tmp_path: Path) -> None:
    """A short accepted pool produces a failed aggregate report."""
    report_path = tmp_path / "gate.json"
    report = run_v8_sanity_gate(
        _candidates(11), retriever=_Retriever(_candidates(11)), report_path=report_path
    )

    assert report["pass"] is False
    assert report["counts"] == {"accepted": 11, "selected": 11}
    assert json.loads(report_path.read_text(encoding="utf-8"))["pass"] is False


class _Retriever:
    def __init__(self, candidates: list[dict[str, object]]) -> None:
        self.rows = {
            candidate["segment_id"]: _row(index)
            for index, candidate in enumerate(candidates)
        }
        self.seen: list[str] = []

    def retrieve_row(self, candidate: dict[str, object]) -> dict[str, object]:
        self.seen.append(str(candidate["audit_id"]))
        return self.rows[candidate["segment_id"]]

    def verify_repository(self) -> None:
        """The fake retriever models private immutable verification."""


def _candidates(count: int) -> list[dict[str, object]]:
    result = []
    for index in range(count):
        row = _row(index)
        result.append(
            {
                "audit_id": f"anonymous-{index}",
                "segment_id": row["segment_id"],
                "pipeline_version": row["pipeline_version"],
                "pipeline_config_sha256": row["pipeline_config_sha256"],
                "status": "accepted",
                "source_file_id": row["source_file_id"],
                "stratum": ["duration-target", f"programme-{index % 3}"],
                "repository": "syvai/p1-segments",
                "revision": _REVISION,
                "parquet_path": "data/train/part.parquet",
                "row_locator": index,
                "metadata_sha256": _metadata_digest(row),
                "audio_sha256": _AUDIO_SHA256,
                "parquet_sha256": "d" * 64,
            }
        )
    return result


def test_gate_failure_report_does_not_claim_unexecuted_checks_pass(
    tmp_path: Path,
) -> None:
    """A row-level schema failure leaves all unproven checks false."""
    candidates = _candidates(12)
    retriever = _Retriever(candidates)
    retriever.rows[candidates[0]["segment_id"]]["revision"] = _REVISION
    report = run_v8_sanity_gate(
        candidates, retriever=retriever, report_path=tmp_path / "failed.json"
    )

    assert report["pass"] is False
    checks = t.cast(dict[str, bool], report["checks"])
    assert checks["private_immutable_revision"] is True
    assert checks["exact_schema"] is False
    assert checks["flac_pcm16_16khz_mono"] is False
    assert checks["trainable_text"] is False


def test_gate_is_deterministic_and_report_contains_aggregate_only(
    tmp_path: Path,
) -> None:
    """The selected set is stable and the report contains no row-level evidence."""
    candidates = _candidates(18)
    first_retriever = _Retriever(candidates)
    first = run_v8_sanity_gate(
        candidates,
        retriever=first_retriever,
        seed="test-seed",
        report_path=tmp_path / "first.json",
    )
    second_retriever = _Retriever(list(reversed(candidates)))
    second = run_v8_sanity_gate(
        list(reversed(candidates)), retriever=second_retriever, seed="test-seed"
    )

    assert first["pass"] is True
    assert first["sample_set_digest"] == second["sample_set_digest"]
    assert first_retriever.seen == second_retriever.seen
    report_text = (tmp_path / "first.json").read_text(encoding="utf-8")
    assert "anonymous-" not in report_text
    assert "programme-" not in report_text
    assert "part.parquet" not in report_text
    assert "hej verden" not in report_text
    assert "source_file_id" not in report_text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alignment_word_map", 1),
        ("speaker_ids", ["speaker", 1]),
        ("alignment_score_type", None),
        ("vad_speech_ratio", "not-a-ratio"),
    ],
)
def test_gate_rejects_malformed_exact_schema_values(
    tmp_path: Path, field: str, value: object
) -> None:
    """Malformed list, nullable, and scalar values cannot pass exact_schema."""
    candidates = _candidates(12)
    retriever = _Retriever(candidates)
    segment_id = t.cast(str, candidates[0]["segment_id"])
    malformed = dict(retriever.rows[segment_id])
    malformed[field] = value
    retriever.rows[segment_id] = malformed
    candidates[0]["metadata_sha256"] = _metadata_digest(malformed)

    report = run_v8_sanity_gate(
        candidates, retriever=retriever, report_path=tmp_path / "failed.json"
    )

    assert report["pass"] is False
    checks = t.cast(dict[str, bool], report["checks"])
    assert checks["exact_schema"] is False
