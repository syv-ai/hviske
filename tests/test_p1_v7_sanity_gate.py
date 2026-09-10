"""Tests for the privacy-safe post-pilot v7 sanity gate."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from hviske.p1_v7_sanity_gate import run_v7_sanity_gate
from hviske.p1_validation import _metadata_digest

_REVISION = "a" * 40


def _flac() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.full(16_000, 0.1, dtype=np.float32), 16_000, format="FLAC")
    return stream.getvalue()


_AUDIO = _flac()
_AUDIO_SHA256 = hashlib.sha256(_AUDIO).hexdigest()


def test_gate_fails_without_a_full_dozen(tmp_path: Path) -> None:
    """A short accepted pool produces a failed aggregate report."""
    report_path = tmp_path / "gate.json"
    report = run_v7_sanity_gate(
        _candidates(11),
        retriever=_Retriever(_candidates(11)),
        asr=_ASR(),
        report_path=report_path,
    )

    assert report["pass"] is False
    assert report["counts"] == {"accepted": 11, "selected": 11}
    assert json.loads(report_path.read_text(encoding="utf-8"))["pass"] is False


class _ASR:
    def transcribe(self, audio: bytes) -> str:
        del audio
        return "hej verden"


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
        "pipeline_version": "p1-segmentation-7",
        "pipeline_config_sha256": "c" * 64,
    }


def _candidates(count: int) -> list[dict[str, object]]:
    result = []
    for index in range(count):
        row = _row(index)
        result.append(
            {
                "audit_id": f"anonymous-{index}",
                "segment_id": row["segment_id"],
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


def test_gate_is_deterministic_and_report_contains_aggregate_only(
    tmp_path: Path,
) -> None:
    """The selected set is stable and the report contains no row-level evidence."""
    candidates = _candidates(18)
    first_retriever = _Retriever(candidates)
    first = run_v7_sanity_gate(
        candidates,
        retriever=first_retriever,
        asr=_ASR(),
        seed="test-seed",
        report_path=tmp_path / "first.json",
    )
    second_retriever = _Retriever(list(reversed(candidates)))
    second = run_v7_sanity_gate(
        list(reversed(candidates)),
        retriever=second_retriever,
        asr=_ASR(),
        seed="test-seed",
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
