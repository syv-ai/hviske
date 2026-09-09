"""Focused tests for bounded P1 corpus validation."""

from __future__ import annotations

import collections.abc as c
import typing as t
from pathlib import Path

from hviske.p1_validation import (
    aggregate_rows,
    bounded_remote_aggregates,
    build_final_quality_report,
    check_duplicate_and_overlaps,
    create_blinded_audit_manifest,
    deterministic_deciles,
    normalised_wer,
    review_one_clip,
    score_asr_anomalies,
    stratified_sample,
    summarise_manual_audit,
)


def test_aggregate_totals_and_distributions() -> None:
    """Aggregates preserve status totals and numeric distribution counts."""
    report = aggregate_rows([_row(0), _row(1, "rejected"), _row(2, "borderline")])
    totals = t.cast(dict[str, object], report["totals"])
    distributions = t.cast(dict[str, object], report["distributions"])

    assert t.cast(int, totals["segments"]) == 3
    assert t.cast(int, totals["accepted_segments"]) == 1
    duration = t.cast(dict[str, object], distributions["duration_ms"])
    assert t.cast(int, duration["count"]) == 3


def _row(index: int, status: str = "accepted") -> dict[str, object]:
    return {
        "segment_id": f"segment-{index}",
        "source_file_id": f"programme-{index % 2}",
        "source_start_ms": index * 1_000,
        "source_end_ms": index * 1_000 + 1_000,
        "duration_ms": 1_000 + index * 100,
        "status": status,
        "show": "news" if index % 2 else "music",
        "speaker_ids": (f"speaker-{index % 3}",),
        "music": index % 2 == 0,
        "language_probability": index / 10,
        "programme_position": index / 10,
        "alignment_score": index / 10,
        "drift_ms": index * 10,
        "text": "et test",
        "remote_path": f"clips/{index}.flac",
    }


def test_blinded_manifest_has_all_quotas_without_labels() -> None:
    """The audit manifest samples each quota without exposing its class."""
    records = [_row(0), _row(1, "rejected"), _row(2, "borderline")]
    manifest = create_blinded_audit_manifest(
        records, accepted_quota=1, rejected_quota=1, borderline_quota=1, seed=4
    )

    assert len(manifest) == 3
    assert all("status" not in item and "accepted" not in item for item in manifest)
    assert {item["segment_id"] for item in manifest} == {
        "segment-0",
        "segment-1",
        "segment-2",
    }


def test_deciles_and_strata_are_deterministic() -> None:
    """Sampling remains stable when the remote stream order changes."""
    values = [0.5, 0.1, 0.9, 0.2]
    assert deterministic_deciles(values) == deterministic_deciles(values)
    rows = [_row(index) for index in range(10)]
    first = stratified_sample(rows, sample_size=6, seed="pilot")
    second = stratified_sample(reversed(rows), sample_size=6, seed="pilot")

    assert [row["segment_id"] for row in first] == [row["segment_id"] for row in second]
    assert all(
        len(t.cast(c.Sized, row["stratum"])) if "stratum" in row else True
        for row in first
    )


def test_duplicate_and_overlap_checks_are_metadata_only(tmp_path: Path) -> None:
    """SQLite checks identify duplicate IDs and source interval collisions."""
    first = _row(0)
    duplicate = {**first, "source_start_ms": 2_000, "source_end_ms": 3_000}
    overlap = {**_row(2), "source_start_ms": 500, "source_end_ms": 1_500}
    checks = check_duplicate_and_overlaps(
        [first, duplicate, overlap], tmp_path / "ledger.sqlite"
    )

    assert checks["duplicate_segment_ids"] == ["segment-0"]
    overlaps = t.cast(c.Sized, checks["overlapping_source_intervals"])
    assert len(overlaps) == 1
    assert checks["passes"] is False


def test_final_report_and_manual_audit_summary(tmp_path: Path) -> None:
    """The final report combines structural and independent audit evidence."""
    report = build_final_quality_report(
        [_row(0)],
        database=tmp_path / "final.sqlite",
        audit_records=[{"decision": "accepted", "text_mismatch": False}],
    )

    assert report["report_type"] == "p1-final-quality"
    structural = t.cast(dict[str, object], report["structural"])
    manual_audit = t.cast(dict[str, object], report["manual_audit"])
    assert structural["passes"] is True
    assert manual_audit["clean_rate"] == 1.0
    assert summarise_manual_audit([])["audited"] == 0


def test_independent_asr_metrics_and_normalisation() -> None:
    """An injectable ASR produces normalised WER anomaly metrics."""

    class FakeASR:
        def transcribe(self, audio: bytes) -> str:
            return "et test" if audio == b"good" else "helt forkert"

    rows = [{**_row(0), "text": "Et, test!"}, {**_row(1), "text": "Et test"}]
    result = score_asr_anomalies(
        rows,
        FakeASR(),
        lambda row: b"good" if row["segment_id"] == "segment-0" else b"bad",
    )

    assert normalised_wer("Et, test!", "et test") == 0
    assert result["segments"] == 2
    assert result["anomalies"] == 1


def test_one_item_review_deletes_audio(tmp_path: Path) -> None:
    """The one-item review deletes its temporary clip on successful review."""
    seen: list[Path] = []

    class Retriever:
        def retrieve(self, entry: t.Mapping[str, object]) -> bytes:
            return b"temporary audio"

    def reviewer(path: Path, _entry: t.Mapping[str, object]) -> dict[str, object]:
        seen.append(path)
        assert path.read_bytes() == b"temporary audio"
        return {"decision": "accepted"}

    result = review_one_clip(
        {"audit_id": "audit-1"}, Retriever(), reviewer, temporary_root=tmp_path
    )

    assert result["decision"] == "accepted"
    assert seen and not seen[0].exists()
    assert list(tmp_path.iterdir()) == []


def test_remote_aggregate_is_bounded() -> None:
    """A bounded remote aggregate must not pull one row beyond its limit."""
    consumed = 0

    def stream() -> c.Iterator[dict[str, object]]:
        nonlocal consumed
        for index in range(20):
            consumed += 1
            yield _row(index)

    report = bounded_remote_aggregates(stream(), max_rows=3)

    assert consumed == 3
    totals = t.cast(dict[str, object], report["totals"])
    assert totals["segments"] == 3
