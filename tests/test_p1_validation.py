"""Focused tests for bounded P1 corpus validation."""

from __future__ import annotations

import collections.abc as c
import typing as t
from pathlib import Path

import pytest

from hviske.p1_validation import (
    AuditReservoir,
    MetadataLedger,
    PinnedHubClipRetriever,
    aggregate_rows,
    bounded_remote_aggregates,
    build_final_quality_report,
    build_quality_report,
    check_duplicate_and_overlaps,
    create_blinded_audit_manifest,
    deterministic_deciles,
    export_clip_for_review,
    normalised_wer,
    persist_audit_candidates,
    play_audio,
    review_one_clip,
    score_asr_anomalies,
    stratified_sample,
    summarise_manual_audit,
)


def test_accepted_audit_locator_is_resolved_after_restart(tmp_path: Path) -> None:
    """A pending local candidate survives a failed upload verification."""
    reservoir_path = tmp_path / "reservoir.json"
    local_path = tmp_path / "staging" / "shard.parquet"
    reservoir = AuditReservoir(
        reservoir_path, accepted_quota=1, rejected_quota=0, borderline_quota=0
    )
    reservoir.add(
        [
            {
                "segment_id": "accepted-1",
                "status": "accepted",
                "local_path": str(local_path),
                "local_row_locator": 3,
            }
        ]
    )

    recovered = AuditReservoir(
        reservoir_path, accepted_quota=1, rejected_quota=0, borderline_quota=0
    )
    assert recovered.rows[0]["local_path"] == str(local_path)
    assert (
        recovered.update_remote_locators(
            repository="org/private-p1",
            revision="a" * 40,
            local_paths=[local_path],
            remote_paths=["data/train/shard.parquet"],
            row_counts=[4],
        )
        == 1
    )
    manifest = recovered.finalise(tmp_path / "manifest.jsonl")
    assert manifest[0]["revision"] == "a" * 40
    assert manifest[0]["parquet_path"] == "data/train/shard.parquet"
    assert manifest[0]["row_locator"] == 3
    assert "local_path" not in manifest[0]


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
        "repository": "org/p1",
        "revision": "a" * 40,
        "parquet_path": "data/train/part.parquet",
        "row_locator": index,
    }


def test_blind_decision_is_persisted_without_status_or_audio(tmp_path: Path) -> None:
    """The reviewer sees no segmentation label and the ledger stores scalars only."""
    database = tmp_path / "audit.sqlite"
    candidates = create_blinded_audit_manifest(
        [{**_row(0), "parquet_path": "data/train/part.parquet", "row_locator": 0}],
        accepted_quota=1,
        rejected_quota=0,
    )
    assert persist_audit_candidates(database, candidates) == 1
    ledger = MetadataLedger(database)
    try:
        seen: list[dict[str, object]] = []

        class Retriever:
            def retrieve(self, entry: t.Mapping[str, object]) -> bytes:
                del entry
                return b"clip"

        def reviewer(_path: Path, entry: t.Mapping[str, object]) -> dict[str, object]:
            seen.append(dict(entry))
            return {
                "audit_id": entry["audit_id"],
                "decision": "accepted",
                "independent_asr_anomaly": False,
            }

        result = review_one_clip(
            candidates[0],
            Retriever(),
            reviewer,
            temporary_root=tmp_path,
            decision_store=ledger,
        )
        assert result["decision"] == "accepted"
        assert "status" not in seen[0]
        assert list(ledger.decision_rows())[0]["independent_asr_anomaly"] is False
    finally:
        ledger.close()
    report = build_quality_report([_row(0)], database=database)
    assert t.cast(dict[str, object], report["manual_audit"])["audited"] == 1


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
    required_axes = {
        "duration-",
        "show-",
        "speakers-",
        "music",
        "language-",
        "position-",
        "confidence-",
        "drift-",
    }
    for item in manifest:
        strata = t.cast(list[str], item["stratum"])
        assert len(strata) == 8
        prefixes = {
            "music" if value in {"music", "speech"} else value.split("-", 1)[0] + "-"
            for value in strata
        }
        assert prefixes == required_axes


@pytest.mark.parametrize("digest_key", ["_p1_metadata_sha256", "metadata_sha256"])
def test_blinded_manifest_preserves_producer_metadata_digest(digest_key: str) -> None:
    """The manifest retains the digest calculated from the complete output row."""
    producer_digest = "d" * 64
    row = _row(0)
    row[digest_key] = producer_digest

    manifest = create_blinded_audit_manifest(
        [row], accepted_quota=1, rejected_quota=0, seed="producer-digest"
    )

    assert manifest[0]["metadata_sha256"] == producer_digest
    assert all(not key.startswith("_p1_") for key in manifest[0])


@pytest.mark.parametrize("digest_key", ["_p1_metadata_sha256", "metadata_sha256"])
def test_blinded_manifest_rejects_malformed_producer_metadata_digest(
    digest_key: str,
) -> None:
    """A malformed producer digest must not be silently replaced."""
    row = _row(0)
    row[digest_key] = "not-a-sha256"

    with pytest.raises(ValueError, match="lowercase SHA-256 hex"):
        create_blinded_audit_manifest(
            [row], accepted_quota=1, rejected_quota=0, seed="producer-digest"
        )


def test_bounded_reservoir_does_not_retain_audio_or_grow() -> None:
    """Representative selection keeps only its configured candidate reservoir."""
    rows = (
        {
            **_row(index),
            "parquet_path": "data/train/part-00000.parquet",
            "row_locator": index,
            "audio": b"must not be retained",
        }
        for index in range(10_000)
    )
    selected = stratified_sample(rows, sample_size=17, seed="bounded")

    assert len(selected) == 17
    assert all("audio" not in row and "waveform" not in row for row in selected)


def test_corpus_audit_reservoir_is_bounded_and_recoverable(tmp_path: Path) -> None:
    """One durable reservoir retains every requested status without growing."""
    path = tmp_path / "reservoir.json"
    reservoir = AuditReservoir(
        path, accepted_quota=2, rejected_quota=2, borderline_quota=2, seed="test"
    )
    rows = [
        _row(index, status)
        for status in ("accepted", "rejected", "borderline")
        for index in range(10)
    ]
    reservoir.add(rows)
    assert len(reservoir.rows) == 6
    recovered = AuditReservoir(
        path, accepted_quota=2, rejected_quota=2, borderline_quota=2, seed="test"
    )
    manifest = recovered.finalise(tmp_path / "manifest.jsonl")
    assert len(manifest) == 6
    assert {item["source_file_id"] for item in manifest if "source_file_id" in item}
    assert not (tmp_path / ".reservoir.json.tmp").exists()


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


def test_export_retrieves_one_clip_and_leaves_no_temporary_audio(
    tmp_path: Path,
) -> None:
    """The non-playing export path does not create a decision or retain temp audio."""
    candidate = create_blinded_audit_manifest(
        [_row(0)], accepted_quota=1, rejected_quota=0
    )[0]
    destination = tmp_path / "controlled-review.flac"
    export_clip_for_review(
        candidate,
        type("Retriever", (), {"retrieve": lambda _self, entry: b"audio"})(),
        destination,
    )
    assert destination.read_bytes() == b"audio"
    assert [path for path in tmp_path.iterdir() if path != destination] == []


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


def test_manifest_keeps_exact_remote_locator_and_immutable_revision() -> None:
    """Audit records retain the exact Parquet locator needed for one-row retrieval."""
    row = {
        **_row(4),
        "repository": "org/source",
        "revision": "b" * 40,
        "parquet_path": "shards/train-004.parquet",
        "row_locator": 37,
        "source_file_id": "file-004",
        "source_start_ms": 1200,
        "source_end_ms": 2300,
    }
    candidate = create_blinded_audit_manifest(
        [row], accepted_quota=1, rejected_quota=0
    )[0]
    assert candidate["repository"] == "org/source"
    assert candidate["revision"] == "b" * 40
    assert candidate["remote_parquet_path"] == "shards/train-004.parquet"
    assert candidate["row_locator"] == 37
    assert candidate["source_file_id"] == "file-004"
    assert candidate["source_start_ms"] == 1200
    assert candidate["source_end_ms"] == 2300


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
        {
            "audit_id": "audit-1",
            "parquet_path": "data/train/part.parquet",
            "row_locator": 0,
            "revision": "a" * 40,
        },
        Retriever(),
        reviewer,
        temporary_root=tmp_path,
    )

    assert result["decision"] == "accepted"
    assert seen and not seen[0].exists()
    assert list(tmp_path.iterdir()) == []


def test_pinned_hub_retriever_reads_one_embedded_row() -> None:
    """A pinned fake Hub returns the addressed row, not a direct audio path."""
    revision = "a" * 40
    source = {
        **_row(1),
        "repository": "org/p1",
        "revision": revision,
        "parquet_path": "data/train/part-00000.parquet",
        "row_locator": 1,
        "audio": {"bytes": b"one clip"},
    }
    candidate = create_blinded_audit_manifest(
        [source], accepted_quota=1, rejected_quota=0
    )[0]

    class FakeHub:
        def __init__(self) -> None:
            self.calls = 0

        def load_dataset(
            self, repo_id: str, *, shard_path: str, revision: str, streaming: bool
        ) -> list[dict[str, object]]:
            assert (repo_id, shard_path, revision, streaming) == (
                "org/p1",
                "data/train/part-00000.parquet",
                revision,
                True,
            )
            return [
                {**source, "row_locator": 0, "audio": {"bytes": b"other"}},
                {**source, "row_locator": 1},
            ]

    fake = FakeHub()
    retriever = PinnedHubClipRetriever(fake, repository="org/p1", revision=revision)
    assert retriever.retrieve(candidate) == b"one clip"


def test_play_audio_uses_argv_without_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Configured players are split into argv and never passed through a shell."""
    calls: list[dict[str, object]] = []

    def run(command: list[str], *, check: bool) -> None:
        calls.append({"command": command, "check": check})

    monkeypatch.setattr("hviske.p1_validation.subprocess.run", run)
    path = tmp_path / "clip;touch unsafe"
    path.write_bytes(b"audio")
    play_audio(path, player='player --flag "value with spaces"')
    assert calls == [
        {"command": ["player", "--flag", "value with spaces", str(path)], "check": True}
    ]


def test_rejected_audit_candidates_use_immutable_source_locators() -> None:
    """Rejected and borderline reviews point back to source intervals."""
    rows = [
        {
            **_row(1, "rejected"),
            "source_repository": "org/source",
            "source_revision": "b" * 40,
            "source_start_ms": 100,
            "source_end_ms": 900,
        },
        {
            **_row(2, "borderline"),
            "source_repository": "org/source",
            "source_revision": "b" * 40,
            "source_start_ms": 200,
            "source_end_ms": 800,
        },
    ]
    manifest = create_blinded_audit_manifest(
        rows, accepted_quota=0, rejected_quota=1, borderline_quota=1
    )

    assert all("parquet_path" not in item for item in manifest)
    assert {item["source_file_id"] for item in manifest} == {
        "programme-0",
        "programme-1",
    }


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


def test_retrieval_failure_cannot_persist_decision(tmp_path: Path) -> None:
    """A failed fetch leaves no decision in the metadata-only ledger.

    Raises:
        AssertionError:
            If a broken retriever unexpectedly returns.
    """
    database = tmp_path / "audit.sqlite"
    candidate = create_blinded_audit_manifest(
        [_row(0)], accepted_quota=1, rejected_quota=0
    )[0]
    persist_audit_candidates(database, [candidate])
    ledger = MetadataLedger(database)
    try:

        class BrokenRetriever:
            def retrieve(self, entry: t.Mapping[str, object]) -> bytes:
                del entry
                raise FileNotFoundError("gone")

        try:
            review_one_clip(
                candidate,
                BrokenRetriever(),
                lambda _path, _entry: {"decision": "accepted"},
                decision_store=ledger,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("retrieval failure unexpectedly returned")
        assert list(ledger.decision_rows()) == []
    finally:
        ledger.close()


def test_review_plays_before_decision_and_persists_only_after_retrieval(
    tmp_path: Path,
) -> None:
    """Playback precedes the reviewer and a successful retrieval is recorded."""
    database = tmp_path / "audit.sqlite"
    candidate = create_blinded_audit_manifest(
        [_row(0)], accepted_quota=1, rejected_quota=0
    )[0]
    persist_audit_candidates(database, [candidate])
    events: list[str] = []
    ledger = MetadataLedger(database)
    try:
        result = review_one_clip(
            candidate,
            type("Retriever", (), {"retrieve": lambda _self, entry: b"audio"})(),
            lambda _path, _entry: events.append("decision") or {"decision": "accepted"},
            decision_store=ledger,
            player=lambda _path: events.append("played"),
        )
    finally:
        ledger.close()
    assert result["decision"] == "accepted"
    assert events == ["played", "decision"]
    check = MetadataLedger(database)
    try:
        assert list(check.decision_rows())[0]["decision"] == "accepted"
    finally:
        check.close()
