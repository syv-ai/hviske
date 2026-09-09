"""Focused offline tests for Phase 1B segmentation and sharding."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from hviske.p1_contracts import NormalisationContract, SegmentationContract, SourceWord
from hviske.p1_segments import (
    AlignmentResult,
    VADSignal,
    align_ctc_emissions,
    correct_drift_once,
    encode_flac,
    form_candidate_segments,
    normalise_alignment_text,
    prepare_source_audio,
    stream_sha256,
    validate_output_shard,
    validate_raw_timestamps,
    write_shards,
)

CONFIG_DIGEST = hashlib.sha256(b"p1-test-config").hexdigest()


def test_candidates_partition_words_and_respect_speakers() -> None:
    """Candidates neither duplicate/drop words nor cross a speaker change."""
    source = words(
        ("en.", 0, 1_000, "a"),
        ("to", 1_000, 2_000, "a"),
        ("tre", 2_000, 3_000, "a"),
        ("fire", 3_000, 4_000, "b"),
        ("fem", 4_000, 5_000, "b"),
    )
    proposals = form_candidate_segments(
        words=source, source_file_id="source", contract=segmentation_contract()
    )
    indexes = [
        index
        for proposal in proposals
        for index in range(proposal.word_start_index, proposal.word_end_index)
    ]
    assert indexes == list(range(len(source)))
    assert all(len(proposal.speaker_ids) <= 1 for proposal in proposals)


def segmentation_contract(maximum_duration_ms: int = 10_000) -> SegmentationContract:
    """Return small deterministic test thresholds."""
    return SegmentationContract(
        minimum_duration_ms=1_000,
        target_minimum_duration_ms=2_000,
        target_maximum_duration_ms=8_000,
        maximum_duration_ms=maximum_duration_ms,
        maximum_drift_ms=100,
        minimum_alignment_score=0.0,
        minimum_vad_speech_ratio=0.0,
    )


def words(*spans: tuple[str, int, int, str | None]) -> tuple[SourceWord, ...]:
    """Build source words without hiding the timestamp values under test.

    Returns:
        Source words matching the supplied spans.
    """
    return tuple(
        SourceWord(text=text, start_ms=start, end_ms=end, speaker_id=speaker)
        for text, start, end, speaker in spans
    )


def test_ctc_emission_adapter_returns_token_boundaries() -> None:
    """Synthetic emissions prove model-free CTC boundary extraction."""
    emissions = np.full((6, 3), -5.0, dtype=np.float32)
    emissions[:2, 1] = 5.0
    emissions[3:5, 2] = 5.0
    emissions -= np.logaddexp.reduce(emissions, axis=1, keepdims=True)
    result = align_ctc_emissions(
        emissions=emissions, token_ids=(1, 2), start_ms=100, frame_duration_ms=10.0
    )
    assert result.word_boundaries[0][:2] == (105, 125)
    assert result.word_boundaries[1][:2] == (125, 145)
    assert result.score_type == "ctc-segmentation:min_mean_log_probability"


def test_danish_mapping_is_reversible() -> None:
    """Punctuation/case changes retain the exact Danish source words."""
    result = normalise_alignment_text(
        words=words(("Ærlig,", 0, 1_000, None), ("Ål!", 1_000, 2_000, None)),
        contract=NormalisationContract(
            version="test", case_folding=True, punctuation_removed=True
        ),
    )
    assert result.text == "ærlig ål"
    assert result.word_map == ("Ærlig,", "Ål!")
    assert result.source_word_indexes == (0, 1)


def test_drift_correction_runs_at_most_once() -> None:
    """A persistently bad second alignment is not chased with more corrections."""
    proposal = form_candidate_segments(
        words=words(("hej", 1_000, 3_000, None)),
        source_file_id="source",
        contract=segmentation_contract(),
    )[0]
    calls: list[tuple[int, int]] = []

    def realign(start: int, end: int) -> AlignmentResult:
        calls.append((start, end))
        return AlignmentResult(start + 500, end + 500, 1.0)

    result, count = correct_drift_once(
        proposal=proposal,
        first_alignment=AlignmentResult(1_500, 3_500, 1.0),
        realign=realign,
        maximum_drift_ms=100,
    )
    assert count == 1
    assert len(calls) == 1
    assert result.start_ms == 2_000


def test_exact_ten_seconds_is_not_accepted_by_duration_gate() -> None:
    """The upper duration bound is strict, including at exactly 10,000 ms."""
    from hviske.p1_segments import make_output_row

    proposal = form_candidate_segments(
        words=words(("hej", 0, 10_000, None)),
        source_file_id="source",
        contract=segmentation_contract(),
    )[0]
    decision = make_output_row(
        proposal=proposal,
        alignment=AlignmentResult(0, 10_000, 1.0),
        audio=np.zeros(160_000, dtype=np.float32),
        source_duration_ms=10_000,
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
        segmentation=segmentation_contract(),
    )
    assert decision.rejection == "duration_out_of_range"


def test_flac_is_freshly_decodable() -> None:
    """The encoded payload is lossless PCM-in-FLAC at the required rate."""
    result = encode_flac(np.zeros(16_000, dtype=np.float32))
    assert result.duration_ms == 1_000
    assert result.sha256 == hashlib.sha256(result.payload).hexdigest()


def test_malformed_timestamps_are_rejected() -> None:
    """Raw floats, overlaps, and out-of-range words cannot enter the pipeline."""
    with pytest.raises(ValueError, match="malformed"):
        validate_raw_timestamps(
            words=[{"text": "hej", "start_ms": 0.5, "end_ms": 500}],
            programme_duration_ms=1_000,
        )
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_raw_timestamps(
            words=[
                {"text": "a", "start_ms": 0, "end_ms": 600},
                {"text": "b", "start_ms": 500, "end_ms": 900},
            ],
            programme_duration_ms=1_000,
        )


def test_segment_ids_are_config_sensitive() -> None:
    """The same content under different pipeline identities is not conflated."""
    from hviske.p1_segments import make_output_row

    proposal = form_candidate_segments(
        words=words(("hej", 0, 2_000, None)),
        source_file_id="source",
        contract=segmentation_contract(),
    )[0]
    first = make_output_row(
        proposal=proposal,
        alignment=AlignmentResult(0, 2_000, 1.0),
        audio=np.zeros(32_000, dtype=np.float32),
        source_duration_ms=2_000,
        pipeline_version="test",
        pipeline_config_sha256="a" * 64,
        segmentation=segmentation_contract(),
    )
    second = make_output_row(
        proposal=proposal,
        alignment=AlignmentResult(0, 2_000, 1.0),
        audio=np.zeros(32_000, dtype=np.float32),
        source_duration_ms=2_000,
        pipeline_version="test",
        pipeline_config_sha256="b" * 64,
        segmentation=segmentation_contract(),
    )
    assert first.row is not None and second.row is not None
    assert first.row.segment_id != second.row.segment_id


def test_shards_rotate_and_callback_follows_fsync(tmp_path: Path) -> None:
    """Rotation leaves readable shards and calls deletion only after recovery."""
    from hviske.p1_segments import make_output_row

    rows = []
    for index in range(3):
        proposal = form_candidate_segments(
            words=words((f"hej{index}", 0, 2_000, None)),
            source_file_id=f"source-{index}",
            contract=segmentation_contract(),
        )[0]
        decision = make_output_row(
            proposal=proposal,
            alignment=AlignmentResult(0, 2_000, 1.0),
            audio=np.zeros(32_000, dtype=np.float32),
            source_duration_ms=2_000,
            pipeline_version="test",
            pipeline_config_sha256=CONFIG_DIGEST,
            segmentation=segmentation_contract(),
        )
        assert decision.row is not None
        rows.append(decision.row)
    deleted: list[bool] = []
    result = write_shards(
        rows=rows,
        output_dir=tmp_path,
        target_bytes=1,
        on_source_recoverable=lambda: deleted.append(True),
    )
    assert result.source_recoverable
    assert deleted == [True]
    assert len(result.shards) == 3
    assert all(shard.fsynced for shard in result.shards)
    assert all(shard.evidence.sha256 == shard.sha256 for shard in result.shards)
    assert all(pq.read_table(shard.path).num_rows == 1 for shard in result.shards)
    assert all(stream_sha256(shard.path) == shard.sha256 for shard in result.shards)
    assert all(validate_output_shard(shard.path) is None for shard in result.shards)


def test_source_audio_is_downmixed_and_resampled() -> None:
    """Declared stereo 8 kHz input becomes finite mono 16 kHz audio."""
    source = np.column_stack((np.ones(8_000), -np.ones(8_000))).astype(np.float32)
    result = prepare_source_audio(audio=source, sampling_rate=8_000, channels=2)
    assert result.shape == (16_000,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_vad_ratio_and_edges() -> None:
    """VAD evidence is measurable without loading a VAD model."""
    signal = VADSignal(speech_intervals=((100, 300),), programme_duration_ms=500)
    assert signal.speech_ratio(100, 400) == pytest.approx(2 / 3)
    assert signal.snap_edges(300, 400) == (300, 400)
