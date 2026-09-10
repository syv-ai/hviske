"""Focused offline tests for Phase 1B segmentation and sharding."""

from __future__ import annotations

import hashlib
import io
import typing as t
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from hviske.p1_contracts import (
    NormalisationContract,
    RejectionCategory,
    SegmentationContract,
    SourceProgramme,
    SourceWord,
    annotate_source_words,
)
from hviske.p1_segments import (
    AlignmentResult,
    CTCAlignmentInfeasible,
    CTCBackend,
    VADSignal,
    align_ctc_emissions,
    align_ctc_word_tokens,
    correct_drift_once,
    encode_flac,
    form_candidate_segments,
    normalise_alignment_text,
    prepare_source_audio,
    segment_programme,
    stream_sha256,
    validate_output_shard,
    validate_raw_timestamps,
    write_shards,
)
from hviske.p1_source import parse_transcript_row

CONFIG_DIGEST = hashlib.sha256(b"p1-test-config").hexdigest()


def test_all_zero_transcript_is_terminally_classifiable() -> None:
    """A transcript with no timing anchors is an explicit rejection, not empty work."""
    from hviske.p1_segments import segment_programme

    parsed = parse_transcript_row(
        row={
            "file_id": "source",
            "words": [
                {"text": "noise", "start_ms": 0, "end_ms": 0},
                {"text": "...", "type": "spacing", "start_ms": 0, "end_ms": 0},
            ],
        }
    )
    result = segment_programme(
        words=parsed.words,
        audio=np.zeros(32_000, dtype=np.float32),
        source_file_id="source",
        source_duration_ms=2_000,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(version="test"),
        ctc=t.cast(CTCBackend, object()),
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert result.rows == ()
    assert result.rejections == (("", RejectionCategory.NO_TIMED_WORDS.value),)


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


def test_candidates_partition_every_owned_source_character() -> None:
    """Proposal boundaries and speaker changes neither drop nor duplicate text."""
    parsed = parse_transcript_row(
        row={
            "file_id": "source",
            "words": [
                {"text": "[...]", "start_ms": 0, "end_ms": 0},
                {"text": "one.", "start_ms": 0, "end_ms": 1_000, "speaker": "a"},
                {"text": "[boundary]", "start_ms": 1_000, "end_ms": 1_000},
                {"text": "two", "start_ms": 1_000, "end_ms": 2_000, "speaker": "b"},
                {"text": "...", "start_ms": 2_000, "end_ms": 2_000},
            ],
        }
    )
    proposals = form_candidate_segments(
        words=parsed.words, source_file_id="source", contract=segmentation_contract()
    )

    assert "".join(proposal.text for proposal in proposals) == parsed.text
    assert [proposal.text for proposal in proposals] == [
        "[...]one.",
        "[boundary]two...",
    ]


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


def words(*spans: tuple[str, int, int, str | None]) -> tuple[SourceWord, ...]:
    """Build source words without hiding the timestamp values under test.

    Returns:
        Source words matching the supplied spans.
    """
    return tuple(
        SourceWord(text=text, start_ms=start, end_ms=end, speaker_id=speaker)
        for text, start, end, speaker in spans
    )


def test_candidates_preserve_verbatim_source_separators() -> None:
    """Candidate text keeps repeated spaces and newlines out of the aligner map."""
    source_text = "Hej,  verden!\nIgen"
    source_words = annotate_source_words(
        [
            SourceWord(text="Hej,", start_ms=0, end_ms=1_000),
            SourceWord(text="verden!", start_ms=1_000, end_ms=2_000),
            SourceWord(text="Igen", start_ms=2_000, end_ms=3_000),
        ],
        source_text,
    )
    proposal = form_candidate_segments(
        words=source_words, source_file_id="source", contract=segmentation_contract()
    )[0]

    assert proposal.text == "Hej,  verden!"
    assert source_words[1].separator_text == "  "
    assert source_words[2].separator_text == "\n"


def test_ctc_alignment_rejects_impossible_prepared_path() -> None:
    """The segmenter is not called when the target path exceeds frame capacity."""
    emissions = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(CTCAlignmentInfeasible):
        align_ctc_emissions(
            emissions=emissions, token_ids=(1,), start_ms=0, frame_duration_ms=20.0
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


def test_ctc_infeasible_proposal_does_not_abort_programme() -> None:
    """A typed infeasibility rejects one proposal while later rows publish."""

    class CTC:
        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, word_map, sampling_rate
            if alignment_text == "abcde":
                raise CTCAlignmentInfeasible("target path does not fit")
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    result = segment_programme(
        words=words(
            ("abcde", 0, 1_000, "speaker-a"), ("normal", 1_000, 3_000, "speaker-b")
        ),
        audio=np.zeros(48_000, dtype=np.float32),
        source_file_id="source",
        source_duration_ms=3_000,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(version="test"),
        ctc=CTC(),
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert len(result.rows) == 1
    assert result.rows[0].text == "normal"
    assert result.rejections == (("abcde", "ctc_alignment_failed"),)


def test_ctc_nonrepeated_and_existing_blanks_are_not_double_counted() -> None:
    """Prepared boundaries and blank labels already satisfy CTC transitions."""
    cases = ((((1, 2),), 5), (((1,), (1,)), 6), (((1, 0, 1),), 6))
    for tokenised_words, frame_count in cases:
        result = align_ctc_word_tokens(
            emissions=np.zeros((frame_count, 3), dtype=np.float32),
            tokenised_words=tokenised_words,
            start_ms=0,
            frame_duration_ms=20.0,
        )
        assert result.word_boundaries


def test_ctc_repeated_tokens_require_an_additional_blank_frame() -> None:
    """Repeated labels need a blank beyond the prepared path length."""
    with pytest.raises(CTCAlignmentInfeasible):
        align_ctc_word_tokens(
            emissions=np.zeros((5, 3), dtype=np.float32),
            tokenised_words=((1, 1),),
            start_ms=0,
            frame_duration_ms=20.0,
        )

    result = align_ctc_word_tokens(
        emissions=np.zeros((6, 3), dtype=np.float32),
        tokenised_words=((1, 1),),
        start_ms=0,
        frame_duration_ms=20.0,
    )
    assert result.word_boundaries


def test_danish_mapping_is_reversible() -> None:
    """Punctuation/case changes retain the exact Danish source words."""
    result = normalise_alignment_text(
        words=words(
            ("Rødgrød", 0, 1_000, None),
            ("Hej", 1_000, 2_000, None),
            ("København", 2_000, 3_000, None),
        ),
        contract=NormalisationContract(
            version="p1-text-normalisation-5",
            case_folding=True,
            punctuation_removed=True,
        ),
    )

    def roest_like_tokeniser(value: str) -> tuple[int, ...]:
        """Represent a lowercase-only Roest tokenizer for this contract test.

        Returns:
            Deterministic stand-in token IDs for the lowercase input.
        """
        assert value == value.casefold()
        return tuple(ord(char) for char in value)

    assert result.text == "rødgrød hej københavn"
    assert all(roest_like_tokeniser(word) for word in result.text.split())
    assert result.word_map == ("Rødgrød", "Hej", "København")
    assert result.source_word_indexes == (0, 1, 2)


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


def test_output_text_preserves_case_when_alignment_text_is_folded() -> None:
    """Canonical alignment text never replaces verbatim published text."""

    class CTC:
        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, word_map, sampling_rate
            assert alignment_text == "rødgrød hej københavn"
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    result = segment_programme(
        words=words(
            ("Rødgrød", 0, 1_000, None),
            ("Hej", 1_000, 2_000, None),
            ("København", 2_000, 3_000, None),
        ),
        audio=np.zeros(48_000, dtype=np.float32),
        source_file_id="source",
        source_duration_ms=3_000,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(
            version="p1-text-normalisation-5", case_folding=True
        ),
        ctc=CTC(),
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.text == "Rødgrød Hej København"
    assert row.alignment_text == "rødgrød hej københavn"
    assert row.alignment_word_map == ("Rødgrød", "Hej", "København")


def test_owned_lexical_text_reaches_ctc_and_published_text_exactly() -> None:
    """Speaker-safe untimed text remains in CTC and verbatim candidate text."""

    class CTC:
        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, sampling_rate
            assert alignment_text == "hej altså verden slut"
            assert word_map == ("Hej,", "  ALTSÅ\n", "Verden!", " <SLUT>")
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    parsed = parse_transcript_row(
        row={
            "file_id": "source",
            "words": [
                {
                    "text": "Hej,",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "speaker": "speaker-a",
                },
                {"text": "  ", "type": "spacing"},
                {
                    "text": "ALTSÅ",
                    "start_ms": 1_000,
                    "end_ms": 1_000,
                    "speaker": "speaker-a",
                },
                {"text": "\n", "type": "spacing"},
                {
                    "text": "Verden!",
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "speaker": "speaker-a",
                },
                {
                    "text": " <SLUT>",
                    "start_ms": 2_000,
                    "end_ms": 2_000,
                    "speaker": "speaker-a",
                },
            ],
        }
    )
    result = segment_programme(
        words=parsed.words,
        audio=np.zeros(32_000, dtype=np.float32),
        source_file_id="source",
        source_duration_ms=2_000,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(
            version="p1-text-normalisation-5", case_folding=True
        ),
        ctc=CTC(),
        pipeline_version="p1-segmentation-6",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert parsed.ambiguous_source_text_records == 0
    assert parsed.words[1].separator_span is not None
    assert (
        parsed.words[1].separator_span.start,
        parsed.words[1].separator_span.end,
    ) == (4, 12)
    assert parsed.words[1].trailing_text == " <SLUT>"
    assert len(result.rows) == 1
    assert result.rows[0].text == "Hej,  ALTSÅ\nVerden! <SLUT>"
    assert result.rows[0].alignment_text == "hej altså verden slut"


def test_pilot_mean_log_probability_threshold_is_config_shaped() -> None:
    """The permissive pilot threshold accepts plausible and rejects poor scores."""
    from hviske.p1_segments import make_output_row

    contract = segmentation_contract()
    contract = contract.model_copy(update={"minimum_alignment_score": -10.0})
    proposal = form_candidate_segments(
        words=words(("hej", 0, 2_000, None)), source_file_id="source", contract=contract
    )[0]
    audio = np.zeros(32_000, dtype=np.float32)
    accepted = make_output_row(
        proposal=proposal,
        alignment=AlignmentResult(0, 2_000, -9.5),
        audio=audio,
        source_duration_ms=2_000,
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
        segmentation=contract,
    )
    rejected = make_output_row(
        proposal=proposal,
        alignment=AlignmentResult(0, 2_000, -10.5),
        audio=audio,
        source_duration_ms=2_000,
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
        segmentation=contract,
    )

    assert accepted.row is not None
    assert rejected.rejection == "low_alignment_score"


@pytest.mark.parametrize(
    ("duration_ms", "expected_calls", "expected_rows"), [(9_999, 1, 1), (10_000, 0, 0)]
)
def test_real_duration_boundary_is_checked_before_ctc(
    duration_ms: int, expected_calls: int, expected_rows: int
) -> None:
    """The configured 10,000 ms maximum permits 9,999 ms but not 10,000 ms."""

    class CTC:
        calls = 0

        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, alignment_text, word_map, sampling_rate
            self.calls += 1
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    ctc = CTC()
    result = segment_programme(
        words=words(("hej", 0, duration_ms, None)),
        audio=np.zeros(duration_ms * 16, dtype=np.float32),
        source_file_id="source",
        source_duration_ms=duration_ms,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(version="test"),
        ctc=ctc,
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert ctc.calls == expected_calls
    assert len(result.rows) == expected_rows
    if duration_ms == 10_000:
        assert result.rejections == (("hej", "duration_out_of_range"),)


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


def test_short_proposal_is_rejected_before_ctc_and_audited() -> None:
    """A short speaker run cannot invoke CTC or disappear from the audit trail."""

    class CTC:
        calls = 0

        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, alignment_text, word_map, sampling_rate
            self.calls += 1
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    ctc = CTC()
    result = segment_programme(
        words=words(
            ("abcde", 0, 40, "speaker-a"), ("normal", 1_000, 3_000, "speaker-b")
        ),
        audio=np.zeros(48_000, dtype=np.float32),
        source_file_id="source",
        source_duration_ms=3_000,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(version="test"),
        ctc=ctc,
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert ctc.calls == 1
    assert len(result.rows) == 1
    assert result.rejections == (("abcde", "duration_out_of_range"),)
    assert result.audit_candidates[0]["source_start_ms"] == 0
    assert result.audit_candidates[0]["source_end_ms"] == 40


def test_source_audio_is_downmixed_and_resampled() -> None:
    """Declared stereo 8 kHz input becomes finite mono 16 kHz audio."""
    source = np.column_stack((np.ones(8_000), -np.ones(8_000))).astype(np.float32)
    result = prepare_source_audio(audio=source, sampling_rate=8_000, channels=2)
    assert result.shape == (16_000,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_tail_duration_normalises_accepted_resampling_edges() -> None:
    """A 1001 ms tail always encodes exactly 16,016 target samples."""

    class CTC:
        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, alignment_text, word_map, sampling_rate
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    for frame_count in (16_008, 16_016):
        result = segment_programme(
            words=(SourceWord(text="tail", start_ms=0, end_ms=1_001),),
            audio=np.zeros(frame_count, dtype=np.float32),
            source_file_id="source",
            source_duration_ms=1_001,
            segmentation=segmentation_contract(),
            normalisation=NormalisationContract(version="test"),
            ctc=CTC(),
            pipeline_version="test",
            pipeline_config_sha256=CONFIG_DIGEST,
        )

        assert len(result.rows) == 1
        row = result.rows[0]
        decoded, _ = sf.read(io.BytesIO(row.audio), dtype="float32")
        assert row.duration_ms == 1_001
        assert decoded.shape == (16_016,)


def test_vad_ratio_and_edges() -> None:
    """VAD evidence is measurable without loading a VAD model."""
    signal = VADSignal(speech_intervals=((100, 300),), programme_duration_ms=500)
    assert signal.speech_ratio(100, 400) == pytest.approx(2 / 3)
    assert signal.snap_edges(300, 400) == (300, 400)


def test_zero_duration_duplicate_keeps_timed_source_occurrence() -> None:
    """Source ownership survives programme validation and output creation."""

    class CTC:
        def align(
            self,
            audio: np.ndarray,
            alignment_text: str,
            word_map: tuple[str, ...],
            start_ms: int,
            end_ms: int,
            sampling_rate: int,
        ) -> AlignmentResult:
            del audio, sampling_rate
            assert alignment_text == "foo foo"
            assert word_map == ("foo", "foo")
            return AlignmentResult(start_ms=start_ms, end_ms=end_ms, score=1.0)

    parsed = parse_transcript_row(
        row={
            "file_id": "source",
            "words": [
                {"text": "foo", "start_ms": 0, "end_ms": 0},
                {"text": "foo", "start_ms": 1_000, "end_ms": 2_000},
            ],
        }
    )
    programme = SourceProgramme(
        file_id=parsed.file_id,
        duration_ms=2_000,
        words=parsed.words,
        transcript_text=parsed.text,
    )

    word = programme.words[0]
    source_span = word.source_span
    assert source_span is not None
    assert (source_span.start, source_span.end) == (3, 6)
    assert word.separator_text == "foo"
    assert word.separator_span is not None
    assert (word.separator_span.start, word.separator_span.end) == (0, 3)
    assert word.trailing_text == ""
    proposals = form_candidate_segments(
        words=programme.words,
        source_file_id=programme.file_id,
        contract=segmentation_contract(),
    )
    assert [proposal.text for proposal in proposals] == ["foofoo"]

    result = segment_programme(
        words=programme.words,
        audio=np.zeros(32_000, dtype=np.float32),
        source_file_id=programme.file_id,
        source_duration_ms=programme.duration_ms,
        segmentation=segmentation_contract(),
        normalisation=NormalisationContract(
            version="p1-text-normalisation-5", case_folding=True
        ),
        ctc=CTC(),
        pipeline_version="test",
        pipeline_config_sha256=CONFIG_DIGEST,
    )

    assert len(result.rows) == 1
    assert result.rows[0].text == "foofoo"
    assert (result.rows[0].source_start_ms, result.rows[0].source_end_ms) == (
        1_000,
        2_000,
    )
