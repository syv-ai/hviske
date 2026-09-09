"""Phase 1B proposal, alignment, quality, and local sharding helpers.

The module deliberately keeps model loading outside the pipeline.  VAD and CTC
implementations are small protocols, so production code can provide pinned models
while tests can provide deterministic fakes.
"""

from __future__ import annotations

import collections.abc as c
import hashlib
import io
import math
import os
import re
import tempfile
import typing as t
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from .p1_contracts import (
    NormalisationContract,
    OutputRow,
    RejectionCategory,
    SegmentationContract,
    SegmentProposal,
    ShardEvidence,
    SourceWord,
    segment_id,
)


@dataclass(frozen=True)
class AlignmentResult:
    """CTC boundaries and score for one candidate.

    Boundaries are absolute programme-relative milliseconds.  ``word_boundaries``
    contains optional absolute spans and scores for the normalised units.
    """

    start_ms: int
    end_ms: int
    score: float
    score_type: str = "ctc-segmentation:min_chunk_mean"
    word_boundaries: tuple[tuple[int, int, float], ...] = ()
    raw_score_inputs: tuple[float, ...] = ()


class CTCBackend(t.Protocol):
    """Injectable CTC forced-alignment implementation."""

    def align(
        self,
        audio: np.ndarray,
        alignment_text: str,
        word_map: tuple[str, ...],
        start_ms: int,
        end_ms: int,
        sampling_rate: int,
    ) -> AlignmentResult:
        """Align one local candidate and return absolute boundaries."""


@dataclass(frozen=True)
class CandidateDecision:
    """Result of applying the publication quality gates."""

    row: OutputRow | None
    rejection: str | None = None
    start_drift_ms: int = 0
    end_drift_ms: int = 0
    correction_count: int = 0


@dataclass(frozen=True)
class EncodedAudio:
    """A validated mono 16 kHz FLAC payload."""

    payload: bytes
    duration_ms: int
    sample_count: int
    sha256: str


@dataclass(frozen=True)
class SegmentationResult:
    """Accepted rows, stable rejection categories, and correction evidence."""

    rows: tuple[OutputRow, ...]
    rejections: tuple[tuple[str, str], ...]
    correction_count: int


@dataclass(frozen=True)
class ShardWriteResult:
    """Recoverability evidence for one atomically closed Parquet shard."""

    path: Path
    row_count: int
    byte_size: int
    sha256: str
    fsynced: bool

    @property
    def evidence(self) -> ShardEvidence:
        """Contract metadata used by restart and upload checks."""
        return ShardEvidence(
            path=str(self.path),
            byte_size=self.byte_size,
            row_count=self.row_count,
            sha256=self.sha256,
        )


@dataclass(frozen=True)
class ShardBatchResult:
    """Evidence for a bounded set of locally recoverable shards."""

    shards: tuple[ShardWriteResult, ...]
    source_recoverable: bool


@dataclass(frozen=True)
class VADSignal:
    """Frame-independent VAD evidence used for edge snapping and quality gates.

    ``speech_intervals`` are half-open programme-relative millisecond intervals.
    They are evidence only: this class never labels a transcript or rejects music.
    """

    speech_intervals: tuple[tuple[int, int], ...]
    programme_duration_ms: int
    boundary_clipping: float = 0.0

    def snap_edges(
        self, start_ms: int, end_ms: int, maximum_edge_ms: int = 300
    ) -> tuple[int, int]:
        """Snap edges towards nearby silence without crossing transcript words.

        Returns:
            The bounded, potentially snapped start and end in milliseconds.
        """
        starts = [start for start, _ in self.speech_intervals]
        ends = [end for _, end in self.speech_intervals]
        nearby_start = max((end for end in ends if end <= start_ms), default=start_ms)
        nearby_end = min((start for start in starts if start >= end_ms), default=end_ms)
        if start_ms - nearby_start <= maximum_edge_ms:
            start_ms = nearby_start
        if nearby_end - end_ms <= maximum_edge_ms:
            end_ms = nearby_end
        return max(0, start_ms), min(self.programme_duration_ms, end_ms)

    def speech_ratio(self, start_ms: int, end_ms: int) -> float:
        """Return the fraction of an interval covered by VAD speech."""
        if end_ms <= start_ms:
            return 0.0
        speech_ms = sum(
            max(0, min(end_ms, end) - max(start_ms, start))
            for start, end in self.speech_intervals
        )
        return min(1.0, speech_ms / (end_ms - start_ms))


class VADBackend(t.Protocol):
    """Injectable offline VAD implementation."""

    def analyse(self, audio: np.ndarray, sampling_rate: int) -> VADSignal:
        """Return programme-level speech intervals and edge evidence."""


@dataclass(frozen=True)
class AlignmentText:
    """Canonical aligner text and its reversible source-word mapping."""

    text: str
    word_map: tuple[str, ...]
    source_word_indexes: tuple[int, ...]

    @property
    def alignment_text(self) -> str:
        """Canonical text using the output-contract name."""
        return self.text

    @property
    def alignment_word_map(self) -> tuple[str, ...]:
        """One source word for each canonical alignment unit."""
        return self.word_map


def normalise_alignment_text(
    words: c.Sequence[SourceWord], contract: NormalisationContract
) -> AlignmentText:
    """Normalise words for alignment while retaining a reversible word map.

    Each whitespace-separated output unit points at the exact original word.  A
    number expansion may therefore create several units with the same mapping;
    published text is never reconstructed from this normalised representation.

    Returns:
        Canonical text, source-word mapping, and source indexes.
    """
    units: list[str] = []
    mapping: list[str] = []
    indexes: list[int] = []
    for index, word in enumerate(words):
        value = unicodedata.normalize(
            t.cast(t.Literal["NFC", "NFD", "NFKC", "NFKD"], contract.unicode_form),
            word.text,
        )
        if contract.case_folding:
            value = value.casefold()
        if contract.punctuation_removed:
            value = "".join(
                char if not unicodedata.category(char).startswith("P") else " "
                for char in value
            )
        if contract.number_expansion:
            value = _expand_danish_numbers(value)
        value = _retain_alignment_characters(value)
        for unit in value.split():
            units.append(unit)
            mapping.append(word.text)
            indexes.append(index)
    return AlignmentText(
        text=" ".join(units),
        word_map=tuple(mapping),
        source_word_indexes=tuple(indexes),
    )


def validate_raw_timestamps(
    words: c.Iterable[dict[str, object]], programme_duration_ms: int
) -> tuple[SourceWord, ...]:
    """Parse untrusted word dictionaries and validate their timestamp spans.

    Returns:
        Validated source words.

    Raises:
        SourceValidationError:
            If a dictionary is malformed or its timeline is invalid.
        TypeError:
            If a supplied collection is not iterable.
    """
    parsed: list[SourceWord] = []
    for index, value in enumerate(words):
        try:
            text = value["text"]
            speaker = value.get("speaker_id")
            if "start_ms" in value and "end_ms" in value:
                start = value["start_ms"]
                end = value["end_ms"]
                if not isinstance(start, int) or isinstance(start, bool):
                    raise TypeError("start_ms")
                if not isinstance(end, int) or isinstance(end, bool):
                    raise TypeError("end_ms")
            else:
                start_seconds = value["start"]
                end_seconds = value["end"]
                if (
                    isinstance(start_seconds, bool)
                    or isinstance(end_seconds, bool)
                    or not isinstance(start_seconds, (int, float))
                    or not isinstance(end_seconds, (int, float))
                    or not math.isfinite(float(start_seconds))
                    or not math.isfinite(float(end_seconds))
                ):
                    raise TypeError("seconds")
                start = int(round(float(start_seconds) * 1000))
                end = int(round(float(end_seconds) * 1000))
            if not isinstance(text, str):
                raise TypeError("text")
            if speaker is not None and not isinstance(speaker, str):
                raise TypeError("speaker_id")
            parsed.append(
                SourceWord(text=text, start_ms=start, end_ms=end, speaker_id=speaker)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceValidationError(
                f"word {index} has malformed timestamps"
            ) from exc
    return validate_source_words(
        words=parsed, programme_duration_ms=programme_duration_ms
    )


class SourceValidationError(ValueError):
    """Raised when source words cannot form a monotonic audio timeline."""

    def __init__(
        self,
        message: str,
        category: RejectionCategory = RejectionCategory.INVALID_TIMESTAMPS,
    ) -> None:
        """Create an error with a stable ledger rejection category.

        Args:
            message:
                Human-readable validation detail.
            category:
                Stable category for durable rejection accounting.
        """
        super().__init__(message)
        self.category = category


def validate_source_words(
    words: c.Iterable[SourceWord], programme_duration_ms: int
) -> tuple[SourceWord, ...]:
    """Validate and return source words on a single monotonic millisecond timebase.

    Args:
        words:
            Transcript words, already converted to integer milliseconds.
        programme_duration_ms:
            Decoded source duration in milliseconds.

    Returns:
        The validated words as an immutable tuple.

    Raises:
            SourceValidationError:
                If a timestamp is missing, non-finite, out of range, or non-monotonic.
    """
    if isinstance(programme_duration_ms, bool) or programme_duration_ms <= 0:
        raise SourceValidationError("programme duration must be positive")
    validated = tuple(words)
    previous_end = 0
    for index, word in enumerate(validated):
        values = (word.start_ms, word.end_ms)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise SourceValidationError(f"word {index} has a non-integer timestamp")
        if word.start_ms < 0 or word.end_ms <= word.start_ms:
            raise SourceValidationError(f"word {index} has an invalid span")
        if word.start_ms < previous_end:
            raise SourceValidationError(f"word {index} is substantially non-monotonic")
        if word.end_ms > programme_duration_ms:
            raise SourceValidationError(f"word {index} lies outside source audio")
        previous_end = word.end_ms
    return validated


# American spelling is useful at integration boundaries and remains one implementation.
normalize_alignment_text = normalise_alignment_text


class CTCEmissionsAlignmentAdapter:
    """Adapt an injected emission provider to :class:`CTCBackend`.

    No checkpoint or model is loaded by this class.  ``segmenter`` can be a
    pinned ``ctc-segmentation`` implementation; the deterministic adapter is
    used when it is omitted, which makes boundary logic testable offline.
    """

    def __init__(
        self,
        emissions_provider: c.Callable[[np.ndarray, int], np.ndarray],
        tokeniser: c.Callable[[str], c.Sequence[int]],
        frame_duration_ms: float,
        blank_id: int = 0,
        segmenter: c.Callable[..., AlignmentResult] | None = None,
    ) -> None:
        """Configure injected emission and tokenisation functions.

        Args:
            emissions_provider:
                Function returning frame-by-class emissions.
            tokeniser:
                Function mapping canonical text to CTC token IDs.
            frame_duration_ms:
                Duration represented by one emission frame.
            blank_id:
                CTC blank class.
            segmenter:
                Optional pinned ``ctc-segmentation`` callable.
        """
        self._emissions_provider = emissions_provider
        self._tokeniser = tokeniser
        self._frame_duration_ms = frame_duration_ms
        self._blank_id = blank_id
        self._segmenter = segmenter

    def align(
        self,
        audio: np.ndarray,
        alignment_text: str,
        word_map: tuple[str, ...],
        start_ms: int,
        end_ms: int,
        sampling_rate: int,
    ) -> AlignmentResult:
        """Align audio using injected emissions and optional CTC segmenter.

        Returns:
            Absolute alignment boundaries and backend score.
        """
        del word_map, end_ms
        emissions = self._emissions_provider(audio, sampling_rate)
        tokens = self._tokeniser(alignment_text)
        if self._segmenter is not None:
            return self._segmenter(
                emissions, tokens, start_ms, self._frame_duration_ms, self._blank_id
            )
        return align_ctc_emissions(
            emissions=emissions,
            token_ids=tokens,
            start_ms=start_ms,
            frame_duration_ms=self._frame_duration_ms,
            blank_id=self._blank_id,
        )


def align_ctc_emissions(
    emissions: np.ndarray,
    token_ids: c.Sequence[int],
    start_ms: int,
    frame_duration_ms: float,
    blank_id: int = 0,
) -> AlignmentResult:
    """Derive model-independent token spans from CTC emissions.

    This deterministic adapter is also a safe test double for the pinned
    ``ctc-segmentation`` backend.  Production callers may replace it with the
    package's chunk scorer without changing the surrounding contracts.

    Returns:
        Absolute token boundaries and the minimum token confidence.

    Raises:
        ValueError:
            If emissions or token IDs are empty or have an invalid shape.
    """
    values = np.asarray(emissions)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("emissions must be a non-empty frame by class matrix")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    classes = np.argmax(values, axis=1)
    spans: list[tuple[int, int, float]] = []
    cursor = 0
    frame_count = len(classes)
    for token in token_ids:
        locations: list[int] = []
        while cursor < frame_count:
            if int(classes[cursor]) == token:
                locations.append(cursor)
            cursor += 1
            if (
                locations
                and cursor < frame_count
                and int(classes[cursor]) not in (token, blank_id)
            ):
                break
        if locations:
            first, last = locations[0], locations[-1] + 1
            score = float(np.mean(values[first:last, token]))
        else:
            first = last = cursor
            score = 0.0
        spans.append(
            (
                int(round(start_ms + first * frame_duration_ms)),
                int(round(start_ms + last * frame_duration_ms)),
                score,
            )
        )
        if cursor >= frame_count:
            cursor = frame_count
    valid_scores = [score for _, _, score in spans]
    return AlignmentResult(
        start_ms=spans[0][0],
        end_ms=spans[-1][1],
        score=min(valid_scores),
        word_boundaries=tuple(spans),
        raw_score_inputs=tuple(valid_scores),
    )


def form_candidate_segments(
    words: c.Sequence[SourceWord],
    source_file_id: str,
    contract: SegmentationContract,
    vad: VADSignal | None = None,
) -> tuple[SegmentProposal, ...]:
    """Form consecutive, speaker-safe proposals with deterministic boundaries.

    Returns:
        Proposals whose word ranges partition the supplied words.
    """
    if not words:
        return ()
    proposals: list[SegmentProposal] = []
    run_start = 0
    while run_start < len(words):
        run_speaker = words[run_start].speaker_id
        run_end = run_start + 1
        while run_end < len(words) and words[run_end].speaker_id == run_speaker:
            run_end += 1
        proposals.extend(
            _proposals_for_speaker_run(
                words=words,
                run_start=run_start,
                run_end=run_end,
                source_file_id=source_file_id,
                contract=contract,
                vad=vad,
            )
        )
        run_start = run_end
    return tuple(proposals)


# Short aliases keep integration adapters readable without changing the protocol.
CTCSegmentationAdapter = CTCEmissionsAlignmentAdapter
CTCAlignmentAdapter = CTCEmissionsAlignmentAdapter


class ShardWriter:
    """Incremental atomic Parquet writer with bounded rotation."""

    def __init__(
        self,
        output_dir: Path,
        target_bytes: int = 500_000_000,
        on_source_recoverable: c.Callable[[], None] | None = None,
    ) -> None:
        """Create a writer whose temporary files share the destination directory.

        Args:
            output_dir:
                Directory receiving atomically renamed shards.
            target_bytes:
                Approximate rotation target in bytes.
            on_source_recoverable:
                Optional callback after close has fsynced every shard.

        Raises:
            ValueError:
                If the target byte bound is not positive.
        """
        if target_bytes <= 0:
            raise ValueError("target_bytes must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_bytes = target_bytes
        self._pending: list[OutputRow] = []
        self._results: list[ShardWriteResult] = []
        self._next_index = _next_shard_index(self.output_dir)
        self._on_source_recoverable = on_source_recoverable
        self._closed = False

    def close(self) -> tuple[ShardWriteResult, ...]:
        """Close the writer and return all fsync evidence.

        Returns:
            Evidence for every shard written by this writer.
        """
        if not self._closed:
            if self._pending:
                self._results.append(
                    _write_one_shard(
                        rows=self._pending,
                        output_dir=self.output_dir,
                        index=self._next_index,
                    )
                )
                self._pending = []
            self._closed = True
            if self._results and all(result.fsynced for result in self._results):
                if self._on_source_recoverable is not None:
                    self._on_source_recoverable()
        return tuple(self._results)

    def append(self, row: OutputRow) -> ShardWriteResult | None:
        """Append a row and rotate when the shard reaches its target.

        Returns:
            Fsync evidence when this append rotates a shard, otherwise ``None``.

        Raises:
            RuntimeError:
                If the writer has already been closed.
        """
        if self._closed:
            raise RuntimeError("cannot append to a closed shard writer")
        self._pending.append(row)
        if _table_size(self._pending) < self.target_bytes:
            return None
        result = _write_one_shard(
            rows=self._pending, output_dir=self.output_dir, index=self._next_index
        )
        self._results.append(result)
        self._next_index += 1
        self._pending = []
        return result


def _next_shard_index(output_dir: Path) -> int:
    indexes = [
        int(match.group(1))
        for path in output_dir.glob("part-*.parquet")
        if (match := re.fullmatch(r"part-(\d+)\.parquet", path.name))
    ]
    return max(indexes, default=-1) + 1


def _table_size(rows: c.Sequence[OutputRow]) -> int:
    sink = pa.BufferOutputStream()
    pq.write_table(_rows_table(rows), sink, compression="zstd")
    return sink.getvalue().size


def _rows_table(rows: c.Sequence[OutputRow]) -> pa.Table:
    schema = pa.schema(
        [
            ("audio", pa.binary()),
            ("audio_sha256", pa.string()),
            ("text", pa.string()),
            ("alignment_text", pa.string()),
            ("alignment_word_map", pa.list_(pa.string())),
            ("language", pa.string()),
            ("segment_id", pa.string()),
            ("source_file_id", pa.string()),
            ("source_start_ms", pa.int64()),
            ("source_end_ms", pa.int64()),
            ("duration_ms", pa.int32()),
            ("speaker_ids", pa.list_(pa.string())),
            ("proposal_start_ms", pa.int64()),
            ("proposal_end_ms", pa.int64()),
            ("alignment_score", pa.float32()),
            ("alignment_score_type", pa.string()),
            ("start_drift_ms", pa.int32()),
            ("end_drift_ms", pa.int32()),
            ("vad_speech_ratio", pa.float32()),
            ("alignment_backend", pa.string()),
            ("pipeline_version", pa.string()),
            ("pipeline_config_sha256", pa.string()),
        ]
    )
    return pa.table(
        {
            "audio": [row.audio for row in rows],
            "audio_sha256": [row.audio_sha256 for row in rows],
            "text": [row.text for row in rows],
            "alignment_text": [row.alignment_text for row in rows],
            "alignment_word_map": [list(row.alignment_word_map) for row in rows],
            "language": [row.language for row in rows],
            "segment_id": [row.segment_id for row in rows],
            "source_file_id": [row.source_file_id for row in rows],
            "source_start_ms": [row.source_start_ms for row in rows],
            "source_end_ms": [row.source_end_ms for row in rows],
            "duration_ms": [row.duration_ms for row in rows],
            "speaker_ids": [list(row.speaker_ids) for row in rows],
            "proposal_start_ms": [row.proposal_start_ms for row in rows],
            "proposal_end_ms": [row.proposal_end_ms for row in rows],
            "alignment_score": [row.alignment_score for row in rows],
            "alignment_score_type": [row.alignment_score_type for row in rows],
            "start_drift_ms": [row.start_drift_ms for row in rows],
            "end_drift_ms": [row.end_drift_ms for row in rows],
            "vad_speech_ratio": [row.vad_speech_ratio for row in rows],
            "alignment_backend": [row.alignment_backend for row in rows],
            "pipeline_version": [row.pipeline_version for row in rows],
            "pipeline_config_sha256": [row.pipeline_config_sha256 for row in rows],
        },
        schema=schema,
    )


def _write_one_shard(
    rows: c.Sequence[OutputRow], output_dir: Path, index: int
) -> ShardWriteResult:
    final_path = output_dir / f"part-{index:05d}.parquet"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=output_dir
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(_rows_table(rows), temporary_path, compression="zstd")
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        payload = final_path.read_bytes()
        return ShardWriteResult(
            path=final_path,
            row_count=len(rows),
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            fsynced=True,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _expand_danish_numbers(value: str) -> str:
    numbers = {
        "0": "nul",
        "1": "en",
        "2": "to",
        "3": "tre",
        "4": "fire",
        "5": "fem",
        "6": "seks",
        "7": "syv",
        "8": "otte",
        "9": "ni",
        "10": "ti",
        "11": "elleve",
        "12": "tolv",
        "13": "tretten",
        "14": "fjorten",
        "15": "femten",
        "16": "seksten",
        "17": "sytten",
        "18": "atten",
        "19": "nitten",
        "20": "tyve",
    }
    return " ".join(numbers.get(token, token) for token in value.split())


def _proposals_for_speaker_run(
    words: c.Sequence[SourceWord],
    run_start: int,
    run_end: int,
    source_file_id: str,
    contract: SegmentationContract,
    vad: VADSignal | None,
) -> list[SegmentProposal]:
    proposals: list[SegmentProposal] = []
    start = run_start
    while start < run_end:
        end = start + 1
        while end < run_end:
            duration = words[end - 1].end_ms - words[start].start_ms
            next_duration = words[end].end_ms - words[start].start_ms
            if duration >= contract.target_minimum_duration_ms and (
                _punctuation_boundary(words[end - 1].text)
                or _vad_boundary(vad, words[end - 1].end_ms, words[end].start_ms)
                or next_duration > contract.target_maximum_duration_ms
            ):
                break
            end += 1
        while (
            end < run_end
            and words[end - 1].end_ms - words[start].start_ms
            < contract.target_minimum_duration_ms
        ):
            end += 1
        selected = words[start:end]
        proposals.append(
            SegmentProposal(
                source_file_id=source_file_id,
                word_start_index=start,
                word_end_index=end,
                proposal_start_ms=selected[0].start_ms,
                proposal_end_ms=selected[-1].end_ms,
                text=" ".join(word.text for word in selected),
                speaker_ids=tuple(
                    dict.fromkeys(
                        word.speaker_id
                        for word in selected
                        if word.speaker_id is not None
                    )
                ),
            )
        )
        start = end
    return proposals


def _punctuation_boundary(text: str) -> bool:
    return bool(re.search(r"[.!?;:]$", text.rstrip()))


def _vad_boundary(vad: VADSignal | None, left_end: int, right_start: int) -> bool:
    if vad is None or right_start <= left_end:
        return False
    return vad.speech_ratio(left_end, right_start) == 0.0


def _retain_alignment_characters(value: str) -> str:
    return "".join(
        char if (char.isalnum() or char.isspace() or char in "æøåÆØÅ") else " "
        for char in value
    )


def segment_programme(
    words: c.Sequence[SourceWord],
    audio: np.ndarray,
    source_file_id: str,
    source_duration_ms: int,
    segmentation: SegmentationContract,
    normalisation: NormalisationContract,
    ctc: CTCBackend,
    pipeline_version: str,
    pipeline_config_sha256: str,
    vad: VADBackend | None = None,
) -> SegmentationResult:
    """Run bounded proposal, VAD, CTC, correction, filtering, and encoding.

    Returns:
        Accepted rows, rejection categories, and correction evidence.

    Raises:
        SourceValidationError:
            If source audio does not match its declared duration or timestamps fail.
    """
    values = np.asarray(audio)
    if values.ndim != 1 or values.size != source_duration_ms * 16:
        raise SourceValidationError(
            "source audio length does not match its declared duration",
            category=RejectionCategory.MISSING_AUDIO,
        )
    validated = validate_source_words(
        words=words, programme_duration_ms=source_duration_ms
    )
    vad_signal = None if vad is None else vad.analyse(values, 16000)
    proposals = form_candidate_segments(
        words=validated,
        source_file_id=source_file_id,
        contract=segmentation,
        vad=vad_signal,
    )
    rows: list[OutputRow] = []
    rejections: list[tuple[str, str]] = []
    identifiers: set[str] = set()
    correction_count = 0
    for proposal in proposals:
        canonical = normalise_alignment_text(
            words=validated[proposal.word_start_index : proposal.word_end_index],
            contract=normalisation,
        )
        if not canonical.text:
            rejections.append((proposal.text, RejectionCategory.EMPTY_TEXT.value))
            continue
        local_audio = np.asarray(audio)[
            proposal.proposal_start_ms * 16 : proposal.proposal_end_ms * 16
        ]
        first = ctc.align(
            audio=local_audio,
            alignment_text=canonical.text,
            word_map=canonical.word_map,
            start_ms=proposal.proposal_start_ms,
            end_ms=proposal.proposal_end_ms,
            sampling_rate=16000,
        )
        final_alignment, applied = correct_drift_once(
            proposal=proposal,
            first_alignment=first,
            maximum_drift_ms=segmentation.maximum_drift_ms,
            realign=lambda start, end: ctc.align(
                audio=np.asarray(audio)[start * 16 : end * 16],
                alignment_text=canonical.text,
                word_map=canonical.word_map,
                start_ms=start,
                end_ms=end,
                sampling_rate=16000,
            ),
        )
        correction_count += applied
        if vad_signal is not None:
            snapped_start, snapped_end = vad_signal.snap_edges(
                start_ms=final_alignment.start_ms, end_ms=final_alignment.end_ms
            )
            final_alignment = AlignmentResult(
                start_ms=min(snapped_start, proposal.proposal_start_ms),
                end_ms=max(snapped_end, proposal.proposal_end_ms),
                score=final_alignment.score,
                score_type=final_alignment.score_type,
                word_boundaries=final_alignment.word_boundaries,
                raw_score_inputs=final_alignment.raw_score_inputs,
            )
        decision = make_output_row(
            proposal=proposal,
            alignment=final_alignment,
            audio=np.asarray(audio),
            source_duration_ms=source_duration_ms,
            pipeline_version=pipeline_version,
            pipeline_config_sha256=pipeline_config_sha256,
            segmentation=segmentation,
            vad_signal=vad_signal,
            alignment_backend=type(ctc).__name__,
            alignment_text=canonical.text,
            alignment_word_map=canonical.word_map,
        )
        if decision.row is None:
            rejections.append((proposal.text, decision.rejection or "rejected"))
        elif decision.row.segment_id in identifiers:
            rejections.append(
                (proposal.text, RejectionCategory.DUPLICATE_SEGMENT_ID.value)
            )
        else:
            identifiers.add(decision.row.segment_id)
            rows.append(decision.row)
    return SegmentationResult(
        rows=tuple(rows),
        rejections=tuple(rejections),
        correction_count=correction_count,
    )


def correct_drift_once(
    proposal: SegmentProposal,
    first_alignment: AlignmentResult,
    realign: c.Callable[[int, int], AlignmentResult],
    maximum_drift_ms: int,
) -> tuple[AlignmentResult, int]:
    """Apply at most one robust offset correction and return alignment evidence.

    Returns:
        The first or once-corrected alignment and the number of corrections applied.
    """
    deltas = (
        first_alignment.start_ms - proposal.proposal_start_ms,
        first_alignment.end_ms - proposal.proposal_end_ms,
    )
    if max(abs(delta) for delta in deltas) <= maximum_drift_ms:
        return first_alignment, 0
    if abs(deltas[0] - deltas[1]) > maximum_drift_ms:
        return first_alignment, 0
    offset = int(round(float(np.median(deltas))))
    corrected = realign(
        proposal.proposal_start_ms + offset, proposal.proposal_end_ms + offset
    )
    return corrected, 1


def make_output_row(
    proposal: SegmentProposal,
    alignment: AlignmentResult,
    audio: np.ndarray,
    source_duration_ms: int,
    pipeline_version: str,
    pipeline_config_sha256: str,
    segmentation: SegmentationContract,
    vad_signal: VADSignal | None = None,
    alignment_backend: str = "ctc-segmentation",
    alignment_text: str | None = None,
    alignment_word_map: tuple[str, ...] | None = None,
) -> CandidateDecision:
    """Apply quality gates and create one publication-ready output row.

    Returns:
        An accepted output row or a stable rejection category.
    """
    final_start = max(0, alignment.start_ms)
    final_end = min(source_duration_ms, alignment.end_ms)
    duration = final_end - final_start
    if (
        duration < segmentation.minimum_duration_ms
        or duration >= segmentation.maximum_duration_ms
    ):
        return CandidateDecision(row=None, rejection="duration_out_of_range")
    if not proposal.text.strip():
        return CandidateDecision(row=None, rejection=RejectionCategory.EMPTY_TEXT.value)
    if len(proposal.speaker_ids) > 1:
        return CandidateDecision(
            row=None, rejection=RejectionCategory.SPEAKER_OVERLAP.value
        )
    if vad_signal is not None and vad_signal.boundary_clipping > 0.5:
        return CandidateDecision(
            row=None, rejection=RejectionCategory.BOUNDARY_CLIPPING.value
        )
    if not any(_is_trainable_character(char) for char in proposal.text):
        return CandidateDecision(
            row=None, rejection=RejectionCategory.UNSUPPORTED_TEXT.value
        )
    drift_start = final_start - proposal.proposal_start_ms
    drift_end = final_end - proposal.proposal_end_ms
    if max(abs(drift_start), abs(drift_end)) > segmentation.maximum_drift_ms:
        return CandidateDecision(
            row=None,
            rejection=RejectionCategory.EXCESSIVE_DRIFT.value,
            start_drift_ms=drift_start,
            end_drift_ms=drift_end,
        )
    if (
        not math.isfinite(alignment.score)
        or alignment.score < segmentation.minimum_alignment_score
    ):
        return CandidateDecision(
            row=None, rejection=RejectionCategory.LOW_ALIGNMENT_SCORE.value
        )
    ratio = (
        1.0 if vad_signal is None else vad_signal.speech_ratio(final_start, final_end)
    )
    if ratio < segmentation.minimum_vad_speech_ratio:
        return CandidateDecision(
            row=None, rejection=RejectionCategory.LOW_SPEECH_RATIO.value
        )
    first_sample = int(round(final_start * 16))
    last_sample = int(round(final_end * 16))
    try:
        encoded = encode_flac(samples=np.asarray(audio)[first_sample:last_sample])
    except (RuntimeError, ValueError, sf.LibsndfileError):
        return CandidateDecision(
            row=None, rejection=RejectionCategory.DECODE_ERROR.value
        )
    identifier = segment_id(
        pipeline_config_sha256=pipeline_config_sha256,
        source_file_id=proposal.source_file_id,
        source_start_ms=final_start,
        source_end_ms=final_end,
        text=proposal.text,
    )
    row = OutputRow(
        audio=encoded.payload,
        audio_sha256=encoded.sha256,
        text=proposal.text,
        alignment_text=alignment_text if alignment_text is not None else proposal.text,
        alignment_word_map=(
            alignment_word_map
            if alignment_word_map is not None
            else tuple(proposal.text.split())
        ),
        segment_id=identifier,
        source_file_id=proposal.source_file_id,
        source_start_ms=final_start,
        source_end_ms=final_end,
        duration_ms=duration,
        speaker_ids=proposal.speaker_ids,
        proposal_start_ms=proposal.proposal_start_ms,
        proposal_end_ms=proposal.proposal_end_ms,
        alignment_score=float(alignment.score),
        alignment_score_type=alignment.score_type,
        start_drift_ms=drift_start,
        end_drift_ms=drift_end,
        vad_speech_ratio=float(ratio),
        alignment_backend=alignment_backend,
        pipeline_version=pipeline_version,
        pipeline_config_sha256=pipeline_config_sha256,
    )
    return CandidateDecision(
        row=row, start_drift_ms=drift_start, end_drift_ms=drift_end
    )


def _is_trainable_character(char: str) -> bool:
    return char.isalpha() or char.isdigit()


def encode_flac(samples: np.ndarray, sampling_rate: int = 16000) -> EncodedAudio:
    """Encode mono audio as lossless FLAC and validate it with a fresh decode.

    Returns:
        Encoded payload, exact sample count, duration, and payload digest.

    Raises:
        ValueError:
            If the input is not non-empty mono 16 kHz audio or fresh decoding fails.
    """
    if sampling_rate != 16000:
        raise ValueError("P1 audio must be sampled at 16000 Hz")
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("audio must be a non-empty mono array")
    output = io.BytesIO()
    sf.write(
        output,
        values.astype(np.float32),
        sampling_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    payload = output.getvalue()
    decoded = decode_flac(payload=payload)
    if decoded.size != values.size:
        raise ValueError("fresh FLAC decode changed sample count")
    return EncodedAudio(
        payload=payload,
        duration_ms=int(round(values.size * 1000 / sampling_rate)),
        sample_count=int(values.size),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def decode_flac(payload: bytes) -> np.ndarray:
    """Decode and validate a FLAC payload as fresh mono 16 kHz audio.

    Returns:
        Decoded mono float32 samples.

    Raises:
        ValueError:
            If the payload is not a valid mono 16 kHz FLAC stream.
    """
    try:
        with sf.SoundFile(io.BytesIO(payload)) as decoded_file:
            decoded = decoded_file.read(dtype="float32", always_2d=True)
            if decoded_file.samplerate != 16000 or decoded_file.channels != 1:
                raise ValueError("fresh FLAC decode is not mono 16 kHz")
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("payload is not a readable FLAC stream") from exc
    return decoded[:, 0]


def write_shards(
    rows: c.Iterable[OutputRow],
    output_dir: Path,
    target_bytes: int = 500_000_000,
    on_source_recoverable: c.Callable[[], None] | None = None,
) -> ShardBatchResult:
    """Atomically rotate bounded Parquet shards and fsync their evidence.

    The optional callback is invoked only after every shard has been renamed into
    place and its directory entry fsynced.  A caller can therefore delete source
    temporary bytes without coupling deletion to a later upload.

    Returns:
        Shard evidence and whether all shard bytes are locally recoverable.

    Raises:
        ValueError:
            If the target byte bound is not positive.
    """
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pending: list[OutputRow] = []
    results: list[ShardWriteResult] = []
    next_index = _next_shard_index(destination)
    for row in rows:
        pending.append(row)
        if _table_size(pending) >= target_bytes:
            results.append(_write_one_shard(pending, destination, next_index))
            next_index += 1
            pending = []
    if pending:
        results.append(_write_one_shard(pending, destination, next_index))
    recoverable = bool(results) and all(result.fsynced for result in results)
    if recoverable and on_source_recoverable is not None:
        on_source_recoverable()
    return ShardBatchResult(shards=tuple(results), source_recoverable=recoverable)


# These names describe the same operations in pipeline and test code.
parse_source_words = validate_raw_timestamps
validate_timestamps = validate_raw_timestamps
candidate_segments = form_candidate_segments
quality_gate = make_output_row


__all__ = [
    "AlignmentResult",
    "AlignmentText",
    "CandidateDecision",
    "CTCBackend",
    "CTCEmissionsAlignmentAdapter",
    "CTCSegmentationAdapter",
    "CTCAlignmentAdapter",
    "EncodedAudio",
    "SegmentationResult",
    "ShardBatchResult",
    "ShardWriter",
    "ShardWriteResult",
    "SourceValidationError",
    "VADBackend",
    "VADSignal",
    "align_ctc_emissions",
    "correct_drift_once",
    "decode_flac",
    "encode_flac",
    "form_candidate_segments",
    "candidate_segments",
    "make_output_row",
    "quality_gate",
    "normalise_alignment_text",
    "normalize_alignment_text",
    "parse_source_words",
    "segment_programme",
    "validate_raw_timestamps",
    "validate_source_words",
    "validate_timestamps",
    "write_shards",
]
