"""Typed, deterministic contracts for the P1 segmentation pipeline.

The models in this module describe metadata only.  They deliberately do not model
source audio or model weights, which keeps the Phase 1A contract safe to serialise
and suitable for the durable ledger.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_BLOB_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ContractModel(BaseModel):
    """Base class for immutable, strict pipeline contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OutputField(ContractModel):
    """One field in the published training schema."""

    name: StrictStr
    type: StrictStr


class OutputRow(ContractModel):
    """One accepted, publication-ready P1 training row."""

    audio: bytes
    audio_sha256: StrictStr
    text: StrictStr
    alignment_text: StrictStr
    alignment_word_map: tuple[StrictStr, ...]
    language: StrictStr = "da"
    segment_id: StrictStr
    source_file_id: StrictStr
    source_start_ms: StrictInt = Field(ge=0)
    source_end_ms: StrictInt = Field(gt=0)
    duration_ms: StrictInt = Field(gt=0)
    speaker_ids: tuple[StrictStr, ...]
    proposal_start_ms: StrictInt = Field(ge=0)
    proposal_end_ms: StrictInt = Field(gt=0)
    alignment_score: StrictFloat
    alignment_score_type: StrictStr
    start_drift_ms: StrictInt
    end_drift_ms: StrictInt
    vad_speech_ratio: StrictFloat = Field(ge=0.0, le=1.0)
    alignment_backend: StrictStr
    pipeline_version: StrictStr
    pipeline_config_sha256: StrictStr

    @field_validator("audio_sha256", "pipeline_config_sha256", "segment_id")
    def _digest_is_hex(value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("digest fields must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def _row_is_consistent(self) -> OutputRow:
        if self.language != "da":
            raise ValueError("P1 output language must be da")
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError("source boundaries must be ordered")
        if self.duration_ms != self.source_end_ms - self.source_start_ms:
            raise ValueError("duration_ms must equal the source interval")
        if self.proposal_end_ms <= self.proposal_start_ms:
            raise ValueError("proposal boundaries must be ordered")
        if not self.text.strip() or not self.alignment_text.strip():
            raise ValueError("published and alignment text must not be empty")
        if not math.isfinite(self.alignment_score):
            raise ValueError("alignment_score must be finite")
        if not self.audio:
            raise ValueError("audio must contain an encoded FLAC payload")
        return self


class OutputSchema(ContractModel):
    """The exact published P1 training-row schema."""

    schema_version: StrictStr
    fields: tuple[OutputField, ...]

    @model_validator(mode="after")
    def _fields_are_unique(self) -> OutputSchema:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("output schema field names must be unique")
        return self


class RepositoryRevision(ContractModel):
    """An immutable repository coordinate."""

    repository: StrictStr
    revision: StrictStr

    @field_validator("repository")
    def _repository_is_complete(value: str) -> str:
        if not value or value.startswith("/") or "@" in value:
            raise ValueError("repository must be a non-empty Hub-style identifier")
        return value

    @field_validator("revision")
    def _revision_is_immutable(value: str) -> str:
        if not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("revision must be a complete 40-character commit SHA")
        return value


class SegmentProposal(ContractModel):
    """A candidate segment formed from consecutive source words."""

    source_file_id: StrictStr
    word_start_index: StrictInt = Field(ge=0)
    word_end_index: StrictInt = Field(gt=0)
    proposal_start_ms: StrictInt = Field(ge=0)
    proposal_end_ms: StrictInt = Field(gt=0)
    text: StrictStr
    speaker_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def _proposal_is_complete(self) -> SegmentProposal:
        if self.word_end_index <= self.word_start_index:
            raise ValueError("proposal word indexes must form a non-empty range")
        if self.proposal_end_ms <= self.proposal_start_ms:
            raise ValueError("proposal timestamps must be ordered")
        if not self.text.strip():
            raise ValueError("proposal text must not be empty")
        return self


class SourceCoordinates(ContractModel):
    """Pinned source dataset coordinates and join columns."""

    audio: RepositoryRevision
    transcripts: RepositoryRevision
    join_key: StrictStr = "file_id"
    text_column: StrictStr = "transcript_text"

    @model_validator(mode="after")
    def _columns_are_p1_columns(self) -> SourceCoordinates:
        if self.join_key != "file_id":
            raise ValueError("P1 sources must be joined on file_id")
        if self.text_column != "transcript_text":
            raise ValueError("P1 transcripts must use transcript_text")
        return self


class SourceTextSpan(ContractModel):
    """A half-open character span in the verbatim programme transcript."""

    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> SourceTextSpan:
        if self.end <= self.start:
            raise ValueError("source text span end must be greater than start")
        return self


class SourceWord(ContractModel):
    """A transcript word with timing and auditable source-text spans."""

    text: StrictStr
    start_ms: StrictInt = Field(ge=0)
    end_ms: StrictInt = Field(gt=0)
    speaker_id: StrictStr | None = None
    source_span: SourceTextSpan | None = None
    separator_span: SourceTextSpan | None = None
    separator_text: StrictStr = ""

    @model_validator(mode="after")
    def _span_is_ordered(self) -> SourceWord:
        if self.end_ms <= self.start_ms:
            raise ValueError("word end_ms must be greater than start_ms")
        if not self.text.strip():
            raise ValueError("word text must not be empty")
        if self.source_span is not None and (
            self.source_span.end - self.source_span.start != len(self.text)
        ):
            raise ValueError("source character span does not match word text")
        if self.separator_span is None and self.separator_text:
            raise ValueError("separator text requires a separator span")
        if self.separator_span is not None and (
            self.separator_span.end - self.separator_span.start
            != len(self.separator_text)
        ):
            raise ValueError("separator span does not match separator text")
        return self


class SourceProgramme(ContractModel):
    """A joined P1 programme and its ordered words and separator spans."""

    file_id: StrictStr
    duration_ms: StrictInt = Field(gt=0)
    words: tuple[SourceWord, ...]
    transcript_text: StrictStr | None = None
    separator_spans: tuple[SourceTextSpan, ...] = ()

    @field_validator("file_id")
    def _file_id_is_present(value: str) -> str:
        if not value.strip():
            raise ValueError("file_id must not be empty")
        return value

    @model_validator(mode="after")
    def _words_fit_duration(self) -> SourceProgramme:
        previous_end = 0
        for word in self.words:
            if word.start_ms < previous_end:
                raise ValueError("source words must be ordered and non-overlapping")
            if word.end_ms > self.duration_ms:
                raise ValueError("word span must fit programme duration")
            previous_end = word.end_ms
        if self.transcript_text is not None and self.words:
            annotated = annotate_source_words(self.words, self.transcript_text)
            object.__setattr__(self, "words", annotated)
            object.__setattr__(
                self,
                "separator_spans",
                tuple(
                    word.separator_span
                    for word in annotated
                    if word.separator_span is not None
                ),
            )
        return self


def annotate_source_words(
    words: tuple[SourceWord, ...] | list[SourceWord], transcript_text: str
) -> tuple[SourceWord, ...]:
    """Attach exact character and separator spans to transcript words.

    The source text is treated as authoritative.  A word that cannot be found in
    order is rejected rather than silently normalised or re-spaced.

    Returns:
        Words carrying their source and separator spans.

    Raises:
        ValueError:
            If a word is absent or out of order in the transcript.
    """
    cursor = 0
    annotated: list[SourceWord] = []
    for index, word in enumerate(words):
        start = transcript_text.find(word.text, cursor)
        if start < 0:
            raise ValueError(f"word {index} is not present in transcript text")
        source_span = SourceTextSpan(start=start, end=start + len(word.text))
        separator_span = (
            SourceTextSpan(start=cursor, end=start) if start > cursor else None
        )
        annotated.append(
            word.model_copy(
                update={
                    "source_span": source_span,
                    "separator_span": separator_span,
                    "separator_text": transcript_text[cursor:start],
                }
            )
        )
        cursor = start + len(word.text)
    return tuple(annotated)


OUTPUT_SCHEMA = OutputSchema(
    schema_version="p1-segments-v1",
    fields=(
        OutputField(name="audio", type="Audio(16000)"),
        OutputField(name="audio_sha256", type="string"),
        OutputField(name="text", type="string"),
        OutputField(name="alignment_text", type="string"),
        OutputField(name="alignment_word_map", type="list[string]"),
        OutputField(name="language", type="string"),
        OutputField(name="segment_id", type="string"),
        OutputField(name="source_file_id", type="string"),
        OutputField(name="source_start_ms", type="int64"),
        OutputField(name="source_end_ms", type="int64"),
        OutputField(name="duration_ms", type="int32"),
        OutputField(name="speaker_ids", type="list[string]"),
        OutputField(name="proposal_start_ms", type="int64"),
        OutputField(name="proposal_end_ms", type="int64"),
        OutputField(name="alignment_score", type="float32"),
        OutputField(name="alignment_score_type", type="string"),
        OutputField(name="start_drift_ms", type="int32"),
        OutputField(name="end_drift_ms", type="int32"),
        OutputField(name="vad_speech_ratio", type="float32"),
        OutputField(name="alignment_backend", type="string"),
        OutputField(name="pipeline_version", type="string"),
        OutputField(name="pipeline_config_sha256", type="string"),
    ),
)


class LedgerState(str, enum.Enum):
    """Durable programme or shard states."""

    DISCOVERED = "discovered"
    PROCESSING = "processing"
    SHARDED = "sharded"
    COMMITTED = "committed"
    VERIFIED = "verified"
    PURGED = "purged"
    REJECTED = "rejected"
    RETRYABLE = "retryable"


class RejectionCategory(str, enum.Enum):
    """Stable categories for rejected programmes and proposals."""

    EMPTY_TEXT = "empty_text"
    DURATION_OUT_OF_RANGE = "duration_out_of_range"
    INVALID_TIMESTAMPS = "invalid_timestamps"
    LOW_ALIGNMENT_SCORE = "low_alignment_score"
    EXCESSIVE_DRIFT = "excessive_drift"
    LOW_SPEECH_RATIO = "low_speech_ratio"
    SPEAKER_OVERLAP = "speaker_overlap"
    BOUNDARY_CLIPPING = "boundary_clipping"
    DECODE_ERROR = "decode_error"
    MISSING_AUDIO = "missing_audio"
    DUPLICATE_SEGMENT_ID = "duplicate_segment_id"
    UNSUPPORTED_TEXT = "unsupported_text"
    MUSIC_DOMINANT = "music_dominant"
    NO_ACCEPTED_SEGMENTS = "no_accepted_segments"


_ALLOWED_TRANSITIONS: dict[LedgerState, frozenset[LedgerState]] = {
    LedgerState.DISCOVERED: frozenset({LedgerState.PROCESSING, LedgerState.REJECTED}),
    LedgerState.PROCESSING: frozenset(
        {LedgerState.SHARDED, LedgerState.RETRYABLE, LedgerState.REJECTED}
    ),
    LedgerState.SHARDED: frozenset({LedgerState.COMMITTED, LedgerState.RETRYABLE}),
    LedgerState.COMMITTED: frozenset({LedgerState.VERIFIED, LedgerState.RETRYABLE}),
    LedgerState.VERIFIED: frozenset({LedgerState.PURGED}),
    LedgerState.PURGED: frozenset(),
    LedgerState.REJECTED: frozenset(),
    LedgerState.RETRYABLE: frozenset({LedgerState.PROCESSING, LedgerState.REJECTED}),
}


class ModelContract(ContractModel):
    """Pinned model and its declared licence."""

    repository: RepositoryRevision
    license: StrictStr


class CTCContract(ContractModel):
    """Pinned CTC library, source distribution, and model."""

    name: StrictStr = "ctc-segmentation"
    version: StrictStr = "1.7.4"
    source_commit: StrictStr
    sdist_sha256: StrictStr
    license: StrictStr
    model: ModelContract

    @field_validator("sdist_sha256")
    def _sdist_digest_is_immutable(value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sdist_sha256 must be lowercase SHA-256 hex")
        return value

    @field_validator("source_commit")
    def _source_commit_is_immutable(value: str) -> str:
        if not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("source_commit must be a complete SHA")
        return value


class NormalisationContract(ContractModel):
    """Versioned text normalisation rules used by the aligner."""

    version: StrictStr
    unicode_form: StrictStr = "NFC"
    case_folding: bool = False
    punctuation_removed: bool = True
    number_expansion: bool = False
    preserves_source_word_map: bool = True


class OutputEncodingContract(ContractModel):
    """Audio and shard encoding identity."""

    audio_format: StrictStr = "flac"
    sampling_rate: StrictInt = Field(default=16000, gt=0)
    channels: StrictInt = Field(default=1, gt=0)
    shard_format: StrictStr = "parquet"
    schema: OutputSchema = OUTPUT_SCHEMA


class SegmentationContract(ContractModel):
    """Versioned segmentation and quality thresholds."""

    minimum_duration_ms: StrictInt = Field(gt=0)
    target_minimum_duration_ms: StrictInt = Field(gt=0)
    target_maximum_duration_ms: StrictInt = Field(gt=0)
    maximum_duration_ms: StrictInt = Field(gt=0)
    maximum_drift_ms: StrictInt = Field(ge=0)
    minimum_alignment_score: StrictFloat
    minimum_vad_speech_ratio: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> SegmentationContract:
        if not (
            self.minimum_duration_ms
            <= self.target_minimum_duration_ms
            <= self.target_maximum_duration_ms
            < self.maximum_duration_ms
        ):
            raise ValueError(
                "duration thresholds must be ordered and below the maximum"
            )
        if not math.isfinite(self.minimum_alignment_score):
            raise ValueError("minimum_alignment_score must be finite")
        return self


class ShardEvidence(ContractModel):
    """Metadata needed to verify one local or remote Parquet shard."""

    path: StrictStr
    byte_size: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(ge=0)
    sha256: StrictStr

    @field_validator("sha256")
    def _sha256_is_hex(value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be lowercase SHA-256 hex")
        return value


class BatchEvidence(ContractModel):
    """Metadata-only evidence for one bounded upload batch."""

    batch_id: StrictStr
    state: LedgerState
    shards: tuple[ShardEvidence, ...]
    commit_id: StrictStr | None = None
    programme_count: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(ge=0)
    rejection_counts: dict[RejectionCategory, StrictInt] = Field(default_factory=dict)

    @field_validator("commit_id")
    def _commit_id_is_immutable(value: str | None) -> str | None:
        if value is not None and not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("commit_id must be a complete 40-character commit SHA")
        return value

    @model_validator(mode="after")
    def _commit_matches_state(self) -> BatchEvidence:
        if (
            self.state
            in {LedgerState.COMMITTED, LedgerState.VERIFIED, LedgerState.PURGED}
            and not self.commit_id
        ):
            raise ValueError("committed batches require an immutable commit_id")
        if self.state in {LedgerState.VERIFIED, LedgerState.PURGED} and not self.shards:
            raise ValueError("verified batches require shard evidence")
        return self


class VADContract(ContractModel):
    """Pinned VAD implementation and model blob."""

    name: StrictStr
    repository: RepositoryRevision
    model_blob: StrictStr
    license: StrictStr

    @field_validator("model_blob")
    def _model_blob_is_immutable(value: str) -> str:
        if not _BLOB_PATTERN.fullmatch(value):
            raise ValueError("model_blob must be a complete hexadecimal blob ID")
        return value


class CanonicalIdentityManifest(ContractModel):
    """Complete identity manifest used to derive the pipeline digest."""

    schema_version: StrictStr
    pipeline_version: StrictStr
    source: SourceCoordinates
    vad: VADContract
    ctc: CTCContract
    anomaly_model: ModelContract
    normalisation: NormalisationContract
    segmentation: SegmentationContract
    output: OutputEncodingContract


def canonical_json(value: object) -> str:
    """Return deterministic canonical JSON text for a contract or JSON value."""
    return canonical_json_bytes(value).decode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON for a contract or JSON-compatible value.

    Strings are recursively NFC-normalised before serialisation.  Non-finite
    floats, non-string mapping keys, bytes, and incomplete contract values are
    rejected instead of being coerced into a different identity.
    """
    normalised = _canonical_value(value)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_value(value: object) -> JSONValue:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, enum.Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("canonical JSON cannot contain non-finite floats")
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value)
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalised_key = unicodedata.normalize("NFC", key)
            if normalised_key in result:
                raise ValueError("NFC-normalised object keys must be unique")
            result[normalised_key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def pipeline_config_sha256(manifest: CanonicalIdentityManifest) -> str:
    """Return the identity digest for a complete pipeline manifest."""
    return sha256_digest(manifest)


def sha256_digest(value: object) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def segment_id(
    pipeline_config_sha256: str,
    source_file_id: str,
    source_start_ms: int,
    source_end_ms: int,
    text: str,
) -> str:
    """Return the deterministic identity of one published segment.

    Raises:
        ValueError: If the digest, boundaries, or required text is invalid.
    """
    if not _SHA256_PATTERN.fullmatch(pipeline_config_sha256):
        raise ValueError("pipeline_config_sha256 must be lowercase SHA-256 hex")
    if (
        isinstance(source_start_ms, bool)
        or isinstance(source_end_ms, bool)
        or not isinstance(source_start_ms, int)
        or not isinstance(source_end_ms, int)
        or source_start_ms < 0
        or source_end_ms <= source_start_ms
    ):
        raise ValueError("segment boundaries must be ordered non-negative integers")
    if not source_file_id or not text:
        raise ValueError("source_file_id and text must not be empty")
    return sha256_digest(
        {
            "pipeline_config_sha256": pipeline_config_sha256,
            "source_file_id": source_file_id,
            "source_start_ms": source_start_ms,
            "source_end_ms": source_end_ms,
            "text": text,
        }
    )


def valid_ledger_transition(current: LedgerState, target: LedgerState) -> bool:
    """Return whether a ledger state transition is permitted."""
    return target in _ALLOWED_TRANSITIONS[current]
