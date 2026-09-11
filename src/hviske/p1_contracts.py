"""Typed, deterministic contracts for the P1 segmentation pipeline.

The models in this module describe metadata only.  They deliberately do not model
source audio or model weights, which keeps the Phase 1A contract safe to serialise
and suitable for the durable ledger.
"""

from __future__ import annotations

import enum
import hashlib
import io
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import soundfile as sf
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
_IMMUTABLE_HUB_URL_PATTERN = re.compile(
    r"^https://huggingface\.co/(?:datasets/)?[^/]+/[^/]+/resolve/"
    r"[0-9a-f]{40}/.+$"
)

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class P1RuntimeContract:
    """The immutable implementation and data-processing contract for P1."""

    pipeline_version: str
    alignment_method: str
    ctc_name: str
    ctc_version: str
    ctc_source_commit: str
    ctc_sdist_sha256: str
    ctc_license: str
    roest_repository: str
    roest_revision: str
    roest_license: str
    roest_license_url: str
    roest_license_repository: str
    roest_license_revision: str
    roest_license_sha256: str
    roest_model_card_url: str
    roest_model_card_sha256: str
    roest_license_meaning: str
    roest_architecture: str
    roest_model_type: str
    roest_sampling_rate: int
    roest_frame_stride_samples: int
    roest_frame_duration_ms: float
    roest_vocab_size: int
    roest_blank_token_id: int
    roest_word_delimiter_token_id: int
    roest_required_tokens: frozenset[str]
    roest_tokenizer_case: str
    normalisation_version: str
    normalisation_source_text_ownership: str
    normalisation_unicode_form: str
    normalisation_case_folding: bool
    normalisation_punctuation_removed: bool
    normalisation_number_expansion: bool
    normalisation_preserves_source_word_map: bool
    dataset_license_template_repository: str
    dataset_license_template_revision: str
    dataset_license_template_url: str
    dataset_license_template_sha256: str
    dataset_license_template_bytes: int
    dataset_license_adaptation: str
    dataset_license_target_path: str
    dataset_license_target_sha256: str


P1_RUNTIME_CONTRACT = P1RuntimeContract(
    pipeline_version="p1-segmentation-8",
    alignment_method="timestamp-native:p1-transcripts.words",
    ctc_name="ctc-segmentation",
    ctc_version="1.7.4",
    ctc_source_commit="69bd9b53b7b82ad926d35e7b280f957ed299a7db",
    ctc_sdist_sha256=(
        "19d383ea5f22438ebb1699d72b22078b63f351a33fa50bedb19c14077ba6a116"
    ),
    ctc_license="Apache-2.0",
    roest_repository="CoRal-project/roest-v3-wav2vec2-315m",
    roest_revision="beb3e790246d6b9dec1df596b0b21d5c42f4d99c",
    roest_license="openrail",
    roest_license_url=(
        "https://huggingface.co/Alvenir/coral-1-whisper-large/resolve/"
        "a6c1e24d9f10e6289607a1ba32341b68e8660688/LICENSE"
    ),
    roest_license_repository="Alvenir/coral-1-whisper-large",
    roest_license_revision="a6c1e24d9f10e6289607a1ba32341b68e8660688",
    roest_license_sha256=(
        "f575b6361ff69b52388967f69219f0cc7f9ae91f96482e5ad038261b2728799e"
    ),
    roest_model_card_url=(
        "https://huggingface.co/CoRal-project/roest-v3-wav2vec2-315m/resolve/"
        "beb3e790246d6b9dec1df596b0b21d5c42f4d99c/README.md"
    ),
    roest_model_card_sha256=(
        "64b3a837fdcb580eeebe31d457113f0a84b200ca90ac5fe1f27475d8fc257cfb"
    ),
    roest_license_meaning=(
        "Roest model-card metadata is openrail; its pinned card describes a custom "
        "OpenRAIL-M licence. P1 uses the checkpoint for ASR alignment only. Model "
        "weights are internal and are not distributed by this dataset."
    ),
    roest_architecture="Wav2Vec2ForCTC",
    roest_model_type="wav2vec2",
    roest_sampling_rate=16_000,
    roest_frame_stride_samples=320,
    roest_frame_duration_ms=20.0,
    roest_vocab_size=46,
    roest_blank_token_id=45,
    roest_word_delimiter_token_id=36,
    roest_required_tokens=frozenset("0123456789abcdefghijklmnopqrstuvwxyzåæéøü"),
    roest_tokenizer_case="lowercase-only",
    normalisation_version="p1-text-normalisation-6",
    normalisation_source_text_ownership=(
        "best-effort-following-word-with-terminal-suffix-v6"
    ),
    normalisation_unicode_form="NFC",
    normalisation_case_folding=True,
    normalisation_punctuation_removed=True,
    normalisation_number_expansion=False,
    normalisation_preserves_source_word_map=True,
    dataset_license_template_repository="CoRal-project/coral-v3",
    dataset_license_template_revision=("01f7c93c21fc9dec87fe9f7149c79569cc433f08"),
    dataset_license_template_url=(
        "https://huggingface.co/datasets/CoRal-project/coral-v3/resolve/"
        "01f7c93c21fc9dec87fe9f7149c79569cc433f08/LICENSE"
    ),
    dataset_license_template_sha256=(
        "ee93c98df9a894464d1c042b66c6039543776c463d06d1a6e93f176e08ca67bd"
    ),
    dataset_license_template_bytes=14112,
    dataset_license_adaptation=(
        "Replace only the exact UTF-8 byte/text sequence 'The Licensed Material "
        "(as defined below) is made available to You by Alexandra\nInstituttet A/S, "
        "Åbogade 34, 8200 Aarhus N, Denmark' with 'The Licensed Material (as "
        "defined below) is made available to You by syv.ai ApS,\nRosenvængets Allé "
        "11, 1. tv, 2100 København Ø, Denmark'; preserve all other bytes."
    ),
    dataset_license_target_path="LICENSE",
    dataset_license_target_sha256=(
        "e06010caf8ea36292a241339c08eedc7ea399778954cf9a546e989d421a48cd6"
    ),
)


class ContractModel(BaseModel):
    """Base class for immutable, strict pipeline contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetLicenseContract(ContractModel):
    """Immutable source and adaptation identity for the target data licence."""

    template_repository: StrictStr
    template_revision: StrictStr
    template_url: StrictStr
    template_sha256: StrictStr
    template_bytes: StrictInt = Field(gt=0)
    adaptation: StrictStr
    target_path: StrictStr
    target_sha256: StrictStr

    @field_validator("template_sha256", "target_sha256")
    @classmethod
    def _digest_is_immutable(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("licence digest must be lowercase SHA-256 hex")
        return value

    @field_validator("template_revision")
    @classmethod
    def _revision_is_immutable(cls, value: str) -> str:
        if not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("licence template revision must be a complete SHA")
        return value

    @field_validator("template_url")
    @classmethod
    def _template_url_is_immutable(cls, value: str) -> str:
        return _validate_immutable_hub_url(value=value)


class OutputField(ContractModel):
    """One field in the published training schema."""

    name: StrictStr
    type: StrictStr
    nullable: bool = False


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
    source_duration_ms: StrictInt = Field(gt=0)
    duration_ms: StrictInt = Field(gt=0)
    speaker_ids: tuple[StrictStr, ...]
    proposal_start_ms: StrictInt = Field(ge=0)
    proposal_end_ms: StrictInt = Field(gt=0)
    alignment_score: StrictFloat | None = None
    alignment_score_type: StrictStr = "not_applicable"
    start_drift_ms: StrictInt | None = None
    end_drift_ms: StrictInt | None = None
    vad_speech_ratio: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    alignment_backend: StrictStr = "ctc-segmentation"
    pipeline_version: StrictStr
    pipeline_config_sha256: StrictStr
    alignment_method: StrictStr = "ctc-segmentation"

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
        if self.source_end_ms > self.source_duration_ms:
            raise ValueError("source boundaries must fit programme duration")
        if self.duration_ms != self.source_end_ms - self.source_start_ms:
            raise ValueError("duration_ms must equal the source interval")
        if self.proposal_end_ms <= self.proposal_start_ms:
            raise ValueError("proposal boundaries must be ordered")
        if not self.text.strip() or not self.alignment_text.strip():
            raise ValueError("published and alignment text must not be empty")
        if self.alignment_score is not None and not math.isfinite(self.alignment_score):
            raise ValueError("alignment_score must be finite when supplied")
        if self.pipeline_version == "p1-segmentation-7":
            raise ValueError("v7 rows cannot be mixed with the active v8 contract")
        if self.pipeline_version == "p1-segmentation-8":
            if self.alignment_method != "timestamp-native:p1-transcripts.words":
                raise ValueError("v8 rows must use timestamp-native alignment")
            if self.alignment_backend != "timestamp-native":
                raise ValueError("v8 rows must use the timestamp-native backend")
            if self.alignment_score_type != "not_applicable:source_timestamps":
                raise ValueError("v8 rows must have a not-applicable score type")
            if (
                self.source_start_ms != self.proposal_start_ms
                or self.source_end_ms != self.proposal_end_ms
                or self.alignment_score is not None
                or self.start_drift_ms is not None
                or self.end_drift_ms is not None
                or self.vad_speech_ratio is not None
            ):
                raise ValueError(
                    "timestamp-native rows cannot contain acoustic evidence"
                )
        if not self.audio:
            raise ValueError("audio must contain an encoded OGG/Opus payload")
        if self.pipeline_version == "p1-segmentation-8":
            try:
                with sf.SoundFile(io.BytesIO(self.audio)) as audio_file:
                    decoded = audio_file.read(dtype="float32", always_2d=True)
                    valid_audio = (
                        audio_file.format == "OGG"
                        and audio_file.subtype == "OPUS"
                        and audio_file.samplerate == 16_000
                        and audio_file.channels == 1
                        and decoded.shape[0] == self.duration_ms * 16
                        and decoded.shape[0] > 0
                        and all(
                            math.isfinite(float(value)) for value in decoded.ravel()
                        )
                    )
            except Exception as error:
                raise ValueError("v8 audio must be readable OGG/Opus") from error
            if not valid_audio:
                raise ValueError(
                    "v8 audio must be non-empty finite OGG/Opus mono 16 kHz "
                    "with the expected sample count"
                )
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
    """A timed word with auditable ownership of adjacent source characters.

    ``separator_text`` belongs to this word, including any leading or inter-word
    untimed records. ``trailing_text`` belongs to the final timed word.
    """

    text: StrictStr
    start_ms: StrictInt = Field(ge=0)
    end_ms: StrictInt = Field(gt=0)
    speaker_id: StrictStr | None = None
    source_span: SourceTextSpan | None = None
    separator_span: SourceTextSpan | None = None
    separator_text: StrictStr = ""
    trailing_span: SourceTextSpan | None = None
    trailing_text: StrictStr = ""

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
        if self.trailing_span is None and self.trailing_text:
            raise ValueError("trailing text requires a trailing span")
        if self.trailing_span is not None and (
            self.trailing_span.end - self.trailing_span.start != len(self.trailing_text)
        ):
            raise ValueError("trailing span does not match trailing text")
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
            has_source_spans = any(word.source_span is not None for word in self.words)
            if has_source_spans and not all(
                word.source_span is not None for word in self.words
            ):
                raise ValueError(
                    "source character spans must be present for every source word"
                )
            expected_starts = (
                tuple(
                    word.source_span.start
                    for word in self.words
                    if word.source_span is not None
                )
                if has_source_spans
                else None
            )
            annotated = annotate_source_words(
                self.words, self.transcript_text, expected_starts=expected_starts
            )
            object.__setattr__(self, "words", annotated)
            object.__setattr__(
                self,
                "separator_spans",
                tuple(
                    span
                    for word in annotated
                    for span in (word.separator_span, word.trailing_span)
                    if span is not None
                ),
            )
        return self


def annotate_source_words(
    words: tuple[SourceWord, ...] | list[SourceWord],
    transcript_text: str,
    expected_starts: tuple[int, ...] | list[int] | None = None,
) -> tuple[SourceWord, ...]:
    """Attach exact character and separator spans to transcript words.

    The source text is treated as authoritative. A word that cannot be found in
    order is rejected rather than silently normalised or re-spaced. Expected
    offsets identify timed words exactly, even when an owned untimed token has the
    same text.

    Args:
        words:
            Timed transcript words in source order.
        transcript_text:
            The authoritative verbatim transcript.
        expected_starts (optional):
            Record offsets used to reject ambiguous duplicate lexical matches.

    Returns:
        Words carrying their source and separator spans.

    Raises:
        ValueError:
            If a word is absent or out of order in the transcript.
    """
    cursor = 0
    annotated: list[SourceWord] = []
    if expected_starts is not None and len(expected_starts) != len(words):
        raise ValueError("expected source offsets do not match timed words")
    for index, word in enumerate(words):
        if expected_starts is None:
            start = transcript_text.find(word.text, cursor)
        else:
            start = expected_starts[index]
        if (
            start < cursor
            or transcript_text[start : start + len(word.text)] != word.text
        ):
            raise ValueError(f"word {index} is not present at its source offset")
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
    if annotated and cursor < len(transcript_text):
        trailing_span = SourceTextSpan(start=cursor, end=len(transcript_text))
        annotated[-1] = annotated[-1].model_copy(
            update={
                "trailing_span": trailing_span,
                "trailing_text": transcript_text[cursor:],
            }
        )
    return tuple(annotated)


OUTPUT_SCHEMA = OutputSchema(
    schema_version="p1-segments-v2",
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
        OutputField(name="source_duration_ms", type="int64"),
        OutputField(name="duration_ms", type="int32"),
        OutputField(name="speaker_ids", type="list[string]"),
        OutputField(name="proposal_start_ms", type="int64"),
        OutputField(name="proposal_end_ms", type="int64"),
        OutputField(name="alignment_score", type="float32", nullable=True),
        OutputField(name="alignment_score_type", type="string"),
        OutputField(name="start_drift_ms", type="int32", nullable=True),
        OutputField(name="end_drift_ms", type="int32", nullable=True),
        OutputField(name="vad_speech_ratio", type="float32", nullable=True),
        OutputField(name="alignment_backend", type="string"),
        OutputField(name="alignment_method", type="string"),
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
    CTC_ALIGNMENT_FAILED = "ctc_alignment_failed"
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
    INVALID_SOURCE_RECORD = "invalid_source_record"
    TRANSCRIPT_OVER_AUDIO = "transcript_over_audio"
    AMBIGUOUS_SOURCE_TEXT = "ambiguous_source_text"
    NO_TIMED_WORDS = "no_timed_words"
    NO_ACCEPTED_SEGMENTS = "no_accepted_segments"


_ALLOWED_TRANSITIONS: dict[LedgerState, frozenset[LedgerState]] = {
    LedgerState.DISCOVERED: frozenset({LedgerState.PROCESSING, LedgerState.REJECTED}),
    LedgerState.PROCESSING: frozenset(
        {LedgerState.SHARDED, LedgerState.RETRYABLE, LedgerState.REJECTED}
    ),
    # Once shard bytes are durable, recovery must use those bytes rather than
    # re-entering processing. A committed batch likewise has an immutable Hub
    # coordinate and can only advance through verification.
    LedgerState.SHARDED: frozenset({LedgerState.COMMITTED}),
    LedgerState.COMMITTED: frozenset({LedgerState.VERIFIED}),
    LedgerState.VERIFIED: frozenset({LedgerState.PURGED}),
    LedgerState.PURGED: frozenset(),
    LedgerState.REJECTED: frozenset(),
    LedgerState.RETRYABLE: frozenset({LedgerState.PROCESSING, LedgerState.REJECTED}),
}


class ModelContract(ContractModel):
    """Pinned model, declared licence, and optional architecture evidence."""

    repository: RepositoryRevision
    license: StrictStr
    license_url: StrictStr | None = None
    license_repository: StrictStr | None = None
    license_revision: StrictStr | None = None
    license_sha256: StrictStr | None = None
    model_card_url: StrictStr | None = None
    model_card_sha256: StrictStr | None = None
    license_notes: StrictStr | None = None
    architecture: StrictStr | None = None
    model_type: StrictStr | None = None
    sampling_rate: StrictInt | None = None
    frame_stride_samples: StrictInt | None = None
    vocab_size: StrictInt | None = None
    blank_token_id: StrictInt | None = None
    word_delimiter_token_id: StrictInt | None = None

    @field_validator("license_sha256", "model_card_sha256")
    @classmethod
    def _optional_digest_is_immutable(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("licence metadata digest must be lowercase SHA-256 hex")
        return value

    @field_validator("license_url", "model_card_url")
    @classmethod
    def _optional_provenance_url_is_immutable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_immutable_hub_url(value=value)

    @field_validator("license_revision")
    @classmethod
    def _optional_revision_is_immutable(cls, value: str | None) -> str | None:
        if value is not None and not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("licence metadata revision must be a complete SHA")
        return value


class CTCContract(ContractModel):
    """Pinned CTC library, source distribution, and model."""

    name: StrictStr = "ctc-segmentation"
    version: StrictStr = "1.7.4"
    source_commit: StrictStr
    sdist_sha256: StrictStr
    license: StrictStr
    model: ModelContract

    @field_validator("sdist_sha256")
    @classmethod
    def _sdist_digest_is_immutable(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sdist_sha256 must be lowercase SHA-256 hex")
        return value

    @field_validator("source_commit")
    def _source_commit_is_immutable(value: str) -> str:
        if not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("source_commit must be a complete SHA")
        return value


def _validate_immutable_hub_url(*, value: str) -> str:
    """Reject mutable Hugging Face provenance URLs.

    Args:
        value:
            URL to validate.

    Returns:
        The immutable URL.

    Raises:
        ValueError:
            If the URL is not a commit-pinned Hugging Face resolve URL.
    """
    if not _IMMUTABLE_HUB_URL_PATTERN.fullmatch(value):
        raise ValueError("licence URL must use a complete SHA in a Hub resolve URL")
    return value


class NormalisationContract(ContractModel):
    """Versioned text normalisation rules used by the aligner."""

    version: StrictStr
    source_text_ownership: StrictStr = (
        "best-effort-following-word-with-terminal-suffix-v6"
    )
    unicode_form: StrictStr = "NFC"
    case_folding: bool = False
    punctuation_removed: bool = True
    number_expansion: bool = False
    preserves_source_word_map: bool = True


class OutputEncodingContract(ContractModel):
    """Audio and shard encoding identity."""

    audio_format: StrictStr = "ogg-opus"
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


def _default_dataset_license() -> DatasetLicenseContract:
    """Return the repository's immutable target dataset licence identity."""
    contract = P1_RUNTIME_CONTRACT
    return DatasetLicenseContract(
        template_repository=contract.dataset_license_template_repository,
        template_revision=contract.dataset_license_template_revision,
        template_url=contract.dataset_license_template_url,
        template_sha256=contract.dataset_license_template_sha256,
        template_bytes=contract.dataset_license_template_bytes,
        adaptation=contract.dataset_license_adaptation,
        target_path=contract.dataset_license_target_path,
        target_sha256=contract.dataset_license_target_sha256,
    )


class CanonicalIdentityManifest(ContractModel):
    """Complete identity manifest used to derive the pipeline digest."""

    schema_version: StrictStr
    pipeline_version: StrictStr
    alignment_method: StrictStr = "timestamp-native:p1-transcripts.words"
    source: SourceCoordinates
    normalisation: NormalisationContract
    segmentation: SegmentationContract
    output: OutputEncodingContract
    vad: VADContract | None = None
    ctc: CTCContract | None = None
    anomaly_model: ModelContract | None = None
    dataset_license: DatasetLicenseContract = Field(
        default_factory=_default_dataset_license
    )
    max_decoded_audio_bytes: StrictInt = Field(default=2 * 1024**3, gt=0)


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
    """Return the digest for the active pipeline identity.

    The v8 pipeline consumes source timestamps and therefore has no model-backed
    alignment identity.  Keep the legacy model fields on the manifest so the
    generic future aligner remains representable, but do not let them affect v8
    segment identities.
    """
    if manifest.pipeline_version == P1_RUNTIME_CONTRACT.pipeline_version:
        value = manifest.model_dump(
            mode="json",
            include={
                "schema_version",
                "pipeline_version",
                "alignment_method",
                "source",
                "normalisation",
                "segmentation",
                "output",
                "dataset_license",
                "max_decoded_audio_bytes",
            },
        )
        return sha256_digest(value)
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


def validate_p1_runtime_contract(
    *,
    pipeline_version: str,
    ctc: CTCContract | None,
    alignment_method: str | None = None,
    normalisation: NormalisationContract,
    dataset_license: DatasetLicenseContract | None = None,
) -> None:
    """Reject configuration that diverges from the active P1 contract.

    Args:
        pipeline_version:
            Version of the P1 pipeline implementation.
        ctc:
            CTC contract for a future model-backed alignment. It is not required by
            timestamp-native v8.
        alignment_method (optional):
            Active alignment method identity.
        normalisation:
            Text normalisation rules used before alignment.
        dataset_license (optional):
            Immutable target dataset licence provenance.

    Raises:
        ValueError:
            If any identity field differs from the active contract.
    """
    contract = P1_RUNTIME_CONTRACT
    if alignment_method is not None and alignment_method != contract.alignment_method:
        raise ValueError(
            "P1 alignment method must be timestamp-native:p1-transcripts.words"
        )
    if pipeline_version == contract.pipeline_version:
        values: dict[str, tuple[object, object]] = {
            "normalisation.version": (
                normalisation.version,
                contract.normalisation_version,
            ),
            "normalisation.source_text_ownership": (
                normalisation.source_text_ownership,
                contract.normalisation_source_text_ownership,
            ),
            "normalisation.unicode_form": (
                normalisation.unicode_form,
                contract.normalisation_unicode_form,
            ),
            "normalisation.case_folding": (
                normalisation.case_folding,
                contract.normalisation_case_folding,
            ),
            "normalisation.punctuation_removed": (
                normalisation.punctuation_removed,
                contract.normalisation_punctuation_removed,
            ),
            "normalisation.number_expansion": (
                normalisation.number_expansion,
                contract.normalisation_number_expansion,
            ),
            "normalisation.preserves_source_word_map": (
                normalisation.preserves_source_word_map,
                contract.normalisation_preserves_source_word_map,
            ),
        }
        if dataset_license is not None:
            expected_license = {
                "template_repository": contract.dataset_license_template_repository,
                "template_revision": contract.dataset_license_template_revision,
                "template_url": contract.dataset_license_template_url,
                "template_sha256": contract.dataset_license_template_sha256,
                "template_bytes": contract.dataset_license_template_bytes,
                "adaptation": contract.dataset_license_adaptation,
                "target_path": contract.dataset_license_target_path,
                "target_sha256": contract.dataset_license_target_sha256,
            }
            values.update(
                {
                    f"dataset_license.{name}": (
                        getattr(dataset_license, name),
                        expected,
                    )
                    for name, expected in expected_license.items()
                }
            )
        mismatches = [
            name
            for name, (actual, expected) in values.items()
            if type(actual) is not type(expected) or actual != expected
        ]
        if mismatches:
            raise ValueError("P1 runtime contract mismatch: " + ", ".join(mismatches))
        return
    if ctc is None:
        raise ValueError("model-backed alignment requires a CTC contract")
    values = {
        "pipeline_version": (pipeline_version, contract.pipeline_version),
        "ctc.name": (ctc.name, contract.ctc_name),
        "ctc.version": (ctc.version, contract.ctc_version),
        "ctc.source_commit": (ctc.source_commit, contract.ctc_source_commit),
        "ctc.sdist_sha256": (ctc.sdist_sha256, contract.ctc_sdist_sha256),
        "ctc.license": (ctc.license, contract.ctc_license),
        "ctc.model.repository.repository": (
            ctc.model.repository.repository,
            contract.roest_repository,
        ),
        "ctc.model.repository.revision": (
            ctc.model.repository.revision,
            contract.roest_revision,
        ),
        "ctc.model.license": (ctc.model.license, contract.roest_license),
        "ctc.model.license_url": (ctc.model.license_url, contract.roest_license_url),
        "ctc.model.license_repository": (
            ctc.model.license_repository,
            contract.roest_license_repository,
        ),
        "ctc.model.license_revision": (
            ctc.model.license_revision,
            contract.roest_license_revision,
        ),
        "ctc.model.license_sha256": (
            ctc.model.license_sha256,
            contract.roest_license_sha256,
        ),
        "ctc.model.model_card_url": (
            ctc.model.model_card_url,
            contract.roest_model_card_url,
        ),
        "ctc.model.model_card_sha256": (
            ctc.model.model_card_sha256,
            contract.roest_model_card_sha256,
        ),
        "ctc.model.license_notes": (
            ctc.model.license_notes,
            contract.roest_license_meaning,
        ),
        "ctc.model.architecture": (ctc.model.architecture, contract.roest_architecture),
        "ctc.model.model_type": (ctc.model.model_type, contract.roest_model_type),
        "ctc.model.sampling_rate": (ctc.model.sampling_rate, None),
        "ctc.model.frame_stride_samples": (
            ctc.model.frame_stride_samples,
            contract.roest_frame_stride_samples,
        ),
        "ctc.model.vocab_size": (ctc.model.vocab_size, contract.roest_vocab_size),
        "ctc.model.blank_token_id": (
            ctc.model.blank_token_id,
            contract.roest_blank_token_id,
        ),
        "ctc.model.word_delimiter_token_id": (
            ctc.model.word_delimiter_token_id,
            contract.roest_word_delimiter_token_id,
        ),
        "normalisation.version": (
            normalisation.version,
            contract.normalisation_version,
        ),
        "normalisation.source_text_ownership": (
            normalisation.source_text_ownership,
            contract.normalisation_source_text_ownership,
        ),
        "normalisation.unicode_form": (
            normalisation.unicode_form,
            contract.normalisation_unicode_form,
        ),
        "normalisation.case_folding": (
            normalisation.case_folding,
            contract.normalisation_case_folding,
        ),
        "normalisation.punctuation_removed": (
            normalisation.punctuation_removed,
            contract.normalisation_punctuation_removed,
        ),
        "normalisation.number_expansion": (
            normalisation.number_expansion,
            contract.normalisation_number_expansion,
        ),
        "normalisation.preserves_source_word_map": (
            normalisation.preserves_source_word_map,
            contract.normalisation_preserves_source_word_map,
        ),
    }
    if ctc.model.sampling_rate not in (None, contract.roest_sampling_rate):
        values["ctc.model.sampling_rate"] = (
            ctc.model.sampling_rate,
            contract.roest_sampling_rate,
        )
    if dataset_license is not None:
        expected_license = {
            "template_repository": contract.dataset_license_template_repository,
            "template_revision": contract.dataset_license_template_revision,
            "template_url": contract.dataset_license_template_url,
            "template_sha256": contract.dataset_license_template_sha256,
            "template_bytes": contract.dataset_license_template_bytes,
            "adaptation": contract.dataset_license_adaptation,
            "target_path": contract.dataset_license_target_path,
            "target_sha256": contract.dataset_license_target_sha256,
        }
        values.update(
            {
                f"dataset_license.{name}": (actual, expected)
                for name, expected in expected_license.items()
                for actual in [getattr(dataset_license, name)]
            }
        )
    mismatches = [
        name
        for name, (actual, expected) in values.items()
        if type(actual) is not type(expected) or actual != expected
    ]
    if mismatches:
        raise ValueError("P1 runtime contract mismatch: " + ", ".join(mismatches))
