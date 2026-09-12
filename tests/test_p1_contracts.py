"""Tests for the deterministic Phase 1A P1 contracts."""

import json

import pytest

from p1_dataset.contracts import (
    CanonicalIdentityManifest,
    CTCContract,
    DatasetLicenseContract,
    LedgerState,
    ModelContract,
    NormalisationContract,
    OutputEncodingContract,
    RepositoryRevision,
    SegmentationContract,
    SourceCoordinates,
    SourceProgramme,
    SourceTextSpan,
    SourceWord,
    VADContract,
    canonical_json,
    pipeline_config_sha256,
    segment_id,
    valid_ledger_transition,
)

AUDIO_REVISION = "449b9c2294026df6d0d37538f279fdec03f565ff"
TRANSCRIPT_REVISION = "41132579816d86e889635f84f30511279f026359"
VAD_REVISION = "867c2aa692646a1f1de3e94a15c9dd9f614c0acb"
VAD_MODEL = "5c6988d663950a93a5f0d6c38c2fe024653ec552b"
CTC_REVISION = "69bd9b53b7b82ad926d35e7b280f957ed299a7db"
CTC_MODEL_REVISION = "beb3e790246d6b9dec1df596b0b21d5c42f4d99c"
ANOMALY_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"


def test_canonical_json_normalises_unicode_and_preserves_numbers() -> None:
    """Canonical JSON is compact, sorted, NFC, and type-preserving."""
    first = canonical_json({"z": "e\u0301", "integer": 1, "float": 1.0})
    second = canonical_json({"float": 1.0, "integer": 1, "z": "é"})

    assert first == second
    assert first == '{"float":1.0,"integer":1,"z":"é"}'
    assert json.loads(first)["integer"] == 1
    assert isinstance(json.loads(first)["float"], float)


def test_canonical_json_rejects_non_finite_floats() -> None:
    """Non-finite values cannot create ambiguous pipeline identities."""
    with pytest.raises((TypeError, ValueError)):
        canonical_json({"score": float("nan")})


@pytest.mark.parametrize(
    "template_url",
    [
        "https://huggingface.co/datasets/org/data/raw/main/LICENSE",
        "https://huggingface.co/datasets/org/data/resolve/main/LICENSE",
        "https://huggingface.co/datasets/org/data/blob/main/LICENSE",
        "https://huggingface.co/datasets/org/data/LICENSE",
    ],
)
def test_dataset_license_rejects_mutable_urls(template_url: str) -> None:
    """Dataset licence provenance accepts only commit-pinned Hub URLs."""
    license_data = make_manifest().dataset_license.model_dump()
    license_data["template_url"] = template_url
    with pytest.raises(ValueError, match="complete SHA"):
        DatasetLicenseContract.model_validate(license_data)


def make_manifest() -> CanonicalIdentityManifest:
    """Build the pinned Phase 1A identity fixture.

    Returns:
        A complete immutable identity manifest.
    """
    return CanonicalIdentityManifest(
        schema_version="p1-segments-v1",
        pipeline_version="p1-segmentation-1a",
        source=SourceCoordinates(
            audio=RepositoryRevision(repository="syvai/p1", revision=AUDIO_REVISION),
            transcripts=RepositoryRevision(
                repository="syvai/p1-transcripts", revision=TRANSCRIPT_REVISION
            ),
        ),
        vad=VADContract(
            name="silero-vad",
            repository=RepositoryRevision(
                repository="snakers4/silero-vad", revision=VAD_REVISION
            ),
            model_blob=VAD_MODEL,
            license="MIT",
        ),
        ctc=CTCContract(
            source_commit=CTC_REVISION,
            sdist_sha256=(
                "19d383ea5f22438ebb1699d72b22078b63f351a33fa50bedb19c14077ba6a116"
            ),
            license="Apache-2.0",
            model=ModelContract(
                repository=RepositoryRevision(
                    repository="CoRal-project/roest-v3-wav2vec2-315m",
                    revision=CTC_MODEL_REVISION,
                ),
                license="openrail",
                license_url=(
                    "https://huggingface.co/Alvenir/coral-1-whisper-large/resolve/"
                    "a6c1e24d9f10e6289607a1ba32341b68e8660688/LICENSE"
                ),
                architecture="Wav2Vec2ForCTC",
                model_type="wav2vec2",
                sampling_rate=16000,
                frame_stride_samples=320,
                vocab_size=46,
                blank_token_id=45,
                word_delimiter_token_id=36,
            ),
        ),
        anomaly_model=ModelContract(
            repository=RepositoryRevision(
                repository="openai/whisper-small", revision=ANOMALY_REVISION
            ),
            license="Apache-2.0",
        ),
        normalisation=NormalisationContract(version="p1-text-normalisation-1"),
        segmentation=SegmentationContract(
            minimum_duration_ms=1000,
            target_minimum_duration_ms=2000,
            target_maximum_duration_ms=8000,
            maximum_duration_ms=10000,
            maximum_drift_ms=500,
            minimum_alignment_score=0.0,
            minimum_vad_speech_ratio=0.0,
        ),
        output=OutputEncodingContract(),
    )


def test_identity_and_segment_ids_are_deterministic() -> None:
    """The manifest and segment identities are stable and framed by JSON."""
    manifest = make_manifest()
    digest = pipeline_config_sha256(manifest)

    assert digest == pipeline_config_sha256(manifest.model_copy())
    assert segment_id(
        pipeline_config_sha256=digest,
        source_file_id="programme-1",
        source_start_ms=100,
        source_end_ms=2100,
        text="Hej, verden!",
    ) == segment_id(
        pipeline_config_sha256=digest,
        source_file_id="programme-1",
        source_start_ms=100,
        source_end_ms=2100,
        text="Hej, verden!",
    )
    assert digest != pipeline_config_sha256(
        manifest.model_copy(update={"pipeline_version": "p1-segmentation-2"})
    )
    assert digest != pipeline_config_sha256(
        manifest.model_copy(
            update={
                "normalisation": manifest.normalisation.model_copy(
                    update={"source_text_ownership": "legacy"}
                )
            }
        )
    )
    assert digest != pipeline_config_sha256(
        manifest.model_copy(
            update={
                "segmentation": manifest.segmentation.model_copy(
                    update={"maximum_duration_ms": 9999}
                )
            }
        )
    )


def test_ledger_transition_contract() -> None:
    """Ledger transitions follow the restart-safe state machine."""
    assert valid_ledger_transition(LedgerState.DISCOVERED, LedgerState.PROCESSING)
    assert valid_ledger_transition(LedgerState.PROCESSING, LedgerState.RETRYABLE)
    assert valid_ledger_transition(LedgerState.COMMITTED, LedgerState.VERIFIED)
    assert not valid_ledger_transition(LedgerState.SHARDED, LedgerState.RETRYABLE)
    assert not valid_ledger_transition(LedgerState.COMMITTED, LedgerState.RETRYABLE)
    assert not valid_ledger_transition(LedgerState.VERIFIED, LedgerState.PROCESSING)
    assert not valid_ledger_transition(LedgerState.PURGED, LedgerState.PROCESSING)


def test_revisions_reject_mutable_or_incomplete_coordinates() -> None:
    """Branches and abbreviated SHAs cannot enter the identity manifest."""
    with pytest.raises(ValueError, match="complete 40-character"):
        RepositoryRevision(repository="syvai/p1", revision="main")
    with pytest.raises(ValueError, match="complete 40-character"):
        RepositoryRevision(repository="syvai/p1", revision=AUDIO_REVISION[:-1])


def test_source_contract_records_verbatim_character_spans() -> None:
    """Programme validation records separators without altering source text."""
    programme = SourceProgramme(
        file_id="programme",
        duration_ms=3_000,
        words=(
            SourceWord(text="Hej,", start_ms=0, end_ms=1_000),
            SourceWord(text="verden!", start_ms=1_000, end_ms=2_000),
        ),
        transcript_text="Hej,  verden!",
    )

    assert programme.words[0].source_span is not None
    assert programme.words[0].source_span.start == 0
    assert programme.words[1].separator_text == "  "
    assert programme.words[1].separator_span is not None
    assert programme.words[-1].trailing_text == ""
    assert programme.transcript_text == "Hej,  verden!"


def test_source_contract_rejects_inconsistent_character_spans() -> None:
    """Annotated source offsets must identify their declared word text."""
    with pytest.raises(ValueError, match="not present at its source offset"):
        SourceProgramme(
            file_id="programme",
            duration_ms=3_000,
            words=(
                SourceWord(
                    text="Hej",
                    start_ms=0,
                    end_ms=1_000,
                    source_span=SourceTextSpan(start=1, end=4),
                ),
            ),
            transcript_text="Hej verden",
        )


def test_source_contract_rejects_partial_character_spans() -> None:
    """A partial source annotation cannot silently fall back to lexical matching."""
    with pytest.raises(ValueError, match="present for every source word"):
        SourceProgramme(
            file_id="programme",
            duration_ms=3_000,
            words=(
                SourceWord(
                    text="Hej",
                    start_ms=0,
                    end_ms=1_000,
                    source_span=SourceTextSpan(start=0, end=3),
                ),
                SourceWord(text="verden", start_ms=1_000, end_ms=2_000),
            ),
            transcript_text="Hej verden",
        )
