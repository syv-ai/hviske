"""Acceptance tests for the timestamp-native P1 v8 path."""

from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from p1_dataset.contracts import (
    OUTPUT_SCHEMA,
    NormalisationContract,
    SegmentationContract,
    SourceWord,
)
from p1_dataset.pipeline import (
    PipelineSettings,
    configure_scratch,
    preflight_pipeline,
    run_pipeline,
)
from p1_dataset.segments import (
    CTCBackend,
    TimestampAlignmentBackend,
    VADBackend,
    segment_programme,
)
from tests.test_p1_publish import MemoryHub


@pytest.mark.parametrize(("end_ms", "accepted"), [(9_999, True), (10_000, False)])
def test_v8_duration_upper_bound_is_exclusive(end_ms: int, accepted: bool) -> None:
    """The timestamp span accepts 9,999 ms but rejects exactly 10 seconds."""
    words = (SourceWord(text="ord", start_ms=0, end_ms=end_ms),)
    result = segment_programme(
        words=words,
        audio=np.zeros(end_ms * 16, dtype=np.float32),
        source_file_id="programme",
        source_duration_ms=end_ms,
        segmentation=_segmentation(),
        normalisation=NormalisationContract(version="unused"),
        ctc=TimestampAlignmentBackend(),
        pipeline_version="p1-segmentation-8",
        pipeline_config_sha256="a" * 64,
    )

    assert bool(result.rows) is accepted


def _segmentation() -> SegmentationContract:
    """Return the production duration contract for direct tests."""
    return SegmentationContract(
        minimum_duration_ms=1_000,
        target_minimum_duration_ms=2_000,
        target_maximum_duration_ms=8_000,
        maximum_duration_ms=10_000,
        maximum_drift_ms=500,
        minimum_alignment_score=-10.0,
        minimum_vad_speech_ratio=0.0,
    )


def test_v8_initialise_never_probes_cuda_or_model_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initialisation records explicit non-applicability without probing models."""
    monkeypatch.setattr(
        "p1_dataset.pipeline.cuda_status",
        lambda *_args, **_kwargs: pytest.fail("CUDA initialisation probe was called"),
    )
    config = _config(tmp_path)
    config.mode = "initialise"
    report = run_pipeline(config=config, hub=MemoryHub())

    assert report.preflight.cuda == {
        "checked": False,
        "reason": "not_applicable:timestamp-native",
    }
    assert report.preflight.model_revisions == {}
    assert report.preflight.checks["model_provenance_not_applicable"] is True


def _config(tmp_path: Path) -> DictConfig:
    """Load the active v8 configuration with an isolated scratch directory.

    Returns:
        The v8 configuration with temporary scratch storage.
    """
    config = OmegaConf.load("config/p1_segments.yaml")
    config.mode = "plan"
    config.runtime.scratch_root = str(tmp_path)
    return t.cast(DictConfig, config)


def test_v8_pipeline_rejects_injected_model_backends(tmp_path: Path) -> None:
    """The orchestration entry point refuses injected model implementations."""
    with pytest.raises(ValueError, match="injected model-backed"):
        run_pipeline(config=_config(tmp_path), ctc=t.cast(CTCBackend, object()))


def test_v8_preflight_never_checks_models_or_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The active plan path has no model, VAD, or CUDA preflight dependency."""
    settings = PipelineSettings.from_config(_config(tmp_path))
    monkeypatch.setattr(
        "p1_dataset.pipeline.check_model_revisions",
        lambda *_args, **_kwargs: pytest.fail("model revision check was called"),
    )
    monkeypatch.setattr(
        "p1_dataset.pipeline.cuda_status",
        lambda *_args, **_kwargs: pytest.fail("CUDA preflight was called"),
    )

    report = preflight_pipeline(
        settings=settings,
        source=object(),
        hub=None,
        shards=(),
        selected_programmes=0,
        maximum_source_bytes=0,
    )

    assert report.cuda["reason"] == "not_applicable:timestamp-native"
    assert report.checks["cuda_device_checked"] is False
    assert report.model_revisions == {}


def test_v8_rejects_injected_model_backends() -> None:
    """The active path refuses both injected CTC and VAD implementations."""
    words = (SourceWord(text="ord", start_ms=0, end_ms=1_000),)
    with pytest.raises(ValueError, match="model-backed"):
        segment_programme(
            words=words,
            audio=np.zeros(16_000, dtype=np.float32),
            source_file_id="programme",
            source_duration_ms=1_000,
            segmentation=_segmentation(),
            normalisation=NormalisationContract(version="unused"),
            ctc=t.cast(CTCBackend, object()),
            pipeline_version="p1-segmentation-8",
            pipeline_config_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="VAD"):
        segment_programme(
            words=words,
            audio=np.zeros(16_000, dtype=np.float32),
            source_file_id="programme",
            source_duration_ms=1_000,
            segmentation=_segmentation(),
            normalisation=NormalisationContract(version="unused"),
            ctc=TimestampAlignmentBackend(),
            pipeline_version="p1-segmentation-8",
            pipeline_config_sha256="a" * 64,
            vad=t.cast(VADBackend, object()),
        )


def test_v8_uses_exact_word_boundaries_and_no_acoustic_evidence() -> None:
    """Timestamp-native rows preserve words, speakers, and source endpoints."""
    words = (
        SourceWord(text="Hej", start_ms=123, end_ms=1_100, speaker_id="a"),
        SourceWord(text="verden", start_ms=1_100, end_ms=3_123, speaker_id="a"),
    )
    result = segment_programme(
        words=words,
        audio=np.zeros(3_123 * 16, dtype=np.float32),
        source_file_id="programme",
        source_duration_ms=3_123,
        segmentation=_segmentation(),
        normalisation=NormalisationContract(version="unused"),
        ctc=TimestampAlignmentBackend(),
        pipeline_version="p1-segmentation-8",
        pipeline_config_sha256="a" * 64,
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert (row.source_start_ms, row.source_end_ms) == (123, 3_123)
    assert row.text == "Hej verden"
    assert row.speaker_ids == ("a",)
    assert row.alignment_method == "timestamp-native:p1-transcripts.words"
    assert row.alignment_score is None
    assert row.start_drift_ms is None
    assert row.end_drift_ms is None
    assert row.vad_speech_ratio is None
    assert OUTPUT_SCHEMA.schema_version == "p1-segments-v2"
    assert "alignment_method" in {field.name for field in OUTPUT_SCHEMA.fields}


def test_v8_uses_model_free_scratch_layout(tmp_path: Path) -> None:
    """The active scratch layout does not create model-cache directories."""
    root = configure_scratch(tmp_path / "p1-v8", model_free=True)

    assert root.name == "p1-v8"
    assert not (root / "transformers").exists()
    assert not (root / "torch").exists()
