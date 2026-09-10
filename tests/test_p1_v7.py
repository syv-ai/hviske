"""Acceptance tests for the timestamp-native P1 v7 path."""

from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from hviske.p1_contracts import (
    OUTPUT_SCHEMA,
    NormalisationContract,
    SegmentationContract,
    SourceWord,
)
from hviske.p1_pipeline import PipelineSettings, preflight_pipeline, run_pipeline
from hviske.p1_segments import TimestampAlignmentBackend, segment_programme
from tests.test_p1_publish import MemoryHub


@pytest.mark.parametrize(("end_ms", "accepted"), [(9_999, True), (10_000, False)])
def test_v7_duration_upper_bound_is_exclusive(end_ms: int, accepted: bool) -> None:
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
        pipeline_version="p1-segmentation-7",
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


def test_v7_initialise_never_probes_cuda_or_model_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initialisation records explicit non-applicability without probing models."""
    monkeypatch.setattr(
        "hviske.p1_pipeline.cuda_status",
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
    """Load the active v7 configuration with an isolated scratch directory.

    Returns:
        The v7 configuration with temporary scratch storage.
    """
    config = OmegaConf.load("config/p1_segments.yaml")
    config.mode = "plan"
    config.runtime.scratch_root = str(tmp_path)
    return t.cast(DictConfig, config)


def test_v7_preflight_never_checks_models_or_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The active plan path has no model, VAD, or CUDA preflight dependency."""
    settings = PipelineSettings.from_config(_config(tmp_path))
    monkeypatch.setattr(
        "hviske.p1_pipeline.check_model_revisions",
        lambda *_args, **_kwargs: pytest.fail("model revision check was called"),
    )
    monkeypatch.setattr(
        "hviske.p1_pipeline.cuda_status",
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


def test_v7_uses_exact_word_boundaries_and_no_acoustic_evidence() -> None:
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
        pipeline_version="p1-segmentation-7",
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
