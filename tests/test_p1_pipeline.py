"""Offline tests for the reusable P1 orchestration layer."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf

from hviske.p1_pipeline import (
    MetadataLog,
    P1PreflightError,
    PipelineSettings,
    _native_candidates,
    preflight_pipeline,
    run_pipeline,
)
from hviske.p1_source import SourcePlan, SourceShard
from tests.test_p1_publish import MemoryHub


def test_existing_scratch_is_included_in_hard_budget(tmp_path: Path) -> None:
    """An existing scratch tree cannot be ignored by the preflight quota."""
    config = pipeline_config(tmp_path)
    settings = PipelineSettings.from_config(config)
    settings.scratch_root.mkdir(parents=True)
    (settings.scratch_root / "old.bin").write_bytes(b"x" * 100)
    settings = dataclasses.replace(settings, max_scratch_bytes=100)
    with pytest.raises(P1PreflightError, match="existing scratch"):
        preflight_pipeline(
            settings=settings,
            source=object(),
            hub=None,
            shards=(),
            selected_programmes=0,
            maximum_source_bytes=0,
        )


def pipeline_config(tmp_path: Path, mode: str = "plan") -> DictConfig:
    """Return a small test-owned pipeline configuration."""
    config = OmegaConf.load("config/p1_segments.yaml")
    config.mode = mode
    config.runtime.scratch_root = str(tmp_path / "scratch")
    config.runtime.device = "cpu"
    return cast(DictConfig, config)


def test_initialise_commits_only_private_metadata(tmp_path: Path) -> None:
    """Initialisation creates the private target without source retrieval."""
    source = MetadataSource()
    hub = MemoryHub()
    run_pipeline(
        config=pipeline_config(tmp_path, mode="initialise"), source=source, hub=hub
    )

    assert hub.private is True
    assert hub.commits == [("README.md", ".gitattributes")]
    assert source.iterated is False


class MetadataSource:
    """A source fake that exposes repository metadata but no payload methods."""

    def __init__(self) -> None:
        """Set a guard for accidental Parquet iteration."""
        self.iterated = False

    def iter_programme_metadata(self, **_: object) -> None:
        """Fail if planning starts reading a Parquet row.

        Raises:
            AssertionError:
                If metadata row iteration is attempted during planning.
        """
        self.iterated = True
        raise AssertionError("plan mode must not iterate Parquet rows")

    def plan(self, **_: object) -> SourcePlan:
        """Return a metadata-only source plan."""
        return SourcePlan(
            audio_repository="audio",
            transcript_repository="transcripts",
            audio_revision="a" * 40,
            transcript_revision="b" * 40,
            audio_shards=(SourceShard("data/audio.parquet", 10),),
            transcript_objects=(("data/transcripts.parquet", 10, None),),
        )


def test_pilot_selection_is_bounded_and_not_first_rows(tmp_path: Path) -> None:
    """The pilot uses a stable reservoir rather than taking stream order."""

    class Source:
        def iter_programme_metadata(self, **_: object) -> list[dict[str, object]]:
            return [{"file_id": f"programme-{i}"} for i in range(20)]

    class Index:
        def get(self, _: str) -> object:
            return object()

    shard = SourceShard("data/audio.parquet", 10)
    first = _native_candidates(
        source=Source(),
        shards=(shard,),
        index=Index(),
        programme_limit=3,
        source_file_id=None,
        pilot=True,
        log=MetadataLog(tmp_path / "events-1.jsonl"),
    )
    second = _native_candidates(
        source=Source(),
        shards=(shard,),
        index=Index(),
        programme_limit=3,
        source_file_id=None,
        pilot=True,
        log=MetadataLog(tmp_path / "events-2.jsonl"),
    )
    assert [item[0] for item in first] == [item[0] for item in second]
    assert [item[0] for item in first] != ["programme-0", "programme-1", "programme-2"]


def test_plan_reads_tree_metadata_only(tmp_path: Path) -> None:
    """Plan mode does not build a pointer index or touch a Parquet row."""
    source = MetadataSource()
    report = run_pipeline(config=pipeline_config(tmp_path), source=source)

    assert report.selected_file_ids == ()
    assert source.iterated is False
    assert report.preflight.target["present"] is False


def test_plan_verifies_github_vad_and_hub_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning verifies model provenance without loading model weights."""
    calls: list[tuple[str, str, str]] = []

    def verify_vad(**kwargs: object) -> str:
        calls.append(("github", str(kwargs["repository"]), str(kwargs["revision"])))
        return "https://github.test/asset"

    def verify_model(**kwargs: object) -> None:
        calls.append(("hub", str(kwargs["repository"]), str(kwargs["revision"])))

    monkeypatch.setattr("hviske.p1_models.verify_silero_vad_revision", verify_vad)
    monkeypatch.setattr("hviske.p1_models.verify_hub_model_revision", verify_model)
    source = MetadataSource()
    source._client = lambda: object()  # type: ignore[attr-defined]

    run_pipeline(config=pipeline_config(tmp_path), source=source)

    assert [item[0] for item in calls] == ["github", "hub", "hub"]
