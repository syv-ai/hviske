"""Offline integration tests for the bounded P1 build entry point."""

from __future__ import annotations

import collections.abc as c
from pathlib import Path
from typing import cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from hviske.p1_segments import AlignmentResult, VADSignal
from scripts.build_p1_segments import (
    P1PreflightError,
    TranscriptIndex,
    build_transcript_index,
    run_pipeline,
)
from tests.test_p1_publish import MemoryHub


def test_build_uses_one_worker_and_purges_verified_publication(tmp_path: Path) -> None:
    """A fake build keeps one programme in flight and purges verified files."""
    source = FakeSource()
    settings = config(tmp_path, mode="build")
    report = run_pipeline(
        config=settings, source=source, hub=MemoryHub(), ctc=FakeCtc(), vad=FakeVad()
    )

    assert source.audio_calls == 1
    assert report.max_in_flight == 1
    assert report.processed == 1
    assert not list((tmp_path / "scratch" / "staging").rglob("*.parquet"))

    run_pipeline(
        config=settings, source=source, hub=MemoryHub(), ctc=FakeCtc(), vad=FakeVad()
    )
    assert source.audio_calls == 1


class FakeCtc:
    """Model-free aligner used by the build smoke test."""

    def align(
        self,
        audio: np.ndarray,
        alignment_text: str,
        word_map: tuple[str, ...],
        start_ms: int,
        end_ms: int,
        sampling_rate: int,
    ) -> AlignmentResult:
        """Return proposal boundaries without a model call."""
        del audio, alignment_text, word_map, sampling_rate
        return AlignmentResult(start_ms, end_ms, 1.0, "fake", ())


class FakeSource:
    """Source fake that records retrieval and exposes deliberately unsorted shards."""

    def __init__(self) -> None:
        """Initialise the retrieval counter."""
        self.audio_calls = 0

    def iter_programmes(
        self, *, shard: object, index: TranscriptIndex
    ) -> c.Iterable[object]:
        """Yield the one source row without touching audio.

        Returns:
            One metadata row.
        """
        return [
            {
                "file_id": "programme-1",
                "duration_ms": 4_000,
                "words": index.records["programme-1"].row["words"],
            }
        ]

    def iter_transcripts(self, *, revision: str) -> c.Iterable[object]:
        """Return one usable transcript and metadata defects."""
        return [
            {
                "file_id": "programme-1",
                "transcript_text": "hej verden",
                "duration_ms": 4_000,
                "words": [
                    {"text": "hej", "start_ms": 0, "end_ms": 2_000},
                    {"text": "verden", "start_ms": 2_000, "end_ms": 4_000},
                ],
            },
            {"file_id": "empty-1", "transcript_text": ""},
            {"file_id": None, "transcript_text": "missing key"},
            {"file_id": "programme-1", "transcript_text": "duplicate"},
        ]

    def list_audio_shards(self, *, revision: str) -> c.Iterable[object]:
        """Return source metadata in a non-deterministic order."""
        return [{"path": "b.parquet", "size": 200}, {"path": "a.parquet", "size": 100}]

    def retrieve_audio(self, *, programme: object, shard: object) -> object:
        """Record retrieval and return a four-second silent clip.

        Returns:
            A synthetic audio array.
        """
        self.audio_calls += 1
        return np.zeros(64_000, dtype=np.float32)


class FakeVad:
    """Model-free VAD adapter."""

    def analyse(self, audio: np.ndarray, sampling_rate: int) -> VADSignal:
        """Keep the smoke test independent of Silero weights.

        Returns:
            Full-programme speech evidence.
        """
        del sampling_rate
        return VADSignal(
            ((0, len(audio) * 1000 // 16_000),), len(audio) * 1000 // 16_000
        )


def config(tmp_path: Path, mode: str = "plan") -> DictConfig:
    """Load the pinned config with a test-owned scratch root.

    Returns:
        A resolved Hydra configuration for the test.
    """
    value = OmegaConf.load("config/p1_segments.yaml")
    value.mode = mode
    value.programme_limit = 1
    value.runtime.scratch_root = str(tmp_path / "scratch")
    value.runtime.target_shard_bytes = 1
    return cast(DictConfig, value)


def test_plan_has_no_audio_and_records_index_defects(tmp_path: Path) -> None:
    """Planning performs selection and quotas without invoking the audio loader."""
    source = FakeSource()
    report = run_pipeline(config=config(tmp_path), source=source)

    assert source.audio_calls == 0
    assert report.selected_file_ids == ("programme-1",)
    assert report.preflight.target["present"] is False
    events = (tmp_path / "scratch" / "p1-events.jsonl").read_text()
    assert "empty_text" in events and "duplicate_file_id" in events
    assert "transcript_text" not in events


def test_source_cap_aborts_before_audio(tmp_path: Path) -> None:
    """A source-object cap fails before a source retrieval can occur.

    Raises:
        AssertionError:
            If the cap does not abort the run.
    """
    source = FakeSource()
    value = config(tmp_path)
    value.max_source_bytes = 1

    try:
        run_pipeline(config=value, source=source)
    except P1PreflightError:
        pass
    else:
        raise AssertionError("source cap did not abort planning")
    assert source.audio_calls == 0


def test_transcript_index_rejects_null_empty_and_duplicate_keys() -> None:
    """The authenticated metadata index has explicit, deterministic rejections."""
    index = build_transcript_index(
        [
            {"file_id": "ok", "transcript_text": "text"},
            {"file_id": "", "transcript_text": "text"},
            {"file_id": "empty", "transcript_text": ""},
            {"file_id": "ok", "transcript_text": "again"},
            {"file_id": None, "transcript_text": "text"},
        ]
    )

    assert tuple(index.records) == ("ok",)
    assert [item.reason for item in index.rejections] == [
        "null_file_id",
        "empty_text",
        "duplicate_file_id",
        "null_file_id",
    ]
