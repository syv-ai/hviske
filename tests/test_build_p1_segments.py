"""Offline integration tests for the native bounded P1 build entry point."""

from __future__ import annotations

import collections.abc as c
import io
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from hviske.p1_ledger import Ledger
from hviske.p1_pipeline import run_pipeline
from hviske.p1_segments import AlignmentResult, VADSignal
from hviske.p1_source import (
    AudioPointer,
    ParsedAudio,
    ParsedTranscript,
    SourcePlan,
    SourceShard,
)
from tests.test_p1_publish import MemoryHub


def test_allocation_failure_never_leaves_programme_sharded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before allocation cannot publish a programme without a batch."""
    source = FakeSource()

    def crash(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("simulated allocation crash")

    monkeypatch.setattr(Ledger, "allocate_batch_with_shards", crash)
    run_pipeline(
        config=config(tmp_path, mode="build"),
        source=source,
        hub=MemoryHub(),
        ctc=FakeCtc(),
        vad=FakeVad(),
    )

    with Ledger(
        tmp_path / "scratch" / "ledger.sqlite", reset_processing=False
    ) as ledger:
        assert not ledger.pending_batches()
        assert ledger.programme("p1-programme-1").state.value == "retryable"


class FakeCtc:
    """Model-free aligner used by the production-shaped build smoke test."""

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


class FakeIndex:
    """Pointer index for one immutable transcript row."""

    def get(self, file_id: str) -> object | None:
        """Return the transcript pointer for the selected programme."""
        return object() if file_id == "programme-1" else None


class FakeSource:
    """Native source fake that keeps metadata and payload access separate."""

    def __init__(self) -> None:
        """Initialise the retrieval counter."""
        self.audio_calls = 0
        self.plan_calls = 0

    def build_transcript_index(
        self, *, revision: str, path: Path, objects: c.Iterable[object]
    ) -> FakeIndex:
        """Build the source-owned pointer index without retaining transcript text.

        Returns:
            A fake pointer index.
        """
        del revision, path, objects
        return FakeIndex()

    def fetch_audio(self, *, pointer: AudioPointer) -> ParsedAudio:
        """Return genuine FLAC bytes through the native source contract."""
        self.audio_calls += 1
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros((64_000, 1), dtype=np.float32), 16_000, format="FLAC")
        return ParsedAudio("programme-1", buffer.getvalue(), 16_000, 1)

    def fetch_transcript(self, pointer: object) -> ParsedTranscript:
        """Fetch the one transcript addressed by its immutable pointer.

        Returns:
            The parsed transcript.
        """
        del pointer
        from hviske.p1_contracts import SourceWord

        return ParsedTranscript(
            file_id="programme-1",
            text="hej verden",
            words=(
                SourceWord(text="hej", start_ms=0, end_ms=2_000, speaker_id=None),
                SourceWord(
                    text="verden", start_ms=2_000, end_ms=4_000, speaker_id=None
                ),
            ),
        )

    def iter_programme_metadata(self, *, shard: SourceShard) -> c.Iterable[object]:
        """Yield projected source metadata only.

        Returns:
            The projected metadata row.
        """
        del shard
        return [{"file_id": "programme-1", "duration_ms": 4_000}]

    def iter_programme_pointers(
        self, *, shard: SourceShard
    ) -> c.Iterable[AudioPointer]:
        """Yield an immutable audio row locator without decoding it."""
        yield AudioPointer("programme-1", shard, 0, 0)

    def plan(self, *, audio_revision: str, transcript_revision: str) -> SourcePlan:
        """Return source tree metadata without opening a Parquet row."""
        self.plan_calls += 1
        return SourcePlan(
            audio_repository="audio",
            transcript_repository="transcripts",
            audio_revision=audio_revision,
            transcript_revision=transcript_revision,
            audio_shards=(SourceShard("data/audio.parquet", 100),),
            transcript_objects=(("data/transcripts.parquet", 100, None),),
        )


class FakeVad:
    """Model-free VAD adapter."""

    def analyse(self, audio: np.ndarray, sampling_rate: int) -> VADSignal:
        """Keep the smoke test independent of Silero weights.

        Returns:
            Full-programme speech evidence.
        """
        del sampling_rate
        duration = len(audio) * 1000 // 16_000
        return VADSignal(((0, duration),), duration)


def config(tmp_path: Path, mode: str = "plan") -> DictConfig:
    """Load the pinned config with a test-owned scratch root.

    Returns:
        A Hydra configuration for the test.
    """
    value = OmegaConf.load("config/p1_segments.yaml")
    value.mode = mode
    value.programme_limit = 1
    value.runtime.scratch_root = str(tmp_path / "scratch")
    value.runtime.target_shard_bytes = 1
    return cast(DictConfig, value)


def test_build_decodes_flac_and_purges_only_verified_output(tmp_path: Path) -> None:
    """A native build uses FLAC bytes and purges after publisher verification."""
    source = FakeSource()
    report = run_pipeline(
        config=config(tmp_path, mode="build"),
        source=source,
        hub=MemoryHub(),
        ctc=FakeCtc(),
        vad=FakeVad(),
    )

    assert source.audio_calls == 1
    assert report.max_in_flight == 1
    assert report.processed == 1
    assert not list((tmp_path / "scratch" / "staging").rglob("*.parquet"))
    audit = (tmp_path / "scratch" / "audit-candidates.jsonl").read_text()
    assert '"parquet_path": "data/train/p1-programme-1-00000.parquet"' in audit
    assert '"row_locator": 0' in audit


def test_plan_dispatches_metadata_only_source_tree_access(tmp_path: Path) -> None:
    """Planning calls source metadata APIs and never retrieves audio."""
    source = FakeSource()
    report = run_pipeline(config=config(tmp_path), source=source)

    assert source.plan_calls == 1
    assert source.audio_calls == 0
    assert report.selected_file_ids == ()
    assert report.preflight.target["present"] is False


def test_quality_rejection_is_terminal_and_keeps_source_audit_locator(
    tmp_path: Path,
) -> None:
    """A fully decoded but rejected programme is not retried or published."""
    source = FakeSource()
    build_config = config(tmp_path, mode="build")
    build_config.segmentation.minimum_alignment_score = 2.0
    hub = MemoryHub()

    report = run_pipeline(
        config=build_config, source=source, hub=hub, ctc=FakeCtc(), vad=FakeVad()
    )

    with Ledger(
        tmp_path / "scratch" / "ledger.sqlite", reset_processing=False
    ) as ledger:
        assert ledger.programme("p1-programme-1").state.value == "rejected"
    assert report.rejection_counts["low_alignment_score"] == 1
    assert not hub.commits
    manifest = (tmp_path / "scratch" / "audit-candidates.jsonl").read_text()
    assert '"source_shard_path": "data/audio.parquet"' in manifest
    assert '"source_row_index": 0' in manifest


def test_sharding_crash_recovers_equivalent_audit_manifest(tmp_path: Path) -> None:
    """A second invocation recovers audit metadata committed with sharding."""
    crash_code = """
import os
import sys
from pathlib import Path
from hviske.p1_ledger import Ledger
from hviske.p1_pipeline import run_pipeline
from tests.test_build_p1_segments import FakeCtc, FakeSource, FakeVad, config
from tests.test_p1_publish import MemoryHub

root = Path(sys.argv[1])
original = Ledger.allocate_batch_with_shards

def crash(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    os._exit(91)

Ledger.allocate_batch_with_shards = crash
run_pipeline(
    config=config(root, mode="build"),
    source=FakeSource(),
    hub=MemoryHub(),
    ctc=FakeCtc(),
    vad=FakeVad(),
)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_code, str(tmp_path / "recovery")],
        cwd=Path.cwd(),
        check=False,
    )
    assert crashed.returncode == 91

    recovery_code = """
import sys
from pathlib import Path
from hviske.p1_pipeline import run_pipeline
from tests.test_build_p1_segments import FakeCtc, FakeSource, FakeVad, config
from tests.test_p1_publish import MemoryHub

root = Path(sys.argv[1])
run_pipeline(
    config=config(root, mode="build"),
    source=FakeSource(),
    hub=MemoryHub(),
    ctc=FakeCtc(),
    vad=FakeVad(),
)
"""
    recovered = subprocess.run(
        [sys.executable, "-c", recovery_code, str(tmp_path / "recovery")],
        cwd=Path.cwd(),
        check=False,
    )
    assert recovered.returncode == 0

    control_root = tmp_path / "control"
    run_pipeline(
        config=config(control_root, mode="build"),
        source=FakeSource(),
        hub=MemoryHub(),
        ctc=FakeCtc(),
        vad=FakeVad(),
    )
    recovery_manifest = (
        tmp_path / "recovery" / "scratch" / "audit-candidates.jsonl"
    ).read_text()
    control_manifest = (control_root / "scratch" / "audit-candidates.jsonl").read_text()
    assert recovery_manifest
    assert sorted(recovery_manifest.splitlines()) == sorted(
        control_manifest.splitlines()
    )


def test_verification_failure_retains_local_shard(tmp_path: Path) -> None:
    """A failed remote verification leaves the ledger batch and bytes recoverable."""
    source = FakeSource()
    report = run_pipeline(
        config=config(tmp_path, mode="build"),
        source=source,
        hub=VerifyFailHub(),
        ctc=FakeCtc(),
        vad=FakeVad(),
    )

    assert report.processed == 0
    assert list((tmp_path / "scratch" / "staging").rglob("*.parquet"))
    with Ledger(
        tmp_path / "scratch" / "ledger.sqlite", reset_processing=False
    ) as ledger:
        assert ledger.pending_batches()


class VerifyFailHub(MemoryHub):
    """Hub fake that fails during remote digest verification."""

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> list[bytes]:
        """Fail before the publisher is allowed to purge local files.

        Raises:
            RuntimeError:
                Always, to simulate a remote verification failure.
        """
        del repo_id, path, repo_type, revision
        raise RuntimeError("simulated verification failure")
