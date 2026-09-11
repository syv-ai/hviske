"""Offline integration tests for the native bounded P1 build entry point."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from hviske.p1_ledger import Ledger
from hviske.p1_pipeline import PipelineSettings, initialise_target, run_pipeline
from hviske.p1_source import (
    AudioPointer,
    ParsedAudio,
    ParsedTranscript,
    SourcePlan,
    SourceShard,
)
from tests.test_p1_publish import MemoryHub


class VerifyFailHub(MemoryHub):
    """Hub fake that fails during remote digest verification."""

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> list[bytes]:
        """Fail before the publisher is allowed to purge local files.

        Returns:
            README bytes during preflight; never returns verification bytes.

        Raises:
            RuntimeError:
                When a shard is streamed for remote verification.
        """
        if path == "README.md":
            return super().stream_file(
                repo_id, path, repo_type=repo_type, revision=revision
            )
        raise RuntimeError("simulated verification failure")


def test_allocation_failure_never_leaves_programme_sharded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before allocation cannot publish a programme without a batch."""
    source = FakeSource()

    def crash(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("simulated allocation crash")

    monkeypatch.setattr(Ledger, "allocate_batch_with_shards", crash)
    with pytest.raises(RuntimeError, match="simulated allocation crash"):
        run_pipeline(
            config=config(tmp_path, mode="build"),
            source=source,
            hub=prepared_hub(tmp_path),
        )

    with Ledger(
        tmp_path / "scratch" / "ledger.sqlite", reset_processing=False
    ) as ledger:
        assert not ledger.reconstruct_work()
        programme = ledger.programme("p1-programme-1")
        assert programme.state.value == "retryable"
        assert programme.last_error == "runtime_error"


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


def prepared_hub(tmp_path: Path) -> MemoryHub:
    """Return a fake target initialised with the active v8 metadata card."""
    hub = MemoryHub()
    settings = PipelineSettings.from_config(config(tmp_path, mode="build"))
    initialise_target(hub=hub, settings=settings)
    return hub


def test_build_encodes_ogg_opus_and_purges_only_verified_output(tmp_path: Path) -> None:
    """A native v8 build uses OGG/Opus and purges after verification."""
    source = FakeSource()
    hub = prepared_hub(tmp_path)
    report = run_pipeline(config=config(tmp_path, mode="build"), source=source, hub=hub)

    assert source.audio_calls == 1
    assert report.max_in_flight == 1
    assert report.processed == 1
    assert report.selected_programmes == 1
    assert not list((tmp_path / "scratch" / "staging").rglob("*.parquet"))
    published = pq.read_table(
        io.BytesIO(hub.files["data/train/p1-programme-1-00000.parquet"])
    ).to_pylist()[0]["audio"]["bytes"]
    with sf.SoundFile(io.BytesIO(published)) as audio_file:
        assert audio_file.format == "OGG"
        assert audio_file.subtype == "OPUS"
        assert audio_file.samplerate == 16_000
        assert audio_file.channels == 1
    audit = (tmp_path / "scratch" / "audit-candidates.jsonl").read_text()
    assert '"parquet_path": "data/train/p1-programme-1-00000.parquet"' in audit
    assert '"row_locator": 0' in audit
    assert any(path.startswith("manifests/p0-batch-") for path in hub.files)


def test_plan_dispatches_metadata_only_source_tree_access(tmp_path: Path) -> None:
    """Planning calls source metadata APIs and never retrieves audio."""
    source = FakeSource()
    report = run_pipeline(config=config(tmp_path), source=source)

    assert source.plan_calls == 1
    assert source.audio_calls == 0
    assert report.selected_programmes == 0
    assert report.preflight.target["present"] is False


def test_quality_rejection_is_terminal_and_keeps_source_audit_locator(
    tmp_path: Path,
) -> None:
    """A fully decoded but rejected programme is not retried or published."""
    source = FakeSource()
    build_config = config(tmp_path, mode="build")
    build_config.segmentation.minimum_duration_ms = 5_000
    build_config.segmentation.target_minimum_duration_ms = 5_000
    hub = MemoryHub()
    initialise_target(hub=hub, settings=PipelineSettings.from_config(build_config))

    report = run_pipeline(config=build_config, source=source, hub=hub)

    with Ledger(
        tmp_path / "scratch" / "ledger.sqlite", reset_processing=False
    ) as ledger:
        assert ledger.programme("p1-programme-1").state.value == "rejected"
    assert report.rejection_counts["duration_out_of_range"] == 1
    assert hub.commits == [("README.md", ".gitattributes", "LICENSE")]
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
from hviske.p1_pipeline import PipelineSettings, initialise_target, run_pipeline
from tests.test_build_p1_segments import FakeSource, config
from tests.test_p1_publish import MemoryHub

root = Path(sys.argv[1])
hub = MemoryHub()
initialise_target(
    hub=hub,
    settings=PipelineSettings.from_config(config(root, mode="build")),
)
original = Ledger.allocate_batch_with_shards

def crash(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    os._exit(91)

Ledger.allocate_batch_with_shards = crash
run_pipeline(
    config=config(root, mode="build"),
    source=FakeSource(),
    hub=hub,
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
from hviske.p1_pipeline import PipelineSettings, initialise_target, run_pipeline
from tests.test_build_p1_segments import FakeSource, config
from tests.test_p1_publish import MemoryHub

root = Path(sys.argv[1])
hub = MemoryHub()
initialise_target(
    hub=hub,
    settings=PipelineSettings.from_config(config(root, mode="build")),
)
run_pipeline(
    config=config(root, mode="build"),
    source=FakeSource(),
    hub=hub,
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
        hub=prepared_hub(tmp_path),
    )
    recovery_manifest = (
        tmp_path / "recovery" / "scratch" / "audit-candidates.jsonl"
    ).read_text()
    control_manifest = (control_root / "scratch" / "audit-candidates.jsonl").read_text()
    assert recovery_manifest

    def stable_audit_fields(manifest: str) -> list[dict[str, object]]:
        return [
            {
                key: value
                for key, value in json.loads(line).items()
                if key not in {"audio_sha256", "metadata_sha256", "parquet_sha256"}
            }
            for line in manifest.splitlines()
        ]

    assert sorted(stable_audit_fields(recovery_manifest), key=str) == sorted(
        stable_audit_fields(control_manifest), key=str
    )


def test_unlimited_production_counts_consumed_candidates_without_ids(
    tmp_path: Path,
) -> None:
    """Streaming production reports each consumed candidate without source IDs."""
    source = FakeSource()
    production_config = config(tmp_path, mode="production")
    production_config.programme_limit = None
    report = run_pipeline(
        config=production_config, source=source, hub=prepared_hub(tmp_path)
    )

    payload = report.as_dict()
    dataclass_payload = dataclasses.asdict(report)
    representations = (repr(report), repr(dataclass_payload), repr(payload))
    preflight_payload = cast(dict[str, object], payload["preflight"])

    assert report.processed == 1
    assert report.selected_programmes == 1
    assert report.preflight.selected_programmes == 1
    assert payload["selected_programmes"] == 1
    assert preflight_payload["selected_programmes"] == 1
    assert dataclass_payload["selected_programmes"] == 1
    assert dataclass_payload["preflight"]["selected_programmes"] == 1
    assert all("programme-1" not in value for value in representations)


def test_verification_failure_retains_local_shard(tmp_path: Path) -> None:
    """A failed remote verification leaves the ledger batch and bytes recoverable."""
    source = FakeSource()
    with pytest.raises(RuntimeError, match="simulated verification failure"):
        run_pipeline(
            config=config(tmp_path, mode="build"),
            source=source,
            hub=prepared_verify_hub(tmp_path),
        )

    local_shards = list((tmp_path / "scratch" / "staging").rglob("*.parquet"))
    assert local_shards
    with Ledger(
        tmp_path / "scratch" / "ledger.sqlite", reset_processing=False
    ) as ledger:
        programme = ledger.programme("p1-programme-1")
        assert programme.state.value == "sharded"
        assert programme.last_error is None
        pending = ledger.pending_batches()
        assert len(pending) == 1
        batch = pending[0]
        assert batch.state.value == "committed"
        assert batch.commit_id is not None
        assert batch.last_error is None
        shards = ledger.shards(batch.batch_id)
        assert len(shards) == 1
        shard = shards[0]
        assert shard.state.value == "sharded"
        assert shard.verification_time is None
        assert shard.purge_time is None
        assert shard.local_path is not None
        assert shard.local_path == str(local_shards[0].resolve())
        assert Path(shard.local_path).is_file()


def prepared_verify_hub(tmp_path: Path) -> VerifyFailHub:
    """Return an initialised target whose later verification fails."""
    hub = MemoryHub()
    settings = PipelineSettings.from_config(config(tmp_path, mode="build"))
    initialise_target(hub=hub, settings=settings)
    hub.__class__ = VerifyFailHub
    return cast(VerifyFailHub, hub)
