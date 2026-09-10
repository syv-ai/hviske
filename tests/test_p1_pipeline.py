"""Offline tests for the reusable P1 orchestration layer."""

from __future__ import annotations

import dataclasses
import io
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from hviske.p1_ledger import Ledger
from hviske.p1_pipeline import (
    BuildReport,
    MetadataLog,
    P1PreflightError,
    PipelineSettings,
    PreflightReport,
    _native_candidates,
    _process_native_programmes,
    _unlink_recovered,
    preflight_pipeline,
    run_pipeline,
)
from hviske.p1_segments import (
    CTCBackend,
    SegmentationResult,
    ShardBatchResult,
    VADBackend,
)
from hviske.p1_source import SourcePlan, SourceShard, harden_p1_logging
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


def test_pipeline_hardens_hydra_root_and_file_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Transport records cannot leak signed URLs through Hydra-style handlers."""
    root_logger = logging.getLogger()
    file_path = tmp_path / "p1.log"
    file_handler = logging.FileHandler(file_path)
    late_stream = io.StringIO()
    late_handler = logging.StreamHandler(late_stream)
    root_logger.addHandler(file_handler)
    transport_names = ("httpx", "httpcore", "huggingface_hub", "fsspec")
    transport_loggers = tuple(logging.getLogger(name) for name in transport_names)
    previous_levels = tuple(item.level for item in transport_loggers)
    transport_logger = transport_loggers[0]
    signed_url = (
        "https://cas-server.xethub.hf.co/reconstruction/object"
        "?X-Xet-Cas-Uid=cas-uid&Policy=policy-value&Signature=signature-value"
        "&X-Amz-Signature=aws-signature-value&token=token-value"
    )
    levels_hardened = False
    try:
        with caplog.at_level(logging.INFO, logger="hviske.p1_pipeline"):
            run_pipeline(config=pipeline_config(tmp_path), source=MetadataSource())
            levels_hardened = all(
                item.level == logging.WARNING for item in transport_loggers
            )
            transport_logger.warning(
                'HTTP Request: GET %s "HTTP/1.1 200 OK"', signed_url
            )
            logging.getLogger("hviske.p1_pipeline").info(
                "P1 metadata event retained: source revision checked"
            )
            root_logger.addHandler(late_handler)
            harden_p1_logging()
            transport_logger.warning("Xet transport retry: %s", signed_url)
        file_handler.flush()
        late_handler.flush()
        log_output = caplog.text + file_path.read_text() + late_stream.getvalue()
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.removeHandler(late_handler)
        file_handler.close()
        late_handler.close()
        for item, level in zip(transport_loggers, previous_levels):
            item.setLevel(level)

    assert levels_hardened
    assert not _contains_signed_url_material(log_output)
    assert _contains_log_text(log_output, "P1 metadata event retained")


def _contains_log_text(value: str, expected: str) -> bool:
    """Return whether a safe metadata event survived transport hardening."""
    return expected in value


def _contains_signed_url_material(value: str) -> bool:
    """Return whether a captured log contains any fixture credential material."""
    return any(
        marker in value
        for marker in (
            "X-Xet-Cas-Uid",
            "Policy",
            "Signature",
            "X-Amz-Signature",
            "cas-uid",
            "policy-value",
            "signature-value",
            "aws-signature-value",
            "token-value",
            "?X-Xet-Cas-Uid=",
        )
    )


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


def test_recovery_purges_only_matching_survivors(tmp_path: Path) -> None:
    """Partial unlink recovery tolerates missing files and protects replacements."""
    good = tmp_path / "good.parquet"
    replaced = tmp_path / "replaced.parquet"
    good.write_bytes(b"good")
    replaced.write_bytes(b"new content")

    _unlink_recovered(
        (good, replaced, tmp_path / "missing.parquet"),
        expected={
            good: (
                4,
                "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c",
            ),
            replaced: (3, "0" * 64),
        },
    )

    assert not good.exists()
    assert replaced.exists()


def test_zero_accepted_programme_is_skipped_on_the_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty proposal result is terminal and is not processed twice."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))
    source_calls = 0

    class Source:
        last_temporary = None

        def fetch_audio(self, *, pointer: object) -> object:
            del pointer
            return type("Audio", (), {"sampling_rate": 16_000, "channels": 1})()

        def fetch_transcript(self, _pointer: object) -> object:
            nonlocal source_calls
            source_calls += 1
            return type("Transcript", (), {"words": (), "text": ""})()

        def iter_programme_pointers(self, *, shard: object) -> object:
            del shard
            return iter(
                (
                    type(
                        "Pointer",
                        (),
                        {"file_id": "file-1", "row_group": 0, "row_index": 0},
                    )(),
                )
            )

    monkeypatch.setattr(
        "hviske.p1_pipeline._decoded_native_audio",
        lambda _audio, file_id: np.zeros(16_000),
    )
    monkeypatch.setattr(
        "hviske.p1_pipeline.segment_programme",
        lambda **_: SegmentationResult(rows=(), rejections=(), correction_count=0),
    )
    monkeypatch.setattr(
        "hviske.p1_pipeline.write_shards",
        lambda *_, **__: ShardBatchResult(shards=(), source_recoverable=False),
    )
    preflight = PreflightReport(
        mode="build",
        selected_programmes=1,
        maximum_source_bytes=0,
        required_scratch_bytes=0,
        free_bytes=1,
        scratch_bytes=0,
        source_revisions={},
        model_revisions={},
        cuda={},
        target={},
        checks={},
    )
    report = BuildReport(preflight=preflight, selected_file_ids=("file-1",))
    source_shard = type(
        "Shard", (), {"path": "source/part.parquet", "byte_size": 1_000}
    )()
    candidate = [("file-1", {"duration_ms": 1_000}, (source_shard, object()))]
    database = tmp_path / "ledger.sqlite"
    with Ledger(database) as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=candidate,
            ledger=ledger,
            hub=object(),
            vad=cast(VADBackend, object()),
            ctc=cast(CTCBackend, object()),
            report=report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
        record = ledger.programme("p1-file-1")
        assert (record.state.value, record.last_error) == (
            "rejected",
            "no_accepted_segments",
        )
    with Ledger(database) as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=candidate,
            ledger=ledger,
            hub=object(),
            vad=cast(VADBackend, object()),
            ctc=cast(CTCBackend, object()),
            report=report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
    assert source_calls == 1
    assert report.rejected == 1
