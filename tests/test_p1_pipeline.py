"""Offline tests for the reusable P1 orchestration layer."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from hviske.p1_contracts import SourceWord
from hviske.p1_ledger import Ledger
from hviske.p1_pipeline import (
    BuildReport,
    MetadataLog,
    NativeCandidate,
    P1PreflightError,
    PipelineSettings,
    PreflightReport,
    _native_candidates,
    _process_native_programmes,
    _SelectionDedup,
    _unlink_recovered,
    enforce_scratch_cap,
    preflight_pipeline,
    run_pipeline,
)
from hviske.p1_segments import (
    CTCBackend,
    SegmentationResult,
    ShardBatchResult,
    VADBackend,
)
from hviske.p1_source import (
    AudioPointer,
    InvalidSourceTimestamp,
    ParsedAudio,
    ParsedTranscript,
    SourcePlan,
    SourceShard,
    TranscriptPointer,
    harden_p1_logging,
    parse_transcript_row,
)
from hviske.p1_validation import stratified_sample
from tests.test_p1_publish import MemoryHub


@pytest.mark.parametrize(
    "iterator_name", ["iter_programme_metadata", "iter_programme_pointers"]
)
def test_audio_scan_progress_reports_threshold_crossings(
    iterator_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Large shards report every crossed progress threshold in either scan path."""
    monkeypatch.setattr("hviske.p1_pipeline._PROGRESS_INTERVAL", 2)
    rows = tuple({"file_id": f"file-{index}"} for index in range(5))

    class Source:
        def iter_programme_metadata(self, **_: object) -> object:
            yield from rows

        def iter_programme_pointers(self, *, shard: object) -> object:
            del shard
            return iter(
                type(
                    "Pointer",
                    (),
                    {
                        "file_id": row["file_id"],
                        "shard": SourceShard("audio.parquet", 10),
                        "row_group": 0,
                        "row_index": index,
                    },
                )()
                for index, row in enumerate(rows)
            )

    class Index:
        def get(self, _file_id: str) -> object:
            return object()

    source = Source()
    if iterator_name == "iter_programme_metadata":
        setattr(source, "iter_programme_pointers", None)
    else:
        setattr(source, "iter_programme_metadata", None)

    with caplog.at_level(logging.INFO, logger="hviske.p1_pipeline"):
        candidates = _native_candidates(
            source=source,
            shards=(SourceShard("audio.parquet", 10),),
            index=Index(),
            programme_limit=None,
            source_file_id=None,
            pilot=False,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
        assert len(list(candidates)) == len(rows)

    progress = [
        record.getMessage()
        for record in caplog.records
        if "Audio metadata scan progress" in record.getMessage()
    ]
    assert progress == [
        "Audio metadata scan progress: 2 rows across 1 shards",
        "Audio metadata scan progress: 4 rows across 1 shards",
    ]


def test_build_report_serialises_only_selected_programme_count() -> None:
    """Report representations never expose selected source identifiers."""
    report = _pipeline_test_report(selected_programmes=1)
    payload = report.as_dict()
    dataclass_payload = dataclasses.asdict(report)
    representations = (repr(report), repr(dataclass_payload), repr(payload))

    assert report.selected_programmes == 1
    assert payload["selected_programmes"] == 1
    assert dataclass_payload["selected_programmes"] == 1
    assert "selected_file_ids" not in payload
    assert "selected_file_ids" not in dataclass_payload
    assert all("selected_file_ids" not in value for value in representations)
    assert all("file-1" not in value for value in representations)
    assert "file-1" not in json.dumps(payload)


def _pipeline_test_report(selected_programmes: int = 0) -> BuildReport:
    """Return a report suitable for direct native-programme tests."""
    preflight = PreflightReport(
        mode="build",
        selected_programmes=selected_programmes,
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
    return BuildReport(preflight=preflight, selected_programmes=selected_programmes)


def test_empty_timed_words_precede_ambiguous_source_text(tmp_path: Path) -> None:
    """Lexical text with no timed words stops before ambiguity classification."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))
    candidate = _pipeline_test_candidate()

    class Source:
        last_temporary = None

        def fetch_audio(self, *, pointer: object) -> object:
            del pointer
            raise AssertionError("untimed transcript must not retrieve audio")

        def fetch_transcript(self, _pointer: object) -> ParsedTranscript:
            return ParsedTranscript(
                file_id="file-1",
                text="lexical text",
                words=(),
                ambiguous_source_text_records=1,
            )

    report = _pipeline_test_report()
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=candidate,
            ledger=ledger,
            hub=object(),
            vad=None,
            ctc=None,
            report=report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
        record = ledger.programme("p1-file-1")

    assert record.last_error == "no_timed_words"
    assert report.rejection_counts == {"no_timed_words": 1}


def _pipeline_test_candidate() -> list[tuple[str, object, object]]:
    """Return one metadata-only candidate for direct native-programme tests."""
    shard = type("Shard", (), {"path": "source/part.parquet", "byte_size": 1_000})()
    return [("file-1", {"duration_ms": 1_000}, (shard, object()))]


def pipeline_config(tmp_path: Path, mode: str = "plan") -> DictConfig:
    """Return a small test-owned pipeline configuration."""
    config = OmegaConf.load("config/p1_segments.yaml")
    config.mode = mode
    config.runtime.scratch_root = str(tmp_path / "scratch")
    config.runtime.device = "cpu"
    return cast(DictConfig, config)


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


def test_invalid_transcript_never_constructs_models_or_retrieves_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal transcript defects stop before audio and model work."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))
    shard = SourceShard("audio.parquet", 10)
    candidate = NativeCandidate(
        file_id="file-1",
        metadata={"duration_ms": 1_000},
        audio_pointer=AudioPointer("file-1", shard, 0, 0),
        transcript_pointer=cast(TranscriptPointer, object()),
    )
    calls = {"audio": 0, "vad": 0, "ctc": 0}

    class Source:
        last_temporary = None

        def fetch_audio(self, *, pointer: object) -> object:
            del pointer
            calls["audio"] += 1
            raise AssertionError("invalid transcript must not retrieve audio")

        def fetch_transcript(self, _pointer: object) -> ParsedTranscript:
            return ParsedTranscript(file_id="file-1", text="  ", words=())

    def make_vad(_settings: PipelineSettings) -> object:
        calls["vad"] += 1
        return object()

    def make_ctc(_settings: PipelineSettings) -> object:
        calls["ctc"] += 1
        return object()

    monkeypatch.setattr("hviske.p1_pipeline.make_silero_vad", make_vad)
    monkeypatch.setattr("hviske.p1_pipeline.make_ctc_backend", make_ctc)
    report = _pipeline_test_report()
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=[candidate],
            ledger=ledger,
            hub=object(),
            vad=None,
            ctc=None,
            report=report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
        assert ledger.programme("p1-file-1").last_error == "empty_text"
    assert calls == {"audio": 0, "vad": 0, "ctc": 0}


def test_missing_transcripts_are_aggregated_without_ids(tmp_path: Path) -> None:
    """Missing transcript evidence contains only an aggregate count."""

    class Source:
        def iter_programme_metadata(self, **_: object) -> object:
            yield from ({"file_id": "missing-a"}, {"file_id": "missing-b"})

    class Index:
        def get(self, _: str) -> None:
            return None

    events = tmp_path / "events.jsonl"
    candidates = _native_candidates(
        source=Source(),
        shards=(SourceShard("audio.parquet", 10),),
        index=Index(),
        programme_limit=None,
        source_file_id=None,
        pilot=False,
        log=MetadataLog(events),
    )
    assert list(candidates) == []
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert records == [
        {
            "count": 2,
            "event": "programme_rejection_summary",
            "reason": "missing_transcript",
        }
    ]
    assert "missing-a" not in events.read_text()
    assert "missing-b" not in events.read_text()


def test_native_candidates_retain_discovered_audio_pointers(tmp_path: Path) -> None:
    """Selection joins transcript pointers without a later audio rescan."""

    class Source:
        def iter_programme_metadata(self, **_: object) -> object:
            raise AssertionError("pointer discovery should be the metadata scan")

        def iter_programme_pointers(self, *, shard: SourceShard) -> object:
            yield AudioPointer("wanted", shard, 3, 7, (("duration_ms", "1000"),))

    class Index:
        def get(self, file_id: str) -> object:
            return object() if file_id == "wanted" else None

    shard = SourceShard("audio.parquet", 10)
    candidates = list(
        _native_candidates(
            source=Source(),
            shards=(shard,),
            index=Index(),
            programme_limit=None,
            source_file_id=None,
            pilot=False,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert isinstance(candidate, NativeCandidate)
    assert candidate.audio_pointer == AudioPointer(
        "wanted", shard, 3, 7, (("duration_ms", "1000"),)
    )
    assert candidate.metadata["duration_ms"] == 1000


def test_native_selection_dedup_is_disk_backed_and_rebuilt(tmp_path: Path) -> None:
    """Deduplication keeps the first row without retaining IDs in Python."""

    class Source:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        def iter_programme_metadata(self, **_: object) -> object:
            yield from self.rows

    class Index:
        def get(self, _: str) -> object:
            return object()

    shard = SourceShard("data/audio.parquet", 10)
    source = Source(
        [
            {"file_id": "programme-1", "duration_ms": 1_000},
            {"file_id": "programme-1", "duration_ms": 9_000},
        ]
    )
    first = list(
        _native_candidates(
            source=source,
            shards=(shard,),
            index=Index(),
            programme_limit=None,
            source_file_id=None,
            pilot=False,
            log=MetadataLog(tmp_path / "events.jsonl"),
            scratch_root=tmp_path,
        )
    )
    assert [candidate.metadata["duration_ms"] for candidate in first] == [1_000]

    source.rows = [{"file_id": "programme-2", "duration_ms": 2_000}]
    second = list(
        _native_candidates(
            source=source,
            shards=(shard,),
            index=Index(),
            programme_limit=None,
            source_file_id=None,
            pilot=False,
            log=MetadataLog(tmp_path / "events.jsonl"),
            scratch_root=tmp_path,
        )
    )
    assert [candidate.file_id for candidate in second] == ["programme-2"]
    with sqlite3.connect(tmp_path / "native-selection-dedup.sqlite") as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(selected_ids)")
        ]
    assert columns == ["file_id"]


def test_overlong_transcript_is_a_terminal_invalid_timestamp_rejection(
    tmp_path: Path,
) -> None:
    """Transcript words beyond audio are rejected before programme construction."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))

    class Source:
        last_temporary = None

        def fetch_transcript(self, _pointer: object) -> ParsedTranscript:
            return ParsedTranscript(
                file_id="file-1",
                text="too long",
                words=(SourceWord(text="too", start_ms=0, end_ms=1_001),),
            )

    database = tmp_path / "ledger.sqlite"
    report = _pipeline_test_report()
    with Ledger(database) as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=_pipeline_test_candidate(),
            ledger=ledger,
            hub=object(),
            vad=cast(VADBackend, object()),
            ctc=cast(CTCBackend, object()),
            report=report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
        record = ledger.programme("p1-file-1")
        assert record.state.value == "rejected"
        assert record.last_error == "invalid_timestamps"
        assert record.rejection_counts == {"invalid_timestamps": 1}
    assert report.processed == 1
    assert report.rejection_counts == {"invalid_timestamps": 1}


def test_parser_timestamp_failure_is_invalid_timestamp_rejection(
    tmp_path: Path,
) -> None:
    """A parser timestamp exception is not swallowed by its parent class."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))
    candidate = _pipeline_test_candidate()

    class Source:
        last_temporary = None

        def fetch_audio(self, *, pointer: object) -> object:
            del pointer
            raise AssertionError("invalid transcript must not retrieve audio")

        def fetch_transcript(self, _pointer: object) -> ParsedTranscript:
            raise InvalidSourceTimestamp("invalid source timestamp")

    report = _pipeline_test_report()
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=candidate,
            ledger=ledger,
            hub=object(),
            vad=None,
            ctc=None,
            report=report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
        record = ledger.programme("p1-file-1")

    assert record.last_error == "invalid_timestamps"
    assert report.rejection_counts == {"invalid_timestamps": 1}


def test_pilot_scans_each_shard_once_and_recovers_exact_pointers(
    tmp_path: Path,
) -> None:
    """Pilot selection keeps bounded locators from its single metadata pass."""
    shards = (SourceShard("audio-a.parquet", 10), SourceShard("audio-b.parquet", 20))
    pointers = tuple(
        AudioPointer(
            f"programme-{index}",
            shards[index // 4],
            index % 4,
            index + 10,
            (("duration_ms", json.dumps(1_000 + index)),),
        )
        for index in range(8)
    )

    class Source:
        def __init__(self) -> None:
            self.shards_opened: list[str] = []

        def iter_programme_metadata(self, **_: object) -> object:
            raise AssertionError("pilot must not make a second metadata pass")

        def iter_programme_pointers(self, *, shard: SourceShard) -> object:
            self.shards_opened.append(shard.path)
            return iter(pointer for pointer in pointers if pointer.shard == shard)

    class Index:
        def get(self, _: str) -> object:
            return object()

    source = Source()
    expected = stratified_sample(
        [
            {"id": pointer.file_id, "duration_ms": 1_000 + index}
            for index, pointer in enumerate(pointers)
        ],
        sample_size=3,
        seed="p1-pilot",
    )
    candidates = _native_candidates(
        source=source,
        shards=shards,
        index=Index(),
        programme_limit=3,
        source_file_id=None,
        pilot=True,
        log=MetadataLog(tmp_path / "events.jsonl"),
    )

    assert source.shards_opened == [shard.path for shard in shards]
    assert [candidate.file_id for candidate in candidates] == sorted(
        str(row["id"]) for row in expected
    )
    recovered = {candidate.file_id: candidate.audio_pointer for candidate in candidates}
    assert all(
        recovered[pointer.file_id] == pointer
        for pointer in pointers
        if pointer.file_id in recovered
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


def test_pilot_selection_preserves_nonfinite_pointer_metadata(tmp_path: Path) -> None:
    """Pilot reconstruction preserves the encoded scalar tuple byte-for-byte."""
    shard = SourceShard("audio.parquet", 10)
    pointer = AudioPointer(
        "programme-1",
        shard,
        2,
        7,
        (("duration_ms", "NaN"), ("title", json.dumps("A", sort_keys=True))),
    )

    class Source:
        def iter_programme_pointers(self, *, shard: SourceShard) -> object:
            assert shard == pointer.shard
            yield pointer

    class Index:
        def get(self, _: str) -> object:
            return object()

    candidates = _native_candidates(
        source=Source(),
        shards=(shard,),
        index=Index(),
        programme_limit=1,
        source_file_id=None,
        pilot=True,
        log=MetadataLog(tmp_path / "events.jsonl"),
        scratch_root=tmp_path,
    )
    selected = list(candidates)
    assert selected[0].audio_pointer == pointer
    assert "_p1_audio_pointer_metadata" not in selected[0].metadata


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

    assert report.selected_programmes == 0
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


def test_progress_logs_do_not_include_source_identifiers_or_paths(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Operational progress reports counts, not source URLs or paths."""
    target = "https://source.invalid/audio.parquet?signature=secret"

    class Source:
        def iter_programme_metadata(self, **_: object) -> object:
            yield {"file_id": target}

    class Index:
        def get(self, file_id: str) -> object | None:
            return object() if file_id == target else None

    with caplog.at_level(logging.INFO, logger="hviske.p1_pipeline"):
        candidates = _native_candidates(
            source=Source(),
            shards=(SourceShard("private/audio.parquet", 10),),
            index=Index(),
            programme_limit=None,
            source_file_id=target,
            pilot=False,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
    assert len(list(candidates)) == 1
    progress = " ".join(record.getMessage() for record in caplog.records)
    assert "selection complete" in progress
    assert target not in progress
    assert "signature=secret" not in progress
    assert "private/audio.parquet" not in progress


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


def test_selection_dedup_rebuild_reclaims_high_water_file(tmp_path: Path) -> None:
    """A fresh selection database does not retain the previous high water mark."""
    path = tmp_path / "selection.sqlite"
    dedup = _SelectionDedup(path)
    for number in range(2_000):
        assert dedup.add_if_new(f"programme-{number}")
    dedup.close()
    grown_size = path.stat().st_size

    rebuilt = _SelectionDedup(path)
    rebuilt.close()

    assert path.stat().st_size < grown_size


def test_selection_quota_guard_aborts_before_audio_or_models(tmp_path: Path) -> None:
    """A tiny quota stops selection before a candidate can reach payload work."""
    config = pipeline_config(tmp_path, mode="build")
    settings = dataclasses.replace(
        PipelineSettings.from_config(config), max_scratch_bytes=0
    )
    calls = {"audio": 0, "model": 0}

    class Source:
        def fetch_audio(self, **_: object) -> object:
            calls["audio"] += 1
            raise AssertionError("quota failure must precede audio retrieval")

        def iter_programme_metadata(self, **_: object) -> object:
            yield {"file_id": "programme-1", "duration_ms": 1_000}

        def load_model(self) -> object:
            calls["model"] += 1
            raise AssertionError("quota failure must precede model construction")

    class Index:
        def get(self, _: str) -> object:
            return object()

    def guard() -> None:
        enforce_scratch_cap(settings)

    with pytest.raises(P1PreflightError, match="scratch hard cap exceeded"):
        list(
            _native_candidates(
                source=Source(),
                shards=(SourceShard("audio.parquet", 10),),
                index=Index(),
                programme_limit=None,
                source_file_id=None,
                pilot=False,
                log=MetadataLog(tmp_path / "events.jsonl"),
                scratch_root=settings.scratch_root,
                scratch_guard=guard,
            )
        )
    assert calls == {"audio": 0, "model": 0}


def test_targeted_selection_stops_after_matching_audio_metadata(tmp_path: Path) -> None:
    """A requested programme is found without scanning later source rows."""

    class Source:
        def __init__(self) -> None:
            self.rows_seen = 0

        def iter_programme_metadata(self, **_: object) -> object:
            for file_id in ("before", "wanted", "after"):
                self.rows_seen += 1
                yield {"file_id": file_id, "duration_ms": 1}

    class Index:
        def get(self, file_id: str) -> object | None:
            return object() if file_id == "wanted" else None

    source = Source()
    candidates = _native_candidates(
        source=source,
        shards=(SourceShard("audio.parquet", 10),),
        index=Index(),
        programme_limit=None,
        source_file_id="wanted",
        pilot=False,
        log=MetadataLog(tmp_path / "events.jsonl"),
    )
    assert [candidate[0] for candidate in candidates] == ["wanted"]
    assert source.rows_seen == 2


def test_unexpected_native_failure_is_retryable_and_aborts_without_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Unexpected failures abort the run and persist only a safe error category."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))

    class Source:
        last_temporary = None

        def fetch_audio(self, *, pointer: object) -> object:
            del pointer
            raise RuntimeError("secret transcript payload must not leak")

        def fetch_transcript(self, _pointer: object) -> ParsedTranscript:
            return ParsedTranscript(
                file_id="file-1",
                text="secret transcript",
                words=(SourceWord(text="secret", start_ms=0, end_ms=100),),
            )

        def iter_programme_pointers(self, *, shard: object) -> object:
            del shard
            return iter((type("Pointer", (), {"file_id": "file-1"})(),))

    database = tmp_path / "ledger.sqlite"
    events = tmp_path / "events.jsonl"
    report = _pipeline_test_report()
    with caplog.at_level(logging.INFO, logger="hviske.p1_pipeline"):
        with pytest.raises(RuntimeError):
            with Ledger(database) as ledger:
                _process_native_programmes(
                    source=Source(),
                    settings=settings,
                    candidates=_pipeline_test_candidate(),
                    ledger=ledger,
                    hub=object(),
                    vad=cast(VADBackend, object()),
                    ctc=cast(CTCBackend, object()),
                    report=report,
                    log=MetadataLog(events),
                )
    with Ledger(database) as ledger:
        record = ledger.programme("p1-file-1")
        assert record.state.value == "retryable"
        assert record.last_error == "runtime_error"
    assert report.selected_programmes == 1
    assert report.processed == 0
    event_text = events.read_text(encoding="utf-8")
    assert "secret transcript" not in event_text
    assert "runtime_error" in event_text
    assert "secret transcript" not in caplog.text
    assert "runtime_error" in caplog.text


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
        selected_programmes=0,
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
    report = BuildReport(preflight=preflight, selected_programmes=0)
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
    assert report.selected_programmes == 1
    assert report.processed == 1

    second_report = BuildReport(preflight=preflight, selected_programmes=0)
    with Ledger(database) as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=candidate,
            ledger=ledger,
            hub=object(),
            vad=cast(VADBackend, object()),
            ctc=cast(CTCBackend, object()),
            report=second_report,
            log=MetadataLog(tmp_path / "events.jsonl"),
        )
    assert source_calls == 1
    assert second_report.selected_programmes == 1
    assert second_report.processed == 0
    assert report.rejected == 1


def test_zero_duration_normalisation_is_reported_as_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline reports omission counts without recording token content."""
    settings = PipelineSettings.from_config(pipeline_config(tmp_path, mode="build"))

    class Source:
        last_temporary = None

        def fetch_audio(self, *, pointer: object) -> ParsedAudio:
            del pointer
            return ParsedAudio(
                file_id="file-1", value=np.zeros(16_000), sampling_rate=16_000
            )

        def fetch_transcript(self, _pointer: object) -> ParsedTranscript:
            return parse_transcript_row(
                row={
                    "file_id": "file-1",
                    "words": [
                        {"text": "a", "start_ms": 0, "end_ms": 100},
                        {"text": "<hidden>", "start_ms": 100, "end_ms": 100},
                        {"text": "b", "start_ms": 100, "end_ms": 200},
                    ],
                }
            )

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
        "hviske.p1_pipeline.segment_programme",
        lambda **_: SegmentationResult(rows=(), rejections=(), correction_count=0),
    )
    monkeypatch.setattr(
        "hviske.p1_pipeline.write_shards",
        lambda *_, **__: ShardBatchResult(shards=(), source_recoverable=False),
    )
    events = tmp_path / "events.jsonl"
    report = _pipeline_test_report()
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        _process_native_programmes(
            source=Source(),
            settings=settings,
            candidates=_pipeline_test_candidate(),
            ledger=ledger,
            hub=object(),
            vad=cast(VADBackend, object()),
            ctc=cast(CTCBackend, object()),
            report=report,
            log=MetadataLog(events),
        )
    assert report.normalization_counts == {"zero_duration_tokens_omitted": 1}
    assert report.as_dict()["normalization_counts"] == {
        "zero_duration_tokens_omitted": 1
    }
    event_text = events.read_text(encoding="utf-8")
    assert "transcript_normalized" in event_text
    assert "<hidden>" not in event_text
