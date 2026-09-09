"""Reusable orchestration for the bounded Phase 1 P1 segmentation dataset.

Planning deliberately uses only Hub repository/tree metadata.  Build mode then uses a
disk-backed transcript pointer index and retrieves one selected programme at a time.
"""

from __future__ import annotations

import collections.abc as c
import dataclasses
import gc
import importlib
import json
import os
import shutil
import time
import typing as t
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from hviske.p1_contracts import (
    CanonicalIdentityManifest,
    CTCContract,
    LedgerState,
    ModelContract,
    NormalisationContract,
    OutputEncodingContract,
    RejectionCategory,
    RepositoryRevision,
    SegmentationContract,
    SourceCoordinates,
    SourceProgramme,
    SourceWord,
    VADContract,
    pipeline_config_sha256,
)
from hviske.p1_ledger import Ledger
from hviske.p1_segments import CTCBackend, VADBackend, segment_programme, write_shards
from hviske.p1_source import SourceShard


@dataclass(frozen=True)
class PipelineSettings:
    """Resolved Hydra configuration and derived P1 identity."""

    mode: str
    pipeline_version: str
    scratch_root: Path
    max_source_bytes: int
    max_scratch_bytes: int
    shards_per_commit: int
    target_shard_bytes: int
    workers: int
    queue_slots: int
    upload_concurrency: int
    verification_concurrency: int
    programme_limit: int | None
    source_file_id: str | None
    resume: bool
    device: str
    target_private_repo: str
    source_audio_repository: str
    source_audio_revision: str
    source_transcript_repository: str
    source_transcript_revision: str
    ctc_model_repository: str
    ctc_model_revision: str
    anomaly_model_repository: str
    anomaly_model_revision: str
    model_revisions: dict[str, object]
    segmentation: SegmentationContract
    normalisation: NormalisationContract
    pipeline_digest: str

    @classmethod
    def from_config(cls, config: DictConfig) -> PipelineSettings:
        """Resolve Hydra values and build the complete canonical identity.

        Returns:
            Resolved pipeline settings with a canonical digest.
        """
        config_object = (
            config if isinstance(config, DictConfig) else OmegaConf.create(config)
        )
        raw = OmegaConf.to_container(config_object, resolve=True)
        root = t.cast(dict[str, object], raw)
        source = t.cast(dict[str, object], root["source"])
        audio = t.cast(dict[str, object], source["audio"])
        transcripts = t.cast(dict[str, object], source["transcripts"])
        runtime = t.cast(dict[str, object], root["runtime"])
        max_source = root.get("max_source_bytes") or runtime["max_source_bytes"]
        max_scratch = root.get("max_scratch_bytes") or runtime["max_scratch_bytes"]
        shard_limit = root.get("shards_per_commit") or runtime["shards_per_commit"]
        vad_raw = t.cast(dict[str, object], root["vad"])
        vad_repo = t.cast(dict[str, object], vad_raw["repository"])
        ctc_raw = t.cast(dict[str, object], root["ctc"])
        ctc_repo = t.cast(
            dict[str, object], t.cast(dict[str, object], ctc_raw["model"])["repository"]
        )
        anomaly_raw = t.cast(dict[str, object], root["anomaly_model"])
        output_raw = t.cast(dict[str, object], root["output"])
        manifest = CanonicalIdentityManifest(
            schema_version=str(root["schema_version"]),
            pipeline_version=str(root["pipeline_version"]),
            source=SourceCoordinates.model_validate(source),
            vad=VADContract(
                name=str(vad_raw["name"]),
                repository=RepositoryRevision.model_validate(vad_repo),
                model_blob=str(vad_raw["model_blob"]),
                license=str(vad_raw["license"]),
            ),
            ctc=CTCContract(
                name=str(ctc_raw["name"]),
                version=str(ctc_raw["version"]),
                source_commit=str(ctc_raw["source_commit"]),
                sdist_sha256=str(ctc_raw["sdist_sha256"]),
                license=str(ctc_raw["license"]),
                model=ModelContract(
                    repository=RepositoryRevision.model_validate(ctc_repo),
                    license=str(t.cast(dict[str, object], ctc_raw["model"])["license"]),
                ),
            ),
            anomaly_model=ModelContract(
                repository=RepositoryRevision.model_validate(anomaly_raw["repository"]),
                license=str(anomaly_raw["license"]),
            ),
            normalisation=NormalisationContract.model_validate(root["normalisation"]),
            segmentation=SegmentationContract.model_validate(root["segmentation"]),
            output=OutputEncodingContract.model_validate(output_raw),
        )
        digest = pipeline_config_sha256(manifest)
        return cls(
            mode=str(root.get("mode", "build")),
            pipeline_version=str(root["pipeline_version"]),
            scratch_root=Path(str(runtime["scratch_root"])).expanduser().resolve(),
            max_source_bytes=_as_int(max_source),
            max_scratch_bytes=_as_int(max_scratch),
            shards_per_commit=_as_int(shard_limit),
            target_shard_bytes=_as_int(runtime["target_shard_bytes"]),
            workers=_as_int(runtime.get("workers", 1)),
            queue_slots=_as_int(runtime.get("queue_slots", 1)),
            upload_concurrency=_as_int(runtime.get("upload_concurrency", 1)),
            verification_concurrency=_as_int(
                runtime.get("verification_concurrency", 1)
            ),
            programme_limit=_optional_int(root.get("programme_limit")),
            source_file_id=_optional_str(root.get("source_file_id")),
            resume=bool(root.get("resume", True)),
            device=str(runtime.get("device", "cuda:0")),
            target_private_repo=str(runtime["target_private_repo"]),
            source_audio_repository=str(audio["repository"]),
            source_audio_revision=str(audio["revision"]),
            source_transcript_repository=str(transcripts["repository"]),
            source_transcript_revision=str(transcripts["revision"]),
            ctc_model_repository=str(ctc_repo["repository"]),
            ctc_model_revision=str(ctc_repo["revision"]),
            anomaly_model_repository=str(
                t.cast(dict[str, object], anomaly_raw["repository"])["repository"]
            ),
            anomaly_model_revision=str(
                t.cast(dict[str, object], anomaly_raw["repository"])["revision"]
            ),
            model_revisions={
                "vad": vad_repo,
                "ctc": ctc_repo,
                "anomaly": anomaly_raw["repository"],
            },
            segmentation=manifest.segmentation,
            normalisation=manifest.normalisation,
            pipeline_digest=digest,
        )


def _as_int(value: object, default: int = 0) -> int:
    """Convert a numeric dataset/config value without weakening type checks.

    Returns:
        An integer value, or ``default`` for a non-numeric object.
    """
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _optional_int(value: object) -> int | None:
    return None if value is None else _as_int(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _unlink_recovered(paths: tuple[Path, ...]) -> None:
    """Remove only digest-matched shard paths supplied by recovery."""
    for path in paths:
        if path.is_file() and not path.is_symlink():
            path.unlink()


@dataclass(frozen=True)
class PreflightReport:
    """JSON-safe evidence from the checks performed before audio retrieval."""

    mode: str
    selected_programmes: int
    maximum_source_bytes: int
    required_scratch_bytes: int
    free_bytes: int
    scratch_bytes: int
    source_revisions: dict[str, object]
    model_revisions: dict[str, object]
    cuda: dict[str, object]
    target: dict[str, object]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, object]:
        """Return the report as metadata suitable for JSONL logging."""
        return {
            "mode": self.mode,
            "selected_programmes": self.selected_programmes,
            "maximum_source_bytes": self.maximum_source_bytes,
            "required_scratch_bytes": self.required_scratch_bytes,
            "free_bytes": self.free_bytes,
            "scratch_bytes": self.scratch_bytes,
            "source_revisions": self.source_revisions,
            "model_revisions": self.model_revisions,
            "cuda": self.cuda,
            "target": self.target,
            "checks": self.checks,
        }


@dataclass
class BuildReport:
    """Bounded run counters and preflight evidence."""

    preflight: PreflightReport
    selected_file_ids: tuple[str, ...]
    processed: int = 0
    rejected: int = 0
    accepted_segments: int = 0
    shard_count: int = 0
    max_in_flight: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return metadata-only run evidence."""
        return {
            "preflight": self.preflight.as_dict(),
            "selected_file_ids": list(self.selected_file_ids),
            "processed": self.processed,
            "rejected": self.rejected,
            "accepted_segments": self.accepted_segments,
            "shard_count": self.shard_count,
            "max_in_flight": self.max_in_flight,
            "rejection_counts": self.rejection_counts,
        }


def run_pipeline(
    *,
    config: DictConfig,
    source: object | None = None,
    hub: object | None = None,
    vad: VADBackend | None = None,
    ctc: CTCBackend | None = None,
) -> BuildReport:
    """Execute the P1 pipeline, using metadata-only planning when available.

    The production source exposes ``plan``.  Keeping the small legacy dispatch is
    intentional: it preserves the injectable API used by older offline tests while
    all real Hub work follows the disk-backed pointer path below.

    Returns:
        Metadata-only or completed build evidence.
    """
    settings = PipelineSettings.from_config(config)
    if settings.mode == "initialise":
        configure_scratch(settings.scratch_root)
        if hub is None:
            hub = make_hub()
        initialise_target(hub=hub, settings=settings)
        return _initialise_report(settings=settings, hub=hub)
    if source is None:
        from hviske.p1_source import HfP1Source

        source = HfP1Source(
            audio_repository=settings.source_audio_repository,
            transcript_repository=settings.source_transcript_repository,
            max_source_object_bytes=settings.max_source_bytes,
        )
    if not hasattr(source, "plan"):
        return _run_legacy_pipeline(
            config=config,
            source=t.cast(SourceAdapter, source),
            hub=hub,
            vad=vad,
            ctc=ctc,
        )
    return _run_native_pipeline(config=config, source=source, hub=hub, vad=vad, ctc=ctc)


def _initialise_report(*, settings: PipelineSettings, hub: object) -> BuildReport:
    """Return metadata evidence after private-target initialisation."""
    required = calculate_scratch_requirement(
        settings=settings,
        maximum_source_bytes=0,
        shard_count=settings.shards_per_commit,
    )
    scratch = settings.scratch_root
    free_bytes = shutil.disk_usage(scratch).free
    scratch_bytes = directory_size(scratch)
    target = target_privacy(hub, settings.target_private_repo)
    preflight = PreflightReport(
        mode=settings.mode,
        selected_programmes=0,
        maximum_source_bytes=0,
        required_scratch_bytes=required,
        free_bytes=free_bytes,
        scratch_bytes=scratch_bytes,
        source_revisions={
            "audio": {
                "repository": settings.source_audio_repository,
                "revision": settings.source_audio_revision,
            },
            "transcripts": {
                "repository": settings.source_transcript_repository,
                "revision": settings.source_transcript_revision,
            },
        },
        model_revisions=settings.model_revisions,
        cuda=cuda_status(settings.device),
        target=target,
        checks={"target_checked": target["private"] is True},
    )
    return BuildReport(preflight=preflight, selected_file_ids=())


def calculate_scratch_requirement(
    *, settings: PipelineSettings, maximum_source_bytes: int, shard_count: int
) -> int:
    """Calculate the conservative active-worker, queue, shard, and safety budget.

    Returns:
        Required scratch bytes.
    """
    worker_slots = 1 + settings.queue_slots
    source_and_decode = maximum_source_bytes * worker_slots * 2
    open_and_pending_shards = settings.target_shard_bytes * max(1, shard_count)
    encoder_and_upload = settings.target_shard_bytes + maximum_source_bytes
    checksum_and_verification = (
        64
        * 1024
        * 1024
        * (settings.upload_concurrency + settings.verification_concurrency)
    )
    model_cache = 2 * 1024 * 1024 * 1024
    ledger_and_safety = 256 * 1024 * 1024
    return (
        source_and_decode
        + open_and_pending_shards
        + encoder_and_upload
        + checksum_and_verification
        + model_cache
        + ledger_and_safety
    )


def cuda_status(device: str) -> dict[str, object]:
    """Inspect the selected CUDA device without constructing a model.

    Returns:
        Device availability and memory evidence.
    """
    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        index = (
            torch.cuda.current_device()
            if device.startswith("cuda") and available
            else None
        )
        if device.startswith("cuda:") and available:
            index = int(device.split(":", 1)[1])
            torch.cuda.get_device_properties(index)
        total = (
            int(torch.cuda.get_device_properties(index).total_memory)
            if index is not None
            else 0
        )
        free, _ = torch.cuda.mem_get_info(index) if index is not None else (0, 0)
        return {
            "checked": True,
            "available": available,
            "device": device,
            "index": index,
            "free_bytes": int(free),
            "total_bytes": total,
        }
    except Exception as exc:
        return {
            "checked": True,
            "available": False,
            "device": device,
            "error": type(exc).__name__,
        }


def directory_size(root: Path) -> int:
    """Return regular-file bytes below a scratch root without following symlinks."""
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def target_privacy(hub: object | None, repo_id: str) -> dict[str, object]:
    """Report target access while allowing a missing target during planning.

    Returns:
        Metadata-only target presence and privacy evidence.
    """
    if hub is None:
        return {"checked": False, "present": False, "private": None, "repo_id": repo_id}
    try:
        info = hub.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        if exc.__class__.__name__ in {"RepositoryNotFoundError", "EntryNotFoundError"}:
            return {
                "checked": True,
                "present": False,
                "private": None,
                "repo_id": repo_id,
            }
        raise
    private = (
        info.get("private")
        if isinstance(info, dict)
        else getattr(info, "private", None)
    )
    return {
        "checked": True,
        "present": True,
        "private": private is True,
        "repo_id": repo_id,
    }


class MetadataLog:
    """Append-only metadata JSONL log that refuses payload-bearing values."""

    def __init__(self, path: Path) -> None:
        """Create a log at ``path`` and create its parent directory."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, object]) -> None:
        """Append one event after removing likely content-bearing fields."""
        forbidden = {
            "audio",
            "audio_bytes",
            "transcript_text",
            "text",
            "waveform",
            "path_local",
        }
        safe = {key: value for key, value in event.items() if key not in forbidden}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True)
class IndexRejection:
    """A metadata-only transcript-index rejection."""

    reason: str
    file_id: str | None


@dataclass(frozen=True)
class TranscriptRecord:
    """One indexed transcript retained only for the current bounded run."""

    file_id: str
    row: dict[str, object]


@dataclass(frozen=True)
class TranscriptIndex:
    """Deterministic file-id index and its metadata-only rejection evidence."""

    records: dict[str, TranscriptRecord]
    rejections: tuple[IndexRejection, ...]


class SourceAdapter(t.Protocol):
    """Metadata and bounded-audio interface used by the pipeline."""

    def iter_programmes(
        self, *, shard: SourceShard, index: TranscriptIndex
    ) -> c.Iterable[object]:
        """Yield joined programme metadata for one source shard."""

    def iter_transcripts(self, *, revision: str) -> c.Iterable[object]:
        """Yield transcript rows without decoding audio."""

    def list_audio_shards(self, *, revision: str) -> c.Iterable[object]:
        """Return source object metadata."""

    def retrieve_audio(self, *, programme: object, shard: SourceShard) -> object:
        """Retrieve exactly one programme's audio."""


def _run_legacy_pipeline(
    *,
    config: DictConfig,
    source: SourceAdapter | None = None,
    hub: object | None = None,
    vad: VADBackend | None = None,
    ctc: CTCBackend | None = None,
) -> BuildReport:
    """Execute one restartable P1 run.

    All source and transcript discovery is completed before a build can retrieve audio.
    The default adapters are lazy, allowing tests and plan runs to use fakes without
    importing or downloading any model.

    Returns:
        Bounded counters and preflight evidence.

    Raises:
        ValueError:
            If ``mode`` is not a supported pipeline mode.
    """
    settings = PipelineSettings.from_config(config)
    if settings.mode not in {"plan", "pilot", "production", "build"}:
        raise ValueError("mode must be plan, pilot, production, or build")
    if settings.mode == "pilot" and settings.programme_limit is None:
        raise ValueError("pilot mode requires programme_limit")
    if settings.workers != 1:
        raise ValueError("P1 permits exactly one programme worker")
    if settings.queue_slots < 1:
        raise ValueError("queue_slots must be positive")
    scratch = configure_scratch(settings.scratch_root)
    log = MetadataLog(scratch / "p1-events.jsonl")
    default_source = source is None
    if source is None:
        from hviske.p1_source import HfP1Source

        source = t.cast(
            SourceAdapter,
            HfP1Source(
                audio_repository=settings.source_audio_repository,
                transcript_repository=settings.source_transcript_repository,
                max_source_object_bytes=settings.max_source_bytes,
            ),
        )
    if hub is None and (settings.mode != "plan" or default_source):
        hub = make_hub()
    shards = sorted_source_shards(source, settings.source_audio_revision)
    index = build_transcript_index(
        source.iter_transcripts(revision=settings.source_transcript_revision), log=log
    )
    programmes = list(
        iter_selected_programmes(
            source=source,
            shards=shards,
            index=index,
            programme_limit=settings.programme_limit,
            source_file_id=settings.source_file_id,
            log=log,
        )
    )
    maximum_source_bytes = max((item[2].byte_size for item in programmes), default=0)
    preflight = preflight_pipeline(
        settings=settings,
        source=source,
        hub=hub,
        shards=shards,
        selected_programmes=len(programmes),
        maximum_source_bytes=maximum_source_bytes,
    )
    log.write({"event": "preflight", **preflight.as_dict()})
    index_rejection_counts: dict[str, int] = {}
    for rejection in index.rejections:
        index_rejection_counts[rejection.reason] = (
            index_rejection_counts.get(rejection.reason, 0) + 1
        )
    report = BuildReport(
        preflight=preflight,
        selected_file_ids=tuple(item[0] for item in programmes),
        rejected=len(index.rejections),
        rejection_counts=index_rejection_counts,
    )
    if settings.mode == "plan":
        return report

    if vad is None:
        vad = make_silero_vad(settings)
    if ctc is None:
        ctc = make_ctc_backend(settings)
    ledger_path = scratch / "ledger.sqlite"
    with Ledger(ledger_path) as ledger:
        initialise_target(hub=hub, settings=settings)
        process_programmes(
            source=source,
            settings=settings,
            programmes=programmes,
            ledger=ledger,
            hub=hub,
            vad=vad,
            ctc=ctc,
            report=report,
            log=log,
        )
    return report


def build_transcript_index(
    rows: c.Iterable[object], *, log: MetadataLog | None = None
) -> TranscriptIndex:
    """Index authenticated transcript metadata.

    Returns:
        A file-id index and metadata-only rejection evidence.
    """
    records: dict[str, TranscriptRecord] = {}
    seen_keys: set[str] = set()
    rejections: list[IndexRejection] = []
    for raw in rows:
        row = as_mapping(raw)
        raw_id = row.get("file_id")
        file_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else None
        text = row.get("transcript_text")
        duplicate = file_id is not None and file_id in seen_keys
        if file_id is not None:
            seen_keys.add(file_id)
        if file_id is None:
            rejection = IndexRejection("null_file_id", None)
        elif duplicate:
            rejection = IndexRejection("duplicate_file_id", file_id)
        elif file_id in records:
            rejection = IndexRejection("duplicate_file_id", file_id)
        elif not isinstance(text, str) or not text.strip():
            rejection = IndexRejection("empty_text", file_id)
        else:
            records[file_id] = TranscriptRecord(file_id=file_id, row=row)
            continue
        rejections.append(rejection)
        if log is not None:
            log.write(
                {
                    "event": "transcript_rejection",
                    "reason": rejection.reason,
                    "source_file_id": rejection.file_id,
                }
            )
    return TranscriptIndex(records=records, rejections=tuple(rejections))


def as_mapping(value: object) -> dict[str, object]:
    """Convert a mapping-like dataset row without retaining a dataset object.

    Returns:
        A plain string-keyed row mapping.

    Raises:
        TypeError:
            If ``value`` is not mapping-like.
    """
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(key, str):
                result[key] = item
        return result
    raise TypeError("source rows must be mappings")


def configure_scratch(root: Path) -> Path:
    """Create the dedicated scratch tree and route supported caches into it.

    Returns:
        The resolved scratch root.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("hf", "datasets", "hub", "transformers", "torch", "tmp", "staging"):
        (root / name).mkdir(exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(root / "hf"),
            "HF_DATASETS_CACHE": str(root / "datasets"),
            "HUGGINGFACE_HUB_CACHE": str(root / "hub"),
            "TRANSFORMERS_CACHE": str(root / "transformers"),
            "TORCH_HOME": str(root / "torch"),
            "TMPDIR": str(root / "tmp"),
            "TEMP": str(root / "tmp"),
            "TMP": str(root / "tmp"),
        }
    )
    return root


def initialise_target(*, hub: object, settings: PipelineSettings) -> None:
    """Create and initialise the private target before uploading any shard bytes."""
    from hviske.p1_publish import (
        HubClient,
        build_dataset_card,
        initialise_private_dataset,
    )

    card = build_dataset_card(
        source_provenance=(
            "Pinned P1 sources: "
            f"{settings.source_audio_revision}, {settings.source_transcript_revision}."
        ),
        permitted_use="Private commercial data preparation and model training only.",
        private_access_terms="Access is restricted to authorised syv.ai members.",
        alignment_method="Pinned Silero VAD and Danish CTC segmentation.",
        field_schema="p1-segments-v1 OutputRow schema.",
        known_limitations="Pilot thresholds and anomaly statistics require review.",
        rejection_policy=(
            "Invalid, empty, low-confidence, and undecodable programmes are "
            "recorded in the ledger."
        ),
        source_revisions=str(settings.source_audio_revision),
        model_revisions=json.dumps(settings.model_revisions, sort_keys=True),
    )
    initialise_private_dataset(
        t.cast(HubClient, hub), settings.target_private_repo, card=card
    )


def iter_selected_programmes(
    *,
    source: SourceAdapter,
    shards: c.Sequence[SourceShard],
    index: TranscriptIndex,
    programme_limit: int | None,
    source_file_id: str | None,
    log: MetadataLog | None = None,
) -> c.Iterator[tuple[str, SourceProgramme, SourceShard]]:
    """Join metadata in shard/path order with deterministic de-duplication.

    Yields:
        Joined programmes in the selected deterministic order.
    """
    seen: set[str] = set()
    candidates: list[tuple[str, SourceProgramme, SourceShard]] = []
    for shard in shards:
        for raw in source.iter_programmes(shard=shard, index=index):
            row = as_mapping(raw)
            file_id = row.get("file_id")
            if (
                not isinstance(file_id, str)
                or file_id not in index.records
                or file_id in seen
            ):
                continue
            if source_file_id is not None and file_id != source_file_id:
                continue
            try:
                programme = programme_from_row(row, index.records[file_id])
            except (TypeError, ValueError):
                if log is not None:
                    log.write(
                        {
                            "event": "programme_rejection",
                            "source_file_id": file_id,
                            "reason": "invalid_timestamps",
                        }
                    )
                continue
            candidates.append((file_id, programme, shard))
            seen.add(file_id)
    candidates.sort(key=lambda item: (item[2].path, item[0]))
    if programme_limit is not None:
        candidates = candidates[:programme_limit]
    yield from candidates


def programme_from_row(
    row: dict[str, object], transcript: TranscriptRecord
) -> SourceProgramme:
    """Build a validated joined programme from source metadata and transcript words.

    Returns:
        A validated source programme.
    """
    merged = {**transcript.row, **row}
    duration = merged.get("duration_ms", merged.get("audio_duration_ms"))
    if duration is None:
        duration_seconds = merged.get("duration")
        duration = (
            float(duration_seconds) * 1000
            if isinstance(duration_seconds, (int, float))
            else 0
        )
    words = merged.get(
        "words", merged.get("word_timestamps", merged.get("timestamps", ()))
    )
    parsed_words: list[SourceWord] = []
    for word in t.cast(c.Iterable[object], words or ()):
        item = as_mapping(word)
        parsed_words.append(
            SourceWord(
                text=str(item.get("text", item.get("word", ""))),
                start_ms=_as_int(item.get("start_ms", item.get("start", 0))),
                end_ms=_as_int(item.get("end_ms", item.get("end", 0))),
                speaker_id=_optional_str(item.get("speaker_id", item.get("speaker"))),
            )
        )
    return SourceProgramme(
        file_id=transcript.file_id,
        duration_ms=_as_int(duration),
        words=tuple(parsed_words),
        transcript_text=str(merged.get("transcript_text", "")),
    )


def make_ctc_backend(settings: PipelineSettings) -> CTCBackend:
    """Construct the pinned Danish CTC and ctc-segmentation adapter lazily.

    Returns:
        The real pinned CTC backend.
    """
    from hviske.p1_models import HuggingFaceCTCBackend

    return HuggingFaceCTCBackend(
        settings.ctc_model_repository,
        settings.ctc_model_revision,
        device=settings.device,
    )


def make_hub() -> object:
    """Construct the authenticated Hub adapter only for a non-plan run.

    Returns:
        An authenticated Hub adapter.
    """
    from hviske.p1_publish import HfApiAdapter

    return HfApiAdapter()


def make_silero_vad(settings: PipelineSettings) -> VADBackend:
    """Construct the exact pinned Silero asset from the P1 scratch cache.

    Returns:
        The real pinned Silero VAD backend.
    """
    from hviske.p1_models import make_silero_vad as make_pinned_silero_vad

    return make_pinned_silero_vad(
        settings.scratch_root / "models", device=settings.device
    )


def preflight_pipeline(
    *,
    settings: PipelineSettings,
    source: object,
    hub: object | None,
    shards: c.Sequence[object],
    selected_programmes: int,
    maximum_source_bytes: int,
) -> PreflightReport:
    """Check immutable inputs, storage, device, and privacy before audio retrieval.

    Returns:
        Metadata-only preflight evidence.

    Raises:
        P1PreflightError:
            If an immutable, storage, or privacy gate fails.
    """
    if maximum_source_bytes > settings.max_source_bytes:
        raise P1PreflightError(
            f"largest source object ({maximum_source_bytes}) exceeds "
            f"max_source_bytes ({settings.max_source_bytes})"
        )
    source_revisions = {
        "audio": {
            "repository": settings.source_audio_repository,
            "revision": settings.source_audio_revision,
        },
        "transcripts": {
            "repository": settings.source_transcript_repository,
            "revision": settings.source_transcript_revision,
        },
    }
    # ``HfP1Source.plan`` already resolves both revisions using repository/tree
    # metadata.  Calling its compatibility checker here would repeat the network
    # request (and local mirrors have no Hub client at all).
    source_revision_ok = (
        True
        if hasattr(source, "plan")
        else check_source_revisions(source, source_revisions)
    )
    model_revision_ok = check_model_revisions(source, settings.model_revisions)
    required = calculate_scratch_requirement(
        settings=settings,
        maximum_source_bytes=maximum_source_bytes,
        shard_count=settings.shards_per_commit,
    )
    scratch = settings.scratch_root
    scratch.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch).free
    scratch_bytes = directory_size(scratch)
    if scratch_bytes + required > settings.max_scratch_bytes:
        raise P1PreflightError(
            "existing scratch plus required working space exceeds max_scratch_bytes"
        )
    if free_bytes < required:
        raise P1PreflightError(
            f"insufficient free space: need {required}, have {free_bytes}"
        )
    target = target_privacy(hub, settings.target_private_repo)
    cuda = cuda_status(settings.device)
    checks = {
        "source_revisions": source_revision_ok,
        "model_revisions": model_revision_ok,
        "free_space": free_bytes >= required,
        "scratch_quota": scratch_bytes <= settings.max_scratch_bytes,
        "scratch_budget": scratch_bytes + required <= settings.max_scratch_bytes,
        "source_object_cap": maximum_source_bytes <= settings.max_source_bytes,
        "cuda_device_checked": bool(cuda.get("checked", False)),
        "target_checked": bool(target.get("checked", False)),
    }
    if not source_revision_ok:
        raise P1PreflightError("source revision is not available at the pinned SHA")
    if not model_revision_ok:
        raise P1PreflightError("model revision is not available at the pinned SHA")
    if settings.mode != "plan" and not target.get("private", False):
        raise P1PreflightError("target repository is not demonstrably private")
    return PreflightReport(
        mode=settings.mode,
        selected_programmes=selected_programmes,
        maximum_source_bytes=maximum_source_bytes,
        required_scratch_bytes=required,
        free_bytes=free_bytes,
        scratch_bytes=scratch_bytes,
        source_revisions=source_revisions,
        model_revisions=settings.model_revisions,
        cuda=cuda,
        target=target,
        checks=checks,
    )


class P1PreflightError(RuntimeError):
    """Raised when a safety gate fails before source audio retrieval."""


def check_model_revisions(source: object, revisions: dict[str, object]) -> bool:
    """Validate pinned model coordinates without loading model weights.

    Returns:
        Whether the source adapter confirmed all model revisions.
    """
    checker = getattr(source, "check_model_revisions", None)
    if checker is not None:
        return bool(checker(revisions=revisions))
    if getattr(source, "local_root", None) is not None:
        return True
    client_getter = getattr(source, "_client", None)
    if client_getter is None:
        return True
    api = client_getter()
    for coordinate in revisions.values():
        item = t.cast(dict[str, object], coordinate)
        api.repo_info(item["repository"], repo_type="model", revision=item["revision"])
    return True


def check_source_revisions(source: object, revisions: dict[str, object]) -> bool:
    """Ask an adapter to validate immutable coordinates without reading payloads.

    Returns:
        Whether all source revisions are available.
    """
    checker = getattr(source, "check_revisions", None)
    if checker is None:
        return True
    try:
        return bool(checker(revisions=revisions))
    except TypeError:
        audio = t.cast(dict[str, object], revisions["audio"])
        transcripts = t.cast(dict[str, object], revisions["transcripts"])
        return bool(
            checker(
                audio_revision=str(audio["revision"]),
                transcript_revision=str(transcripts["revision"]),
            )
        )


def process_programmes(
    *,
    source: SourceAdapter,
    settings: PipelineSettings,
    programmes: c.Sequence[tuple[str, SourceProgramme, SourceShard]],
    ledger: Ledger,
    hub: object,
    vad: VADBackend,
    ctc: CTCBackend,
    report: BuildReport,
    log: MetadataLog,
) -> None:
    """Process one programme at a time and publish bounded shard batches."""
    from hviske.p1_publish import LocalShard

    pending: list[LocalShard] = []
    pending_ids: list[str] = []
    for file_id, programme, shard in programmes:
        report.max_in_flight = max(report.max_in_flight, 1)
        programme_id = f"p1-{file_id}"
        ledger.discover_programme(
            programme_id,
            source_file_id=file_id,
            source_revisions={
                "audio": {
                    "repository": settings.source_audio_repository,
                    "revision": settings.source_audio_revision,
                },
                "transcripts": {
                    "repository": settings.source_transcript_repository,
                    "revision": settings.source_transcript_revision,
                },
            },
            pipeline_digest=settings.pipeline_digest,
            source_duration_ms=programme.duration_ms,
        )
        state = ledger.programme(programme_id).state.value
        if settings.resume and state in {"purged", "verified"}:
            continue
        ledger.start_processing(programme_id)
        started = time.monotonic()
        source_audio: object | None = None
        try:
            source_audio = source.retrieve_audio(programme=programme, shard=shard)
            audio = decode_source_audio(source_audio)
            result = segment_programme(
                words=programme.words,
                audio=audio,
                source_file_id=file_id,
                source_duration_ms=programme.duration_ms,
                segmentation=settings.segmentation,
                normalisation=settings.normalisation,
                ctc=ctc,
                pipeline_version=settings.pipeline_version,
                pipeline_config_sha256=settings.pipeline_digest,
                vad=vad,
            )
            report.accepted_segments += len(result.rows)
            report.rejected += len(result.rejections)
            for _, reason in result.rejections:
                report.rejection_counts[reason] = (
                    report.rejection_counts.get(reason, 0) + 1
                )
            out = settings.scratch_root / "staging" / file_id
            shards_written = write_shards(
                result.rows, output_dir=out, target_bytes=settings.target_shard_bytes
            )
            rejection_counts = {
                reason: sum(
                    1 for _, item_reason in result.rejections if item_reason == reason
                )
                for _, reason in result.rejections
            }
            ledger.transition_programme(
                programme_id,
                target=_state("sharded"),
                evidence={
                    "accepted_count": len(result.rows),
                    "rejected_count": len(result.rejections),
                    "processed_duration_ms": int((time.monotonic() - started) * 1000),
                    "rejection_counts": rejection_counts,
                },
            )
            # Register every fsynced shard before allowing source deletion. This
            # leaves enough durable evidence to regenerate a failed upload.
            for item in shards_written.shards:
                shard_id = f"{programme_id}-{Path(item.path).stem}"
                shard_sequence = ledger.allocate_shard_sequence()
                repo_path = f"data/train/part-{shard_sequence:05d}.parquet"
                ledger.register_shard(
                    shard_id,
                    path=repo_path,
                    sha256=item.evidence.sha256,
                    byte_size=item.evidence.byte_size,
                    row_count=item.evidence.row_count,
                    programme_id=programme_id,
                )
                pending.append(
                    LocalShard(Path(item.path), repo_path, item.evidence.row_count)
                )
                pending_ids.append(shard_id)
                report.shard_count += 1
            # No source bytes survive this point: all local shard files are fsynced
            # and their metadata is already durable in SQLite.
            purge_source_temporary(source_audio)
            ledger.mark_source_temps_purged(
                programme_id, evidence={"deleted": True, "kind": "source-temporary"}
            )
            report.processed += 1
            enforce_scratch_cap(settings)
        except Exception as exc:
            report.rejected += 1
            report.rejection_counts[RejectionCategory.DECODE_ERROR.value] = (
                report.rejection_counts.get(RejectionCategory.DECODE_ERROR.value, 0) + 1
            )
            ledger.transition_programme(
                programme_id, target=_state("retryable"), last_error=str(exc)[:500]
            )
            log.write(
                {
                    "event": "programme_error",
                    "source_file_id": file_id,
                    "reason": "decode_error",
                }
            )
        finally:
            del source_audio
            gc.collect()
        if len(pending) >= settings.shards_per_commit:
            publish_pending(
                hub=hub,
                settings=settings,
                ledger=ledger,
                batch_id=ledger.allocate_batch_id(),
                pending=pending,
                pending_ids=pending_ids,
            )
            pending, pending_ids = [], []
            enforce_scratch_cap(settings)
    if pending:
        publish_pending(
            hub=hub,
            settings=settings,
            ledger=ledger,
            batch_id=ledger.allocate_batch_id(),
            pending=pending,
            pending_ids=pending_ids,
        )
    enforce_scratch_cap(settings)


def _state(value: str) -> LedgerState:
    return LedgerState(value)


def decode_source_audio(value: object) -> np.ndarray:
    """Decode one programme lazily, accepting arrays and production FLAC files.

    Returns:
        One mono audio array.

    Raises:
        TypeError:
            If the adapter returns an unsupported payload.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray)):
        import io

        import soundfile as sf

        return np.asarray(sf.read(io.BytesIO(bytes(value)), dtype="float32")[0])
    if isinstance(value, (str, Path)):
        import soundfile as sf

        return np.asarray(sf.read(str(value), dtype="float32")[0])
    raise TypeError("audio adapter must return an ndarray, bytes, or path")


def enforce_scratch_cap(settings: PipelineSettings) -> None:
    """Raise when hard-cap monitoring observes an over-quota scratch tree.

    Raises:
        P1PreflightError:
            If the configured scratch quota has been exceeded.
    """
    used = directory_size(settings.scratch_root)
    if used > settings.max_scratch_bytes:
        raise P1PreflightError("scratch hard cap exceeded")


def publish_pending(
    *,
    hub: object,
    settings: PipelineSettings,
    ledger: Ledger,
    batch_id: str,
    pending: c.Sequence[object],
    pending_ids: c.Sequence[str],
) -> None:
    """Verify and purge a bounded publication batch through the shared publisher."""
    from hviske.p1_publish import HubClient, LocalShard, publish_batch

    api = t.cast(HubClient, hub)
    shards = t.cast(c.Sequence[LocalShard], pending)
    existing = ledger.register_batch(batch_id, pipeline_digest=settings.pipeline_digest)
    programme_id_values: list[str] = []
    for shard_id in pending_ids:
        programme_id = ledger.shard(shard_id).programme_id
        if programme_id is not None:
            programme_id_values.append(programme_id)
    programme_ids = tuple(sorted(set(programme_id_values)))
    if existing.state is _state("committed"):
        present = ledger.reconcile_committed_batch(
            batch_id,
            lambda commit_id, paths: remote_paths_present(
                hub=hub,
                repo_id=settings.target_private_repo,
                commit_id=commit_id,
                paths=paths,
            ),
        )
        if present:
            ledger.transition_batch(batch_id, _state("verified"))
            for shard_id in pending_ids:
                ledger.transition_shard(shard_id, _state("committed"))
                ledger.transition_shard(shard_id, _state("verified"))
            purge_publication_files(pending=pending)
            ledger.purge_batch(
                batch_id, evidence={"deleted": True, "kind": "publication-artifact"}
            )
            for shard_id in pending_ids:
                ledger.transition_shard(shard_id, _state("purged"))
            for programme_id in programme_ids:
                ledger.transition_programme(programme_id, _state("purged"))
            return
    if existing.state is not _state("sharded"):
        ledger.transition_batch(batch_id, _state("processing"))
        for shard_id in pending_ids:
            ledger.attach_shard(batch_id, shard_id)
        ledger.transition_batch(batch_id, _state("sharded"))

    def durable(evidence: object) -> None:
        commit_id = getattr(evidence, "commit_id")
        ledger.transition_batch(batch_id, _state("committed"), commit_id=commit_id)
        ledger.transition_batch(batch_id, _state("verified"))
        for shard_id in pending_ids:
            ledger.transition_shard(shard_id, _state("committed"))
            ledger.transition_shard(shard_id, _state("verified"))
        for programme_id in programme_ids:
            ledger.transition_programme(
                programme_id, _state("committed"), commit_id=commit_id
            )
            ledger.transition_programme(programme_id, _state("verified"))

    def purge(paths: tuple[Path, ...]) -> None:
        for path in paths:
            if path.exists():
                path.unlink()
        ledger.purge_batch(
            batch_id, evidence={"deleted": True, "kind": "publication-artifact"}
        )
        for shard_id in pending_ids:
            ledger.transition_shard(shard_id, _state("purged"))
        for programme_id in programme_ids:
            ledger.transition_programme(programme_id, _state("purged"))

    publish_batch(
        api,
        settings.target_private_repo,
        batch_id,
        shards,
        durable_verification=durable,
        purge_callback=purge,
        staging_dir=Path(shards[0].path).parent,
    )


def purge_publication_files(*, pending: c.Sequence[object]) -> None:
    """Remove only local shard files and their generated batch manifests."""
    parents: set[Path] = set()
    for item in pending:
        path = getattr(item, "path")
        if isinstance(path, Path):
            if path.is_file() and not path.is_symlink():
                path.unlink()
            parents.add(path.parent)
    for parent in parents:
        manifest = parent / "batch-manifest.json"
        if manifest.is_file() and not manifest.is_symlink():
            manifest.unlink()


def remote_paths_present(
    *, hub: object, repo_id: str, commit_id: str, paths: tuple[str, ...]
) -> bool:
    """Check all committed shard paths before retrying an interrupted upload.

    Returns:
        True only when every publication path is present at the immutable commit.
    """
    getter = getattr(hub, "get_paths_info")
    found = tuple(getter(repo_id, list(paths), repo_type="dataset", revision=commit_id))
    return len(found) == len(paths)


def purge_source_temporary(value: object) -> None:
    """Delete only a source temporary after local shard fsync, never arbitrary paths."""
    if isinstance(value, Path) and value.is_file() and not value.is_symlink():
        value.unlink()


def sorted_source_shards(
    source: SourceAdapter, revision: str
) -> tuple[SourceShard, ...]:
    """Return source objects in an explicit path order, never in Hub listing order."""
    result = []
    for raw in source.list_audio_shards(revision=revision):
        item = as_mapping(raw)
        path = item.get("path", item.get("name"))
        if not isinstance(path, str) or not path:
            continue
        size = item.get("size", item.get("byte_size", 0))
        result.append(
            SourceShard(
                path=path,
                byte_size=_as_int(size),
                revision=revision,
                oid=_optional_str(item.get("oid")),
            )
        )
    return tuple(sorted(result, key=lambda item: item.path))


def _run_native_pipeline(
    *,
    config: DictConfig,
    source: object,
    hub: object | None,
    vad: VADBackend | None,
    ctc: CTCBackend | None,
) -> BuildReport:
    """Run a production-shaped source with no in-memory corpus materialisation.

    Returns:
        Metadata-only or completed build evidence.

    Raises:
        ValueError:
            If the mode or worker configuration is unsafe.
    """
    settings = PipelineSettings.from_config(config)
    if settings.mode not in {"plan", "pilot", "production", "build", "initialise"}:
        raise ValueError("mode must be plan, pilot, production, build, or initialise")
    if settings.mode == "pilot" and settings.programme_limit is None:
        raise ValueError("pilot mode requires programme_limit")
    if settings.workers != 1:
        raise ValueError("P1 permits exactly one programme worker")
    scratch = configure_scratch(settings.scratch_root)
    log = MetadataLog(scratch / "p1-events.jsonl")
    from hviske.p1_source import SourcePlan

    plan = t.cast(
        SourcePlan,
        source.plan(
            audio_revision=settings.source_audio_revision,
            transcript_revision=settings.source_transcript_revision,
        ),
    )
    shards = tuple(sorted(plan.audio_shards, key=lambda item: item.path))
    maximum_source_bytes = max((item.byte_size for item in shards), default=0)
    preflight = preflight_pipeline(
        settings=settings,
        source=source,
        hub=hub,
        shards=shards,
        selected_programmes=0,
        maximum_source_bytes=maximum_source_bytes,
    )
    log.write({"event": "preflight", **preflight.as_dict()})
    report = BuildReport(preflight=preflight, selected_file_ids=(), rejection_counts={})
    if settings.mode == "plan":
        return report
    if hub is None:
        hub = make_hub()
    if settings.mode == "initialise":
        initialise_target(hub=hub, settings=settings)
        return report

    index_path = scratch / "transcript-pointers.sqlite"
    index_builder = getattr(source, "build_transcript_index")
    index = index_builder(
        revision=settings.source_transcript_revision,
        path=index_path,
        objects=plan.transcript_objects,
    )
    candidates = _native_candidates(
        source=source,
        shards=shards,
        index=index,
        programme_limit=settings.programme_limit,
        source_file_id=settings.source_file_id,
        pilot=settings.mode == "pilot",
        log=log,
    )
    if isinstance(candidates, list):
        report.selected_file_ids = tuple(item[0] for item in candidates)
        report.preflight = dataclasses.replace(
            report.preflight, selected_programmes=len(candidates)
        )
    if vad is None:
        vad = make_silero_vad(settings)
    if ctc is None:
        ctc = make_ctc_backend(settings)
    ledger_path = scratch / "ledger.sqlite"
    with Ledger(ledger_path) as ledger:
        _recover_native_batches(
            source=source, settings=settings, ledger=ledger, hub=hub
        )
        _process_native_programmes(
            source=source,
            settings=settings,
            candidates=candidates,
            ledger=ledger,
            hub=hub,
            vad=vad,
            ctc=ctc,
            report=report,
            log=log,
        )
    return report


def _native_candidates(
    *,
    source: object,
    shards: c.Sequence[object],
    index: object,
    programme_limit: int | None,
    source_file_id: str | None,
    pilot: bool,
    log: MetadataLog,
) -> c.Iterable[tuple[str, object, object]]:
    """Select metadata pointers, using a stable hash reservoir for pilots.

    Returns:
        A bounded list for a limited run or a streaming iterator for production.
    """

    def stream() -> c.Iterator[tuple[str, object, object]]:
        # Unlimited production runs do not retain a corpus-sized identifier set;
        # bounded runs keep only a small duplicate window around their reservoir.
        seen: set[str] = set()
        seen_limit = 0 if programme_limit is None else max(32, programme_limit * 4)
        for shard in shards:
            metadata_iterator = getattr(source, "iter_programme_metadata", None)
            if metadata_iterator is None:
                metadata_iterator = getattr(source, "iter_programmes")
            for raw in metadata_iterator(shard=shard):
                row = as_mapping(raw)
                file_id = row.get("file_id")
                if not isinstance(file_id, str) or not file_id or file_id in seen:
                    continue
                if source_file_id is not None and file_id != source_file_id:
                    continue
                pointer = getattr(index, "get")(file_id)
                if pointer is None:
                    log.write(
                        {
                            "event": "programme_rejection",
                            "source_file_id": file_id,
                            "reason": "missing_transcript",
                        }
                    )
                    continue
                safe_metadata = {
                    key: value
                    for key, value in row.items()
                    if key.casefold()
                    not in {
                        "audio",
                        "text",
                        "transcript",
                        "transcript_text",
                        "words",
                        "word_timestamps",
                        "timestamps",
                    }
                    and isinstance(value, (str, int, float, bool, type(None)))
                }
                if seen_limit:
                    seen.add(file_id)
                    if len(seen) > seen_limit:
                        seen.clear()
                yield file_id, safe_metadata, (shard, pointer)

    if programme_limit is None:
        return stream()
    selected: list[tuple[str, object, object]] = []
    for candidate in stream():
        selected.append(candidate)
        if pilot:
            selected.sort(key=lambda item: _candidate_rank(item[0]))
            del selected[programme_limit:]
        elif len(selected) == programme_limit:
            break
    selected.sort(key=lambda item: _candidate_rank(item[0]) if pilot else item[0])
    return selected


def _candidate_rank(file_id: str) -> str:
    """Return a stable pseudo-random rank without retaining source content."""
    import hashlib

    return hashlib.sha256(f"p1-pilot\\0{file_id}".encode()).hexdigest()


def _process_native_programmes(
    *,
    source: object,
    settings: PipelineSettings,
    candidates: c.Iterable[tuple[str, object, object]],
    ledger: Ledger,
    hub: object,
    vad: VADBackend,
    ctc: CTCBackend,
    report: BuildReport,
    log: MetadataLog,
) -> None:
    """Retrieve, segment, and publish one selected programme at a time."""
    from hviske.p1_publish import LocalShard

    pending: list[LocalShard] = []
    pending_ids: list[str] = []
    for file_id, metadata, locator in candidates:
        report.max_in_flight = max(report.max_in_flight, 1)
        shard, transcript_pointer = t.cast(tuple[object, object], locator)
        programme_id = f"p1-{file_id}"
        ledger.discover_programme(
            programme_id,
            source_file_id=file_id,
            source_revisions={
                "audio": {
                    "repository": settings.source_audio_repository,
                    "revision": settings.source_audio_revision,
                },
                "transcripts": {
                    "repository": settings.source_transcript_repository,
                    "revision": settings.source_transcript_revision,
                },
            },
            pipeline_digest=settings.pipeline_digest,
            source_duration_ms=_as_int(as_mapping(metadata).get("duration_ms", 1)),
        )
        state = ledger.programme(programme_id).state.value
        if settings.resume and state in {"purged", "verified", "sharded"}:
            continue
        try:
            enforce_scratch_cap(settings)
            ledger.start_processing(programme_id)
            transcript = source.fetch_transcript(transcript_pointer)
            enforce_scratch_cap(settings)
            duration = _as_int(as_mapping(metadata).get("duration_ms", 0))
            if duration <= 0:
                duration = max((word.end_ms for word in transcript.words), default=0)
            programme = SourceProgramme(
                file_id=file_id,
                duration_ms=duration,
                words=tuple(transcript.words),
                transcript_text=transcript.text,
            )
            audio_pointer = next(
                item
                for item in source.iter_programme_pointers(shard=shard)
                if item.file_id == file_id
            )
            parsed_audio = source.fetch_audio(pointer=audio_pointer)
            enforce_scratch_cap(settings)
            result = segment_programme(
                words=programme.words,
                audio=np.asarray(parsed_audio.value),
                source_file_id=file_id,
                source_duration_ms=programme.duration_ms,
                segmentation=settings.segmentation,
                normalisation=settings.normalisation,
                ctc=ctc,
                pipeline_version=settings.pipeline_version,
                pipeline_config_sha256=settings.pipeline_digest,
                vad=vad,
                sampling_rate=parsed_audio.sampling_rate,
            )
            report.accepted_segments += len(result.rows)
            report.rejected += len(result.rejections)
            _record_audit_candidates(
                rows=result.rows, path=settings.scratch_root / "audit-candidates.jsonl"
            )
            for _, reason in result.rejections:
                report.rejection_counts[reason] = (
                    report.rejection_counts.get(reason, 0) + 1
                )
            output = settings.scratch_root / "staging" / file_id
            written = write_shards(
                result.rows, output_dir=output, target_bytes=settings.target_shard_bytes
            )
            enforce_scratch_cap(settings)
            ledger.transition_programme(
                programme_id,
                target=_state("sharded"),
                evidence={
                    "accepted_count": len(result.rows),
                    "rejected_count": len(result.rejections),
                },
            )
            for item in written.shards:
                sequence = ledger.allocate_shard_sequence()
                shard_id = f"shard-{sequence:08d}"
                repo_path = f"data/train/part-{sequence:05d}.parquet"
                ledger.register_shard(
                    shard_id,
                    path=repo_path,
                    sha256=item.evidence.sha256,
                    byte_size=item.evidence.byte_size,
                    row_count=item.evidence.row_count,
                    programme_id=programme_id,
                )
                pending.append(
                    LocalShard(Path(item.path), repo_path, item.evidence.row_count)
                )
                pending_ids.append(shard_id)
                report.shard_count += 1
            purge_source_temporary(getattr(source, "last_temporary", None))
            ledger.mark_source_temps_purged(programme_id, evidence={"deleted": True})
            report.processed += 1
        except Exception as exc:
            report.rejected += 1
            report.rejection_counts[RejectionCategory.DECODE_ERROR.value] = (
                report.rejection_counts.get(RejectionCategory.DECODE_ERROR.value, 0) + 1
            )
            ledger.transition_programme(
                programme_id, target=_state("retryable"), last_error=str(exc)[:500]
            )
            log.write(
                {
                    "event": "programme_error",
                    "source_file_id": file_id,
                    "reason": "decode_error",
                }
            )
        finally:
            gc.collect()
        if len(pending) >= settings.shards_per_commit:
            _publish_native_pending(
                hub=hub,
                settings=settings,
                ledger=ledger,
                pending=pending,
                pending_ids=pending_ids,
            )
            pending, pending_ids = [], []
        enforce_scratch_cap(settings)
    if pending:
        _publish_native_pending(
            hub=hub,
            settings=settings,
            ledger=ledger,
            pending=pending,
            pending_ids=pending_ids,
        )
    enforce_scratch_cap(settings)


def _publish_native_pending(
    *,
    hub: object,
    settings: PipelineSettings,
    ledger: Ledger,
    pending: c.Sequence[object],
    pending_ids: c.Sequence[str],
    batch_id: str | None = None,
) -> None:
    """Allocate and publish a complete, bounded pending batch."""
    if batch_id is None:
        batch_id = ledger.allocate_batch_id()
    publish_pending(
        hub=hub,
        settings=settings,
        ledger=ledger,
        batch_id=batch_id,
        pending=pending,
        pending_ids=pending_ids,
    )


def _record_audit_candidates(*, rows: c.Sequence[object], path: Path) -> None:
    """Persist bounded blinded audit metadata, never training payload fields."""
    from hviske.p1_validation import build_representative_audit_candidates

    candidates = [
        {
            "status": "accepted",
            "segment_id": getattr(row, "segment_id"),
            "source_file_id": getattr(row, "source_file_id"),
            "source_start_ms": getattr(row, "source_start_ms"),
            "source_end_ms": getattr(row, "source_end_ms"),
            "duration_ms": getattr(row, "duration_ms"),
        }
        for row in rows
    ]
    if not candidates:
        return
    selected = build_representative_audit_candidates(
        candidates, accepted_quota=min(200, len(candidates)), seed="p1"
    )
    with path.open("a", encoding="utf-8") as stream:
        for candidate in selected:
            stream.write(json.dumps(candidate, sort_keys=True) + "\\n")
        stream.flush()
        os.fsync(stream.fileno())


def _recover_native_batches(
    *, source: object, settings: PipelineSettings, ledger: Ledger, hub: object
) -> None:
    """Verify committed work recorded before a process interruption.

    Local paths are recovered only when their basename and durable digest both match
    a ledger record.  No arbitrary staging file is eligible for deletion.
    """
    del source
    from hviske.p1_publish import HubClient, LocalShard, verify_batch

    staging = settings.scratch_root / "staging"
    for batch, records in ledger.reconstruct_work():
        paths = tuple(
            path
            for record in records
            if (
                path := _matching_local_shard(
                    ledger=ledger, record=record, staging=staging
                )
            )
            is not None
        )
        if (
            batch.commit_id is None
            and batch.state
            in {_state("discovered"), _state("processing"), _state("sharded")}
            and records
            and len(paths) == len(records)
        ):
            _publish_native_pending(
                hub=hub,
                settings=settings,
                ledger=ledger,
                pending=tuple(
                    LocalShard(path, record.path, record.row_count)
                    for record, path in zip(records, paths)
                ),
                pending_ids=tuple(record.shard_id for record in records),
                batch_id=batch.batch_id,
            )
            continue
        if batch.commit_id is None or batch.state not in {
            _state("committed"),
            _state("verified"),
        }:
            continue
        verify_batch(
            t.cast(HubClient, hub),
            settings.target_private_repo,
            batch.batch_id,
            ledger=ledger,
            purge_callback=None if not paths else _unlink_recovered,
            local_paths=paths,
        )
        recovered_batch = ledger.batch(batch.batch_id)
        for programme_id in {
            record.programme_id for record in records if record.programme_id is not None
        }:
            programme = ledger.programme(programme_id)
            if programme.state is _state("sharded"):
                ledger.transition_programme(
                    programme_id, _state("committed"), commit_id=batch.commit_id
                )
                ledger.transition_programme(programme_id, _state("verified"))
            if recovered_batch.state is _state("purged"):
                ledger.transition_programme(programme_id, _state("purged"))


def _matching_local_shard(
    *, ledger: Ledger, record: object, staging: Path
) -> Path | None:
    """Find one local shard whose bytes match its durable ledger evidence.

    Returns:
        The matching regular file, or ``None`` when recovery evidence is absent.
    """
    shard_id = str(getattr(record, "shard_id"))
    name = Path(str(getattr(record, "path"))).name
    return next(
        (
            path
            for path in staging.rglob(name)
            if ledger.reconcile_local_shard(shard_id, path)
        ),
        None,
    )
