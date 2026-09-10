"""Reusable orchestration for the bounded Phase 1 P1 segmentation dataset.

Planning deliberately uses only Hub repository/tree metadata.  Build mode then uses a
disk-backed transcript pointer index and retrieves one selected programme at a time.
"""

from __future__ import annotations

import collections.abc as c
import dataclasses
import gc
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import os
import shutil
import sqlite3
import time
import typing as t
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from hviske.p1_contracts import (
    OUTPUT_SCHEMA,
    P1_RUNTIME_CONTRACT,
    CanonicalIdentityManifest,
    CTCContract,
    DatasetLicenseContract,
    LedgerState,
    ModelContract,
    NormalisationContract,
    OutputEncodingContract,
    RejectionCategory,
    RepositoryRevision,
    SegmentationContract,
    ShardEvidence,
    SourceCoordinates,
    SourceProgramme,
    VADContract,
    pipeline_config_sha256,
    validate_p1_runtime_contract,
)
from hviske.p1_ledger import Ledger
from hviske.p1_segments import (
    CTCBackend,
    TimestampAlignmentBackend,
    VADBackend,
    segment_programme,
    write_shards,
)
from hviske.p1_source import (
    _AUDIO_POINTER_METADATA_KEY,
    _AUDIO_POINTER_METADATA_MAX_BYTES,
    AudioPointer,
    ParsedTranscript,
    TranscriptOverAudio,
    TranscriptPointer,
    _scalar_metadata,
)

logger = logging.getLogger(__name__)

_PROGRESS_INTERVAL = 10_000


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
    source_max_batch_rows: int
    source_max_batch_bytes: int
    max_decoded_audio_bytes: int
    programme_limit: int | None
    source_file_id: str | None
    resume: bool
    device: str
    target_private_repo: str
    source_audio_repository: str
    source_audio_revision: str
    source_transcript_repository: str
    source_transcript_revision: str
    alignment_method: str
    ctc_model_repository: str
    ctc_model_revision: str
    anomaly_model_repository: str
    anomaly_model_revision: str
    vad_model_path: str
    vad_model_blob: str
    vad_model_sha256: str
    model_revisions: dict[str, object]
    ctc_contract: CTCContract | None
    segmentation: SegmentationContract
    normalisation: NormalisationContract
    dataset_license: DatasetLicenseContract
    pipeline_digest: str

    @classmethod
    def from_config(cls, config: DictConfig) -> PipelineSettings:
        """Resolve Hydra values and build the complete canonical identity.

        Returns:
            Resolved pipeline settings with a canonical digest.

        Raises:
            ValueError:
                If the v7 runtime, output schema, or immutable identity is invalid.
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
        max_decoded_audio = root.get("max_decoded_audio_bytes") or runtime.get(
            "max_decoded_audio_bytes", 2 * 1024**3
        )
        future_alignment = t.cast(dict[str, object], root.get("future_alignment", root))
        if "vad" not in future_alignment:
            future_alignment = {
                **future_alignment,
                "vad": {
                    "name": "not_applicable",
                    "repository": {
                        "repository": "inactive/future-alignment",
                        "revision": "0" * 40,
                    },
                    "model_path": "",
                    "model_blob": "0" * 40,
                    "model_sha256": "0" * 64,
                    "license": "not_applicable",
                },
                "ctc": {
                    "source_commit": "0" * 40,
                    "sdist_sha256": "0" * 64,
                    "license": "not_applicable",
                    "model": {
                        "repository": {
                            "repository": "inactive/future-alignment",
                            "revision": "0" * 40,
                        },
                        "license": "not_applicable",
                    },
                },
                "anomaly_model": {
                    "repository": {
                        "repository": "inactive/future-alignment",
                        "revision": "0" * 40,
                    },
                    "license": "not_applicable",
                },
            }
        vad_raw = t.cast(dict[str, object], future_alignment["vad"])
        vad_repo = t.cast(dict[str, object], vad_raw["repository"])
        ctc_raw = t.cast(dict[str, object], future_alignment["ctc"])
        ctc_model = t.cast(dict[str, object], ctc_raw["model"])
        ctc_repo = t.cast(dict[str, object], ctc_model["repository"])
        normalisation_raw = t.cast(dict[str, object], root["normalisation"])
        anomaly_raw = t.cast(dict[str, object], future_alignment["anomaly_model"])
        output_raw = t.cast(dict[str, object], root["output"])
        dataset_license_raw = t.cast(dict[str, object], root["dataset_license"])
        manifest = CanonicalIdentityManifest(
            schema_version=str(root["schema_version"]),
            pipeline_version=str(root["pipeline_version"]),
            alignment_method=str(
                root.get("alignment_method", "timestamp-native:p1-transcripts.words")
            ),
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
                    license=str(ctc_model["license"]),
                    license_url=_optional_str(ctc_model.get("license_url")),
                    license_repository=_optional_str(
                        ctc_model.get("license_repository")
                    ),
                    license_revision=_optional_str(ctc_model.get("license_revision")),
                    license_sha256=_optional_str(ctc_model.get("license_sha256")),
                    model_card_url=_optional_str(ctc_model.get("model_card_url")),
                    model_card_sha256=_optional_str(ctc_model.get("model_card_sha256")),
                    license_notes=_optional_str(ctc_model.get("license_notes")),
                    architecture=_optional_str(ctc_model.get("architecture")),
                    model_type=_optional_str(ctc_model.get("model_type")),
                    sampling_rate=_optional_int(ctc_model.get("sampling_rate")),
                    frame_stride_samples=_optional_int(
                        ctc_model.get("frame_stride_samples")
                    ),
                    vocab_size=_optional_int(ctc_model.get("vocab_size")),
                    blank_token_id=_optional_int(ctc_model.get("blank_token_id")),
                    word_delimiter_token_id=_optional_int(
                        ctc_model.get("word_delimiter_token_id")
                    ),
                ),
            ),
            anomaly_model=ModelContract(
                repository=RepositoryRevision.model_validate(anomaly_raw["repository"]),
                license=str(anomaly_raw["license"]),
            ),
            normalisation=NormalisationContract.model_validate(normalisation_raw),
            segmentation=SegmentationContract.model_validate(root["segmentation"]),
            output=OutputEncodingContract.model_validate(output_raw),
            dataset_license=DatasetLicenseContract.model_validate(dataset_license_raw),
            max_decoded_audio_bytes=_as_int(max_decoded_audio),
        )
        validate_p1_runtime_contract(
            pipeline_version=manifest.pipeline_version,
            ctc=manifest.ctc,
            alignment_method=manifest.alignment_method,
            normalisation=manifest.normalisation,
            dataset_license=manifest.dataset_license,
        )
        if manifest.schema_version != OUTPUT_SCHEMA.schema_version:
            raise ValueError("P1 output schema version must be p1-segments-v2")
        if manifest.output.schema != OUTPUT_SCHEMA:
            raise ValueError("P1 output schema must match the active v7 schema")
        digest = pipeline_config_sha256(manifest)
        if manifest.pipeline_version != P1_RUNTIME_CONTRACT.pipeline_version and (
            manifest.ctc is None
        ):
            raise ValueError("future model-backed alignment requires CTC metadata")
        ctc_contract = manifest.ctc
        active_model_revisions = (
            {}
            if manifest.pipeline_version == P1_RUNTIME_CONTRACT.pipeline_version
            else {
                "vad": {
                    **vad_repo,
                    "model_path": str(vad_raw["model_path"]),
                    "model_blob": str(vad_raw["model_blob"]),
                    "model_sha256": str(vad_raw["model_sha256"]),
                },
                "ctc": {
                    **ctc_contract.model.repository.model_dump(mode="json"),
                    **ctc_contract.model.model_dump(
                        mode="json", exclude={"repository"}, exclude_none=True
                    ),
                },
                "ctc_library": {
                    "name": ctc_contract.name,
                    "version": ctc_contract.version,
                    "source_commit": ctc_contract.source_commit,
                    "sdist_sha256": ctc_contract.sdist_sha256,
                    "license": ctc_contract.license,
                },
                "anomaly": anomaly_raw["repository"],
            }
        )
        return cls(
            mode=str(root.get("mode", "build")),
            pipeline_version=str(root["pipeline_version"]),
            alignment_method=manifest.alignment_method,
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
            source_max_batch_rows=_as_int(runtime.get("source_max_batch_rows", 1024)),
            source_max_batch_bytes=_as_int(
                runtime.get("source_max_batch_bytes", 64 * 1024 * 1024)
            ),
            max_decoded_audio_bytes=_as_int(max_decoded_audio),
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
            vad_model_path=str(vad_raw["model_path"]),
            vad_model_blob=str(vad_raw["model_blob"]),
            vad_model_sha256=str(vad_raw["model_sha256"]),
            model_revisions=active_model_revisions,
            ctc_contract=ctc_contract,
            segmentation=manifest.segmentation,
            normalisation=manifest.normalisation,
            dataset_license=manifest.dataset_license,
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
    """Run counters and preflight evidence."""

    preflight: PreflightReport
    selected_programmes: int
    processed: int = 0
    rejected: int = 0
    accepted_segments: int = 0
    shard_count: int = 0
    max_in_flight: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    normalization_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return metadata-only run evidence."""
        return {
            "preflight": self.preflight.as_dict(),
            "selected_programmes": self.selected_programmes,
            "processed": self.processed,
            "rejected": self.rejected,
            "accepted_segments": self.accepted_segments,
            "shard_count": self.shard_count,
            "max_in_flight": self.max_in_flight,
            "rejection_counts": self.rejection_counts,
            "normalization_counts": self.normalization_counts,
        }


def run_pipeline(
    *,
    config: DictConfig,
    source: object | None = None,
    hub: object | None = None,
    vad: VADBackend | None = None,
    ctc: CTCBackend | None = None,
) -> BuildReport:
    """Execute the P1 pipeline through the native P1 source contract.

    Returns:
        Metadata-only or completed build evidence.

    Raises:
        TypeError:
            If the source does not expose the native planning API.
    """
    from hviske.p1_source import harden_p1_logging

    harden_p1_logging()
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
            max_batch_rows=settings.source_max_batch_rows,
            max_batch_bytes=settings.source_max_batch_bytes,
            max_decoded_audio_bytes=settings.max_decoded_audio_bytes,
        )
    if not hasattr(source, "plan"):
        raise TypeError("source must expose the p1_source planning API")
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
    target = target_privacy(hub, settings.target_private_repo, settings.pipeline_digest)
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
        model_revisions={},
        cuda={"checked": False, "reason": "not_applicable:timestamp-native"},
        target=target,
        checks={
            "target_checked": target["private"] is True,
            "model_provenance_not_applicable": True,
            "cuda_device_checked": False,
        },
    )
    return BuildReport(preflight=preflight, selected_programmes=0)


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


def directory_size(root: Path) -> int:
    """Return regular-file bytes below a scratch root without following symlinks."""
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def target_privacy(
    hub: object | None, repo_id: str, expected_digest: str | None = None
) -> dict[str, object]:
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
    contract_v7 = _target_card_is_v7(
        hub=hub, repo_id=repo_id, expected_digest=expected_digest
    )
    return {
        "checked": True,
        "present": True,
        "private": private is True,
        "contract_v7": contract_v7,
        "repo_id": repo_id,
    }


def _target_card_is_v7(
    *, hub: object | None, repo_id: str, expected_digest: str | None
) -> bool:
    """Check the remote card without retaining its text in pipeline state.

    Returns:
        Whether the remote card proves the active v7 identity.
    """
    reader = getattr(hub, "stream_file", None)
    if not callable(reader):
        return False
    try:
        payload = b"".join(
            reader(repo_id, "README.md", repo_type="dataset", revision="main")
        )
        card = payload.decode("utf-8")
    except Exception:
        return False
    required = (
        "pipeline_version: p1-segmentation-7",
        "timestamp-native:p1-transcripts.words",
        "p1-segments-v2",
    )
    if not all(marker in card for marker in required):
        return False
    if "Model revisions" in card or any(
        marker in card.lower()
        for marker in ("silero", "roest", "whisper", "ctc", "vad")
    ):
        return False
    return expected_digest is None or (
        f"pipeline_config_sha256: {expected_digest}" in card
    )


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
    from hviske.p1_validation import AuditReservoir

    audit_reservoir = AuditReservoir(scratch / "audit-reservoir.json")
    plan = t.cast(
        SourcePlan,
        source.plan(
            audio_revision=settings.source_audio_revision,
            transcript_revision=settings.source_transcript_revision,
        ),
    )
    shards = tuple(sorted(plan.audio_shards, key=lambda item: item.path))
    maximum_source_bytes = max((item.byte_size for item in shards), default=0)
    if settings.mode != "plan" and hub is None:
        hub = make_hub()
    preflight = preflight_pipeline(
        settings=settings,
        source=source,
        hub=hub,
        shards=shards,
        selected_programmes=0,
        maximum_source_bytes=maximum_source_bytes,
    )
    log.write({"event": "preflight", **preflight.as_dict()})
    report = BuildReport(
        preflight=preflight, selected_programmes=0, rejection_counts={}
    )
    if settings.mode == "plan":
        return report
    if settings.mode == "initialise":
        initialise_target(hub=hub, settings=settings)
        return report

    ledger_path = scratch / "ledger.sqlite"
    # Open and bind the ledger before constructing models or selecting new work.  A
    # changed configuration must fail without touching a populated run.
    with Ledger(ledger_path, pipeline_digest=settings.pipeline_digest) as ledger:
        _recover_native_batches(
            source=source,
            settings=settings,
            ledger=ledger,
            hub=hub,
            audit_reservoir=audit_reservoir,
        )

        def scratch_guard() -> None:
            enforce_scratch_cap(settings)

        scratch_guard()
        index_path = scratch / "transcript-pointers.sqlite"
        index_builder = getattr(source, "build_transcript_index")
        index_kwargs: dict[str, object] = {
            "revision": settings.source_transcript_revision,
            "path": index_path,
            "objects": plan.transcript_objects,
        }
        if settings.source_file_id is not None:
            index_kwargs["source_file_id"] = settings.source_file_id
        index_signature = inspect.signature(index_builder)
        if "scratch_guard" in index_signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in index_signature.parameters.values()
        ):
            index_kwargs["scratch_guard"] = scratch_guard
        index = index_builder(**index_kwargs)
        candidates = _native_candidates(
            source=source,
            shards=shards,
            index=index,
            programme_limit=settings.programme_limit,
            source_file_id=settings.source_file_id,
            pilot=settings.mode == "pilot",
            log=log,
            scratch_root=settings.scratch_root,
            scratch_guard=scratch_guard,
        )
        if isinstance(candidates, list):
            logger.info(
                "P1 selection complete: %d programmes selected", len(candidates)
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
            audit_reservoir=audit_reservoir,
        )
    audit_reservoir.finalise(scratch / "audit-candidates.jsonl")
    return report


class MetadataLog:
    """Append-only metadata JSONL log that refuses payload-bearing values."""

    def __init__(self, path: Path) -> None:
        """Create a log at ``path`` and create its parent directory."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, object]) -> None:
        """Append one event after removing content and locator-bearing fields."""
        forbidden = {
            "audio",
            "audio_bytes",
            "cache",
            "file_id",
            "id",
            "path",
            "path_local",
            "programme_id",
            "query",
            "row_group",
            "row_index",
            "source_file_id",
            "source_row_group",
            "source_row_index",
            "source_shard_byte_size",
            "source_shard_index",
            "source_shard_path",
            "text",
            "transcript_id",
            "transcript_text",
            "url",
            "waveform",
        }

        def sanitise(value: object, key: str | None = None) -> object | None:
            if key is not None and (
                key.casefold() in forbidden or key.startswith("_p1_")
            ):
                return None
            if isinstance(value, dict):
                return {
                    child_key: child_value
                    for child_key, child in value.items()
                    if isinstance(child_key, str)
                    and (child_value := sanitise(child, child_key)) is not None
                }
            if isinstance(value, list):
                return [sanitise(child) for child in value]
            if isinstance(value, str) and "://" in value:
                return "<redacted-url>"
            return value

        safe = {
            key: value
            for key, item in event.items()
            if (value := sanitise(item, key)) is not None
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True)
class NativeCandidate:
    """One joined programme selected from immutable source metadata."""

    file_id: str
    metadata: dict[str, object]
    audio_pointer: AudioPointer
    transcript_pointer: TranscriptPointer

    def __getitem__(self, index: int) -> object:
        """Retain read-only tuple indexing for older callers.

        Returns:
            The selected compatibility field.
        """
        fields = (
            self.file_id,
            self.metadata,
            self.audio_pointer,
            self.transcript_pointer,
        )
        return fields[index]


def _native_candidates(
    *,
    source: object,
    shards: c.Sequence[object],
    index: object,
    programme_limit: int | None,
    source_file_id: str | None,
    pilot: bool,
    log: MetadataLog,
    scratch_root: Path | None = None,
    scratch_guard: c.Callable[[], None] | None = None,
) -> c.Iterable[NativeCandidate]:
    """Select joined transcript and audio pointers with bounded metadata scans.

    Audio pointers are discovered during selection and retained in each candidate.  A
    build therefore never has to rescan a source shard to locate an already selected
    programme.  Pilot selection samples bounded metadata and reconstructs only the
    selected candidates from their retained scalar locators.

    Returns:
        A bounded list or streaming iterator of joined source candidates.
    """
    from hviske.p1_source import SourceSelectionError, SourceShard

    pointer_iterator = getattr(source, "iter_programme_pointers", None)
    metadata_iterator = getattr(source, "iter_programme_metadata", None)
    dedup_root = scratch_root or log.path.parent
    if scratch_guard is not None:
        scratch_guard()
    dedup = _SelectionDedup(
        dedup_root / "native-selection-dedup.sqlite", scratch_guard=scratch_guard
    )

    def safe_metadata(
        row: Mapping[str, object], pointer: object, file_id: str, shard_index: int
    ) -> dict[str, object]:
        forbidden = {
            "audio",
            "text",
            "transcript",
            "transcript_text",
            "words",
            "word_timestamps",
            "timestamps",
        }
        safe = {
            key: value
            for key, value in row.items()
            if key.casefold() not in forbidden
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        for key, value in getattr(pointer, "metadata", ()):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                continue
            if isinstance(decoded, float) and not np.isfinite(decoded):
                continue
            if isinstance(decoded, (str, int, float, bool, type(None))):
                safe.setdefault(key, decoded)
        safe["id"] = file_id
        shard = getattr(pointer, "shard", None)
        if isinstance(shard_index, int) and shard_index >= 0:
            safe["source_shard_index"] = shard_index
        shard_path = getattr(shard, "path", None)
        if isinstance(shard_path, str):
            safe["source_shard_path"] = shard_path
        shard_byte_size = getattr(shard, "byte_size", None)
        if isinstance(shard_byte_size, int) and not isinstance(shard_byte_size, bool):
            safe["source_shard_byte_size"] = shard_byte_size
        shard_revision = getattr(shard, "revision", None)
        if isinstance(shard_revision, str):
            safe["source_revision"] = shard_revision
        row_group = getattr(pointer, "row_group", None)
        if isinstance(row_group, int) and not isinstance(row_group, bool):
            safe["source_row_group"] = row_group
        row_index = getattr(pointer, "row_index", None)
        if isinstance(row_index, int) and not isinstance(row_index, bool):
            safe["source_row_index"] = row_index
        return t.cast(dict[str, object], safe)

    def candidate_from_pointer(
        pointer: object, shard_index: int
    ) -> NativeCandidate | None:
        file_id = getattr(pointer, "file_id", None)
        if not isinstance(file_id, str) or not file_id:
            return None
        transcript_pointer = getattr(index, "get")(file_id)
        if transcript_pointer is None:
            return None
        if not isinstance(pointer, AudioPointer):
            pointer = AudioPointer(
                file_id=file_id,
                shard=t.cast(SourceShard, getattr(pointer, "shard")),
                row_group=int(getattr(pointer, "row_group")),
                row_index=int(getattr(pointer, "row_index")),
                metadata=tuple(getattr(pointer, "metadata", ())),
            )
        return NativeCandidate(
            file_id=file_id,
            metadata=safe_metadata({}, pointer, file_id, shard_index),
            audio_pointer=pointer,
            transcript_pointer=transcript_pointer,
        )

    def metadata_candidate(
        raw: object, shard: object, shard_index: int, row_index: int
    ) -> NativeCandidate | None:
        row = as_mapping(raw)
        file_id = row.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            return None
        transcript_pointer = getattr(index, "get")(file_id)
        if transcript_pointer is None:
            return None
        metadata_row_group = row.get("source_row_group", 0)
        if (
            not isinstance(metadata_row_group, int)
            or isinstance(metadata_row_group, bool)
            or metadata_row_group < 0
        ):
            metadata_row_group = 0
        metadata_row_index = row.get("source_row_index", row_index)
        if (
            not isinstance(metadata_row_index, int)
            or isinstance(metadata_row_index, bool)
            or metadata_row_index < 0
        ):
            metadata_row_index = row_index
        pointer = AudioPointer(
            file_id=file_id,
            shard=t.cast(SourceShard, shard),
            row_group=metadata_row_group,
            row_index=metadata_row_index,
            metadata=_scalar_metadata(row),
        )
        return NativeCandidate(
            file_id=file_id,
            metadata=safe_metadata(row, pointer, file_id, shard_index),
            audio_pointer=pointer,
            transcript_pointer=transcript_pointer,
        )

    def stream(
        *, emit_summary: bool = True, include_pointer_metadata: bool = False
    ) -> c.Iterator[NativeCandidate]:
        rows_scanned = 0
        shards_scanned = 0
        missing_transcript_count = 0
        next_progress = _PROGRESS_INTERVAL

        def emit_progress() -> None:
            nonlocal next_progress
            while rows_scanned >= next_progress:
                logger.info(
                    "Audio metadata scan progress: %d rows across %d shards",
                    rows_scanned,
                    shards_scanned,
                )
                next_progress += _PROGRESS_INTERVAL

        try:
            for shard_index, shard in enumerate(shards):
                shards_scanned += 1
                if callable(pointer_iterator):
                    for pointer in pointer_iterator(shard=shard):
                        rows_scanned += 1
                        emit_progress()
                        candidate = candidate_from_pointer(pointer, shard_index)
                        if candidate is None:
                            file_id = getattr(pointer, "file_id", None)
                            if isinstance(file_id, str) and file_id:
                                missing_transcript_count += 1
                            continue
                        if not dedup.add_if_new(candidate.file_id):
                            continue
                        if include_pointer_metadata:
                            metadata = dict(candidate.metadata)
                            metadata[_AUDIO_POINTER_METADATA_KEY] = (
                                _encode_audio_pointer_metadata(
                                    candidate.audio_pointer.metadata
                                )
                            )
                            candidate = dataclasses.replace(
                                candidate, metadata=metadata
                            )
                        yield candidate
                elif callable(metadata_iterator):
                    for row_index, raw in enumerate(metadata_iterator(shard=shard)):
                        rows_scanned += 1
                        emit_progress()
                        candidate = metadata_candidate(
                            raw, shard, shard_index, row_index
                        )
                        if candidate is None:
                            file_id = as_mapping(raw).get("file_id")
                            if isinstance(file_id, str) and file_id:
                                missing_transcript_count += 1
                            continue
                        if not dedup.add_if_new(candidate.file_id):
                            continue
                        if include_pointer_metadata:
                            metadata = dict(candidate.metadata)
                            metadata[_AUDIO_POINTER_METADATA_KEY] = (
                                _encode_audio_pointer_metadata(
                                    candidate.audio_pointer.metadata
                                )
                            )
                            candidate = dataclasses.replace(
                                candidate, metadata=metadata
                            )
                        yield candidate
        finally:
            if emit_summary and missing_transcript_count:
                log.write(
                    {
                        "event": "programme_rejection_summary",
                        "reason": "missing_transcript",
                        "count": missing_transcript_count,
                    }
                )
            if emit_summary:
                logger.info(
                    "Audio metadata selection complete: %d rows, %d shards, "
                    "%d missing transcripts",
                    rows_scanned,
                    shards_scanned,
                    missing_transcript_count,
                )
            dedup.close()

    def targeted() -> list[NativeCandidate]:
        target_id = t.cast(str, source_file_id)
        selection = stream()
        try:
            for candidate in selection:
                if candidate.file_id == target_id:
                    logger.info("Audio metadata selection complete: target selected")
                    return [candidate]
        finally:
            selection.close()
        raise SourceSelectionError(
            "requested source_file_id was not found in audio metadata"
        )

    if source_file_id is not None:
        return targeted()
    if programme_limit is None:
        return stream()
    if not pilot:
        selected = list(itertools.islice(stream(), programme_limit))
        dedup.close()
        return selected
    from hviske.p1_validation import stratified_sample

    selected_rows = stratified_sample(
        (candidate.metadata for candidate in stream(include_pointer_metadata=True)),
        sample_size=programme_limit,
        seed="p1-pilot",
    )

    def selected_candidate(row: Mapping[str, object]) -> NativeCandidate:
        file_id = row.get("id")
        shard_index = row.get("source_shard_index")
        row_group = row.get("source_row_group")
        row_index = row.get("source_row_index")
        if (
            not isinstance(file_id, str)
            or not file_id
            or not isinstance(shard_index, int)
            or isinstance(shard_index, bool)
            or not 0 <= shard_index < len(shards)
            or not isinstance(row_group, int)
            or isinstance(row_group, bool)
            or row_group < 0
            or not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index < 0
        ):
            raise SourceSelectionError("selected audio metadata has no valid locator")
        shard = t.cast(SourceShard, shards[shard_index])
        shard_path = row.get("source_shard_path")
        if isinstance(shard_path, str) and shard_path != shard.path:
            raise SourceSelectionError("selected audio metadata locator disagrees")
        transcript_pointer = getattr(index, "get")(file_id)
        if transcript_pointer is None:
            raise SourceSelectionError("selected programme has no transcript pointer")
        pointer = AudioPointer(
            file_id=file_id,
            shard=shard,
            row_group=row_group,
            row_index=row_index,
            metadata=_decode_audio_pointer_metadata(
                row.get(_AUDIO_POINTER_METADATA_KEY)
            ),
        )
        metadata = {
            key: value
            for key, value in row.items()
            if key != _AUDIO_POINTER_METADATA_KEY
        }
        return NativeCandidate(
            file_id=file_id,
            metadata=metadata,
            audio_pointer=pointer,
            transcript_pointer=transcript_pointer,
        )

    selected = [selected_candidate(row) for row in selected_rows]
    selected.sort(key=lambda candidate: candidate.file_id)
    logger.info(
        "Audio metadata selection complete: %d programmes selected", len(selected)
    )
    return selected


class _SelectionDedup:
    """Crash-safe, metadata-only deduplication for one selection pass."""

    def __init__(
        self, path: Path, scratch_guard: c.Callable[[], None] | None = None
    ) -> None:
        """Create an empty deduplication database at ``path``.

        Args:
            path:
                Database path in the owned scratch tree.
            scratch_guard (optional):
                Callback that rejects a run after scratch growth.
        """
        self.path = Path(path)
        self._scratch_guard = scratch_guard
        self._validate_path()
        self._guard_scratch()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recreate_database()
        self.connection = sqlite3.connect(self.path)
        try:
            self.connection.execute("PRAGMA journal_mode = DELETE")
            self.connection.execute("PRAGMA synchronous = FULL")
            self.connection.execute(
                "CREATE TABLE selected_ids "
                "(file_id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID"
            )
            self.connection.commit()
            self._guard_scratch()
        except Exception:
            self.connection.close()
            raise

    def _guard_scratch(self) -> None:
        """Check the scratch quota after a database transaction is durable."""
        if self._scratch_guard is not None:
            self._scratch_guard()

    def _recreate_database(self) -> None:
        """Remove old high-water files before creating the next selection state."""
        if self.path.exists():
            self.path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()

    def _validate_path(self) -> None:
        """Reject paths that could make quota accounting or replacement unsafe.

        Raises:
            ValueError:
                If the database or one of its sidecars is an unexpected path.
        """
        if self.path.is_symlink():
            raise ValueError(f"deduplication path must not be a symlink: {self.path}")
        if self.path.exists() and not self.path.is_file():
            raise ValueError(f"deduplication path must be a regular file: {self.path}")
        if self.path.parent.is_symlink():
            raise ValueError(
                f"deduplication parent must not be a symlink: {self.path.parent}"
            )
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            if sidecar.is_symlink():
                raise ValueError(
                    f"deduplication sidecar must not be a symlink: {sidecar}"
                )
            if sidecar.exists() and not sidecar.is_file():
                raise ValueError(
                    f"deduplication sidecar must be a regular file: {sidecar}"
                )

    def close(self) -> None:
        """Close the durable selection database connection."""
        self.connection.close()

    def add_if_new(self, file_id: str) -> bool:
        """Record ``file_id`` and return whether this is its first occurrence.

        Returns:
            Whether this call inserted the identifier.
        """
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO selected_ids (file_id) VALUES (?)", (file_id,)
        )
        is_new = cursor.rowcount == 1
        self.connection.commit()
        if is_new:
            self._guard_scratch()
        return is_new


def _decode_audio_pointer_metadata(value: object) -> tuple[tuple[str, str], ...]:
    """Decode the bounded internal pointer-metadata scalar.

    Returns:
        The original scalar metadata tuple.

    Raises:
        ValueError:
            If the scalar is missing, oversized, or malformed.
    """
    if not isinstance(value, str):
        raise ValueError("selected audio metadata has no pointer metadata")
    if len(value.encode("utf-8")) > _AUDIO_POINTER_METADATA_MAX_BYTES:
        raise ValueError("selected audio pointer metadata exceeds its size limit")
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(
        isinstance(item, list)
        and len(item) == 2
        and all(isinstance(part, str) for part in item)
        for item in decoded
    ):
        raise ValueError("selected audio pointer metadata is malformed")
    return tuple((item[0], item[1]) for item in decoded)


def _encode_audio_pointer_metadata(metadata: tuple[tuple[str, str], ...]) -> str:
    """Encode pointer metadata without decoding non-finite scalar values.

    Returns:
        The bounded JSON scalar.

    Raises:
        ValueError:
            If the encoded metadata exceeds the bounded selection limit.
    """
    encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _AUDIO_POINTER_METADATA_MAX_BYTES:
        raise ValueError("audio pointer metadata exceeds the bounded selection limit")
    return encoded


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


def _process_native_programmes(
    *,
    source: object,
    settings: PipelineSettings,
    candidates: c.Iterable[NativeCandidate | tuple[str, object, object]],
    ledger: Ledger,
    hub: object,
    vad: VADBackend | None,
    ctc: CTCBackend | None,
    report: BuildReport,
    log: MetadataLog,
    audit_reservoir: object | None = None,
) -> None:
    """Retrieve, segment, and publish one selected programme at a time.

    Raises:
        ValueError:
            If a selected programme produces no shard or has invalid state.
    """
    from hviske.p1_ledger import ShardAllocation
    from hviske.p1_source import InvalidSourceRecord, InvalidSourceTimestamp

    for programme_number, candidate in enumerate(candidates, start=1):
        # Count at consumption time so skipped and retryable candidates are included
        # without pre-counting bounded lists or retaining source identifiers.
        report.selected_programmes += 1
        report.preflight = dataclasses.replace(
            report.preflight, selected_programmes=report.selected_programmes
        )
        logger.info("Programme %d start", programme_number)
        report.max_in_flight = max(report.max_in_flight, 1)
        if isinstance(candidate, NativeCandidate):
            file_id = candidate.file_id
            metadata = candidate.metadata
            audio_pointer: object = candidate.audio_pointer
            transcript_pointer = candidate.transcript_pointer
            shard = candidate.audio_pointer.shard
        else:
            file_id, metadata, locator = candidate
            shard, transcript_pointer = t.cast(tuple[object, object], locator)
            # Compatibility for direct callers of this private orchestration helper.
            # Production candidates always carry an AudioPointer from discovery.
            audio_pointer = shard
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
        )
        state = ledger.programme(programme_id).state.value
        if state == "rejected" or (
            settings.resume and state in {"purged", "verified", "sharded"}
        ):
            logger.info(
                "Programme %d terminal outcome: skipped (%s)", programme_number, state
            )
            continue
        try:
            started = time.monotonic()
            enforce_scratch_cap(settings)
            ledger.start_processing(programme_id)
            try:
                transcript = source.fetch_transcript(transcript_pointer)
            except TranscriptOverAudio:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.TRANSCRIPT_OVER_AUDIO.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            except InvalidSourceTimestamp:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.INVALID_TIMESTAMPS.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            except InvalidSourceRecord:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.INVALID_SOURCE_RECORD.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            enforce_scratch_cap(settings)
            omitted = getattr(transcript, "zero_duration_tokens_omitted", 0)
            untimed = getattr(transcript, "untimed_tokens_owned", 0)
            if omitted or untimed:
                if omitted:
                    report.normalization_counts["zero_duration_tokens_omitted"] = (
                        report.normalization_counts.get(
                            "zero_duration_tokens_omitted", 0
                        )
                        + omitted
                    )
                if untimed > omitted:
                    report.normalization_counts["untimed_tokens_owned"] = (
                        report.normalization_counts.get("untimed_tokens_owned", 0)
                        + untimed
                        - omitted
                    )
                log.write(
                    {
                        "event": "transcript_normalized",
                        "operation": "assign_source_text_ownership",
                        "count": untimed,
                    }
                )
            if getattr(transcript, "file_id", file_id) != file_id:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.INVALID_SOURCE_RECORD.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            text = getattr(transcript, "text", None)
            words = getattr(transcript, "words", None)
            if not isinstance(text, str) or not isinstance(words, tuple):
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.INVALID_SOURCE_RECORD.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            legacy_empty_result = (
                text == ""
                and not words
                and not isinstance(transcript, ParsedTranscript)
            )
            if not text.strip() and not legacy_empty_result:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.EMPTY_TEXT.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            if not words and not legacy_empty_result:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.NO_TIMED_WORDS.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            ambiguous = getattr(transcript, "ambiguous_source_text_records", 0)
            if ambiguous:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.AMBIGUOUS_SOURCE_TEXT.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            try:
                # Transcript qualification is deliberately independent of metadata
                # duration.  The source audio is the only duration authority.
                _validate_native_timestamps(words=transcript.words)
            except InvalidSourceTimestamp:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.INVALID_TIMESTAMPS.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            if isinstance(audio_pointer, AudioPointer):
                if audio_pointer.file_id != file_id:
                    _reject_native_programme(
                        ledger=ledger,
                        report=report,
                        log=log,
                        programme_id=programme_id,
                        source_file_id=file_id,
                        reason=RejectionCategory.MISSING_AUDIO.value,
                    )
                    purge_source_temporary(getattr(source, "last_temporary", None))
                    continue
            try:
                parsed_audio = source.fetch_audio(pointer=audio_pointer)
            except InvalidSourceRecord:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.MISSING_AUDIO.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            enforce_scratch_cap(settings)
            try:
                audio = _decoded_native_audio(
                    parsed_audio,
                    file_id=file_id,
                    max_decoded_audio_bytes=settings.max_decoded_audio_bytes,
                )
                duration = _decoded_duration_ms(
                    audio=audio,
                    sampling_rate=int(getattr(parsed_audio, "sampling_rate", 0)),
                )
            except (InvalidSourceRecord, TypeError, ValueError):
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.MISSING_AUDIO.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            if getattr(parsed_audio, "file_id", file_id) != file_id:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.MISSING_AUDIO.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            ledger.record_source_duration(programme_id, duration)
            try:
                _validate_native_timestamps(
                    words=transcript.words, duration_ms=duration
                )
            except TranscriptOverAudio:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.TRANSCRIPT_OVER_AUDIO.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            except InvalidSourceTimestamp:
                _reject_native_programme(
                    ledger=ledger,
                    report=report,
                    log=log,
                    programme_id=programme_id,
                    source_file_id=file_id,
                    reason=RejectionCategory.INVALID_TIMESTAMPS.value,
                )
                purge_source_temporary(getattr(source, "last_temporary", None))
                continue
            programme = SourceProgramme(
                file_id=file_id,
                duration_ms=duration,
                words=tuple(transcript.words),
                transcript_text=transcript.text,
            )
            if settings.alignment_method == "timestamp-native:p1-transcripts.words":
                # P1 v7 deliberately consumes the source word timestamps.  Keep the
                # generic model-backed path below intact for future datasets, but do
                # not even construct its backends for the active source.
                active_vad = None
                active_ctc: CTCBackend = TimestampAlignmentBackend()
            else:
                if vad is None:
                    logger.info("P1 model stage: loading VAD backend")
                    vad = make_silero_vad(settings)
                    logger.info("P1 model stage: VAD backend ready")
                if ctc is None:
                    logger.info("P1 model stage: loading CTC backend")
                    ctc = make_ctc_backend(settings)
                    logger.info("P1 model stage: CTC backend ready")
                active_vad = vad
                active_ctc = ctc
            result = segment_programme(
                words=programme.words,
                audio=audio,
                source_file_id=file_id,
                source_duration_ms=programme.duration_ms,
                segmentation=settings.segmentation,
                normalisation=settings.normalisation,
                ctc=active_ctc,
                pipeline_version=settings.pipeline_version,
                pipeline_config_sha256=settings.pipeline_digest,
                vad=active_vad,
                sampling_rate=parsed_audio.sampling_rate,
                channels=parsed_audio.channels,
                source_locator={
                    "source_repository": settings.source_audio_repository,
                    "source_revision": settings.source_audio_revision,
                    "source_shard_path": shard.path,
                    "source_row_group": getattr(audio_pointer, "row_group", 0),
                    "source_row_index": getattr(audio_pointer, "row_index", 0),
                    "source_shard_byte_size": shard.byte_size,
                    "source_duration_ms": programme.duration_ms,
                },
            )
            report.accepted_segments += len(result.rows)
            report.rejected += len(result.rejections)
            if audit_reservoir is not None and result.audit_candidates:
                getattr(audit_reservoir, "add")(result.audit_candidates)
            for _, reason in result.rejections:
                report.rejection_counts[reason] = (
                    report.rejection_counts.get(reason, 0) + 1
                )
            output = settings.scratch_root / "staging" / file_id
            written = write_shards(
                result.rows, output_dir=output, target_bytes=settings.target_shard_bytes
            )
            enforce_scratch_cap(settings)
            if not written.shards:
                if not result.rows:
                    reason_counts = {
                        reason: sum(
                            1 for _, value in result.rejections if value == reason
                        )
                        for _, reason in result.rejections
                    }
                    if not reason_counts:
                        reason_counts = {
                            RejectionCategory.NO_ACCEPTED_SEGMENTS.value: 1
                        }
                        # Proposal-level rejection counts are already included above;
                        # an empty proposal set still needs one terminal programme
                        # outcome so that the run report has a durable count.
                        report.rejected += 1
                        report.rejection_counts[
                            RejectionCategory.NO_ACCEPTED_SEGMENTS.value
                        ] = (
                            report.rejection_counts.get(
                                RejectionCategory.NO_ACCEPTED_SEGMENTS.value, 0
                            )
                            + 1
                        )
                    ledger.reject_programme(
                        programme_id,
                        reason=next(iter(reason_counts)),
                        rejection_counts=reason_counts,
                        accepted_count=0,
                        rejected_count=sum(reason_counts.values()),
                    )
                    purge_source_temporary(getattr(source, "last_temporary", None))
                    report.processed += 1
                    log.write(
                        {
                            "event": "programme_rejected",
                            "reason": next(iter(reason_counts)),
                        }
                    )
                    logger.info(
                        "Programme %d terminal outcome: rejected (%s)",
                        programme_number,
                        next(iter(reason_counts)),
                    )
                    continue
                raise ValueError("programme produced no publication shard")
            allocations = tuple(
                ShardAllocation(
                    local_path=item.path,
                    remote_path=f"data/train/{programme_id}-{ordinal:05d}.parquet",
                    sha256=item.evidence.sha256,
                    byte_size=item.evidence.byte_size,
                    row_count=item.evidence.row_count,
                )
                for ordinal, item in enumerate(written.shards)
            )
            audit_candidates = _record_audit_candidates(
                rows=result.rows,
                path=None,
                repository=settings.target_private_repo,
                revision=None,
                remote_paths=tuple(item.remote_path for item in allocations),
                row_counts=tuple(item.row_count for item in allocations),
                local_paths=tuple(Path(item.local_path) for item in allocations),
            )
            batch, shard_records = ledger.allocate_batch_with_shards(
                programme_id,
                allocations,
                accepted_count=len(result.rows),
                rejected_count=len(result.rejections),
                processed_duration_ms=int((time.monotonic() - started) * 1000),
                rejection_counts={
                    reason: sum(1 for _, value in result.rejections if value == reason)
                    for _, reason in result.rejections
                },
                audit_candidates=audit_candidates,
            )
            if audit_reservoir is not None and audit_candidates:
                getattr(audit_reservoir, "add")(audit_candidates)
            report.shard_count += len(shard_records)
            purge_source_temporary(getattr(source, "last_temporary", None))
            ledger.mark_source_temps_purged(programme_id, evidence={"deleted": True})
            _publish_native_pending(
                hub=hub,
                settings=settings,
                ledger=ledger,
                pending=tuple(
                    _local_shard_from_record(record) for record in shard_records
                ),
                pending_ids=tuple(record.shard_id for record in shard_records),
                batch_id=batch.batch_id,
                audit_rows=result.rows,
                audit_reservoir=audit_reservoir,
            )
            report.processed += 1
            logger.info("Programme %d terminal outcome: accepted", programme_number)
        except InvalidSourceTimestamp:
            _reject_native_programme(
                ledger=ledger,
                report=report,
                log=log,
                programme_id=programme_id,
                source_file_id=file_id,
                reason=RejectionCategory.INVALID_TIMESTAMPS.value,
            )
            purge_source_temporary(getattr(source, "last_temporary", None))
            logger.info(
                "Programme %d terminal outcome: rejected (invalid timestamps)",
                programme_number,
            )
            continue
        except Exception as exc:
            category = _safe_exception_category(exc)
            current = ledger.programme(programme_id)
            if current.state.value == "processing":
                ledger.transition_programme(
                    programme_id, target=_state("retryable"), last_error=category
                )
            log.write({"event": "programme_error", "reason": category})
            logger.info(
                "Programme %d terminal outcome: error (%s)", programme_number, category
            )
            raise
        finally:
            gc.collect()
        enforce_scratch_cap(settings)
    enforce_scratch_cap(settings)


def _decoded_duration_ms(*, audio: np.ndarray, sampling_rate: int) -> int:
    """Derive a nearest-millisecond duration from decoded frames.

    The decoded source array is frames-by-channels (or a mono vector).  Channel count
    must never affect the frame count used for duration.

    Returns:
        The nearest-millisecond duration of the decoded source.

    Raises:
        ValueError:
            If the decoded array or sampling rate is unusable.
    """
    if sampling_rate <= 0:
        raise ValueError("decoded audio has an invalid sampling rate")
    if audio.ndim == 1:
        frames = audio.shape[0]
    elif audio.ndim == 2 and audio.shape[1] > 0:
        frames = audio.shape[0]
    else:
        raise ValueError("decoded audio has an invalid shape")
    if frames <= 0:
        raise ValueError("decoded audio is empty")
    return (int(frames) * 1000 + sampling_rate // 2) // sampling_rate


def _decoded_native_audio(
    parsed_audio: object, *, file_id: str, max_decoded_audio_bytes: int
) -> np.ndarray:
    """Decode and validate one native audio payload under the pipeline cap.

    The source adapter is not trusted to enforce the bound: callers may inject a
    predecoded ``ParsedAudio`` object, and compressed bytes may be returned by a
    custom adapter. The final float32 array is therefore checked independently.

    Returns:
        A decoded source array; resampling is performed by ``segment_programme``.

    Raises:
        TypeError:
            If the source did not return a parsed array payload.
        ValueError:
            If the configured cap is not positive or the array shape is invalid.
        DecodedAudioTooLarge:
            If the decoded PCM exceeds the configured cap.
    """
    from hviske.p1_source import DecodedAudioTooLarge, ParsedAudio, parse_audio_row

    if max_decoded_audio_bytes <= 0:
        raise ValueError("max_decoded_audio_bytes must be positive")
    if not isinstance(parsed_audio, ParsedAudio):
        raise TypeError("native source must return ParsedAudio")
    value = parsed_audio.value
    if isinstance(value, (bytes, bytearray)):
        parsed_audio = parse_audio_row(
            {
                "file_id": file_id,
                "audio": {
                    "bytes": bytes(value),
                    "sampling_rate": parsed_audio.sampling_rate,
                    "channels": parsed_audio.channels,
                },
            },
            expected_file_id=file_id,
            max_decoded_audio_bytes=max_decoded_audio_bytes,
        )
        value = parsed_audio.value
    if not isinstance(value, np.ndarray):
        raise TypeError("native source audio must decode to an ndarray")
    if value.ndim not in (1, 2) or (value.ndim == 2 and value.shape[1] <= 0):
        raise ValueError("native source audio has an invalid shape")
    if value.size <= 0 or value.nbytes > max_decoded_audio_bytes:
        raise DecodedAudioTooLarge("native source audio exceeds the configured limit")
    decoded = np.asarray(value, dtype=np.float32)
    if decoded.ndim not in (1, 2) or (decoded.ndim == 2 and decoded.shape[1] <= 0):
        raise ValueError("native source audio has an invalid shape")
    if decoded.size <= 0 or decoded.nbytes > max_decoded_audio_bytes:
        raise DecodedAudioTooLarge("native source audio exceeds the configured limit")
    if not np.isfinite(decoded).all():
        raise ValueError("native source audio must be finite")
    return decoded


def _local_shard_from_record(record: object) -> object:
    """Build the publisher's local shard from durable ledger identity.

    Returns:
        A publisher-compatible local shard.

    Raises:
        ValueError:
            If the ledger record has no local path.
    """
    from hviske.p1_publish import LocalShard

    local_path = getattr(record, "local_path", None)
    if local_path is None:
        raise ValueError("ledger shard has no durable local path")
    return LocalShard(
        path=Path(local_path),
        repo_path=str(getattr(record, "path")),
        row_count=int(getattr(record, "row_count")),
    )


def _publish_native_pending(
    *,
    hub: object,
    settings: PipelineSettings,
    ledger: Ledger,
    pending: c.Sequence[object],
    pending_ids: c.Sequence[str],
    batch_id: str,
    audit_rows: c.Sequence[object] = (),
    audit_reservoir: object | None = None,
) -> None:
    """Publish a ledger-allocated batch and add locators after immutable commit."""
    del pending_ids
    publish_pending(
        hub=hub,
        settings=settings,
        ledger=ledger,
        batch_id=batch_id,
        pending=pending,
        pending_ids=(),
        audit_rows=audit_rows,
        audit_reservoir=audit_reservoir,
    )


def publish_pending(
    *,
    hub: object,
    settings: PipelineSettings,
    ledger: Ledger,
    batch_id: str,
    pending: c.Sequence[object],
    pending_ids: c.Sequence[str],
    audit_rows: c.Sequence[object] = (),
    audit_reservoir: object | None = None,
) -> object:
    """Publish a complete ledger batch through the shared verified publisher.

    Returns:
        Verified publication evidence.

    Raises:
        ValueError:
            If the batch has no local shards.
        P1PreflightError:
            If the target card does not prove the active v7 contract.
    """
    from hviske.p1_publish import HubClient, LocalShard, publish_batch

    if settings.alignment_method == "timestamp-native:p1-transcripts.words":
        if not _target_card_is_v7(
            hub=hub,
            repo_id=settings.target_private_repo,
            expected_digest=settings.pipeline_digest,
        ):
            raise P1PreflightError(
                "refusing payload commit: target card is not the P1 v7 contract"
            )

    shards = t.cast(c.Sequence[LocalShard], pending)
    try:
        record = ledger.batch(batch_id)
    except KeyError:
        record = ledger.register_batch(
            batch_id, pipeline_digest=settings.pipeline_digest
        )
        ledger.transition_batch(batch_id, _state("processing"))
        for shard_id in pending_ids:
            ledger.attach_shard(batch_id, shard_id)
        ledger.transition_batch(batch_id, _state("sharded"))
        record = ledger.batch(batch_id)
    if record.state in {_state("verified"), _state("purged")}:
        from hviske.p1_publish import HubClient, verify_batch

        evidence = verify_batch(
            t.cast(HubClient, hub),
            settings.target_private_repo,
            batch_id,
            ledger=ledger,
        )
        if ledger.batch(batch_id).state is _state("verified"):
            expected = {
                Path(item.local_path): (item.byte_size, item.sha256)
                for item in ledger.shards(batch_id)
                if item.local_path is not None
            }
            _unlink_recovered(tuple(expected), expected=expected)
            ledger.purge_batch(
                batch_id, evidence={"deleted": True, "kind": "publication-artifact"}
            )
        ledger.finalise_batch_children(batch_id)
        return evidence.model_copy(update={"state": _state("purged")})
    if not shards:
        raise ValueError("publication batch has no local shards")
    ledger_shards = ledger.shards(batch_id)
    programme_ids = tuple(
        sorted(
            {
                item.programme_id
                for item in ledger_shards
                if item.programme_id is not None
            }
        )
    )
    local_paths = tuple(shard.path for shard in shards)
    remote_paths = tuple(item.path for item in ledger_shards)
    row_counts = tuple(item.row_count for item in ledger_shards)
    if not (len(shards) == len(remote_paths) == len(row_counts)):
        raise ValueError("pending shards differ from their durable ledger records")
    if audit_reservoir is not None:
        stored_candidates = ledger.audit_candidates(batch_id)
        if stored_candidates:
            getattr(audit_reservoir, "add")(stored_candidates)
        elif audit_rows:
            candidates = _record_audit_candidates(
                rows=audit_rows,
                path=None,
                repository=settings.target_private_repo,
                revision=None,
                remote_paths=remote_paths,
                row_counts=row_counts,
                local_paths=local_paths,
            )
            getattr(audit_reservoir, "add")(candidates)

    def purge(paths: tuple[Path, ...]) -> None:
        for path in paths:
            if path.is_file() and not path.is_symlink():
                path.unlink()

    def commit_recorded(commit_id: str) -> None:
        if audit_reservoir is not None:
            updater = getattr(audit_reservoir, "update_remote_locators", None)
            if callable(updater):
                updater(
                    repository=settings.target_private_repo,
                    revision=commit_id,
                    local_paths=local_paths,
                    remote_paths=remote_paths,
                    row_counts=row_counts,
                    parquet_sha256=tuple(_file_sha256(shard.path) for shard in shards),
                )

    evidence = publish_batch(
        t.cast(HubClient, hub),
        settings.target_private_repo,
        batch_id,
        shards,
        programme_count=record.programme_count,
        rejection_counts={
            RejectionCategory(key): value
            for key, value in record.rejection_counts.items()
        },
        staging_dir=Path(shards[0].path).parent,
        ledger=ledger,
        purge_callback=purge,
        commit_recorded=commit_recorded,
    )
    ledger.finalise_batch_children(batch_id)
    for programme_id in programme_ids:
        programme = ledger.programme(programme_id)
        if programme.state is _state("sharded"):
            ledger.transition_programme(
                programme_id, _state("committed"), commit_id=evidence.commit_id
            )
            ledger.transition_programme(programme_id, _state("verified"))
        if ledger.batch(batch_id).state is _state("purged"):
            for shard in ledger.shards(batch_id):
                if shard.state is _state("verified"):
                    ledger.transition_shard(shard.shard_id, _state("purged"))
            if ledger.programme(programme_id).state is _state("verified"):
                ledger.transition_programme(programme_id, _state("purged"))
    return evidence


class P1PreflightError(RuntimeError):
    """Raised when a safety gate fails before source audio retrieval."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_audit_candidates(
    *,
    rows: c.Sequence[object],
    path: Path | None,
    repository: str,
    revision: str | None,
    remote_paths: tuple[str, ...],
    row_counts: tuple[int, ...],
    local_paths: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    """Build bounded blinded audit metadata with local or remote locators.

    Returns:
        The metadata-only candidates with local or committed locators.

    Raises:
        TypeError:
            If an audit row is neither a mapping nor a contract model.
        ValueError:
            If local and remote shard paths are mismatched.
    """
    candidates: list[dict[str, object]] = []
    row_index = 0
    if local_paths is not None and len(local_paths) != len(remote_paths):
        raise ValueError("local and remote shard paths must have equal lengths")
    for shard_number, (shard_path, shard_rows) in enumerate(
        zip(remote_paths, row_counts, strict=True)
    ):
        for offset, row in enumerate(rows[row_index : row_index + shard_rows]):
            model_dump = getattr(row, "model_dump", None)
            if callable(model_dump):
                raw_candidate = t.cast(dict[str, object], model_dump(mode="python"))
            elif isinstance(row, Mapping):
                raw_candidate = dict(row)
            else:
                raise TypeError("audit rows must be mappings or contract models")
            from hviske.p1_validation import _metadata_copy, _metadata_digest

            metadata_digest = _metadata_digest(raw_candidate)
            candidate = _metadata_copy(raw_candidate)
            candidate["status"] = "accepted"
            # Keep the digest produced from the complete OutputRow while dropping
            # transcript and audio fields from the reservoir candidate.
            candidate["_p1_metadata_sha256"] = metadata_digest
            candidate["metadata_sha256"] = metadata_digest
            if local_paths is None:
                candidate.update(
                    {
                        "repository": repository,
                        "revision": revision,
                        "parquet_path": shard_path,
                        "row_locator": offset,
                    }
                )
            else:
                candidate.update(
                    {
                        "local_path": str(local_paths[shard_number]),
                        "local_row_locator": offset,
                    }
                )
            candidates.append(candidate)
            row_index += 1
    if not candidates:
        return []
    if path is not None:
        from hviske.p1_validation import create_blinded_audit_manifest

        selected = create_blinded_audit_manifest(
            candidates, accepted_quota=min(200, len(candidates)), rejected_quota=0
        )
        with path.open("a", encoding="utf-8") as stream:
            for candidate in selected:
                stream.write(json.dumps(candidate, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return candidates


def _state(value: str) -> LedgerState:
    """Resolve a ledger state while keeping transition calls concise.

    Returns:
        The corresponding ledger state.
    """
    return LedgerState(value)


def _unlink_recovered(
    paths: tuple[Path, ...], expected: Mapping[Path, tuple[int, str]] | None = None
) -> None:
    """Remove only surviving regular files matching durable recovery evidence."""
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        evidence = None if expected is None else expected.get(path)
        if evidence is not None:
            size, digest = evidence
            if path.stat().st_size != size:
                continue
            checksum_builder = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum_builder.update(chunk)
            if checksum_builder.hexdigest() != digest:
                continue
        path.unlink()


def _reject_native_programme(
    *,
    ledger: Ledger,
    report: BuildReport,
    log: MetadataLog,
    programme_id: str,
    source_file_id: str,
    reason: str,
) -> None:
    """Persist a terminal source rejection without recording source content."""
    del source_file_id
    ledger.reject_programme(
        programme_id,
        reason=reason,
        rejection_counts={reason: 1},
        accepted_count=0,
        rejected_count=1,
    )
    report.processed += 1
    report.rejected += 1
    report.rejection_counts[reason] = report.rejection_counts.get(reason, 0) + 1
    log.write({"event": "programme_rejected", "reason": reason})


def _safe_exception_category(exc: Exception) -> str:
    """Return a bounded error category without serialising exception details."""
    if exc.__class__.__name__ in {
        "AllowListError",
        "PrivacyError",
        "PublicationError",
        "VerificationError",
    }:
        return "publication_error"
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)):
        return "infrastructure_error"
    if isinstance(exc, (RuntimeError, TypeError, ValueError)):
        return "runtime_error"
    return "unexpected_error"


def _validate_native_timestamps(
    *, words: c.Iterable[object], duration_ms: int | None = None
) -> None:
    """Validate the source timeline before constructing ``SourceProgramme``.

    ``duration_ms`` is optional because transcript qualification must happen before
    audio retrieval.  When supplied, it is the decoded audio duration, never metadata
    from the source row.

    Raises:
        InvalidSourceTimestamp:
            If a word is malformed or overlaps a previous word.
        TranscriptOverAudio:
            If a word endpoint exceeds the decoded programme duration.
    """
    from hviske.p1_source import InvalidSourceTimestamp

    previous_end = 0
    for word in words:
        start = getattr(word, "start_ms", None)
        end = getattr(word, "end_ms", None)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or start < previous_end
        ):
            raise InvalidSourceTimestamp("source word timeline is invalid")
        if duration_ms is not None and end > duration_ms:
            raise TranscriptOverAudio("source word endpoint exceeds decoded audio")
        previous_end = end


def enforce_scratch_cap(settings: PipelineSettings) -> None:
    """Reject a run that exceeds its hard scratch quota.

    Raises:
        P1PreflightError:
            If regular files exceed the configured quota.
    """
    if directory_size(settings.scratch_root) > settings.max_scratch_bytes:
        raise P1PreflightError("scratch hard cap exceeded")


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
        case_folding=settings.normalisation.case_folding,
    )


def make_silero_vad(settings: PipelineSettings) -> VADBackend:
    """Construct the exact pinned Silero asset from the P1 scratch cache.

    Returns:
        The real pinned Silero VAD backend.
    """
    from hviske.p1_models import make_silero_vad as make_pinned_silero_vad

    return make_pinned_silero_vad(
        settings.scratch_root / "models", device=settings.device
    )


def purge_source_temporary(value: object) -> None:
    """Delete only a source adapter's explicitly owned temporary path."""
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_file() and not path.is_symlink():
            path.unlink()


def _recover_native_batches(
    *,
    source: object,
    settings: PipelineSettings,
    ledger: Ledger,
    hub: object,
    audit_reservoir: object | None = None,
) -> None:
    """Reconcile every durable local and remote item after a restart.

    Recovery uses only the exact local path persisted by the atomic allocation.  A
    basename or a count of files is never evidence of identity.  Remote verification
    remains mandatory even when all local files are present.
    """
    del source
    from hviske.p1_publish import HubClient, _manifest_bytes, verify_batch

    unattached, _ = ledger.recovery_work()
    for record in unattached:
        local_path = getattr(record, "local_path", None)
        if local_path is not None:
            ledger.reconcile_local_shard(record.shard_id, Path(local_path))

    for batch, records in ledger.reconstruct_work():
        manifest_path = next(
            (
                Path(record.local_path).parent / "manifests" / f"{batch.batch_id}.json"
                for record in records
                if record.local_path is not None
            ),
            None,
        )
        paths = tuple(
            Path(record.local_path)
            for record in records
            if record.local_path is not None
            and ledger.reconcile_local_shard(record.shard_id, Path(record.local_path))
        )
        if (
            batch.commit_id is None
            and batch.state is _state("sharded")
            and records
            and len(paths) == len(records)
        ):
            _publish_native_pending(
                hub=hub,
                settings=settings,
                ledger=ledger,
                pending=tuple(_local_shard_from_record(record) for record in records),
                pending_ids=tuple(record.shard_id for record in records),
                batch_id=batch.batch_id,
                audit_reservoir=audit_reservoir,
            )
            continue
        if batch.commit_id is None:
            continue
        if audit_reservoir is not None and all(
            record.local_path is not None for record in records
        ):
            stored_candidates = ledger.audit_candidates(batch.batch_id)
            if stored_candidates:
                getattr(audit_reservoir, "add")(stored_candidates)
            updater = getattr(audit_reservoir, "update_remote_locators", None)
            if callable(updater):
                updater(
                    repository=settings.target_private_repo,
                    revision=batch.commit_id,
                    local_paths=tuple(
                        t.cast(str, record.local_path) for record in records
                    ),
                    remote_paths=tuple(record.path for record in records),
                    row_counts=tuple(record.row_count for record in records),
                    parquet_sha256=tuple(record.sha256 for record in records),
                )
        verify_batch(
            t.cast(HubClient, hub),
            settings.target_private_repo,
            batch.batch_id,
            ledger=ledger,
            manifest_path=manifest_path,
            local_paths=paths,
        )
        expected_local: dict[Path, tuple[int, str]] = {
            Path(record.local_path): (record.byte_size, record.sha256)
            for record in records
            if record.local_path is not None
        }
        if manifest_path is not None:
            manifest = _manifest_bytes(
                batch_id=batch.batch_id,
                shards=tuple(
                    ShardEvidence(
                        path=record.path,
                        byte_size=record.byte_size,
                        row_count=record.row_count,
                        sha256=record.sha256,
                    )
                    for record in records
                ),
                programme_count=batch.programme_count,
                rejection_counts={
                    RejectionCategory(key): value
                    for key, value in batch.rejection_counts.items()
                },
            )
            expected_local[manifest_path] = (
                len(manifest),
                hashlib.sha256(manifest).hexdigest(),
            )
        _unlink_recovered(tuple(expected_local), expected=expected_local)
        recovered_batch = ledger.batch(batch.batch_id)
        if recovered_batch.state is _state("verified"):
            ledger.purge_batch(
                batch.batch_id,
                evidence={"deleted": True, "kind": "publication-artifact"},
            )
        if ledger.batch(batch.batch_id).state is _state("purged"):
            ledger.finalise_batch_children(batch.batch_id)
    for batch in ledger.purged_batches_with_pending_children():
        ledger.finalise_batch_children(batch.batch_id)


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
        permitted_use=(
            "Private commercial data preparation and authorised internal use only."
        ),
        private_access_terms="Access is restricted to authorised syv.ai members.",
        alignment_method=(
            "pipeline_version: p1-segmentation-7; "
            "alignment_method: timestamp-native:p1-transcripts.words; "
            f"pipeline_config_sha256: {settings.pipeline_digest}; "
            "Source word timestamps are authoritative; boundaries are the first "
            "timed word start and last timed word end. Text follows "
            f"{P1_RUNTIME_CONTRACT.normalisation_version} and "
            f"{P1_RUNTIME_CONTRACT.normalisation_source_text_ownership}."
        ),
        field_schema=(
            "p1-segments-v2 OutputRow schema; alignment_backend is timestamp-native, "
            "alignment_score_type is not_applicable:source_timestamps, and all "
            "acoustic evidence fields are null."
        ),
        known_limitations="Timestamp quality is limited by the source word timestamps.",
        rejection_policy=(
            "Invalid, empty, short or overlong timestamp proposals, source timeline "
            "defects, speaker mixing, and undecodable programmes are recorded in the "
            "ledger; unexpected source and programming failures remain fatal."
        ),
        source_revisions=json.dumps(
            {
                "audio": {
                    "repository": settings.source_audio_repository,
                    "revision": settings.source_audio_revision,
                },
                "transcripts": {
                    "repository": settings.source_transcript_repository,
                    "revision": settings.source_transcript_revision,
                },
            },
            sort_keys=True,
        ),
        model_revisions=None,
        dataset_license=json.dumps(
            settings.dataset_license.model_dump(mode="json"), sort_keys=True
        ),
    )
    license_path = Path(__file__).resolve().parents[2] / "LICENSE-DATASET"
    initialise_private_dataset(
        t.cast(HubClient, hub),
        settings.target_private_repo,
        card=card,
        license_text=license_path.read_text(encoding="utf-8"),
    )


def make_hub() -> object:
    """Construct the authenticated Hub adapter only for a non-plan run.

    Returns:
        An authenticated Hub adapter.
    """
    from hviske.p1_publish import HfApiAdapter

    return HfApiAdapter()


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
    timestamp_native = settings.alignment_method == (
        "timestamp-native:p1-transcripts.words"
    )
    try:
        validate_p1_runtime_contract(
            pipeline_version=settings.pipeline_version,
            ctc=None if timestamp_native else _settings_ctc_contract(settings),
            alignment_method=settings.alignment_method,
            normalisation=settings.normalisation,
            dataset_license=settings.dataset_license,
        )
    except ValueError as exc:
        raise P1PreflightError(str(exc)) from exc
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
    # The source plan has already resolved both immutable repository revisions.
    source_revision_ok = True
    # The v7 path has no model or device dependency.  In particular, do not call
    # the legacy revision verifier: it may inspect inactive future-alignment metadata.
    model_revision_ok = (
        True
        if timestamp_native
        else check_model_revisions(source, settings.model_revisions)
    )
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
    target = target_privacy(hub, settings.target_private_repo, settings.pipeline_digest)
    cuda = (
        {"checked": False, "reason": "not_applicable:timestamp-native"}
        if timestamp_native
        else cuda_status(settings.device)
    )
    checks = {
        "source_revisions": source_revision_ok,
        "model_provenance_not_applicable": timestamp_native,
        "normalisation_compatible": True,
        "free_space": free_bytes >= required,
        "scratch_quota": scratch_bytes <= settings.max_scratch_bytes,
        "scratch_budget": scratch_bytes + required <= settings.max_scratch_bytes,
        "source_object_cap": maximum_source_bytes <= settings.max_source_bytes,
        "cuda_device_checked": (
            False if timestamp_native else bool(cuda.get("checked", False))
        ),
        "target_checked": bool(target.get("checked", False)),
        "target_contract_v7": bool(target.get("contract_v7", False)),
    }
    if not source_revision_ok:
        raise P1PreflightError("source revision is not available at the pinned SHA")
    if not model_revision_ok:
        raise P1PreflightError("model revision is not available at the pinned SHA")
    if settings.mode != "plan" and not target.get("private", False):
        raise P1PreflightError("target repository is not demonstrably private")
    if settings.mode not in {"plan", "initialise"} and not target.get(
        "contract_v7", False
    ):
        raise P1PreflightError(
            "target card does not prove the P1 v7 timestamp-native contract"
        )
    return PreflightReport(
        mode=settings.mode,
        selected_programmes=selected_programmes,
        maximum_source_bytes=maximum_source_bytes,
        required_scratch_bytes=required,
        free_bytes=free_bytes,
        scratch_bytes=scratch_bytes,
        source_revisions=source_revisions,
        model_revisions=({} if timestamp_native else settings.model_revisions),
        cuda=cuda,
        target=target,
        checks=checks,
    )


def _settings_ctc_contract(settings: PipelineSettings) -> CTCContract:
    """Return the CTC contract with all externally exposed settings applied."""
    model = settings.ctc_contract.model.model_copy(
        update={
            "repository": RepositoryRevision(
                repository=settings.ctc_model_repository,
                revision=settings.ctc_model_revision,
            )
        }
    )
    return settings.ctc_contract.model_copy(update={"model": model})


def check_model_revisions(source: object, revisions: dict[str, object]) -> bool:
    """Verify every pinned model without loading model weights.

    The VAD is a GitHub asset, while the CTC and anomaly models are Hub models.  A
    real source supplies the authenticated Hub API used for both model checks.  Small
    offline fakes intentionally opt out by not exposing a client.

    Returns:
        ``True`` when all pinned coordinates have been verified.
    """
    if getattr(source, "local_root", None) is not None:
        return True
    client_getter = getattr(source, "_client", None)
    if client_getter is None:
        checker = getattr(source, "check_model_revisions", None)
        return True if checker is None else bool(checker(revisions=revisions))
    from hviske.p1_models import verify_hub_model_revision, verify_silero_vad_revision

    api = client_getter()
    vad = t.cast(dict[str, object], revisions["vad"])
    verify_silero_vad_revision(
        repository=str(vad["repository"]),
        revision=str(vad["revision"]),
        model_path=str(vad.get("model_path")),
        expected_blob=str(vad.get("model_blob")),
        expected_sha256=str(vad.get("model_sha256")),
    )
    for name in ("ctc", "anomaly"):
        coordinate = t.cast(dict[str, object], revisions[name])
        verify_hub_model_revision(
            repository=str(coordinate["repository"]),
            revision=str(coordinate["revision"]),
            api=api,
        )
    return True


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
