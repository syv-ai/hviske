"""Privacy-safe, structural sanity gate for the P1 v8 dataset.

The active gate retrieves a small, deterministic sample from one private immutable
revision. It validates publication structure and audio encoding only; transcription
agreement and model-backed quality checks are intentionally out of scope.
"""

from __future__ import annotations

import collections.abc as c
import hashlib
import io
import json
import logging
import re
import typing as t
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from .contracts import OUTPUT_SCHEMA, OutputRow
from .validation import _metadata_digest

logger = logging.getLogger(__name__)

PILOT_REPOSITORY = "syvai/p1-segments"
PIPELINE_VERSION = "p1-segmentation-8"
SCHEMA_VERSION = "p1-segments-v2"
SAMPLE_SIZE = 12
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MetadataRow = c.Mapping[str, object]


def run_v8_sanity_gate(
    candidates: c.Iterable[MetadataRow],
    *,
    retriever: object,
    pilot_head: str | None = None,
    seed: str | int = "p1-v8-dozen",
    report_path: Path | str | None = None,
    expected_pipeline_version: str = PIPELINE_VERSION,
    expected_pipeline_config_sha256: str | None = None,
) -> dict[str, object]:
    """Run the model-free post-pilot v8 structural gate.

    Args:
        candidates:
            Metadata-only records from the bounded audit manifest.
        retriever:
            A retriever which verifies private immutable rows, shard hashes, row
            hashes, and audio hashes before returning one row.
        pilot_head (optional):
            Expected complete immutable commit SHA. If omitted, the sole revision in
            the accepted candidates is used.
        seed (optional):
            Stable sample-selection seed. Defaults to ``p1-v8-dozen``.
        report_path (optional):
            Aggregate JSON destination. No candidate identifiers are written there.
        expected_pipeline_version (optional):
            Active pipeline version. Defaults to the v8 contract version.
        expected_pipeline_config_sha256 (optional):
            Active configuration digest, or the sole digest in audit evidence.

    Returns:
        An aggregate structural report safe to retain outside the private dataset.

    Raises:
        ValueError:
            If the explicitly supplied pilot head is not immutable.
    """
    rows = [dict(row) for row in candidates]
    accepted = [row for row in rows if _is_accepted_candidate(row)]
    expected_head = _resolve_pilot_head(accepted, pilot_head)
    selected = _select_candidates(accepted, seed=seed)
    expected_digest = _resolve_pipeline_digest(
        selected, expected_pipeline_config_sha256
    )
    report = _empty_report(expected_head=expected_head, seed=seed, selected=selected)
    report["pipeline_version"] = expected_pipeline_version
    if len(selected) < SAMPLE_SIZE:
        report["counts"] = {"accepted": len(accepted), "selected": len(selected)}
        _write_report(report_path, report)
        return report

    if expected_pipeline_version != PIPELINE_VERSION or expected_digest is None:
        report["counts"] = {
            "accepted": len(accepted),
            "selected": len(selected),
            "retrieved": 0,
            "structural_failures": len(selected),
            "pipeline_digests": 0,
        }
        _write_report(report_path, report)
        return report

    verify_repository = getattr(retriever, "verify_repository", None)
    if not callable(verify_repository):
        report["counts"] = {
            "accepted": len(accepted),
            "selected": len(selected),
            "retrieved": 0,
            "structural_failures": len(selected),
            "pipeline_digests": 0,
        }
        report["pass"] = False
        _write_report(report_path, report)
        return report
    try:
        verify_repository()
        t.cast(dict[str, bool], report["checks"])["private_immutable_revision"] = True
    except Exception as error:
        logger.warning("v8 repository verification failed: %s", error)
        report["counts"] = {
            "accepted": len(accepted),
            "selected": len(selected),
            "retrieved": 0,
            "structural_failures": len(selected),
            "pipeline_digests": 0,
        }
        report["pass"] = False
        _write_report(report_path, report)
        return report

    structural_failures = 0
    retrievals = 0
    digests: set[str] = set()
    for ordinal, candidate in enumerate(selected, 1):
        try:
            if expected_head is None:
                raise ValueError("an explicit final pilot HEAD is required")
            rebound_candidate = _rebind_candidate(candidate, pilot_head=expected_head)
            _validate_candidate_locator(
                rebound_candidate, pilot_head=expected_head, repository=PILOT_REPOSITORY
            )
            row = _retrieve_row(retriever, rebound_candidate)
            retrievals += 1
            audio = _validate_row(
                row=row,
                candidate=candidate,
                expected_pipeline_version=expected_pipeline_version,
                expected_pipeline_config_sha256=expected_digest,
            )
            duration = row.get("duration_ms")
            if not isinstance(duration, int) or isinstance(duration, bool):
                raise ValueError("retrieved row has invalid duration metadata")
            _decode_audio(audio, duration_ms=duration)
            digest = row.get("pipeline_config_sha256")
            if not isinstance(digest, str) or digest != expected_digest:
                raise ValueError("retrieved row has a different pipeline digest")
            digests.add(digest)
        except Exception as error:
            structural_failures += 1
            logger.warning(
                "v8 sanity sample %d/%d failed structural checks: %s",
                ordinal,
                SAMPLE_SIZE,
                error,
            )
            continue
        logger.info("v8 sanity sample %d/%d validated", ordinal, SAMPLE_SIZE)

    checks = t.cast(dict[str, bool], report["checks"])
    if retrievals == SAMPLE_SIZE and structural_failures == 0:
        for name in (
            "hashes",
            "ogg_opus_16khz_mono",
            "exact_schema",
            "timestamp_native_v8",
            "exact_duration_and_bounds",
            "trainable_text",
            "timed_anchor_single_speaker",
        ):
            checks[name] = True
        checks["one_pipeline_digest"] = len(digests) == 1
    report["counts"] = {
        "accepted": len(accepted),
        "selected": len(selected),
        "retrieved": retrievals,
        "structural_failures": structural_failures,
        "pipeline_digests": len(digests),
    }
    report["pass"] = bool(
        len(selected) == SAMPLE_SIZE
        and retrievals == SAMPLE_SIZE
        and structural_failures == 0
        and len(digests) == 1
    )
    _write_report(report_path, report)
    return report


def _decode_audio(audio: bytes, *, duration_ms: int) -> np.ndarray:
    with sf.SoundFile(io.BytesIO(audio)) as audio_file:
        if (
            audio_file.format != "OGG"
            or audio_file.subtype != "OPUS"
            or audio_file.samplerate != 16_000
            or audio_file.channels != 1
        ):
            raise ValueError("audio is not OGG/Opus mono 16 kHz")
        decoded = audio_file.read(dtype="float32", always_2d=True)
    if (
        not np.isfinite(decoded).all()
        or decoded.shape[0] != duration_ms * 16
        or decoded.shape[0] < 16_000
    ):
        raise ValueError("audio is empty, non-finite, or duration-inconsistent")
    return decoded[:, 0]


def _empty_report(
    *, expected_head: str | None, seed: str | int, selected: list[MetadataRow]
) -> dict[str, object]:
    selected_digest = hashlib.sha256(
        json.dumps(
            [_stable_candidate_identity(row) for row in selected],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "report_type": "p1-v8-dozen-sanity-gate",
        "pilot_head": expected_head,
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "sample_set_digest": selected_digest,
        "sample_size": SAMPLE_SIZE,
        "seed": str(seed),
        "checks": {
            "private_immutable_revision": False,
            "hashes": False,
            "ogg_opus_16khz_mono": False,
            "exact_schema": False,
            "one_pipeline_digest": False,
            "timestamp_native_v8": False,
            "exact_duration_and_bounds": False,
            "trainable_text": False,
            "timed_anchor_single_speaker": False,
        },
        "counts": {"accepted": 0, "selected": 0},
        "pass": False,
    }


def _stable_candidate_identity(row: MetadataRow) -> str:
    for key in ("audit_id", "segment_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return hashlib.sha256(
        json.dumps(
            sorted((str(key), repr(value)) for key, value in row.items())
        ).encode()
    ).hexdigest()


def _is_accepted_candidate(row: MetadataRow) -> bool:
    status = row.get("status", row.get("quality_status"))
    if status is not None:
        return status == "accepted"
    return isinstance(row.get("parquet_path", row.get("remote_parquet_path")), str)


def _rebind_candidate(candidate: MetadataRow, *, pilot_head: str) -> dict[str, object]:
    revision = candidate.get("revision")
    if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("candidate revision is not immutable")
    rebound = dict(candidate)
    rebound["revision"] = pilot_head
    return rebound


def _resolve_pilot_head(
    candidates: c.Iterable[MetadataRow], pilot_head: str | None
) -> str | None:
    if pilot_head is not None and not _COMMIT_SHA.fullmatch(pilot_head):
        raise ValueError("pilot_head must be a complete immutable commit SHA")
    revisions = {
        row.get("revision")
        for row in candidates
        if isinstance(row.get("revision"), str)
    }
    if pilot_head is None and len(revisions) == 1:
        value = next(iter(revisions))
        if isinstance(value, str) and _COMMIT_SHA.fullmatch(value):
            return value
    if pilot_head is None:
        raise ValueError("an explicit final pilot HEAD is required")
    return pilot_head


def _resolve_pipeline_digest(
    selected: c.Sequence[MetadataRow], expected: str | None
) -> str | None:
    """Resolve the active digest from explicit evidence or audit candidates.

    Returns:
        The validated digest when evidence is unambiguous, otherwise ``None``.
    """
    if expected is not None:
        return expected if _SHA256.fullmatch(expected) else None
    digests = {
        value
        for row in selected
        if isinstance(value := row.get("pipeline_config_sha256"), str)
    }
    return next(iter(digests)) if len(digests) == 1 else None


def _retrieve_row(retriever: object, candidate: MetadataRow) -> dict[str, object]:
    retrieve_row = getattr(retriever, "retrieve_row", None)
    if callable(retrieve_row):
        row = retrieve_row(candidate)
        if not isinstance(row, c.Mapping):
            raise TypeError("retriever returned a non-mapping row")
        return dict(row)
    retrieve = getattr(retriever, "retrieve", None)
    if not callable(retrieve):
        raise TypeError("retriever must provide retrieve_row or retrieve")
    audio = retrieve(candidate)
    if not isinstance(audio, bytes):
        raise TypeError("sanity gate retrievers must return in-memory bytes")
    row = dict(candidate)
    row["audio"] = audio
    return row


def _select_candidates(
    candidates: c.Iterable[MetadataRow], *, seed: str | int
) -> list[MetadataRow]:
    groups: dict[tuple[str, tuple[str, ...]], list[MetadataRow]] = {}
    for candidate in candidates:
        key = (_programme_key(candidate), _stratum(candidate))
        groups.setdefault(key, []).append(candidate)
    ranked_groups: list[tuple[str, list[MetadataRow]]] = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: _rank(seed, row))
        ranked_groups.append((hashlib.sha256(repr(key).encode()).hexdigest(), ordered))
    ranked_groups.sort(key=lambda item: item[0])
    selected: list[MetadataRow] = []
    while len(selected) < SAMPLE_SIZE and ranked_groups:
        progressed = False
        for _, group in ranked_groups:
            if group:
                selected.append(group.pop(0))
                progressed = True
                if len(selected) == SAMPLE_SIZE:
                    break
        if not progressed:
            break
    return selected


def _programme_key(row: MetadataRow) -> str:
    for key in ("source_file_id", "programme_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown-programme"


def _rank(seed: str | int, row: MetadataRow) -> str:
    return hashlib.sha256(
        f"{seed}\0{_stable_candidate_identity(row)}".encode()
    ).hexdigest()


def _stratum(row: MetadataRow) -> tuple[str, ...]:
    value = row.get("stratum", ())
    if isinstance(value, (str, bytes)):
        return (str(value),)
    if isinstance(value, c.Iterable):
        return tuple(str(item) for item in value)
    return ()


def _validate_candidate_locator(
    candidate: MetadataRow, *, pilot_head: str | None, repository: str
) -> None:
    if candidate.get("repository") != repository:
        raise ValueError("candidate is not from the private P1 repository")
    revision = candidate.get("revision")
    if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("candidate revision is not immutable")
    if pilot_head is not None and revision != pilot_head:
        raise ValueError("candidate is not from the pilot head")
    path = candidate.get("parquet_path", candidate.get("remote_parquet_path"))
    if not isinstance(path, str) or not path.endswith(".parquet"):
        raise ValueError("candidate has no Parquet locator")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or "\\" in path or ".." in pure_path.parts:
        raise ValueError("candidate Parquet locator is unsafe")
    locator = candidate.get("row_locator")
    if not isinstance(locator, int) or isinstance(locator, bool) or locator < 0:
        raise ValueError("candidate row locator is invalid")
    parquet_hash = candidate.get("parquet_sha256")
    if not isinstance(parquet_hash, str) or not _SHA256.fullmatch(parquet_hash):
        raise ValueError("candidate Parquet hash is missing")


def _validate_row(
    *,
    row: dict[str, object],
    candidate: MetadataRow,
    expected_pipeline_version: str,
    expected_pipeline_config_sha256: str,
) -> bytes:
    audio = _validate_exact_schema(row=row)
    if row.get("pipeline_version") != expected_pipeline_version:
        raise ValueError("retrieved row has the wrong pipeline version")
    if row.get("language") != "da":
        raise ValueError("retrieved row has the wrong language")
    expected_segment = candidate.get("segment_id")
    if expected_segment is not None and row.get("segment_id") != expected_segment:
        raise ValueError("candidate segment does not match retrieved row")
    if row.get("alignment_backend") != "timestamp-native":
        raise ValueError("retrieved row has the wrong alignment backend")
    if row.get("alignment_score_type") != "not_applicable:source_timestamps":
        raise ValueError("retrieved row has the wrong alignment score type")
    if row.get("alignment_method") != "timestamp-native:p1-transcripts.words":
        raise ValueError("retrieved row has the wrong alignment method")
    speakers = row.get("speaker_ids")
    if isinstance(speakers, (str, bytes)) or not isinstance(speakers, c.Iterable):
        raise ValueError("retrieved row has no single timed-anchor speaker")
    speaker_values = tuple(speakers)
    if (
        len(speaker_values) != 1
        or not isinstance(speaker_values[0], str)
        or not speaker_values[0]
    ):
        raise ValueError("retrieved row has no single timed-anchor speaker")
    text = row.get("text")
    alignment_text = row.get("alignment_text")
    if not _has_trainable_text(text) or not _has_trainable_text(alignment_text):
        raise ValueError("retrieved row has no trainable text")
    duration = row.get("duration_ms")
    start = row.get("source_start_ms")
    end = row.get("source_end_ms")
    source_duration = row.get("source_duration_ms")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (duration, start, end, source_duration)
    ):
        raise ValueError("retrieved row has invalid duration metadata")
    if (
        not 1_000 <= duration < 10_000
        or start < 0
        or end <= start
        or end - start != duration
        or source_duration <= 0
        or end > source_duration
    ):
        raise ValueError("retrieved row has inconsistent duration metadata")
    if (
        start != row.get("proposal_start_ms")
        or end != row.get("proposal_end_ms")
        or row.get("alignment_score") is not None
        or row.get("start_drift_ms") is not None
        or row.get("end_drift_ms") is not None
        or row.get("vad_speech_ratio") is not None
    ):
        raise ValueError("retrieved row has non-native v8 alignment evidence")
    expected_audio = candidate.get("audio_sha256")
    expected_metadata = candidate.get("metadata_sha256")
    for field in ("segment_id", "pipeline_config_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError("retrieved row has invalid identity metadata")
    if row.get("pipeline_config_sha256") != expected_pipeline_config_sha256:
        raise ValueError("retrieved row has a different pipeline digest")
    actual_audio = hashlib.sha256(audio).hexdigest()
    if (
        not isinstance(expected_audio, str)
        or not _SHA256.fullmatch(expected_audio)
        or actual_audio != expected_audio
        or row.get("audio_sha256") != actual_audio
    ):
        raise ValueError("candidate audio hash does not match retrieved audio")
    if (
        not isinstance(expected_metadata, str)
        or not _SHA256.fullmatch(expected_metadata)
        or _metadata_digest(row) != expected_metadata
    ):
        raise ValueError("candidate metadata hash does not match retrieved row")
    return audio


def _has_trainable_text(value: object) -> bool:
    return isinstance(value, str) and any(char.isalnum() for char in value)


def _validate_exact_schema(*, row: dict[str, object]) -> bytes:
    """Validate one retrieved row against the strict P1 output contract.

    The remote Arrow representation uses lists and an HF Audio mapping, whereas the
    in-process contract uses tuples and encoded bytes.  Convert only those transport
    wrappers before asking the authoritative Pydantic contract to validate types and
    structure.  No audio decoding or content is performed here.

    Returns:
        Embedded audio bytes for the subsequent encoding checks.

    Raises:
        ValueError:
            If a field has the wrong type, nullability, or structure.
    """
    expected_fields = {field.name for field in OUTPUT_SCHEMA.fields}
    if set(row) != expected_fields:
        raise ValueError("retrieved row does not have the exact v8 schema")

    audio_value = row.get("audio")
    if not isinstance(audio_value, c.Mapping):
        raise ValueError("retrieved row has an invalid audio structure")
    if set(audio_value) - {"bytes", "path"}:
        raise ValueError("retrieved row has an invalid audio structure")
    payload = audio_value.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("retrieved row has an invalid audio structure")
    path = audio_value.get("path")
    if path is not None and not isinstance(path, str):
        raise ValueError("retrieved row has an invalid audio structure")

    validated = dict(row)
    validated["audio"] = payload
    for name in ("alignment_word_map", "speaker_ids"):
        value = row.get(name)
        if not isinstance(value, list):
            raise ValueError(f"retrieved row has an invalid {name} structure")
        validated[name] = tuple(value)
    try:
        OutputRow.model_validate(validated)
    except Exception as error:
        raise ValueError(
            "retrieved row has invalid field types or nullability"
        ) from error
    return payload


def _write_report(path: Path | str | None, report: dict[str, object]) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "PILOT_REPOSITORY",
    "PIPELINE_VERSION",
    "SAMPLE_SIZE",
    "SCHEMA_VERSION",
    "run_v8_sanity_gate",
]
