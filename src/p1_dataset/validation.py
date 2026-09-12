"""Bounded validation and audit utilities for the P1 derived corpus.

The validation path deliberately treats remote rows as metadata records.  Audio is
accepted only by the one-item audit helper and is deleted in a ``finally`` block;
there is no API here which materialises a corpus locally.
"""

from __future__ import annotations

import collections.abc as c
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import typing as t
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean, median

from .contracts import OUTPUT_SCHEMA
from .source import _AUDIO_POINTER_METADATA_KEY, _AUDIO_POINTER_METADATA_MAX_BYTES

MetadataRow = c.Mapping[str, object]
RowStream = c.Iterable[MetadataRow]

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuditReservoir:
    """Crash-safe, corpus-wide reservoir for the bounded audit manifest.

    The state file contains only the currently selected metadata records.  Each
    update is written to a temporary sibling and atomically renamed, so a crash
    cannot leave a half-written reservoir or an unbounded per-programme log.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        accepted_quota: int = 200,
        rejected_quota: int = 100,
        borderline_quota: int = 100,
        seed: int | str = "p1",
    ) -> None:
        """Open a bounded reservoir, recovering its previous selection.

        Raises:
            ValueError:
                If quotas or a recovered state file are invalid.
        """
        self.path = Path(path)
        self.quotas = {
            "accepted": accepted_quota,
            "rejected": rejected_quota,
            "borderline": borderline_quota,
        }
        if any(value < 0 for value in self.quotas.values()):
            raise ValueError("audit quotas must not be negative")
        self.seed = seed
        self.rows: list[dict[str, object]] = []
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, list) or not all(
                isinstance(item, dict) for item in value
            ):
                raise ValueError("audit reservoir is not a JSON list of records")
            self.rows = [t.cast(dict[str, object], item) for item in value]

    def add(self, candidates: RowStream) -> None:
        """Merge candidates and persist the bounded, stratified selection.

        Raises:
            ValueError:
                If a candidate has no valid temporary or immutable locator.
        """
        combined_by_identity: dict[tuple[str, str], MetadataRow] = {}
        for row in [*self.rows, *candidates]:
            key = (_status(row), _identity(row))
            existing = combined_by_identity.get(key)
            if existing is not None and _parquet_path(existing) is not None:
                incoming_path = _parquet_path(row)
                incoming_revision = _string(row, "revision", "hub_revision")
                existing_revision = _string(existing, "revision", "hub_revision")
                if incoming_path is None:
                    continue
                if incoming_path != _parquet_path(existing) or (
                    incoming_revision != existing_revision
                ):
                    raise ValueError("audit candidate remote locator is immutable")
            combined_by_identity[key] = row
        combined = list(combined_by_identity.values())
        reservoirs = {
            status: _StratifiedReservoir(quota) for status, quota in self.quotas.items()
        }
        for ordinal, row in enumerate(combined):
            status = _status(row)
            if status not in reservoirs or not self.quotas[status]:
                continue
            _validate_candidate_locator(row, status)
            safe_row = _metadata_copy(row)
            metadata_digest = _metadata_digest_for_candidate(row)
            safe_row["_p1_metadata_sha256"] = metadata_digest
            audio_digest = _audio_digest(row)
            if audio_digest is not None:
                safe_row["_p1_audio_sha256"] = audio_digest
            reservoirs[status].add(
                safe_row, seed=f"{self.seed}:{status}", ordinal=ordinal
            )
        self.rows = [
            row
            for status, reservoir in reservoirs.items()
            for row in reservoir.rows()
            if _status(row) == status
        ]
        self._write_state()

    def update_remote_locators(
        self,
        *,
        repository: str,
        revision: str,
        local_paths: c.Sequence[Path | str],
        remote_paths: c.Sequence[str],
        row_counts: c.Sequence[int],
        parquet_sha256: c.Sequence[str] | None = None,
    ) -> int:
        """Resolve selected local candidates to immutable remote locations.

        Local identity is retained in the crash-safe reservoir until the commit SHA
        and publication-relative path are known.  This makes a restart after a
        verification failure able to finish the same audit selection without
        sampling the accepted rows again.

        Returns:
            Number of selected candidates resolved to remote locators.

        Raises:
            ValueError:
                If shard locator sequences, hashes, or the immutable revision are
                invalid.
        """
        if not _COMMIT_SHA.fullmatch(revision):
            raise ValueError("audit candidates require a complete immutable revision")
        if not (len(local_paths) == len(remote_paths) == len(row_counts)):
            raise ValueError("local and remote shard evidence must have equal lengths")
        if parquet_sha256 is not None and len(parquet_sha256) != len(remote_paths):
            raise ValueError("Parquet hashes must match remote shard evidence")
        by_identity: dict[tuple[str, int], tuple[str, str, str | None]] = {}
        for index, (local, remote, count) in enumerate(
            zip(local_paths, remote_paths, row_counts, strict=True)
        ):
            shard_hash = parquet_sha256[index] if parquet_sha256 is not None else None
            if shard_hash is not None and not _SHA256.fullmatch(shard_hash):
                raise ValueError("Parquet hashes must be lowercase SHA-256 hex")
            if count < 0:
                raise ValueError("audit shard row counts must not be negative")
            local_key = str(Path(local).expanduser().resolve(strict=False))
            for row_locator in range(count):
                by_identity[(local_key, row_locator)] = (remote, repository, shard_hash)

        resolved = 0
        for row in self.rows:
            if _status(row) != "accepted":
                continue
            local = _string(row, "local_path")
            row_locator = _integer(row, "local_row_locator")
            if local is None or row_locator is None:
                continue
            target = by_identity.get(
                (str(Path(local).expanduser().resolve(strict=False)), row_locator)
            )
            if target is None:
                continue
            remote, target_repository, shard_hash = target
            existing_path = _parquet_path(row)
            existing_revision = _string(row, "revision", "hub_revision")
            if existing_path is not None:
                if existing_path != remote or existing_revision != revision:
                    raise ValueError("audit candidate remote locator is immutable")
                continue
            row.update(
                {
                    "repository": target_repository,
                    "revision": revision,
                    "parquet_path": remote,
                    "remote_parquet_path": remote,
                    "row_locator": row_locator,
                }
            )
            if shard_hash is not None:
                row["parquet_sha256"] = shard_hash
            row.pop("local_path", None)
            row.pop("local_row_locator", None)
            resolved += 1
        if resolved:
            self._write_state()
        return resolved

    resolve_remote_locators = update_remote_locators

    def _write_state(self) -> None:
        """Atomically persist the bounded selection."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.rows, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def finalise(self, path: Path | str) -> list[dict[str, object]]:
        """Write the sole final blinded manifest after all commits are known.

        Returns:
            The metadata-only manifest records.
        """
        manifest = create_blinded_audit_manifest(
            self.rows,
            accepted_quota=self.quotas["accepted"],
            rejected_quota=self.quotas["rejected"],
            borderline_quota=self.quotas["borderline"],
            seed=self.seed,
        )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for candidate in manifest:
                stream.write(json.dumps(candidate, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        return manifest


class _StratifiedReservoir:
    """Bounded, multi-axis reservoir with one fair bucket per stratum.

    A bucket is a complete combination of the required axes, rather than one axis
    selected after the fact.  Each bucket receives a fair share of the bounded
    capacity; a seeded ``random.Random`` priority makes selection reproducible
    without depending on Python's process-randomised ``hash`` function.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("reservoir capacity must not be negative")
        self.capacity = capacity
        self.groups: dict[tuple[str, ...], list[tuple[float, dict[str, object]]]] = {}

    def add(self, row: dict[str, object], *, seed: str, ordinal: int) -> None:
        """Add a row while retaining at most ``capacity`` metadata records."""
        del ordinal
        if not self.capacity:
            return
        key = _stratum_key(row)
        if key not in self.groups:
            if len(self.groups) >= self.capacity:
                new_priority = _random_priority(seed, "stratum", key)
                worst = max(
                    self.groups,
                    key=lambda item: _random_priority(seed, "stratum", item),
                )
                if new_priority >= _random_priority(seed, "stratum", worst):
                    return
                del self.groups[worst]
            self.groups[key] = []
        bucket = self.groups[key]
        bucket.append((_random_priority(seed, "row", key, _identity(row)), row))
        bucket.sort(key=lambda item: (item[0], _identity(item[1])))
        limit = max(1, self.capacity // len(self.groups))
        for values in self.groups.values():
            del values[limit:]

    def rows(self) -> list[dict[str, object]]:
        """Return round-robin records without exceeding the reservoir capacity."""
        result: list[dict[str, object]] = []
        ordered = sorted(self.groups.items())
        depth = 0
        while len(result) < self.capacity:
            added = False
            for _, bucket in ordered:
                if depth < len(bucket):
                    result.append(bucket[depth][1])
                    added = True
                    if len(result) == self.capacity:
                        break
            if not added:
                break
            depth += 1
        return result


def _identity(row: MetadataRow) -> str:
    return (
        _string(row, "segment_id", "id")
        or hashlib.sha256(
            json.dumps(_metadata_copy(row), sort_keys=True, default=str).encode()
        ).hexdigest()
    )


def _metadata_copy(row: MetadataRow) -> dict[str, object]:
    """Copy only bounded scalar metadata needed to choose an audit candidate.

    Returns:
        Bounded metadata with no audio or transcript payload.
    """
    allowed = {
        "segment_id",
        "pipeline_version",
        "pipeline_config_sha256",
        "id",
        "source_file_id",
        "source_repository",
        "source_revision",
        "programme_id",
        "source_start_ms",
        "source_end_ms",
        "duration_ms",
        "programme_duration_ms",
        "source_duration_ms",
        "status",
        "quality_status",
        "accepted",
        "borderline",
        "show",
        "show_type",
        "programme_type",
        "speaker_count",
        "speaker_ids",
        "speakers",
        "music",
        "music_dominant",
        "language_probability",
        "language_prob",
        "language_probability_decile",
        "language_decile",
        "programme_position",
        "programme_position_decile",
        "position_decile",
        "position",
        "alignment_score",
        "alignment_score_type",
        "alignment_method",
        "alignment_backend",
        "confidence",
        "confidence_decile",
        "drift_ms",
        "start_drift_ms",
        "end_drift_ms",
        "drift_decile",
        "repository",
        "repo_id",
        "hub_repo",
        "revision",
        "hub_revision",
        "parquet_path",
        "remote_parquet_path",
        "shard_path",
        "source_parquet",
        "source_shard_path",
        "source_shard",
        "source_shard_index",
        "source_parquet_path",
        "source_row_group",
        "source_row_index",
        "source_row_locator",
        "source_shard_byte_size",
        "parquet_sha256",
        "shard_sha256",
        "audio_sha256",
        "metadata_sha256",
        "row_locator",
        "row_index",
        "parquet_row",
        "local_path",
        "local_row_locator",
        "rejection_reason",
        "reject_reason",
    }
    return {key: value for key, value in row.items() if key in allowed}


def _string(row: MetadataRow, *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str):
            return value
        if value is not None and not isinstance(
            value, (bytes, bytearray, dict, list, tuple)
        ):
            return str(value)
    return None


def _random_priority(seed: str, *parts: object) -> float:
    """Return a stable pseudo-random priority without using process hash state."""
    generator = random.Random("\0".join([seed, *(str(part) for part in parts)]))
    return generator.random()


def _stratum_key(row: MetadataRow) -> tuple[str, ...]:
    """Build the fixed-dimensional key used by the bounded sampler.

    Returns:
        Duration, show, speaker, music, probability, position, confidence, and
        drift dimensions.
    """
    duration = _number(row, "duration_ms") or 0
    duration_class = (
        "short" if duration < 2_000 else "long" if duration > 8_000 else "target"
    )
    speakers = row.get("speaker_ids", row.get("speakers", ()))
    if isinstance(speakers, (str, bytes)):
        speaker_count = 1
    elif isinstance(speakers, c.Sized):
        speaker_count = len(speakers)
    else:
        speaker_count = int(_number(row, "speaker_count") or 0)
    language_probability = _number(row, "language_probability", "language_prob") or 0
    confidence = _number(row, "alignment_score", "confidence") or 0
    drift = abs(
        _number(row, "drift_ms")
        or max(
            abs(_number(row, "start_drift_ms") or 0),
            abs(_number(row, "end_drift_ms") or 0),
        )
    )
    return (
        f"duration-{duration_class}",
        f"show-{_string(row, 'show', 'show_type', 'programme_type') or 'unknown'}",
        f"speakers-{speaker_count}",
        "music" if _boolean(row, "music", "music_dominant") else "speech",
        "language-"
        + _explicit_or_decile(
            row,
            ("language_probability_decile", "language_decile"),
            language_probability,
        ),
        "position-"
        + _explicit_or_decile(
            row,
            ("programme_position_decile", "position_decile"),
            _number(row, "programme_position", "position") or 0,
        ),
        f"confidence-{_explicit_or_decile(row, ('confidence_decile',), confidence)}",
        f"drift-{_explicit_or_decile(row, ('drift_decile',), drift / 1_000)}",
    )


def _boolean(row: MetadataRow, *keys: str) -> bool:
    """Read common JSON boolean spellings without treating ``"false"`` as true.

    Returns:
        The first recognised boolean value, or ``False``.
    """
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.casefold() in {"1", "true", "yes", "music"}
    return False


def _explicit_or_decile(row: MetadataRow, keys: tuple[str, ...], value: float) -> str:
    """Use a supplied decile or derive one from a normalised numeric value.

    Returns:
        A canonical ``d0`` to ``d9`` label.
    """
    explicit = _string(row, *keys)
    if explicit:
        label = explicit.casefold()
        if label.startswith("d"):
            label = label[1:]
        try:
            decile = int(label)
        except ValueError:
            decile = int(value * 10)
        return f"d{min(9, max(0, decile))}"
    return f"d{min(9, max(0, int(value * 10)))}"


def _number(row: MetadataRow, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _audio_digest(row: MetadataRow) -> str | None:
    value = row.get("audio", row.get("waveform"))
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, c.Mapping) and isinstance(value.get("bytes"), bytes):
        return hashlib.sha256(t.cast(bytes, value["bytes"])).hexdigest()
    return None


def _integer(row: MetadataRow, *keys: str) -> int | None:
    value = _number(row, *keys)
    return int(value) if value is not None else None


def _metadata_digest_for_candidate(row: MetadataRow) -> str:
    """Return the producer digest, or calculate one for legacy rows.

    The producer digest is calculated from the complete output row before the audit
    record is reduced to metadata.  Recalculating after that reduction would hash a
    different object and make valid audit records unretrievable.

    Raises:
        ValueError:
            If a supplied digest is not lowercase SHA-256 hexadecimal text.
    """
    supplied = [
        (key, row[key])
        for key in ("_p1_metadata_sha256", "metadata_sha256")
        if key in row
    ]
    for key, value in supplied:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{key} must be lowercase SHA-256 hex")
    if supplied:
        return t.cast(str, supplied[0][1])
    return _metadata_digest(row)


def _metadata_digest(row: MetadataRow) -> str:
    """Hash the canonical metadata of the complete published row.

    The digest is calculated from the output schema before audit metadata is reduced
    to its bounded reservoir representation. Audio bytes and the digest itself are
    excluded because they are payload and transport values, respectively. Keeping
    every other published field (including source and segment identities) means that
    a row changing between local generation and remote retrieval cannot pass audit.

    Returns:
        A SHA-256 digest of the canonical published metadata.
    """
    published_fields = {
        field.name for field in OUTPUT_SCHEMA.fields if field.name != "audio"
    }
    metadata = {
        key: _canonical_published_value(key, row.get(key)) for key in published_fields
    }
    payload = _canonical_json_value(metadata)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: object) -> object:
    """Convert a remote row value to the canonical JSON projection.

    Returns:
        A JSON-compatible scalar, list, or mapping.
    """
    if isinstance(value, c.Mapping):
        return {
            str(key): _canonical_json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_published_value(key: str, value: object) -> object:
    """Match scalar coercions performed by the published Arrow schema.

    Returns:
        The value in the representation used by the published row.
    """
    if key in {"alignment_score", "vad_speech_ratio"} and isinstance(value, float):
        try:
            return struct.unpack("!f", struct.pack("!f", value))[0]
        except OverflowError:
            return value
    return value


def _parquet_path(row: MetadataRow) -> str | None:
    """Read a Parquet object path, never treating an audio path as one.

    Returns:
        A Parquet path, or ``None`` when the row has no such locator.
    """
    value = _string(
        row, "parquet_path", "remote_parquet_path", "shard_path", "source_parquet"
    )
    return value if value and value.lower().endswith(".parquet") else None


def _status(row: MetadataRow) -> str:
    value = _string(row, "status", "quality_status", "decision")
    if value in {"accepted", "pass", "keep"}:
        return "accepted"
    if value in {"rejected", "fail", "reject"}:
        return "rejected"
    if value == "borderline":
        return value
    accepted = row.get("accepted")
    if accepted is False:
        return "rejected"
    if row.get("borderline") is True:
        return "borderline"
    return "accepted"


def _validate_candidate_locator(row: MetadataRow, status: str) -> None:
    """Validate the locator appropriate to an audit status.

    Raises:
        ValueError:
            If the status-specific immutable locator is incomplete.
    """
    if status == "accepted" and _string(row, "local_path") is not None:
        local_locator = _integer(row, "local_row_locator")
        if local_locator is None or local_locator < 0:
            raise ValueError("accepted candidates require a local row locator")
        return
    if _parquet_path(row) is not None and (
        status == "accepted" or not _has_source_locator(row)
    ):
        _validate_audit_locator(row)
        return
    for key in ("source_repository", "source_file_id"):
        if not _string(row, key):
            raise ValueError(f"{status} candidates require {key}")
    revision = _string(row, "source_revision")
    if revision is None or not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"{status} candidates require an immutable source revision")
    start = _integer(row, "source_start_ms")
    end = _integer(row, "source_end_ms")
    if start is None or end is None or start < 0 or end <= start:
        raise ValueError(f"{status} candidates require a valid source interval")
    shard_path = _source_shard_path(row)
    row_group = _integer(row, "source_row_group")
    if row_group is None:
        row_group = 0
    row_index = _integer(row, "source_row_index", "source_row_locator", "row_locator")
    if shard_path is None or row_group is None or row_index is None:
        raise ValueError(f"{status} candidates require an exact source row locator")
    if not shard_path.lower().endswith(".parquet") or row_group < 0 or row_index < 0:
        raise ValueError(f"{status} candidates have an invalid source row locator")


def _has_source_locator(row: MetadataRow) -> bool:
    """Return whether a row contains the complete source interval locator."""
    return all(
        (
            _string(row, "source_repository"),
            _string(row, "source_revision"),
            _string(row, "source_file_id"),
            _integer(row, "source_start_ms") is not None,
            _integer(row, "source_end_ms") is not None,
        )
    )


def _source_shard_path(row: MetadataRow) -> str | None:
    """Return the exact source Parquet path, including legacy aliases."""
    value = _string(
        row,
        "source_shard_path",
        "source_shard",
        "source_parquet_path",
        "source_parquet",
        "parquet_path",
    )
    return value if value and value.lower().endswith(".parquet") else None


def _validate_audit_locator(row: MetadataRow) -> tuple[str, int, str]:
    """Validate the immutable remote locator required by an audit record.

    Returns:
        The Parquet path, row locator, and immutable commit SHA.

    Raises:
        ValueError:
            If any required locator field is absent or malformed.
    """
    parquet_path = _parquet_path(row)
    if parquet_path is None:
        raise ValueError("audit candidates require a remote Parquet path")
    pure_path = PurePosixPath(parquet_path)
    if (
        pure_path.is_absolute()
        or "\\" in parquet_path
        or "://" in parquet_path
        or ".." in pure_path.parts
    ):
        raise ValueError("audit candidate Parquet locator is unsafe")
    locator = _explicit_row_locator(row)
    if locator is None or locator < 0:
        raise ValueError("audit candidates require a non-negative row locator")
    revision = _string(row, "revision", "hub_revision")
    if revision is None or not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("audit candidates require a complete immutable commit SHA")
    return parquet_path, locator, revision


def _explicit_row_locator(row: MetadataRow) -> int | None:
    """Return the supplied Parquet row locator, without inventing one."""
    for key in (
        "row_locator",
        "row_index",
        "parquet_row",
        "source_row_index",
        "source_row_locator",
    ):
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def create_blinded_audit_manifest(
    rows: RowStream,
    *,
    accepted_quota: int = 200,
    rejected_quota: int = 100,
    borderline_quota: int = 0,
    seed: int | str = 0,
) -> list[dict[str, object]]:
    """Create bounded, label-free candidate records for a blinded audit.

    Sampling is performed by a bounded reservoir for each status.  At most the
    requested quota is retained for a status, irrespective of corpus size; source
    labels, transcripts, and audio are never copied into the result.  The default
    quotas are the production pilot quotas (200 accepted and 100 rejected).

    Returns:
        Metadata-only candidate records. Accepted rows have a Parquet locator;
        rejected and borderline rows have a source interval locator.

    Raises:
        ValueError:
            If a quota is negative.
    """
    quotas = {
        "accepted": accepted_quota,
        "rejected": rejected_quota,
        "borderline": borderline_quota,
    }
    if any(value < 0 for value in quotas.values()):
        raise ValueError("audit quotas must not be negative")
    reservoirs = {
        status: _StratifiedReservoir(quota) for status, quota in quotas.items()
    }
    for ordinal, row in enumerate(rows):
        status = _status(row)
        if quotas[status]:
            _validate_candidate_locator(row, status)
            metadata_digest = _metadata_digest_for_candidate(row)
            safe_row = _metadata_copy(row)
            safe_row["_p1_metadata_sha256"] = metadata_digest
            audio_digest = _audio_digest(row)
            if audio_digest is not None:
                safe_row["_p1_audio_sha256"] = audio_digest
            reservoirs[status].add(safe_row, seed=f"{seed}:{status}", ordinal=ordinal)
    selected: list[dict[str, object]] = []
    for status, quota in quotas.items():
        selected.extend(
            _candidate_source_rows(
                reservoirs[status].rows(), status=status, seed=f"{seed}:{status}"
            )
        )
    selected.sort(key=lambda item: (_identity(item), _stratum_key(item)))
    manifest: list[dict[str, object]] = []
    for ordinal, row in enumerate(selected):
        identity = _identity(row)
        status = _status(row)
        audit_id = hashlib.sha256(f"{seed}\0{ordinal}\0{identity}".encode()).hexdigest()
        candidate = AuditCandidate(
            audit_id=audit_id,
            segment_id=identity,
            repository=(
                _string(row, "repository", "repo_id", "hub_repo")
                if status == "accepted" or not _has_source_locator(row)
                else None
            ),
            revision=(
                _string(row, "revision", "hub_revision")
                if status == "accepted" or not _has_source_locator(row)
                else None
            ),
            parquet_path=(
                _parquet_path(row)
                if status == "accepted" or not _has_source_locator(row)
                else None
            ),
            row_locator=(
                _row_locator(row, ordinal)
                if status == "accepted" or not _has_source_locator(row)
                else 0
            ),
            stratum=_stratum_key(row),
            metadata_sha256=_metadata_digest_for_candidate(row),
            pipeline_version=_string(row, "pipeline_version"),
            pipeline_config_sha256=_string(row, "pipeline_config_sha256"),
            audio_sha256=_string(row, "_p1_audio_sha256", "audio_sha256"),
            parquet_sha256=_string(
                row, "parquet_sha256", "shard_sha256", "_p1_parquet_sha256"
            ),
            source_file_id=_string(row, "source_file_id", "programme_id"),
            source_repository=_string(row, "source_repository"),
            source_revision=_string(row, "source_revision"),
            source_start_ms=_integer(row, "source_start_ms"),
            source_end_ms=_integer(row, "source_end_ms"),
            source_shard_path=(
                _source_shard_path(row) if status != "accepted" else None
            ),
            source_row_group=(
                (_integer(row, "source_row_group") or 0)
                if status != "accepted"
                else None
            ),
            source_row_index=(
                _integer(row, "source_row_index", "source_row_locator", "row_locator")
                if status != "accepted"
                else None
            ),
            source_shard_byte_size=(
                _integer(row, "source_shard_byte_size")
                if status != "accepted"
                else None
            ),
        )
        manifest.append(candidate.as_dict())
    return manifest


@dataclass(frozen=True)
class AuditCandidate:
    """Durable, metadata-only location for one blinded audit item.

    Accepted candidates contain a Parquet object and row locator. Rejected and
    borderline candidates contain an immutable source file and interval locator.
    It is safe to persist in a manifest or SQLite database.
    """

    audit_id: str
    segment_id: str
    repository: str | None = None
    revision: str | None = None
    parquet_path: str | None = None
    row_locator: int = 0
    stratum: tuple[str, ...] = ()
    metadata_sha256: str = ""
    pipeline_version: str | None = None
    pipeline_config_sha256: str | None = None
    audio_sha256: str | None = None
    parquet_sha256: str | None = None
    source_file_id: str | None = None
    source_repository: str | None = None
    source_revision: str | None = None
    source_start_ms: int | None = None
    source_end_ms: int | None = None
    source_shard_path: str | None = None
    source_row_group: int | None = None
    source_row_index: int | None = None
    source_shard_byte_size: int | None = None

    def __post_init__(self) -> None:
        """Reject durable records that cannot be retrieved later.

        Raises:
            ValueError:
                If the remote locator is incomplete or malformed.
        """
        if self.parquet_path is not None:
            if not self.parquet_path.lower().endswith(".parquet"):
                raise ValueError("audit candidates require a remote Parquet path")
            if self.row_locator < 0:
                raise ValueError("audit candidates require a non-negative row locator")
            if self.revision is None or not _COMMIT_SHA.fullmatch(self.revision):
                raise ValueError(
                    "audit candidates require a complete immutable commit SHA"
                )
        elif not self.source_file_id or not self.source_repository:
            raise ValueError("source audit candidates require a source locator")
        if self.parquet_path is None:
            if self.source_revision is None or not _COMMIT_SHA.fullmatch(
                self.source_revision
            ):
                raise ValueError(
                    "source audit candidates require an immutable revision"
                )
            if (
                self.source_start_ms is None
                or self.source_end_ms is None
                or self.source_start_ms < 0
                or self.source_end_ms <= self.source_start_ms
            ):
                raise ValueError("source audit candidates require a valid interval")
            if (
                self.source_shard_path is None
                or not self.source_shard_path.lower().endswith(".parquet")
                or self.source_row_group is None
                or self.source_row_group < 0
                or self.source_row_index is None
                or self.source_row_index < 0
            ):
                raise ValueError(
                    "source audit candidates require an exact source row locator"
                )
            if (
                self.source_shard_byte_size is not None
                and self.source_shard_byte_size <= 0
            ):
                raise ValueError("source shard byte size must be positive")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible metadata record without the source label."""
        result: dict[str, object] = {
            "audit_id": self.audit_id,
            "segment_id": self.segment_id,
            "stratum": list(self.stratum),
            "metadata_sha256": self.metadata_sha256,
        }
        if self.pipeline_version is not None:
            result["pipeline_version"] = self.pipeline_version
        if self.pipeline_config_sha256 is not None:
            result["pipeline_config_sha256"] = self.pipeline_config_sha256
        if self.parquet_path is not None:
            result.update(
                {
                    "repository": self.repository,
                    "revision": self.revision,
                    "parquet_path": self.parquet_path,
                    "remote_parquet_path": self.parquet_path,
                    "row_locator": self.row_locator,
                }
            )
        if self.audio_sha256 is not None:
            result["audio_sha256"] = self.audio_sha256
        if self.parquet_sha256 is not None:
            result["parquet_sha256"] = self.parquet_sha256
        if self.source_file_id is not None:
            result["source_file_id"] = self.source_file_id
        if self.source_repository is not None:
            result["source_repository"] = self.source_repository
        if self.source_revision is not None:
            result["source_revision"] = self.source_revision
        if self.source_start_ms is not None:
            result["source_start_ms"] = self.source_start_ms
        if self.source_end_ms is not None:
            result["source_end_ms"] = self.source_end_ms
        if self.source_shard_path is not None:
            result["source_shard_path"] = self.source_shard_path
        if self.source_row_group is not None:
            result["source_row_group"] = self.source_row_group
        if self.source_row_index is not None:
            result["source_row_index"] = self.source_row_index
        if self.source_shard_byte_size is not None:
            result["source_shard_byte_size"] = self.source_shard_byte_size
        return result


def _candidate_source_rows(
    rows: c.Iterable[dict[str, object]], *, status: str, seed: str
) -> list[dict[str, object]]:
    """Keep status selection internal while returning only selected source rows.

    Returns:
        The selected metadata rows.
    """
    del status, seed
    return list(rows)


def _row_locator(row: MetadataRow, fallback: int) -> int:
    explicit = _explicit_row_locator(row)
    return fallback if explicit is None else explicit


class ClipRetriever(t.Protocol):
    """Retrieve one audit clip into a temporary location or memory."""

    def retrieve(self, entry: MetadataRow) -> bytes | Path:
        """Retrieve the clip represented by an audit manifest entry."""


class IndependentASR(t.Protocol):
    """Minimal interface required by the independent-ASR anomaly detector."""

    def transcribe(self, audio: bytes) -> str:
        """Transcribe one clip without modifying or retaining it."""


class PinnedHubClipRetriever:
    """Read exactly one embedded-audio row from an immutable Hub revision.

    ``hub`` exposes the small ``repo_info``, ``get_paths_info``, and ``load_dataset``
    methods provided by the publication adapter. Dataset iteration may scan preceding
    rows, but only the addressed row's audio is returned and no dataset or row is
    retained.
    """

    def __init__(
        self,
        hub: object,
        *,
        repository: str | None = None,
        repo_id: str | None = None,
        revision: str,
        expected_pipeline_version: str | None = None,
        expected_pipeline_config_sha256: str | None = None,
    ) -> None:
        """Initialise a retriever pinned to a complete Hub commit SHA.

        Raises:
            ValueError:
                If revision is not a complete commit SHA.
        """
        if not _COMMIT_SHA.fullmatch(revision):
            raise ValueError("revision must be a complete 40-character commit SHA")
        if (expected_pipeline_version is None) != (
            expected_pipeline_config_sha256 is None
        ):
            raise ValueError(
                "expected pipeline version and digest must be supplied together"
            )
        if expected_pipeline_config_sha256 is not None and not _SHA256.fullmatch(
            expected_pipeline_config_sha256
        ):
            raise ValueError("expected pipeline configuration digest must be SHA-256")
        selected_repository = repository or repo_id
        if not selected_repository:
            raise ValueError("repository must be supplied")
        self.hub = hub
        self.repository = selected_repository
        self.revision = revision
        self.expected_pipeline_version = expected_pipeline_version
        self.expected_pipeline_config_sha256 = expected_pipeline_config_sha256
        self._repository_verified = False

    def retrieve(self, entry: MetadataRow) -> bytes:
        """Retrieve and verify one clip represented by a candidate record.

        Returns:
            The addressed embedded audio bytes.
        """
        return _embedded_audio(self.retrieve_row(entry))

    def retrieve_row(self, entry: MetadataRow) -> MetadataRow:
        """Retrieve and verify one complete row represented by a candidate record.

        Returns:
            The addressed row mapping. Only this one row remains reachable by the
            caller; the streaming dataset itself is not retained.

        Raises:
            ValueError:
                If the candidate is malformed or metadata/audio hashes differ.
            TypeError:
                If the Hub adapter or returned rows are not stream-compatible.
        """
        if (
            callable(getattr(self.hub, "repo_info", None))
            and not self._repository_verified
        ):
            self.verify_repository()
        repository = _string(entry, "repository", "repo_id")
        revision = _string(entry, "revision")
        if repository and repository != self.repository:
            raise ValueError("candidate repository differs from pinned repository")
        if revision and revision != self.revision:
            raise ValueError("candidate revision differs from pinned revision")
        parquet_path, locator, candidate_revision = _validate_audit_locator(entry)
        if candidate_revision != self.revision:
            raise ValueError("candidate revision differs from pinned revision")
        expected_shard = _string(entry, "parquet_sha256", "shard_sha256")
        if expected_shard:
            _validate_remote_shard_hash(
                self.hub,
                repository=self.repository,
                revision=self.revision,
                parquet_path=parquet_path,
                expected=expected_shard,
            )
        loader = getattr(self.hub, "load_dataset", None)
        if loader is None:
            raise TypeError("Hub adapter must provide load_dataset")
        try:
            dataset = loader(
                self.repository,
                shard_path=parquet_path,
                revision=self.revision,
                streaming=True,
            )
        except Exception:
            raise ValueError("unable to retrieve the pinned dataset row") from None
        row = _row_at(dataset, locator)
        expected_segment = _string(entry, "segment_id")
        actual_segment = _string(row, "segment_id", "id")
        if expected_segment and actual_segment and expected_segment != actual_segment:
            raise ValueError("retrieved row has a different segment ID")
        expected_metadata = _string(entry, "metadata_sha256")
        if expected_metadata and expected_metadata != _metadata_digest(row):
            raise ValueError("retrieved row metadata hash does not match candidate")
        if self.expected_pipeline_version is not None:
            expected_fields = {field.name for field in OUTPUT_SCHEMA.fields}
            if set(row) != expected_fields:
                raise ValueError("retrieved row does not have the exact output schema")
            if row.get("pipeline_version") != self.expected_pipeline_version:
                raise ValueError("retrieved row has a different pipeline version")
            if row.get("pipeline_config_sha256") != (
                self.expected_pipeline_config_sha256
            ):
                raise ValueError("retrieved row has a different pipeline digest")
        audio = _embedded_audio(row)
        expected_audio = _string(entry, "audio_sha256")
        if expected_audio and hashlib.sha256(audio).hexdigest() != expected_audio:
            raise ValueError("retrieved audio hash does not match candidate")
        return row

    def verify_repository(self) -> None:
        """Verify the pinned private dataset revision before retrieval.

        Raises:
            ValueError:
                If repository metadata is unavailable, public, or resolves to a
                different commit than the pinned revision.
        """
        getter = getattr(self.hub, "repo_info", None)
        if not callable(getter):
            raise ValueError("Hub adapter cannot verify the pinned dataset revision")
        try:
            info = getter(self.repository, repo_type="dataset", revision=self.revision)
        except Exception:
            raise ValueError("unable to verify the pinned dataset revision") from None
        private = (
            info.get("private")
            if isinstance(info, c.Mapping)
            else getattr(info, "private", None)
        )
        resolved_sha = (
            next(
                (
                    info.get(name)
                    for name in ("sha", "oid", "commit_id")
                    if isinstance(info, c.Mapping) and isinstance(info.get(name), str)
                ),
                None,
            )
            if isinstance(info, c.Mapping)
            else next(
                (
                    getattr(info, name)
                    for name in ("sha", "oid", "commit_id")
                    if isinstance(getattr(info, name, None), str)
                ),
                None,
            )
        )
        if private is not True or resolved_sha != self.revision:
            raise ValueError("pinned dataset is not private at the requested revision")
        self._repository_verified = True


def _embedded_audio(row: MetadataRow) -> bytes:
    """Extract bytes from a decoded Parquet audio feature.

    Returns:
        The embedded audio bytes.

    Raises:
        ValueError:
            If the row contains no embedded audio.
    """
    value = row.get("audio", row.get("waveform"))
    if isinstance(value, bytes):
        return value
    if isinstance(value, c.Mapping) and isinstance(value.get("bytes"), bytes):
        return t.cast(bytes, value["bytes"])
    raise ValueError("Parquet row does not contain embedded audio bytes")


def _row_at(dataset: object, locator: int) -> MetadataRow:
    iterator = iter(dataset) if isinstance(dataset, c.Iterable) else None
    if iterator is None:
        raise TypeError("Hub dataset must be iterable in streaming mode")
    for index, row in enumerate(iterator):
        if index == locator:
            if not isinstance(row, c.Mapping):
                raise TypeError("Hub dataset rows must be mappings")
            return row
    raise FileNotFoundError("requested Parquet row was not found")


def _validate_remote_shard_hash(
    hub: object, *, repository: str, revision: str, parquet_path: str, expected: str
) -> None:
    """Check the Hub's immutable file metadata without downloading the shard.

    Raises:
        FileNotFoundError:
            If Hub metadata has no entry for the shard.
        ValueError:
            If the expected or remote hash does not match.
    """
    if not _SHA256.fullmatch(expected):
        raise ValueError("parquet_sha256 must be lowercase SHA-256 hex")
    getter = getattr(hub, "get_paths_info", None)
    if getter is None:
        raise ValueError("Hub adapter cannot verify the candidate Parquet hash")
    try:
        info = list(
            getter(repository, [parquet_path], repo_type="dataset", revision=revision)
        )
    except Exception:
        raise ValueError("unable to verify the remote Parquet shard") from None
    if not info:
        raise FileNotFoundError("requested Parquet shard was not found")
    value = _object_value(info[0], "sha256") or _object_value(info[0], "oid")
    if value != expected:
        raise ValueError("remote Parquet hash does not match candidate")


def _object_value(value: object, name: str) -> str | None:
    if isinstance(value, c.Mapping):
        candidate = value.get(name)
        if isinstance(candidate, str):
            return candidate
        nested = value.get("lfs")
        if isinstance(nested, c.Mapping) and isinstance(nested.get(name), str):
            return t.cast(str, nested[name])
    candidate = getattr(value, name, None)
    if isinstance(candidate, str):
        return candidate
    nested = getattr(value, "lfs", None)
    candidate = getattr(nested, name, None)
    return candidate if isinstance(candidate, str) else None


class SourceClipRetriever:
    """Adapt a one-at-a-time source retrieval callback for rejected clips.

    The callback receives only the source locator from a candidate and must return
    bytes or a path.  It is deliberately not a cache: callers get one temporary
    review file and the file is removed when review finishes.
    """

    def __init__(self, callback: c.Callable[[MetadataRow], bytes | Path]) -> None:
        """Initialise the non-retaining source adapter."""
        self._callback = callback

    def retrieve(self, entry: MetadataRow) -> bytes | Path:
        """Retrieve one source interval.

        Returns:
            One source clip as bytes or a temporary path.

        Raises:
            ValueError:
                If the candidate lacks a complete source locator.
        """
        for key in ("source_repository", "source_revision", "source_file_id"):
            if not _string(entry, key):
                raise ValueError(f"candidate has no {key}")
        source_revision = _string(entry, "source_revision")
        if source_revision is None or not _COMMIT_SHA.fullmatch(source_revision):
            raise ValueError(
                "candidate source revision must be an immutable commit SHA"
            )
        for key in ("source_start_ms", "source_end_ms"):
            if _integer(entry, key) is None:
                raise ValueError(f"candidate has no {key}")
        return self._callback(entry)


def _asr_report(
    scores: c.Iterable[float], *, threshold: float = 0.5
) -> dict[str, object]:
    """Build an anomaly report from bounded, metadata-only scores.

    Returns:
        Counts, anomaly rate, and a normalised-WER distribution.
    """
    accumulator = _DistributionAccumulator()
    anomalies = 0
    for score in scores:
        accumulator.add(score)
        anomalies += score > threshold
    return {
        "segments": accumulator.count,
        "anomalies": anomalies,
        "anomaly_rate": anomalies / accumulator.count if accumulator.count else 0.0,
        "normalised_wer": accumulator.as_distribution().as_dict(),
    }


@dataclass(frozen=True)
class Distribution:
    """Small, JSON-safe descriptive distribution."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    p90: float | None

    def as_dict(self) -> dict[str, object]:
        """Return the distribution as a report-compatible mapping."""
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
        }


class _DistributionAccumulator:
    """Bounded-memory accumulator with a deterministic quantile reservoir."""

    def __init__(self, limit: int = 4_096) -> None:
        """Initialise an accumulator with a fixed reservoir size."""
        self.limit = limit
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.values: list[float] = []

    def add(self, value: float) -> None:
        """Add a value while retaining only a bounded deterministic reservoir."""
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        slot = (
            int(hashlib.sha256(str(self.count).encode("ascii")).hexdigest()[:16], 16)
            % self.count
        )
        if slot < self.limit:
            self.values[slot] = value

    def as_distribution(self) -> Distribution:
        """Return an approximate quantile distribution with exact totals."""
        if not self.count:
            return Distribution(0, None, None, None, None, None)
        ordered = sorted(self.values)
        return Distribution(
            count=self.count,
            minimum=self.minimum,
            maximum=self.maximum,
            mean=self.total / self.count,
            median=ordered[len(ordered) // 2],
            p90=ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.9) - 1)],
        )


def _distribution(values: list[float]) -> Distribution:
    if not values:
        return Distribution(0, None, None, None, None, None)
    ordered = sorted(values)
    return Distribution(
        count=len(values),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=mean(values),
        median=median(values),
        p90=ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.9) - 1)],
    )


def bounded_remote_aggregates(
    remote: object, *, split: str = "train", max_rows: int = 100_000
) -> dict[str, object]:
    """Compute metadata aggregates while consuming at most ``max_rows`` rows.

    Returns:
        A JSON-compatible totals and distributions report.
    """
    return aggregate_rows(
        stream_remote_rows(remote, split=split, max_rows=max_rows), bounded=True
    )


def aggregate_rows(rows: RowStream, *, bounded: bool = False) -> dict[str, object]:
    """Compute totals and distributions in one streaming pass.

    The function retains only scalar counters and numeric values used for quantiles;
    audio, transcript payloads, and row dictionaries are never copied.

    Returns:
        A JSON-compatible totals and distributions report.
    """
    totals = {
        "segments": 0,
        "accepted_segments": 0,
        "rejected_segments": 0,
        "borderline_segments": 0,
        "programmes": set[str](),
        "audio_ms": 0,
    }
    distributions = {
        "duration_ms": _DistributionAccumulator(),
        "word_count": _DistributionAccumulator(),
        "alignment_score": _DistributionAccumulator(),
        "vad_speech_ratio": _DistributionAccumulator(),
        "language_probability": _DistributionAccumulator(),
        "programme_position": _DistributionAccumulator(),
        "drift_ms": _DistributionAccumulator(),
    }
    by_show: dict[str, dict[str, float | int]] = {}
    by_position: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    music_counts = {"music": 0, "speech": 0}
    usable_transcript_ms = 0
    for row in rows:
        totals["segments"] += 1
        status = _status(row)
        totals[f"{status}_segments"] += 1
        file_id = _string(row, "source_file_id", "programme_id")
        if file_id:
            totals["programmes"].add(file_id)
        duration = _number(row, "duration_ms")
        if duration is not None:
            totals["audio_ms"] += int(duration)
            distributions["duration_ms"].add(duration)
        for name, keys in {
            "word_count": ("word_count", "words"),
            "alignment_score": ("alignment_score", "confidence"),
            "vad_speech_ratio": ("vad_speech_ratio", "speech_ratio"),
        }.items():
            value = _first_number(row, keys)
            if value is not None:
                distributions[name].add(value)
        language_probability = _number(row, "language_probability", "language_prob")
        if language_probability is not None:
            distributions["language_probability"].add(language_probability)
        programme_position = _number(row, "programme_position", "position")
        if programme_position is not None:
            distributions["programme_position"].add(programme_position)
        drift = _first_number(row, ("drift_ms", "start_drift_ms", "end_drift_ms"))
        if drift is not None:
            if "drift_ms" not in row:
                drift = max(
                    abs(_number(row, "start_drift_ms") or 0),
                    abs(_number(row, "end_drift_ms") or 0),
                )
            distributions["drift_ms"].add(abs(drift))
        show = _string(row, "show", "show_type", "programme_type") or "unknown"
        show_data = by_show.setdefault(show, {"segments": 0, "audio_ms": 0})
        show_data["segments"] = int(show_data["segments"]) + 1
        show_data["audio_ms"] = int(show_data["audio_ms"]) + int(duration or 0)
        position = _position_decile(row)
        by_position[position] = by_position.get(position, 0) + 1
        reason = _string(row, "rejection_reason", "reject_reason")
        if status == "rejected" and reason:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        speakers = row.get("speaker_ids", row.get("speakers", ()))
        stored_speaker_count = _number(row, "speaker_count")
        speaker_count = (
            int(stored_speaker_count)
            if stored_speaker_count is not None
            else (
                len(speakers)
                if isinstance(speakers, c.Sized)
                and not isinstance(speakers, (str, bytes))
                else 0
            )
        )
        speaker_key = str(speaker_count)
        speaker_counts[speaker_key] = speaker_counts.get(speaker_key, 0) + 1
        music_counts[
            "music" if _boolean(row, "music", "music_dominant") else "speech"
        ] += 1
        usable_transcript_ms += int(
            _number(row, "usable_transcript_duration_ms", "transcript_duration_ms") or 0
        )
    total_audio_hours = int(totals["audio_ms"]) / 3_600_000
    result: dict[str, object] = {
        "bounded": bounded,
        "totals": {
            "segments": totals["segments"],
            "accepted_segments": totals["accepted_segments"],
            "rejected_segments": totals["rejected_segments"],
            "borderline_segments": totals["borderline_segments"],
            "programmes": len(totals["programmes"]),
            "audio_ms": totals["audio_ms"],
            "audio_hours": total_audio_hours,
            "usable_transcript_duration_ms": usable_transcript_ms,
            "coverage": (
                int(totals["audio_ms"]) / usable_transcript_ms
                if usable_transcript_ms
                else None
            ),
        },
        "distributions": {
            name: values.as_distribution().as_dict()
            for name, values in distributions.items()
        },
        "per_show": {
            name: {**data, "audio_hours": int(data["audio_ms"]) / 3_600_000}
            for name, data in sorted(by_show.items())
        },
        "programme_position_deciles": dict(sorted(by_position.items())),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "music_counts": music_counts,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejection_reason_rates": {
            reason: count / totals["rejected_segments"]
            for reason, count in sorted(rejection_reasons.items())
        },
        "rejection_rate": totals["rejected_segments"] / totals["segments"]
        if totals["segments"]
        else 0.0,
    }
    return result


def _first_number(row: MetadataRow, keys: tuple[str, ...]) -> float | None:
    return _number(row, *keys)


def _position_decile(row: MetadataRow) -> str:
    value = _number(row, "programme_position", "position")
    if value is None:
        start = _number(row, "source_start_ms", "start_ms") or 0
        duration = _number(row, "programme_duration_ms", "source_duration_ms") or 1
        value = start / duration
    return f"d{min(9, max(0, int(value * 10)))}"


def stream_remote_rows(
    remote: object, *, split: str = "train", max_rows: int | None = None
) -> c.Iterator[MetadataRow]:
    """Adapt common streaming-client interfaces without loading a split.

    ``remote`` may itself be iterable, or expose ``iter_rows(split=...)`` or
    ``stream(split=...)``.  The optional limit is applied before any caller can
    retain rows.

    Yields:
        Metadata mappings from the selected split.

    Raises:
        TypeError:
            If the remote adapter does not provide an iterable or yields a non-mapping.
    """
    if hasattr(remote, "iter_rows"):
        iterator = getattr(remote, "iter_rows")(split=split)
    elif hasattr(remote, "stream"):
        iterator = getattr(remote, "stream")(split=split)
    else:
        iterator = remote
    if not isinstance(iterator, c.Iterable):
        raise TypeError("remote source must expose an iterable row stream")
    rows = iterator
    if max_rows is not None:
        rows = iter_bounded(source=rows, max_rows=max_rows)
    for row in rows:
        if not isinstance(row, c.Mapping):
            raise TypeError("remote rows must be mappings")
        yield row


def iter_bounded(
    source: c.Iterable[MetadataRow], max_rows: int
) -> c.Iterator[MetadataRow]:
    """Yield at most ``max_rows`` rows from a remote stream.

    Args:
        source:
            A one-shot iterable such as a Hugging Face streaming split.
        max_rows:
            Maximum number of rows to request from the iterable.

    Raises:
        ValueError:
            If ``max_rows`` is negative.
        TypeError:
            If a yielded row is not a mapping.
    """
    if max_rows < 0:
        raise ValueError("max_rows must not be negative")
    iterator = iter(source)
    for _ in range(max_rows):
        try:
            row = next(iterator)
        except StopIteration:
            return
        if not isinstance(row, c.Mapping):
            raise TypeError("stream rows must be mappings")
        yield row


def build_final_quality_report(
    rows: RowStream,
    *,
    database: Path | str,
    asr: IndependentASR | None = None,
    audio_loader: c.Callable[[MetadataRow], bytes] | None = None,
    audit_records: RowStream | None = None,
    asr_anomaly_threshold: float = 0.5,
) -> dict[str, object]:
    """Build the final report with optional independent-ASR anomaly metrics.

    Returns:
        A final quality report.
    """
    return build_quality_report(
        rows=rows,
        database=database,
        asr=asr,
        audio_loader=audio_loader,
        audit_records=audit_records,
        asr_anomaly_threshold=asr_anomaly_threshold,
    )


def build_quality_report(
    rows: RowStream,
    *,
    database: Path | str,
    asr: IndependentASR | None = None,
    audio_loader: c.Callable[[MetadataRow], bytes] | None = None,
    audit_records: RowStream | None = None,
    asr_anomaly_threshold: float = 0.5,
) -> dict[str, object]:
    """Build the structural/final report from one remote metadata stream.

    Returns:
        A final quality report containing totals, distributions, structural checks,
        and optional independent-ASR metrics.

    Raises:
        ValueError:
            If only one of ``asr`` and ``audio_loader`` is supplied.
    """
    if asr is not None or audio_loader is not None:
        if asr is None or audio_loader is None:
            raise ValueError("asr and audio_loader must be supplied together")
    if not math.isfinite(asr_anomaly_threshold) or asr_anomaly_threshold < 0:
        raise ValueError("asr_anomaly_threshold must be finite and non-negative")
    ledger = MetadataLedger(database, reset=True)
    asr_scores: _DistributionAccumulator | None = (
        _DistributionAccumulator() if asr is not None else None
    )
    asr_anomalies = 0
    try:
        for row in rows:
            ledger.add(row)
            if asr is not None and audio_loader is not None:
                hypothesis = asr.transcribe(audio_loader(row))
                reference = _string(row, "text", "transcript") or ""
                score = normalised_wer(reference, hypothesis)
                if asr_scores is not None:
                    asr_scores.add(score)
                    asr_anomalies += score > asr_anomaly_threshold
        report = aggregate_rows(ledger.rows())
        report["structural"] = ledger.quality_checks()
        if asr_scores is not None:
            report["independent_asr"] = {
                "segments": asr_scores.count,
                "anomalies": asr_anomalies,
                "anomaly_rate": (
                    asr_anomalies / asr_scores.count if asr_scores.count else 0.0
                ),
                "normalised_wer": asr_scores.as_distribution().as_dict(),
            }
        if audit_records is not None:
            report["manual_audit"] = summarise_manual_audit(audit_records)
        else:
            persisted_decisions = list(ledger.decision_rows())
            if persisted_decisions:
                report["manual_audit"] = summarise_manual_audit(persisted_decisions)
        report["report_type"] = "p1-final-quality"
        return report
    finally:
        ledger.close()


class MetadataLedger:
    """SQLite ledger containing only segment identity and scalar metadata."""

    def __init__(self, database: Path | str, *, reset: bool = False) -> None:
        """Open or create a metadata-only validation database.

        Args:
            database:
                SQLite database path.
            reset (optional):
                Whether to replace a previous report run. Defaults to ``False``.
        """
        database_path = Path(database)
        if str(database_path) != ":memory:":
            database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database))
        self.connection.execute("PRAGMA journal_mode=WAL")
        if reset:
            # A report rerun must not destroy the durable audit manifest or its
            # already-blinded decisions.
            self.connection.execute("DROP TABLE IF EXISTS segments")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS segments (
                row_id INTEGER PRIMARY KEY,
                segment_id TEXT NOT NULL,
                source_file_id TEXT NOT NULL,
                source_start_ms INTEGER,
                source_end_ms INTEGER,
                status TEXT NOT NULL,
                show TEXT,
                duration_ms INTEGER,
                word_count INTEGER,
                alignment_score REAL,
                vad_speech_ratio REAL,
                speaker_count INTEGER,
                language_probability REAL,
                programme_position REAL,
                drift_ms REAL,
                music INTEGER,
                usable_transcript_duration_ms INTEGER,
                rejection_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_candidates (
                audit_id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL,
                repository TEXT,
                revision TEXT,
                parquet_path TEXT,
                row_locator INTEGER NOT NULL,
                stratum_json TEXT NOT NULL,
                metadata_sha256 TEXT NOT NULL,
                audio_sha256 TEXT,
                parquet_sha256 TEXT,
                source_file_id TEXT,
                source_repository TEXT,
                source_revision TEXT,
                source_start_ms INTEGER,
                source_end_ms INTEGER,
                source_shard_path TEXT,
                source_row_group INTEGER,
                source_row_index INTEGER,
                source_shard_byte_size INTEGER,
                retrieved_at REAL
            );
            CREATE TABLE IF NOT EXISTS blind_decisions (
                audit_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                material_defect INTEGER NOT NULL DEFAULT 0,
                independent_asr_anomaly INTEGER,
                normalised_wer REAL,
                reviewed_at REAL NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS segments_id ON segments(segment_id);
            CREATE INDEX IF NOT EXISTS segments_source ON segments(source_file_id,
                source_start_ms, source_end_ms);
            """
        )
        for table, column, definition in (
            ("audit_candidates", "parquet_sha256", "TEXT"),
            ("audit_candidates", "source_file_id", "TEXT"),
            ("audit_candidates", "source_repository", "TEXT"),
            ("audit_candidates", "source_revision", "TEXT"),
            ("audit_candidates", "source_start_ms", "INTEGER"),
            ("audit_candidates", "source_end_ms", "INTEGER"),
            ("audit_candidates", "source_shard_path", "TEXT"),
            ("audit_candidates", "source_row_group", "INTEGER"),
            ("audit_candidates", "source_row_index", "INTEGER"),
            ("audit_candidates", "source_shard_byte_size", "INTEGER"),
            ("audit_candidates", "retrieved_at", "REAL"),
            ("blind_decisions", "details_json", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            columns = {
                str(value[1])
                for value in self.connection.execute(f"PRAGMA table_info({table})")
            }
            if column not in columns:
                self.connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )
        self.connection.commit()

    def add(self, row: MetadataRow) -> None:
        """Insert scalar metadata from one row; audio and text are excluded."""
        self.connection.execute(
            """INSERT INTO segments VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )""",
            (
                _identity(row),
                _string(row, "source_file_id", "programme_id") or "",
                _integer(row, "source_start_ms", "start_ms"),
                _integer(row, "source_end_ms", "end_ms"),
                _status(row),
                _string(row, "show", "show_type", "programme_type"),
                _integer(row, "duration_ms"),
                _integer(row, "word_count", "words"),
                _number(row, "alignment_score", "confidence"),
                _number(row, "vad_speech_ratio", "speech_ratio"),
                _integer(row, "speaker_count") or _speaker_count(row),
                _number(row, "language_probability", "language_prob"),
                _number(row, "programme_position", "position"),
                _number(row, "drift_ms", "start_drift_ms", "end_drift_ms"),
                int(_boolean(row, "music", "music_dominant")),
                _integer(
                    row, "usable_transcript_duration_ms", "transcript_duration_ms"
                ),
                _string(row, "rejection_reason", "reject_reason"),
            ),
        )
        self.connection.commit()

    def add_candidate(self, candidate: MetadataRow) -> None:
        """Persist one metadata-only candidate record.

        Raises:
            TypeError:
                If the stratum is not iterable.
        """
        _validate_candidate_locator(candidate, _status(candidate))
        stratum = candidate.get("stratum", ())
        if isinstance(stratum, (str, bytes)):
            stratum = [str(stratum)]
        if not isinstance(stratum, c.Iterable):
            raise TypeError("candidate stratum must be iterable")
        values = [str(value) for value in stratum]
        self.connection.execute(
            """INSERT OR REPLACE INTO audit_candidates
            (audit_id, segment_id, repository, revision, parquet_path,
             row_locator, stratum_json, metadata_sha256, audio_sha256,
             parquet_sha256, source_file_id, source_repository, source_revision,
             source_start_ms, source_end_ms, source_shard_path, source_row_group,
             source_row_index, source_shard_byte_size, retrieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _string(candidate, "audit_id") or "",
                _string(candidate, "segment_id") or "",
                _string(candidate, "repository", "repo_id"),
                _string(candidate, "revision", "hub_revision"),
                _parquet_path(candidate),
                _explicit_row_locator(candidate) or 0,
                json.dumps(values, separators=(",", ":")),
                _string(candidate, "metadata_sha256") or "",
                _string(candidate, "audio_sha256"),
                _string(candidate, "parquet_sha256", "shard_sha256"),
                _string(candidate, "source_file_id"),
                _string(candidate, "source_repository"),
                _string(candidate, "source_revision"),
                _integer(candidate, "source_start_ms"),
                _integer(candidate, "source_end_ms"),
                _source_shard_path(candidate),
                _integer(candidate, "source_row_group"),
                _integer(candidate, "source_row_index", "source_row_locator"),
                _integer(candidate, "source_shard_byte_size"),
                None,
            ),
        )
        self.connection.commit()

    def add_decision(self, decision: MetadataRow) -> None:
        """Persist a blinded decision and scalar anomaly evidence only.

        Raises:
            ValueError:
                If the decision lacks an audit ID or decision value.
        """
        audit_id = _string(decision, "audit_id")
        value = _string(decision, "decision")
        if not audit_id or value not in {"accepted", "rejected", "borderline"}:
            raise ValueError(
                "a blinded decision needs audit_id and an accepted, rejected, "
                "or borderline decision"
            )
        candidate_exists = self.connection.execute(
            "SELECT retrieved_at FROM audit_candidates WHERE audit_id = ?", (audit_id,)
        ).fetchone()
        if candidate_exists is None:
            raise ValueError("a decision needs a persisted audit candidate")
        if candidate_exists[0] is None:
            raise ValueError("a decision needs successful clip retrieval")
        wer = _number(decision, "normalised_wer", "wer")
        anomaly = decision.get("independent_asr_anomaly")
        details = {
            key: decision[key]
            for key in (
                "text_mismatch",
                "speech_clipping",
                "missing_words",
                "inserted_words",
                "speaker_mixing",
                "music_dominance",
                "boundary_clipping",
            )
            if key in decision and isinstance(decision[key], (bool, int, float))
        }
        self.connection.execute(
            """INSERT OR REPLACE INTO blind_decisions
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                audit_id,
                value,
                int(_boolean(decision, "material_defect")),
                int(anomaly) if isinstance(anomaly, bool) else None,
                wer,
                _number(decision, "reviewed_at") or time.time(),
                json.dumps(details, separators=(",", ":")),
            ),
        )
        self.connection.commit()

    def candidate_rows(self) -> c.Iterator[dict[str, object]]:
        """Yield persisted candidate metadata without source labels."""
        cursor = self.connection.execute(
            "SELECT audit_id, segment_id, repository, revision, parquet_path, "
            "row_locator, stratum_json, metadata_sha256, audio_sha256, "
            "parquet_sha256, source_file_id, source_repository, source_revision, "
            "source_start_ms, source_end_ms, source_shard_path, source_row_group, "
            "source_row_index, source_shard_byte_size FROM audit_candidates "
            "ORDER BY audit_id"
        )
        for values in cursor:
            yield {
                "audit_id": values[0],
                "segment_id": values[1],
                "repository": values[2],
                "revision": values[3],
                "parquet_path": values[4],
                "remote_parquet_path": values[4],
                "row_locator": values[5],
                "stratum": json.loads(values[6]),
                "metadata_sha256": values[7],
                **({"audio_sha256": values[8]} if values[8] else {}),
                **({"parquet_sha256": values[9]} if values[9] else {}),
                **({"source_file_id": values[10]} if values[10] else {}),
                **({"source_repository": values[11]} if values[11] else {}),
                **({"source_revision": values[12]} if values[12] else {}),
                **({"source_start_ms": values[13]} if values[13] is not None else {}),
                **({"source_end_ms": values[14]} if values[14] is not None else {}),
                **({"source_shard_path": values[15]} if values[15] is not None else {}),
                **({"source_row_group": values[16]} if values[16] is not None else {}),
                **({"source_row_index": values[17]} if values[17] is not None else {}),
                **(
                    {"source_shard_byte_size": values[18]}
                    if values[18] is not None
                    else {}
                ),
            }

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()

    def decision_rows(self) -> c.Iterator[dict[str, object]]:
        """Yield persisted blinded decisions and scalar anomaly evidence."""
        cursor = self.connection.execute(
            "SELECT audit_id, decision, material_defect, independent_asr_anomaly, "
            "normalised_wer, reviewed_at, details_json FROM blind_decisions "
            "ORDER BY audit_id"
        )
        for values in cursor:
            result: dict[str, object] = {
                "audit_id": values[0],
                "decision": values[1],
                "material_defect": bool(values[2]),
                "reviewed_at": values[5],
            }
            if values[3] is not None:
                result["independent_asr_anomaly"] = bool(values[3])
            if values[4] is not None:
                result["normalised_wer"] = values[4]
            details = json.loads(values[6])
            if isinstance(details, dict):
                result.update(details)
            yield result

    def mark_retrieved(self, audit_id: str) -> None:
        """Record successful validation of one temporary clip retrieval.

        Raises:
            ValueError:
                If the audit ID is not persisted.
        """
        updated = self.connection.execute(
            "UPDATE audit_candidates SET retrieved_at = ? WHERE audit_id = ?",
            (time.time(), audit_id),
        )
        if updated.rowcount != 1:
            raise ValueError("cannot mark an unknown audit candidate as retrieved")
        self.connection.commit()

    def quality_checks(self) -> dict[str, object]:
        """Return duplicate-ID and same-source overlapping-interval findings."""
        duplicate_ids = [
            row[0]
            for row in self.connection.execute(
                "SELECT segment_id FROM segments "
                "GROUP BY segment_id HAVING COUNT(*) > 1 ORDER BY segment_id"
            )
        ]
        overlap_rows = self.connection.execute(
            """SELECT a.segment_id, b.segment_id, a.source_file_id,
                      a.source_start_ms, a.source_end_ms,
                      b.source_start_ms, b.source_end_ms
               FROM segments a JOIN segments b
                 ON a.source_file_id = b.source_file_id AND a.row_id < b.row_id
                AND a.source_start_ms IS NOT NULL AND b.source_start_ms IS NOT NULL
                AND a.source_end_ms IS NOT NULL AND b.source_end_ms IS NOT NULL
                AND a.source_start_ms < b.source_end_ms
                AND b.source_start_ms < a.source_end_ms
              ORDER BY a.row_id, b.row_id"""
        )
        overlaps = [
            {
                "segment_id": values[0],
                "other_segment_id": values[1],
                "source_file_id": values[2],
                "interval": [values[3], values[4]],
                "other_interval": [values[5], values[6]],
            }
            for values in overlap_rows
        ]
        return {
            "duplicate_segment_ids": duplicate_ids,
            "overlapping_source_intervals": overlaps,
            "passes": not duplicate_ids and not overlaps,
        }

    def rows(self) -> c.Iterator[dict[str, object]]:
        """Yield report rows reconstructed from scalar database columns."""
        cursor = self.connection.execute("SELECT * FROM segments ORDER BY row_id")
        columns = [description[0] for description in cursor.description or ()]
        for values in cursor:
            yield dict(zip(columns, values, strict=True))


def _speaker_count(row: MetadataRow) -> int:
    """Return a scalar speaker count for metadata ledger storage."""
    speakers = row.get("speaker_ids", row.get("speakers", ()))
    if isinstance(speakers, (str, bytes)):
        return 1
    if isinstance(speakers, c.Sized):
        return len(speakers)
    return 0


def normalised_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate after deterministic Danish-safe normalisation.

    Returns:
        Word error rate in the range zero to one or above for many errors.
    """
    reference_words = _normalise_words(reference)
    hypothesis_words = _normalise_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for ref_index, reference_word in enumerate(reference_words, 1):
        current = [ref_index]
        for hyp_index, hypothesis_word in enumerate(hypothesis_words, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def _normalise_words(text: str) -> list[str]:
    normalised = unicodedata.normalize("NFC", text).casefold()
    return [
        word
        for word in "".join(
            char if char.isalnum() else " " for char in normalised
        ).split()
    ]


def summarise_manual_audit(records: RowStream) -> dict[str, object]:
    """Summarise blinded manual decisions and material defects.

    Returns:
        Counts by decision and the clean-audit rate.  Defect fields are deliberately
        read only from reviewer output, never from segmentation status.
    """
    by_decision: dict[str, int] = {}
    total = 0
    material_defects = 0
    asr_anomalies = 0
    asr_observations = 0
    defect_fields = (
        "material_defect",
        "text_mismatch",
        "speech_clipping",
        "missing_words",
        "inserted_words",
        "speaker_mixing",
        "music_dominance",
        "boundary_clipping",
    )
    for record in records:
        total += 1
        decision = _string(record, "decision") or "unrecorded"
        by_decision[decision] = by_decision.get(decision, 0) + 1
        if any(_boolean(record, field) for field in defect_fields):
            material_defects += 1
        if "independent_asr_anomaly" in record:
            asr_observations += 1
            asr_anomalies += _boolean(record, "independent_asr_anomaly")
    return {
        "audited": total,
        "by_decision": dict(sorted(by_decision.items())),
        "material_defects": material_defects,
        "clean_rate": (total - material_defects) / total if total else 0.0,
        "independent_asr_observations": asr_observations,
        "independent_asr_anomalies": asr_anomalies,
        "independent_asr_anomaly_rate": (
            asr_anomalies / asr_observations if asr_observations else 0.0
        ),
    }


def build_representative_audit_candidates(
    rows: RowStream,
    *,
    accepted_quota: int = 200,
    rejected_quota: int = 100,
    borderline_quota: int = 0,
    seed: int | str = 0,
) -> list[dict[str, object]]:
    """Return bounded representative candidates for the later assembler."""
    return create_blinded_audit_manifest(
        rows,
        accepted_quota=accepted_quota,
        rejected_quota=rejected_quota,
        borderline_quota=borderline_quota,
        seed=seed,
    )


def build_structural_report(
    rows: RowStream, *, database: Path | str
) -> dict[str, object]:
    """Build totals, distributions, and metadata integrity findings.

    Returns:
        A structural validation report.
    """
    return build_quality_report(rows=rows, database=database)


def check_duplicate_and_overlaps(
    rows: RowStream, database: Path | str
) -> dict[str, object]:
    """Check IDs and source intervals using a metadata-only SQLite database.

    Returns:
        Duplicate and overlap findings.
    """
    ledger = MetadataLedger(database, reset=True)
    try:
        for row in rows:
            ledger.add(row)
        return ledger.quality_checks()
    finally:
        ledger.close()


def deterministic_deciles(values: c.Iterable[float]) -> list[int]:
    """Return stable rank deciles, retaining input order in the result."""
    numbers = list(values)
    order = sorted(range(len(numbers)), key=lambda index: (numbers[index], index))
    result = [0] * len(numbers)
    for rank, index in enumerate(order):
        result[index] = min(9, rank * 10 // max(1, len(numbers)))
    return result


def export_clip_for_review(
    entry: MetadataRow, retriever: ClipRetriever, destination: Path
) -> Path:
    """Retrieve one clip to a caller-selected path without recording a decision.

    The temporary retrieval file is removed before this function returns.  This is
    the non-playing path for a reviewer who controls playback outside the CLI.

    Returns:
        The requested destination path.
    """
    source = retrieve_one_for_review(entry, retriever)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination
    finally:
        source.unlink(missing_ok=True)


def retrieve_one_for_review(
    entry: MetadataRow, retriever: ClipRetriever, *, temporary_root: Path | None = None
) -> Path:
    """Retrieve one audit clip into a uniquely named temporary file.

    Returns:
        A temporary path containing the single retrieved clip.

    Raises:
        TypeError:
            If the retriever returns neither bytes nor a path.
        ValueError:
            If the locator is invalid or retrieved audio is empty.
    """
    root = temporary_root.expanduser() if temporary_root else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    _validate_candidate_locator(entry, _status(entry))
    result = retriever.retrieve(entry)
    if isinstance(result, Path):
        suffix = result.suffix or ".audio"
        handle, target = tempfile.mkstemp(prefix="p1-review-", suffix=suffix, dir=root)
        os.close(handle)
        shutil.copyfile(result, target)
        if Path(target).stat().st_size == 0:
            Path(target).unlink(missing_ok=True)
            raise ValueError("retrieved clip is empty")
        return Path(target)
    if not isinstance(result, bytes):
        raise TypeError("clip retriever must return bytes or a Path")
    if not result:
        raise ValueError("retrieved clip is empty")
    handle, target = tempfile.mkstemp(prefix="p1-review-", suffix=".ogg", dir=root)
    with os.fdopen(handle, "wb") as output:
        output.write(result)
        output.flush()
        os.fsync(output.fileno())
    return Path(target)


def persist_audit_candidates(database: Path | str, candidates: RowStream) -> int:
    """Persist metadata-only audit candidates and return the number written.

    Returns:
        Number of candidate records written.
    """
    ledger = MetadataLedger(database)
    count = 0
    try:
        for candidate in candidates:
            ledger.add_candidate(candidate)
            count += 1
    finally:
        ledger.close()
    return count


def persist_blinded_decision(
    database: Path | str, decision: MetadataRow
) -> dict[str, object]:
    """Persist one metadata-only decision in a SQLite validation ledger.

    Returns:
        The scalar-only decision that was stored.
    """
    result = _decision_copy(decision, audit_id=_string(decision, "audit_id"))
    ledger = MetadataLedger(database)
    try:
        ledger.add_decision(result)
    finally:
        ledger.close()
    return result


def _decision_copy(decision: MetadataRow, *, audit_id: str | None) -> dict[str, object]:
    """Retain only fields permitted in a durable blinded-decision record.

    Returns:
        A scalar-only decision record.

    Raises:
        ValueError:
            If ``audit_id`` is missing.
    """
    if not audit_id:
        raise ValueError("a blinded decision needs audit_id")
    allowed = {
        "decision",
        "material_defect",
        "text_mismatch",
        "speech_clipping",
        "missing_words",
        "inserted_words",
        "speaker_mixing",
        "music_dominance",
        "boundary_clipping",
        "independent_asr_anomaly",
        "normalised_wer",
        "wer",
        "reviewed_at",
    }
    result: dict[str, object] = {"audit_id": audit_id}
    for key in allowed:
        if key in decision and isinstance(decision[key], (str, int, float, bool)):
            result[key] = decision[key]
    if result.get("decision") not in {"accepted", "rejected", "borderline"}:
        raise ValueError("a review decision must be accepted, rejected, or borderline")
    return result


def play_audio(path: Path, player: str | c.Sequence[str] | None = None) -> None:
    """Play one temporary clip using a safe argument vector.

    Args:
        path:
            Audio file to play.
        player (optional):
            Executable and optional arguments, either as a sequence or a shell-like
            command string. Defaults to the first available platform player.

    """
    command = _player_command(player)
    subprocess.run([*command, str(path)], check=True)


def _player_command(player: str | c.Sequence[str] | None) -> list[str]:
    if player is not None:
        command = shlex.split(player) if isinstance(player, str) else list(player)
        if not command:
            raise ValueError("audio player command must not be empty")
        return command
    candidates = (
        (("afplay",) if sys.platform == "darwin" else ())
        + (("ffplay", "-nodisp", "-autoexit", "-loglevel", "error"),)
        + (("paplay",), ("aplay",))
    )
    for candidate in candidates:
        if shutil.which(candidate[0]):
            return list(candidate)
    raise FileNotFoundError("no local audio player is available")


def review_one_clip(
    entry: MetadataRow,
    retriever: ClipRetriever,
    reviewer: c.Callable[[Path, MetadataRow], MetadataRow],
    *,
    temporary_root: Path | None = None,
    decision_store: MetadataLedger | None = None,
    player: c.Callable[[Path], None] | None = None,
) -> dict[str, object]:
    """Run one blind review, persist its decision, and delete temporary audio.

    The reviewer sees only a sanitised candidate, so segmentation status cannot leak
    through the callback.  ``decision_store`` receives scalar fields only.

    Returns:
        The persisted scalar-only decision.

    Raises:
        TypeError:
            If the reviewer does not return a mapping.
    """
    path = retrieve_one_for_review(entry, retriever, temporary_root=temporary_root)
    try:
        if decision_store is not None:
            decision_store.mark_retrieved(_string(entry, "audit_id") or "")
        if player is not None:
            player(path)
        blind_entry = _blind_candidate(entry)
        decision = reviewer(path, blind_entry)
        if not isinstance(decision, c.Mapping):
            raise TypeError("reviewer must return a mapping")
        result = _decision_copy(decision, audit_id=_string(entry, "audit_id"))
        if decision_store is not None:
            decision_store.add_decision(result)
        return result
    finally:
        path.unlink(missing_ok=True)


def _blind_candidate(entry: MetadataRow) -> dict[str, object]:
    """Remove source labels and sensitive payloads before invoking a reviewer.

    Returns:
        A candidate mapping safe to pass to a blind reviewer.
    """
    hidden = {
        "status",
        "quality_status",
        "accepted",
        "borderline",
        "rejection_reason",
        "reject_reason",
        "decision",
        "text",
        "transcript",
        "audio",
        "waveform",
    }
    return {key: value for key, value in entry.items() if key not in hidden}


def score_asr_anomalies(
    rows: RowStream,
    asr: IndependentASR,
    audio_loader: c.Callable[[MetadataRow], bytes],
    *,
    anomaly_threshold: float = 0.5,
) -> dict[str, object]:
    """Score independent-ASR normalised WER without storing audio or hypotheses.

    Returns:
        Counts, anomaly rate, and a WER distribution.

    Raises:
        ValueError:
            If ``anomaly_threshold`` is negative or non-finite.
    """
    if not math.isfinite(anomaly_threshold) or anomaly_threshold < 0:
        raise ValueError("anomaly_threshold must be a finite non-negative number")
    accumulator = _DistributionAccumulator()
    anomalies = 0
    for row in rows:
        audio = audio_loader(row)
        hypothesis = asr.transcribe(audio)
        reference = _string(row, "text", "transcript") or ""
        score = normalised_wer(reference, hypothesis)
        accumulator.add(score)
        anomalies += score > anomaly_threshold
    return {
        "segments": accumulator.count,
        "anomalies": anomalies,
        "anomaly_rate": anomalies / accumulator.count if accumulator.count else 0.0,
        "normalised_wer": accumulator.as_distribution().as_dict(),
    }


def stratified_sample(
    rows: RowStream, *, sample_size: int, seed: int | str = 0
) -> list[dict[str, object]]:
    """Select a deterministic, multi-dimensional sample from metadata rows.

    Strata contain duration, show, speaker count, music, language probability,
    programme position, confidence and drift deciles.  A stable digest rather than
    process-global random state makes the result independent of hash randomisation.

    Returns:
        The selected metadata rows in stable identity order.

    Raises:
        ValueError:
            If ``sample_size`` is negative.
    """
    if sample_size < 0:
        raise ValueError("sample_size must not be negative")
    reservoir = _StratifiedReservoir(sample_size)
    for ordinal, raw_row in enumerate(rows):
        row = _metadata_copy(raw_row)
        internal_metadata = raw_row.get(_AUDIO_POINTER_METADATA_KEY)
        if internal_metadata is not None:
            if not isinstance(internal_metadata, str):
                raise ValueError("audio pointer metadata must be an internal string")
            if (
                len(internal_metadata.encode("utf-8"))
                > _AUDIO_POINTER_METADATA_MAX_BYTES
            ):
                raise ValueError("audio pointer metadata exceeds its size limit")
            row[_AUDIO_POINTER_METADATA_KEY] = internal_metadata
        reservoir.add(row, seed=str(seed), ordinal=ordinal)
    selected = reservoir.rows()
    selected.sort(key=lambda row: (_identity(row), _stratum_key(row)))
    return selected


# Descriptive aliases keep the public API discoverable for report callers.
bounded_stream_aggregates = bounded_remote_aggregates
build_audit_manifest = create_blinded_audit_manifest
HfClipRetriever = PinnedHubClipRetriever
find_duplicate_segment_ids = check_duplicate_and_overlaps
normalised_word_error_rate = normalised_wer

__all__ = [
    "AuditCandidate",
    "ClipRetriever",
    "SourceClipRetriever",
    "PinnedHubClipRetriever",
    "HfClipRetriever",
    "IndependentASR",
    "MetadataLedger",
    "aggregate_rows",
    "bounded_remote_aggregates",
    "bounded_stream_aggregates",
    "build_audit_manifest",
    "build_final_quality_report",
    "build_quality_report",
    "build_structural_report",
    "check_duplicate_and_overlaps",
    "find_duplicate_segment_ids",
    "create_blinded_audit_manifest",
    "build_representative_audit_candidates",
    "deterministic_deciles",
    "iter_bounded",
    "normalised_wer",
    "normalised_word_error_rate",
    "review_one_clip",
    "export_clip_for_review",
    "play_audio",
    "persist_audit_candidates",
    "persist_blinded_decision",
    "score_asr_anomalies",
    "stratified_sample",
    "summarise_manual_audit",
    "stream_remote_rows",
]
