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
import re
import shutil
import sqlite3
import tempfile
import time
import typing as t
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

MetadataRow = c.Mapping[str, object]
RowStream = c.Iterable[MetadataRow]

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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

    ``hub`` only needs the small ``load_dataset`` method exposed by the publication
    adapter, which keeps this class straightforward to test without network access.
    Dataset iteration may scan preceding rows, but only the addressed row's audio is
    returned and no dataset or row is retained.
    """

    def __init__(
        self,
        hub: object,
        *,
        repository: str | None = None,
        repo_id: str | None = None,
        revision: str,
    ) -> None:
        """Initialise a retriever pinned to a complete Hub commit SHA.

        Raises:
            ValueError:
                If revision is not a complete commit SHA.
        """
        if not _COMMIT_SHA.fullmatch(revision):
            raise ValueError("revision must be a complete 40-character commit SHA")
        selected_repository = repository or repo_id
        if not selected_repository:
            raise ValueError("repository must be supplied")
        self.hub = hub
        self.repository = selected_repository
        self.revision = revision

    def retrieve(self, entry: MetadataRow) -> bytes:
        """Retrieve and verify one clip represented by a candidate record.

        Returns:
            The addressed embedded audio bytes.

        Raises:
            ValueError:
                If the candidate is malformed or metadata/audio hashes differ.
            TypeError:
                If the Hub adapter or returned rows are not stream-compatible.
        """
        repository = _string(entry, "repository", "repo_id")
        revision = _string(entry, "revision")
        if repository and repository != self.repository:
            raise ValueError("candidate repository differs from pinned repository")
        if revision and revision != self.revision:
            raise ValueError("candidate revision differs from pinned revision")
        parquet_path = _parquet_path(entry)
        if not parquet_path:
            raise ValueError("candidate has no remote Parquet path")
        expected_shard = _string(entry, "parquet_sha256", "shard_sha256")
        if expected_shard:
            _validate_remote_shard_hash(
                self.hub,
                repository=self.repository,
                revision=self.revision,
                parquet_path=parquet_path,
                expected=expected_shard,
            )
        locator = _row_locator(entry, 0)
        if locator < 0:
            raise ValueError("candidate row locator must be non-negative")
        loader = getattr(self.hub, "load_dataset", None)
        if loader is None:
            raise TypeError("Hub adapter must provide load_dataset")
        dataset = loader(
            self.repository,
            shard_path=parquet_path,
            revision=self.revision,
            streaming=True,
        )
        row = _row_at(dataset, locator)
        expected_segment = _string(entry, "segment_id")
        actual_segment = _string(row, "segment_id", "id")
        if expected_segment and actual_segment and expected_segment != actual_segment:
            raise ValueError("retrieved row has a different segment ID")
        expected_metadata = _string(entry, "metadata_sha256")
        if expected_metadata and expected_metadata != _metadata_digest(row):
            raise ValueError("retrieved row metadata hash does not match candidate")
        audio = _embedded_audio(row)
        expected_audio = _string(entry, "audio_sha256")
        if expected_audio and hashlib.sha256(audio).hexdigest() != expected_audio:
            raise ValueError("retrieved audio hash does not match candidate")
        return audio


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


def _metadata_digest(row: MetadataRow) -> str:
    """Hash metadata without retaining its values in an audit record.

    Returns:
        A SHA-256 digest of non-audio metadata.
    """
    ignored = {
        "audio",
        "waveform",
        "array",
        "input_values",
        "status",
        "quality_status",
        "accepted",
        "borderline",
        "rejection_reason",
        "reject_reason",
        "_p1_metadata_sha256",
        "_p1_audio_sha256",
        "audio_sha256",
        "repository",
        "repo_id",
        "hub_repo",
        "revision",
        "hub_revision",
        "parquet_path",
        "remote_parquet_path",
        "shard_path",
        "source_parquet",
        "parquet_sha256",
        "shard_sha256",
        "audio_sha256",
        "metadata_sha256",
        "row_locator",
        "row_index",
        "parquet_row",
    }
    payload = {key: value for key, value in row.items() if key not in ignored}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _parquet_path(row: MetadataRow) -> str | None:
    """Read a Parquet object path, never treating an audio path as one.

    Returns:
        A Parquet path, or ``None`` when the row has no such locator.
    """
    value = _string(
        row, "parquet_path", "remote_parquet_path", "shard_path", "source_parquet"
    )
    return value if value and value.lower().endswith(".parquet") else None


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


def _row_at(dataset: object, locator: int) -> MetadataRow:
    iterator = iter(dataset) if isinstance(dataset, c.Iterable) else None
    if iterator is None:
        raise TypeError("Hub dataset must be iterable in streaming mode")
    for index, row in enumerate(iterator):
        if index == locator:
            if not isinstance(row, c.Mapping):
                raise TypeError("Hub dataset rows must be mappings")
            return row
    raise FileNotFoundError(f"Parquet row {locator} was not found")


def _row_locator(row: MetadataRow, fallback: int) -> int:
    for key in ("row_locator", "row_index", "parquet_row"):
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


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
    info = list(
        getter(repository, [parquet_path], repo_type="dataset", revision=revision)
    )
    if not info:
        raise FileNotFoundError(parquet_path)
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


def _first_number(row: MetadataRow, keys: tuple[str, ...]) -> float | None:
    return _number(row, *keys)


def _number(row: MetadataRow, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _position_decile(row: MetadataRow) -> str:
    value = _number(row, "programme_position", "position")
    if value is None:
        start = _number(row, "source_start_ms", "start_ms") or 0
        duration = _number(row, "programme_duration_ms", "source_duration_ms") or 1
        value = start / duration
    return f"d{min(9, max(0, int(value * 10)))}"


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
                parquet_sha256 TEXT
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
        stratum = candidate.get("stratum", ())
        if isinstance(stratum, (str, bytes)):
            stratum = [str(stratum)]
        if not isinstance(stratum, c.Iterable):
            raise TypeError("candidate stratum must be iterable")
        values = [str(value) for value in stratum]
        self.connection.execute(
            """INSERT OR REPLACE INTO audit_candidates
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _string(candidate, "audit_id") or "",
                _string(candidate, "segment_id") or "",
                _string(candidate, "repository", "repo_id"),
                _string(candidate, "revision"),
                _parquet_path(candidate),
                _row_locator(candidate, 0),
                json.dumps(values, separators=(",", ":")),
                _string(candidate, "metadata_sha256") or "",
                _string(candidate, "audio_sha256"),
                _string(candidate, "parquet_sha256", "shard_sha256"),
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
        if not audit_id or not value:
            raise ValueError("a blinded decision needs audit_id and decision")
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
            "parquet_sha256 FROM audit_candidates ORDER BY audit_id"
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
        "id",
        "source_file_id",
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
        "parquet_sha256",
        "shard_sha256",
        "audio_sha256",
        "metadata_sha256",
        "row_locator",
        "row_index",
        "parquet_row",
    }
    return {key: value for key, value in row.items() if key in allowed}


def _integer(row: MetadataRow, *keys: str) -> int | None:
    value = _number(row, *keys)
    return int(value) if value is not None else None


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
        Metadata-only candidate records.  A candidate has a Parquet path and row
        locator, not a presumed direct audio path.

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
            safe_row = _metadata_copy(row)
            safe_row["_p1_metadata_sha256"] = _string(
                row, "metadata_sha256"
            ) or _metadata_digest(row)
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
        audit_id = hashlib.sha256(f"{seed}\0{ordinal}\0{identity}".encode()).hexdigest()
        candidate = AuditCandidate(
            audit_id=audit_id,
            segment_id=identity,
            repository=_string(row, "repository", "repo_id", "hub_repo"),
            revision=_string(row, "revision", "hub_revision"),
            parquet_path=_parquet_path(row),
            row_locator=_row_locator(row, ordinal),
            stratum=_stratum_key(row),
            metadata_sha256=(
                _string(row, "_p1_metadata_sha256") or _metadata_digest(row)
            ),
            audio_sha256=_string(row, "_p1_audio_sha256", "audio_sha256"),
            parquet_sha256=_string(
                row, "parquet_sha256", "shard_sha256", "_p1_parquet_sha256"
            ),
        )
        manifest.append(candidate.as_dict())
    return manifest


@dataclass(frozen=True)
class AuditCandidate:
    """Durable, metadata-only location for one blinded audit item.

    The candidate deliberately contains a Parquet object and row locator rather than
    an audio path.  It is safe to persist in a manifest or SQLite database.
    """

    audit_id: str
    segment_id: str
    repository: str | None
    revision: str | None
    parquet_path: str | None
    row_locator: int
    stratum: tuple[str, ...]
    metadata_sha256: str
    audio_sha256: str | None = None
    parquet_sha256: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible metadata record without the source label."""
        result: dict[str, object] = {
            "audit_id": self.audit_id,
            "segment_id": self.segment_id,
            "repository": self.repository,
            "revision": self.revision,
            "parquet_path": self.parquet_path,
            "remote_parquet_path": self.parquet_path,
            "row_locator": self.row_locator,
            "stratum": list(self.stratum),
            "metadata_sha256": self.metadata_sha256,
        }
        if self.audio_sha256 is not None:
            result["audio_sha256"] = self.audio_sha256
        if self.parquet_sha256 is not None:
            result["parquet_sha256"] = self.parquet_sha256
        return result


class _StratifiedReservoir:
    """Order-independent reservoir with bounded groups and records."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.groups: dict[tuple[str, ...], list[tuple[str, dict[str, object]]]] = {}

    def add(self, row: dict[str, object], *, seed: str, ordinal: int) -> None:
        if not self.capacity:
            return
        key = _stratum_key(row)
        identity = _identity(row)
        digest = hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()
        if key not in self.groups and len(self.groups) >= self.capacity:
            new_priority = hashlib.sha256(
                f"{seed}\0stratum\0{key}".encode()
            ).hexdigest()
            worst = max(
                self.groups,
                key=lambda item: hashlib.sha256(
                    f"{seed}\0stratum\0{item}".encode()
                ).hexdigest(),
            )
            if (
                new_priority
                >= hashlib.sha256(f"{seed}\0stratum\0{worst}".encode()).hexdigest()
            ):
                return
            del self.groups[worst]
        bucket = self.groups.setdefault(key, [])
        bucket.append((digest, row))
        bucket.sort(key=lambda item: (item[0], _identity(item[1])))
        limit = max(1, self.capacity // max(1, len(self.groups)))
        for values in self.groups.values():
            del values[limit:]
        # ``ordinal`` is accepted to make callers explicit about stream position;
        # identity-derived priorities are used so reversing a remote stream is safe.
        del ordinal

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


def _explicit_or_decile(row: MetadataRow, keys: tuple[str, ...], value: float) -> str:
    """Use a supplied decile or derive one from a normalised numeric value.

    Returns:
        A canonical ``d0`` to ``d9`` label.
    """
    explicit = _string(row, *keys)
    if explicit:
        return explicit if explicit.startswith("d") else f"d{explicit}"
    return f"d{min(9, max(0, int(value * 10)))}"


def _audio_digest(row: MetadataRow) -> str | None:
    value = row.get("audio", row.get("waveform"))
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, c.Mapping) and isinstance(value.get("bytes"), bytes):
        return hashlib.sha256(t.cast(bytes, value["bytes"])).hexdigest()
    return None


def _candidate_source_rows(
    rows: c.Iterable[dict[str, object]], *, status: str, seed: str
) -> list[dict[str, object]]:
    """Keep status selection internal while returning only selected source rows.

    Returns:
        The selected metadata rows.
    """
    del status, seed
    return list(rows)


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
    if "decision" not in result:
        result["decision"] = "unrecorded"
    return result


def review_one_clip(
    entry: MetadataRow,
    retriever: ClipRetriever,
    reviewer: c.Callable[[Path, MetadataRow], MetadataRow],
    *,
    temporary_root: Path | None = None,
    decision_store: MetadataLedger | None = None,
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


def retrieve_one_for_review(
    entry: MetadataRow, retriever: ClipRetriever, *, temporary_root: Path | None = None
) -> Path:
    """Retrieve one audit clip into a uniquely named temporary file.

    Returns:
        A temporary path containing the single retrieved clip.

    Raises:
        TypeError:
            If the retriever returns neither bytes nor a path.
    """
    root = temporary_root.expanduser() if temporary_root else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    result = retriever.retrieve(entry)
    if isinstance(result, Path):
        suffix = result.suffix or ".audio"
        handle, target = tempfile.mkstemp(prefix="p1-review-", suffix=suffix, dir=root)
        os.close(handle)
        shutil.copyfile(result, target)
        return Path(target)
    if not isinstance(result, bytes):
        raise TypeError("clip retriever must return bytes or a Path")
    handle, target = tempfile.mkstemp(prefix="p1-review-", suffix=".flac", dir=root)
    with os.fdopen(handle, "wb") as output:
        output.write(result)
        output.flush()
        os.fsync(output.fileno())
    return Path(target)


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
        reservoir.add(_metadata_copy(raw_row), seed=str(seed), ordinal=ordinal)
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
    "persist_audit_candidates",
    "persist_blinded_decision",
    "score_asr_anomalies",
    "stratified_sample",
    "summarise_manual_audit",
    "stream_remote_rows",
]
