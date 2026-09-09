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
import shutil
import sqlite3
import tempfile
import typing as t
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

MetadataRow = c.Mapping[str, object]
RowStream = c.Iterable[MetadataRow]


class ClipRetriever(t.Protocol):
    """Retrieve one audit clip into a temporary location or memory."""

    def retrieve(self, entry: MetadataRow) -> bytes | Path:
        """Retrieve the clip represented by an audit manifest entry."""


class IndependentASR(t.Protocol):
    """Minimal interface required by the independent-ASR anomaly detector."""

    def transcribe(self, audio: bytes) -> str:
        """Transcribe one clip without modifying or retaining it."""


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
    )


def build_quality_report(
    rows: RowStream,
    *,
    database: Path | str,
    asr: IndependentASR | None = None,
    audio_loader: c.Callable[[MetadataRow], bytes] | None = None,
    audit_records: RowStream | None = None,
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
    ledger = MetadataLedger(database, reset=True)
    asr_scores: list[float] = []
    try:
        for row in rows:
            ledger.add(row)
            if asr is not None and audio_loader is not None:
                hypothesis = asr.transcribe(audio_loader(row))
                reference = _string(row, "text", "transcript") or ""
                asr_scores.append(normalised_wer(reference, hypothesis))
        report = aggregate_rows(ledger.rows())
        report["structural"] = ledger.quality_checks()
        if asr is not None:
            report["independent_asr"] = _asr_report(asr_scores)
        if audit_records is not None:
            report["manual_audit"] = summarise_manual_audit(audit_records)
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
            CREATE INDEX IF NOT EXISTS segments_id ON segments(segment_id);
            CREATE INDEX IF NOT EXISTS segments_source ON segments(source_file_id,
                source_start_ms, source_end_ms);
            """
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

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()

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
    return {
        key: value for key, value in row.items() if key not in {"audio", "waveform"}
    }


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


def _asr_report(scores: list[float], *, threshold: float = 0.5) -> dict[str, object]:
    """Build an anomaly report from already computed, metadata-only scores.

    Returns:
        Counts, anomaly rate, and a WER distribution.
    """
    anomalies = sum(score > threshold for score in scores)
    return {
        "segments": len(scores),
        "anomalies": anomalies,
        "anomaly_rate": anomalies / len(scores) if scores else 0.0,
        "normalised_wer": _distribution(scores).as_dict(),
    }


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
    return {
        "audited": total,
        "by_decision": dict(sorted(by_decision.items())),
        "material_defects": material_defects,
        "clean_rate": (total - material_defects) / total if total else 0.0,
    }


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


def create_blinded_audit_manifest(
    rows: RowStream,
    *,
    accepted_quota: int = 200,
    rejected_quota: int = 100,
    borderline_quota: int = 100,
    seed: int | str = 0,
) -> list[dict[str, object]]:
    """Create a label-free manifest with independent accepted/rejected quotas.

    The quota class is used only for selection and is never written to the manifest.
    A reviewer therefore cannot infer whether an item passed segmentation.

    Returns:
        Label-free audit entries, with fewer entries when a quota is unavailable.

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
    grouped: dict[str, list[dict[str, object]]] = {status: [] for status in quotas}
    for row in rows:
        status = _status(row)
        if quotas[status]:
            grouped[status].append(_metadata_copy(row))
    selected: list[dict[str, object]] = []
    for status, quota in quotas.items():
        if quota:
            selected.extend(
                stratified_sample(
                    grouped[status], sample_size=quota, seed=f"{seed}:{status}"
                )
            )
    manifest: list[dict[str, object]] = []
    for ordinal, row in enumerate(
        sorted(selected, key=lambda item: (_identity(item), _status(item)))
    ):
        identity = _identity(row)
        audit_id = hashlib.sha256(f"{seed}\0{ordinal}\0{identity}".encode()).hexdigest()
        entry = t.cast(
            dict[str, object],
            {
                "audit_id": audit_id,
                "segment_id": identity,
                "remote_path": _string(row, "remote_path", "path", "audio_path"),
                "stratum": _stratum_key(row),
            },
        )
        manifest.append(entry)
    return manifest


def _stratum_key(row: MetadataRow) -> tuple[str, ...]:
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
        duration_class,
        _string(row, "show", "show_type", "programme_type") or "unknown",
        f"speakers-{speaker_count}",
        "music" if _boolean(row, "music", "music_dominant") else "speech",
        f"language-d{min(9, max(0, int(language_probability * 10)))}",
        f"position-{_position_decile(row)}",
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
    candidates: dict[tuple[str, ...], list[tuple[str, dict[str, object]]]] = {}
    for raw_row in rows:
        row = _metadata_copy(raw_row)
        key = _stratum_key(row)
        identity = _identity(row)
        digest = hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()
        bucket = candidates.setdefault(key, [])
        bucket.append((digest, row))
        bucket.sort(key=lambda item: (item[0], _identity(item[1])))
        if len(bucket) > sample_size:
            bucket.pop()
    if sample_size == 0:
        return []
    # Round-robin gives every available stratum a chance before filling large ones.
    selected: list[tuple[str, dict[str, object]]] = []
    ordered = sorted(candidates.items(), key=lambda item: item[0])
    depth = 0
    while len(selected) < sample_size:
        added = False
        for _, bucket in ordered:
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        depth += 1
    selected.sort(key=lambda item: (_identity(item[1]), item[0]))
    return [row for _, row in selected]


def deterministic_deciles(values: c.Iterable[float]) -> list[int]:
    """Return stable rank deciles, retaining input order in the result."""
    numbers = list(values)
    order = sorted(range(len(numbers)), key=lambda index: (numbers[index], index))
    result = [0] * len(numbers)
    for rank, index in enumerate(order):
        result[index] = min(9, rank * 10 // max(1, len(numbers)))
    return result


def review_one_clip(
    entry: MetadataRow,
    retriever: ClipRetriever,
    reviewer: c.Callable[[Path, MetadataRow], MetadataRow],
    *,
    temporary_root: Path | None = None,
) -> dict[str, object]:
    """Run one blinded review and delete its temporary audio before returning.

    Returns:
        The reviewer's metadata-only decision.

    Raises:
        TypeError:
            If the reviewer does not return a mapping.
    """
    path = retrieve_one_for_review(entry, retriever, temporary_root=temporary_root)
    try:
        decision = reviewer(path, entry)
        if not isinstance(decision, c.Mapping):
            raise TypeError("reviewer must return a mapping")
        return _metadata_copy(decision)
    finally:
        path.unlink(missing_ok=True)


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
    scores: list[float] = []
    anomalies = 0
    for row in rows:
        audio = audio_loader(row)
        hypothesis = asr.transcribe(audio)
        reference = _string(row, "text", "transcript") or ""
        score = normalised_wer(reference, hypothesis)
        scores.append(score)
        if score > anomaly_threshold:
            anomalies += 1
    distribution = _distribution(scores)
    return {
        "segments": len(scores),
        "anomalies": anomalies,
        "anomaly_rate": anomalies / len(scores) if scores else 0.0,
        "normalised_wer": distribution.as_dict(),
    }


# Descriptive aliases keep the public API discoverable for report callers.
bounded_stream_aggregates = bounded_remote_aggregates
build_audit_manifest = create_blinded_audit_manifest
find_duplicate_segment_ids = check_duplicate_and_overlaps
normalised_word_error_rate = normalised_wer

__all__ = [
    "ClipRetriever",
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
    "deterministic_deciles",
    "iter_bounded",
    "normalised_wer",
    "normalised_word_error_rate",
    "review_one_clip",
    "score_asr_anomalies",
    "stratified_sample",
    "summarise_manual_audit",
    "stream_remote_rows",
]
