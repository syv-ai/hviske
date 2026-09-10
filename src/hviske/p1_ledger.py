"""Crash-safe metadata ledger for the Phase 1 P1 segmentation pipeline.

The ledger deliberately stores identities, counters and checksums, never source
content.  Remote paths are publication-relative; local shard paths are persisted
only as durable recovery identities.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .p1_contracts import (
    BatchEvidence,
    ContractModel,
    LedgerState,
    RejectionCategory,
    ShardEvidence,
    valid_ledger_transition,
)

_SCHEMA_VERSION = 3
_SEQUENCE_TABLE = "ledger_sequences"
_METADATA_TABLE = "ledger_metadata"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_KEYS = {
    "access_token",
    "api_key",
    "audio_bytes",
    "auth",
    "credential",
    "credentials",
    "full_transcript",
    "hf_token",
    "password",
    "secret",
    "text",
    "text_corpus",
    "token",
    "transcript",
    "transcript_text",
    "waveform",
}
_FORBIDDEN_PATH_PARTS = {".cache", "cache", "scratch", "tmp"}


@dataclass(frozen=True)
class BatchRecord:
    """Metadata and durable evidence for one bounded publication batch."""

    batch_id: str
    state: LedgerState
    pipeline_digest: str
    commit_id: str | None
    attempts: int
    programme_count: int
    row_count: int
    rejection_counts: dict[str, int]
    duration_ms: int | None
    verification_time: str | None
    purge_time: str | None
    publication_artifact_purged_at: str | None
    publication_artifact_purge_evidence: dict[str, object]
    remote_checked_at: str | None
    remote_present: bool | None
    last_error: str | None


class LedgerError(RuntimeError):
    """Base class for ledger errors."""


class EvidenceError(LedgerError):
    """Raised when evidence would violate the metadata-only ledger contract."""


class InvalidTransition(LedgerError):
    """Raised when a requested state transition is not allowed."""


@dataclass(frozen=True)
class ProgrammeRecord:
    """Metadata and durable evidence for one source programme."""

    programme_id: str
    source_file_id: str
    state: LedgerState
    source_revisions: dict[str, object]
    pipeline_digest: str
    attempts: int
    accepted_count: int
    rejected_count: int
    source_duration_ms: int | None
    processed_duration_ms: int | None
    commit_id: str | None
    rejection_counts: dict[str, int]
    source_temp_purged_at: str | None
    source_temp_purge_evidence: dict[str, object]
    verification_time: str | None
    purge_time: str | None
    last_error: str | None


@dataclass(frozen=True)
class ShardRecord:
    """Metadata and checksum evidence for one local or published shard."""

    shard_id: str
    programme_id: str | None
    batch_id: str | None
    state: LedgerState
    path: str
    byte_size: int
    row_count: int
    sha256: str
    verification_time: str | None
    purge_time: str | None
    local_path: str | None = None


@dataclass(frozen=True)
class ShardAllocation:
    """Local evidence needed for one atomic batch allocation."""

    local_path: str | Path
    remote_path: str
    sha256: str
    byte_size: int
    row_count: int
    shard_id: str | None = None


class Ledger:
    """SQLite-backed, restart-safe ledger.

    Args:
        path:
            SQLite database path.  ``":memory:"`` is useful for tests.
        reset_processing:
            Reset processing rows when opening the database.  This is enabled by
            default because a new connection represents a restarted worker.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_digest: str | None = None,
        reset_processing: bool = True,
    ) -> None:
        """Open or create a durable ledger.

        Args:
            path:
                SQLite database path.
            pipeline_digest (optional):
                Expected identity of the pipeline using this ledger. An empty ledger
                is bound to this digest; a populated legacy ledger is accepted only
                when all its records already use it.
            reset_processing (optional):
                Whether to reset processing rows on open. Defaults to True.
        """
        if pipeline_digest is not None:
            self._validate_digest(pipeline_digest, "pipeline_digest")
        self.pipeline_digest: str | None = pipeline_digest
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        database = str(path)
        self._connection = sqlite3.connect(database, isolation_level=None, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA synchronous = FULL")
        try:
            with self.transaction() as connection:
                self._migrate(connection)
                self._bind_pipeline_digest(connection, pipeline_digest)
            if reset_processing:
                self.reset_abandoned_processing()
            if database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a caller-supplied group of writes as one durable transaction.

        Yields:
            The active SQLite connection.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
            self._sync_database()

    def __enter__(self) -> Ledger:
        """Return this ledger for a context-managed session."""
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object | None,
    ) -> None:
        """Close the ledger after leaving a context-managed session."""
        self.close()

    def allocate_batch_with_shards(
        self,
        programme_id: str,
        shards: Sequence[ShardAllocation | Mapping[str, object] | object],
        *,
        batch_id: str | None = None,
        batch_prefix: str = "batch",
        shard_prefix: str = "shard",
        accepted_count: int = 0,
        rejected_count: int = 0,
        processed_duration_ms: int | None = None,
        rejection_counts: Mapping[RejectionCategory | str, int] | None = None,
        audit_candidates: Sequence[Mapping[str, object]] = (),
    ) -> tuple[BatchRecord, tuple[ShardRecord, ...]]:
        """Allocate and attach a complete batch in one durable transaction.

        The caller supplies already-fsynced local shards. Their local path, remote
        path, size and digest are committed together with the generated identities;
        only then is the programme allowed to become ``sharded``. This removes the
        crash window between a programme transition and shard registration.

        Args:
            programme_id:
                Programme whose output is being published.
            shards:
                Shard allocations, mappings, or objects exposing ``path``,
                ``repo_path`` and ``row_count``. Objects may omit digest and size;
                those values are computed from the local file.
            batch_id (optional):
                Explicit id for retry or recovery. Defaults to a durable allocation.
            batch_prefix (optional):
                Prefix for generated batch IDs. Defaults to ``batch``.
            shard_prefix (optional):
                Prefix for generated shard IDs. Defaults to ``shard``.
            accepted_count (optional):
                Number of accepted rows. Defaults to 0.
            rejected_count (optional):
                Number of rejected rows. Defaults to 0.
            processed_duration_ms (optional):
                Processing duration. Defaults to None.
            rejection_counts (optional):
                Metadata-only rejection counts. Defaults to an empty mapping.
            audit_candidates (optional):
                Accepted metadata-only audit candidates with local row locators.
                They are committed with the batch and shard rows. Defaults to an
                empty sequence.

        Returns:
            The atomically created batch and its attached shards.

        Raises:
            EvidenceError:
                If local evidence is incomplete or inconsistent.
            InvalidTransition:
                If the programme cannot become sharded.
        """
        if not shards:
            raise EvidenceError("a publication batch must contain at least one shard")
        self._validate_identifier(programme_id, "programme_id")
        self._validate_identifier(batch_prefix, "batch_prefix")
        self._validate_identifier(shard_prefix, "shard_prefix")
        self._validate_nonnegative(accepted_count, "accepted_count")
        self._validate_nonnegative(rejected_count, "rejected_count")
        self._validate_nonnegative(processed_duration_ms, "processed_duration_ms")
        safe_rejections = self._rejection_counts(rejection_counts or {})
        prepared = tuple(self._prepare_allocation(item) for item in shards)
        prepared_audit = tuple(
            self._prepare_audit_candidate(candidate) for candidate in audit_candidates
        )
        remote_paths = [cast(str, item["remote_path"]) for item in prepared]
        if len(remote_paths) != len(set(remote_paths)):
            raise EvidenceError("publication paths must be unique")
        if batch_id is not None:
            self._validate_identifier(batch_id, "batch_id")

        with self.transaction() as connection:
            programme = self._require_row(
                connection, "programmes", "programme_id", programme_id
            )
            pipeline_digest = str(programme["pipeline_digest"])
            self._ensure_pipeline_digest(connection, pipeline_digest)
            if LedgerState(programme["state"]) not in {
                LedgerState.DISCOVERED,
                LedgerState.PROCESSING,
                LedgerState.SHARDED,
            }:
                raise InvalidTransition("programme cannot allocate new shards")
            resolved_batch_id = batch_id or self._next_identifier(
                connection, kind="batch", prefix=batch_prefix
            )
            if (
                connection.execute(
                    "SELECT 1 FROM batches WHERE batch_id = ?", (resolved_batch_id,)
                ).fetchone()
                is not None
            ):
                raise EvidenceError("batch identity already exists")
            now = self._now()
            connection.execute(
                """INSERT INTO batches (
                    batch_id, state, pipeline_digest, programme_count, row_count,
                    rejection_counts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    resolved_batch_id,
                    LedgerState.SHARDED.value,
                    pipeline_digest,
                    1,
                    sum(item["row_count"] for item in prepared),
                    safe_rejections,
                    now,
                    now,
                ),
            )
            records: list[ShardRecord] = []
            for item in prepared:
                resolved_shard_id = cast(
                    str,
                    item["shard_id"]
                    or self._next_identifier(
                        connection, kind="shard", prefix=shard_prefix
                    ),
                )
                self._validate_identifier(resolved_shard_id, "shard_id")
                connection.execute(
                    """INSERT INTO shards (
                        shard_id, programme_id, batch_id, state, path, byte_size,
                        row_count, sha256, local_path, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        resolved_shard_id,
                        programme_id,
                        resolved_batch_id,
                        LedgerState.SHARDED.value,
                        item["remote_path"],
                        item["byte_size"],
                        item["row_count"],
                        item["sha256"],
                        item["local_path"],
                        now,
                        now,
                    ),
                )
                records.append(
                    self._shard_record(
                        self._require_row(
                            connection, "shards", "shard_id", resolved_shard_id
                        )
                    )
                )
            for candidate in prepared_audit:
                connection.execute(
                    """INSERT INTO audit_candidates (
                        batch_id, programme_id, candidate_json, local_path,
                        local_row_locator, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        resolved_batch_id,
                        programme_id,
                        candidate[0],
                        candidate[1],
                        candidate[2],
                        now,
                    ),
                )
            programme_updates = {
                "state": LedgerState.SHARDED.value,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "processed_duration_ms": processed_duration_ms,
                "rejection_counts": safe_rejections,
                "updated_at": now,
            }
            assignments = ", ".join(f"{key} = ?" for key in programme_updates)
            connection.execute(
                f"UPDATE programmes SET {assignments} WHERE programme_id = ?",
                (*programme_updates.values(), programme_id),
            )
            batch = self._batch_record(
                self._require_row(connection, "batches", "batch_id", resolved_batch_id)
            )
        return batch, tuple(records)

    def discover_programme(
        self,
        programme_id: str,
        *,
        source_file_id: str,
        source_revisions: ContractModel | Mapping[str, object],
        pipeline_digest: str,
        source_duration_ms: int | None = None,
    ) -> ProgrammeRecord:
        """Alias for :meth:`register_programme` used by discovery workers.

        Returns:
            The resulting programme record.
        """
        return self.register_programme(
            programme_id,
            source_file_id=source_file_id,
            source_revisions=source_revisions,
            pipeline_digest=pipeline_digest,
            source_duration_ms=source_duration_ms,
        )

    def register_programme(
        self,
        programme_id: str,
        *,
        source_file_id: str,
        source_revisions: ContractModel | Mapping[str, object],
        pipeline_digest: str,
        source_duration_ms: int | None = None,
    ) -> ProgrammeRecord:
        """Register a discovered programme idempotently.

        Source revisions are serialised as contract metadata. Transcript and
        audio payloads are rejected before SQLite is touched.

        Returns:
            The resulting programme record.

        Raises:
            EvidenceError:
                If identity or metadata is invalid.
        """
        self._validate_identifier(programme_id, "programme_id")
        self._validate_identifier(source_file_id, "source_file_id")
        self._validate_digest(pipeline_digest, "pipeline_digest")
        self._validate_nonnegative(source_duration_ms, "source_duration_ms")
        revisions = self._metadata_json(source_revisions)
        now = self._now()
        with self.transaction() as connection:
            self._ensure_pipeline_digest(connection, pipeline_digest)
            existing = connection.execute(
                "SELECT * FROM programmes WHERE programme_id = ?", (programme_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO programmes (
                        programme_id, source_file_id, state, source_revisions,
                        pipeline_digest, source_duration_ms, discovered_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        programme_id,
                        source_file_id,
                        LedgerState.DISCOVERED.value,
                        revisions,
                        pipeline_digest,
                        source_duration_ms,
                        now,
                        now,
                    ),
                )
            elif (
                existing["source_file_id"] != source_file_id
                or existing["source_revisions"] != revisions
                or existing["pipeline_digest"] != pipeline_digest
            ):
                raise EvidenceError("programme identity differs from the ledger")
        return self.programme(programme_id)

    def purge_programme(
        self, programme_id: str, *, evidence: Mapping[str, object] | None = None
    ) -> ProgrammeRecord:
        """Record source-temporary deletion and finish a programme.

        Returns:
            The resulting programme record.
        """
        self.mark_source_temps_purged(programme_id, evidence=evidence)
        return self.transition_programme(programme_id, LedgerState.PURGED)

    def mark_source_temps_purged(
        self, programme_id: str, *, evidence: Mapping[str, object] | None = None
    ) -> ProgrammeRecord:
        """Record deletion of source/alignment temporary data, without paths.

        Returns:
            The resulting programme record.

        Raises:
            InvalidTransition:
                If local shards are not yet recoverable.
        """
        safe_evidence = self._metadata_json(evidence or {})
        now = self._now()
        with self.transaction() as connection:
            programme = self._require_row(
                connection, "programmes", "programme_id", programme_id
            )
            if programme["state"] not in {
                LedgerState.SHARDED.value,
                LedgerState.COMMITTED.value,
                LedgerState.VERIFIED.value,
                LedgerState.PURGED.value,
            }:
                raise InvalidTransition("source temporaries require recoverable shards")
            connection.execute(
                """UPDATE programmes SET source_temp_purged_at = ?,
                source_temp_purge_evidence = ?, updated_at = ?
                WHERE programme_id = ?""",
                (now, safe_evidence, now, programme_id),
            )
        return self.programme(programme_id)

    def transition_programme(
        self,
        programme_id: str,
        target: LedgerState,
        *,
        evidence: Mapping[str, object] | None = None,
        **fields: object,
    ) -> ProgrammeRecord:
        """Atomically transition a programme and persist its evidence.

        Returns:
            The resulting programme record.
        """
        combined = self._fields_from_evidence(evidence, fields)
        return cast(
            ProgrammeRecord,
            self._transition(
                table="programmes",
                identifier_column="programme_id",
                identifier=programme_id,
                target=target,
                evidence=evidence,
                fields=combined,
            ),
        )

    def register_shard(
        self,
        shard_id: str,
        *,
        path: str,
        sha256: str,
        byte_size: int,
        row_count: int,
        programme_id: str | None = None,
        batch_id: str | None = None,
        local_path: str | Path | None = None,
        state: LedgerState = LedgerState.SHARDED,
    ) -> ShardRecord:
        """Register a complete shard using only publication-relative metadata.

        Returns:
            The resulting shard record.

        Raises:
            EvidenceError:
                If the path, digest, or evidence is unsafe.
        """
        self._validate_identifier(shard_id, "shard_id")
        safe_path = self._publication_path(path)
        self._validate_digest(sha256, "sha256")
        self._validate_nonnegative(byte_size, "byte_size")
        self._validate_nonnegative(row_count, "row_count")
        durable_local_path = self._local_path(local_path)
        if state not in {LedgerState.DISCOVERED, LedgerState.SHARDED}:
            raise EvidenceError("new shards must be discovered or sharded")
        with self.transaction() as connection:
            if programme_id is not None:
                self._require_row(
                    connection, "programmes", "programme_id", programme_id
                )
            if batch_id is not None:
                self._require_row(connection, "batches", "batch_id", batch_id)
            collision = connection.execute(
                """SELECT shard_id FROM shards WHERE path = ? AND shard_id != ?""",
                (safe_path, shard_id),
            ).fetchone()
            if collision is not None:
                raise EvidenceError(
                    f"publication path already belongs to shard {collision[0]!r}"
                )
            existing = connection.execute(
                "SELECT * FROM shards WHERE shard_id = ?", (shard_id,)
            ).fetchone()
            values = (
                programme_id,
                batch_id,
                state.value,
                safe_path,
                byte_size,
                row_count,
                sha256,
                durable_local_path,
            )
            if existing is None:
                connection.execute(
                    """INSERT INTO shards (
                        shard_id, programme_id, batch_id, state, path, byte_size,
                        row_count, sha256, local_path, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (shard_id, *values, self._now(), self._now()),
                )
            elif (
                tuple(
                    existing[key]
                    for key in (
                        "path",
                        "byte_size",
                        "row_count",
                        "sha256",
                        "local_path",
                    )
                )
                != values[3:]
            ):
                if existing["state"] in {
                    LedgerState.VERIFIED.value,
                    LedgerState.PURGED.value,
                }:
                    raise EvidenceError("a verified shard path is immutable")
                raise EvidenceError("shard evidence differs from the ledger")
        return self.shard(shard_id)

    prepare_batch = allocate_batch_with_shards
    allocate_publication_batch = allocate_batch_with_shards
    register_batch_with_shards = allocate_batch_with_shards
    allocate_and_attach_shards = allocate_batch_with_shards

    def reject_programme(
        self,
        programme_id: str,
        *,
        reason: RejectionCategory | str,
        accepted_count: int = 0,
        rejected_count: int = 1,
        rejection_counts: Mapping[RejectionCategory | str, int] | None = None,
    ) -> ProgrammeRecord:
        """Permanently reject a programme with a counted reason.

        Returns:
            The resulting programme record.
        """
        category = self._rejection_category(reason)
        counts = (
            {category: rejected_count} if rejection_counts is None else rejection_counts
        )
        return self.transition_programme(
            programme_id,
            LedgerState.REJECTED,
            rejection_counts=counts,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            last_error=category,
        )

    @property
    def schema_version(self) -> int:
        """Expose the current on-disk schema version."""
        return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    def start_processing(self, programme_id: str) -> ProgrammeRecord:
        """Move a programme to processing and increment its attempt count.

        Returns:
            The resulting programme record.
        """
        return self.transition_programme(programme_id, LedgerState.PROCESSING)

    record_shard = register_shard
    create_shard = register_shard

    def allocate_batch_id(self, prefix: str = "batch") -> str:
        """Return a globally monotonic, collision-free batch identifier."""
        self._validate_identifier(prefix, "prefix")
        return f"{prefix}-{self.allocate_batch_sequence():08d}"

    def allocate_batch_sequence(self) -> int:
        """Allocate the next durable batch sequence number.

        Allocation is performed under the ledger write transaction, so closing and
        reopening a ledger cannot reuse a number and concurrent assemblers cannot
        receive the same number.

        Returns:
            The newly allocated positive sequence number.
        """
        return self._allocate_sequence("batch")

    def allocate_shard_id(self, prefix: str = "shard") -> str:
        """Return a globally monotonic, collision-free shard identifier."""
        self._validate_identifier(prefix, "prefix")
        return f"{prefix}-{self.allocate_shard_sequence():08d}"

    def allocate_shard_sequence(self) -> int:
        """Allocate the next durable shard sequence number.

        Returns:
            The newly allocated positive sequence number.
        """
        return self._allocate_sequence("shard")

    # These names are intentionally small aliases for assembler integrations.
    next_batch_sequence = allocate_batch_sequence
    next_shard_sequence = allocate_shard_sequence
    allocate_batch_number = allocate_batch_sequence
    allocate_shard_number = allocate_shard_sequence
    new_batch_id = allocate_batch_id
    new_shard_id = allocate_shard_id
    next_batch_id = allocate_batch_id
    next_shard_id = allocate_shard_id

    def register_batch(
        self, batch_id: str, *, pipeline_digest: str, duration_ms: int | None = None
    ) -> BatchRecord:
        """Register an empty bounded publication batch idempotently.

        Returns:
            The resulting batch record.

        Raises:
            EvidenceError:
                If the batch identity is invalid.
        """
        self._validate_identifier(batch_id, "batch_id")
        self._validate_digest(pipeline_digest, "pipeline_digest")
        self._validate_nonnegative(duration_ms, "duration_ms")
        now = self._now()
        with self.transaction() as connection:
            self._ensure_pipeline_digest(connection, pipeline_digest)
            existing = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """INSERT INTO batches (
                        batch_id, state, pipeline_digest, duration_ms, created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id,
                        LedgerState.DISCOVERED.value,
                        pipeline_digest,
                        duration_ms,
                        now,
                        now,
                    ),
                )
            elif existing["pipeline_digest"] != pipeline_digest:
                raise EvidenceError("batch pipeline digest differs from the ledger")
        return self.batch(batch_id)

    def transition_shard(
        self,
        shard_id: str,
        target: LedgerState,
        *,
        evidence: Mapping[str, object] | None = None,
        **fields: object,
    ) -> ShardRecord:
        """Atomically transition a shard.

        Returns:
            The resulting shard record.
        """
        combined = self._fields_from_evidence(evidence, fields)
        return cast(
            ShardRecord,
            self._transition(
                table="shards",
                identifier_column="shard_id",
                identifier=shard_id,
                target=target,
                evidence=evidence,
                fields=combined,
            ),
        )

    create_batch = register_batch

    def attach_shard(self, batch_id: str, shard_id: str) -> BatchRecord:
        """Attach a registered shard and refresh batch counts atomically.

        Returns:
            The resulting batch record.

        Raises:
            EvidenceError:
                If the shard belongs to another pipeline or batch.
            InvalidTransition:
                If the batch has already been committed.
        """
        with self.transaction() as connection:
            batch = self._require_row(connection, "batches", "batch_id", batch_id)
            shard = self._require_row(connection, "shards", "shard_id", shard_id)
            if shard["programme_id"] is not None:
                programme = self._require_row(
                    connection, "programmes", "programme_id", shard["programme_id"]
                )
                if programme["pipeline_digest"] != batch["pipeline_digest"]:
                    raise EvidenceError(
                        "shard programme uses a different pipeline digest"
                    )
            if batch["state"] not in {
                LedgerState.DISCOVERED.value,
                LedgerState.PROCESSING.value,
                LedgerState.SHARDED.value,
            }:
                raise InvalidTransition("shards cannot be attached after commit")
            if shard["batch_id"] not in (None, batch_id):
                raise EvidenceError("shard already belongs to another batch")
            connection.execute(
                "UPDATE shards SET batch_id = ?, updated_at = ? WHERE shard_id = ?",
                (batch_id, self._now(), shard_id),
            )
            self._refresh_batch_counts(connection, batch_id)
        return self.batch(batch_id)

    def audit_candidates(self, batch_id: str) -> tuple[dict[str, object], ...]:
        """Return metadata-only audit candidates durably attached to a batch.

        Args:
            batch_id:
                Batch whose accepted candidate locators should be returned.

        Returns:
            Candidates in their deterministic allocation order.

        Raises:
            LedgerError:
                If persisted candidate metadata is not an object.
        """
        rows = self._connection.execute(
            "SELECT candidate_json, local_path, local_row_locator "
            "FROM audit_candidates WHERE batch_id = ? ORDER BY candidate_id",
            (batch_id,),
        ).fetchall()
        candidates: list[dict[str, object]] = []
        for row in rows:
            value = json.loads(str(row[0]))
            if not isinstance(value, dict):
                raise LedgerError("audit candidate metadata is not an object")
            candidate = cast(dict[str, object], value)
            candidate["local_path"] = str(row[1])
            candidate["local_row_locator"] = int(row[2])
            candidates.append(candidate)
        return tuple(candidates)

    def finalise_batch_children(self, batch_id: str) -> BatchRecord:
        """Idempotently finish every shard and programme in a purged batch.

        The physical purge is necessarily outside SQLite.  Keeping this recovery
        step transactional means a crash between unlinking files and child-state
        updates can be repaired without re-uploading the batch.

        Returns:
            The purged batch record.

        Raises:
            KeyError:
                If an attached programme is absent.
            InvalidTransition:
                If the batch or a child has not reached a finalisable state.
            EvidenceError:
                If the batch has no commit or a child has conflicting evidence.
        """
        batch = self.batch(batch_id)
        if batch.state is not LedgerState.PURGED:
            raise InvalidTransition("child finalisation requires a purged batch")
        if batch.commit_id is None:
            raise EvidenceError("purged batches require a commit SHA")
        now = self._now()
        with self.transaction() as connection:
            shard_rows = connection.execute(
                "SELECT shard_id, programme_id, state "
                "FROM shards WHERE batch_id = ? ORDER BY shard_id",
                (batch_id,),
            ).fetchall()
            programme_ids = sorted(
                {row["programme_id"] for row in shard_rows if row["programme_id"]}
            )
            for row in shard_rows:
                if row["state"] not in {
                    LedgerState.SHARDED.value,
                    LedgerState.COMMITTED.value,
                    LedgerState.VERIFIED.value,
                    LedgerState.PURGED.value,
                }:
                    raise InvalidTransition(
                        f"cannot finalise shard in state {row['state']}"
                    )
                connection.execute(
                    """UPDATE shards SET state = ?,
                    verification_time = COALESCE(verification_time, ?),
                    purge_time = COALESCE(purge_time, ?), updated_at = ?
                    WHERE shard_id = ?""",
                    (LedgerState.PURGED.value, now, now, now, row["shard_id"]),
                )
            for programme_id in programme_ids:
                row = connection.execute(
                    "SELECT state, commit_id FROM programmes WHERE programme_id = ?",
                    (programme_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown programme {programme_id!r}")
                if row["commit_id"] is not None and row["commit_id"] != batch.commit_id:
                    raise EvidenceError("child commit SHA differs from batch")
                if row["state"] not in {
                    LedgerState.SHARDED.value,
                    LedgerState.COMMITTED.value,
                    LedgerState.VERIFIED.value,
                    LedgerState.PURGED.value,
                }:
                    raise InvalidTransition(
                        f"cannot finalise programme in state {row['state']}"
                    )
                connection.execute(
                    """UPDATE programmes SET state = ?, commit_id = ?,
                    verification_time = COALESCE(verification_time, ?),
                    purge_time = COALESCE(purge_time, ?), updated_at = ?
                    WHERE programme_id = ?""",
                    (
                        LedgerState.PURGED.value,
                        batch.commit_id,
                        now,
                        now,
                        now,
                        programme_id,
                    ),
                )
        return self.batch(batch_id)

    def mark_batch_verified(
        self,
        batch_id: str,
        *,
        commit_id: str | None = None,
        evidence: Mapping[str, object] | None = None,
        shards: tuple[ShardEvidence, ...] | None = None,
    ) -> BatchRecord:
        """Durably record complete remote verification before local deletion.

        Returns:
            The verified batch record.

        Raises:
            EvidenceError:
                If commit or shard evidence differs from the ledger.
        """
        record = self.batch(batch_id)
        expected = commit_id or record.commit_id
        if expected is None:
            raise EvidenceError("verified batches require a commit SHA")
        if record.commit_id != expected:
            raise EvidenceError("verification commit SHA differs from the ledger")
        if shards is not None:
            durable = tuple(
                ShardEvidence(
                    path=item.path,
                    byte_size=item.byte_size,
                    row_count=item.row_count,
                    sha256=item.sha256,
                )
                for item in self.shards(batch_id)
            )
            if tuple(shards) != durable:
                raise EvidenceError("verification shard evidence differs from ledger")
        if record.state is LedgerState.VERIFIED or record.state is LedgerState.PURGED:
            return record
        return self.transition_batch(
            batch_id, LedgerState.VERIFIED, commit_id=expected, evidence=evidence
        )

    def purge_batch(
        self, batch_id: str, *, evidence: Mapping[str, object] | None = None
    ) -> BatchRecord:
        """Record verified publication deletion and finish a batch.

        Returns:
            The resulting batch record.
        """
        self.mark_publication_artifacts_purged(batch_id, evidence=evidence)
        return self.transition_batch(batch_id, LedgerState.PURGED)

    def mark_publication_artifacts_purged(
        self, batch_id: str, *, evidence: Mapping[str, object] | None = None
    ) -> BatchRecord:
        """Record deletion of local publication artefacts separately from sources.

        Returns:
            The resulting batch record.

        Raises:
            InvalidTransition:
                If remote verification has not completed.
        """
        now = self._now()
        with self.transaction() as connection:
            batch = self._require_row(connection, "batches", "batch_id", batch_id)
            if batch["state"] != LedgerState.VERIFIED.value:
                raise InvalidTransition(
                    "publication artefacts require remote verification"
                )
            safe_evidence = (
                batch["publication_artifact_purge_evidence"]
                if evidence is None
                else self._metadata_json(evidence)
            )
            connection.execute(
                """UPDATE batches SET publication_artifact_purged_at = ?,
                publication_artifact_purge_evidence = ?, updated_at = ?
                WHERE batch_id = ?""",
                (now, safe_evidence, now, batch_id),
            )
        return self.batch(batch_id)

    def record_commit(self, batch_id: str, commit_id: str) -> BatchRecord:
        """Durably record a Hub commit immediately after it is created.

        This is deliberately separate from verification: a restart must be able to
        resume verification of a commit that exists remotely but was not verified.

        Returns:
            The committed batch record.

        Raises:
            EvidenceError:
                If the commit differs from an already recorded commit.
        """
        self._validate_commit(commit_id)
        record = self.batch(batch_id)
        if record.state in {
            LedgerState.COMMITTED,
            LedgerState.VERIFIED,
            LedgerState.PURGED,
        }:
            if record.commit_id != commit_id:
                raise EvidenceError("a batch commit SHA is immutable")
            return record
        return self.transition_batch(
            batch_id, LedgerState.COMMITTED, commit_id=commit_id
        )

    record_verification = mark_batch_verified
    mark_verified = mark_batch_verified

    def reconcile_committed_batch(
        self, batch_id: str, remote_commit_exists: object
    ) -> bool:
        """Ask a remote store whether a committed batch already exists.

        The hook receives ``(commit_id, publication_paths)`` and must return a
        boolean. A present commit is left committed for the normal verification
        step. A missing commit is recorded as an unresolved incident; it never
        makes already-sharded or committed work retryable.

        Returns:
            Whether the immutable remote commit exists.

        Raises:
            InvalidTransition:
                If the batch is not committed.
            TypeError:
                If the hook is not callable.
        """
        if not callable(remote_commit_exists):
            raise TypeError("remote_commit_exists must be callable")
        record = self.batch(batch_id)
        if record.state != LedgerState.COMMITTED or record.commit_id is None:
            raise InvalidTransition("only committed batches have remote reconciliation")
        paths = tuple(shard.path for shard in self.shards(batch_id))
        hook = cast(Callable[[str, tuple[str, ...]], bool], remote_commit_exists)
        present = bool(hook(record.commit_id, paths))
        now = self._now()
        with self.transaction() as connection:
            connection.execute(
                """UPDATE batches SET remote_checked_at = ?, remote_present = ?,
                last_error = ?, updated_at = ? WHERE batch_id = ?""",
                (
                    now,
                    int(present),
                    None if present else "remote commit was not found",
                    now,
                    batch_id,
                ),
            )
        return present

    def transition_batch(
        self,
        batch_id: str,
        target: LedgerState,
        *,
        evidence: Mapping[str, object] | None = None,
        commit_id: str | None = None,
        **fields: object,
    ) -> BatchRecord:
        """Atomically transition a batch and persist publication evidence.

        Returns:
            The resulting batch record.
        """
        combined = self._fields_from_evidence(evidence, fields)
        if commit_id is not None:
            self._validate_commit(commit_id)
            combined["commit_id"] = commit_id
        return cast(
            BatchRecord,
            self._transition(
                table="batches",
                identifier_column="batch_id",
                identifier=batch_id,
                target=target,
                evidence=evidence,
                fields=combined,
            ),
        )

    reconcile_remote = reconcile_committed_batch

    def reconcile_local_shard(
        self, shard_id: str, local_path: str | Path | None = None
    ) -> bool:
        """Return whether the durable local identity still matches its digest.

        When a shard has a persisted local path, a candidate at another path is not
        interchangeable merely because its basename happens to match.  Older ledger
        rows without that column remain explicitly addressable by their caller.

        Returns:
            True only for the persisted regular file with matching size and digest.
        """
        shard = self.shard(shard_id)
        if local_path is None:
            if shard.local_path is None:
                return False
            local_path = shard.local_path
        candidate = Path(local_path)
        if shard.local_path is not None:
            try:
                if candidate.resolve() != Path(shard.local_path).resolve():
                    return False
            except OSError:
                return False
        if not candidate.is_file() or candidate.is_symlink():
            return False
        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        return size == shard.byte_size and digest.hexdigest() == shard.sha256

    def unattached_local_shards(self) -> tuple[ShardRecord, ...]:
        """Return durable local shards not yet attached to a publication batch.

        These rows are safe recovery candidates because both their local identity
        and content digest were committed together in SQLite.
        """
        rows = self._connection.execute(
            """SELECT * FROM shards WHERE batch_id IS NULL AND local_path IS NOT NULL
            AND state IN (?, ?, ?) ORDER BY shard_id""",
            (
                LedgerState.DISCOVERED.value,
                LedgerState.SHARDED.value,
                LedgerState.RETRYABLE.value,
            ),
        ).fetchall()
        return tuple(self._shard_record(row) for row in rows)

    unattached_shards = unattached_local_shards

    def recoverable_committed_batches(self) -> tuple[BatchRecord, ...]:
        """Return batches whose remote publication may need restart recovery."""
        rows = self._connection.execute(
            """SELECT * FROM batches WHERE state IN (?, ?) ORDER BY batch_id""",
            (LedgerState.COMMITTED.value, LedgerState.VERIFIED.value),
        ).fetchall()
        return tuple(self._batch_record(row) for row in rows)

    committed_for_recovery = recoverable_committed_batches

    def recovery_work(self) -> tuple[tuple[ShardRecord, ...], tuple[BatchRecord, ...]]:
        """Enumerate unattached local shards and committed batches for restart.

        Returns:
            Unattached local shard records followed by committed batch records.
            Both collections contain only durable ledger identities.
        """
        return self.unattached_local_shards(), self.recoverable_committed_batches()

    enumerate_recovery = recovery_work

    local_shard_matches = reconcile_local_shard

    @staticmethod
    def _prepare_allocation(item: object) -> dict[str, object]:
        def value(*names: str) -> object:
            if isinstance(item, Mapping):
                for name in names:
                    if name in item:
                        return item[name]
            else:
                for name in names:
                    candidate = getattr(item, name, None)
                    if candidate is not None:
                        return candidate
            return None

        local = value("local_path", "path")
        remote = value("remote_path", "repo_path")
        row_count = value("row_count")
        if local is None or remote is None or row_count is None:
            raise EvidenceError(
                "shard allocation needs local and remote paths and rows"
            )
        local_path = Ledger._local_path(cast(str | Path, local))
        if local_path is None:
            raise EvidenceError("shard allocation needs a local path")
        candidate = Path(local_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise EvidenceError("local shard must be a regular non-symlink file")
        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        supplied_digest = value("sha256")
        if supplied_digest is not None and supplied_digest != digest.hexdigest():
            raise EvidenceError("local shard digest differs from its bytes")
        supplied_size = value("byte_size", "size")
        if supplied_size is not None and supplied_size != size:
            raise EvidenceError("local shard size differs from its bytes")
        Ledger._validate_nonnegative(row_count, "row_count")
        safe_remote = Ledger._publication_path(cast(str, remote))
        return {
            "local_path": local_path,
            "remote_path": safe_remote,
            "sha256": digest.hexdigest(),
            "byte_size": size,
            "row_count": cast(int, row_count),
            "shard_id": value("shard_id"),
        }

    @staticmethod
    def _prepare_audit_candidate(
        candidate: Mapping[str, object],
    ) -> tuple[str, str, int]:
        if not isinstance(candidate, Mapping):
            raise EvidenceError("audit candidates must be metadata mappings")
        if candidate.get("status") != "accepted":
            raise EvidenceError("ledger audit candidates must be accepted")
        segment_id = candidate.get("segment_id")
        local_path = candidate.get("local_path")
        row_locator = candidate.get("local_row_locator")
        if not isinstance(segment_id, str) or not segment_id:
            raise EvidenceError("audit candidates need a segment identity")
        if not isinstance(local_path, (str, Path)) or not str(local_path):
            raise EvidenceError("accepted audit candidates need a local path")
        if not isinstance(row_locator, int) or isinstance(row_locator, bool):
            raise EvidenceError("accepted audit candidates need a local row locator")
        if row_locator < 0:
            raise EvidenceError("accepted audit row locators must not be negative")
        resolved_path = str(Path(local_path).expanduser().resolve(strict=False))
        metadata = dict(candidate)
        metadata.pop("local_path")
        metadata.pop("local_row_locator")
        return Ledger._metadata_json(metadata), resolved_path, row_locator

    def allocate_sequence(self, kind: str) -> int:
        """Allocate a durable sequence for ``batch`` or ``shard`` work.

        Returns:
            The newly allocated positive sequence number.
        """
        return self._allocate_sequence(kind)

    def _allocate_sequence(self, kind: str) -> int:
        if kind not in {"batch", "shard"}:
            raise ValueError(f"unknown sequence kind: {kind}")
        with self.transaction() as connection:
            value = self._allocate_sequence_in_transaction(connection, kind)
        return value

    def _allocate_sequence_in_transaction(
        self, connection: sqlite3.Connection, kind: str
    ) -> int:
        identifier = self._next_identifier(
            connection, kind=kind, prefix="batch" if kind == "batch" else "shard"
        )
        return int(identifier.rsplit("-", 1)[1])

    @staticmethod
    def _next_identifier(
        connection: sqlite3.Connection, *, kind: str, prefix: str
    ) -> str:
        if kind not in {"batch", "shard"}:
            raise ValueError(f"unknown sequence kind: {kind}")
        row = connection.execute(
            f"SELECT next_value FROM {_SEQUENCE_TABLE} WHERE kind = ?", (kind,)
        ).fetchone()
        value = 1 if row is None else int(row[0])
        table = "batches" if kind == "batch" else "shards"
        identifier = "batch_id" if kind == "batch" else "shard_id"
        suffixes = [
            int(match.group(1))
            for item in connection.execute(f"SELECT {identifier} FROM {table}")
            if (match := re.search(r"-(\d+)$", str(item[0]))) is not None
        ]
        value = max(value, max(suffixes, default=0) + 1)
        if row is None:
            connection.execute(
                f"INSERT INTO {_SEQUENCE_TABLE} (kind, next_value) VALUES (?, ?)",
                (kind, value + 1),
            )
        else:
            connection.execute(
                f"UPDATE {_SEQUENCE_TABLE} SET next_value = ? WHERE kind = ?",
                (value + 1, kind),
            )
        return f"{prefix}-{value:08d}"

    def committed_batches(self) -> tuple[BatchRecord, ...]:
        """Return committed batches, including ones awaiting verification."""
        rows = self._connection.execute(
            "SELECT * FROM batches WHERE state = ? ORDER BY batch_id",
            (LedgerState.COMMITTED.value,),
        ).fetchall()
        return tuple(self._batch_record(row) for row in rows)

    def pending_shards(self, batch_id: str | None = None) -> tuple[ShardRecord, ...]:
        """Return local shard records that have not completed publication."""
        if batch_id is None:
            rows = self._connection.execute(
                """SELECT * FROM shards WHERE state IN (?, ?, ?, ?, ?, ?)
                ORDER BY shard_id""",
                (
                    LedgerState.DISCOVERED.value,
                    LedgerState.PROCESSING.value,
                    LedgerState.SHARDED.value,
                    LedgerState.RETRYABLE.value,
                    LedgerState.COMMITTED.value,
                    LedgerState.VERIFIED.value,
                ),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """SELECT * FROM shards WHERE batch_id = ?
                AND state IN (?, ?, ?, ?, ?, ?) ORDER BY shard_id""",
                (
                    batch_id,
                    LedgerState.DISCOVERED.value,
                    LedgerState.PROCESSING.value,
                    LedgerState.SHARDED.value,
                    LedgerState.RETRYABLE.value,
                    LedgerState.COMMITTED.value,
                    LedgerState.VERIFIED.value,
                ),
            ).fetchall()
        return tuple(self._shard_record(row) for row in rows)

    def programme(self, programme_id: str) -> ProgrammeRecord:
        """Return one programme record."""
        row = self._fetch("programmes", "programme_id", programme_id)
        return self._programme_record(row)

    def purged_batches_with_pending_children(self) -> tuple[BatchRecord, ...]:
        """Return purged batches whose child transitions were interrupted."""
        rows = self._connection.execute(
            """SELECT DISTINCT b.* FROM batches AS b
            JOIN shards AS s ON s.batch_id = b.batch_id
            LEFT JOIN programmes AS p ON p.programme_id = s.programme_id
            WHERE b.state = ? AND (s.state != ? OR p.state != ?)
            ORDER BY b.batch_id""",
            (
                LedgerState.PURGED.value,
                LedgerState.PURGED.value,
                LedgerState.PURGED.value,
            ),
        ).fetchall()
        return tuple(self._batch_record(row) for row in rows)

    def reconstruct_work(
        self,
    ) -> tuple[tuple[BatchRecord, tuple[ShardRecord, ...]], ...]:
        """Reconstruct pending and committed work for a restarted assembler.

        Returns:
            Batches and their deterministically ordered shard records.
        """
        return tuple(
            (batch, self.shards(batch.batch_id)) for batch in self.pending_batches()
        )

    def pending_batches(self) -> tuple[BatchRecord, ...]:
        """Return all batches that may need assembly or recovery after restart."""
        states = (
            LedgerState.DISCOVERED.value,
            LedgerState.PROCESSING.value,
            LedgerState.SHARDED.value,
            LedgerState.RETRYABLE.value,
            LedgerState.COMMITTED.value,
            LedgerState.VERIFIED.value,
        )
        rows = self._connection.execute(
            "SELECT * FROM batches WHERE state IN (?, ?, ?, ?, ?, ?) ORDER BY batch_id",
            states,
        ).fetchall()
        return tuple(self._batch_record(row) for row in rows)

    def reset_abandoned_processing(self) -> int:
        """Reset all processing rows, as a newly opened ledger is a restart.

        Returns:
            The number of rows reset.
        """
        now = self._now()
        with self.transaction() as connection:
            total = 0
            for table in ("programmes", "batches"):
                cursor = connection.execute(
                    f"""UPDATE {table} SET state = ?, last_error = ?,
                    processing_started_at = NULL, updated_at = ?
                    WHERE state = ?""",
                    (
                        LedgerState.RETRYABLE.value,
                        "abandoned processing reset",
                        now,
                        LedgerState.PROCESSING.value,
                    ),
                )
                total += cursor.rowcount
            cursor = connection.execute(
                """UPDATE shards SET state = ?, last_error = ?, updated_at = ?
                WHERE state = ?""",
                (
                    LedgerState.RETRYABLE.value,
                    "abandoned processing reset",
                    now,
                    LedgerState.PROCESSING.value,
                ),
            )
            total += cursor.rowcount
        return total

    get_programme = programme

    def shard(self, shard_id: str) -> ShardRecord:
        """Return one shard record."""
        row = self._fetch("shards", "shard_id", shard_id)
        return self._shard_record(row)

    get_shard = shard

    def batch(self, batch_id: str) -> BatchRecord:
        """Return one batch record."""
        row = self._fetch("batches", "batch_id", batch_id)
        return self._batch_record(row)

    get_batch = batch

    def _bind_pipeline_digest(
        self, connection: sqlite3.Connection, expected: str | None
    ) -> None:
        """Bind this ledger to one pipeline identity before workers can mutate it.

        Raises:
            EvidenceError:
                If stored records contain conflicting or unexpected digests.
        """
        metadata = connection.execute(
            f"SELECT value FROM {_METADATA_TABLE} WHERE key = 'pipeline_digest'"
        ).fetchone()
        bound = None if metadata is None else str(metadata[0])
        if bound is not None:
            self._validate_digest(bound, "pipeline_digest")
        stored = {
            str(row[0])
            for row in connection.execute(
                """SELECT pipeline_digest FROM programmes
                UNION SELECT pipeline_digest FROM batches"""
            )
        }
        if len(stored) > 1:
            raise EvidenceError("ledger contains multiple pipeline digests")
        stored_digest = next(iter(stored), None)
        if bound is not None and stored_digest not in {None, bound}:
            raise EvidenceError("ledger metadata disagrees with stored digests")
        if expected is not None and stored_digest not in {None, expected}:
            raise EvidenceError("ledger records use a different pipeline digest")
        if expected is not None and bound not in {None, expected}:
            raise EvidenceError("ledger is bound to a different pipeline digest")
        target = expected or bound or stored_digest
        if target is not None and bound is None:
            connection.execute(
                f"INSERT INTO {_METADATA_TABLE} (key, value) VALUES (?, ?)",
                ("pipeline_digest", target),
            )
        self.pipeline_digest = target

    @staticmethod
    def _validate_digest(value: object, name: str) -> None:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise EvidenceError(f"{name} must be lowercase SHA-256 hex")

    def _ensure_pipeline_digest(
        self, connection: sqlite3.Connection, pipeline_digest: str
    ) -> None:
        """Ensure an insert uses the ledger's singleton pipeline identity.

        Raises:
            EvidenceError:
                If the digest differs from the identity already bound to the ledger.
        """
        self._validate_digest(pipeline_digest, "pipeline_digest")
        row = connection.execute(
            f"SELECT value FROM {_METADATA_TABLE} WHERE key = 'pipeline_digest'"
        ).fetchone()
        bound = None if row is None else str(row[0])
        if bound is not None and bound != pipeline_digest:
            raise EvidenceError("ledger is bound to a different pipeline digest")
        if bound is None:
            stored = {
                str(item[0])
                for item in connection.execute(
                    """SELECT pipeline_digest FROM programmes
                    UNION SELECT pipeline_digest FROM batches"""
                )
            }
            if stored and stored != {pipeline_digest}:
                raise EvidenceError("ledger records use a different pipeline digest")
            connection.execute(
                f"INSERT INTO {_METADATA_TABLE} (key, value) VALUES (?, ?)",
                ("pipeline_digest", pipeline_digest),
            )
        self.pipeline_digest = pipeline_digest

    @staticmethod
    def _fields_from_evidence(
        evidence: Mapping[str, object] | None, fields: Mapping[str, object]
    ) -> dict[str, object]:
        combined = dict(fields)
        if evidence is not None:
            for key in (
                "accepted_count",
                "rejected_count",
                "source_duration_ms",
                "processed_duration_ms",
                "duration_ms",
                "rejection_counts",
                "commit_id",
                "last_error",
            ):
                if key in evidence and key not in combined:
                    combined[key] = evidence[key]
        return combined

    @staticmethod
    def _local_path(value: str | Path | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        if not str(candidate) or "\x00" in str(candidate):
            raise EvidenceError("local shard paths must be non-empty paths")
        try:
            return str(candidate.expanduser().resolve(strict=False))
        except OSError as error:
            raise EvidenceError("local shard path cannot be resolved") from error

    def _migrate(self, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise LedgerError("ledger schema is newer than this package")
        connection.execute(
            f"""CREATE TABLE IF NOT EXISTS {_SEQUENCE_TABLE} (
                kind TEXT PRIMARY KEY,
                next_value INTEGER NOT NULL CHECK (next_value > 0)
            )"""
        )
        if version < 1:
            connection.execute(
                """CREATE TABLE programmes (
                    programme_id TEXT PRIMARY KEY,
                    source_file_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    source_revisions TEXT NOT NULL,
                    pipeline_digest TEXT NOT NULL,
                    commit_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    accepted_count INTEGER NOT NULL DEFAULT 0,
                    rejected_count INTEGER NOT NULL DEFAULT 0,
                    source_duration_ms INTEGER,
                    processed_duration_ms INTEGER,
                    rejection_counts TEXT NOT NULL DEFAULT '{}',
                    processing_started_at TEXT,
                    discovered_at TEXT NOT NULL,
                    verification_time TEXT,
                    purge_time TEXT,
                    source_temp_purged_at TEXT,
                    source_temp_purge_evidence TEXT NOT NULL DEFAULT '{}',
                    last_evidence TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE batches (
                    batch_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    pipeline_digest TEXT NOT NULL,
                    commit_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    programme_count INTEGER NOT NULL DEFAULT 0,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    rejection_counts TEXT NOT NULL DEFAULT '{}',
                    duration_ms INTEGER,
                    processing_started_at TEXT,
                    verification_time TEXT,
                    purge_time TEXT,
                    publication_artifact_purged_at TEXT,
                    publication_artifact_purge_evidence TEXT NOT NULL DEFAULT '{}',
                    remote_checked_at TEXT,
                    remote_present INTEGER,
                    last_evidence TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE shards (
                    shard_id TEXT PRIMARY KEY,
                    programme_id TEXT REFERENCES programmes(programme_id),
                    batch_id TEXT REFERENCES batches(batch_id),
                    state TEXT NOT NULL,
                    path TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    row_count INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    local_path TEXT,
                    verification_time TEXT,
                    purge_time TEXT,
                    last_evidence TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX shards_batch ON shards(batch_id, shard_id)"
            )
            connection.execute("CREATE INDEX programmes_state ON programmes(state)")
            connection.execute("CREATE INDEX batches_state ON batches(state)")
            connection.execute("PRAGMA user_version = 1")
        if version < 2:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(shards)")
            }
            if "local_path" not in columns:
                connection.execute("ALTER TABLE shards ADD COLUMN local_path TEXT")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS audit_candidates (
                    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
                    programme_id TEXT NOT NULL REFERENCES programmes(programme_id),
                    candidate_json TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    local_row_locator INTEGER NOT NULL
                        CHECK (local_row_locator >= 0),
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS audit_candidates_batch "
                "ON audit_candidates(batch_id, candidate_id)"
            )
            connection.execute("PRAGMA user_version = 2")
        if version < 3:
            connection.execute(
                f"""CREATE TABLE IF NOT EXISTS {_METADATA_TABLE} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )"""
            )
            connection.execute("PRAGMA user_version = 3")

    @staticmethod
    def _publication_path(value: str) -> str:
        if not isinstance(value, str) or not value or "\\" in value:
            raise EvidenceError("shard paths must be non-empty POSIX relative paths")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise EvidenceError("shard paths must be publication-relative")
        if any(part.lower() in _FORBIDDEN_PATH_PARTS for part in path.parts):
            raise EvidenceError("machine or cache paths are not allowed")
        return path.as_posix()

    def _refresh_batch_counts(
        self, connection: sqlite3.Connection, batch_id: str
    ) -> None:
        row = connection.execute(
            """SELECT COUNT(DISTINCT programme_id), COALESCE(SUM(row_count), 0)
            FROM shards WHERE batch_id = ?""",
            (batch_id,),
        ).fetchone()
        connection.execute(
            """UPDATE batches SET programme_count = ?, row_count = ?, updated_at = ?
            WHERE batch_id = ?""",
            (row[0], row[1], self._now(), batch_id),
        )

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")

    def _sync_database(self) -> None:
        if self.path == Path(":memory:"):
            return
        for filename in (self.path, Path(f"{self.path}-wal")):
            try:
                descriptor = os.open(filename, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _transition(
        self,
        *,
        table: str,
        identifier_column: str,
        identifier: str,
        target: LedgerState,
        evidence: Mapping[str, object] | None,
        fields: Mapping[str, object],
    ) -> ProgrammeRecord | ShardRecord | BatchRecord:
        safe_evidence = self._metadata_json(evidence or {})
        now = self._now()
        with self.transaction() as connection:
            row = self._require_row(connection, table, identifier_column, identifier)
            current = LedgerState(row["state"])
            requested_commit = fields.get("commit_id")
            if requested_commit is not None and row["commit_id"] is not None:
                self._validate_commit(cast(str, requested_commit))
                if requested_commit != row["commit_id"]:
                    raise EvidenceError(
                        "a committed object cannot change its commit SHA"
                    )
            if current == target:
                return self._record_for_table(table, row)
            if not valid_ledger_transition(current, target):
                raise InvalidTransition(
                    f"{current.value} -> {target.value} is not allowed"
                )
            updates: dict[str, object] = {"state": target.value, "updated_at": now}
            if table == "programmes":
                updates.update(
                    self._programme_updates(target, row, fields, safe_evidence, now)
                )
            elif table == "shards":
                updates.update(self._shard_updates(target, fields, safe_evidence, now))
            else:
                updates.update(
                    self._batch_updates(target, row, fields, safe_evidence, now)
                )
            assignments = ", ".join(f"{column} = ?" for column in updates)
            connection.execute(
                f"UPDATE {table} SET {assignments} WHERE {identifier_column} = ?",
                (*updates.values(), identifier),
            )
        return self._record_for_table(
            table, self._fetch(table, identifier_column, identifier)
        )

    def _batch_updates(
        self,
        target: LedgerState,
        row: sqlite3.Row,
        fields: Mapping[str, object],
        evidence: str,
        now: str,
    ) -> dict[str, object]:
        updates = self._common_updates(
            fields,
            evidence,
            allowed_keys={"duration_ms", "rejection_counts", "commit_id", "last_error"},
        )
        if target in {LedgerState.COMMITTED, LedgerState.VERIFIED, LedgerState.PURGED}:
            attached = self._connection.execute(
                "SELECT COUNT(*) FROM shards WHERE batch_id = ?", (row["batch_id"],)
            ).fetchone()[0]
            if attached == 0:
                raise EvidenceError("committed batches require at least one shard")
            commit_id = fields.get("commit_id", row["commit_id"])
            if not isinstance(commit_id, str) or not _COMMIT_RE.fullmatch(commit_id):
                raise EvidenceError("committed batches require a complete commit SHA")
        if target == LedgerState.PROCESSING:
            updates["attempts"] = int(row["attempts"]) + 1
            updates["processing_started_at"] = now
        if target == LedgerState.COMMITTED:
            updates["remote_checked_at"] = None
            updates["remote_present"] = None
        if target == LedgerState.VERIFIED:
            updates["verification_time"] = now
        if target == LedgerState.PURGED:
            updates["purge_time"] = now
        return updates

    @staticmethod
    def _common_updates(
        fields: Mapping[str, object], evidence: str, *, allowed_keys: set[str]
    ) -> dict[str, object]:
        allowed = {
            "accepted_count": "accepted_count",
            "rejected_count": "rejected_count",
            "source_duration_ms": "source_duration_ms",
            "processed_duration_ms": "processed_duration_ms",
            "duration_ms": "duration_ms",
            "rejection_counts": "rejection_counts",
            "commit_id": "commit_id",
            "last_error": "last_error",
        }
        updates: dict[str, object] = {"last_evidence": evidence}
        for key, column in allowed.items():
            if key not in allowed_keys or key not in fields:
                continue
            value = fields[key]
            if (
                key.endswith("_count")
                or key.endswith("_duration_ms")
                or key == "duration_ms"
            ):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise EvidenceError(f"{key} must be a non-negative integer")
            if key == "rejection_counts":
                value = Ledger._rejection_counts(value)
            if key == "commit_id":
                Ledger._validate_commit(cast(str, value))
            if key == "last_error" and value is not None and not isinstance(value, str):
                raise EvidenceError("last_error must be text")
            updates[column] = value
        return updates

    @staticmethod
    def _rejection_counts(value: object) -> str:
        if not isinstance(value, Mapping):
            raise EvidenceError("rejection_counts must be a mapping")
        counts: dict[str, int] = {}
        for key, count in value.items():
            if not isinstance(key, (str, RejectionCategory)):
                raise EvidenceError("rejection categories must be strings")
            category = Ledger._rejection_category(
                key.value if isinstance(key, RejectionCategory) else key
            )
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise EvidenceError("rejection counts must be non-negative integers")
            counts[category] = count
        return json.dumps(counts, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _rejection_category(value: RejectionCategory | str) -> str:
        try:
            return RejectionCategory(value).value
        except ValueError as error:
            raise EvidenceError(f"unknown rejection category: {value!r}") from error

    @staticmethod
    def _validate_commit(value: object) -> None:
        if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
            raise EvidenceError("commit_id must be a complete 40-character commit SHA")

    def _fetch(self, table: str, column: str, value: str) -> sqlite3.Row:
        row = self._connection.execute(
            f"SELECT * FROM {table} WHERE {column} = ?", (value,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown {table[:-1]} {value!r}")
        return row

    @staticmethod
    def _metadata_json(value: object) -> str:
        if isinstance(value, ContractModel):
            serialisable = Ledger._metadata_value(value.model_dump(mode="json"))
        elif isinstance(value, Mapping):
            serialisable = Ledger._metadata_value(value)
        elif value is None:
            serialisable = {}
        else:
            raise EvidenceError("evidence must be a mapping or contract")
        Ledger._validate_metadata(serialisable)
        try:
            return json.dumps(serialisable, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise EvidenceError("evidence is not JSON metadata") from error

    @staticmethod
    def _metadata_value(value: object) -> object:
        if isinstance(value, ContractModel):
            return Ledger._metadata_value(value.model_dump(mode="json"))
        if isinstance(value, Mapping):
            converted: dict[str, object] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise EvidenceError("metadata keys must be strings")
                converted[key] = Ledger._metadata_value(child)
            return converted
        if isinstance(value, (list, tuple)):
            return [Ledger._metadata_value(child) for child in value]
        return value

    @staticmethod
    def _validate_metadata(value: object, key: str = "") -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise EvidenceError("audio or binary payloads are not allowed")
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                if not isinstance(raw_key, str):
                    raise EvidenceError("metadata keys must be strings")
                normalised = raw_key.lower().replace("-", "_")
                if normalised in _FORBIDDEN_KEYS:
                    raise EvidenceError(f"metadata field {raw_key!r} is not permitted")
                Ledger._validate_metadata(child, normalised)
        elif isinstance(value, (list, tuple)):
            for child in value:
                Ledger._validate_metadata(child, key)
        elif isinstance(value, str):
            lowered = value.lower()
            if "bearer " in lowered or re.search(r"\bhf_[a-z0-9]{20,}\b", lowered):
                raise EvidenceError("credentials are not allowed in metadata")
            if (
                key == "path"
                or key.endswith(("_path", "_dir", "_file"))
                or "cache" in key
                or "scratch" in key
            ) and Ledger._looks_like_local_path(value):
                raise EvidenceError("machine or cache paths are not allowed")
        elif isinstance(value, float) and not math.isfinite(value):
            raise EvidenceError("metadata numbers must be finite")
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise EvidenceError("metadata must contain JSON-compatible values")

    @staticmethod
    def _looks_like_local_path(value: str) -> bool:
        path = Path(value)
        return path.is_absolute() or any(
            part.lower() in _FORBIDDEN_PATH_PARTS for part in path.parts
        )

    def _programme_updates(
        self,
        target: LedgerState,
        row: sqlite3.Row,
        fields: Mapping[str, object],
        evidence: str,
        now: str,
    ) -> dict[str, object]:
        updates = self._common_updates(
            fields,
            evidence,
            allowed_keys={
                "accepted_count",
                "rejected_count",
                "source_duration_ms",
                "processed_duration_ms",
                "rejection_counts",
                "commit_id",
                "last_error",
            },
        )
        if target in {LedgerState.COMMITTED, LedgerState.VERIFIED, LedgerState.PURGED}:
            commit_id = fields.get("commit_id", row["commit_id"])
            if not isinstance(commit_id, str) or not _COMMIT_RE.fullmatch(commit_id):
                raise EvidenceError(
                    "committed programmes require a complete commit SHA"
                )
        if target == LedgerState.PROCESSING:
            updates["attempts"] = int(row["attempts"]) + 1
            updates["processing_started_at"] = now
        if target == LedgerState.RETRYABLE:
            updates["commit_id"] = None
        if target == LedgerState.VERIFIED:
            updates["verification_time"] = now
        if target == LedgerState.PURGED:
            updates["purge_time"] = now
        return updates

    def _record_for_table(
        self, table: str, row: sqlite3.Row
    ) -> ProgrammeRecord | ShardRecord | BatchRecord:
        if table == "programmes":
            return self._programme_record(row)
        if table == "shards":
            return self._shard_record(row)
        return self._batch_record(row)

    @staticmethod
    def _batch_record(row: sqlite3.Row) -> BatchRecord:
        return BatchRecord(
            batch_id=row["batch_id"],
            state=LedgerState(row["state"]),
            pipeline_digest=row["pipeline_digest"],
            commit_id=row["commit_id"],
            attempts=row["attempts"],
            programme_count=row["programme_count"],
            row_count=row["row_count"],
            rejection_counts=json.loads(row["rejection_counts"]),
            duration_ms=row["duration_ms"],
            verification_time=row["verification_time"],
            purge_time=row["purge_time"],
            publication_artifact_purged_at=row["publication_artifact_purged_at"],
            publication_artifact_purge_evidence=json.loads(
                row["publication_artifact_purge_evidence"]
            ),
            remote_checked_at=row["remote_checked_at"],
            remote_present=None
            if row["remote_present"] is None
            else bool(row["remote_present"]),
            last_error=row["last_error"],
        )

    @staticmethod
    def _programme_record(row: sqlite3.Row) -> ProgrammeRecord:
        return ProgrammeRecord(
            programme_id=row["programme_id"],
            source_file_id=row["source_file_id"],
            state=LedgerState(row["state"]),
            source_revisions=json.loads(row["source_revisions"]),
            pipeline_digest=row["pipeline_digest"],
            attempts=row["attempts"],
            accepted_count=row["accepted_count"],
            rejected_count=row["rejected_count"],
            source_duration_ms=row["source_duration_ms"],
            processed_duration_ms=row["processed_duration_ms"],
            commit_id=row["commit_id"],
            rejection_counts=json.loads(row["rejection_counts"]),
            source_temp_purged_at=row["source_temp_purged_at"],
            source_temp_purge_evidence=json.loads(row["source_temp_purge_evidence"]),
            verification_time=row["verification_time"],
            purge_time=row["purge_time"],
            last_error=row["last_error"],
        )

    @staticmethod
    def _shard_record(row: sqlite3.Row) -> ShardRecord:
        return ShardRecord(
            shard_id=row["shard_id"],
            programme_id=row["programme_id"],
            batch_id=row["batch_id"],
            state=LedgerState(row["state"]),
            path=row["path"],
            byte_size=row["byte_size"],
            row_count=row["row_count"],
            sha256=row["sha256"],
            verification_time=row["verification_time"],
            purge_time=row["purge_time"],
            local_path=row["local_path"],
        )

    @staticmethod
    def _require_row(
        connection: sqlite3.Connection, table: str, column: str, value: str
    ) -> sqlite3.Row:
        row = connection.execute(
            f"SELECT * FROM {table} WHERE {column} = ?", (value,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown {table[:-1]} {value!r}")
        return row

    def _shard_updates(
        self, target: LedgerState, fields: Mapping[str, object], evidence: str, now: str
    ) -> dict[str, object]:
        updates = self._common_updates(fields, evidence, allowed_keys={"last_error"})
        if target == LedgerState.VERIFIED:
            updates["verification_time"] = now
        if target == LedgerState.PURGED:
            updates["purge_time"] = now
        return updates

    @staticmethod
    def _validate_identifier(value: object, name: str) -> None:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise EvidenceError(f"{name} must be a non-empty identifier")
        if name.endswith("_file_id") and Ledger._looks_like_local_path(value):
            raise EvidenceError("machine or cache paths are not allowed")
        Ledger._validate_metadata({name: value})

    @staticmethod
    def _validate_nonnegative(value: object, name: str) -> None:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise EvidenceError(f"{name} must be a non-negative integer")

    def batch_evidence(self, batch_id: str) -> BatchEvidence:
        """Build the serialisable contract evidence for a batch.

        Returns:
            Metadata validated by the Phase 1A batch contract.
        """
        batch = self.batch(batch_id)
        return BatchEvidence(
            batch_id=batch.batch_id,
            state=batch.state,
            shards=tuple(
                ShardEvidence(
                    path=shard.path,
                    byte_size=shard.byte_size,
                    row_count=shard.row_count,
                    sha256=shard.sha256,
                )
                for shard in self.shards(batch_id)
            ),
            commit_id=batch.commit_id,
            programme_count=batch.programme_count,
            row_count=batch.row_count,
            rejection_counts={
                RejectionCategory(key): value
                for key, value in batch.rejection_counts.items()
            },
        )

    def shards(self, batch_id: str) -> tuple[ShardRecord, ...]:
        """Return all shards attached to a batch in deterministic order."""
        rows = self._connection.execute(
            "SELECT * FROM shards WHERE batch_id = ? ORDER BY shard_id", (batch_id,)
        ).fetchall()
        return tuple(self._shard_record(row) for row in rows)


__all__ = [
    "BatchRecord",
    "EvidenceError",
    "InvalidTransition",
    "Ledger",
    "LedgerError",
    "ProgrammeRecord",
    "ShardRecord",
]
