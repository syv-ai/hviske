"""Focused tests for the crash-safe Phase 1B metadata ledger."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from hviske.p1_contracts import LedgerState, RejectionCategory
from hviske.p1_ledger import EvidenceError, InvalidTransition, Ledger, ShardAllocation

DIGEST = "a" * 64
COMMIT = "b" * 40
REVISIONS = {
    "audio": {"repository": "syvai/p1-audio", "revision": "c" * 40},
    "transcripts": {"repository": "syvai/p1-transcripts", "revision": "d" * 40},
}


def test_atomic_batch_allocation_persists_local_identity_before_sharding(
    tmp_path: Path,
) -> None:
    """Batch attachment and programme sharding commit as one transaction."""
    local = tmp_path / "part.parquet"
    local.write_bytes(b"durable shard")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_programme(ledger)
        batch, shards = ledger.allocate_batch_with_shards(
            "programme-1",
            [
                ShardAllocation(
                    local_path=local,
                    remote_path="data/part.parquet",
                    sha256=digest,
                    byte_size=local.stat().st_size,
                    row_count=2,
                )
            ],
        )
        assert batch.state is LedgerState.SHARDED
        assert ledger.programme("programme-1").state is LedgerState.SHARDED
        assert shards[0].batch_id == batch.batch_id
        assert shards[0].local_path == str(local.resolve())
        assert ledger.unattached_local_shards() == ()
        alternate = tmp_path / "elsewhere" / local.name
        alternate.parent.mkdir()
        alternate.write_bytes(local.read_bytes())
        assert not ledger.reconcile_local_shard(shards[0].shard_id, alternate)
        assert ledger.reconcile_local_shard(shards[0].shard_id)


def add_programme(ledger: Ledger, programme_id: str = "programme-1") -> None:
    """Add the standard metadata-only programme fixture."""
    ledger.register_programme(
        programme_id,
        source_file_id=f"source-{programme_id}",
        source_revisions=REVISIONS,
        pipeline_digest=DIGEST,
        source_duration_ms=12_000,
    )


def test_batch_shard_state_machine_and_separate_publication_purge(
    tmp_path: Path,
) -> None:
    """Remote verification precedes publication-artifact deletion."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_batch(ledger)
        ledger.transition_batch("batch-1", LedgerState.PROCESSING)
        ledger.transition_batch("batch-1", LedgerState.SHARDED)
        ledger.transition_batch("batch-1", LedgerState.COMMITTED, commit_id=COMMIT)
        assert ledger.batch("batch-1").commit_id == COMMIT
        with pytest.raises(InvalidTransition):
            ledger.mark_publication_artifacts_purged("batch-1")
        ledger.transition_batch("batch-1", LedgerState.VERIFIED)
        verified = ledger.mark_publication_artifacts_purged(
            "batch-1", evidence={"deleted": True, "kind": "publication-artifact"}
        )
        assert verified.publication_artifact_purged_at is not None
        assert verified.publication_artifact_purge_evidence["kind"] == (
            "publication-artifact"
        )
        assert ledger.purge_batch("batch-1").state is LedgerState.PURGED
        assert ledger.batch_evidence("batch-1").commit_id == COMMIT


def add_batch(ledger: Ledger, batch_id: str = "batch-1") -> None:
    """Add a batch and its checksum-backed local shard."""
    add_programme(ledger)
    ledger.register_batch(batch_id, pipeline_digest=DIGEST)
    ledger.register_shard(
        "shard-1",
        path="train/shard-000.parquet",
        sha256=DIGEST,
        byte_size=3,
        row_count=2,
        programme_id="programme-1",
    )
    ledger.attach_shard(batch_id, "shard-1")


def test_local_reconciliation_requires_exact_digest_and_regular_file(
    tmp_path: Path,
) -> None:
    """Only matching ordinary files can be reused after a restart."""
    content = b"parquet metadata fixture"
    digest = hashlib.sha256(content).hexdigest()
    local = tmp_path / "shard.parquet"
    local.write_bytes(content)
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        ledger.register_shard(
            "shard-1",
            path="train/shard.parquet",
            sha256=digest,
            byte_size=len(content),
            row_count=1,
        )
        assert ledger.reconcile_local_shard("shard-1", local)
        local.write_bytes(b"changed")
        assert not ledger.reconcile_local_shard("shard-1", local)


def test_programme_complete_state_machine_and_evidence(tmp_path: Path) -> None:
    """A programme records attempts, counts, source purge, and timestamps."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_programme(ledger)
        assert ledger.programme("programme-1").state is LedgerState.DISCOVERED
        assert ledger.start_processing("programme-1").attempts == 1
        record = ledger.transition_programme(
            "programme-1",
            LedgerState.SHARDED,
            evidence={"accepted_count": 4, "rejected_count": 1},
            processed_duration_ms=11_500,
            rejection_counts={RejectionCategory.LOW_SPEECH_RATIO: 1},
        )
        assert record.accepted_count == 4
        assert record.rejection_counts == {"low_speech_ratio": 1}
        assert record.source_temp_purged_at is None
        ledger.mark_source_temps_purged(
            "programme-1", evidence={"deleted": True, "kind": "source-temporary"}
        )
        ledger.transition_programme(
            "programme-1", LedgerState.COMMITTED, commit_id=COMMIT
        )
        ledger.transition_programme("programme-1", LedgerState.VERIFIED)
        record = ledger.transition_programme("programme-1", LedgerState.PURGED)
        assert record.state is LedgerState.PURGED
        assert record.source_temp_purged_at is not None
        assert record.purge_time is not None


def test_publication_path_collision_is_refused(tmp_path: Path) -> None:
    """Two shard identities cannot claim the same publication path."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_batch(ledger)
        with pytest.raises(EvidenceError, match="publication path"):
            ledger.register_shard(
                "shard-2",
                path="train/shard-000.parquet",
                sha256=DIGEST,
                byte_size=4,
                row_count=1,
                batch_id="batch-1",
            )


def test_purged_batch_child_finalisation_is_idempotent(tmp_path: Path) -> None:
    """A restart after physical purge repairs all child states without reupload."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_batch(ledger)
        ledger.transition_programme("programme-1", LedgerState.PROCESSING)
        ledger.transition_programme("programme-1", LedgerState.SHARDED)
        ledger.transition_batch("batch-1", LedgerState.PROCESSING)
        ledger.transition_batch("batch-1", LedgerState.SHARDED)
        ledger.transition_batch("batch-1", LedgerState.COMMITTED, commit_id=COMMIT)
        ledger.transition_batch("batch-1", LedgerState.VERIFIED)
        ledger.purge_batch("batch-1")

        assert ledger.finalise_batch_children("batch-1").state is LedgerState.PURGED
        assert ledger.programme("programme-1").state is LedgerState.PURGED
        assert ledger.shard("shard-1").state is LedgerState.PURGED
        ledger.finalise_batch_children("batch-1")


def test_rejected_programmes_and_retryable_transitions_are_durable(
    tmp_path: Path,
) -> None:
    """Failure states retain their reason and can retry before final rejection."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_programme(ledger)
        ledger.start_processing("programme-1")
        retry = ledger.transition_programme(
            "programme-1", LedgerState.RETRYABLE, last_error="temporary decode failure"
        )
        assert retry.state is LedgerState.RETRYABLE
        assert retry.attempts == 1
        rejected = ledger.reject_programme(
            "programme-1", reason=RejectionCategory.DECODE_ERROR
        )
        assert rejected.state is LedgerState.REJECTED
        assert rejected.rejection_counts == {"decode_error": 1}


def test_rejected_programme_is_terminal_across_two_runs(tmp_path: Path) -> None:
    """A restart skips rejected work and cannot reopen its terminal state."""
    database = tmp_path / "ledger.sqlite"
    with Ledger(database) as ledger:
        add_programme(ledger)
        ledger.start_processing("programme-1")
        rejected = ledger.reject_programme(
            "programme-1", reason=RejectionCategory.NO_ACCEPTED_SEGMENTS
        )
        assert rejected.state is LedgerState.REJECTED
        assert rejected.attempts == 1

    with Ledger(database) as restarted:
        record = restarted.programme("programme-1")
        assert record.state is LedgerState.REJECTED
        assert record.attempts == 1
        with pytest.raises(InvalidTransition):
            restarted.start_processing("programme-1")
        with pytest.raises(InvalidTransition):
            restarted.transition_programme("programme-1", LedgerState.RETRYABLE)


def test_remote_reconciliation_avoids_reupload_or_marks_retryable(
    tmp_path: Path,
) -> None:
    """The remote hook sees immutable commit and publication-relative paths."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        add_batch(ledger)
        ledger.transition_batch("batch-1", LedgerState.PROCESSING)
        ledger.transition_batch("batch-1", LedgerState.SHARDED)
        ledger.transition_batch("batch-1", LedgerState.COMMITTED, commit_id=COMMIT)
        seen: list[tuple[str, tuple[str, ...]]] = []

        def remote(commit: str, paths: tuple[str, ...]) -> bool:
            seen.append((commit, paths))
            return True

        assert ledger.reconcile_committed_batch("batch-1", remote)
        assert seen == [(COMMIT, ("train/shard-000.parquet",))]
        assert ledger.batch("batch-1").state is LedgerState.COMMITTED

        ledger.register_batch("batch-2", pipeline_digest=DIGEST)
        ledger.register_shard(
            "shard-2",
            path="train/shard-001.parquet",
            sha256=DIGEST,
            byte_size=3,
            row_count=1,
        )
        ledger.attach_shard("batch-2", "shard-2")
        ledger.transition_batch("batch-2", LedgerState.PROCESSING)
        ledger.transition_batch("batch-2", LedgerState.SHARDED)
        ledger.transition_batch("batch-2", LedgerState.COMMITTED, commit_id=COMMIT)
        assert not ledger.reconcile_remote("batch-2", lambda _commit, _paths: False)
        assert ledger.batch("batch-2").state is LedgerState.RETRYABLE


def test_restart_enumerates_unattached_local_shards_and_committed_batches(
    tmp_path: Path,
) -> None:
    """Recovery exposes durable local identities without basename matching."""
    local = tmp_path / "part.parquet"
    local.write_bytes(b"local shard")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        shard = ledger.register_shard(
            "orphan",
            path="data/orphan.parquet",
            local_path=local,
            sha256=digest,
            byte_size=local.stat().st_size,
            row_count=1,
        )
        assert ledger.unattached_local_shards() == (shard,)
        assert ledger.recovery_work()[0] == (shard,)


def test_restart_resets_abandoned_processing_and_preserves_attempts(
    tmp_path: Path,
) -> None:
    """Opening a new connection makes abandoned work retryable."""
    database = tmp_path / "ledger.sqlite"
    first = Ledger(database)
    add_programme(first)
    first.start_processing("programme-1")
    first.close()

    with Ledger(database) as second:
        record = second.programme("programme-1")
        assert record.state is LedgerState.RETRYABLE
        assert record.attempts == 1
        assert second.start_processing("programme-1").attempts == 2


def test_schema_is_atomic_and_metadata_only(tmp_path: Path) -> None:
    """Schema creation is durable and contains no content-bearing columns."""
    database = tmp_path / "ledger.sqlite"
    with Ledger(database) as ledger:
        version = ledger._connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == 1
        columns = {
            row[1]
            for row in ledger._connection.execute("PRAGMA table_info(programmes)")
        }
        assert "transcript_text" not in columns
        assert "audio_bytes" not in columns
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    connection.close()


def test_sequences_are_durable_monotonic_and_work_is_reconstructable(
    tmp_path: Path,
) -> None:
    """Restarting never reuses an allocation and exposes committed work."""
    database = tmp_path / "ledger.sqlite"
    with Ledger(database) as ledger:
        batch_id = ledger.allocate_batch_id()
        shard_id = ledger.allocate_shard_id()
        assert batch_id == "batch-00000001"
        assert shard_id == "shard-00000001"
        ledger.register_batch(batch_id, pipeline_digest=DIGEST)
        ledger.register_shard(
            shard_id,
            path="data/part-00001.parquet",
            sha256=DIGEST,
            byte_size=4,
            row_count=1,
            batch_id=batch_id,
        )
        ledger.transition_batch(batch_id, LedgerState.PROCESSING)
        ledger.transition_batch(batch_id, LedgerState.SHARDED)
        ledger.record_commit(batch_id, COMMIT)
    with Ledger(database) as ledger:
        assert ledger.allocate_batch_id() == "batch-00000002"
        assert ledger.allocate_shard_id() == "shard-00000002"
        work = ledger.reconstruct_work()
        assert work[0][0].state is LedgerState.COMMITTED
        assert work[0][1][0].shard_id == "shard-00000001"


def test_transaction_rolls_back_state_and_evidence(tmp_path: Path) -> None:
    """An exception in an explicit transaction leaves no partial row.

    Raises:
        RuntimeError:
            Simulated failure inside the transaction.
    """
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        with pytest.raises(RuntimeError):
            with ledger.transaction() as connection:
                connection.execute(
                    "INSERT INTO batches(batch_id, state, pipeline_digest, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("never-committed", "discovered", DIGEST, "now", "now"),
                )
                raise RuntimeError("simulated crash")
        with pytest.raises(KeyError):
            ledger.batch("never-committed")


def test_verified_shard_path_cannot_change_and_metadata_is_restricted(
    tmp_path: Path,
) -> None:
    """The ledger refuses payloads, credentials, and local paths."""
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        ledger.register_shard(
            "shard-1",
            path="train/shard.parquet",
            sha256=DIGEST,
            byte_size=3,
            row_count=1,
        )
        ledger.transition_shard("shard-1", LedgerState.COMMITTED)
        ledger.transition_shard("shard-1", LedgerState.VERIFIED)
        with pytest.raises(EvidenceError):
            ledger.register_shard(
                "shard-1",
                path="train/shard.parquet",
                sha256="e" * 64,
                byte_size=3,
                row_count=1,
            )
        with pytest.raises(EvidenceError):
            ledger.transition_shard(
                "shard-1", LedgerState.PURGED, evidence={"text": "secret"}
            )
        with pytest.raises(EvidenceError):
            ledger.register_shard(
                "shard-2",
                path="/Users/dan/.cache/shard.parquet",
                sha256=DIGEST,
                byte_size=3,
                row_count=1,
            )
        with pytest.raises(EvidenceError):
            ledger.transition_shard(
                "shard-1", LedgerState.PURGED, evidence={"token": "hf_" + "x" * 24}
            )
        assert (
            ledger.transition_shard("shard-1", LedgerState.PURGED).state
            is LedgerState.PURGED
        )
