"""Direct tests for the local P1 publication-layout migration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from p1_dataset.layout_migration import migrate_layout
from p1_dataset.ledger import Ledger, ShardAllocation
from p1_dataset.publication_layout import (
    is_allowed_shard_path,
    legacy_shard_path,
    new_shard_path,
    shard_bucket,
)

DIGEST = "a" * 64


@pytest.mark.parametrize(
    "path",
    [
        "data-shards/train/00/programme-0/part-00000.parquet",
        "data-shards/train/../programme-0/part-00000.parquet",
        "data-shards/train/00//programme-0/part-00000.parquet",
        "data-shards/train/00/programme-0\\part-00000.parquet",
    ],
)
def test_canonical_aliases_and_bucket_mismatches_are_rejected(path: str) -> None:
    """Allowlisting requires exact spelling and the derived bucket."""
    assert not is_allowed_shard_path(path)
    assert is_allowed_shard_path(
        f"data-shards/train/{shard_bucket('programme-0')}/programme-0/part-00000.parquet"
    )


def test_flat_legacy_layout_migrates_all_partitions_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """A stopped eight-partition run migrates metadata, never Parquet bytes."""
    local_bytes = _make_run(tmp_path)
    before = _parquet_bytes(tmp_path)

    dry_run = migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=False)
    assert dry_run.shard_count == 8
    assert not (tmp_path / "supervisor" / "DONE").exists()

    applied = migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=True)
    assert applied.shard_count == 8
    assert (tmp_path / "supervisor" / "DONE").exists()
    assert not (tmp_path / "supervisor" / "FAILED").exists()
    assert _parquet_bytes(tmp_path) == before == local_bytes

    for index in range(8):
        with Ledger(
            tmp_path / f"partition-{index}" / "ledger.sqlite",
            pipeline_digest=DIGEST,
            reset_processing=False,
        ) as ledger:
            assert ledger.shards(f"batch-{index}")[0].path == new_shard_path(
                f"programme-{index}", 0
            )
    rerun = migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=True)
    assert rerun.shard_count == 0


def _make_run(
    root: Path, *, audit: bool = False, sealed: bool = False, manifest: bool = False
) -> bytes:
    """Create representative metadata-only ledgers with synthetic identifiers.

    Returns:
        The bytes written to every synthetic shard.
    """
    root.mkdir(exist_ok=True)
    marker_root = root / "supervisor" / "markers"
    marker_root.mkdir(parents=True)
    parquet_bytes = b"synthetic parquet placeholder"
    sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    for index in range(8):
        partition = root / f"partition-{index}"
        partition.mkdir()
        local = partition / "part-00000.parquet"
        local.write_bytes(parquet_bytes)
        path = legacy_shard_path(f"programme-{index}", 0)
        with Ledger(
            partition / "ledger.sqlite", pipeline_digest=DIGEST, reset_processing=False
        ) as ledger:
            ledger.register_programme(
                f"programme-{index}",
                source_file_id=f"source-{index}",
                source_revisions={},
                pipeline_digest=DIGEST,
            )
            ledger.start_processing(f"programme-{index}")
            batch, shards = ledger.allocate_batch_with_shards(
                f"programme-{index}",
                [
                    ShardAllocation(
                        local_path=local,
                        remote_path=path,
                        sha256=sha256,
                        byte_size=len(parquet_bytes),
                        row_count=1,
                    )
                ],
                batch_id=f"batch-{index}",
                audit_candidates=(
                    {
                        "path": path,
                        "segment_id": f"segment-{index}",
                        "status": "accepted",
                        "opaque": "preserve-me",
                        "local_path": str(local),
                        "local_row_locator": 0,
                    },
                )
                if audit
                else (),
            )
            if sealed:
                ledger.seal_batch(batch.batch_id)
        if manifest:
            manifest_path = local.parent / "manifests" / f"batch-{index}.json"
            manifest_path.parent.mkdir()
            manifest_path.write_bytes(
                json.dumps(
                    {
                        "batch_id": batch.batch_id,
                        "programme_count": 1,
                        "row_count": 1,
                        "rejection_counts": {},
                        "shards": [
                            {
                                "path": shards[0].path,
                                "byte_size": shards[0].byte_size,
                                "row_count": shards[0].row_count,
                                "sha256": shards[0].sha256,
                            }
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
    (root / "supervisor" / "FAILED").write_text(
        "".join(f"partition={index} attempts=1\n" for index in range(8)),
        encoding="utf-8",
    )
    return parquet_bytes


def _parquet_bytes(root: Path) -> bytes:
    """Read one synthetic local shard for the no-byte-write assertion.

    Returns:
        The bytes in the first partition's local shard.
    """
    return (root / "partition-0" / "part-00000.parquet").read_bytes()


def test_manifest_is_validated_before_any_ledger_mutation(tmp_path: Path) -> None:
    """Malformed owned manifests abort before changing the ledger or marker."""
    _make_run(tmp_path, sealed=True, manifest=True)
    manifest = next((tmp_path / "partition-0").rglob("batch-0.json"))
    manifest.write_bytes(b"not json")
    before = _ledger_bytes(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=True)
    assert _ledger_bytes(tmp_path) == before
    assert (tmp_path / "supervisor" / "FAILED").exists()


def _ledger_bytes(root: Path) -> tuple[bytes, ...]:
    """Snapshot all eight primary SQLite files.

    Returns:
        The bytes of each ledger in partition order.
    """
    return tuple(
        (root / f"partition-{index}" / "ledger.sqlite").read_bytes()
        for index in range(8)
    )


def test_markers_must_prove_a_stopped_run(tmp_path: Path) -> None:
    """Markerless and inconsistent aggregate states are refused."""
    _make_run(tmp_path)
    failed = tmp_path / "supervisor" / "FAILED"
    failed.unlink()
    with pytest.raises(ValueError, match="FAILED"):
        migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=False)
    failed.write_text("partition=8 attempts=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=False)


def test_migration_preserves_audit_json_and_excludes_other_states(
    tmp_path: Path,
) -> None:
    """Only uncommitted sharded rows are remapped; audit evidence is untouched."""
    _make_run(tmp_path, audit=True)
    ledger_path = tmp_path / "partition-0" / "ledger.sqlite"
    with sqlite3.connect(ledger_path) as connection:
        audit_before = connection.execute(
            "SELECT candidate_json FROM audit_candidates"
        ).fetchone()[0]
        connection.execute("UPDATE shards SET state = 'processing'")
        connection.commit()
    migrate_layout(run_root=tmp_path, expected_digest=DIGEST, apply=True)
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute("SELECT path FROM shards").fetchone()[
            0
        ] == legacy_shard_path("programme-0", 0)
        assert (
            connection.execute(
                "SELECT candidate_json FROM audit_candidates"
            ).fetchone()[0]
            == audit_before
        )
