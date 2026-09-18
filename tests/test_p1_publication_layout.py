"""Offline tests for the P1 publication layout contract."""

from __future__ import annotations

import hashlib

import pytest

from p1_dataset.publication_layout import (
    is_allowed_shard_path,
    legacy_shard_path,
    new_shard_path,
    shard_bucket,
)


def test_allowlist_accepts_both_roots_and_rejects_traversal() -> None:
    """Only exact immutable legacy and active shard paths are accepted."""
    assert is_allowed_shard_path(legacy_shard_path("programme-1", 0))
    assert is_allowed_shard_path(new_shard_path("programme-1", 0))
    assert not is_allowed_shard_path("data/train/part.parquet")
    assert not is_allowed_shard_path(
        "data-shards/train/aa/../programme-1/part-00000.parquet"
    )
    assert not is_allowed_shard_path("data-shards/train/zz/programme-1/part.parquet")


def test_invalid_layout_identity_is_refused() -> None:
    """Path traversal and unbounded ordinals cannot enter the layout."""
    with pytest.raises(ValueError):
        new_shard_path("../programme", 0)
    with pytest.raises(ValueError):
        new_shard_path("programme", 100_000)


def test_layout_is_deterministic_and_fanned_out() -> None:
    """The same identity always selects one bounded two-hex bucket."""
    identifiers = [f"programme-{index}" for index in range(10_000)]
    buckets = {shard_bucket(identifier) for identifier in identifiers}
    assert len(buckets) > 200
    assert all(len(bucket) == 2 for bucket in buckets)
    assert new_shard_path("programme-1", 0) == new_shard_path("programme-1", 0)
    assert hashlib.sha256(b"programme-1").hexdigest()[:2] == shard_bucket("programme-1")
