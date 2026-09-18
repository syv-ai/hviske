"""Safe, deterministic publication paths for P1 Parquet shards.

Publication layout is deliberately separate from the canonical row identity.  It can
therefore change without changing the pipeline digest or any Parquet bytes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

LEGACY_SHARD_ROOT = "data/train"
SHARD_ROOT = "data-shards/train"
BUCKET_WIDTH = 2
_PART_NAME = re.compile(r"part-[0-9]{5}\.parquet\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_LEGACY_NAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)-([0-9]{5})\.parquet\Z")
_BUCKET = re.compile(r"[0-9a-f]{2}\Z")


@dataclass(frozen=True)
class PublicationLayout:
    """Runtime publication layout, excluded from canonical row identity."""

    version: str = "p1-publication-layout-2"
    shard_root: str = SHARD_ROOT
    bucket_width: int = BUCKET_WIDTH

    def __post_init__(self) -> None:
        """Validate the fixed active layout values.

        Raises:
            ValueError:
                If a layout value differs from the active contract.
        """
        if self.shard_root != SHARD_ROOT:
            raise ValueError("the active P1 shard root is fixed")
        if self.bucket_width != BUCKET_WIDTH:
            raise ValueError("the active P1 bucket width is fixed")


def is_allowed_shard_path(path: str) -> bool:
    """Whether ``path`` is an exact legacy or active P1 shard path.

    Returns:
        True only for a canonical path in one of the two approved roots.
    """
    if not isinstance(path, str) or "\\" in path or "\x00" in path:
        return False
    parsed = PurePosixPath(path)
    # PurePosixPath deliberately normalises aliases.  Comparing its spelling is
    # therefore part of the allowlist, rather than merely inspecting its parts.
    if (
        parsed.is_absolute()
        or parsed.as_posix() != path
        or ".." in parsed.parts
        or "." in parsed.parts
    ):
        return False
    parts = parsed.parts
    if len(parts) == 3 and parts[:2] == ("data", "train"):
        match = _LEGACY_NAME.fullmatch(parts[2])
        return match is not None and _ID.fullmatch(match.group(1)) is not None
    if len(parts) == 5 and parts[:2] == ("data-shards", "train"):
        if not _BUCKET.fullmatch(parts[2]) or not _ID.fullmatch(parts[3]):
            return False
        if not _PART_NAME.fullmatch(parts[4]):
            return False
        return parts[2] == shard_bucket(parts[3])
    return False


def shard_bucket(deterministic_id: str) -> str:
    """Return the two-hex-digit bucket for an existing deterministic path ID."""
    _validate_id(deterministic_id)
    return hashlib.sha256(deterministic_id.encode("utf-8")).hexdigest()[:BUCKET_WIDTH]


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("deterministic path ID is not safe")


def legacy_shard_path(deterministic_id: str, ordinal: int) -> str:
    """Return the legacy path used by existing local ledgers and Hub objects.

    Raises:
        ValueError:
            If the identifier or ordinal is unsafe.
    """
    _validate_id(deterministic_id)
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 0 <= ordinal <= 99999
    ):
        raise ValueError("shard ordinal must be between zero and 99999")
    return f"{LEGACY_SHARD_ROOT}/{deterministic_id}-{ordinal:05d}.parquet"


def new_shard_path(deterministic_id: str, ordinal: int) -> str:
    """Return a fan-out path without changing the deterministic shard identity.

    Raises:
        ValueError:
            If the identifier or ordinal is unsafe.
    """
    _validate_id(deterministic_id)
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 0 <= ordinal <= 99999
    ):
        raise ValueError("shard ordinal must be between zero and 99999")
    return (
        f"{SHARD_ROOT}/{shard_bucket(deterministic_id)}/{deterministic_id}/"
        f"part-{ordinal:05d}.parquet"
    )


bucket_for_deterministic_id = shard_bucket
programme_shard_path = new_shard_path


__all__ = [
    "BUCKET_WIDTH",
    "LEGACY_SHARD_ROOT",
    "PublicationLayout",
    "SHARD_ROOT",
    "bucket_for_deterministic_id",
    "is_allowed_shard_path",
    "legacy_shard_path",
    "new_shard_path",
    "programme_shard_path",
    "shard_bucket",
]
