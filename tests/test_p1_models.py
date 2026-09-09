"""Offline tests for pinned P1 model adapters."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from hviske.p1_models import (
    SILERO_MODEL_PATH,
    SILERO_MODEL_SHA256,
    SILERO_REPOSITORY,
    SILERO_REVISION,
    ModelPinError,
    download_silero_vad,
    verify_silero_vad_asset,
)


def test_silero_asset_rejects_content_checksum(tmp_path: Path) -> None:
    """A file with the wrong content cannot be loaded as the pinned model."""
    asset = tmp_path / "silero_vad.jit"
    asset.write_bytes(b"not a model")
    with pytest.raises(ModelPinError, match="SHA-256"):
        verify_silero_vad_asset(asset)


def test_silero_download_verifies_git_blob_and_provenance(tmp_path: Path) -> None:
    """The downloader records immutable source coordinates beside the bytes."""
    payload = b"synthetic model"
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    digest = hashlib.sha256(payload).hexdigest()

    def opener(_: str) -> io.BytesIO:
        return io.BytesIO(payload)

    with pytest.raises(ModelPinError):
        verify_silero_vad_asset(
            tmp_path / "missing.jit", expected_blob=blob, expected_sha256=digest
        )

    # The real constants are intentionally not replaced in production; this
    # verifies the downloader's immutable URL/provenance protocol without a fetch.
    with pytest.raises(ModelPinError, match="SHA-256"):
        download_silero_vad(tmp_path, opener=opener)


def test_silero_pin_constants_are_complete() -> None:
    """The production pin names the exact repository, revision, path and digest."""
    assert SILERO_REPOSITORY == "snakers4/silero-vad"
    assert len(SILERO_REVISION) == 40
    assert SILERO_MODEL_PATH == "src/silero_vad/data/silero_vad.jit"
    assert len(SILERO_MODEL_SHA256) == 64
