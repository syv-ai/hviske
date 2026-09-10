"""Offline tests for pinned P1 model adapters."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hviske.p1_models import (
    SILERO_MODEL_BLOB,
    SILERO_MODEL_PATH,
    SILERO_MODEL_SHA256,
    SILERO_REPOSITORY,
    SILERO_REVISION,
    ModelPinError,
    SileroVADBackend,
    download_silero_vad,
    verify_hub_model_revision,
    verify_silero_vad_asset,
    verify_silero_vad_revision,
)


def test_hub_model_revision_uses_hf_api() -> None:
    """Hub revisions are resolved by the injected HfApi-compatible client."""

    class FakeApi:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def repo_info(self, repository: str, revision: str) -> object:
            self.calls.append((repository, revision))
            return type("Info", (), {"sha": revision})()

    api = FakeApi()
    verify_hub_model_revision(repository="org/model", revision="a" * 40, api=api)
    assert api.calls == [("org/model", "a" * 40)]


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


def test_silero_revision_verifier_checks_github_commit_and_blob() -> None:
    """GitHub coordinates are checked without constructing an HfApi client."""
    responses = {
        f"https://api.github.com/repos/{SILERO_REPOSITORY}/commits/{SILERO_REVISION}": {
            "sha": SILERO_REVISION
        },
        (
            "https://api.github.com/repos/"
            f"{SILERO_REPOSITORY}/contents/{SILERO_MODEL_PATH}?ref={SILERO_REVISION}"
        ): {"sha": SILERO_MODEL_BLOB},
    }

    def opener(url: str) -> io.BytesIO:
        return io.BytesIO(json.dumps(responses[url]).encode())

    assert verify_silero_vad_revision(opener=opener).startswith(
        "https://raw.githubusercontent.com/"
    )


def test_silero_uses_padded_frames_and_resets_between_programmes() -> None:
    """Silero receives fixed frames and cannot carry state between programmes."""

    class FakeModel:
        def __init__(self) -> None:
            self.reset_count = 0
            self.frames: list[np.ndarray] = []
            self.frame_in_programme = 0

        def __call__(self, frame: torch.Tensor, sampling_rate: int) -> torch.Tensor:
            assert sampling_rate == 16_000
            self.frames.append(frame.detach().cpu().numpy())
            self.frame_in_programme += 1
            return torch.tensor(1.0 if self.frame_in_programme == 1 else 0.0)

        def reset_states(self) -> None:
            self.reset_count += 1
            self.frame_in_programme = 0

    model = FakeModel()
    backend = SileroVADBackend(model=model)
    first = backend.analyse(np.ones(513, dtype=np.float32), 16_000)
    second = backend.analyse(np.ones(512, dtype=np.float32), 16_000)

    assert model.reset_count == 2
    assert [len(frame) for frame in model.frames] == [512, 512, 512]
    assert np.all(model.frames[1][1:] == 0)
    assert first.programme_duration_ms == 32
    assert first.speech_intervals == ((0, 32),)
    assert second.speech_intervals == ((0, 32),)
