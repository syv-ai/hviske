"""Focused tests for the bounded finetuning-data preflight."""

import collections.abc as c
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
from datasets import Dataset
from omegaconf import OmegaConf

from scripts.preflight_finetuning_data import (
    _preflight_local_manifest,
    preflight_finetuning_data,
)


def test_local_preflight_accepts_valid_wav(tmp_path: Path) -> None:
    """A readable WAV with an in-bounds cue passes."""
    _preflight_local_manifest("local", _write_local_manifest(tmp_path))


def _write_local_manifest(
    tmp_path: Path, *, start: float = 0.0, end: float = 0.5
) -> Path:
    """Write a one-row local manifest for audio preflight tests.

    Returns:
        The generated manifest path.
    """
    audio_path = tmp_path / "audio.wav"
    soundfile.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_wav_path": str(audio_path),
                "start": start,
                "end": end,
                "text": "hej",
                "id": "local-1",
                "duration": end - start,
                "language": "da",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_local_preflight_rejects_cue_beyond_eof(tmp_path: Path) -> None:
    """A cue extending beyond the WAV is rejected."""
    manifest_path = _write_local_manifest(tmp_path, end=2.0)
    with pytest.raises(ValueError, match="extends beyond audio"):
        _preflight_local_manifest("local", manifest_path)


def test_local_preflight_rejects_zero_byte_wav(tmp_path: Path) -> None:
    """A zero-byte WAV is rejected before training starts."""
    manifest_path = _write_local_manifest(tmp_path)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")
    with pytest.raises(ValueError, match="Unreadable audio"):
        _preflight_local_manifest("local", manifest_path)


def test_preflight_consumes_at_most_one_row_per_source(tmp_path: Path) -> None:
    """Audio and validation stay bounded while transcripts are fully indexed."""
    audio_path = tmp_path / "audio.wav"
    soundfile.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_wav_path": str(audio_path),
                "start": 0.0,
                "end": 1.0,
                "text": "hej",
                "id": "local-1",
                "duration": 1.0,
                "language": "da",
            }
        )
        + "\nthis second row must not be parsed\n",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "org/gated-model",
                "revision": "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
            },
            "cache_dir": None,
            "datasets": {
                "p1": {
                    "id": "org/audio",
                    "subset": None,
                    "train_name": "train",
                    "text_column": "text",
                    "audio_column": "audio",
                    "audio_join_column": "recording_id",
                    "revision": "audio-sha",
                    "trust_remote_code": False,
                    "transcript_dataset_id": "org/transcripts",
                    "transcript_subset": None,
                    "transcript_split": "train",
                    "transcript_revision": "0123456789abcdef0123456789abcdef01234567",
                    "transcript_join_column": "recording_id",
                    "transcript_text_column": "text",
                    "transcript_trust_remote_code": False,
                },
                "local": {"type": "local_vtt", "manifest_path": str(manifest_path)},
            },
            "evaluation_datasets": [
                {
                    "id": "org/evaluation",
                    "subset": "en",
                    "val_name": "validation",
                    "text_column": "text",
                    "audio_column": "audio",
                    "revision": "evaluation-sha",
                    "trust_remote_code": False,
                }
            ],
        }
    )
    calls: list[dict[str, object]] = []

    def fake_loader(**kwargs: object) -> Dataset:
        calls.append(kwargs)
        path = kwargs["path"]
        if path == "org/audio":
            rows = [
                {"audio": [0.0], "recording_id": "one"},
                {"audio": [0.0], "recording_id": "two"},
            ]
        elif path == "org/transcripts":
            rows = [
                {"recording_id": "one", "text": "hello"},
                {"recording_id": "two", "text": "world"},
            ]
        else:
            rows = [
                {"audio": [0.0], "text": "hello"},
                {"audio": [0.0], "text": "world"},
            ]
        return Dataset.from_list(rows)

    api = FakeHubApi()
    preflight_finetuning_data(config=config, dataset_loader=fake_loader, hub_api=api)

    assert api.model_ids == [
        ("org/gated-model", "b1eacc2686a3d08ceaae5f24a88b1d519620bc09")
    ]
    assert len(calls) == 3
    assert [call["streaming"] for call in calls] == [True, False, True]
    assert all(call["trust_remote_code"] is False for call in calls)
    assert [call["revision"] for call in calls] == [
        "audio-sha",
        "0123456789abcdef0123456789abcdef01234567",
        "evaluation-sha",
    ]


class FakeHubApi:
    """Record authentication and model-access checks."""

    def __init__(self) -> None:
        """Initialise an empty model access log."""
        self.model_ids: list[tuple[str, str]] = []

    def model_info(self, repo_id: str, *, revision: str) -> object:
        """Record the model repository checked by the preflight.

        Returns:
            Placeholder model metadata.
        """
        self.model_ids.append((repo_id, revision))
        return object()

    def whoami(self) -> dict[str, object]:
        """Return a test identity."""
        return {"name": "tester"}


def test_preflight_rejects_missing_schema_without_consuming_a_second_row() -> None:
    """A schema error stops after the first streamed row."""
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "org/gated-model",
                "revision": "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
            },
            "cache_dir": None,
            "datasets": {
                "broken": {
                    "id": "org/broken",
                    "subset": None,
                    "train_name": "train",
                    "text_column": "text",
                    "audio_column": "audio",
                    "revision": "sha",
                    "trust_remote_code": False,
                }
            },
            "evaluation_datasets": [],
        }
    )
    dataset = CountingDataset(rows=[{"audio": object()}, {"text": "late"}])

    def fake_loader(**kwargs: object) -> CountingDataset:
        del kwargs
        return dataset

    with pytest.raises(ValueError, match="text"):
        preflight_finetuning_data(
            config=config, dataset_loader=fake_loader, hub_api=FakeHubApi()
        )

    assert dataset.consumed == 1


class CountingDataset:
    """Iterable that records how many rows a caller consumes."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        """Store rows without exposing a length-based materialisation path."""
        self.rows = rows
        self.consumed = 0

    def __iter__(self) -> c.Iterator[dict[str, object]]:
        """Yield rows while recording consumption."""
        for row in self.rows:
            self.consumed += 1
            yield row
