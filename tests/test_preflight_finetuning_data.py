"""Focused tests for the bounded finetuning-data preflight."""

import collections.abc as c
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.preflight_finetuning_data import preflight_finetuning_data


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


class FakeHubApi:
    """Record authentication and model-access checks."""

    def __init__(self) -> None:
        """Initialise an empty model access log."""
        self.model_ids: list[str] = []

    def whoami(self) -> dict[str, object]:
        """Return a test identity."""
        return {"name": "tester"}

    def model_info(self, repo_id: str) -> object:
        """Record the model repository checked by the preflight.

        Returns:
            Placeholder model metadata.
        """
        self.model_ids.append(repo_id)
        return object()


def test_preflight_consumes_at_most_one_row_per_source(tmp_path: Path) -> None:
    """Hub streams and local manifests remain bounded to their first row."""
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
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
            "model": {"pretrained_model_id": "org/gated-model"},
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
                    "transcript_revision": "transcript-sha",
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
    datasets: list[CountingDataset] = []
    calls: list[dict[str, object]] = []

    def fake_loader(**kwargs: object) -> CountingDataset:
        calls.append(kwargs)
        rows: list[dict[str, object]]
        path = kwargs["path"]
        if path == "org/audio":
            rows = [
                {"audio": object(), "recording_id": "one"},
                {"audio": object(), "recording_id": "two"},
            ]
        elif path == "org/transcripts":
            rows = [
                {"recording_id": "one", "text": "hello"},
                {"recording_id": "two", "text": "world"},
            ]
        else:
            rows = [
                {"audio": object(), "text": "hello"},
                {"audio": object(), "text": "world"},
            ]
        dataset = CountingDataset(rows=rows)
        datasets.append(dataset)
        return dataset

    api = FakeHubApi()
    preflight_finetuning_data(config=config, dataset_loader=fake_loader, hub_api=api)

    assert api.model_ids == ["org/gated-model"]
    assert len(datasets) == 3
    assert all(dataset.consumed == 1 for dataset in datasets)
    assert all(call["streaming"] is True for call in calls)
    assert all(call["trust_remote_code"] is False for call in calls)
    assert [call["revision"] for call in calls] == [
        "audio-sha",
        "transcript-sha",
        "evaluation-sha",
    ]


def test_preflight_rejects_missing_schema_without_consuming_a_second_row() -> None:
    """A schema error stops after the first streamed row."""
    config = OmegaConf.create(
        {
            "model": {"pretrained_model_id": "org/gated-model"},
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
