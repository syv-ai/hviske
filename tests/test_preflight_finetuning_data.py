"""Focused tests for the bounded finetuning-data preflight."""

import collections.abc as c
import json
import typing as t
from pathlib import Path

import numpy as np
import pytest
import soundfile
from datasets import Audio, Dataset, Features, IterableDataset, Value
from omegaconf import DictConfig, OmegaConf

import scripts.preflight_finetuning_data as preflight_module
from scripts.preflight_finetuning_data import (
    _preflight_local_manifest,
    preflight_finetuning_data,
)


def test_grouped_positional_preflight_loads_and_consumes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent source views share one strict base/overlay consumption."""
    config = _grouped_config()
    base = Dataset.from_list(
        [
            {"audio": [0.0], "text": "one", "source": "one"},
            {"audio": [0.0], "text": "two", "source": "two"},
            {"audio": [0.0], "text": "one", "source": "one"},
            {"audio": [0.0], "text": "two", "source": "two"},
        ]
    )
    overlay = Dataset.from_list(
        [
            {"action": "keep", "source": "one", "reference_text": "one"},
            {"action": "keep", "source": "two", "reference_text": "two"},
            {"action": "keep", "source": "one", "reference_text": "one"},
            {"action": "keep", "source": "two", "reference_text": "two"},
        ]
    )
    calls: list[dict[str, object]] = []
    consumption_count = 0
    real_apply = preflight_module.apply_dataset_overlay

    def fake_loader(**kwargs: object) -> Dataset:
        calls.append(kwargs)
        return base if kwargs["path"] == "org/audio" else overlay

    def counting_overlay(**kwargs: object) -> c.Iterator[object]:
        nonlocal consumption_count
        result = real_apply(
            base_dataset=t.cast(Dataset | IterableDataset, kwargs["base_dataset"]),
            overlay_dataset=t.cast(Dataset, kwargs["overlay_dataset"]),
            overlay_config=t.cast(c.Mapping[str, object], kwargs["overlay_config"]),
        )

        def stream() -> c.Iterator[object]:
            nonlocal consumption_count
            consumption_count += 1
            yield from result

        return stream()

    before = [config.datasets[name].filters.copy() for name in ("one", "two")]
    monkeypatch.setattr(preflight_module, "apply_dataset_overlay", counting_overlay)
    preflight_finetuning_data(
        config=config, dataset_loader=fake_loader, hub_api=FakeHubApi()
    )

    assert [call["streaming"] for call in calls] == [True, False]
    assert consumption_count == 1
    assert [config.datasets[name].filters for name in ("one", "two")] == before


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


def _grouped_config() -> DictConfig:
    """Build a minimal two-view training configuration.

    Returns:
        An OmegaConf configuration object.
    """
    return OmegaConf.create(
        {
            "model": {"pretrained_model_id": "org/model", "revision": "model-sha"},
            "cache_dir": None,
            "datasets": {
                "one": _grouped_overlay_config("one"),
                "two": _grouped_overlay_config("two"),
            },
            "evaluation_datasets": [],
        }
    )


def _grouped_overlay_config(value: str) -> dict[str, object]:
    """Build one source config for grouped positional-preflight tests.

    Returns:
        A source configuration mapping.
    """
    return {
        "id": "org/audio",
        "subset": "default",
        "train_name": "train",
        "text_column": "text",
        "audio_column": "audio",
        "filters": {"source": value},
        "revision": "audio-sha",
        "trust_remote_code": False,
        "overlay": {
            "id": "org/overlay",
            "subset": "default",
            "split": "train",
            "revision": "1" * 40,
            "strategy": "positional",
            "base_filters": {"source": value},
            "filters": {"source": value},
            "equality_checks": {"source": "source", "text": "reference_text"},
            "action_column": "action",
            "allowed_actions": ["keep", "relabel", "strip"],
            "recognised_actions": [
                "keep",
                "relabel",
                "strip",
                "drop",
                "flag",
                "quarantine",
            ],
            "text_policy": {"candidates": [{"column": "reference_text"}]},
            "trust_remote_code": False,
        },
    }


def test_grouped_positional_preflight_requires_each_source_to_emit_a_row() -> None:
    """A valid global overlay cannot hide an empty configured source view."""
    config = _grouped_config()
    base = Dataset.from_list(
        [
            {"audio": [0.0], "text": "one", "source": "one"},
            {"audio": [0.0], "text": "two", "source": "two"},
        ]
    )
    overlay = Dataset.from_list(
        [
            {"action": "keep", "source": "one", "reference_text": "one"},
            {"action": "drop", "source": "two", "reference_text": "two"},
        ]
    )

    def fake_loader(**kwargs: object) -> Dataset:
        return base if kwargs["path"] == "org/audio" else overlay

    with pytest.raises(ValueError, match="two.*no accepted rows"):
        preflight_finetuning_data(
            config=config, dataset_loader=fake_loader, hub_api=FakeHubApi()
        )


def test_grouped_positional_preflight_surfaces_length_mismatch() -> None:
    """The shared path retains strict positional length validation."""
    config = _grouped_config()
    base = Dataset.from_list(
        [
            {"audio": [0.0], "text": "one", "source": "one"},
            {"audio": [0.0], "text": "two", "source": "two"},
        ]
    )
    overlay = Dataset.from_list(
        [{"action": "keep", "source": "one", "reference_text": "one"}]
    )

    def fake_loader(**kwargs: object) -> Dataset:
        return base if kwargs["path"] == "org/audio" else overlay

    with pytest.raises(ValueError, match="length mismatch"):
        preflight_finetuning_data(
            config=config, dataset_loader=fake_loader, hub_api=FakeHubApi()
        )


def test_ineligible_positional_views_use_independent_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-column filters do not enter the shared path."""
    config = _grouped_config()
    config.datasets.two.filters = {"source": "two", "language": "da"}
    handled: list[str] = []

    def fake_preflight(**kwargs: object) -> None:
        handled.append(str(kwargs["source_name"]))

    monkeypatch.setattr(preflight_module, "_preflight_hub_source", fake_preflight)
    preflight_finetuning_data(config=config, hub_api=FakeHubApi())

    assert handled == ["one", "two"]


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


def test_overlay_preflight_disables_audio_decoding_before_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay preflight removes audio before filtering and joining."""
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "org/gated-model",
                "revision": "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
            },
            "cache_dir": None,
            "datasets": {
                "overlaid": {
                    "id": "org/audio",
                    "subset": None,
                    "train_name": "train",
                    "text_column": "text",
                    "audio_column": "audio",
                    "filters": {"source": "demo"},
                    "revision": "audio-sha",
                    "trust_remote_code": False,
                    "overlay": {
                        "id": "org/overlay",
                        "revision": "1" * 40,
                        "strategy": "keyed",
                        "base_filters": {"metadata": "keep"},
                        "equality_checks": {"text": "reference_text"},
                        "base_join_column": "recording_id",
                    },
                }
            },
            "evaluation_datasets": [],
        }
    )
    dataset = Dataset.from_dict(
        {
            "audio": [{"bytes": b"metadata-only", "path": None}],
            "text": ["hello"],
            "source": ["demo"],
            "metadata": ["keep"],
            "recording_id": ["recording-1"],
        },
        features=Features(
            {
                "audio": Audio(),
                "text": Value("string"),
                "source": Value("string"),
                "metadata": Value("string"),
                "recording_id": Value("string"),
            }
        ),
    )
    observed: list[tuple[bool, set[str]]] = []

    def fake_loader(**kwargs: object) -> Dataset:
        if kwargs["path"] == "org/audio":
            return dataset
        return Dataset.from_list([{"action": "keep", "reference_text": "hello"}])

    def fake_overlay(**kwargs: object) -> Dataset:
        overlaid = kwargs["base_dataset"]
        assert isinstance(overlaid, Dataset)
        observed.append(("audio" in overlaid.column_names, set(overlaid.column_names)))
        return overlaid

    monkeypatch.setattr(preflight_module, "apply_dataset_overlay", fake_overlay)
    preflight_finetuning_data(
        config=config, dataset_loader=fake_loader, hub_api=FakeHubApi()
    )

    assert observed == [(False, {"text", "source", "metadata", "recording_id"})]


def test_overlay_preflight_rejects_missing_source_audio_column() -> None:
    """Overlay sources must validate audio schema before loading overlay rows."""
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "org/model",
                "revision": "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
            },
            "datasets": {
                "overlaid": {
                    "id": "org/audio",
                    "subset": None,
                    "train_name": "train",
                    "text_column": "text",
                    "audio_column": "audio",
                    "revision": "audio-sha",
                    "overlay": {"id": "org/overlay", "revision": "1" * 40},
                }
            },
            "evaluation_datasets": [],
        }
    )
    overlay_loaded = False

    def fake_loader(**kwargs: object) -> Dataset:
        nonlocal overlay_loaded
        if kwargs["path"] == "org/overlay":
            overlay_loaded = True
        return Dataset.from_list([{"text": "hello"}])

    with pytest.raises(ValueError, match="Missing audio column"):
        preflight_finetuning_data(
            config=config, dataset_loader=fake_loader, hub_api=FakeHubApi()
        )
    assert not overlay_loaded


def test_preflight_bounds_non_overlay_source_consumption(tmp_path: Path) -> None:
    """Ordinary audio validation stays bounded while transcripts are fully indexed."""
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


@pytest.mark.parametrize("revision", ["", "main", "0123456", "g" * 40])
def test_preflight_rejects_missing_or_invalid_immutable_source_revision(
    revision: str,
) -> None:
    """Preflight requires configured environment-backed source revisions."""
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "org/gated-model",
                "revision": "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
            },
            "cache_dir": None,
            "datasets": {
                "p1": {
                    "id": "syvai/p1-segments",
                    "subset": None,
                    "train_name": "train",
                    "text_column": "text",
                    "audio_column": "audio",
                    "revision": revision,
                    "immutable_revision_env": "P1_SEGMENTS_REVISION",
                }
            },
            "evaluation_datasets": [],
        }
    )

    with pytest.raises(ValueError, match="P1_SEGMENTS_REVISION"):
        preflight_finetuning_data(
            config=config, dataset_loader=lambda **kwargs: None, hub_api=FakeHubApi()
        )


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
