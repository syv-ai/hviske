"""Tests for bounded Hub shard loading and pre-decode shuffling."""

import typing as t

import pytest
from datasets import Audio, Dataset, Features, IterableDataset, Value
from omegaconf import OmegaConf

import hviske.data as data_module
from hviske.data import (
    _load_transcript_dataset,
    _resolve_hub_data_files,
    _validate_positional_shard_parity,
    load_data_for_finetuning,
)

REVISION = "1" * 40


def test_hub_shard_range_rejects_empty_selection() -> None:
    """A valid range still fails when no pinned filename exists."""
    with pytest.raises(ValueError, match="resolved no existing files"):
        _resolve_hub_data_files(
            dataset_id="organisation/dataset",
            revision=REVISION,
            selection={
                "template": "data/train-{shard:05d}.parquet",
                "start": 1,
                "end": 2,
            },
            hub_api=FakeHubApi(["data/train-00000.parquet"]),
        )


class FakeHubApi:
    """Return a fixed repository tree and record pinned lookups."""

    def __init__(self, files: list[str]) -> None:
        """Store the repository-relative files returned by the fake."""
        self.files = files
        self.calls: list[tuple[str, str, str]] = []

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[str]:
        """Return configured repository-relative paths."""
        self.calls.append((repo_id, repo_type, revision))
        return self.files


def test_hub_shard_range_resolves_only_existing_pinned_files() -> None:
    """Unrelated files and a missing in-range shard never reach the loader."""
    api = FakeHubApi(
        [
            "data/train-00000.parquet",
            "data/train-00001.parquet",
            "data/train-00003.parquet",
            "data/train-00004.parquet",
            "data/train-00005.parquet",
            "README.md",
        ]
    )

    resolved = _resolve_hub_data_files(
        dataset_id="organisation/dataset",
        revision=REVISION,
        selection={"template": "data/train-{shard:05d}.parquet", "start": 1, "end": 4},
        hub_api=api,
    )

    assert resolved == [
        "data/train-00001.parquet",
        "data/train-00003.parquet",
        "data/train-00004.parquet",
    ]
    assert resolved is not None
    assert api.calls == [("organisation/dataset", "dataset", REVISION)]
    assert all("?" not in path and "://" not in path for path in resolved)

    calls: list[dict[str, object]] = []

    def loader(**kwargs: object) -> Dataset:
        calls.append(kwargs)
        return Dataset.from_dict({"text": ["one"]})

    _load_transcript_dataset(
        dataset_id="organisation/dataset",
        subset="default",
        split="train",
        revision=REVISION,
        cache_dir=None,
        dataset_loader=loader,
        data_files=resolved,
    )
    assert calls[0]["data_files"] == resolved


@pytest.mark.parametrize(
    ("selection", "revision", "message"),
    [
        (None, None, None),
        (
            {"template": "data/{shard}.parquet", "start": -1, "end": 2},
            REVISION,
            "bounds",
        ),
        (
            {"template": "data/{shard}.parquet", "start": 3, "end": 2},
            REVISION,
            "bounds",
        ),
        (
            {"template": "data/{other}.parquet", "start": 0, "end": 2},
            REVISION,
            "template",
        ),
        (
            {"template": "https://host/{shard}", "start": 0, "end": 2},
            REVISION,
            "template",
        ),
        (
            {"template": "data/{shard}.parquet", "start": 0, "end": 2},
            "main",
            "immutable",
        ),
    ],
)
def test_hub_shard_range_validates_configuration(
    selection: dict[str, object] | None, revision: str | None, message: str | None
) -> None:
    """Malformed templates, bounds, and mutable revisions fail before listing."""
    if selection is None:
        assert (
            _resolve_hub_data_files(
                dataset_id="organisation/dataset",
                revision=revision,
                selection=selection,
                hub_api=FakeHubApi([]),
            )
            is None
        )
        return
    with pytest.raises(ValueError, match=t.cast(str, message)):
        _resolve_hub_data_files(
            dataset_id="organisation/dataset",
            revision=revision,
            selection=selection,
            hub_api=FakeHubApi(["data/0.parquet"]),
        )


def test_positional_shard_parity_tolerates_only_mirrored_gaps() -> None:
    """Mirrored missing numbers pass, while a missing or reordered side fails."""
    mirrored = ["data/train-00001.parquet", "data/train-00003.parquet"]
    _validate_positional_shard_parity(
        base_files=mirrored,
        overlay_files=["overlay/train-00001.parquet", "overlay/train-00003.parquet"],
    )

    with pytest.raises(ValueError, match="basenames/orders"):
        _validate_positional_shard_parity(
            base_files=mirrored,
            overlay_files=[
                "overlay/train-00003.parquet",
                "overlay/train-00001.parquet",
            ],
        )
    with pytest.raises(ValueError, match="both base and overlay"):
        _validate_positional_shard_parity(base_files=mirrored, overlay_files=None)


@pytest.mark.parametrize("strategy", ["positional", "keyed"])
def test_streaming_shuffle_precedes_audio_decode_after_overlay(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """Overlay joins and shuffle operate on decode-free metadata."""
    events: list[str] = []
    original_cast = IterableDataset.cast_column
    original_shuffle = IterableDataset.shuffle

    def recording_cast(
        self: IterableDataset, column: str, feature: Audio
    ) -> IterableDataset:
        events.append("decode" if feature.decode else "metadata")
        return original_cast(self, column=column, feature=feature)

    def recording_shuffle(
        self: IterableDataset, seed: int | None = None, buffer_size: int = 1000
    ) -> IterableDataset:
        events.append("shuffle")
        return original_shuffle(self, seed=seed, buffer_size=buffer_size)

    base = IterableDataset.from_generator(
        lambda: iter(
            [
                {
                    "audio": {"path": "unused.wav", "bytes": None},
                    "text": "original",
                    "source": "source",
                }
            ]
        ),
        features=Features(
            audio=Audio(decode=False), text=Value("string"), source=Value("string")
        ),
    )
    overlay = Dataset.from_list(
        [
            {
                "source": "source",
                "reference_text": "original",
                "new_text": "revised",
                "action": "relabel",
            }
        ]
    )

    monkeypatch.setattr(IterableDataset, "cast_column", recording_cast)
    monkeypatch.setattr(IterableDataset, "shuffle", recording_shuffle)
    monkeypatch.setattr(data_module, "load_dataset", lambda **_: base)
    monkeypatch.setattr(data_module, "_load_transcript_dataset", lambda **_: overlay)

    config = OmegaConf.create(
        {
            "seed": 7,
            "dataset_probabilities": [1.0],
            "datasets": {
                "source": {
                    "id": "organisation/base",
                    "subset": "default",
                    "train_name": "train",
                    "text_column": "text",
                    "audio_column": "audio",
                    "filter_dataset": True,
                    "filters": {"source": "source"},
                    "language": "da",
                    "revision": REVISION,
                    "overlay": {
                        "id": "organisation/overlay",
                        "revision": REVISION,
                        "strategy": strategy,
                        "base_join_column": "source",
                        "overlay_join_column": "source",
                        "base_filters": {"source": "source"},
                        "filters": {"source": "source"},
                        "equality_checks": {
                            "source": "source",
                            "text": "reference_text",
                        },
                        "action_column": "action",
                        "allowed_actions": ["keep", "relabel"],
                        "text_policy": {
                            "candidates": [
                                {"column": "new_text", "actions": ["relabel"]},
                                {"column": "reference_text"},
                            ]
                        },
                    },
                }
            },
            "streaming": True,
            "cache_dir": None,
            "shuffle_buffer_size": 1,
            "dataset_num_workers": 1,
            "min_seconds_per_example": 0.1,
            "max_seconds_per_example": 10.0,
            "model": {
                "sampling_rate": 16_000,
                "lower_case": True,
                "characters_to_keep": None,
                "language": "da",
                "punctuation": True,
            },
            "evaluation_datasets": [],
            "evaluation_lower_case": True,
            "evaluation_characters_to_keep": None,
        }
    )

    if strategy == "keyed":
        config.datasets.source.data_file_shards = {
            "template": "data/base-{shard}.parquet",
            "start": 0,
            "end": 0,
        }
        config.datasets.source.overlay.data_file_shards = {
            "template": "data/overlay-{shard}.parquet",
            "start": 1,
            "end": 1,
        }
        monkeypatch.setattr(
            data_module,
            "_resolve_hub_data_files",
            lambda dataset_id, revision, selection: [
                "data/base-000.parquet"
                if dataset_id == "organisation/base"
                else "data/overlay-001.parquet"
            ],
        )

    dataset = load_data_for_finetuning(config=config)

    assert events == ["metadata", "shuffle", "decode"]
    assert dataset["train"] is not None
