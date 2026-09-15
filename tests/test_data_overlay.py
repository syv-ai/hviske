"""Focused tests for reusable metadata-only ASR overlays."""

import concurrent.futures
import functools
import json
import multiprocessing
import pickle
import struct
import sys
import typing as t
import wave
from pathlib import Path

import pytest
from datasets import (
    Audio,
    Dataset,
    Features,
    IterableDataset,
    Value,
    interleave_datasets,
)
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

import hviske.data as data_module
from hviske.data import (
    _filter_dataset_rows,
    _standardise_training_dataset,
    apply_dataset_overlay,
    process_dataset,
)
from hviske.local_vtt import decode_vtt_audio, load_vtt_manifest


def _consume_pickled_overlay(payload: bytes) -> list[str]:
    """Consume an overlay in a spawn child process.

    Returns:
        Text values emitted by the overlay.
    """
    dataset = pickle.loads(payload)
    return [row["text"] for row in dataset]


def _first_spawn_batch(rows: list[dict[str, object]]) -> dict[str, object]:
    """Keep nested audio dictionaries intact in the spawn probe's batch.

    Returns:
        The first row in the batch.
    """
    return rows[0]


def _spawn_remote_rows() -> t.Iterator[dict[str, object]]:
    """Yield typed, Hub-shaped rows for the Linux spawn graph test."""
    for text in ("remote one", "remote two"):
        yield {
            "audio": {"array": [0.0] * 16_000, "sampling_rate": 16_000},
            "text": text,
            "language": "en",
            "source": "remote",
        }


def test_keyed_overlay_rejects_duplicate_keys() -> None:
    """Keyed indexes fail before a duplicate can silently overwrite a row."""
    base = Dataset.from_list([{"id": 1, "text": "one", "source": "demo"}])
    overlay = Dataset.from_list(
        [
            {
                "id": 1,
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
            {
                "id": 1,
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
        ]
    )

    with pytest.raises(ValueError, match="Duplicate overlay join key"):
        apply_dataset_overlay(
            base,
            overlay,
            _config(
                strategy="keyed",
                base_join_column="id",
                overlay_join_column="id",
                equality_checks={"text": "reference_text"},
            ),
        )


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "strategy": "positional",
        "base_filters": {"source": "demo"},
        "filters": {"source": "demo"},
        "equality_checks": {"source": "source", "text": "reference_text"},
        "action_column": "action",
        "allowed_actions": ["keep", "relabel", "strip"],
        "text_policy": {
            "candidates": [
                {"column": "new_text", "actions": ["relabel", "strip"]},
                {"column": "reference_text"},
            ]
        },
    }
    config.update(overrides)
    return config


def test_keyed_overlay_rejects_missing_and_mismatched_keys() -> None:
    """Keyed overlays report absent rows and incompatible key types."""
    base = Dataset.from_list([{"id": 2, "text": "two", "source": "demo"}])
    overlay = Dataset.from_list(
        [
            {
                "id": 1,
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            }
        ]
    )
    config = _config(
        strategy="keyed",
        base_join_column="id",
        overlay_join_column="id",
        equality_checks={"text": "reference_text"},
    )
    with pytest.raises(ValueError, match="missing base key"):
        list(apply_dataset_overlay(base, overlay, config))

    mismatched_types = Dataset.from_list(
        [
            {
                "id": "2",
                "source": "demo",
                "reference_text": "two",
                "new_text": None,
                "action": "keep",
            }
        ]
    )
    with pytest.raises(ValueError, match="key type mismatch"):
        list(apply_dataset_overlay(base, mismatched_types, config))


def test_overlay_consumers_are_pickleable_and_isolated() -> None:
    """Independent consumers can run concurrently without shared seen-key state."""
    base = Dataset.from_list(
        [
            {"id": 1, "source": "demo", "text": "one"},
            {"id": 2, "source": "demo", "text": "two"},
        ]
    )
    overlay = Dataset.from_list(
        [
            {
                "id": 1,
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
            {
                "id": 2,
                "source": "demo",
                "reference_text": "two",
                "new_text": "deux",
                "action": "relabel",
            },
        ]
    )
    overlaid = apply_dataset_overlay(
        base,
        overlay,
        _config(
            strategy="keyed",
            base_join_column="id",
            overlay_join_column="id",
            equality_checks={"text": "reference_text"},
        ),
    )

    payload = pickle.dumps(overlaid)
    pickle.loads(payload)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: list(overlaid), range(2)))

    assert [row["text"] for row in results[0]] == ["one", "deux"]
    assert [row["text"] for row in results[1]] == ["one", "deux"]

    context = multiprocessing.get_context("spawn")
    with context.Pool(1) as pool:
        assert pool.apply(_consume_pickled_overlay, (payload,)) == ["one", "deux"]


def test_overlay_filters_recognised_but_disallowed_review_action() -> None:
    """Recognised review rows are filtered without weakening positional joins."""
    base = Dataset.from_list(
        [{"source": "demo", "text": "one"}, {"source": "demo", "text": "two"}]
    )
    overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
            {
                "source": "demo",
                "reference_text": "two",
                "new_text": "",
                "action": "review",
            },
        ]
    )

    overlaid = apply_dataset_overlay(
        base,
        overlay,
        _config(recognised_actions=["keep", "relabel", "strip", "review"]),
    )

    assert list(overlaid) == [{"source": "demo", "text": "one"}]


def test_overlay_projects_columns_before_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay filters run after large unused columns are projected away."""
    observed_columns: list[list[str]] = []
    original_filter = data_module._filter_dataset_rows

    def spy_filter(dataset: Dataset, filters: dict[str, object]) -> Dataset:
        observed_columns.append(list(dataset.column_names or []))
        return original_filter(dataset=dataset, filters=filters)

    monkeypatch.setattr(data_module, "_filter_dataset_rows", spy_filter)
    base = Dataset.from_list([{"source": "demo", "text": "one"}])
    overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
                "unused_model_logits": [0.0] * 4096,
            }
        ]
    )

    result = list(apply_dataset_overlay(base, overlay, _config(base_filters=None)))

    assert observed_columns == [["action", "new_text", "reference_text", "source"]]
    assert result[0]["text"] == "one"


def test_overlay_projects_unused_columns_before_indexing() -> None:
    """Only join, action, equality, and text candidate columns are indexed."""
    base = Dataset.from_list([{"source": "demo", "text": "one"}])
    overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
                "unused_model_logits": [1.0, 2.0, 3.0],
            }
        ]
    )
    result = list(apply_dataset_overlay(base, overlay, _config()))

    assert result[0]["text"] == "one"
    assert "unused_model_logits" not in result[0]


def test_overlay_rejects_absent_configured_columns() -> None:
    """Configured equality and text columns are required before streaming."""
    base = Dataset.from_list([{"source": "demo", "text": "one"}])
    overlay = Dataset.from_list(
        [{"source": "demo", "new_text": None, "action": "keep"}]
    )

    with pytest.raises(ValueError, match="Missing overlay dataset columns"):
        apply_dataset_overlay(base, overlay, _config())


def test_overlay_rejects_missing_usable_text() -> None:
    """An allowed action with only blank candidates cannot create silent rows."""
    base = Dataset.from_list([{"source": "demo", "text": "one"}])
    overlay = Dataset.from_list(
        [{"source": "demo", "reference_text": " ", "new_text": None, "action": "keep"}]
    )

    with pytest.raises(ValueError, match="no usable rows"):
        apply_dataset_overlay(base, overlay, _config())


def test_overlay_rejects_null_and_unknown_actions() -> None:
    """Actions outside the configured recognised domain fail strictly."""
    base = Dataset.from_list([{"source": "demo", "text": "one"}])
    for action in [None, "mystery"]:
        overlay = Dataset.from_list(
            [
                {
                    "source": "demo",
                    "reference_text": "one",
                    "new_text": None,
                    "action": action,
                }
            ]
        )
        with pytest.raises(ValueError, match="expected one of"):
            apply_dataset_overlay(
                base,
                overlay,
                _config(
                    recognised_actions=[
                        "keep",
                        "relabel",
                        "strip",
                        "drop",
                        "flag",
                        "quarantine",
                        "review",
                    ]
                ),
            )


def test_overlay_rejects_retained_rows_without_text_among_usable_rows() -> None:
    """A bad retained row cannot be hidden by another usable overlay row."""
    base = Dataset.from_list(
        [{"source": "demo", "text": "one"}, {"source": "demo", "text": "two"}]
    )
    overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
            {
                "source": "demo",
                "reference_text": " ",
                "new_text": None,
                "action": "keep",
            },
        ]
    )

    with pytest.raises(ValueError, match="no usable text candidate"):
        apply_dataset_overlay(base, overlay, _config())


def test_positional_overlay_filters_and_uses_ordered_text_fallback() -> None:
    """Mirrored rows retain only allowed actions and select usable text."""
    base = Dataset.from_list(
        [
            {"source": "demo", "text": "one", "audio": "a"},
            {"source": "demo", "text": "two", "audio": "b"},
            {"source": "other", "text": "ignored", "audio": "c"},
        ]
    )
    overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
            {
                "source": "demo",
                "reference_text": "two",
                "new_text": " revised ",
                "action": "relabel",
            },
            {
                "source": "other",
                "reference_text": "ignored",
                "new_text": None,
                "action": "drop",
            },
        ]
    )

    overlaid = apply_dataset_overlay(base, overlay, _config())
    payload = pickle.dumps(overlaid)
    result = list(overlaid)
    restored = list(pickle.loads(payload))

    assert [row["text"] for row in result] == ["one", " revised "]
    assert [row["audio"] for row in result] == ["a", "b"]
    assert [row["text"] for row in restored] == ["one", " revised "]


def test_positional_overlay_infers_features_for_untyped_base() -> None:
    """Positional overlays retain exact rows from an untyped base stream."""
    base = IterableDataset.from_generator(
        lambda: iter(
            [
                {"source": "demo", "text": "one", "audio": "a"},
                {"source": "demo", "text": "two", "audio": "b"},
            ]
        )
    )
    overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            },
            {
                "source": "demo",
                "reference_text": "two",
                "new_text": "revised",
                "action": "relabel",
            },
        ]
    )

    result = list(apply_dataset_overlay(base, overlay, _config()))

    assert [row["text"] for row in result] == ["one", "revised"]
    assert [row["audio"] for row in result] == ["a", "b"]


def test_positional_overlay_rejects_length_and_equality_mismatches() -> None:
    """Strict positional overlays do not silently shift or truncate rows."""
    base = Dataset.from_list(
        [{"source": "demo", "text": "one"}, {"source": "demo", "text": "two"}]
    )
    short_overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "one",
                "new_text": None,
                "action": "keep",
            }
        ]
    )
    with pytest.raises(ValueError, match="length mismatch"):
        list(apply_dataset_overlay(base, short_overlay, _config()))

    mismatched_overlay = Dataset.from_list(
        [
            {
                "source": "demo",
                "reference_text": "wrong",
                "new_text": None,
                "action": "keep",
            },
            {
                "source": "demo",
                "reference_text": "two",
                "new_text": None,
                "action": "keep",
            },
        ]
    )
    with pytest.raises(ValueError, match="equality mismatch"):
        list(apply_dataset_overlay(base, mismatched_overlay, _config()))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker runtime regression")
def test_production_graph_feeds_spawn_dataloader(tmp_path: Path) -> None:
    """A production-shaped local/remote graph remains pickleable under spawn."""
    wav_path = tmp_path / "programme.wav"
    _write_spawn_wav(wav_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_wav_path": str(wav_path),
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "local",
                "id": "local",
                "language": "da",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    local = load_vtt_manifest(manifest_path, min_seconds=0.1, max_seconds=2.0)
    local_features = local.features.copy()
    local_features["audio"] = Audio(sampling_rate=16_000)
    local = local.map(
        function=functools.partial(decode_vtt_audio, sampling_rate=16_000),
        features=local_features,
    )
    local = t.cast(
        IterableDataset, _standardise_training_dataset(local, sampling_rate=16_000)
    )

    remote = IterableDataset.from_generator(
        _spawn_remote_rows,
        features=Features(
            audio=Audio(sampling_rate=16_000),
            text=Value("string"),
            language=Value("string"),
            source=Value("string"),
        ),
    )
    remote = t.cast(
        IterableDataset, _standardise_training_dataset(remote, sampling_rate=16_000)
    )
    remote = _filter_dataset_rows(dataset=remote, filters={"language": "en"})
    remote = _filter_dataset_rows(dataset=remote, filters={"language": "en"})
    overlay = Dataset.from_list(
        [
            {"reference_text": "remote one", "new_text": None, "action": "keep"},
            {
                "reference_text": "remote two",
                "new_text": "remote revised",
                "action": "relabel",
            },
        ]
    )
    remote = t.cast(
        IterableDataset,
        apply_dataset_overlay(
            base_dataset=remote,
            overlay_dataset=overlay,
            overlay_config={
                "strategy": "positional",
                "action_column": "action",
                "allowed_actions": ["keep", "relabel"],
                "text_policy": {
                    "candidates": [
                        {"column": "new_text", "actions": ["relabel"]},
                        {"column": "reference_text"},
                    ]
                },
            },
        ),
    )
    train = interleave_datasets(
        datasets=[local, remote],
        probabilities=[0.5, 0.5],
        seed=4242,
        stopping_strategy="all_exhausted",
    )
    train = process_dataset(
        dataset=train,
        lower_case=True,
        characters_to_keep=None,
        text_column="text",
        remove_input_dataset_columns=False,
        audio_column="audio",
        convert_numerals=False,
        normalise_audio=False,
        augment_audio=False,
        processor=None,
        language_column="language",
    )
    pickle.dumps(train)
    torch_dataset = t.cast(TorchIterableDataset[dict[str, object]], train)
    loader = DataLoader(
        torch_dataset,
        batch_size=1,
        num_workers=1,
        multiprocessing_context="spawn",
        collate_fn=_first_spawn_batch,
    )

    batches = list(loader)
    assert len(batches) >= 1
    assert all("audio" in batch and "text" in batch for batch in batches)


def _write_spawn_wav(path: Path) -> None:
    """Write a short mono WAV without requiring a fixture or network access."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(struct.pack("<16000h", *([0] * 16_000)))
