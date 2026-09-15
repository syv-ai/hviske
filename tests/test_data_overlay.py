"""Focused tests for reusable metadata-only ASR overlays."""

import concurrent.futures
import multiprocessing
import pickle

import pytest
from datasets import Dataset, IterableDataset

import hviske.data as data_module
from hviske.data import apply_dataset_overlay


def _consume_pickled_overlay(payload: bytes) -> list[str]:
    """Consume an overlay in a spawn child process.

    Returns:
        Text values emitted by the overlay.
    """
    dataset = pickle.loads(payload)
    return [row["text"] for row in dataset]


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
            apply_dataset_overlay(base, overlay, _config())


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

    result = list(apply_dataset_overlay(base, overlay, _config()))

    assert [row["text"] for row in result] == ["one", " revised "]
    assert [row["audio"] for row in result] == ["a", "b"]


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
