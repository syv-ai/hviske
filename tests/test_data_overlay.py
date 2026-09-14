"""Focused tests for reusable metadata-only ASR overlays."""

import pytest
from datasets import Dataset

from hviske.data import apply_dataset_overlay


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
