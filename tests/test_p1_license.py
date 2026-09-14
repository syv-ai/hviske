"""Tests for the codebase licence."""

from pathlib import Path


def test_codebase_uses_mit_license_without_dataset_license() -> None:
    """The root licence covers the software rather than the P1 dataset."""
    root = Path(__file__).parents[1]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026- syv.ai ApS" in license_text
    assert "Copyright (c) 2023-2024 Alexandra Instituttet A/S" in license_text
    assert "Permission is hereby granted" in license_text
    assert "DR P1 SPEECH SEGMENTS" not in license_text
    assert not (root / "LICENSE-DATASET").exists()
