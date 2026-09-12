"""Regression tests for the renamed package namespace."""

import importlib.util
import pkgutil

import hviske
import p1_dataset


def test_coral_namespace_is_absent() -> None:
    """The old package namespace is no longer present."""
    assert importlib.util.find_spec("coral") is None


def test_hviske_has_no_p1_modules() -> None:
    """The ASR namespace does not contain P1 dataset modules."""
    assert not any(
        module.name.startswith("p1_")
        for module in pkgutil.iter_modules(hviske.__path__)
    )


def test_hviske_namespace_is_available() -> None:
    """The renamed package can be imported from its public namespace."""
    assert hviske.__name__ == "hviske"
    assert importlib.util.find_spec("hviske") is not None


def test_p1_dataset_namespace_is_available() -> None:
    """P1 dataset creation is available outside the ASR namespace."""
    assert p1_dataset.__name__ == "p1_dataset"
    assert importlib.util.find_spec("p1_dataset") is not None
