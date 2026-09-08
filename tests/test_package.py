"""Regression tests for the renamed package namespace."""

import importlib.util

import hviske


def test_hviske_namespace_is_available() -> None:
    """The renamed package can be imported from its public namespace."""
    assert hviske.__name__ == "hviske"
    assert importlib.util.find_spec("hviske") is not None


def test_coral_namespace_is_absent() -> None:
    """The old package namespace is no longer present."""
    assert importlib.util.find_spec("coral") is None
