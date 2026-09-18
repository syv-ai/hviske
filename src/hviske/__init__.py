"""The Hviske project.

.. include:: ../../README.md
"""

import importlib.metadata

# Fetch the version of the package as defined in pyproject.toml. Keep package import
# lightweight by deferring model utilities until they are used.
__version__ = importlib.metadata.version(__package__ or "")
__all__ = ["__version__", "block_terminal_output"]


def block_terminal_output() -> None:
    """Apply the legacy output suppression settings on demand.

    The implementation is imported only when this compatibility export is called,
    keeping package imports free of model dependencies.
    """
    from .utils import block_terminal_output as implementation

    implementation()
