"""The Hviske project.

.. include:: ../../README.md
"""

import importlib.metadata
import logging

# Fetch the version of the package as defined in pyproject.toml.  Keep package import
# lightweight: model utilities import Transformers and must not run for P1 modules.
__version__ = importlib.metadata.version(__package__ or "")
__all__ = ["__version__", "block_terminal_output"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def block_terminal_output() -> None:
    """Apply the legacy output suppression settings on demand.

    The implementation is imported only when this compatibility export is called,
    keeping P1 imports free of model dependencies.
    """
    from .utils import block_terminal_output as implementation

    implementation()
