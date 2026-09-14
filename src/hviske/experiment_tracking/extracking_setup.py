"""This module contains the base class for an experiment tracking setup."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig


class ExTrackingSetup(ABC):
    """Base class for an experiment tracking setup."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the experiment tracking setup.

        Args:
            config:
                The configuration object.
        """
        self.config = config

    @abstractmethod
    def run_finalization(self, exit_code: int = 0) -> None:
        """Finalise the tracking run with its process exit status.

        Args:
            exit_code (optional):
                The process exit code to report. Defaults to ``0``.
        """

    @abstractmethod
    def run_initialization(self) -> None:
        """Run the initialization of the experiment tracking setup.

        Returns:
            True if the initialization was successful, False otherwise.
        """
