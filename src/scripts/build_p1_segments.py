"""Hydra entry point for the native P1 segmentation pipeline."""

from __future__ import annotations

import json
import logging

import hydra
from omegaconf import DictConfig

from hviske.p1_pipeline import P1PreflightError, run_pipeline

logger = logging.getLogger("build_p1_segments")

__all__ = ["P1PreflightError", "main", "run_pipeline"]


@hydra.main(config_path="../../config", config_name="p1_segments", version_base=None)
def main(config: DictConfig) -> None:
    """Run the native P1 pipeline using a Hydra-resolved configuration."""
    report = run_pipeline(config=config)
    logger.info("P1 run complete: %s", json.dumps(report.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
