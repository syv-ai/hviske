"""Hydra entry point for the native P1 segmentation pipeline."""

from __future__ import annotations

import json
import logging

import hydra
from omegaconf import DictConfig

from hviske.p1_pipeline import P1PreflightError, _safe_exception_category, run_pipeline

logger = logging.getLogger("build_p1_segments")

__all__ = ["P1PreflightError", "main", "run_pipeline"]


@hydra.main(config_path="../../config", config_name="p1_segments", version_base=None)
def main(config: DictConfig) -> None:
    """Run the native P1 pipeline using a Hydra-resolved configuration.

    Raises:
        SystemExit:
            If an unexpected failure occurs; the logged output contains only its
            stable category.
    """
    try:
        report = run_pipeline(config=config)
    except Exception as exc:
        logger.error("P1 run failed: %s", _safe_exception_category(exc))
        raise SystemExit(1) from None
    logger.info("P1 run complete: %s", json.dumps(report.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
