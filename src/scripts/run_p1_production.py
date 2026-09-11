"""Hydra entry point for one supervised P1 production run."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from hviske.p1_supervisor import run_supervisor

logger = logging.getLogger("run_p1_production")


@hydra.main(config_path="../../config", config_name="p1_production", version_base=None)
def main(config: DictConfig) -> None:
    """Run all deterministic P1 partitions in one parent process.

    Raises:
        SystemExit:
            If a partition fails after its bounded retry budget.
    """
    result = run_supervisor(config=config)
    logger.info(
        "P1 production complete: completed=%s failed=%s interrupted=%s",
        result.completed,
        result.failed,
        result.interrupted,
    )
    if result.exit_code:
        raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
