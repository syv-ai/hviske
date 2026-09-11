"""Hydra entry point for one supervised P1 production run."""

from __future__ import annotations

import logging
import os

import hydra
from huggingface_hub import get_token
from omegaconf import DictConfig

from hviske.p1_source import harden_p1_logging
from hviske.p1_supervisor import run_supervisor

logger = logging.getLogger("run_p1_production")


@hydra.main(config_path="../../config", config_name="p1_production", version_base=None)
def main(config: DictConfig) -> None:
    """Run all deterministic P1 partitions in one parent process.

    Raises:
        SystemExit:
            If a partition fails after its bounded retry budget.
    """
    token = get_token()
    if not token:
        raise SystemExit("P1 production requires an authenticated Hub session")
    os.environ["HF_TOKEN"] = token
    harden_p1_logging()
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
