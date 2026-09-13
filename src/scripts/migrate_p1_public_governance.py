"""Safely migrate existing P1 metadata from private to public governance."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from p1_dataset.governance import migrate_public_governance
from p1_dataset.pipeline import PipelineSettings
from p1_dataset.publish import HfApiAdapter

logger = logging.getLogger(__name__)


def main() -> None:
    """Validate and optionally apply the one-shot governance migration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-old-head", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("config/p1_segments.yaml"))
    args = parser.parse_args()
    settings = PipelineSettings.from_config(
        cast(DictConfig, OmegaConf.load(args.config))
    )
    report = migrate_public_governance(
        api=HfApiAdapter(),
        settings=settings,
        expected_old_head=args.expected_old_head,
        apply=args.apply,
    )
    logger.info(
        "P1 governance migration applied=%s file_count=%d "
        "readme_sha256=%s license_sha256=%s commit_sha256=%s",
        report.applied,
        report.file_count,
        report.readme_sha256,
        report.license_sha256,
        report.commit_sha256 or "not-applied",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
