"""Safely migrate existing P1 metadata from private to public governance."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from p1_dataset.governance import migrate_public_governance
from p1_dataset.hub_diagnostics import classify_hub_error
from p1_dataset.pipeline import PipelineSettings
from p1_dataset.publish import HfApiAdapter

logger = logging.getLogger(__name__)


def main() -> int:
    """Validate and optionally apply the one-shot governance migration.

    Returns:
        Zero on success, otherwise one after a privacy-safe diagnostic.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-old-head", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("config/p1_segments.yaml"))
    args = parser.parse_args()
    try:
        settings = PipelineSettings.from_config(
            cast(DictConfig, OmegaConf.load(args.config))
        )
        report = migrate_public_governance(
            api=HfApiAdapter(),
            settings=settings,
            expected_old_head=args.expected_old_head,
            apply=args.apply,
        )
    except Exception as error:
        try:
            diagnostic = classify_hub_error(error)
        except Exception:
            diagnostic = None
        status = (
            str(diagnostic.status_code)
            if diagnostic is not None
            and diagnostic.status_code is not None
            and 100 <= diagnostic.status_code <= 599
            else "none"
        )
        phase = diagnostic.phase if diagnostic is not None else "unknown"
        reason = diagnostic.reason if diagnostic is not None else "unknown"
        logger.error(
            "P1 governance migration failed error_class=%s status=%s phase=%s "
            "reason=%s",
            _safe_class_name(type(error).__name__),
            status,
            _safe_diagnostic(phase),
            _safe_diagnostic(reason),
        )
        return 1
    logger.info(
        "P1 governance migration applied=%s file_count=%d "
        "readme_sha256=%s license_sha256=%s commit_sha256=%s",
        report.applied,
        report.file_count,
        report.readme_sha256,
        report.license_sha256,
        report.commit_sha256 or "not-applied",
    )
    return 0


def _safe_class_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        return value
    return "unknown"


def _safe_diagnostic(value: object) -> str:
    if isinstance(value, str) and re.fullmatch(r"[a-z_]{1,32}", value):
        return value
    return "unknown"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
