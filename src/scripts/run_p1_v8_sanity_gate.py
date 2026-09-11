"""Run the privacy-safe, model-free P1 v8 structural sanity gate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import typing as t
from pathlib import Path

from hviske.p1_publish import HfApiAdapter
from hviske.p1_v8_sanity_gate import (
    PILOT_REPOSITORY,
    PIPELINE_VERSION,
    run_v8_sanity_gate,
)
from hviske.p1_validation import PinnedHubClipRetriever

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run one structural gate and return a shell-friendly status code.

    Returns:
        Zero when the aggregate gate passes, otherwise one.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate exactly 12 accepted P1 v8 clips after publication. "
            "The gate is structural and model-free."
        ),
        epilog=(
            "Pass requires private immutable retrieval, matching hashes, the exact "
            "schema and audio encoding, one v8 digest, and single-speaker anchors."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scratch/audit-candidates.jsonl"),
        help="audit candidates JSONL (default: scratch/audit-candidates.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("scratch/p1-v8-sanity-gate.json"),
        help="aggregate JSON report destination",
    )
    parser.add_argument(
        "--pilot-head", required=True, help="complete immutable publication commit SHA"
    )
    parser.add_argument(
        "--seed", default="p1-v8-dozen", help="deterministic stratified sampling seed"
    )
    parser.add_argument(
        "--pipeline-config-sha256",
        help=(
            "expected active pipeline configuration digest; defaults to audit evidence"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        candidates = _read_jsonl(args.input)
        digest = args.pipeline_config_sha256
        if digest is None:
            candidate_digests = {
                value
                for candidate in candidates
                if isinstance(value := candidate.get("pipeline_config_sha256"), str)
            }
            digest = (
                next(iter(candidate_digests)) if len(candidate_digests) == 1 else None
            )
        retriever = PinnedHubClipRetriever(
            HfApiAdapter(),
            repository=PILOT_REPOSITORY,
            revision=args.pilot_head,
            expected_pipeline_version=PIPELINE_VERSION,
            expected_pipeline_config_sha256=digest,
        )
        report = run_v8_sanity_gate(
            candidates,
            retriever=retriever,
            pilot_head=args.pilot_head,
            seed=args.seed,
            report_path=args.report,
            expected_pipeline_version=PIPELINE_VERSION,
            expected_pipeline_config_sha256=digest,
        )
    except Exception:
        logger.error("P1 v8 sanity gate could not run")
        return 1
    if not report.get("pass", False):
        logger.error("P1 v8 sanity gate failed")
        return 1
    logger.info("P1 v8 sanity gate passed")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("candidate records must be JSON objects")
                records.append(t.cast(dict[str, object], value))
    return records


if __name__ == "__main__":
    sys.exit(main())
