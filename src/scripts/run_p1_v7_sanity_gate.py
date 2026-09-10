"""Run the privacy-safe post-pilot P1 v7 dozen-clip sanity gate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import typing as t
from pathlib import Path

from hviske.p1_publish import HfApiAdapter
from hviske.p1_v7_sanity_gate import (
    PILOT_REPOSITORY,
    WhisperSmallASR,
    run_v7_sanity_gate,
)
from hviske.p1_validation import PinnedHubClipRetriever

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run one post-pilot gate and return a shell-friendly status code.

    Returns:
        Zero when the aggregate gate passes, otherwise one.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate exactly 12 accepted P1 v7 clips after pilot publication. "
            "The report contains aggregate WER only."
        ),
        epilog=(
            "Pass requires all structural checks, no empty/non-speech ASR, "
            "median WER <= 0.5, and at most 3 samples with WER > 0.8."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scratch/audit-candidates.jsonl"),
        help="pilot audit candidates JSONL (default: scratch/audit-candidates.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("scratch/p1-v7-sanity-gate.json"),
        help="aggregate JSON report destination",
    )
    parser.add_argument(
        "--pilot-head", required=True, help="complete immutable final pilot commit SHA"
    )
    parser.add_argument(
        "--seed", default="p1-v7-dozen", help="deterministic stratified sampling seed"
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Transformers device index (default: CUDA 0 when available, otherwise CPU)"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        candidates = _read_jsonl(args.input)
        retriever = PinnedHubClipRetriever(
            HfApiAdapter(), repository=PILOT_REPOSITORY, revision=args.pilot_head
        )
        report = run_v7_sanity_gate(
            candidates,
            retriever=retriever,
            asr=WhisperSmallASR(device=_device(args.device)),
            pilot_head=args.pilot_head,
            seed=args.seed,
            report_path=args.report,
        )
    except Exception:
        logger.error("P1 v7 sanity gate could not run")
        return 1
    if not report.get("pass", False):
        logger.error("P1 v7 sanity gate failed")
        return 1
    logger.info("P1 v7 sanity gate passed")
    return 0


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


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
