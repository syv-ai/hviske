"""Sparkie entry point for reproducible P1 v8 finalisation."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=(
        r'^Field name "schema" in "OutputEncodingContract" shadows '
        r'an attribute in parent "ContractModel"$'
    ),
    category=UserWarning,
    module=r"p1_dataset\.contracts",
)

from p1_dataset.finalisation import finalise_p1_corpus  # noqa: E402
from p1_dataset.publish import HfApiAdapter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Validate the completed private corpus without exposing sensitive diagnostics.

    Returns:
        Zero when validation succeeds, otherwise one.
    """
    parser = argparse.ArgumentParser(
        description="Validate the completed private syvai/p1-segments v8 corpus."
    )
    parser.add_argument("--revision", required=True, help="pinned 40-character HEAD")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--pipeline-config-sha256", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repository", default="syvai/p1-segments")
    parser.add_argument(
        "--update-card",
        action="store_true",
        help="explicitly CAS-add final structural statistics to README.md",
    )
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        report = finalise_p1_corpus(
            hub=HfApiAdapter(),
            run_root=args.run_root,
            revision=args.revision,
            expected_pipeline_config_sha256=args.pipeline_config_sha256,
            repository=args.repository,
            report_path=args.report,
            update_card=args.update_card,
        )
    except Exception:
        return 1
    # Deliberately emit only a status; the aggregate report is the machine output.
    sys.stdout.write("P1 finalisation passed\n")
    if args.update_card:
        head = report.get("final_head")
        if isinstance(head, str):
            sys.stdout.write(f"P1 card update committed head={head}\n")
    sys.stdout.flush()
    del report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
