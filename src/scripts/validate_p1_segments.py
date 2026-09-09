"""Generate P1 validation reports and perform an explicit one-item audit review."""

from __future__ import annotations

import argparse
import json
import logging
import typing as t
from pathlib import Path

import pyarrow.parquet as pq

from hviske.p1_publish import HfApiAdapter
from hviske.p1_validation import (
    ClipRetriever,
    MetadataLedger,
    PinnedHubClipRetriever,
    build_quality_report,
    create_blinded_audit_manifest,
    iter_bounded,
    persist_audit_candidates,
    review_one_clip,
)

logger = logging.getLogger("hviske_p1_validation")


def main() -> None:
    """Run report generation or the explicitly requested single-item review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="metadata JSONL exported from the remote split"
    )
    parser.add_argument("--database", type=Path, default=Path("p1-validation.sqlite"))
    parser.add_argument(
        "--max-rows", type=int, help="bound report generation to this many input rows"
    )
    parser.add_argument(
        "--report", type=Path, help="write the structural report as JSON"
    )
    parser.add_argument(
        "--manifest", type=Path, help="write or read a blinded audit manifest"
    )
    parser.add_argument("--sample-seed", default="0")
    parser.add_argument("--accepted-quota", type=int, default=200)
    parser.add_argument("--rejected-quota", type=int, default=100)
    parser.add_argument("--borderline-quota", type=int, default=0)
    parser.add_argument("--review-id", help="review exactly this manifest audit_id")
    parser.add_argument(
        "--hub-repo", help="pinned Hub dataset repository for one-item reviews"
    )
    parser.add_argument("--hub-revision", help="complete 40-character Hub commit SHA")
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="optional local Parquet staging root for offline smoke tests",
    )
    parser.add_argument(
        "--decision",
        choices=("accepted", "rejected", "borderline"),
        help="explicit decision for the one-item review",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.review_id:
        _run_review(args)
        return
    if args.input is None:
        parser.error("--input is required when generating a report")
    input_rows = _read_jsonl(args.input)
    if args.max_rows is not None:
        input_rows = iter_bounded(input_rows, max_rows=args.max_rows)
    report = build_quality_report(rows=input_rows, database=args.database)
    if args.report:
        _write_json(args.report, report)
    else:
        logger.info(
            "P1 quality report generated: %s", json.dumps(report, sort_keys=True)
        )
    if args.manifest:
        manifest = create_blinded_audit_manifest(
            rows=_read_jsonl(args.input),
            accepted_quota=args.accepted_quota,
            rejected_quota=args.rejected_quota,
            borderline_quota=args.borderline_quota,
            seed=args.sample_seed,
        )
        _write_json(args.manifest, manifest)
        persist_audit_candidates(args.database, manifest)


def _read_jsonl(path: Path) -> t.Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield t.cast(dict[str, object], value)


def _run_review(args: argparse.Namespace) -> None:
    if args.manifest is None or args.decision is None:
        raise SystemExit("--review-id requires --manifest and --decision")
    if args.audio_root is None and (args.hub_repo is None or args.hub_revision is None):
        raise SystemExit(
            "--review-id requires --hub-repo and --hub-revision "
            "(or an explicit --audio-root for offline smoke tests)"
        )
    raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, list):
        raise ValueError("audit manifest must be a JSON list")
    entry = next(
        (
            item
            for item in raw_manifest
            if isinstance(item, dict) and item.get("audit_id") == args.review_id
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"unknown audit_id: {args.review_id}")
    retriever: ClipRetriever
    if args.audio_root is not None:
        retriever = LocalClipRetriever(args.audio_root)
    else:
        retriever = PinnedHubClipRetriever(
            HfApiAdapter(), repository=args.hub_repo, revision=args.hub_revision
        )
    store = _decision_store(args.database)
    try:
        result = review_one_clip(
            entry=t.cast(dict[str, object], entry),
            retriever=retriever,
            reviewer=lambda _path, item: {
                "audit_id": item["audit_id"],
                "decision": args.decision,
            },
            decision_store=store,
        )
    finally:
        store.close()
    logger.info("Blinded review recorded: %s", json.dumps(result, sort_keys=True))


class LocalClipRetriever(ClipRetriever):
    """Read one embedded-audio row from a local Parquet staging root.

    This adapter exists only for offline smoke tests.  In particular, it does not
    interpret a candidate's old-style ``remote_path`` as an audio file.
    """

    def __init__(self, root: Path) -> None:
        """Initialise a retriever rooted at ``root``."""
        self.root = root.resolve()

    def retrieve(self, entry: t.Mapping[str, object]) -> bytes:
        """Read exactly one row from the candidate's local Parquet shard.

        Returns:
            Embedded audio bytes from the addressed row.

        Raises:
            FileNotFoundError:
                If the shard or row is absent.
            ValueError:
                If the candidate is not a local Parquet locator or has no audio.
        """
        raw_path = entry.get("parquet_path")
        if not isinstance(raw_path, str) or not raw_path.endswith(".parquet"):
            raise ValueError("manifest entry has no local Parquet path")
        path = (self.root / raw_path).resolve()
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        locator = entry.get("row_locator", 0)
        if not isinstance(locator, int) or locator < 0:
            raise ValueError("manifest row_locator must be a non-negative integer")
        parquet = pq.ParquetFile(path)
        for index, batch in enumerate(parquet.iter_batches(batch_size=1)):
            if index != locator:
                continue
            row = batch.to_pylist()[0]
            audio = row.get("audio", row.get("waveform"))
            if isinstance(audio, bytes):
                return audio
            if isinstance(audio, dict) and isinstance(audio.get("bytes"), bytes):
                return t.cast(bytes, audio["bytes"])
            raise ValueError("Parquet row does not contain embedded audio bytes")
        raise FileNotFoundError(f"Parquet row {locator} was not found")


def _decision_store(database: Path) -> MetadataLedger:
    """Open the metadata-only store used by one-item CLI reviews.

    Returns:
        An open validation ledger.
    """
    return MetadataLedger(database)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
