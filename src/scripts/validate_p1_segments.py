"""Generate P1 validation reports and perform an explicit one-item audit review."""

from __future__ import annotations

import argparse
import io
import json
import logging
import typing as t
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from p1_dataset.publish import HfApiAdapter
from p1_dataset.source import AudioPointer, HfP1Source, SourceShard
from p1_dataset.validation import (
    ClipRetriever,
    MetadataLedger,
    PinnedHubClipRetriever,
    SourceClipRetriever,
    build_quality_report,
    create_blinded_audit_manifest,
    export_clip_for_review,
    iter_bounded,
    persist_audit_candidates,
    play_audio,
    review_one_clip,
)

logger = logging.getLogger("p1_dataset.validation")


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
    parser.add_argument(
        "--player", help="local player command and arguments; never run through a shell"
    )
    parser.add_argument(
        "--export-audio",
        type=Path,
        help="retrieve one clip for controlled non-playing review; records no decision",
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
        _write_jsonl(args.manifest, manifest)
        persist_audit_candidates(args.database, manifest)


def _read_jsonl(path: Path) -> t.Iterator[dict[str, object]]:
    """Read JSONL records, also accepting the legacy JSON-array format.

    Args:
        path:
            Manifest or metadata file to read.

    Yields:
        JSON object records in file order.

    Raises:
        ValueError:
            If a record is not a JSON object or the file is malformed.
    """
    with path.open(encoding="utf-8") as stream:
        first_line: str | None = None
        first_line_number = 0
        for first_line_number, line in enumerate(stream, 1):
            if line.strip():
                first_line = line
                break
        if first_line is None:
            return
        if first_line.lstrip().startswith("["):
            raw = json.loads(first_line + stream.read())
            if not isinstance(raw, list):
                raise ValueError("JSON-array manifest must contain a list")
            for index, value in enumerate(raw, 1):
                if not isinstance(value, dict):
                    raise ValueError(f"array item {index} is not a JSON object")
                yield t.cast(dict[str, object], value)
            return
        try:
            value = json.loads(first_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {first_line_number} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"line {first_line_number} is not a JSON object")
        yield t.cast(dict[str, object], value)
        for line_number, line in enumerate(stream, first_line_number + 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number} is not valid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield t.cast(dict[str, object], value)


def _run_review(args: argparse.Namespace) -> None:
    if args.manifest is None:
        raise SystemExit("--review-id requires --manifest")
    if (args.decision is None) == (args.export_audio is None):
        raise SystemExit(
            "choose exactly one of --decision (playing review) or --export-audio "
            "(non-playing review)"
        )
    entry = next(
        (
            item
            for item in _read_jsonl(args.manifest)
            if item.get("audit_id") == args.review_id
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"unknown audit_id: {args.review_id}")
    if (
        args.audio_root is None
        and isinstance(entry, dict)
        and isinstance(entry.get("parquet_path"), str)
        and (args.hub_repo is None or args.hub_revision is None)
    ):
        raise SystemExit(
            "remote audit reviews require --hub-repo and --hub-revision "
            "(or an explicit --audio-root for offline smoke tests)"
        )
    retriever: ClipRetriever
    if args.audio_root is not None:
        retriever = LocalClipRetriever(args.audio_root)
    elif isinstance(entry, dict) and isinstance(entry.get("parquet_path"), str):
        retriever = PinnedHubClipRetriever(
            HfApiAdapter(), repository=args.hub_repo, revision=args.hub_revision
        )
    else:
        retriever = SourceClipRetriever(_source_clip_callback)
    candidate = entry
    if args.export_audio is not None:
        destination = export_clip_for_review(
            entry=candidate, retriever=retriever, destination=args.export_audio
        )
        logger.info(
            "Exported one retrieved clip for controlled review: %s", destination
        )
        return

    store = _decision_store(args.database)
    try:
        result = review_one_clip(
            entry=candidate,
            retriever=retriever,
            reviewer=lambda _path, item: {
                "audit_id": item["audit_id"],
                "decision": args.decision,
            },
            decision_store=store,
            player=lambda path: play_audio(path, args.player),
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


def _write_jsonl(path: Path, records: t.Iterable[dict[str, object]]) -> None:
    """Write one JSON object per line for streaming review interoperability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("Wrote %s", path)


def _source_clip_callback(entry: t.Mapping[str, object]) -> bytes:
    """Retrieve and encode exactly one rejected source interval.

    Returns:
        A temporary-review OGG/Opus payload.

    Raises:
        ValueError:
            If the source locator or decoded payload is invalid.
    """
    repository = entry.get("source_repository")
    revision = entry.get("source_revision")
    file_id = entry.get("source_file_id")
    start = entry.get("source_start_ms")
    end = entry.get("source_end_ms")
    if not isinstance(repository, str) or not isinstance(revision, str):
        raise ValueError("source audit entry has an incomplete source locator")
    if not isinstance(file_id, str):
        raise ValueError("source audit entry has no source file identifier")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("source audit entry has an incomplete interval")
    shard_path = entry.get(
        "source_shard_path", entry.get("source_shard", entry.get("source_parquet_path"))
    )
    row_group = entry.get("source_row_group", 0)
    row_index = entry.get("source_row_index", entry.get("source_row_locator"))
    shard_size = entry.get("source_shard_byte_size")
    if not isinstance(shard_path, str) or not shard_path.endswith(".parquet"):
        raise ValueError("source audit entry has no exact source shard")
    if not isinstance(row_group, int) or row_group < 0:
        raise ValueError("source audit entry has no source row group")
    if not isinstance(row_index, int) or row_index < 0:
        raise ValueError("source audit entry has no source row")
    if not isinstance(shard_size, int) or shard_size <= 0:
        raise ValueError("source audit entry has no source shard size")
    source = HfP1Source(
        audio_repository=repository, max_source_object_bytes=6_197_291_423
    )
    shard = SourceShard(path=shard_path, byte_size=shard_size, revision=revision)
    pointer = AudioPointer(
        file_id=file_id, shard=shard, row_group=row_group, row_index=row_index
    )
    audio = source.fetch_audio(pointer=pointer)
    if not isinstance(audio.value, np.ndarray):
        raise ValueError("source audio did not decode to samples")
    first = max(0, start * audio.sampling_rate // 1000)
    last = min(audio.value.shape[0], end * audio.sampling_rate // 1000)
    if last <= first:
        raise ValueError("source interval is empty")
    output = io.BytesIO()
    sf.write(
        output,
        audio.value[first:last],
        audio.sampling_rate,
        format="OGG",
        subtype="OPUS",
    )
    return output.getvalue()


if __name__ == "__main__":
    main()
