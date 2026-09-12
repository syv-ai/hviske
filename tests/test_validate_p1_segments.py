"""CLI compatibility tests for P1 audit manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from p1_dataset.validation import create_blinded_audit_manifest
from scripts import validate_p1_segments as cli


def test_pipeline_manifest_round_trips_through_review_export_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSONL output routes both accepted and rejected entries to export review."""
    rows = [
        {
            "segment_id": "accepted-1",
            "source_file_id": "programme-1",
            "status": "accepted",
            "repository": "org/p1",
            "revision": "a" * 40,
            "parquet_path": "data/train/part.parquet",
            "row_locator": 0,
        },
        {
            "segment_id": "rejected-1",
            "source_file_id": "programme-1",
            "status": "rejected",
            "source_repository": "org/source",
            "source_revision": "b" * 40,
            "source_start_ms": 0,
            "source_end_ms": 1_000,
            "source_shard_path": "source/part.parquet",
            "source_row_group": 0,
            "source_row_index": 0,
        },
    ]
    manifest = create_blinded_audit_manifest(
        rows, accepted_quota=1, rejected_quota=1, seed="cli"
    )
    path = tmp_path / "manifest.jsonl"
    cli._write_jsonl(path, manifest)
    assert list(cli._read_jsonl(path)) == manifest

    legacy = tmp_path / "manifest.json"
    legacy.write_text(json.dumps(manifest), encoding="utf-8")
    assert list(cli._read_jsonl(legacy)) == manifest

    seen: list[dict[str, object]] = []

    def fake_export(
        *, entry: dict[str, object], retriever: object, destination: Path
    ) -> Path:
        del retriever
        seen.append(entry)
        return destination

    monkeypatch.setattr(cli, "export_clip_for_review", fake_export)
    for entry in manifest:
        cli._run_review(
            argparse.Namespace(
                manifest=path,
                review_id=entry["audit_id"],
                decision=None,
                export_audio=tmp_path / f"{entry['audit_id']}.wav",
                audio_root=tmp_path,
                hub_repo=None,
                hub_revision=None,
                database=tmp_path / "audit.sqlite",
                player=None,
            )
        )
    assert seen == manifest
