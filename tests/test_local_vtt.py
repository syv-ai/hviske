"""Regression tests for local WebVTT manifest streams."""

import json
from pathlib import Path

from datasets import IterableDataset

from hviske.local_vtt import load_vtt_manifest


def test_local_vtt_stream_filters_durations_and_preserves_shards(
    tmp_path: Path,
) -> None:
    """Local streams filter metadata before yielding deterministic partitions."""
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path)

    dataset = load_vtt_manifest(
        manifest_path=manifest_path, min_seconds=1.0, max_seconds=8.0, num_shards=2
    )

    assert isinstance(dataset, IterableDataset)
    assert dataset.n_shards == 2
    first_pass = [row["id"] for row in dataset]
    assert set(first_pass) == {"cue-1", "cue-2"}
    assert [row["id"] for row in dataset] == first_pass


def _write_manifest(path: Path) -> None:
    """Write rows covering both sides of the exclusive duration bounds."""
    rows = [
        {
            "source_wav_path": "/tmp/source.wav",
            "start": 0.0,
            "end": duration,
            "duration": duration,
            "text": str(index),
            "id": f"cue-{index}",
            "language": "da",
        }
        for index, duration in enumerate([0.5, 1.5, 2.0, 8.0, 8.5])
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
