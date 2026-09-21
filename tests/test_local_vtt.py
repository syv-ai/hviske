"""Regression tests for local WebVTT manifest streams."""

import collections.abc as c
import json
import typing as t
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import IterableDataset
from omegaconf import DictConfig

import hviske.data as data_module
from hviske.local_vtt import (
    build_vtt_manifest,
    decode_vtt_audio,
    load_vtt_manifest,
    parse_vtt,
)


def test_build_manifest_and_decode_vtt_audio(tmp_path: Path) -> None:
    """A manifest parses VTT cues and decodes only each cue's WAV segment."""
    wav_path = tmp_path / "recording.wav"
    sf.write(wav_path, np.ones(24_000, dtype=np.float32), 8_000)
    vtt_path = wav_path.with_suffix(".vtt")
    vtt_path.write_text(
        """WEBVTT

00:00.000 --> 00:01.000
hello

00:01.000 --> 00:02.000
world
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"

    summary = build_vtt_manifest(
        source_directories=[tmp_path], output_path=manifest_path, language="da"
    )
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert summary.cues_written == 2
    assert [row["text"] for row in rows] == ["hello", "world"]
    decoded = decode_vtt_audio(example=rows[0], sampling_rate=16_000)
    decoded_audio = t.cast(dict[str, object], decoded["audio"])
    assert decoded_audio["sampling_rate"] == 16_000
    assert t.cast(np.ndarray, decoded_audio["array"]).shape == (16_000,)


def test_load_data_for_finetuning_local_vtt_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local VTT training data is decoded and standardised without Hub access."""
    wav_path = tmp_path / "recording.wav"
    sf.write(wav_path, np.linspace(-1, 1, 8_000, dtype=np.float32), 8_000)
    wav_path.with_suffix(".vtt").write_text(
        """WEBVTT

00:00.000 --> 00:01.000
Hello!
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    build_vtt_manifest(
        source_directories=[tmp_path], output_path=manifest_path, language="da"
    )

    monkeypatch.setattr(data_module, "load_dataset", pytest.fail, raising=True)
    real_process_dataset = t.cast(c.Callable[..., object], data_module.process_dataset)

    def process_without_augmentation(**kwargs: object) -> object:
        kwargs["remove_input_dataset_columns"] = False
        kwargs["augment_audio"] = False
        return real_process_dataset(**kwargs)

    monkeypatch.setattr(data_module, "process_dataset", process_without_augmentation)
    config = DictConfig(
        {
            "datasets": {
                "local": {
                    "type": "local_vtt",
                    "manifest_path": str(manifest_path),
                    "language": "da",
                    "local_vtt_num_shards": 1,
                    "shuffle_buffer_size": 1,
                }
            },
            "dataset_probabilities": None,
            "hub_streaming_retries": None,
            "min_seconds_per_example": 0.5,
            "max_seconds_per_example": 2.0,
            "streaming": True,
            "cache_dir": None,
            "seed": 7,
            "dataset_num_workers": 1,
            "evaluation_datasets": [],
            "model": {
                "sampling_rate": 16_000,
                "lower_case": True,
                "characters_to_keep": "abcdefghijklmnopqrstuvwxyz ",
                "language": "da",
                "punctuation": True,
            },
        }
    )

    dataset = data_module.load_data_for_finetuning(config=config)
    rows = list(dataset["train"])

    assert set(rows[0]) == {"audio", "text", "language"}
    assert rows[0]["text"] == "hello"
    assert rows[0]["language"] == "da"
    assert rows[0]["audio"]["sampling_rate"] == 16_000
    assert len(rows[0]["audio"]["array"]) == 16_000


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


def test_parse_vtt_removes_rolling_overlap_from_complete_caption(
    tmp_path: Path,
) -> None:
    """Rolling captions emit only words absent from the preceding full caption."""
    vtt_path = tmp_path / "captions.vtt"
    vtt_path.write_text(
        """WEBVTT

00:00.000 --> 00:01.000
hello

00:01.000 --> 00:02.000
hello world

00:02.000 --> 00:03.000
hello world again
""",
        encoding="utf-8",
    )

    assert [cue["text"] for cue in parse_vtt(vtt_path)] == ["hello", "world", "again"]
