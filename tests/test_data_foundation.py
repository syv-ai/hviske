"""Focused tests for bilingual and zero-copy data foundations."""

import json
import typing as t
import wave
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, Features, IterableDataset, Value, interleave_datasets

from hviske.data import join_audio_and_transcripts, process_dataset
from hviske.local_vtt import build_vtt_manifest, decode_vtt_audio, load_vtt_manifest


class PromptProcessor:
    """Small prompt-aware processor used to inspect language propagation."""

    def __init__(self) -> None:
        """Initialise the language call log."""
        self.languages: list[str] = []

    def get_decoder_prompt_ids(self) -> list[int]:
        """Identify this processor as prompt-aware.

        Returns:
            A marker identifying a prompt-aware processor.
        """
        return [1]

    def __call__(self, audio: object, **kwargs: object) -> dict[str, list[list[int]]]:
        """Record a processor call and return minimal model features.

        Returns:
            Minimal prompt and label features.
        """
        del audio
        self.languages.append(str(kwargs["language"]))
        return {
            "input_features": [[1]],
            "attention_mask": [[1]],
            "decoder_input_ids": [[1]],
            "labels": [[1]],
        }


def test_mixed_language_dataset_reaches_prompt_processor() -> None:
    """Each interleaved source language reaches the Cohere processor."""
    dataset = interleave_datasets(
        datasets=[
            Dataset.from_list([{"text": "Hej", "language": "da", "audio": _audio()}]),
            Dataset.from_list([{"text": "Hello", "language": "en", "audio": _audio()}]),
        ],
        probabilities=[0.5, 0.5],
        seed=4242,
        stopping_strategy="all_exhausted",
    )
    processor = PromptProcessor()

    process_dataset(
        dataset=dataset,
        lower_case=False,
        characters_to_keep=None,
        text_column="text",
        remove_input_dataset_columns=True,
        audio_column="audio",
        convert_numerals=False,
        normalise_audio=False,
        augment_audio=False,
        processor=processor,
        language="da",
        language_column="language",
    )

    assert set(processor.languages) == {"da", "en"}
    assert len(processor.languages) >= 2


def test_single_language_processor_uses_model_default() -> None:
    """The model-level language remains the default for old datasets."""
    dataset = Dataset.from_list([{"text": "Hej", "audio": _audio()}])
    processor = PromptProcessor()

    process_dataset(
        dataset=dataset,
        lower_case=False,
        characters_to_keep=None,
        text_column="text",
        remove_input_dataset_columns=True,
        audio_column="audio",
        convert_numerals=False,
        normalise_audio=False,
        augment_audio=False,
        processor=processor,
        language="da",
    )

    assert processor.languages == ["da"]


def test_streaming_audio_is_joined_to_indexed_transcripts() -> None:
    """Joining does not materialise the audio side and fails for missing keys."""
    audio = IterableDataset.from_generator(
        lambda: iter([{"key": "one"}, {"key": "two"}]),
        features=Features(key=Value("string")),
    )
    transcripts = Dataset.from_list([{"transcript_key": "one", "words": "Et"}])
    joined = join_audio_and_transcripts(
        audio_dataset=audio,
        transcript_dataset=transcripts,
        audio_join_column="key",
        transcript_join_column="transcript_key",
        transcript_text_column="words",
    )

    with pytest.raises(ValueError, match="No transcript"):
        list(joined)


def test_join_validates_configured_columns() -> None:
    """Join configuration errors are reported before consuming audio."""
    with pytest.raises(ValueError, match="Missing audio dataset columns"):
        join_audio_and_transcripts(
            audio_dataset=Dataset.from_list([{"key": "one"}]),
            transcript_dataset=Dataset.from_list([{"id": "one", "text": "Et"}]),
            audio_join_column="missing",
            transcript_join_column="id",
            transcript_text_column="text",
        )


def test_vtt_manifest_slices_original_wav_without_copying(tmp_path: Path) -> None:
    """Manifest loading filters metadata first and reads only a cue on demand."""
    wav_path = tmp_path / "programme.wav"
    vtt_path = tmp_path / "programme.vtt"
    _write_wav(wav_path)
    vtt_path.write_text("WEBVTT\n\n00:00.250 --> 00:00.750\nhello\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"

    build_vtt_manifest(
        source_directories=[tmp_path], output_path=manifest_path, language="en"
    )
    row = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert row["source_wav_path"] == str(wav_path.resolve())
    assert not list(tmp_path.glob("*.wav.*"))

    dataset = load_vtt_manifest(
        manifest_path=manifest_path, min_seconds=0.1, max_seconds=1.0
    )
    decoded = decode_vtt_audio(dataset[0], sampling_rate=16_000)
    audio = t.cast(dict[str, object], decoded["audio"])
    array = t.cast(np.ndarray, audio["array"])
    assert array.shape[0] == 8_000
    assert decoded["text"] == "hello"


def test_vtt_rejects_empty_cues(tmp_path: Path) -> None:
    """Empty cues cannot enter a training manifest."""
    vtt_path = tmp_path / "programme.vtt"
    vtt_path.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed|Empty"):
        from hviske.local_vtt import parse_vtt

        parse_vtt(vtt_path)


def _audio() -> dict[str, object]:
    return {"array": [0.0] * 16_000, "sampling_rate": 16_000}


def _write_wav(path: Path) -> None:
    samples = (np.zeros(16_000, dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples)
