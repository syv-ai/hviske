"""Focused tests for bilingual and zero-copy data foundations."""

import json
import multiprocessing
import pickle
import typing as t
import wave
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import (
    Audio,
    Dataset,
    Features,
    IterableDataset,
    Value,
    interleave_datasets,
    load_dataset,
)
from datasets import config as datasets_config

from hviske.data import (
    _dataset_cache_identity,
    _filter_dataset_rows,
    _filter_streaming_dataset,
    _limit_validation_dataset,
    _set_source_language,
    _standardise_training_dataset,
    _validate_dataset_probabilities,
    join_audio_and_transcripts,
    process_dataset,
)
from hviske.local_vtt import (
    VTTParseStats,
    build_vtt_manifest,
    decode_vtt_audio,
    load_vtt_manifest,
    parse_vtt,
)


def _consume_pickled_join(payload: bytes) -> list[dict[str, object]]:
    """Consume a pickled joined stream in a spawn child.

    Returns:
        Rows emitted by the joined stream.
    """
    dataset = t.cast(IterableDataset, pickle.loads(payload))
    return list(dataset)


def _streaming_join_audio_rows() -> Iterable[dict[str, str]]:
    """Yield small audio rows for the streaming join pickle test."""
    yield {"partition": "train", "key": "one"}
    yield {"partition": "other", "key": "two"}
    yield {"partition": "train", "key": "three"}


def test_build_vtt_manifest_is_atomic_on_fatal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed build cannot replace a previously complete manifest."""
    wav_path = tmp_path / "programme.wav"
    _write_wav(wav_path)
    vtt_path = wav_path.with_suffix(".vtt")
    vtt_path.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("previous\n", encoding="utf-8")
    monkeypatch.setattr(
        "hviske.local_vtt.sf.info", lambda _: (_ for _ in ()).throw(OSError("boom"))
    )

    with pytest.raises(OSError, match="boom"):
        build_vtt_manifest(
            source_directories=[tmp_path], output_path=manifest_path, language="da"
        )

    assert manifest_path.read_text(encoding="utf-8") == "previous\n"
    assert not list(tmp_path.glob(".manifest.jsonl.*.tmp"))


def _write_wav(path: Path) -> None:
    samples = (np.zeros(16_000, dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples)


def test_dataset_cache_identity_includes_all_dataset_coordinates() -> None:
    """Changing any dataset coordinate selects a different cache."""
    identity = _dataset_cache_identity(
        dataset_id="org/data",
        subset="subset",
        split="validation",
        revision="commit",
        purpose="validation",
    )
    assert identity == _dataset_cache_identity(
        dataset_id="org/data",
        subset="subset",
        split="validation",
        revision="commit",
        purpose="validation",
    )
    for keyword, value in {
        "dataset_id": "org/other",
        "subset": "other",
        "split": "test",
        "revision": "other",
    }.items():
        kwargs = {
            "dataset_id": "org/data",
            "subset": "subset",
            "split": "validation",
            "revision": "commit",
            "purpose": "validation",
        }
        kwargs[keyword] = value
        assert (
            _dataset_cache_identity(**kwargs)  # ty: ignore[invalid-argument-type]
            != identity
        )


@pytest.mark.parametrize("probabilities", [[0.5, 0.500000005], [1.0]])
def test_dataset_probabilities_allow_small_floating_point_error(
    probabilities: list[float],
) -> None:
    """Probability validation tolerates only the documented absolute error."""
    assert (
        _validate_dataset_probabilities(
            probabilities=probabilities, dataset_count=len(probabilities)
        )
        == probabilities
    )


@pytest.mark.parametrize(
    "probabilities,dataset_count",
    [
        ([0.5], 2),
        ([True, 0.0], 2),
        (["0.5", 0.5], 2),
        ([np.nan, 0.5], 2),
        ([np.inf, 0.0], 2),
        ([-0.1, 1.1], 2),
        ([0.4, 0.4], 2),
    ],
)
def test_dataset_probabilities_reject_invalid_values(
    probabilities: list[object], dataset_count: int
) -> None:
    """Invalid probabilities fail with ValueError rather than assertions."""
    with pytest.raises(ValueError):
        _validate_dataset_probabilities(
            probabilities=probabilities, dataset_count=dataset_count
        )


def test_exact_row_filters_apply_to_streaming_datasets() -> None:
    """Streaming row filters retain matching rows and explicit features."""
    features = Features(source=Value("string"), text=Value("string"))
    dataset = IterableDataset.from_generator(
        lambda: iter(
            [
                {"source": "voxpopuli", "text": "included"},
                {"source": "ftspeech", "text": "excluded"},
            ]
        ),
        features=features,
    )

    filtered = _filter_dataset_rows(dataset=dataset, filters={"source": "voxpopuli"})

    assert filtered.features == features
    assert [row["text"] for row in filtered] == ["included"]


def test_exact_row_filters_fail_for_missing_features() -> None:
    """Row filters reject unknown columns before consuming a stream."""
    dataset = IterableDataset.from_generator(
        lambda: iter([]), features=Features(text=Value("string"))
    )

    with pytest.raises(ValueError, match="missing dataset features: source"):
        _filter_dataset_rows(dataset=dataset, filters={"source": "voxpopuli"})


def test_exact_row_filters_infer_features_for_untyped_streams() -> None:
    """Row filters infer a bounded schema without consuming the stream."""
    consumed = 0

    def rows() -> Iterable[dict[str, str]]:
        nonlocal consumed
        index = 0
        while True:
            consumed += 1
            yield {"source": "keep" if index % 2 == 0 else "drop", "text": str(index)}
            index += 1

    dataset = IterableDataset.from_generator(rows)
    filtered = _filter_dataset_rows(dataset=dataset, filters={"source": "keep"})

    assert consumed <= 5
    assert [row["text"] for row in filtered.take(3)] == ["0", "2", "4"]
    assert [row["text"] for row in filtered.take(3)] == ["0", "2", "4"]


def test_exact_row_filters_repeat_on_local_parquet_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local Parquet streams retain typed decode-free audio across filters."""
    monkeypatch.setattr(datasets_config, "HF_DATASETS_OFFLINE", True)
    parquet_path = tmp_path / "audio.parquet"
    Dataset.from_list(
        [
            {
                "source": "keep" if index % 2 == 0 else "drop",
                "partition": "train" if index % 4 == 0 else "other",
                "audio": {"path": f"{index}.wav", "bytes": None},
            }
            for index in range(9)
        ]
    ).to_parquet(parquet_path)
    dataset = t.cast(
        IterableDataset,
        load_dataset(
            "parquet", data_files=str(parquet_path), split="train", streaming=True
        ),
    ).cast_column("audio", Audio(decode=False))

    filtered_once = _filter_dataset_rows(dataset=dataset, filters={"source": "keep"})
    assert isinstance(filtered_once, IterableDataset)
    assert filtered_once._ex_iterable.is_typed
    filtered_twice = _filter_dataset_rows(
        dataset=filtered_once, filters={"partition": "train"}
    )

    assert isinstance(filtered_twice, IterableDataset)
    features = filtered_twice.features
    assert features == dataset.features
    assert features is not None
    assert filtered_twice.n_shards == dataset.n_shards
    assert [row["audio"]["path"] for row in filtered_twice] == [
        "0.wav",
        "4.wav",
        "8.wav",
    ]
    assert [row["audio"]["path"] for row in filtered_twice] == [
        "0.wav",
        "4.wav",
        "8.wav",
    ]
    assert isinstance(features["audio"], Audio)
    assert not features["audio"].decode
    assert list(pickle.loads(pickle.dumps(filtered_twice))) == list(filtered_twice)


def test_exact_row_filters_repeat_on_typed_streams_without_read_ahead() -> None:
    """Sequential filters remain restartable and bounded on a typed iterable."""
    consumed = 0
    features = Features(
        source=Value("string"), partition=Value("string"), audio=Audio(decode=False)
    )

    def rows() -> Iterable[dict[str, object]]:
        nonlocal consumed
        index = 0
        while True:
            consumed += 1
            yield {
                "source": "keep" if index % 2 == 0 else "drop",
                "partition": "train" if index % 4 == 0 else "other",
                "audio": {"path": f"{index}.wav", "bytes": None},
            }
            index += 1

    dataset = IterableDataset.from_generator(rows, features=features)
    filtered_once = _filter_dataset_rows(dataset=dataset, filters={"source": "keep"})
    assert isinstance(filtered_once, IterableDataset)
    assert filtered_once.features == features
    assert filtered_once._ex_iterable.is_typed

    filtered_twice = _filter_dataset_rows(
        dataset=filtered_once, filters={"partition": "train"}
    )

    assert isinstance(filtered_twice, IterableDataset)
    assert filtered_twice.features == features
    assert consumed == 0
    first_pass = [row["audio"]["path"] for row in filtered_twice.take(3)]
    assert first_pass == ["0.wav", "4.wav", "8.wav"]
    assert consumed <= 9
    second_pass = [row["audio"]["path"] for row in filtered_twice.take(3)]
    assert second_pass == first_pass
    assert consumed <= 18


def test_filtered_join_is_pickleable_and_spawn_restartable() -> None:
    """A joined stream remains restartable after filtering and pickling."""
    audio = IterableDataset.from_generator(
        _streaming_join_audio_rows,
        features=Features(partition=Value("string"), key=Value("string")),
    )
    filtered = _filter_dataset_rows(dataset=audio, filters={"partition": "train"})
    transcripts = Dataset.from_list(
        [
            {"transcript_key": "one", "words": "Et"},
            {"transcript_key": "three", "words": "Tre"},
        ]
    )
    joined = join_audio_and_transcripts(
        audio_dataset=filtered,
        transcript_dataset=transcripts,
        audio_join_column="key",
        transcript_join_column="transcript_key",
        transcript_text_column="words",
    )
    expected = [
        {"partition": "train", "key": "one", "text": "Et"},
        {"partition": "train", "key": "three", "text": "Tre"},
    ]

    payload = pickle.dumps(joined)
    assert list(joined) == expected
    assert list(joined) == expected
    assert list(pickle.loads(payload)) == expected

    context = multiprocessing.get_context("spawn")
    with context.Pool(1) as pool:
        assert pool.apply(_consume_pickled_join, (payload,)) == expected


def test_join_rejects_duplicate_transcript_keys() -> None:
    """A duplicate index key cannot silently overwrite a transcript."""
    audio = IterableDataset.from_generator(
        lambda: iter([{"key": "one"}]), features=Features(key=Value("string"))
    )
    transcripts = Dataset.from_list(
        [
            {"transcript_key": "one", "words": "Et"},
            {"transcript_key": "one", "words": "To"},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate transcript key"):
        join_audio_and_transcripts(
            audio_dataset=audio,
            transcript_dataset=transcripts,
            audio_join_column="key",
            transcript_join_column="transcript_key",
            transcript_text_column="words",
        )


@pytest.mark.parametrize(
    ("audio_key", "transcript_key", "message"),
    [("one", ["one"], "hashable"), (1, "one", "type")],
)
def test_join_rejects_invalid_first_audio_key(
    audio_key: object, transcript_key: object, message: str
) -> None:
    """A preflight join rejects unhashable keys and key-type errors."""
    audio = Dataset.from_list([{"key": audio_key}])
    transcripts = Dataset.from_list([{"transcript_key": transcript_key, "words": "Et"}])
    with pytest.raises(ValueError, match=message):
        joined = join_audio_and_transcripts(
            audio_dataset=audio,
            transcript_dataset=transcripts,
            audio_join_column="key",
            transcript_join_column="transcript_key",
            transcript_text_column="words",
        )
        next(iter(joined))


def test_join_rejects_transcript_index_without_usable_text() -> None:
    """A transcript index containing only empty text cannot train a model."""
    audio = IterableDataset.from_generator(
        lambda: iter([{"key": "one"}]), features=Features(key=Value("string"))
    )
    transcripts = Dataset.from_list([{"transcript_key": "one", "words": ""}])

    with pytest.raises(ValueError, match="no usable transcripts"):
        join_audio_and_transcripts(
            audio_dataset=audio,
            transcript_dataset=transcripts,
            audio_join_column="key",
            transcript_join_column="transcript_key",
            transcript_text_column="words",
        )


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


def test_joined_audio_consumes_a_matching_transcript() -> None:
    """The first streamed audio example receives its indexed transcript."""
    audio = IterableDataset.from_generator(
        lambda: iter([{"key": "one"}]), features=Features(key=Value("string"))
    )
    transcripts = Dataset.from_list([{"transcript_key": "one", "words": "Et"}])
    joined = join_audio_and_transcripts(
        audio_dataset=audio,
        transcript_dataset=transcripts,
        audio_join_column="key",
        transcript_join_column="transcript_key",
        transcript_text_column="words",
    )
    assert next(iter(joined))["text"] == "Et"


def test_local_and_hub_iterable_datasets_can_be_interleaved(tmp_path: Path) -> None:
    """Local cue streams retain compatible features with Hub audio streams."""
    wav_path = tmp_path / "programme.wav"
    _write_wav(wav_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_wav_path": str(wav_path),
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "local",
                "id": "local",
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    local = load_vtt_manifest(
        manifest_path=manifest_path, min_seconds=0.1, max_seconds=2.0
    )
    local_features = local.features.copy()
    local_features["audio"] = Audio(sampling_rate=16_000)
    local = local.map(
        function=lambda example: decode_vtt_audio(example, sampling_rate=16_000),
        features=local_features,
    ).remove_columns(["source_wav_path", "start", "end", "duration", "id"])
    hub = IterableDataset.from_generator(
        lambda: iter(
            [
                {
                    "audio": {"array": np.zeros(16_000), "sampling_rate": 16_000},
                    "text": "hub",
                    "language": "da",
                }
            ]
        ),
        features=Features(
            audio=Audio(sampling_rate=16_000),
            text=Value("string"),
            language=Value("string"),
        ),
    )

    interleaved = interleave_datasets(
        datasets=[local, hub],
        probabilities=[0.5, 0.5],
        seed=4242,
        stopping_strategy="all_exhausted",
    )
    assert {row["text"] for row in interleaved} == {"local", "hub"}


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


class PromptProcessor:
    """Small prompt-aware processor used to inspect language propagation."""

    def __init__(self) -> None:
        """Initialise the language call log."""
        self.languages: list[str] = []

    def __call__(self, audio: object, **kwargs: object) -> dict[str, object]:
        """Record a processor call and return minimal model features.

        Returns:
            Minimal prompt and label features.
        """
        del audio
        self.languages.append(str(kwargs["language"]))
        return {
            "input_features": [[[1.0]]],
            "attention_mask": [[1]],
            "decoder_input_ids": [[1]],
            "labels": [[1]],
        }

    def get_decoder_prompt_ids(self) -> list[int]:
        """Identify this processor as prompt-aware.

        Returns:
            A marker identifying a prompt-aware processor.
        """
        return [1]


def _audio() -> dict[str, object]:
    return {"array": [0.0] * 16_000, "sampling_rate": 16_000}


def test_production_sources_have_restartable_interleave_schema(tmp_path: Path) -> None:
    """Joined Hub streams and local VTT streams share a restartable schema."""
    wav_path = tmp_path / "programme.wav"
    _write_wav(wav_path)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_wav_path": str(wav_path),
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "local",
                "id": "local",
                "language": "da",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    local = load_vtt_manifest(manifest_path, min_seconds=0.1, max_seconds=2.0)
    local_features = local.features.copy()
    local_features["audio"] = Audio(sampling_rate=16_000)
    local = local.map(
        function=lambda example: decode_vtt_audio(example, sampling_rate=16_000),
        features=local_features,
    )
    local = t.cast(
        IterableDataset, _standardise_training_dataset(local, sampling_rate=16_000)
    )

    hub = IterableDataset.from_generator(
        lambda: iter(
            [
                {
                    "recording_id": "hub",
                    "audio": {"array": np.zeros(16_000), "sampling_rate": 16_000},
                    "extra_metadata": "removed",
                }
            ]
        ),
        features=Features(
            recording_id=Value("string"),
            audio=Audio(sampling_rate=16_000),
            extra_metadata=Value("string"),
        ),
    )
    hub = join_audio_and_transcripts(
        audio_dataset=hub,
        transcript_dataset=Dataset.from_list(
            [{"recording_id": "hub", "transcript": "hub"}]
        ),
        audio_join_column="recording_id",
        transcript_join_column="recording_id",
        transcript_text_column="transcript",
    )
    hub_features = hub.features.copy()
    hub_features["language"] = Value("string")
    hub = hub.map(
        function=lambda example: _set_source_language(example, language="en"),
        features=hub_features,
    )
    hub = t.cast(
        IterableDataset, _standardise_training_dataset(hub, sampling_rate=16_000)
    )

    interleaved = interleave_datasets(
        datasets=[local, hub],
        probabilities=[0.5, 0.5],
        seed=4242,
        stopping_strategy="all_exhausted",
    )
    assert interleaved.features == local.features == hub.features
    rows = list(interleaved)
    restarted_rows = list(interleaved)
    assert [row["text"] for row in rows] == [row["text"] for row in restarted_rows]
    assert set(row["text"] for row in rows) == {"local", "hub"}
    assert all(set(row) == {"audio", "text", "language"} for row in rows)


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


def test_streaming_audio_is_joined_to_partial_transcript_index() -> None:
    """Joining lazily skips audio without a usable indexed transcript."""
    audio = IterableDataset.from_generator(
        lambda: iter([{"key": "one"}, {"key": "two"}, {"key": "three"}]),
        features=Features(key=Value("string")),
    )
    transcripts = Dataset.from_list(
        [
            {"transcript_key": "one", "words": "  "},
            {"transcript_key": "two", "words": "To"},
        ]
    )
    joined = join_audio_and_transcripts(
        audio_dataset=audio,
        transcript_dataset=transcripts,
        audio_join_column="key",
        transcript_join_column="transcript_key",
        transcript_text_column="words",
    )

    assert list(joined) == [{"key": "two", "text": "To"}]


def test_streaming_filter_reraises_unrelated_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated filter TypeError is propagated without replacement."""
    dataset = IterableDataset.from_generator(
        lambda: iter([{"key": "one"}]), features=Features(key=Value("string"))
    )
    filtered_once = dataset.filter(function=lambda row: True)
    error = TypeError("unrelated filter failure")

    def fail_filter(self: IterableDataset, *, function: object) -> IterableDataset:
        del self, function
        raise error

    monkeypatch.setattr(IterableDataset, "filter", fail_filter)

    with pytest.raises(TypeError) as raised:
        _filter_streaming_dataset(dataset=filtered_once, function=lambda row: True)

    assert raised.value is error


def test_validation_sample_cap_is_applied_before_materialisation() -> None:
    """Only capped validation examples are consumed during materialisation."""
    consumed: list[int] = []

    def examples() -> t.Iterator[dict[str, int]]:
        for value in range(5):
            consumed.append(value)
            yield {"value": value}

    dataset = IterableDataset.from_generator(examples)
    capped = _limit_validation_dataset(dataset=dataset, max_samples=2)

    assert list(capped) == [{"value": 0}, {"value": 1}]
    assert consumed == [0, 1]


def test_vtt_cleans_sparkie_rolling_and_style_cues(tmp_path: Path) -> None:
    """YouTube and DRTV caption quirks are cleaned without aborting parsing."""
    vtt_path = tmp_path / "sparkie.vtt"
    vtt_path.write_text(
        "\ufeffWEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hello <00:00:00.100><c>world</c>\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "Hello world\n\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "<i>world</i> again\n\n"
        "00:00:03.000 --> 00:00:04.000\n\n"
        "00:00:bad --> 00:05.000\nignored\n",
        encoding="utf-8",
    )
    stats = VTTParseStats()

    cues = parse_vtt(vtt_path, stats=stats)

    assert [cue["text"] for cue in cues] == ["Hello world", "again"]
    assert stats.cues_skipped == 3


def test_vtt_manifest_does_not_open_audio_before_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest construction remains metadata-only until an item is consumed."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_wav_path": str(tmp_path / "programme.wav"),
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text": "hello",
                "id": "one",
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    opened = False

    def fail_if_opened(*args: object, **kwargs: object) -> None:
        nonlocal opened
        del args, kwargs
        opened = True
        raise AssertionError("audio was opened while loading the manifest")

    monkeypatch.setattr("hviske.local_vtt.sf.SoundFile", fail_if_opened)
    dataset = load_vtt_manifest(
        manifest_path=manifest_path, min_seconds=0.1, max_seconds=2.0
    )
    assert not opened
    assert next(iter(dataset))["text"] == "hello"
    assert not opened


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
    decoded = decode_vtt_audio(next(iter(dataset)), sampling_rate=16_000)
    audio = t.cast(dict[str, object], decoded["audio"])
    array = t.cast(np.ndarray, audio["array"])
    assert array.shape[0] == 8_000
    assert decoded["text"] == "hello"


def test_vtt_resampling_downmixes_and_preserves_cue_metadata(tmp_path: Path) -> None:
    """A stereo 48 kHz cue is anti-aliased and retains its manifest offsets."""
    wav_path = tmp_path / "programme.wav"
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = np.column_stack(
        [np.sin(2 * np.pi * 1_000 * time), np.sin(2 * np.pi * 12_000 * time)]
    )
    sf.write(wav_path, signal, sample_rate, subtype="FLOAT")
    low_wav_path = tmp_path / "low.wav"
    sf.write(low_wav_path, signal[:, :1], sample_rate, subtype="FLOAT")
    example: dict[str, object] = {
        "source_wav_path": str(wav_path),
        "start": 0.25,
        "end": 0.75,
        "duration": 0.5,
        "text": "hello",
        "language": "en",
    }

    decoded = decode_vtt_audio(example, sampling_rate=16_000)
    audio = t.cast(dict[str, object], decoded["audio"])
    array = t.cast(np.ndarray, audio["array"])
    assert array.dtype == np.float32
    assert array.flags.c_contiguous
    assert array.shape == (8_000,)
    assert decoded["start"] == 0.25
    assert decoded["end"] == 0.75
    assert decoded["duration"] == 0.5
    assert np.mean(np.abs(array)) > 0.1

    low_example: dict[str, object] = {**example, "source_wav_path": str(low_wav_path)}
    low_decoded = decode_vtt_audio(low_example, sampling_rate=16_000)
    low_audio = t.cast(dict[str, object], low_decoded["audio"])
    low_array = t.cast(np.ndarray, low_audio["array"])
    assert np.sqrt(np.mean((array - low_array / 2) ** 2)) < 0.01


def test_vtt_skips_empty_cues(tmp_path: Path) -> None:
    """Empty cues cannot enter a training manifest."""
    vtt_path = tmp_path / "programme.vtt"
    vtt_path.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\n\n", encoding="utf-8")
    stats = VTTParseStats()

    assert parse_vtt(vtt_path, stats=stats) == []
    assert stats.cues_skipped == 1
