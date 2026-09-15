"""Unit tests for the `data` module."""

import collections.abc as c
import re
import typing as t
from collections.abc import Generator
from pathlib import Path

import pytest
import soundfile as sf
import torch
import torchaudio
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    IterableDatasetDict,
    Value,
)
from omegaconf import DictConfig
from torch_audiomentations import AddBackgroundNoise

import hviske.data as data
from hviske.audio import SoundfileAudio
from hviske.data import (
    filter_dataset,
    load_data_for_finetuning,
    process_dataset,
    process_example,
)


class TestLoadDataForFinetuning:
    """Unit tests for the `load_data` function."""

    @pytest.fixture(scope="class")
    def finetuning_dataset(
        self, finetuning_config: DictConfig
    ) -> Generator[IterableDatasetDict, None, None]:
        """Load the dataset for testing.

        Yields:
            The dataset for testing.
        """
        yield load_data_for_finetuning(config=finetuning_config)

    def test_dataset_type(self, finetuning_dataset: IterableDatasetDict) -> None:
        """Test that the dataset is of the correct type."""
        assert isinstance(finetuning_dataset, IterableDatasetDict)

    def test_split_names(self, finetuning_dataset: IterableDatasetDict) -> None:
        """Test that the dataset has the correct split names."""
        assert "train" in finetuning_dataset
        for split_name in finetuning_dataset.keys():
            if split_name == "train":
                continue
            assert re.match(r"^val(_.+)?$", split_name) is not None


class TestProcessDataset:
    """Unit tests for the `process_dataset` function."""

    def test_process_dataset(
        self, dataset: Dataset | IterableDataset | DatasetDict | IterableDatasetDict
    ) -> None:
        """Test that the `process_dataset` function works as expected."""
        processed_dataset = process_dataset(
            dataset=dataset,
            characters_to_keep=None,
            text_column="text",
            audio_column=None,
            convert_numerals=False,
            remove_input_dataset_columns=False,
            lower_case=True,
            normalise_audio=True,
            augment_audio=False,
        )
        processed_samples = {sample["text"] for sample in processed_dataset}
        expected_samples = {
            "min fortræffelige lille nattergal!",
            "jeg venter grumme meget af den",
            "men hendes vilje var fast, som hendes tillid til vorherre",
            "her er kommet gode klæder at slide for de fire børn!",
            "hver rose på træet i haven havde sin historie.",
        }
        assert processed_samples == expected_samples


class TestProcessExample:
    """Unit tests for the `process_example` function."""

    transcription = "\nThis is a (test) [sentence]\u0301 with \n{aa} and ğ. "

    empty_conversion_dict: dict[str, str] = {}
    diacritics_conversion_dict = {"aa": "å", "ğ": "g"}
    empty_whitespace_conversion_dict = {"\u0301": " "}

    all_characters = (
        set(transcription)
        | set(empty_conversion_dict.values())
        | set(diacritics_conversion_dict.values())
        | set(empty_whitespace_conversion_dict.values())
    )
    no_parentheses = all_characters - set("()[]{}")
    no_newlines = all_characters - set("\n\r")

    @pytest.mark.parametrize(
        argnames=[
            "transcription",
            "characters_to_keep",
            "conversion_dict",
            "text_column",
            "lower_case",
            "expected",
        ],
        argvalues=[
            (
                transcription,
                all_characters,
                empty_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                all_characters,
                empty_conversion_dict,
                "text",
                False,
                "This is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                all_characters,
                empty_conversion_dict,
                "text2",
                True,
                "this is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                None,
                empty_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                all_characters,
                diacritics_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with\n{å} and g.",
            ),
            (
                transcription,
                all_characters,
                empty_whitespace_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence] with\n{aa} and ğ.",
            ),
            (
                transcription,
                no_parentheses,
                empty_conversion_dict,
                "text",
                True,
                "this is a test sentence \u0301 with\naa and ğ.",
            ),
            (
                transcription,
                no_parentheses,
                diacritics_conversion_dict,
                "text",
                True,
                "this is a test sentence \u0301 with\nå and g.",
            ),
            (
                transcription,
                no_parentheses,
                empty_whitespace_conversion_dict,
                "text",
                True,
                "this is a test sentence with\naa and ğ.",
            ),
            (
                transcription,
                no_newlines,
                empty_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with {aa} and ğ.",
            ),
            (
                transcription,
                no_newlines,
                diacritics_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with {å} and g.",
            ),
            (
                transcription,
                no_newlines,
                empty_whitespace_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence] with {aa} and ğ.",
            ),
        ],
        ids=[
            "empty-empty",
            "empty-empty-no-lower-case",
            "empty-empty-different-text-column",
            "empty-empty-with-None-characters-to-keep",
            "empty-diacritics",
            "empty-empty_whitespace",
            "parans-empty",
            "parans-diacritics",
            "parans-empty_whitespace",
            "newline-empty",
            "newline-diacritics",
            "newline-empty_whitespace",
        ],
    )
    def test_clean_example(
        self,
        transcription: str,
        characters_to_keep: set[str] | None,
        conversion_dict: dict[str, str],
        text_column: str,
        lower_case: bool,
        expected: str,
    ) -> None:
        """Test that the `clean_example` function works as expected."""
        example = {text_column: transcription}
        cleaned_transcription = process_example(
            example=example,
            characters_to_keep=characters_to_keep,
            conversion_dict=conversion_dict,
            text_column=text_column,
            audio_column=None,
            lower_case=lower_case,
            convert_numerals=False,
            processor=None,
            normalise_audio=True,
            augment_audio=False,
        )[text_column]
        assert cleaned_transcription == expected


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("num_proc, expected_num_proc", [(1, None), (2, 2)])
def test_filter_dataset_normalises_single_worker(
    as_dict: bool,
    num_proc: int,
    expected_num_proc: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regular dataset filtering does not fork for one configured worker."""
    base_dataset = Dataset.from_list(
        [
            {
                "audio": {"array": [0.0] * 16_001, "sampling_rate": 16_000},
                "text": "Hello",
            }
        ],
        features=Features(
            {"audio": Audio(sampling_rate=16_000), "text": Value("string")}
        ),
    )
    dataset: Dataset | DatasetDict = (
        DatasetDict({"train": base_dataset}) if as_dict else base_dataset
    )
    calls: list[int | None] = []
    original_filter = Dataset.filter

    def spy_filter(*args: object, **kwargs: object) -> Dataset:
        """Record the process count and call the datasets implementation.

        Returns:
            The filtered dataset.
        """
        calls.append(t.cast(int | None, kwargs["num_proc"]))
        return t.cast(c.Callable[..., Dataset], original_filter)(*args, **kwargs)

    monkeypatch.setattr(Dataset, "filter", spy_filter)

    filtered = filter_dataset(
        dataset=dataset,
        audio_column="audio",
        text_column="text",
        min_seconds_per_example=1.0,
        max_seconds_per_example=10,
        is_main_process=True,
        num_proc=num_proc,
    )

    assert calls == [expected_num_proc]
    if as_dict:
        filtered_dict = t.cast(DatasetDict, filtered)
        assert filtered_dict["train"]["text"] == ["Hello"]
    else:
        filtered_dataset = t.cast(Dataset, filtered)
        assert filtered_dataset["text"] == ["Hello"]


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("num_proc, expected_num_proc", [(1, None), (2, 2)])
def test_process_dataset_normalises_single_worker(
    as_dict: bool,
    num_proc: int,
    expected_num_proc: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regular dataset processing does not fork for one configured worker."""
    base_dataset = Dataset.from_list([{"text": "Hello"}])
    dataset: Dataset | DatasetDict = (
        DatasetDict({"train": base_dataset}) if as_dict else base_dataset
    )
    calls: list[int | None] = []
    original_map = Dataset.map

    def spy_map(*args: object, **kwargs: object) -> Dataset:
        """Record the process count and call the datasets implementation.

        Returns:
            The mapped dataset.
        """
        calls.append(t.cast(int | None, kwargs["num_proc"]))
        return t.cast(c.Callable[..., Dataset], original_map)(*args, **kwargs)

    monkeypatch.setattr(Dataset, "map", spy_map)

    processed = process_dataset(
        dataset=dataset,
        lower_case=True,
        characters_to_keep=None,
        text_column="text",
        remove_input_dataset_columns=False,
        audio_column=None,
        convert_numerals=False,
        normalise_audio=False,
        augment_audio=False,
        num_proc=num_proc,
    )

    assert calls == [expected_num_proc]
    if as_dict:
        processed_dict = t.cast(DatasetDict, processed)
        assert processed_dict["train"]["text"] == ["hello"]
    else:
        processed_dataset = t.cast(Dataset, processed)
        assert processed_dataset["text"] == ["hello"]


def test_process_example_augments_without_torchaudio_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Background noise augmentation uses the soundfile-backed audio adapter."""
    monkeypatch.chdir(tmp_path)
    background_path = tmp_path / "background-noises" / "noise.wav"
    background_path.parent.mkdir()
    sf.write(file=background_path, data=[0.1] * 8_000, samplerate=8_000)
    monkeypatch.delattr(torchaudio, "info", raising=False)

    def load_audio(
        path: str | Path, frame_offset: int = 0, num_frames: int = -1
    ) -> tuple[torch.Tensor, int]:
        """Provide the decoder used by torch-audiomentations in this test.

        Returns:
            A decoded waveform and its sample rate.
        """
        waveform, sample_rate = sf.read(file=path, dtype="float32", always_2d=True)
        waveform_tensor = torch.from_numpy(waveform.T)
        end = None if num_frames < 0 else frame_offset + num_frames
        return waveform_tensor[:, frame_offset:end], sample_rate

    monkeypatch.setattr(torchaudio, "load", load_audio)
    created: list[AddBackgroundNoise] = []
    original_add_background_noise = data.ta.AddBackgroundNoise

    def create_background_noise(
        *, background_paths: Path, p: float, sample_rate: int
    ) -> AddBackgroundNoise:
        """Make the probabilistic transform deterministic for this test.

        Returns:
            A background-noise transform configured to always run.
        """
        del p
        transform = original_add_background_noise(
            background_paths=background_paths, p=1.0, sample_rate=sample_rate
        )
        created.append(transform)
        return transform

    monkeypatch.setattr(data.ta, "AddBackgroundNoise", create_background_noise)
    original_audio = torch.full((16_000,), 0.2)
    example = {
        "text": "Hello",
        "audio": {"array": original_audio.tolist(), "sampling_rate": 16_000},
    }

    processed = process_example(
        example=example,
        characters_to_keep=None,
        conversion_dict={},
        text_column="text",
        audio_column="audio",
        lower_case=False,
        convert_numerals=False,
        processor=None,
        normalise_audio=False,
        augment_audio=True,
    )

    assert len(created) == 1
    assert isinstance(created[0].audio, SoundfileAudio)
    assert created[0].sample_rate == 16_000
    assert processed["audio"]["array"].shape == (16_000,)
    assert not torch.allclose(processed["audio"]["array"], original_audio)


def test_process_example_skips_augmentation_when_disabled() -> None:
    """Disabling augmentation does not require background-noise files."""
    audio = torch.full((16_000,), 0.2)
    example = {
        "text": "Hello",
        "audio": {"array": audio.tolist(), "sampling_rate": 16_000},
    }

    processed = process_example(
        example=example,
        characters_to_keep=None,
        conversion_dict={},
        text_column="text",
        audio_column="audio",
        lower_case=False,
        convert_numerals=False,
        processor=None,
        normalise_audio=False,
        augment_audio=False,
    )

    assert torch.equal(processed["audio"]["array"], audio)


def test_soundfile_audio_reads_metadata_without_torchaudio_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Soundfile metadata supports torch-audiomentations on torchaudio 2.10."""
    audio_path = tmp_path / "noise.wav"
    sf.write(file=audio_path, data=[0.1] * 8_000, samplerate=8_000)
    monkeypatch.delattr(torchaudio, "info", raising=False)

    audio = SoundfileAudio(sample_rate=16_000)

    assert audio.get_num_samples(audio_path) == 16_000
