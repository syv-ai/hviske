"""Functions related to the data loading and processing."""

import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import typing as t
from collections.abc import Callable, Iterable, Sized
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Any
from unicodedata import normalize
from zipfile import ZipFile

import httpx
import torch
import torch_audiomentations as ta
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    NamedSplit,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .local_vtt import decode_vtt_audio, load_vtt_manifest
from .types import Data
from .utils import (
    NUMERAL_REGEX,
    convert_iterable_dataset_to_dataset,
    convert_numeral_to_words,
    interpret_dataset_name,
    no_datasets_progress_bars,
)

logger = logging.getLogger(__package__)


def _validate_dataset_probabilities(
    probabilities: Iterable[object], dataset_count: int
) -> list[float]:
    """Validate and normalise dataset sampling probabilities.

    Args:
        probabilities:
            Candidate probability values.
        dataset_count:
            Number of datasets being interleaved.

    Returns:
        Validated probability values as floats.

    Raises:
        ValueError:
            If the count, values, bounds, or total are invalid.
    """
    try:
        values = list(probabilities)
    except TypeError as error:
        raise ValueError("Dataset probabilities must be an iterable") from error
    if len(values) != dataset_count:
        raise ValueError(
            f"There are {dataset_count:,} datasets, but {len(values):,} "
            "probabilities were provided"
        )

    validated: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Number):
            raise ValueError(
                f"Dataset probability at index {index} must be a real number"
            )
        try:
            probability = float(t.cast(t.SupportsFloat, value))
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"Dataset probability at index {index} must be a real number"
            ) from error
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                f"Dataset probability at index {index} must be finite and in [0, 1]"
            )
        validated.append(probability)

    total = math.fsum(validated)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Dataset probabilities must sum to 1, but sum to {total}")
    return validated


def _limit_validation_dataset(
    dataset: IterableDataset, max_samples: int | None
) -> IterableDataset:
    """Apply a validation sample limit while the dataset is still iterable.

    Args:
        dataset:
            Validation examples to limit.
        max_samples:
            Maximum number of examples, or ``None`` for no limit.

    Returns:
        The limited iterable dataset.
    """
    if max_samples is None:
        return dataset
    return dataset.take(max_samples)


def _dataset_cache_identity(
    dataset_id: str,
    subset: str | None,
    split: str,
    revision: str | None,
    purpose: str,
    max_samples: int | None = None,
) -> str:
    """Build a stable cache identity for one dataset materialisation.

    Args:
        dataset_id:
            Dataset repository or local identifier.
        subset:
            Dataset subset, if any.
        split:
            Dataset split.
        revision:
            Dataset revision, if any.
        purpose:
            Cache purpose, such as validation or evaluation.
        max_samples (optional):
            Materialisation limit, if one is applied.

    Returns:
        A filesystem-safe dataset identifier containing a canonical hash.
    """
    identity = json.dumps(
        {
            "dataset_id": dataset_id,
            "max_samples": max_samples,
            "purpose": purpose,
            "revision": revision,
            "split": split,
            "subset": subset,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{dataset_id.replace('/', '--')}-{digest}"


def join_audio_and_transcripts(
    audio_dataset: Dataset | IterableDataset,
    transcript_dataset: Dataset,
    audio_join_column: str,
    transcript_join_column: str,
    transcript_text_column: str,
) -> Dataset | IterableDataset:
    """Join a streaming audio dataset to an indexed transcript dataset.

    The audio side is never materialised. The compact transcript side is indexed in
    memory, then looked up as audio examples are consumed. Missing transcript keys are
    reported when the corresponding streaming example is read.

    Args:
        audio_dataset:
            Audio dataset, normally loaded with ``streaming=True``.
        transcript_dataset:
            Compact, non-streaming transcript dataset.
        audio_join_column:
            Key column in the audio dataset.
        transcript_join_column:
            Key column in the transcript dataset.
        transcript_text_column:
            Transcript text column in the transcript dataset.

    Returns:
        The audio dataset with a ``text`` column.

    Raises:
        ValueError:
            If a configured column is absent or transcript keys are duplicated.
    """
    _require_columns(
        dataset=audio_dataset, columns=[audio_join_column], dataset_name="audio"
    )
    _require_columns(
        dataset=transcript_dataset,
        columns=[transcript_join_column, transcript_text_column],
        dataset_name="transcript",
    )
    transcript_by_key: dict[object, str] = {}
    for raw_row in transcript_dataset:
        row = t.cast(dict[str, Any], raw_row)
        key = row[transcript_join_column]
        if key in transcript_by_key:
            raise ValueError(f"Duplicate transcript key: {key!r}")
        text = row[transcript_text_column]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty transcript for key: {key!r}")
        transcript_by_key[key] = text

    def add_transcript(example: dict[str, Any]) -> dict[str, Any]:
        key = example[audio_join_column]
        if key not in transcript_by_key:
            raise ValueError(f"No transcript found for audio key: {key!r}")
        example["text"] = transcript_by_key[key]
        return example

    return t.cast(Dataset | IterableDataset, audio_dataset.map(add_transcript))


def _load_transcript_dataset(
    dataset_id: str,
    subset: str | None,
    split: str,
    revision: str | None,
    cache_dir: str | None,
) -> Dataset:
    """Load the compact transcript side without streaming.

    Returns:
        A non-streaming transcript dataset.

    Raises:
        ValueError:
            If the Hub returns a streaming or otherwise unsupported dataset.
    """
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "name": subset,
        "split": split,
        "token": os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        "streaming": False,
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }
    if revision is not None:
        kwargs["revision"] = revision
    with no_datasets_progress_bars():
        dataset = load_dataset(**kwargs)
    if not isinstance(dataset, Dataset):
        raise ValueError("The transcript dataset must be a non-streaming Dataset")
    return dataset


def _require_columns(
    dataset: Dataset | IterableDataset, columns: list[str], dataset_name: str
) -> None:
    available = set(dataset.column_names or [])
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Missing {dataset_name} dataset columns: {missing}")


def _set_source_language(
    example: dict[str, Any], language: str | None
) -> dict[str, Any]:
    example["language"] = language
    return example


# Dictionary that contains characters to be converted (from the key to the value). Some
# values contain spaces to ensure that they're separated from other characters, and
# superfluous spaces are removed later. Note also that these are converted in the order
# they appear in the dictionary.
DEFAULT_CONVERSION_DICT = {
    "aa": "å",
    "ğ": "g",
    "ñ": "n",
    "ń": "n",
    "è": "e",
    "kg": " kilo ",
    "μg": " mikrogram ",
    "hhv": "henholdsvis",
    "fx": "for eksempel",
    "f.eks.": "for eksempel",
    "-": " minus ",
    "+": " plus ",
    "μ": " mikro ",
    "§": " paragraf ",
    "%": " procent ",
    "‰": " promille ",
    "ú": "u",
    "ş": "s",
    "ê": "e",
    "ã": "a",
    "ë": "e",
    "ć": "c",
    "ä": "æ",
    "í": "i",
    "š": "s",
    "î": "i",
    "ě": "e",
    "ð": "d",
    "á": "a",
    "ó": "o",
    "þ": "th",
    "ı": "i",
    "ö": "ø",
    "ç": "c",
    "ș": "s",
    "\u0301": " ",  # Empty whitespace symbol
    "\u200b": " ",  # Empty whitespace symbol
}


FILLER_WORDS_PATTERN = re.compile(
    pattern=r"\b(eh+m*|øh+m*|h+m+|m+h+)\b", flags=re.IGNORECASE
)


def load_data_for_finetuning(
    config: DictConfig, processor: Callable | None = None
) -> IterableDatasetDict:
    """Load an audio dataset for finetuning.

    Args:
        config:
            The Hydra configuration object.
        processor (optional):
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.

    Returns:
        The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    probabilities = config.dataset_probabilities
    if probabilities is not None:
        probabilities = _validate_dataset_probabilities(
            probabilities=probabilities, dataset_count=len(config.datasets)
        )

    all_datasets: list[IterableDataset] | list[Dataset] = list()
    for dataset_name, dataset_config in config.datasets.items():
        if is_main_process:
            logger.info(f"Loading dataset {dataset_name!r}")

        is_local_vtt = dataset_config.get("type") == "local_vtt"
        transcript_dataset_id = dataset_config.get("transcript_dataset_id")

        if is_local_vtt:
            ds = load_vtt_manifest(
                manifest_path=Path(dataset_config.manifest_path),
                min_seconds=config.min_seconds_per_example,
                max_seconds=config.max_seconds_per_example,
            )
        # Load from disk if the dataset ID is a path and it is stored as an arrow
        # dataset
        elif Path(dataset_config.id).exists():
            train_path = Path(dataset_config.id) / dataset_config.train_name
            data_files = list(map(str, train_path.glob("data-*.arrow")))
            if len(data_files) == 0:
                try:
                    ds = load_dataset(
                        path=dataset_config.id,
                        name=dataset_config.subset,
                        split=dataset_config.train_name,
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )

                # In case a single split has been stored to disk, we load it directly
                except ValueError as e:
                    if "load_from_disk" not in str(e):
                        raise e
                    ds = Dataset.load_from_disk(dataset_path=dataset_config.id)
            else:
                try:
                    ds = load_dataset(
                        "arrow",
                        data_files=data_files,
                        split=dataset_config.train_name,
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )
                except ValueError:
                    ds = load_dataset(
                        "arrow",
                        data_files=data_files,
                        split="train",
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )

        # Load dataset from the Hugging Face Hub. The HUGGINGFACE_HUB_TOKEN is only
        # used during CI - normally it is expected that the user is logged in to the
        # Hugging Face Hub using the `huggingface-cli login` command.
        else:
            kwargs: dict[str, Any] = {
                "path": dataset_config.id,
                "name": dataset_config.subset,
                "split": dataset_config.train_name,
                "token": os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                "streaming": (
                    True if transcript_dataset_id is not None else config.streaming
                ),
                "cache_dir": config.cache_dir,
                "trust_remote_code": True,
            }
            if dataset_config.get("revision") is not None:
                kwargs["revision"] = dataset_config.revision
            with no_datasets_progress_bars():
                ds = load_dataset(**kwargs)

        if not isinstance(ds, Dataset | IterableDataset):
            raise ValueError(f"Unsupported dataset type: {type(ds)}")

        if not is_local_vtt and dataset_config.text_column != "text":
            ds = ds.rename_column(dataset_config.text_column, "text")
        if not is_local_vtt and dataset_config.audio_column != "audio":
            ds = ds.rename_column(dataset_config.audio_column, "audio")
        if not is_local_vtt:
            ds = ds.cast_column(
                column="audio", feature=Audio(sampling_rate=config.model.sampling_rate)
            )

        if transcript_dataset_id is not None:
            transcript = _load_transcript_dataset(
                dataset_id=transcript_dataset_id,
                subset=dataset_config.get("transcript_subset"),
                split=dataset_config.get("transcript_split", "train"),
                revision=dataset_config.get("transcript_revision"),
                cache_dir=config.cache_dir,
            )
            ds = join_audio_and_transcripts(
                audio_dataset=ds,
                transcript_dataset=transcript,
                audio_join_column=dataset_config.audio_join_column,
                transcript_join_column=dataset_config.transcript_join_column,
                transcript_text_column=dataset_config.transcript_text_column,
            )

        if not is_local_vtt:
            ds = ds.map(
                function=partial(
                    _set_source_language,
                    language=dataset_config.get("language")
                    or getattr(config.model, "language", None),
                )
            )

        if is_local_vtt:
            if ds.features is None:
                raise ValueError("Local VTT datasets must declare manifest features")
            local_features = ds.features.copy()
            local_features["audio"] = Audio(sampling_rate=config.model.sampling_rate)
            ds = ds.map(
                function=partial(
                    decode_vtt_audio, sampling_rate=config.model.sampling_rate
                ),
                features=local_features,
            )
        elif dataset_config.filter_dataset:
            ds = filter_dataset(
                dataset=ds,
                audio_column="audio",
                text_column="text",
                min_seconds_per_example=config.min_seconds_per_example,
                max_seconds_per_example=config.max_seconds_per_example,
                is_main_process=is_main_process,
                num_proc=config.dataset_num_workers,
            )

        ds = ds.remove_columns(
            column_names=[
                column
                for column in ds.column_names or list()
                if column not in ["audio", "text", "language"]
            ]
        ).shuffle(seed=config.seed)

        all_datasets.append(ds)  # type: ignore[bad-argument-type]

    if len(all_datasets) == 0:
        raise ValueError("No datasets were loaded")

    if len(all_datasets) > 1:
        if is_main_process:
            logger.info("Interleaving datasets...")
            if config.dataset_probabilities is None and len(all_datasets) > 1:
                logger.warning(
                    "No dataset probabilities were specified for the training split. "
                    "This means that each dataset will be sampled with equal "
                    "probability, which means that the smaller datasets will be "
                    "sampled more often than the larger datasets. This is probably "
                    "not what you want."
                )

        if probabilities is None:
            probabilities = [1 / len(all_datasets)] * len(all_datasets)

        train = interleave_datasets(
            datasets=all_datasets,  # type: ignore[bad-argument-type]
            probabilities=probabilities,
            seed=config.seed,
            split=NamedSplit("train"),
            stopping_strategy="all_exhausted",
        )
    else:
        train = all_datasets[0]

    train = process_dataset(
        dataset=train,
        lower_case=config.model.lower_case,
        characters_to_keep=config.model.characters_to_keep,
        text_column="text",
        audio_column="audio",
        convert_numerals=False,
        remove_input_dataset_columns=True,
        normalise_audio=True,
        augment_audio=True,
        processor=processor,
        num_proc=config.dataset_num_workers,
        language=getattr(config.model, "language", None),
        language_column="language",
        punctuation=getattr(config.model, "punctuation", True),
    )

    data_dict = dict(train=train)
    dataset = IterableDatasetDict(data_dict)

    if is_main_process:
        logger.info("Loading CoRal validation dataset...")

    def load_validation_dataset(dataset_config: DictConfig) -> Dataset:
        """Load a validation dataset.

        Args:
            dataset_config:
                The config for the dataset to load.

        Returns:
            The loaded dataset.

        Raises:
            ValueError:
                If the loaded dataset is not iterable.
        """
        validation_split = dataset_config.val_name
        validation_kwargs: dict[str, Any] = {
            "path": dataset_config.id,
            "name": dataset_config.subset,
            "split": validation_split,
            "token": os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            "streaming": True,
            "cache_dir": config.cache_dir,
            "trust_remote_code": True,
        }
        if dataset_config.get("revision") is not None:
            validation_kwargs["revision"] = dataset_config.revision
        with no_datasets_progress_bars():
            val = load_dataset(**validation_kwargs)
        if not isinstance(val, IterableDataset):
            raise ValueError(f"Unsupported validation dataset type: {type(val)}")
        max_samples = config.get("max_validation_samples_per_dataset")
        val = _limit_validation_dataset(dataset=val, max_samples=max_samples)
        val = convert_iterable_dataset_to_dataset(
            iterable_dataset=val,
            split_name=validation_split,
            dataset_id=_dataset_cache_identity(
                dataset_id=dataset_config.id,
                subset=dataset_config.get("subset"),
                split=validation_split,
                revision=dataset_config.get("revision"),
                purpose="validation",
                max_samples=max_samples,
            ),
            cache_dir=config.cache_dir,
        )
        if dataset_config.text_column != "text":
            val = val.rename_column(dataset_config.text_column, "text")
        if dataset_config.audio_column != "audio":
            val = val.rename_column(dataset_config.audio_column, "audio")
        val = val.cast_column(
            column="audio", feature=Audio(sampling_rate=config.model.sampling_rate)
        ).select_columns(column_names=["text", "audio"])
        return val.map(
            function=partial(
                _set_source_language,
                language=dataset_config.get("language")
                or getattr(config.model, "language", None),
            )
        )

    vals = [
        load_validation_dataset(dataset_config=dataset_config)
        for dataset_config in config.evaluation_datasets
    ]
    vals = [
        filter_dataset(
            dataset=val,
            audio_column="audio",
            text_column="text",
            min_seconds_per_example=config.min_seconds_per_example,
            max_seconds_per_example=config.max_seconds_per_example,
            is_main_process=is_main_process,
            num_proc=config.dataset_num_workers,
        )
        for val in vals
    ]
    vals = [
        process_dataset(
            dataset=val,
            lower_case=config.evaluation_lower_case,
            characters_to_keep=config.evaluation_characters_to_keep,
            text_column="text",
            audio_column="audio",
            convert_numerals=False,
            remove_input_dataset_columns=True,
            normalise_audio=True,
            augment_audio=False,
            processor=processor,
            num_proc=config.dataset_num_workers,
            language=getattr(config.model, "language", None),
            language_column="language",
            punctuation=getattr(config.model, "punctuation", True),
        )
        for val in vals
    ]
    for dataset_config, split in zip(config.evaluation_datasets, vals):
        split_name = f"val_{dataset_config.id.split('/')[-1].lower().replace('-', '_')}"
        if dataset_config.subset is not None:
            split_name += f"_{dataset_config.subset.lower().replace('-', '_')}"
        dataset[split_name] = split

    return dataset


def load_dataset_for_evaluation(config: DictConfig) -> Dataset:
    """Load the evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DatasetDict containing the validation and test datasets.

    Raises:
        ValueError:
            If the loaded dataset cannot be streamed or materialised.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    dataset_id, dataset_subset, dataset_revision = interpret_dataset_name(
        dataset_name=config.dataset
    )

    if is_main_process:
        logger.info(
            f"Loading the {config.eval_split_name!r} split of the {dataset_id} "
            "dataset..."
        )

    eval_dataset_path = None
    cache_identity = _dataset_cache_identity(
        dataset_id=dataset_id,
        subset=dataset_subset,
        split=config.eval_split_name,
        revision=dataset_revision,
        purpose="evaluation",
    )
    if config.cache_dir:
        eval_dataset_path = Path(config.cache_dir) / "test-sets" / cache_identity
        if eval_dataset_path.exists():
            return Dataset.load_from_disk(dataset_path=eval_dataset_path)

    dataset = load_dataset(
        path=dataset_id,
        name=dataset_subset,
        split=config.eval_split_name,
        revision=dataset_revision,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        cache_dir=config.cache_dir,
        streaming=True,
        trust_remote_code=True,
    )
    if not isinstance(dataset, IterableDataset):
        raise ValueError(f"Unsupported evaluation dataset type: {type(dataset)}")
    dataset = convert_iterable_dataset_to_dataset(
        iterable_dataset=dataset,
        split_name=config.eval_split_name,
        dataset_id=(
            f"test-sets/{cache_identity}" if eval_dataset_path is not None else None
        ),
        cache_dir=config.cache_dir,
    )
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Unsupported materialised dataset type: {type(dataset)}")
    dataset = filter_dataset(
        dataset=dataset,
        audio_column=config.audio_column,
        text_column=config.text_column,
        min_seconds_per_example=config.min_seconds_per_example,
        max_seconds_per_example=config.max_seconds_per_example,
        is_main_process=is_main_process,
    )
    dataset = dataset.cast_column(
        column=config.audio_column, feature=Audio(sampling_rate=config.sampling_rate)
    )
    dataset = process_dataset(
        dataset=dataset,
        lower_case=config.lower_case,
        characters_to_keep=config.characters_to_keep,
        text_column=config.text_column,
        audio_column=config.audio_column,
        normalise_audio=True,
        augment_audio=False,
        remove_input_dataset_columns=False,
        convert_numerals=True,
    )

    if eval_dataset_path is not None:
        dataset.save_to_disk(dataset_path=eval_dataset_path)

    return dataset


def filter_dataset(
    dataset: Data,
    audio_column: str,
    text_column: str,
    min_seconds_per_example: int | float,
    max_seconds_per_example: int,
    is_main_process: bool,
    num_proc: int | None = None,
) -> Data:
    """Filter the dataset.

    Note that this removes samples from the dataset.

    Args:
        dataset:
            The dataset to filter.
        audio_column:
            The name of the column containing the audio.
        text_column:
            The name of the column containing the transcriptions.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.
        is_main_process:
            Whether the current process is the main process.
        num_proc (optional):
            The number of processes to use for filtering the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.

    Returns:
        The filtered dataset.

    Raises:
        ValueError:
            If the filtered dataset type is unsupported.
    """
    num_samples_before = len(dataset) if isinstance(dataset, Sized) else 0

    filter_fn = partial(
        filter_example,
        text_column=text_column,
        audio_column=audio_column,
        min_seconds_per_example=min_seconds_per_example,
        max_seconds_per_example=max_seconds_per_example,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        filtered = dataset.filter(
            function=filter_fn,
            num_proc=num_proc,
            desc="Filtering dataset",
            keep_in_memory=True,
        )
    else:
        filtered = dataset.filter(function=filter_fn)

    # Add info back in the filtered dataset, as it gets removed after calling `filter`
    if isinstance(dataset, Dataset | IterableDataset) and isinstance(
        filtered, Dataset | IterableDataset
    ):
        filtered.info.features = dataset.info.features
    else:
        if not (
            isinstance(dataset, DatasetDict | IterableDatasetDict)
            and isinstance(filtered, DatasetDict | IterableDatasetDict)
        ):
            raise ValueError(f"Unsupported filtered dataset type: {type(filtered)}")
        for split_name in dataset.keys():
            filtered[split_name].info.features = dataset[split_name].info.features

    if isinstance(dataset, Sized) and isinstance(filtered, Sized) and is_main_process:
        num_samples_removed = num_samples_before - len(filtered)
        logger.info(f"Removed {num_samples_removed:,} samples from the dataset")

    return filtered  # type: ignore[bad-return]


def filter_example(
    sample: dict[str, Any],
    audio_column: str,
    text_column: str,
    min_seconds_per_example: int | float,
    max_seconds_per_example: int,
) -> bool:
    """Filter samples based on the validation status.

    Args:
        sample:
            The sample to filter.
        audio_column:
            The name of the column containing the audio.
        text_column:
            The name of the column containing the transcriptions.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can

    Returns:
        Whether the sample should be kept.
    """
    # Filtering based on audio
    audio = sample[audio_column]
    if audio["array"].shape[0] <= audio["sampling_rate"] * min_seconds_per_example:
        return False
    if audio["array"].shape[0] >= audio["sampling_rate"] * max_seconds_per_example:
        return False

    # Filtering based on text
    if len(sample[text_column].strip()) == 0:
        return False

    # Filtering based on validation
    if "validated" in sample and sample["validated"] == "rejected":
        return False

    return True


def process_dataset(
    dataset: Data,
    lower_case: bool,
    characters_to_keep: Iterable[str] | None,
    text_column: str,
    remove_input_dataset_columns: bool,
    audio_column: str | None,
    convert_numerals: bool,
    normalise_audio: bool,
    augment_audio: bool,
    num_proc: int | None = None,
    processor: Callable | None = None,
    language: str | None = None,
    language_column: str | None = None,
    punctuation: bool = True,
) -> Data:
    """Process the dataset.

    Note that this does not remove any samples from the dataset.

    Args:
        dataset:
            The dataset to be cleaned.
        lower_case:
            Whether to make the text lower case.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept.
        text_column:
            The name of the column containing the text.
        remove_input_dataset_columns:
            Whether to remove all input dataset columns from the output dataset.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        convert_numerals:
            Whether to convert numerals to words.
        normalise_audio:
            Whether to normalise the audio.
        augment_audio:
            Whether to augment the audio.
        num_proc (optional):
            The number of processes to use for processing the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.
        processor (optional):
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.
        language (optional):
            The default language prompt for a prompt-aware processor. Defaults to
            `None`.
        language_column (optional):
            The input column containing a per-example language prompt. Defaults to
            `None`.
        punctuation (optional):
            Whether to enable punctuation in a prompt-aware processor. Defaults to
            `True`.

    Returns:
        The cleaned dataset.

    Raises:
        ValueError:
            If the dataset type is not supported.
    """
    if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
        column_names = dataset.column_names
    elif isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        column_names = dataset["train"].column_names
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    map_fn = partial(
        process_example,
        characters_to_keep=characters_to_keep,
        conversion_dict=DEFAULT_CONVERSION_DICT,
        text_column=text_column,
        audio_column=audio_column,
        lower_case=lower_case,
        convert_numerals=convert_numerals,
        processor=processor,
        normalise_audio=normalise_audio,
        augment_audio=augment_audio,
        language=language,
        language_column=language_column,
        punctuation=punctuation,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        mapped = dataset.map(
            function=map_fn,
            num_proc=num_proc,
            desc="Processing dataset",
            remove_columns=column_names if remove_input_dataset_columns else None,
        )
    else:
        mapped = dataset.map(function=map_fn, remove_columns=column_names)

    return mapped  # type: ignore[bad-return]


def process_example(
    example: dict,
    characters_to_keep: Iterable[str] | None,
    conversion_dict: dict[str, str],
    text_column: str,
    audio_column: str | None,
    lower_case: bool,
    convert_numerals: bool,
    processor: Callable | None,
    normalise_audio: bool,
    augment_audio: bool,
    language: str | None = None,
    language_column: str | None = None,
    punctuation: bool = True,
) -> dict:
    """Helper function which cleans a single example.

    Args:
        example:
            The example to be cleaned.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept.
        conversion_dict:
            A dictionary of characters to be converted.
        text_column:
            The name of the column containing the text.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        lower_case:
            Whether to make the text lower case.
        convert_numerals:
            Whether to convert numerals to words.
        processor:
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Requires `audio_column` to be specified.
        normalise_audio:
            Whether to normalise the audio.
        augment_audio:
            Whether to augment the audio.
        language (optional):
            The default language prompt for a prompt-aware processor. Defaults to
            `None`.
        language_column (optional):
            The input column containing a per-example language prompt. Defaults to
            `None`.
        punctuation (optional):
            Whether to enable punctuation in a prompt-aware processor. Defaults to
            `True`.

    Returns:
        The cleaned example.
    """
    doc = example[text_column]

    if convert_numerals and re.search(pattern=NUMERAL_REGEX, string=doc):
        doc = "".join(
            convert_numeral_to_words(numeral=maybe_numeral)
            for maybe_numeral in re.split(pattern=NUMERAL_REGEX, string=doc)
            if maybe_numeral is not None
        )

    if lower_case:
        doc = doc.lower()

    # Remove filler words such as "ehh"
    doc = FILLER_WORDS_PATTERN.sub(repl="", string=doc)

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    doc = normalize("NFKC", doc)

    # Convert known symbols
    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    # Remove all non-standard characters
    if characters_to_keep is not None:
        characters_to_keep = "".join(char for char in characters_to_keep)
        non_standard_characters_regex = re.compile(
            f"[^{re.escape(characters_to_keep + ' |')}]", flags=re.IGNORECASE
        )
        doc = re.sub(non_standard_characters_regex, " ", doc.strip())

    # Replace superfluous spaces
    doc = re.sub(r" +", " ", doc)

    # Strip each newline
    doc = "\n".join([line.strip() for line in doc.split("\n")]).strip("\n")

    # Re-assign the cleaned transcription
    example[text_column] = doc

    # If we do not have any audio, then we return the example, as the remainder of the
    # function concerns audio processing
    if audio_column is None:
        return example

    # Extract audio from example
    audio = example[audio_column]
    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]

    # Normalise and augment audio
    download_background_noises()
    normalise = ta.PeakNormalization(p=1.0) if normalise_audio else ta.Identity()
    augment = (
        ta.Compose(
            [
                ta.PeakNormalization(p=1.0),
                ta.Gain(p=1.0),
                ta.AddBackgroundNoise(
                    background_paths=Path("background-noises"), p=0.7
                ),
                ta.AddColoredNoise(p=0.2),
                ta.OneOf(
                    [
                        ta.BandPassFilter(p=1.0),
                        ta.BandStopFilter(p=1.0),
                        ta.HighPassFilter(p=1.0),
                        ta.LowPassFilter(p=1.0),
                    ],
                    p=0.2,
                ),
            ],
            p=1.0,
        )
        if augment_audio
        else ta.Identity()
    )
    normalise_and_augment = ta.Compose([normalise, augment], p=1.0)
    audio_array = normalise_and_augment(
        torch.tensor(audio_array).unsqueeze(0).unsqueeze(0), sample_rate=sampling_rate
    )[0, 0]

    # If we don't have a processor then we just re-assign the normalised audio and
    # return the processed example
    if processor is None:
        example[audio_column]["array"] = audio_array
        return example

    # Cohere ASR needs the language prompt and transcript in the same processor call.
    example_language = (
        example.get(language_column) if language_column is not None else None
    ) or language
    if example_language is not None and hasattr(processor, "get_decoder_prompt_ids"):
        processed = processor(
            audio_array,
            language=example_language,
            text=example[text_column],
            punctuation=punctuation,
            sampling_rate=sampling_rate,
        )
        example["input_features"] = processed["input_features"][0]
        example["attention_mask"] = processed["attention_mask"][0]
        example["decoder_input_ids"] = processed["decoder_input_ids"][0]
        example["labels"] = processed["labels"][0]
        example["input_length"] = len(example["labels"])
        example["num_seconds"] = len(example["attention_mask"]) / 100
        return example

    # Process the audio for Whisper and Wav2Vec2.
    processed = processor(audio_array, sampling_rate=sampling_rate)
    audio_feature_name = (
        "input_values" if "input_values" in processed else "input_features"
    )
    audio_array = processed[audio_feature_name][0]
    example[audio_feature_name] = audio_array
    example["num_seconds"] = len(example[audio_feature_name]) / sampling_rate

    # Process the labels
    example["labels"] = processor(text=example[text_column], truncation=True).input_ids
    example["input_length"] = len(example["labels"])

    return example


def download_background_noises() -> None:
    """Download background noises for audio augmentation.

    This function downloads the background noises to the `background-noises` directory,
    and will do nothing if the directory already exists.
    """
    background_noises_path = Path("background-noises")
    if background_noises_path.exists():
        return

    logger.info("Downloading background noises from the ESC-50 dataset...")

    # Download the ESC-50 dataset zip file as a stream
    zip_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    chunks = []
    with httpx.stream(method="GET", url=zip_url, follow_redirects=True) as response:
        for chunk in tqdm(
            response.iter_bytes(),
            desc="Downloading ESC-50 dataset",
            unit="B",
            unit_scale=True,
            total=int(response.headers.get("Content-Length", 0)),
        ):
            chunks.append(chunk)
    content = b"".join(chunks)

    # Unzip only the audio files from the ESC-50 dataset
    with ZipFile(file=io.BytesIO(content)) as zip_file:
        audio_files = [
            file_info
            for file_info in zip_file.infolist()
            if file_info.filename.startswith("ESC-50-master/audio/")
        ]
        zip_file.extractall(members=audio_files, path=background_noises_path)

    # Move audio files to the root of the background-noises directory
    extracted_audio_path = background_noises_path / "ESC-50-master" / "audio"
    for audio_file in extracted_audio_path.iterdir():
        audio_file.rename(background_noises_path / audio_file.name)

    # Remove the extracted directories
    shutil.rmtree(background_noises_path / "ESC-50-master")

    logger.info("Background noises downloaded successfully.")
