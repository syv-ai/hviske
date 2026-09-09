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
from collections.abc import Callable, Iterable, Mapping, Sized
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
    Features,
    IterableDataset,
    IterableDatasetDict,
    NamedSplit,
    Sequence,
    Value,
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


def _load_transcript_dataset(
    dataset_id: str,
    subset: str | None,
    split: str,
    revision: str | None,
    cache_dir: str | None,
    trust_remote_code: bool = False,
    dataset_loader: Callable[..., object] = load_dataset,
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
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        kwargs["revision"] = revision
    with no_datasets_progress_bars():
        dataset = dataset_loader(**kwargs)
    if isinstance(dataset, Dataset):
        return dataset
    try:
        rows = list(t.cast(Iterable[object], dataset))
    except TypeError as error:
        raise ValueError("The transcript dataset must be iterable") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("The transcript dataset rows must be mappings")
    return Dataset.from_list(t.cast(list[dict[str, object]], rows))


def _set_source_language(
    example: dict[str, Any], language: str | None
) -> dict[str, Any]:
    example["language"] = language or ""
    return example


def _standardise_training_dataset(
    dataset: Dataset | IterableDataset, sampling_rate: int
) -> Dataset | IterableDataset:
    """Drop source metadata and cast a source to the interleave schema.

    Args:
        dataset:
            Dataset to normalise.
        sampling_rate:
            Sampling rate for the shared audio feature.

    Returns:
        Dataset with the ``audio``, ``text`` and ``language`` columns.

    Raises:
        ValueError:
            If a required training column is missing.
    """
    required_columns = {"audio", "text", "language"}
    available_columns = set(dataset.column_names or [])
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(
            "Training dataset is missing columns: " + ", ".join(sorted(missing_columns))
        )
    features = _standard_training_features(sampling_rate=sampling_rate)
    standardised = dataset.select_columns(["audio", "text", "language"]).cast(features)
    if isinstance(standardised, IterableDataset):
        standardised = standardised.map(function=lambda example: example)
        standardised.info.features = features
    return standardised


def _standard_training_features(sampling_rate: int) -> Features:
    """Return the exact schema shared by every training source."""
    return Features(
        audio=Audio(sampling_rate=sampling_rate),
        text=Value("string"),
        language=Value("string"),
    )


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
    key_type: type[object] | None = None
    for raw_row in transcript_dataset:
        row = t.cast(dict[str, Any], raw_row)
        key = row[transcript_join_column]
        _validate_join_key(key=key, expected_type=key_type, side="transcript")
        if key_type is None:
            key_type = type(key)
        if key in transcript_by_key:
            raise ValueError(f"Duplicate transcript key: {key!r}")
        text = row[transcript_text_column]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty transcript for key: {key!r}")
        transcript_by_key[key] = text

    def add_transcript(example: dict[str, Any]) -> dict[str, Any]:
        key = example[audio_join_column]
        _validate_join_key(key=key, expected_type=key_type, side="audio")
        if key not in transcript_by_key:
            raise ValueError(f"No transcript found for audio key: {key!r}")
        example["text"] = transcript_by_key[key]
        return example

    if audio_dataset.features is None:
        raise ValueError("Audio dataset must declare features")
    joined_features = audio_dataset.features.copy()
    joined_features["text"] = Value("string")
    return t.cast(
        Dataset | IterableDataset,
        audio_dataset.map(add_transcript, features=joined_features),
    )


def _require_columns(
    dataset: Dataset | IterableDataset, columns: list[str], dataset_name: str
) -> None:
    available = set(dataset.column_names or [])
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Missing {dataset_name} dataset columns: {missing}")


def _validate_join_key(
    key: object, expected_type: type[object] | None, side: str
) -> None:
    """Validate that a join key can be indexed and has the expected type.

    Raises:
        ValueError:
            If the key is unhashable or has an incompatible type.
    """
    try:
        hash(key)
    except TypeError as error:
        raise ValueError(f"The {side} join key must be hashable: {key!r}") from error
    if expected_type is not None and type(key) is not expected_type:
        raise ValueError(
            f"The {side} join key has type {type(key).__name__}, expected "
            f"{expected_type.__name__}"
        )


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
                "trust_remote_code": dataset_config.get("trust_remote_code", False),
            }
            if dataset_config.get("revision") is not None:
                kwargs["revision"] = dataset_config.revision
            with no_datasets_progress_bars():
                ds = load_dataset(**kwargs)

        if not isinstance(ds, Dataset | IterableDataset):
            raise ValueError(f"Unsupported dataset type: {type(ds)}")

        row_filters = dataset_config.get("filters")
        if row_filters is not None:
            ds = _filter_dataset_rows(
                dataset=ds, filters=t.cast(Mapping[str, object], row_filters)
            )

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
                trust_remote_code=dataset_config.get(
                    "transcript_trust_remote_code", False
                ),
            )
            ds = join_audio_and_transcripts(
                audio_dataset=ds,
                transcript_dataset=transcript,
                audio_join_column=dataset_config.audio_join_column,
                transcript_join_column=dataset_config.transcript_join_column,
                transcript_text_column=dataset_config.transcript_text_column,
            )

        if not is_local_vtt:
            if ds.features is None:
                raise ValueError("Hub datasets must declare features")
            language_features = ds.features.copy()
            language_features["language"] = Value("string")
            ds = ds.map(
                function=partial(
                    _set_source_language,
                    language=dataset_config.get("language")
                    or getattr(config.model, "language", None),
                ),
                features=language_features,
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
            ds = _standardise_training_dataset(
                dataset=ds, sampling_rate=config.model.sampling_rate
            )
            ds = filter_dataset(
                dataset=ds,
                audio_column="audio",
                text_column="text",
                min_seconds_per_example=config.min_seconds_per_example,
                max_seconds_per_example=config.max_seconds_per_example,
                is_main_process=is_main_process,
                num_proc=config.dataset_num_workers,
            )

        ds = _standardise_training_dataset(
            dataset=ds, sampling_rate=config.model.sampling_rate
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
            "trust_remote_code": dataset_config.get("trust_remote_code", False),
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


def _filter_dataset_rows(dataset: Data, filters: Mapping[str, object]) -> Data:
    """Keep rows whose configured columns match exact scalar values.

    Args:
        dataset:
            The dataset to filter.
        filters:
            Mapping from feature names to the exact values to retain.

    Returns:
        The filtered dataset with its original explicit features.

    Raises:
        ValueError:
            If a filter column is not present in the dataset features.
    """
    if isinstance(dataset, Dataset | IterableDataset):
        return t.cast(
            Data, _filter_dataset_rows_from_split(dataset=dataset, filters=filters)
        )
    if isinstance(dataset, DatasetDict):
        return t.cast(
            Data,
            DatasetDict(
                {
                    split_name: _filter_dataset_rows_from_split(
                        dataset=split_dataset, filters=filters
                    )
                    for split_name, split_dataset in dataset.items()
                }
            ),
        )
    if isinstance(dataset, IterableDatasetDict):
        return t.cast(
            Data,
            IterableDatasetDict(
                {
                    split_name: _filter_dataset_rows_from_split(
                        dataset=split_dataset, filters=filters
                    )
                    for split_name, split_dataset in dataset.items()
                }
            ),
        )
    raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def _filter_dataset_rows_from_split(
    dataset: Dataset | IterableDataset, filters: Mapping[str, object]
) -> Dataset | IterableDataset:
    """Filter one dataset split while retaining its declared feature schema.

    Args:
        dataset:
            The dataset split to filter.
        filters:
            Mapping from feature names to the exact values to retain.

    Returns:
        The filtered dataset with its original explicit features.

    Raises:
        ValueError:
            If the dataset has no declared features or a filter column is missing.
    """
    if dataset.features is None:
        raise ValueError("Cannot apply row filters without declared dataset features")
    missing_columns = [column for column in filters if column not in dataset.features]
    if missing_columns:
        raise ValueError(
            "Cannot apply row filters; missing dataset features: "
            + ", ".join(missing_columns)
        )

    original_features = dataset.features
    filtered = dataset.filter(
        function=lambda row: all(
            row[column] == expected_value for column, expected_value in filters.items()
        )
    )
    filtered.info.features = original_features
    return filtered


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
        filtered = t.cast(Dataset | DatasetDict, dataset).filter(
            function=filter_fn,
            num_proc=num_proc,
            desc="Filtering dataset",
            keep_in_memory=True,
        )
    else:
        filtered = t.cast(IterableDataset | IterableDatasetDict, dataset).filter(
            function=filter_fn
        )

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

    return t.cast(Data, filtered)


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
        column_names = t.cast(Dataset | IterableDataset, dataset).column_names
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
    mapped_features = _processing_features(
        processor=processor, remove_input_dataset_columns=remove_input_dataset_columns
    )
    if isinstance(dataset, Dataset | DatasetDict):
        mapped = t.cast(Dataset | DatasetDict, dataset).map(
            function=map_fn,
            num_proc=num_proc,
            desc="Processing dataset",
            remove_columns=column_names if remove_input_dataset_columns else None,
            features=mapped_features,
        )
    elif isinstance(dataset, IterableDataset):
        mapped = dataset.map(
            function=map_fn,
            remove_columns=column_names if remove_input_dataset_columns else None,
            features=mapped_features,
        )
    else:
        iterable_dataset_dict = t.cast(IterableDatasetDict, dataset)
        mapped = IterableDatasetDict(
            {
                split: split_dataset.map(
                    function=map_fn,
                    remove_columns=(
                        column_names if remove_input_dataset_columns else None
                    ),
                    features=mapped_features,
                )
                for split, split_dataset in iterable_dataset_dict.items()
            }
        )

    return t.cast(Data, mapped)


def _processing_features(
    processor: Callable | None, remove_input_dataset_columns: bool
) -> Features | None:
    """Describe stable map output features where the processor contract allows it.

    Args:
        processor:
            Optional processor whose output schema is being mapped.
        remove_input_dataset_columns:
            Whether the map removes the source columns.

    Returns:
        Explicit output features when the processor contract is known, otherwise
        ``None``.
    """
    if not remove_input_dataset_columns:
        return None
    if processor is None:
        return None
    if not hasattr(processor, "get_decoder_prompt_ids"):
        return Features(
            input_values=Sequence(Value("float64")),
            labels=Sequence(Value("int64")),
            input_length=Value("int64"),
            num_seconds=Value("float64"),
        )
    return Features(
        input_features=Sequence(Sequence(Value("float64"))),
        attention_mask=Sequence(Value("int64")),
        decoder_input_ids=Sequence(Value("int64")),
        labels=Sequence(Value("int64")),
        input_length=Value("int64"),
        num_seconds=Value("float64"),
    )


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
        trust_remote_code=config.get("trust_remote_code", False),
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
        example["input_features"] = _to_python(processed["input_features"][0])
        example["attention_mask"] = _to_python(processed["attention_mask"][0])
        example["decoder_input_ids"] = _to_python(processed["decoder_input_ids"][0])
        example["labels"] = _to_python(processed["labels"][0])
        labels = t.cast(Sized, example["labels"])
        attention_mask = t.cast(Sized, example["attention_mask"])
        example["input_length"] = len(labels)
        example["num_seconds"] = len(attention_mask) / 100
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


def _to_python(value: object) -> object:
    """Convert tensor-like processor output to a serialisable Python value.

    Args:
        value:
            Processor output value.

    Returns:
        A Python list when the value supports ``tolist``, otherwise the value itself.
    """
    if not hasattr(value, "tolist"):
        return value
    to_list = t.cast(Callable[[], object], getattr(value, "tolist"))
    return to_list()


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
