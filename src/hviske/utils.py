"""General utility functions."""

import collections.abc as c
import contextlib
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import warnings
from functools import partialmethod
from pathlib import Path
from types import TracebackType

import datasets.utils.logging as ds_logging
import tqdm as tqdm_package
import transformers.utils.logging as hf_logging
from datasets import (
    Dataset,
    IterableDataset,
    NamedSplit,
    disable_progress_bar,
    enable_progress_bar,
)
from huggingface_hub import CommitInfo, HfApi, upload_folder
from huggingface_hub.errors import RepositoryNotFoundError
from tqdm.auto import tqdm
from transformers.trainer import Trainer

logger = logging.getLogger(__package__)


NUMERAL_REGEX = re.compile(r"\b(0|[1-9]\d{0,2}(?:(?:\.\d{3})*|\d*)(?:,\d+)?)\b")


_MODEL_ARTEFACT_NAMES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "decoder_config.json",
        "feature_extractor_config.json",
        "generation_config.json",
        "merges.txt",
        "normalizer.json",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "chat_template.jinja",
    }
)
_SHARDED_MODEL_ARTEFACT = re.compile(
    r"(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:bin|safetensors)\Z"
)
_FULL_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


def block_terminal_output() -> None:
    """Blocks undesired terminal output."""
    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Disable logging from Hugging Face libraries
    ds_logging.set_verbosity_error()
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
    logging.getLogger("transformers.models.whisper").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils").setLevel(logging.ERROR)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def convert_iterable_dataset_to_dataset(
    iterable_dataset: IterableDataset,
    split_name: str = "train",
    dataset_id: str | None = None,
    cache_dir: Path | None = None,
) -> Dataset:
    """Convert an IterableDataset to a Dataset.

    Args:
        iterable_dataset:
            The IterableDataset to convert.
        split_name (optional):
            The name of the split. Defaults to "train".
        dataset_id (optional):
            The ID of the dataset, which is used to store and re-load the dataset. If
            None then the dataset is not stored. Defaults to None.
        cache_dir (optional):
            The directory to store the dataset. If None then the default cache
            `~/.cache/huggingface/datasets` is used. Defaults to None.

    Returns:
        The converted Dataset.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"

    dataset_dir = None
    if dataset_id is not None:
        dataset_dir = Path(cache_dir) / dataset_id
        if dataset_dir.exists():
            return Dataset.load_from_disk(str(dataset_dir))

    splits_info = iterable_dataset.info.splits
    num_examples = None if splits_info is None else splits_info[split_name].num_examples

    def gen_from_iterable_dataset() -> c.Generator[dict, None, None]:
        yield from tqdm(  # type: ignore[invalid-yield]
            iterable=iterable_dataset,
            total=num_examples,
            desc="Converting iterable dataset to regular dataset",
        )

    with no_datasets_progress_bars():
        dataset = Dataset.from_generator(
            generator=gen_from_iterable_dataset,
            features=iterable_dataset.features,
            split=NamedSplit(name=split_name),
            num_proc=mp.cpu_count(),
        )
    assert isinstance(dataset, Dataset)

    if dataset_dir is not None:
        dataset_dir.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(str(dataset_dir))

    return dataset


class no_datasets_progress_bars:
    """Context manager that disables the `datasets` progress bars."""

    def __enter__(self) -> None:
        """Disable the progress bar."""
        disable_progress_bar()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Re-enable the progress bar."""
        enable_progress_bar()


def convert_numeral_to_words(numeral: str, inside_larger_numeral: bool = False) -> str:
    """Convert numerals to words.

    Args:
        numeral:
            The numeral to convert.
        inside_larger_numeral (optional):
            Whether the numeral is inside a larger numeral. For instance, if `numeral`
            is 10, but is part of the larger numeral 1,010, then this should be `True`.

    Returns:
        The text with numerals converted to words.
    """
    if re.fullmatch(pattern=NUMERAL_REGEX, string=numeral) is None:
        return numeral

    numeral = numeral.replace(".", "")

    if "," in numeral:
        assert numeral.count(",") == 1, f"Too many commas in {numeral!r}"
        major, minor = numeral.split(",")
        major = convert_numeral_to_words(numeral=major)
        minor = " ".join(convert_numeral_to_words(numeral=char) for char in minor)
        return f"{major} komma {minor.replace('en', 'et')}"

    match len(numeral):
        case 1:
            mapping = {
                "0": "nul",
                "1": "en",
                "2": "to",
                "3": "tre",
                "4": "fire",
                "5": "fem",
                "6": "seks",
                "7": "syv",
                "8": "otte",
                "9": "ni",
            }
            result = mapping[numeral]

        case 2:
            mapping = {
                "10": "ti",
                "11": "elleve",
                "12": "tolv",
                "13": "tretten",
                "14": "fjorten",
                "15": "femten",
                "16": "seksten",
                "17": "sytten",
                "18": "atten",
                "19": "nitten",
                "20": "tyve",
                "30": "tredive",
                "40": "fyrre",
                "50": "halvtreds",
                "60": "tres",
                "70": "halvfjerds",
                "80": "firs",
                "90": "halvfems",
            }
            if numeral in mapping:
                return mapping[numeral]
            minor = convert_numeral_to_words(
                numeral=numeral[1], inside_larger_numeral=True
            )
            major = convert_numeral_to_words(
                numeral=numeral[0] + "0", inside_larger_numeral=True
            )
            result = f"{minor}og{major}"

        case 3:
            mapping = {"100": "hundrede"}
            if not inside_larger_numeral and numeral in mapping:
                return mapping[numeral]
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            ).replace("en", "et")
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "hundrede"
            if minor:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 4:
            mapping = {"1000": "tusind"}
            if not inside_larger_numeral and numeral in mapping:
                return mapping[numeral]
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            ).replace("en", "et")
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}".strip()

        case 5:
            major = convert_numeral_to_words(
                numeral=numeral[:2], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 6:
            major = convert_numeral_to_words(
                numeral=numeral[:3], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 7:
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "million" if int(numeral[0]) == 1 else "millioner"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 8:
            major = convert_numeral_to_words(
                numeral=numeral[:2], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "millioner"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 9:
            major = convert_numeral_to_words(
                numeral=numeral[:3], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "millioner"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case _:
            logger.warning(
                "Cannot convert numerals greater than 999,999,999 to words. Received "
                f"{numeral!r}"
            )
            return numeral

    return re.sub(r" +", " ", result).strip()


@contextlib.contextmanager
def disable_tqdm() -> c.Generator[None, None, None]:
    """Context manager to disable tqdm."""

    def _patch(old_init: c.Callable[..., None]) -> partialmethod:
        return partialmethod(old_init, disable=True)

    with monkeypatched(tqdm_package.std.tqdm, "__init__", _patch):
        yield


@contextlib.contextmanager
def monkeypatched(
    obj: object, name: str, patch: c.Callable
) -> c.Generator[None, None, None]:
    """Temporarily monkeypatch.

    Args:
        obj:
            The object to monkeypatch.
        name:
            The name of the attribute to monkeypatch.
        patch:
            The patch to apply.
    """
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


def interpret_dataset_name(dataset_name: str) -> tuple[str, str | None, str | None]:
    """Interpret the dataset name.

    This extracts the dataset ID, dataset subset and dataset revision from the dataset
    name.

    Args:
        dataset_name:
            The name of the dataset.

    Returns:
        A triple (dataset_id, dataset_subset, dataset_revision) where:
            dataset_id:
                The ID of the dataset.
            dataset_subset:
                The subset of the dataset, which can be None if the default subset
                should be used.
            dataset_revision:
                The revision of the dataset, which can be None if the newest revision
                should be used.
    """
    if ":" in dataset_name and "::" not in dataset_name:
        dataset_name = dataset_name.replace(":", "::")

    assert dataset_name.count("@") <= 1, (
        "You cannot include more than one '@' in the dataset name"
    )
    assert dataset_name.count("::") <= 1, (
        "You cannot include more than one ':' in the dataset name"
    )

    dataset_id = dataset_name
    dataset_subset = None
    dataset_revision = None

    if "@" in dataset_name:
        dataset_id_and_dataset_subset, dataset_revision_and_dataset_subset = (
            dataset_name.split("@")
        )
        if "::" in dataset_id_and_dataset_subset:
            dataset_id, dataset_subset = dataset_id_and_dataset_subset.split("::")
        else:
            dataset_id = dataset_id_and_dataset_subset
            dataset_subset = None
        if "::" in dataset_revision_and_dataset_subset:
            dataset_id, dataset_subset = dataset_revision_and_dataset_subset.split("::")
        else:
            dataset_revision = dataset_revision_and_dataset_subset

    if "::" in dataset_name:
        dataset_id, dataset_subset = dataset_name.split("::")
        if "@" in dataset_subset:
            dataset_subset, dataset_revision = dataset_subset.split("@")
        else:
            dataset_revision = None

    return dataset_id, dataset_subset, dataset_revision


def publish_model_folder(
    folder_path: str | Path,
    repo_id: str,
    finetuned_from: str,
    private: bool,
    model_card_languages: list[str],
    commit_message: str = "Publish private model",
    training_dataset_ids: list[str] | None = None,
    evaluation_status: str = "Not evaluated.",
    training_sources: list[dict[str, object]] | None = None,
    reviewed_model_card: Path | None = None,
    finetuned_from_revision: str | None = None,
) -> CommitInfo:
    """Publish a model folder through one private Hub commit.

    Args:
        folder_path:
            Directory containing the saved model and tokenizer.
        repo_id:
            Destination model repository.
        finetuned_from:
            Base model identifier for the model card.
        private:
            Must be true for this private-only publication path.
        model_card_languages:
            Languages to include in the model-card metadata.
        commit_message (optional):
            Hub commit message. Defaults to "Publish private model".
        training_dataset_ids (optional):
            Exact Hub dataset identifiers used for training.
        evaluation_status (optional):
            Short evaluation-status statement for the model card.
        training_sources (optional):
            Structured source provenance for the model card.
        reviewed_model_card (optional):
            A reviewed README to use instead of the generated card.
        finetuned_from_revision:
            Immutable revision of the base model. Required for publication.

    Returns:
        The model-file upload commit information.

    Raises:
        ValueError:
            If the package or structured provenance is incomplete.
    """
    validate_private_only_config({"private_only": True, "private": private})
    if not training_sources:
        raise ValueError("Structured training-source provenance is required")
    _validate_base_model_metadata(
        finetuned_from=finetuned_from, finetuned_from_revision=finetuned_from_revision
    )
    _validate_training_sources(
        training_sources=training_sources,
        training_dataset_ids=training_dataset_ids or [],
    )
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    api = ensure_private_hub_repository(repo_id=repo_id, token=token)
    languages = list(model_card_languages)
    for required_language in ("da", "en"):
        if required_language not in languages:
            languages.append(required_language)
    with tempfile.TemporaryDirectory(prefix="hviske-model-") as staging_dir:
        staging_path = Path(staging_dir)
        _copy_model_artefacts(source=Path(folder_path), destination=staging_path)
        _stage_model_card(
            destination=staging_path / "README.md",
            finetuned_from=finetuned_from,
            model_card_languages=languages,
            training_dataset_ids=training_dataset_ids or [],
            training_sources=training_sources,
            evaluation_status=evaluation_status,
            reviewed_model_card=reviewed_model_card,
            finetuned_from_revision=finetuned_from_revision,
        )
        verify_private_hub_repository(api=api, repo_id=repo_id, token=token)
        commit = upload_folder(
            repo_id=repo_id,
            folder_path=staging_path,
            token=token or True,
            commit_message=commit_message,
        )
        verify_private_hub_repository(api=api, repo_id=repo_id, token=token)
    return commit


def _copy_model_artefacts(source: Path, destination: Path) -> None:
    """Copy a complete, reloadable Cohere package using the strict allowlist.

    Raises:
        ValueError:
            If the source is not a complete Cohere package.
    """
    if not source.is_dir():
        raise ValueError(f"Model output directory does not exist: {source}")
    _validate_model_package(source=source)
    for candidate in source.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        is_sharded = _SHARDED_MODEL_ARTEFACT.fullmatch(candidate.name) is not None
        if candidate.name not in _MODEL_ARTEFACT_NAMES and not is_sharded:
            continue
        shutil.copy2(candidate, destination / candidate.name)


def _validate_model_package(source: Path) -> None:
    """Check that a saved Cohere model has all reload-critical files.

    Raises:
        ValueError:
            If a required package file or weight set is missing or malformed.
    """
    required = {"config.json", "processor_config.json", "tokenizer_config.json"}
    missing = [name for name in required if not _regular_file(source / name)]
    if missing:
        raise ValueError("Cohere package is missing: " + ", ".join(sorted(missing)))
    documents: dict[str, dict[str, object]] = {}
    for name in required:
        try:
            document = json.loads((source / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cohere {name} is not valid JSON") from error
        if not isinstance(document, dict):
            raise ValueError(f"Cohere {name} must contain an object")
        documents[name] = document
    if documents["config.json"].get("model_type") != "cohere_asr":
        raise ValueError("Cohere config.json has the wrong model_type")
    processor = documents["processor_config.json"]
    if processor.get("processor_class") != "CohereAsrProcessor":
        raise ValueError("Cohere processor config has the wrong processor_class")
    feature_extractor = processor.get("feature_extractor")
    if not isinstance(feature_extractor, dict):
        raise ValueError("Cohere processor config has no feature extractor metadata")
    if feature_extractor.get("feature_extractor_type") != "CohereAsrFeatureExtractor":
        raise ValueError("Cohere processor config has the wrong feature extractor")
    if (
        not isinstance(feature_extractor.get("sampling_rate"), int)
        or feature_extractor["sampling_rate"] <= 0
    ):
        raise ValueError("Cohere processor config has an invalid sampling rate")
    tokenizer = documents["tokenizer_config.json"]
    if tokenizer.get("tokenizer_class") != "TokenizersBackend":
        raise ValueError("Cohere tokenizer config has the wrong tokenizer_class")
    if tokenizer.get("backend") != "tokenizers":
        raise ValueError("Cohere tokenizer config has the wrong backend")
    tokenizer_json = source / "tokenizer.json"
    if not _regular_file(tokenizer_json):
        raise ValueError("Cohere package is missing tokenizer.json")
    try:
        tokenizer_document = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cohere tokenizer.json is not valid JSON") from error
    if not isinstance(tokenizer_document, dict):
        raise ValueError("Cohere tokenizer.json must contain an object")

    single = source / "model.safetensors"
    index_path = source / "model.safetensors.index.json"
    shard_paths = sorted(
        path
        for path in source.iterdir()
        if _SHARDED_MODEL_ARTEFACT.fullmatch(path.name)
        and path.suffix == ".safetensors"
    )
    if _regular_file(single) and (shard_paths or _regular_file(index_path)):
        raise ValueError("Cohere package contains conflicting weight layouts")
    if _regular_file(single):
        _validate_safetensors_file(single)
        return
    if not _regular_file(index_path):
        raise ValueError("Cohere package needs model.safetensors or a complete index")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Malformed sharded weight index") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        )
    ):
        raise ValueError("Malformed sharded weight index")
    referenced = set(weight_map.values())
    if any(
        Path(name).name != name
        or not _SHARDED_MODEL_ARTEFACT.fullmatch(name)
        or not _regular_file(source / name)
        for name in referenced
    ):
        raise ValueError("Incomplete sharded weight index")
    if {path.name for path in shard_paths} != referenced:
        raise ValueError("Sharded weight files do not match the index")
    shard_tensors: dict[str, set[str]] = {}
    for shard_name in sorted(referenced):
        shard_tensors[shard_name] = _validate_safetensors_file(source / shard_name)
    for shard_name, actual_tensors in shard_tensors.items():
        expected_tensors = {
            tensor_name
            for tensor_name, assigned_shard in weight_map.items()
            if assigned_shard == shard_name
        }
        missing = expected_tensors - actual_tensors
        if missing:
            tensor_name = sorted(missing)[0]
            raise ValueError(
                f"Sharded index maps {tensor_name!r} to a shard without that tensor"
            )
        extra = actual_tensors - expected_tensors
        if extra:
            indexed_tensors = set(weight_map)
            unindexed = extra - indexed_tensors
            if unindexed:
                raise ValueError(
                    "Sharded weight files contain unindexed tensors: "
                    + ", ".join(sorted(unindexed))
                )
            raise ValueError(
                f"Sharded tensor set does not match index for {shard_name}: "
                + ", ".join(sorted(extra))
            )


def _regular_file(path: Path) -> bool:
    """Return whether a path is a regular, non-symlink file."""
    return path.is_file() and not path.is_symlink()


def _validate_safetensors_file(path: Path) -> set[str]:
    """Read safetensors metadata without materialising tensor data.

    Returns:
        The tensor names stored in the file.

    Raises:
        ValueError:
            If the file is not a readable safetensors file.
    """
    try:
        from safetensors import SafetensorError, safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensor_names = set(handle.keys())
            if not tensor_names:
                raise ValueError(f"Safetensors file is empty: {path.name}")
            for tensor_name in tensor_names:
                handle.get_slice(tensor_name)
            return tensor_names
    except (OSError, RuntimeError, SafetensorError, ValueError) as error:
        raise ValueError(f"Invalid safetensors weights: {path.name}") from error


def _stage_model_card(
    destination: Path,
    finetuned_from: str,
    model_card_languages: list[str],
    training_dataset_ids: list[str],
    training_sources: list[dict[str, object]],
    evaluation_status: str,
    reviewed_model_card: Path | None,
    finetuned_from_revision: str | None = None,
) -> None:
    """Stage a generated or reviewed card, never arbitrary source files.

    Raises:
        ValueError:
            If a reviewed card is not a safe, complete provenance record.
    """
    _validate_base_model_metadata(
        finetuned_from=finetuned_from, finetuned_from_revision=finetuned_from_revision
    )
    if reviewed_model_card is not None:
        if not _regular_file(reviewed_model_card):
            raise ValueError("Reviewed model card must be a regular file")
        card = reviewed_model_card.read_text(encoding="utf-8")
        forbidden = ("manifest_path", "source_wav_path", "HF_TOKEN", "HUGGINGFACE")
        card_lower = card.lower()
        required_markers = ("license: openrail", "private", "internal")
        frontmatter = _read_model_card_frontmatter(card)
        if (
            any(
                str(value) not in card
                for source in training_sources
                for value in source.values()
            )
            or any(marker not in card_lower for marker in required_markers)
            or frontmatter.get("base_model") != finetuned_from
            or frontmatter.get("base_model_revision") != finetuned_from_revision
            or any(marker in card for marker in forbidden)
            or re.search(r"(?:^|\s)/(?:Users|home|private|tmp|var)/", card)
        ):
            raise ValueError(
                "Reviewed model card does not contain safe complete provenance"
            )
        destination.write_text(card, encoding="utf-8")
        return
    source_lines = "\n".join(
        "- " + "; ".join(f"{key}: {source[key]}" for key in source)
        for source in training_sources
    )
    dataset_lines = "\n".join(f"- {dataset_id}" for dataset_id in training_dataset_ids)
    language_lines = "\n".join(f"- {language}" for language in model_card_languages)
    destination.write_text(
        "---\n"
        f"language:\n{language_lines}\nlicense: openrail\nlibrary_name: transformers\n"
        "pipeline_tag: automatic-speech-recognition\n"
        f"base_model: {finetuned_from}\n"
        + (
            f"base_model_revision: {finetuned_from_revision}\n"
            if finetuned_from_revision is not None
            else ""
        )
        + f"datasets:\n{dataset_lines}\n---\n\n"
        + "# Private internal Danish-English ASR checkpoint\n\n"
        "This private Cohere checkpoint is for internal research, evaluation and "
        "testing only. It is not for public distribution or production use.\n\n"
        "## Training-source provenance\n\n"
        f"{source_lines}\n\n## Evaluation status\n\n{evaluation_status}\n",
        encoding="utf-8",
    )


def _read_model_card_frontmatter(card: str) -> dict[str, str]:
    """Read scalar metadata from a model card's YAML frontmatter.

    Returns:
        The parsed base model metadata, or an empty dictionary for invalid
        frontmatter.
    """
    lines = card.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}
    try:
        end = next(
            index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"
        )
    except StopIteration:
        return {}
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"base_model", "base_model_revision"}:
            field = key.strip()
            if field in metadata:
                return {}
            metadata[field] = value.strip().strip("'\"")
    return metadata


def _validate_base_model_metadata(
    finetuned_from: str, finetuned_from_revision: str | None
) -> None:
    """Require the immutable base model identity used by publication.

    Raises:
        ValueError:
            If the base model ID or pinned revision is missing.
    """
    if (
        not finetuned_from.strip()
        or not finetuned_from_revision
        or not finetuned_from_revision.strip()
    ):
        raise ValueError(
            "Publication requires an exact base model and pinned base model revision"
        )


def _validate_training_sources(
    training_sources: list[dict[str, object]], training_dataset_ids: list[str]
) -> None:
    """Validate source provenance before it can enter a model card.

    Raises:
        ValueError:
            If a configured dataset is missing or metadata is incomplete.
    """
    ids = {str(source.get("id")) for source in training_sources}
    for source in training_sources:
        joined_transcript = source.get("joined_transcript")
        if isinstance(joined_transcript, dict):
            dataset_id = joined_transcript.get("dataset_id")
            if dataset_id is not None:
                ids.add(str(dataset_id))
    missing = sorted(set(training_dataset_ids) - ids)
    if missing:
        raise ValueError("Model-card provenance misses datasets: " + ", ".join(missing))
    required = {
        "id",
        "source",
        "subset",
        "split",
        "revision",
        "probability",
        "language",
    }
    for source in training_sources:
        if not required.issubset(source) or any(
            source.get(key) is None or not str(source.get(key)).strip()
            for key in required
        ):
            raise ValueError("Each training source needs complete structured metadata")
        if any(key in source for key in ("path", "manifest_path")):
            raise ValueError("Local paths are not permitted in model-card provenance")


def ensure_private_hub_repository(repo_id: str, token: str | None) -> HfApi:
    """Create a missing private model repository and verify its visibility.

    Args:
        repo_id:
            Model repository identifier.
        token:
            Hugging Face token used for Hub operations.

    Returns:
        The authenticated Hugging Face API client.

    """
    api = HfApi(token=token or True)
    try:
        verify_private_hub_repository(api=api, repo_id=repo_id, token=token)
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,
            token=token or True,
        )
        verify_private_hub_repository(api=api, repo_id=repo_id, token=token)
    return api


def verify_private_hub_repository(api: HfApi, repo_id: str, token: str | None) -> None:
    """Verify that a model repository exists and is private.

    Args:
        api:
            Authenticated Hugging Face API client.
        repo_id:
            Model repository identifier.
        token:
            Hugging Face token used for the request.

    Raises:
        PermissionError:
            If the repository is public or its visibility is unavailable.
    """
    info = api.repo_info(repo_id=repo_id, repo_type="model", token=token or True)
    if getattr(info, "private", None) is not True:
        raise PermissionError(
            f"Private-only publication refuses public repository {repo_id!r}"
        )


def validate_private_only_config(config: object) -> None:
    """Reject a contradictory private-publication configuration.

    Args:
        config:
            Hydra configuration containing ``private_only`` and ``private``.

    Raises:
        ValueError:
            If private-only publication is enabled without private publication.
    """
    getter = getattr(config, "get", None)
    if callable(getter):
        private_only = bool(getter("private_only", False))
        private = bool(getter("private", False))
    else:
        private_only = bool(getattr(config, "private_only", False))
        private = bool(getattr(config, "private", False))
    if private_only and not private:
        raise ValueError("A private-only run must set private=true")


def push_model_to_hub(
    trainer: Trainer,
    model_name: str,
    finetuned_from: str,
    create_pr: bool,
    language: str = "da",
    license: str = "openrail",
    tasks: list[str] | None = None,
    commit_message: str = "Finished finetuning 🎉",
    private: bool = False,
    private_only: bool = False,
    model_card_languages: list[str] | None = None,
    training_dataset_ids: list[str] | None = None,
    evaluation_status: str = "Not evaluated.",
    finetuned_from_revision: str | None = None,
) -> CommitInfo | None:
    """Upload a filtered model artefact set to the Hugging Face Hub.

    The upload is staged in a temporary directory so trainer output, datasets and
    experiment-tracking artefacts cannot become part of the Hub commit.

    Args:
        trainer:
            The Trainer object containing the model and tokenizer to upload.
        model_name:
            The name of the model.
        finetuned_from:
            The ID of the model that was finetuned.
        create_pr:
            Whether to create a pull request.
        language (optional):
            Retained for API compatibility. The model card is bilingual.
        license (optional):
            Must be ``openrail`` for this publication path.
        tasks (optional):
            Retained for API compatibility; this path publishes ASR models.
        commit_message (optional):
            Message to commit while pushing. Defaults to "Finished finetuning 🎉".
        private (optional):
            Whether the destination repository must be private. Defaults to False.
        private_only (optional):
            Whether to refuse all public repositories. Defaults to False.
        model_card_languages (optional):
            Additional model-card language metadata. Danish and English are always
            included.
        training_dataset_ids (optional):
            Exact Hub dataset identifiers used for training.
        evaluation_status (optional):
            Short evaluation-status statement for the model card.
        finetuned_from_revision:
            Immutable revision of the base model. Required for publication.

    Returns:
        The commit information, or None if the process is not the main process.

    Raises:
        ValueError:
            If private-only publication is requested without private=true, or if a
            licence other than openrail is requested.
    """
    del language, tasks, model_name
    if license.lower() != "openrail":
        raise ValueError("Private ASR publication requires the openrail licence")
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    validate_private_only_config({"private_only": private_only, "private": private})
    repo_id = trainer.hub_model_id or getattr(trainer.args, "hub_model_id", None)
    api: HfApi | None = None
    requires_private = private or private_only
    if requires_private:
        if repo_id is None:
            raise ValueError("Private publication requires a Hub model ID")
        api = ensure_private_hub_repository(repo_id=repo_id, token=token)

    # Trainer's own asynchronous pushes must never publish this output directory.
    trainer.args.push_to_hub = False
    if trainer.hub_model_id is None:
        trainer.init_hf_repo(token=token)
        repo_id = trainer.hub_model_id

    if not trainer.is_world_process_zero():
        return None
    trainer._finish_current_push()

    languages = list(model_card_languages or ["da", "en"])
    for required_language in ("da", "en"):
        if required_language not in languages:
            languages.append(required_language)
    with tempfile.TemporaryDirectory(prefix="hviske-model-") as staging_dir:
        staging_path = Path(staging_dir)
        _copy_model_artefacts(
            source=Path(trainer.args.output_dir or "."), destination=staging_path
        )
        _write_model_card(
            destination=staging_path / "README.md",
            finetuned_from=finetuned_from,
            model_card_languages=languages,
            training_dataset_ids=training_dataset_ids or [],
            evaluation_status=evaluation_status,
            finetuned_from_revision=finetuned_from_revision,
        )
        if requires_private:
            assert api is not None
            verify_private_hub_repository(api=api, repo_id=repo_id or "", token=token)
        commit = upload_folder(
            repo_id=repo_id or "",
            create_pr=create_pr,
            folder_path=staging_path,
            commit_message=commit_message,
            token=token or True,
        )
        if requires_private:
            assert api is not None
            verify_private_hub_repository(api=api, repo_id=repo_id or "", token=token)
    return commit


def _write_model_card(
    destination: Path,
    finetuned_from: str,
    model_card_languages: list[str],
    training_dataset_ids: list[str],
    evaluation_status: str,
    finetuned_from_revision: str | None,
) -> None:
    """Write a backwards-compatible minimal card for trainer publication."""
    sources: list[dict[str, object]] = [
        {
            "id": dataset_id,
            "source": dataset_id,
            "subset": "unspecified",
            "split": "unspecified",
            "revision": "unspecified",
            "probability": "unspecified",
            "language": "da/en",
        }
        for dataset_id in training_dataset_ids
    ]
    _stage_model_card(
        destination=destination,
        finetuned_from=finetuned_from,
        model_card_languages=model_card_languages,
        training_dataset_ids=training_dataset_ids,
        training_sources=sources,
        evaluation_status=evaluation_status,
        reviewed_model_card=None,
        finetuned_from_revision=finetuned_from_revision,
    )


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        hf_logging.set_verbosity_error()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        hf_logging.set_verbosity_info()


def validate_transcript_revision(revision: str) -> str:
    """Validate an immutable private transcript dataset revision.

    Args:
        revision:
            The Hub revision to use for the private transcript dataset.

    Returns:
        The unchanged, validated revision.

    Raises:
        ValueError:
            If ``revision`` is not a complete hexadecimal commit SHA.
    """
    if not _FULL_COMMIT_SHA.fullmatch(revision):
        raise ValueError(
            "P1 transcript revision must be a full 40-character commit SHA; "
            "mutable branches and abbreviated or non-hex revisions are forbidden."
        )
    return revision
