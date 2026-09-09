"""General utility functions."""

import collections.abc as c
import contextlib
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


class transformers_output_ignored:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        hf_logging.set_verbosity_error()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        hf_logging.set_verbosity_info()


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


@contextlib.contextmanager
def disable_tqdm() -> c.Generator[None, None, None]:
    """Context manager to disable tqdm."""

    def _patch(old_init: c.Callable[..., None]) -> partialmethod:
        return partialmethod(old_init, disable=True)

    with monkeypatched(tqdm_package.std.tqdm, "__init__", _patch):
        yield


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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Re-enable the progress bar."""
        enable_progress_bar()


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


def publish_model_folder(
    folder_path: str | Path,
    repo_id: str,
    finetuned_from: str,
    private: bool,
    model_card_languages: list[str],
    commit_message: str = "Publish private model",
    training_dataset_ids: list[str] | None = None,
    evaluation_status: str = "Not evaluated.",
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

    Returns:
        The model-file upload commit information.
    """
    validate_private_only_config({"private_only": True, "private": private})
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    api = ensure_private_hub_repository(repo_id=repo_id, token=token)
    languages = list(model_card_languages)
    for required_language in ("da", "en"):
        if required_language not in languages:
            languages.append(required_language)
    with tempfile.TemporaryDirectory(prefix="hviske-model-") as staging_dir:
        staging_path = Path(staging_dir)
        _copy_model_artefacts(source=Path(folder_path), destination=staging_path)
        _write_model_card(
            destination=staging_path / "README.md",
            finetuned_from=finetuned_from,
            model_card_languages=languages,
            training_dataset_ids=training_dataset_ids or [],
            evaluation_status=evaluation_status,
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


def _copy_model_artefacts(source: Path, destination: Path) -> None:
    """Copy only recognised regular files from a model output directory.

    Raises:
        ValueError:
            If the source is not a directory.
    """
    if not source.is_dir():
        raise ValueError(f"Model output directory does not exist: {source}")
    for candidate in source.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        is_sharded = _SHARDED_MODEL_ARTEFACT.fullmatch(candidate.name) is not None
        if candidate.name not in _MODEL_ARTEFACT_NAMES and not is_sharded:
            continue
        shutil.copy2(candidate, destination / candidate.name)


def _write_model_card(
    destination: Path,
    finetuned_from: str,
    model_card_languages: list[str],
    training_dataset_ids: list[str],
    evaluation_status: str,
) -> None:
    """Write the publication model card without machine-local information."""
    language_lines = "\n".join(f"- {language}" for language in model_card_languages)
    dataset_lines = "\n".join(f"- {dataset_id}" for dataset_id in training_dataset_ids)
    if not dataset_lines:
        dataset_lines = "- Not supplied by the caller."
    destination.write_text(
        "---\n"
        f"language:\n{language_lines}\n"
        "license: openrail\n"
        "library_name: transformers\n"
        "pipeline_tag: automatic-speech-recognition\n"
        f"base_model: {finetuned_from}\n"
        "datasets:\n"
        f"{dataset_lines}\n"
        "---\n\n"
        "# Private internal Danish-English ASR checkpoint\n\n"
        "This is a private internal Danish-English automatic speech recognition "
        "checkpoint. It is intended for internal research, evaluation and testing "
        "only, not for public distribution or production use.\n\n"
        "## Training datasets\n\n"
        f"{dataset_lines}\n\n"
        "## Evaluation status\n\n"
        f"{evaluation_status}\n",
        encoding="utf-8",
    )


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
