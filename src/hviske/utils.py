"""General utility functions."""

import collections.abc as c
import contextlib
import logging
import multiprocessing as mp
import os
import re
import typing as t
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
) -> CommitInfo:
    """Publish a model folder after enforcing private-only Hub visibility.

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

    Returns:
        The model-file upload commit information.

    """
    validate_private_only_config({"private_only": True, "private": private})
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    api = ensure_private_hub_repository(repo_id=repo_id, token=token)
    commit = upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        token=token or True,
        commit_message=commit_message,
        allow_patterns=[
            "*.bin",
            "*.json",
            "*.jinja",
            "*.model",
            "*.safetensors",
            "*.txt",
        ],
        ignore_patterns=[
            "_*",
            "checkpoint-*",
            "*.arrow",
            "*.csv",
            "*.flac",
            "*.jsonl",
            "*.log",
            "*.yaml",
            "*.yml",
            "trainer_state.json",
            "all_results.json",
            "eval_results.json",
            "train_results.json",
            "*.mp3",
            "*.parquet",
            "*.wav",
            "*.vtt",
            ".hydra/**",
            "mlruns/**",
            "runs/**",
            "wandb/**",
        ],
    )
    language_lines = "\n".join(f"- {language}" for language in model_card_languages)
    card = (
        "---\n"
        f"language:\n{language_lines}\n"
        "library_name: transformers\n"
        "pipeline_tag: automatic-speech-recognition\n"
        f"base_model: {finetuned_from}\n"
        "---\n"
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token or True,
        commit_message="Add bilingual model-card metadata",
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
) -> CommitInfo | None:
    """Upload model and tokenizer to the Hugging Face Hub.

    This uses the model stored as `trainer.model` and the tokenizer stored as
    `trainer.tokenizer`, and uploads them to the model ID stored in
    `trainer.args.hub_model_id`.

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
            The language of the model. Defaults to "da" (Danish).
        license (optional):
            The license of the model. Defaults to "openrail".
        tasks (optional):
            The tasks the model is fine-tuned for. Defaults to
            ["automatic-speech-recognition"].
        commit_message (optional):
            Message to commit while pushing. Defaults to "Finished finetuning 🎉".
        private (optional):
            Whether the destination repository must be private. Defaults to False.
        private_only (optional):
            Whether to refuse all public repositories. Defaults to False.
        model_card_languages (optional):
            Language metadata to write to the model card. Defaults to ``language``.

    Returns:
        The commit information, or None if the process is not the main process.

    Raises:
        ValueError:
            If private-only publication is requested without private=true.
    """
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    validate_private_only_config({"private_only": private_only, "private": private})
    repo_id = trainer.hub_model_id or getattr(trainer.args, "hub_model_id", None)
    if private_only:
        if repo_id is None:
            raise ValueError("Private-only publication requires a Hub model ID")
        ensure_private_hub_repository(repo_id=repo_id, token=token)

    # In case the user calls this method with trainer.args.push_to_hub = False
    if trainer.hub_model_id is None:
        trainer.init_hf_repo(token=token)
        repo_id = trainer.hub_model_id

    # Only push from one node
    if not trainer.is_world_process_zero():
        return None

    trainer.create_model_card(
        model_name=model_name,
        language=t.cast(str, model_card_languages or language),
        license=license,
        tasks=tasks or ["automatic-speech-recognition"],
        finetuned_from=finetuned_from,
    )

    # Wait for the current upload to be finished.
    trainer._finish_current_push()
    commit = upload_folder(
        repo_id=repo_id or "",
        create_pr=create_pr,
        folder_path=trainer.args.output_dir or ".",
        commit_message=commit_message,
        token=token or True,
        allow_patterns=[
            "README.md",
            "*.bin",
            "*.json",
            "*.jinja",
            "*.model",
            "*.safetensors",
            "*.txt",
        ],
        ignore_patterns=[
            "_*",
            "checkpoint-*",
            "*.arrow",
            "*.audio",
            "*.csv",
            "*.flac",
            "*.jsonl",
            "*.log",
            "*.yaml",
            "*.yml",
            "trainer_state.json",
            "all_results.json",
            "eval_results.json",
            "train_results.json",
            "*.mp3",
            "*.parquet",
            "*.wav",
            "*.vtt",
            ".hydra/**",
            "mlruns/**",
            "runs/**",
            "wandb/**",
        ],
    )
    if private_only:
        verify_private_hub_repository(
            api=HfApi(token=token or True), repo_id=repo_id or "", token=token
        )
    return commit


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
