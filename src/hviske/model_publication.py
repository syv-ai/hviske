"""Hugging Face model publication helpers."""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import CommitInfo, HfApi, upload_folder
from huggingface_hub.errors import RepositoryNotFoundError
from transformers.trainer import Trainer

logger = logging.getLogger(__package__)

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


def _copy_model_artefacts(
    source: Path, destination: Path, expected_model_type: str | None = None
) -> None:
    """Copy a complete, reloadable model package using the strict allowlist.

    Raises:
        ValueError:
            If the source is incomplete or its family contradicts the provenance.
    """
    if not source.is_dir():
        raise ValueError(f"Model output directory does not exist: {source}")
    model_type = _validate_model_package(source=source)
    if expected_model_type is not None and model_type != expected_model_type:
        raise ValueError(
            "Model package type does not match publication provenance: "
            f"expected {expected_model_type!r}, found {model_type!r}"
        )
    for candidate in source.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        is_sharded = _SHARDED_MODEL_ARTEFACT.fullmatch(candidate.name) is not None
        if candidate.name not in _MODEL_ARTEFACT_NAMES and not is_sharded:
            continue
        shutil.copy2(candidate, destination / candidate.name)


def _validate_model_package(source: Path) -> str:
    """Check that a saved supported model has all reload-critical files.

    Returns:
        The validated package's model type.

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
            raise ValueError(f"Model {name} is not valid JSON") from error
        if not isinstance(document, dict):
            raise ValueError(f"Model {name} must contain an object")
        documents[name] = document
    model_type = documents["config.json"].get("model_type")
    processor = documents["processor_config.json"]
    tokenizer = documents["tokenizer_config.json"]
    if model_type == "cohere_asr":
        family_label = "Cohere"
        if processor.get("processor_class") != "CohereAsrProcessor":
            raise ValueError("Cohere processor config has the wrong processor_class")
        feature_extractor = processor.get("feature_extractor")
        if not isinstance(feature_extractor, dict):
            raise ValueError(
                "Cohere processor config has no feature extractor metadata"
            )
        if (
            feature_extractor.get("feature_extractor_type")
            != "CohereAsrFeatureExtractor"
        ):
            raise ValueError("Cohere processor config has the wrong feature extractor")
        if (
            not isinstance(feature_extractor.get("sampling_rate"), int)
            or feature_extractor["sampling_rate"] <= 0
        ):
            raise ValueError("Cohere processor config has an invalid sampling rate")
        if tokenizer.get("tokenizer_class") != "TokenizersBackend":
            raise ValueError("Cohere tokenizer config has the wrong tokenizer_class")
        if tokenizer.get("backend") != "tokenizers":
            raise ValueError("Cohere tokenizer config has the wrong backend")
    elif model_type == "parakeet_tdt":
        family_label = "Parakeet TDT"
        architectures = documents["config.json"].get("architectures")
        if not isinstance(architectures, list) or "ParakeetForTDT" not in architectures:
            raise ValueError("Parakeet TDT config has the wrong architecture")
        durations = documents["config.json"].get("durations")
        if (
            not isinstance(durations, list)
            or not durations
            or not all(
                isinstance(duration, int) and duration >= 0 for duration in durations
            )
        ):
            raise ValueError("Parakeet TDT config has invalid durations")
        if processor.get("processor_class") != "ParakeetProcessor":
            raise ValueError(
                "Parakeet TDT processor config has the wrong processor_class"
            )
        if processor.get("decoder_type") != "tdt":
            raise ValueError("Parakeet TDT processor config has the wrong decoder_type")
        if (
            not isinstance(processor.get("blank_token"), str)
            or not processor["blank_token"]
        ):
            raise ValueError("Parakeet TDT processor config has no blank token")
        feature_extractor = processor.get("feature_extractor")
        if not isinstance(feature_extractor, dict):
            raise ValueError(
                "Parakeet TDT processor config has no feature extractor metadata"
            )
        if (
            feature_extractor.get("feature_extractor_type")
            != "ParakeetFeatureExtractor"
        ):
            raise ValueError(
                "Parakeet TDT processor config has the wrong feature extractor"
            )
        if feature_extractor.get("sampling_rate") != 16_000:
            raise ValueError(
                "Parakeet TDT processor config has the wrong sampling rate"
            )
        if tokenizer.get("tokenizer_class") != "ParakeetTokenizer":
            raise ValueError(
                "Parakeet TDT tokenizer config has the wrong tokenizer_class"
            )
        if tokenizer.get("backend") != "tokenizers":
            raise ValueError("Parakeet TDT tokenizer config has the wrong backend")
    elif processor.get("processor_class") == "CohereAsrProcessor":
        raise ValueError("Cohere config.json has the wrong model_type")
    else:
        raise ValueError(f"Unsupported model family: {model_type!r}")
    tokenizer_json = source / "tokenizer.json"
    if not _regular_file(tokenizer_json):
        raise ValueError(f"{family_label} package is missing tokenizer.json")
    try:
        tokenizer_document = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{family_label} tokenizer.json is not valid JSON") from error
    if not isinstance(tokenizer_document, dict):
        raise ValueError(f"{family_label} tokenizer.json must contain an object")

    single = source / "model.safetensors"
    index_path = source / "model.safetensors.index.json"
    shard_paths = sorted(
        path
        for path in source.iterdir()
        if _SHARDED_MODEL_ARTEFACT.fullmatch(path.name)
        and path.suffix == ".safetensors"
    )
    if _regular_file(single) and (shard_paths or _regular_file(index_path)):
        raise ValueError(f"{family_label} package contains conflicting weight layouts")
    if _regular_file(single):
        _validate_safetensors_file(single)
        return str(model_type)
    if not _regular_file(index_path):
        raise ValueError(
            f"{family_label} package needs model.safetensors or a complete index"
        )
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
    return str(model_type)


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


def _write_model_card(
    destination: Path,
    finetuned_from: str,
    model_card_languages: list[str],
    training_dataset_ids: list[str],
    evaluation_status: str,
    finetuned_from_revision: str | None,
) -> None:
    """Write the generated card used by integrated trainer publication."""
    _stage_model_card(
        destination=destination,
        finetuned_from=finetuned_from,
        model_card_languages=model_card_languages,
        training_dataset_ids=training_dataset_ids,
        evaluation_status=evaluation_status,
        finetuned_from_revision=finetuned_from_revision,
    )


def _stage_model_card(
    destination: Path,
    finetuned_from: str,
    model_card_languages: list[str],
    training_dataset_ids: list[str],
    evaluation_status: str,
    finetuned_from_revision: str | None = None,
) -> None:
    """Stage the generated model card for an integrated training publication."""
    _validate_base_model_metadata(
        finetuned_from=finetuned_from, finetuned_from_revision=finetuned_from_revision
    )
    source_lines = "\n".join(f"- {dataset_id}" for dataset_id in training_dataset_ids)
    language_lines = "\n".join(f"- {language}" for language in model_card_languages)
    destination.write_text(
        "---\n"
        f"language:\n{language_lines}\nlicense: openrail\n"
        "library_name: transformers\npipeline_tag: automatic-speech-recognition\n"
        f"base_model: {finetuned_from}\n"
        + (
            f"base_model_revision: {finetuned_from_revision}\n"
            if finetuned_from_revision is not None
            else ""
        )
        + f"datasets:\n{source_lines}\n---\n\n"
        + "# Private internal Danish-English ASR checkpoint\n\n"
        + "This private ASR checkpoint is for internal research, evaluation and "
        + "testing only. It is not for public distribution or production use.\n\n"
        + "## Training datasets\n\n"
        + f"{source_lines}\n\n## Evaluation status\n\n{evaluation_status}\n",
        encoding="utf-8",
    )


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
