"""Publish a reviewed model to a verified private Hub repository."""

import pathlib
from collections.abc import Mapping, Sequence

import click
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from hviske.utils import (
    publish_model_folder,
    validate_immutable_source_revision,
    validate_overlay_revision,
    validate_transcript_revision,
)

_MODEL_TYPES_BY_BASE_ID = {
    "CohereLabs/cohere-transcribe-03-2026": "cohere_asr",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet_tdt",
}


@click.command()
@click.argument(
    "model_dir", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.argument("repo_id")
@click.option(
    "--config-name",
    default="bilingual",
    show_default=True,
    help="Hydra preset supplying the reviewed training provenance.",
)
@click.option(
    "--private",
    "private",
    is_flag=True,
    help="Confirm that this publication must remain private.",
)
@click.option(
    "--evaluation-status",
    default="Not evaluated.",
    show_default=True,
    help="Evaluation status to record in the model card.",
)
@click.option(
    "--reviewed-model-card",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Optional reviewed README containing the complete provenance.",
)
def main(
    model_dir: pathlib.Path,
    repo_id: str,
    config_name: str,
    private: bool,
    evaluation_status: str,
    reviewed_model_card: pathlib.Path | None,
) -> None:
    """Upload MODEL_DIR to REPO_ID using the resolved training-preset metadata."""
    config = _load_config(config_name=config_name)
    sources = training_sources_from_config(config=config)
    publish_model_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        finetuned_from=str(config.model.pretrained_model_id),
        private=private,
        model_card_languages=list(config.get("model_card_languages", ["da", "en"])),
        training_dataset_ids=list(config.training_dataset_ids),
        training_sources=sources,
        evaluation_status=evaluation_status,
        reviewed_model_card=reviewed_model_card,
        finetuned_from_revision=str(config.model.revision),
        expected_model_type=_expected_model_type(config=config),
    )


def _expected_model_type(config: DictConfig) -> str:
    """Resolve the only package family valid for the provenance preset.

    Returns:
        The Transformers model type required in the saved package.

    Raises:
        ValueError:
            If the preset's base model has no supported publication contract.
    """
    base_model_id = str(config.model.pretrained_model_id)
    try:
        return _MODEL_TYPES_BY_BASE_ID[base_model_id]
    except KeyError as error:
        raise ValueError(
            f"Unsupported publication base model: {base_model_id!r}"
        ) from error


def _load_config(config_name: str) -> DictConfig:
    """Resolve a publication preset from this repository's config directory.

    Returns:
        The resolved Hydra configuration.
    """
    config_dir = pathlib.Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=config_name)
    OmegaConf.resolve(config)
    return config


def training_sources_from_config(config: DictConfig) -> list[dict[str, object]]:
    """Derive safe, structured training provenance from a resolved preset.

    Returns:
        Source metadata without local paths.

    Raises:
        ValueError:
            If a configured training dataset is not represented by a source.
    """
    probabilities = list(config.dataset_probabilities)
    sources: list[dict[str, object]] = []
    for index, (source_name, source_config) in enumerate(config.datasets.items()):
        immutable_revision_env = source_config.get("immutable_revision_env")
        if immutable_revision_env is not None:
            validate_immutable_source_revision(
                str(source_config.get("revision") or ""),
                revision_label=str(immutable_revision_env),
            )
        is_local = source_config.get("type") == "local_vtt"
        source_id = f"local_vtt:{source_name}" if is_local else str(source_config.id)
        source: dict[str, object] = {
            "id": source_id,
            "source": str(source_name),
            "subset": str(source_config.get("subset") or "none"),
            "split": str(source_config.get("train_name", "train")),
            "revision": (
                "local-manifest" if is_local else str(source_config.get("revision"))
            ),
            "probability": probabilities[index],
            "language": str(source_config.get("language") or "unspecified"),
        }
        overlay_config = source_config.get("overlay")
        if overlay_config is not None:
            overlay_metadata = {
                "dataset_id": str(overlay_config.id),
                "subset": str(overlay_config.get("subset") or "none"),
                "split": str(overlay_config.get("split", "train")),
                "revision": validate_overlay_revision(
                    str(overlay_config.revision or "")
                ),
                "filters": _safe_overlay_value(overlay_config.get("filters", {})),
                "base_filters": _safe_overlay_value(
                    overlay_config.get("base_filters", {})
                ),
                "strategy": str(overlay_config.get("strategy", "keyed")),
                "base_join_column": str(overlay_config.get("base_join_column", "none")),
                "overlay_join_column": str(
                    overlay_config.get("overlay_join_column", "none")
                ),
                "equality_checks": _safe_overlay_value(
                    overlay_config.get("equality_checks", {})
                ),
                "action_column": str(overlay_config.get("action_column", "action")),
                "recognised_actions": _safe_overlay_value(
                    overlay_config.get("recognised_actions", [])
                ),
                "allowed_actions": _safe_overlay_value(
                    overlay_config.get("allowed_actions", [])
                ),
                "text_policy": _safe_overlay_value(
                    overlay_config.get("text_policy", {})
                ),
            }
            source["overlay"] = overlay_metadata
            source["relationship"] = (
                "base rows overlaid by metadata-only transcript revision"
            )
        transcript_id = source_config.get("transcript_dataset_id")
        if transcript_id is not None:
            transcript_metadata = {
                "dataset_id": str(transcript_id),
                "subset": str(source_config.get("transcript_subset") or "none"),
                "split": str(source_config.get("transcript_split", "train")),
                "revision": validate_transcript_revision(
                    str(source_config.transcript_revision)
                ),
                "join_column": str(source_config.transcript_join_column),
                "text_column": str(source_config.transcript_text_column),
            }
            source["audio"] = {
                "dataset_id": source_id,
                "revision": source["revision"],
                "join_column": str(source_config.audio_join_column),
            }
            source["transcript"] = transcript_metadata
            source["joined_transcript"] = {
                **transcript_metadata,
                "audio_join_column": str(source_config.audio_join_column),
                "transcript_join_column": str(source_config.transcript_join_column),
                "transcript_text_column": str(source_config.transcript_text_column),
            }
            source["relationship"] = "audio joined to transcript by configured columns"
        sources.append(source)

    configured_ids = {str(dataset_id) for dataset_id in config.training_dataset_ids}
    derived_ids = {str(source["id"]) for source in sources}
    for source in sources:
        joined_transcript = source.get("joined_transcript")
        if isinstance(joined_transcript, dict):
            derived_ids.add(str(joined_transcript["dataset_id"]))
        overlay = source.get("overlay")
        if isinstance(overlay, dict):
            derived_ids.add(str(overlay["dataset_id"]))
    missing = configured_ids - derived_ids
    if missing:
        raise ValueError(
            "Training preset training_dataset_ids are not represented by its sources: "
            + ", ".join(sorted(missing))
        )
    return sources


def _safe_overlay_value(value: object) -> object:
    """Convert overlay config values to provenance-safe scalar containers.

    Returns:
        A recursively serialisable value without local path resolution.
    """
    if isinstance(value, Mapping):
        return {str(key): _safe_overlay_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_safe_overlay_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
