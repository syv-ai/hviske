"""Boundedly preflight the pinned finetuning data without loading the model."""

import argparse
import collections.abc as c
import json
import logging
import math
import os
import typing as t
from pathlib import Path

import soundfile
from datasets import Audio, Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from hviske.data import (
    _filter_dataset_rows,
    _load_transcript_dataset,
    apply_dataset_overlay,
    join_audio_and_transcripts,
)
from hviske.experiment_tracking.wandb_setup import preflight_wandb_access
from hviske.utils import (
    validate_immutable_source_revision,
    validate_overlay_revision,
    validate_transcript_revision,
)

logger = logging.getLogger("hviske_data_preflight")

DatasetLoader: t.TypeAlias = c.Callable[..., object]


def main() -> None:
    """Resolve a finetuning preset and preflight every configured data source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="sparkie_bilingual")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=args.config_name)
    OmegaConf.resolve(config)
    preflight_finetuning_data(config=config)
    logger.info("Preflight passed for every configured training and validation source")


def _preflight_hub_source(
    source_name: str,
    source_config: DictConfig,
    split_key: str,
    dataset_loader: DatasetLoader,
    cache_dir: str | None,
    token: str | None,
) -> None:
    dataset = dataset_loader(
        path=source_config.id,
        name=source_config.get("subset"),
        split=source_config[split_key],
        revision=source_config.get("revision"),
        token=token or True,
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=source_config.get("trust_remote_code", False),
    )
    audio_column = str(source_config.audio_column)
    if (
        isinstance(dataset, Dataset | IterableDataset)
        and audio_column in (dataset.column_names or [])
        and dataset.features is not None
        and isinstance(dataset.features[audio_column], Audio)
    ):
        dataset = dataset.cast_column(column=audio_column, feature=Audio(decode=False))

    source_filters = source_config.get("filters")
    if source_filters is not None:
        if not isinstance(source_filters, c.Mapping):
            raise ValueError(f"Filters for {source_name} must be a mapping")
        if not isinstance(dataset, Dataset | IterableDataset):
            raise ValueError(f"Unsupported audio dataset type: {type(dataset)}")
        dataset = _filter_dataset_rows(
            dataset=dataset, filters=t.cast(dict[str, object], source_filters)
        )
    overlay_config = source_config.get("overlay")
    has_overlay = overlay_config is not None
    if has_overlay:
        overlay_revision = validate_overlay_revision(
            str(overlay_config.get("revision") or "")
        )
        overlay = _load_transcript_dataset(
            dataset_id=str(overlay_config.id),
            subset=overlay_config.get("subset"),
            split=str(overlay_config.get("split", "train")),
            revision=overlay_revision,
            cache_dir=cache_dir,
            trust_remote_code=overlay_config.get("trust_remote_code", False),
            dataset_loader=dataset_loader,
        )
        if not isinstance(dataset, Dataset | IterableDataset):
            raise ValueError(f"Unsupported audio dataset type: {type(dataset)}")
        dataset = apply_dataset_overlay(
            base_dataset=dataset,
            overlay_dataset=overlay,
            overlay_config=t.cast(c.Mapping[str, object], overlay_config),
        )
    has_transcript_join = source_config.get("transcript_dataset_id") is not None
    if has_transcript_join:
        transcript_revision = validate_transcript_revision(
            str(source_config.transcript_revision)
        )
        transcript = _load_transcript_dataset(
            dataset_id=str(source_config.transcript_dataset_id),
            subset=source_config.get("transcript_subset"),
            split=str(source_config.get("transcript_split", "train")),
            revision=transcript_revision,
            cache_dir=cache_dir,
            trust_remote_code=source_config.get("transcript_trust_remote_code", False),
            dataset_loader=dataset_loader,
        )
        if not isinstance(dataset, Dataset | IterableDataset):
            raise ValueError(f"Unsupported audio dataset type: {type(dataset)}")
        dataset = join_audio_and_transcripts(
            audio_dataset=dataset,
            transcript_dataset=transcript,
            audio_join_column=str(source_config.audio_join_column),
            transcript_join_column=str(source_config.transcript_join_column),
            transcript_text_column=str(source_config.transcript_text_column),
        )
        row = _first_row(dataset=dataset, source_name=f"joined {source_name}")
        _require_columns(
            row=row,
            required_columns=[str(source_config.audio_column), "text"],
            source_name=f"joined {source_name}",
        )
    elif has_overlay:
        row = _consume_overlay(dataset=dataset, source_name=source_name)
        _require_columns(
            row=row,
            required_columns=[str(source_config.audio_column), "text"],
            source_name=f"overlaid {source_name}",
        )
    else:
        row = _first_row(dataset=dataset, source_name=source_name)
        _require_columns(
            row=row,
            required_columns=[
                str(source_config.audio_column),
                str(source_config.text_column),
            ],
            source_name=source_name,
        )
    text_column = (
        "text" if has_transcript_join or has_overlay else str(source_config.text_column)
    )
    text = row[text_column]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Configured text is empty in the first {source_name} row")
    logger.info("Validated joined/overlaid/streamed rows from %s", source_name)


def _consume_overlay(dataset: object, source_name: str) -> dict[str, object]:
    """Consume an overlaid stream to trigger strict length and missing-row checks.

    Returns:
        The first usable row.

    Raises:
        ValueError:
            If the overlay is empty or emits a non-mapping row.
    """
    iterator = iter(t.cast(c.Iterable[object], dataset))
    first: dict[str, object] | None = None
    count = 0
    for raw_row in iterator:
        if not isinstance(raw_row, dict):
            raise ValueError(f"A row from {source_name} is not a mapping")
        if first is None:
            first = t.cast(dict[str, object], raw_row)
        count += 1
    if first is None:
        raise ValueError(f"Overlay produced no usable rows from {source_name}")
    logger.info("Validated %s overlaid rows from %s", f"{count:,}", source_name)
    return first


def _first_row(dataset: object, source_name: str) -> dict[str, object]:
    iterator = iter(t.cast(c.Iterable[object], dataset))
    try:
        raw_row = next(iterator)
    except StopIteration as error:
        raise ValueError(f"No rows available from {source_name}") from error
    if not isinstance(raw_row, dict):
        raise ValueError(f"The first row from {source_name} is not a mapping")
    return t.cast(dict[str, object], raw_row)


def _require_columns(
    row: dict[str, object], required_columns: list[str], source_name: str
) -> None:
    missing = sorted(set(required_columns) - set(row))
    if missing:
        raise ValueError(f"Missing columns from {source_name}: {missing}")


def _preflight_local_manifest(source_name: str, manifest_path: Path) -> None:
    manifest_path = manifest_path.expanduser()
    if not manifest_path.is_file():
        raise ValueError(f"Missing {source_name} manifest: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as manifest_file:
        first_line = next((line for line in manifest_file if line.strip()), None)
    if first_line is None:
        raise ValueError(f"Empty {source_name} manifest: {manifest_path}")
    raw_row = json.loads(first_line)
    if not isinstance(raw_row, dict):
        raise ValueError(f"The first {source_name} manifest row is not an object")
    row = t.cast(dict[str, object], raw_row)
    required_columns = [
        "source_wav_path",
        "start",
        "end",
        "text",
        "id",
        "duration",
        "language",
    ]
    _require_columns(
        row=row, required_columns=required_columns, source_name=source_name
    )

    audio_path = Path(str(row["source_wav_path"])).expanduser()
    if not audio_path.is_file():
        raise ValueError(f"Missing audio referenced by {source_name}: {audio_path}")
    try:
        start = float(t.cast(float | int | str, row["start"]))
        end = float(t.cast(float | int | str, row["end"]))
        duration = float(t.cast(float | int | str, row["duration"]))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid cue timing in the first {source_name} manifest row"
        ) from error
    if (
        not all(math.isfinite(value) for value in (start, end, duration))
        or start < 0
        or end <= start
        or duration <= 0
    ):
        raise ValueError(f"Invalid cue timing in the first {source_name} manifest row")
    try:
        audio_info = soundfile.info(audio_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"Unreadable audio referenced by {source_name}: {audio_path}"
        ) from error
    sample_rate = float(audio_info.samplerate)
    channels = int(audio_info.channels)
    frames = int(audio_info.frames)
    file_duration = float(audio_info.duration)
    if (
        not math.isfinite(sample_rate)
        or sample_rate <= 0
        or channels <= 0
        or frames <= 0
        or not math.isfinite(file_duration)
        or file_duration <= 0
    ):
        raise ValueError(f"Invalid audio metadata in the first {source_name} row")
    if end > file_duration or end * sample_rate > frames + 1:
        raise ValueError(
            f"Cue extends beyond audio referenced by {source_name}: {audio_path}"
        )
    if not str(row["text"]).strip() or not str(row["language"]).strip():
        raise ValueError(
            f"Empty text or language in the first {source_name} manifest row"
        )
    logger.info("Validated one metadata row from %s", source_name)


class HubApi(t.Protocol):
    """Hub operations needed by the data preflight."""

    def model_info(self, repo_id: str, *, revision: str) -> object:
        """Return metadata for an accessible model repository revision."""
        ...

    def whoami(self) -> dict[str, object]:
        """Return the authenticated Hub identity."""
        ...


def preflight_finetuning_data(
    config: DictConfig,
    dataset_loader: DatasetLoader = load_dataset,
    hub_api: HubApi | None = None,
) -> None:
    """Validate Hub access, source samples, and complete configured overlays.

    Args:
        config:
            Resolved finetuning configuration.
        dataset_loader (optional):
            Hugging Face dataset loader. Defaults to ``datasets.load_dataset``.
        hub_api (optional):
            Hub API client. Defaults to an authenticated ``HfApi`` client.
    """
    if config.get("enable_experiment_tracking", False):
        if config.experiment_tracking.type == "wandb":
            preflight_wandb_access(config=config)

    for source_config in config.datasets.values():
        immutable_revision_env = source_config.get("immutable_revision_env")
        if immutable_revision_env is not None:
            validate_immutable_source_revision(
                str(source_config.get("revision") or ""),
                revision_label=str(immutable_revision_env),
            )
        transcript_dataset_id = source_config.get("transcript_dataset_id")
        if transcript_dataset_id is not None:
            validate_transcript_revision(str(source_config.transcript_revision))
        overlay_config = source_config.get("overlay")
        if overlay_config is not None:
            validate_overlay_revision(str(overlay_config.get("revision") or ""))
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api: HubApi = hub_api or HfApi(token=token)
    identity = api.whoami()
    logger.info("Authenticated to the Hugging Face Hub as %s", identity.get("name"))
    api.model_info(
        repo_id=str(config.model.pretrained_model_id),
        revision=str(config.model.revision),
    )
    logger.info("Confirmed access to model %s", config.model.pretrained_model_id)

    for source_name, source_config in config.datasets.items():
        if source_config.get("type") == "local_vtt":
            _preflight_local_manifest(
                source_name=str(source_name),
                manifest_path=Path(source_config.manifest_path),
            )
        else:
            _preflight_hub_source(
                source_name=str(source_name),
                source_config=source_config,
                split_key="train_name",
                dataset_loader=dataset_loader,
                cache_dir=config.get("cache_dir"),
                token=token,
            )

    for index, source_config in enumerate(config.evaluation_datasets):
        _preflight_hub_source(
            source_name=f"evaluation[{index}]",
            source_config=source_config,
            split_key="val_name",
            dataset_loader=dataset_loader,
            cache_dir=config.get("cache_dir"),
            token=token,
        )


if __name__ == "__main__":
    main()
