"""Boundedly preflight the pinned finetuning data without loading the model."""

import argparse
import collections.abc as c
import copy
import dataclasses
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


def _preflight_grouped_hub_sources(
    datasets: c.Mapping[str, DictConfig],
    dataset_loader: DatasetLoader,
    cache_dir: str | None,
    token: str | None,
) -> set[str]:
    """Preflight equivalent positional views with one shared stream.

    Returns:
        Names handled by the shared preflight path.

    Raises:
        ValueError:
            If a shared stream fails the ordinary overlay validation.
    """
    handled: set[str] = set()
    for group in _grouped_source_candidates(datasets=datasets):
        if len(group) < 2:
            continue
        representative = group[0]
        source_config = representative.config
        dataset = dataset_loader(
            path=source_config.id,
            name=source_config.get("subset"),
            split=source_config.train_name,
            revision=source_config.get("revision"),
            token=token or True,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=source_config.get("trust_remote_code", False),
        )
        overlay_config = t.cast(c.Mapping[str, object], source_config.overlay)
        dataset, overlay_base_columns = _project_overlay_base_dataset(
            dataset=dataset,
            audio_column=str(source_config.audio_column),
            source_config=source_config,
            overlay_config=overlay_config,
            source_name=f"group containing {representative.name}",
        )
        overlay_revision = validate_overlay_revision(
            str(overlay_config.get("revision") or "")
        )
        overlay = _load_transcript_dataset(
            dataset_id=str(overlay_config["id"]),
            subset=t.cast(str | None, overlay_config.get("subset")),
            split=str(overlay_config.get("split", "train")),
            revision=overlay_revision,
            cache_dir=cache_dir,
            trust_remote_code=bool(overlay_config.get("trust_remote_code", False)),
            dataset_loader=dataset_loader,
        )
        unfiltered_overlay_config = copy.deepcopy(dict(overlay_config))
        unfiltered_overlay_config.pop("base_filters", None)
        unfiltered_overlay_config.pop("filters", None)
        if not isinstance(dataset, Dataset | IterableDataset):
            raise ValueError(f"Unsupported audio dataset type: {type(dataset)}")
        overlaid = apply_dataset_overlay(
            base_dataset=dataset,
            overlay_dataset=overlay,
            overlay_config=unfiltered_overlay_config,
        )
        _consume_grouped_overlay(
            dataset=overlaid, group=group, required_base_columns=overlay_base_columns
        )
        handled.update(source.name for source in group)
    return handled


def _require_columns(
    row: dict[str, object], required_columns: list[str], source_name: str
) -> None:
    missing = sorted(set(required_columns) - set(row))
    if missing:
        raise ValueError(f"Missing columns from {source_name}: {missing}")


def _same_filter_value(left: object, right: object) -> bool:
    """Compare filter values without collapsing distinct scalar types.

    Returns:
        Whether both values have the same type and exact value.
    """
    return type(left) is type(right) and left == right


@dataclasses.dataclass(frozen=True)
class _GroupedSource:
    """A training source that can participate in a positional overlay group."""

    name: str
    config: DictConfig
    filter_column: str
    filter_value: object
    base_signature: object
    overlay_signature: object


def _consume_grouped_overlay(
    dataset: object, group: list[_GroupedSource], required_base_columns: set[str]
) -> None:
    """Consume a shared overlay and validate every configured source view.

    Raises:
        ValueError:
            If a row is malformed, missing metadata, or no source has an accepted
            row.
    """
    counts = {source.name: 0 for source in group}
    first_rows: dict[str, dict[str, object]] = {}
    filter_column = group[0].filter_column
    iterator = iter(t.cast(c.Iterable[object], dataset))
    for raw_row in iterator:
        if not isinstance(raw_row, dict):
            raise ValueError(
                "A row from the grouped positional overlay is not a mapping"
            )
        row = t.cast(dict[str, object], raw_row)
        _require_columns(
            row=row,
            required_columns=["text", filter_column, *sorted(required_base_columns)],
            source_name="grouped positional overlay",
        )
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Grouped positional overlay emitted empty text")
        for source in group:
            if _same_filter_value(row[filter_column], source.filter_value):
                counts[source.name] += 1
                first_rows.setdefault(source.name, row)
                break

    for source in group:
        count = counts[source.name]
        if count == 0:
            raise ValueError(
                f"Configured source {source.name} emitted no accepted rows for "
                f"{source.filter_column}={source.filter_value!r}"
            )
        # Retaining the first row makes the per-source validation explicit while
        # keeping the full stream consumption above as the single strict pass.
        first_row = first_rows.get(source.name)
        if first_row is None:
            raise ValueError(f"No retained row available from {source.name}")
        _require_columns(
            row=first_row,
            required_columns=["text", source.filter_column],
            source_name=source.name,
        )
        logger.info("Validated %s accepted rows from %s", f"{count:,}", source.name)


def _grouped_source_candidates(
    datasets: c.Mapping[str, DictConfig],
) -> list[list[_GroupedSource]]:
    """Return eligible source groups, retaining singleton groups for the caller."""
    candidates: list[_GroupedSource] = []
    for raw_name, source_config in datasets.items():
        candidate = _grouped_source_candidate(
            name=str(raw_name), source_config=source_config
        )
        if candidate is not None:
            candidates.append(candidate)

    groups: list[list[_GroupedSource]] = []
    for candidate in candidates:
        group = next(
            (
                group
                for group in groups
                if group[0].base_signature == candidate.base_signature
                and group[0].overlay_signature == candidate.overlay_signature
                and group[0].filter_column == candidate.filter_column
            ),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)

    # Reusing a value would make the shared result ambiguous, so it is safer to
    # leave every source in such a group on the established independent path.
    return [
        group
        for group in groups
        if len(group) >= 2
        and all(
            not _same_filter_value(left.filter_value, right.filter_value)
            for index, left in enumerate(group)
            for right in group[index + 1 :]
        )
    ] + [group for group in groups if len(group) == 1]


def _grouped_source_candidate(
    name: str, source_config: DictConfig
) -> _GroupedSource | None:
    """Describe a source only when its overlay is safe to coalesce.

    Returns:
        An eligible source descriptor, or ``None`` when it must be independent.
    """
    if source_config.get("type") == "local_vtt":
        return None
    if source_config.get("transcript_dataset_id") is not None:
        return None
    raw_overlay = source_config.get("overlay")
    if not isinstance(raw_overlay, c.Mapping):
        return None
    overlay_config = t.cast(c.Mapping[str, object], raw_overlay)
    strategy = _effective_overlay_strategy(overlay_config)
    if strategy != "positional" or "overlay_filters" in overlay_config:
        return None
    source_filter = _matching_singleton_filter(source_config.get("filters"))
    base_filter = _matching_singleton_filter(overlay_config.get("base_filters"))
    overlay_filter = _matching_singleton_filter(overlay_config.get("filters"))
    if source_filter is None or base_filter is None or overlay_filter is None:
        return None
    if not (
        source_filter[0] == base_filter[0] == overlay_filter[0]
        and _same_filter_value(source_filter[1], base_filter[1])
        and _same_filter_value(source_filter[1], overlay_filter[1])
    ):
        return None

    source_without_filters = _without_config_keys(
        source_config, excluded={"filters", "overlay"}
    )
    overlay_without_filters = _without_config_keys(
        overlay_config, excluded={"base_filters", "filters"}
    )
    return _GroupedSource(
        name=name,
        config=source_config,
        filter_column=source_filter[0],
        filter_value=source_filter[1],
        base_signature=_normalise_config(source_without_filters),
        overlay_signature=_normalise_config(overlay_without_filters),
    )


def _effective_overlay_strategy(overlay_config: c.Mapping[str, object]) -> str:
    """Resolve the overlay strategy using the same precedence as the data path.

    Returns:
        The normalised configured strategy.
    """
    nested_join = overlay_config.get("join")
    join_config = nested_join if isinstance(nested_join, c.Mapping) else {}
    return str(
        overlay_config.get(
            "strategy",
            overlay_config.get("join_strategy", join_config.get("strategy", "keyed")),
        )
    ).lower()


def _matching_singleton_filter(value: object) -> tuple[str, object] | None:
    """Return a conservative scalar singleton filter, if configured."""
    if not isinstance(value, c.Mapping) or len(value) != 1:
        return None
    column, filter_value = next(iter(value.items()))
    if not isinstance(column, str) or not _is_exact_filter_scalar(filter_value):
        return None
    return column, filter_value


def _is_exact_filter_scalar(value: object) -> bool:
    """Whether a filter value has unambiguous equality semantics.

    Returns:
        Whether the value is a finite, scalar equality operand.
    """
    if value is None or isinstance(value, str | bool | int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _normalise_config(value: object) -> object:
    """Convert config containers into deterministic, comparable values.

    Returns:
        A recursively normalised value suitable for equality comparison.
    """
    if isinstance(value, c.Mapping):
        return tuple(
            sorted((str(key), _normalise_config(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_config(item) for item in value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return (type(value).__name__, value)
    return (type(value).__name__, repr(value))


def _without_config_keys(
    config: c.Mapping[str, object], excluded: set[str]
) -> dict[str, object]:
    """Copy a config mapping while excluding explicitly source-specific keys.

    Returns:
        A deep-copied mapping without excluded keys.
    """
    return {
        str(key): copy.deepcopy(value)
        for key, value in config.items()
        if str(key) not in excluded
    }


def _project_overlay_base_dataset(
    dataset: object,
    audio_column: str,
    source_config: c.Mapping[str, object],
    overlay_config: c.Mapping[str, object],
    source_name: str,
) -> tuple[Dataset | IterableDataset, set[str]]:
    """Validate and remove lazy audio before running an overlay preflight.

    Returns:
        The metadata-only base dataset and the base columns required by the overlay.

    Raises:
        ValueError:
            If the source is not a supported dataset or lacks its audio column.
    """
    if not isinstance(dataset, Dataset | IterableDataset):
        raise ValueError(f"Unsupported audio dataset type: {type(dataset)}")
    column_names = list(dataset.column_names or [])
    if audio_column not in column_names:
        raise ValueError(f"Missing audio column from {source_name}: {audio_column!r}")

    if (
        dataset.features is not None
        and audio_column in dataset.features
        and isinstance(dataset.features[audio_column], Audio)
    ):
        dataset = dataset.cast_column(column=audio_column, feature=Audio(decode=False))

    required_columns = _overlay_base_columns(
        source_config=source_config, overlay_config=overlay_config
    )
    columns = [
        column
        for column in column_names
        if column != audio_column or audio_column in required_columns
    ]
    return dataset.select_columns(columns), required_columns


def _overlay_base_columns(
    source_config: c.Mapping[str, object], overlay_config: c.Mapping[str, object]
) -> set[str]:
    """Return source columns needed while applying an overlay."""
    columns: set[str] = set()
    text_column = source_config.get("text_column")
    if text_column is not None:
        columns.add(str(text_column))
    columns.update(_mapping_keys(source_config.get("filters")))
    columns.update(_mapping_keys(overlay_config.get("base_filters")))

    equality_checks = overlay_config.get(
        "equality_checks", overlay_config.get("checks", {})
    )
    if isinstance(equality_checks, c.Mapping):
        columns.update(str(column) for column in equality_checks)
    elif not isinstance(equality_checks, str) and isinstance(
        equality_checks, c.Iterable
    ):
        for check in equality_checks:
            if isinstance(check, c.Mapping):
                base_column = check.get("base_column", check.get("base"))
                if base_column is not None:
                    columns.add(str(base_column))

    strategy = str(
        overlay_config.get(
            "strategy",
            overlay_config.get(
                "join_strategy",
                _mapping_value(overlay_config, "join", "strategy", "keyed"),
            ),
        )
    ).lower()
    if strategy == "keyed":
        join_config = overlay_config.get("join")
        nested_join = join_config if isinstance(join_config, c.Mapping) else {}
        base_join_column = _first_configured_value(
            overlay_config, ("base_join_column", "base_column")
        )
        if base_join_column is None:
            base_join_column = nested_join.get("base_column")
        if base_join_column is None:
            base_join_column = overlay_config.get("join_column")
        if base_join_column is not None:
            columns.add(str(base_join_column))
    return columns


def _first_configured_value(
    mapping: c.Mapping[str, object], keys: tuple[str, ...]
) -> object:
    """Return the value for the first explicitly configured key."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _mapping_keys(value: object) -> set[str]:
    """Return string keys from a configured mapping."""
    if not isinstance(value, c.Mapping):
        return set()
    return {str(key) for key in value}


def _mapping_value(
    mapping: c.Mapping[str, object], outer_key: str, nested_key: str, default: object
) -> object:
    """Read a nested mapping value without assuming valid overlay configuration.

    Returns:
        The nested value, or ``default`` when the nested configuration is absent.
    """
    nested = mapping.get(outer_key)
    if isinstance(nested, c.Mapping):
        return nested.get(nested_key, default)
    return default


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
    overlay_config = source_config.get("overlay")
    has_overlay = overlay_config is not None
    has_transcript_join = source_config.get("transcript_dataset_id") is not None
    if has_overlay and not has_transcript_join:
        dataset, overlay_base_columns = _project_overlay_base_dataset(
            dataset=dataset,
            audio_column=audio_column,
            source_config=source_config,
            overlay_config=t.cast(c.Mapping[str, object], overlay_config),
            source_name=source_name,
        )
    elif (
        isinstance(dataset, Dataset | IterableDataset)
        and audio_column in (dataset.column_names or [])
        and dataset.features is not None
        and isinstance(dataset.features[audio_column], Audio)
    ):
        dataset = dataset.cast_column(column=audio_column, feature=Audio(decode=False))
        overlay_base_columns = set()
    else:
        overlay_base_columns = set()

    source_filters = source_config.get("filters")
    if source_filters is not None:
        if not isinstance(source_filters, c.Mapping):
            raise ValueError(f"Filters for {source_name} must be a mapping")
        if not isinstance(dataset, Dataset | IterableDataset):
            raise ValueError(f"Unsupported audio dataset type: {type(dataset)}")
        dataset = _filter_dataset_rows(
            dataset=dataset, filters=t.cast(dict[str, object], source_filters)
        )
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
            required_columns=["text", *sorted(overlay_base_columns)],
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

    handled_sources = _preflight_grouped_hub_sources(
        datasets=config.datasets,
        dataset_loader=dataset_loader,
        cache_dir=config.get("cache_dir"),
        token=token,
    )
    for source_name, source_config in config.datasets.items():
        source_name = str(source_name)
        if source_name in handled_sources:
            continue
        if source_config.get("type") == "local_vtt":
            _preflight_local_manifest(
                source_name=source_name, manifest_path=Path(source_config.manifest_path)
            )
        else:
            _preflight_hub_source(
                source_name=source_name,
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
