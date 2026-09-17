"""Functions related to the data loading and processing."""

import atexit
import copy
import hashlib
import io
import json
import logging
import math
import os
import pickle
import re
import shutil
import sqlite3
import tempfile
import typing as t
from collections.abc import Callable, Iterable, Mapping, Sized
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Any
from unicodedata import normalize
from zipfile import ZipFile

import datasets
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
from datasets import iterable_dataset as datasets_iterable
from huggingface_hub import HfApi
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .audio import SoundfileAudio
from .hub_retries import configure_hub_streaming_retries
from .local_vtt import decode_vtt_audio, load_vtt_manifest
from .types import Data
from .utils import (
    NUMERAL_REGEX,
    convert_iterable_dataset_to_dataset,
    convert_numeral_to_words,
    interpret_dataset_name,
    no_datasets_progress_bars,
    validate_immutable_source_revision,
    validate_overlay_revision,
    validate_transcript_revision,
)

logger = logging.getLogger(__package__)

_MAX_HUB_SHARD_CANDIDATES = 100_000


class _HubFileLister(t.Protocol):
    """Hub operation needed for bounded data-file resolution."""

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[str]:
        """Return repository-relative filenames at one revision."""
        ...


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


def _identity_example(example: dict[str, Any]) -> dict[str, Any]:
    """Return an example unchanged while preserving its streaming features."""
    return example


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
    data_files: list[str] | None = None,
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
    if data_files is not None:
        kwargs["data_files"] = data_files
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


def _resolve_hub_data_files(
    dataset_id: str,
    revision: str | None,
    selection: Mapping[str, object] | None,
    *,
    hub_api: object | None = None,
) -> list[str] | None:
    """Resolve an inclusive, numeric Hub shard selection at a pinned revision.

    The result contains repository-relative filenames only. Missing shard numbers are
    allowed because some exports omit empty shards, but at least one configured file
    must exist.

    Args:
        dataset_id:
            Hub dataset repository identifier.
        revision:
            Immutable dataset commit SHA.
        selection:
            Optional mapping with ``template``, ``start``, and inclusive ``end``.
        hub_api (optional):
            Hugging Face API-compatible client. Defaults to an authenticated client.

    Returns:
        Ordered existing filenames, or ``None`` when no selection is configured.

    Raises:
        ValueError:
            If the configuration is invalid or selects no existing pinned files.
    """
    if selection is None:
        return None
    if not isinstance(selection, Mapping):
        raise ValueError("Hub data_file_shards must be a mapping")
    if set(selection) != {"template", "start", "end"}:
        raise ValueError(
            "Hub data_file_shards requires exactly template, start, and end"
        )
    template = selection["template"]
    start = selection["start"]
    end = selection["end"]
    if not isinstance(template, str) or not template:
        raise ValueError("Hub data-file shard template must be a non-empty string")
    fields = re.findall(r"\{([^{}]+)\}", template)
    if (
        template.startswith(("/", "http://", "https://"))
        or "?" in template
        or "#" in template
        or len(fields) != 1
        or re.fullmatch(r"shard(?::0?\d*d)?", fields[0]) is None
        or "{" in re.sub(r"\{[^{}]+\}", "", template)
        or "}" in re.sub(r"\{[^{}]+\}", "", template)
    ):
        raise ValueError(
            "Hub data-file shard template must be a relative path with one "
            "{shard} field"
        )
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end < start
        or end - start + 1 > _MAX_HUB_SHARD_CANDIDATES
    ):
        raise ValueError(
            "Hub data-file shard bounds must be non-negative integers with start <= "
            f"end and at most {_MAX_HUB_SHARD_CANDIDATES:,} candidates"
        )
    validate_immutable_source_revision(
        str(revision or ""), revision_label=f"{dataset_id} data-file revision"
    )
    try:
        candidates = [template.format(shard=shard) for shard in range(start, end + 1)]
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("Invalid Hub data-file shard template") from error
    if len(set(candidates)) != len(candidates) or any(
        candidate.startswith(("/", "http://", "https://"))
        or "?" in candidate
        or "#" in candidate
        or "{" in candidate
        or "}" in candidate
        for candidate in candidates
    ):
        raise ValueError(
            "Hub data-file shard template must produce unique relative filenames"
        )

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api = t.cast(_HubFileLister, hub_api) if hub_api is not None else HfApi(token=token)
    try:
        repo_files = api.list_repo_files(
            repo_id=dataset_id, repo_type="dataset", revision=str(revision)
        )
    except AttributeError as error:
        raise ValueError("Hub API client cannot list repository files") from error
    existing = set(repo_files)
    resolved = [candidate for candidate in candidates if candidate in existing]
    if not resolved:
        raise ValueError(
            f"Hub data-file shard selection for {dataset_id} resolved no existing files"
        )
    return resolved


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
    dataset = _resolve_streaming_features(dataset=dataset)
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
        standardised = standardised.map(function=_identity_example)
        standardised.info.features = features
    return standardised


def _resolve_streaming_features(
    dataset: Dataset | IterableDataset,
) -> Dataset | IterableDataset:
    """Infer features for an untyped streaming dataset without materialising it.

    ``IterableDataset._resolve_features`` is private in the supported datasets
    release, so all use of that compatibility API is kept in this helper.

    Args:
        dataset:
            Dataset whose features may need to be inferred.

    Returns:
        The original dataset when features are already known, otherwise a new
        restartable iterable dataset with inferred features.

    Raises:
        RuntimeError:
            If the datasets version cannot infer features for an untyped stream.
    """
    if not isinstance(dataset, IterableDataset) or dataset.features is not None:
        return dataset

    resolve_features = getattr(dataset, "_resolve_features", None)
    if not callable(resolve_features):
        raise RuntimeError(
            "Cannot infer features for an untyped streaming dataset: "
            "datasets.IterableDataset._resolve_features is unavailable"
        )
    resolved = t.cast(Callable[[], IterableDataset], resolve_features)()
    if resolved.features is None:
        raise RuntimeError(
            "datasets.IterableDataset._resolve_features did not infer streaming "
            "dataset features"
        )
    return resolved


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


def _validate_positional_shard_parity(
    base_files: list[str] | None, overlay_files: list[str] | None
) -> None:
    """Require selected positional base and overlay shards to mirror each other.

    Raises:
        ValueError:
            If only one side is selected or their basename order differs.
    """
    if base_files is None and overlay_files is None:
        return
    if base_files is None or overlay_files is None:
        raise ValueError(
            "Positional shard selection must be configured for both base and overlay"
        )
    base_names = [Path(path).name for path in base_files]
    overlay_names = [Path(path).name for path in overlay_files]
    if base_names != overlay_names:
        raise ValueError(
            "Positional base and overlay shard basenames/orders do not match"
        )


def apply_dataset_overlay(
    base_dataset: Dataset | IterableDataset,
    overlay_dataset: Dataset,
    overlay_config: Mapping[str, object],
) -> Dataset | IterableDataset:
    """Apply a validated, metadata-only transcript overlay to an audio stream.

    The overlay is indexed in a temporary SQLite database.  Consequently the audio
    side remains lazy and the amount of Python memory used by a multi-million-row
    overlay does not grow with its size.  ``strategy`` may be ``keyed`` or
    ``positional``; positional joins validate row order and length while the stream
    is consumed.

    Args:
        base_dataset:
            The filtered audio/metadata dataset.  It is never materialised.
        overlay_dataset:
            The finite, non-streaming overlay dataset.
        overlay_config:
            Nested overlay settings. Supported keys include ``strategy``,
            ``base_join_column``, ``overlay_join_column``, ``equality_checks``,
            ``action_column``, ``recognised_actions``, ``allowed_actions``,
            ``text_policy``, and ``base_filters``.

    Returns:
        A lazy dataset containing only rows with an allowed action and usable text.

    Raises:
        ValueError:
            If the overlay schema, immutable configuration, join, equality checks,
            or text policy is invalid.
    """
    config = overlay_config
    base_dataset = _resolve_streaming_features(dataset=base_dataset)
    base_filters = config.get("base_filters")
    if base_filters is not None:
        if not isinstance(base_filters, Mapping):
            raise ValueError("Overlay base_filters must be a mapping")
        base_dataset = _filter_dataset_rows(
            dataset=base_dataset, filters=t.cast(Mapping[str, object], base_filters)
        )
    overlay_filters = config.get("filters", config.get("overlay_filters"))
    if overlay_filters is not None and not isinstance(overlay_filters, Mapping):
        raise ValueError("Overlay filters must be a mapping")

    join_config = config.get("join", {})
    if not isinstance(join_config, Mapping):
        raise ValueError("Overlay join must be a mapping")
    strategy = _effective_overlay_strategy(overlay_config=config)
    if strategy not in {"keyed", "positional"}:
        raise ValueError(
            f"Unsupported overlay join strategy {strategy!r}; use keyed or positional"
        )
    equality_checks = _overlay_equality_checks(
        config.get("equality_checks", config.get("checks", {}))
    )
    action_column = str(config.get("action_column", "action"))
    recognised_actions = config.get(
        "recognised_actions",
        config.get(
            "recognized_actions",
            ["keep", "relabel", "strip", "drop", "flag", "quarantine"],
        ),
    )
    if isinstance(recognised_actions, str) or not isinstance(
        recognised_actions, Iterable
    ):
        raise ValueError("Overlay recognised_actions must be a non-empty list")
    recognised = {str(action) for action in recognised_actions}
    if not recognised:
        raise ValueError("Overlay recognised_actions must not be empty")
    allowed_actions = config.get("allowed_actions", ["keep", "relabel", "strip"])
    if isinstance(allowed_actions, str) or not isinstance(allowed_actions, Iterable):
        raise ValueError("Overlay allowed_actions must be a non-empty list")
    allowed = {str(action) for action in allowed_actions}
    if not allowed:
        raise ValueError("Overlay allowed_actions must not be empty")
    unknown_allowed = allowed - recognised
    if unknown_allowed:
        raise ValueError(
            "Overlay allowed_actions contains actions outside recognised_actions: "
            + ", ".join(sorted(unknown_allowed))
        )
    text_policy = config.get(
        "text_policy", {"candidates": config.get("text_candidates")}
    )
    if isinstance(text_policy, list):
        text_policy = {"candidates": text_policy}
    if not isinstance(text_policy, Mapping):
        raise ValueError("Overlay text_policy must be a mapping")
    candidates = _overlay_text_candidates(text_policy=t.cast(Mapping, text_policy))

    required_overlay = {action_column}
    required_overlay.update(column for _, column in equality_checks)
    required_overlay.update(column for _, column, _ in candidates)
    if isinstance(overlay_filters, Mapping):
        required_overlay.update(str(column) for column in overlay_filters)
    required_base = {column for column, _ in equality_checks}
    if strategy == "keyed":
        base_join_column = config.get(
            "base_join_column",
            config.get(
                "base_column", join_config.get("base_column", config.get("join_column"))
            ),
        )
        overlay_join_column = config.get(
            "overlay_join_column",
            config.get(
                "overlay_column",
                join_config.get("overlay_column", config.get("join_column")),
            ),
        )
        if base_join_column is None or overlay_join_column is None:
            raise ValueError(
                "Keyed overlays require base_join_column and overlay_join_column"
            )
        base_join_column = str(base_join_column)
        overlay_join_column = str(overlay_join_column)
        required_base.add(base_join_column)
        required_overlay.add(overlay_join_column)
    _require_columns(
        dataset=overlay_dataset,
        columns=sorted(required_overlay),
        dataset_name="overlay",
    )
    _require_columns(
        dataset=base_dataset, columns=sorted(required_base), dataset_name="base"
    )
    # Store only metadata needed by the join. In particular, model logits and other
    # large overlay columns must never enter the SQLite record blobs.
    overlay_dataset = overlay_dataset.select_columns(sorted(required_overlay))
    if isinstance(overlay_filters, Mapping):
        overlay_dataset = _filter_dataset_rows(
            dataset=overlay_dataset,
            filters=t.cast(Mapping[str, object], overlay_filters),
        )
    output_column = str(config.get("output_text_column", "text"))
    index_path = _build_overlay_index(
        overlay_dataset=overlay_dataset,
        strategy=strategy,
        overlay_join_column=(
            str(
                config.get(
                    "overlay_join_column",
                    config.get(
                        "overlay_column",
                        join_config.get("overlay_column", config.get("join_column")),
                    ),
                )
            )
            if strategy == "keyed"
            else None
        ),
        action_column=action_column,
        allowed_actions=allowed,
        recognised_actions=recognised,
        candidates=candidates,
    )
    features = (
        base_dataset.features.copy() if base_dataset.features is not None else None
    )
    if features is None:
        raise ValueError("Base dataset must declare features for an overlay")
    features[output_column] = Value("string")

    generator = _OverlayRows(
        base_dataset=base_dataset,
        index_path=index_path,
        strategy=strategy,
        base_join_column=base_join_column if strategy == "keyed" else None,
        action_column=action_column,
        allowed_actions=allowed,
        candidates=candidates,
        equality_checks=equality_checks,
        output_column=output_column,
    )
    atexit.register(Path(index_path).unlink, missing_ok=True)
    return IterableDataset.from_generator(generator=generator, features=features)


class _OverlayRows:
    """Pickleable per-consumer reader for an overlay index."""

    def __init__(
        self,
        base_dataset: Dataset | IterableDataset,
        index_path: str,
        strategy: str,
        base_join_column: str | None,
        action_column: str,
        allowed_actions: set[str],
        candidates: list[tuple[frozenset[str] | None, str, str | None]],
        equality_checks: list[tuple[str, str]],
        output_column: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.index_path = index_path
        self.strategy = strategy
        self.base_join_column = base_join_column
        self.action_column = action_column
        self.allowed_actions = allowed_actions
        self.candidates = candidates
        self.equality_checks = equality_checks
        self.output_column = output_column

    def __call__(self) -> Iterable[dict[str, Any]]:
        """Yield joined rows using a connection-local duplicate-key table.

        Raises:
            ValueError:
                If a base row cannot be matched or fails an overlay check.
        """
        connection = sqlite3.connect(self.index_path)
        connection.execute(
            "CREATE TEMP TABLE seen_base_keys (key_blob BLOB PRIMARY KEY)"
        )
        position = 0
        try:
            for raw_row in self.base_dataset:
                row = t.cast(dict[str, Any], raw_row)
                if self.strategy == "positional":
                    record = connection.execute(
                        "SELECT row_blob FROM overlay_rows WHERE position = ?",
                        (position,),
                    ).fetchone()
                    if record is None:
                        raise ValueError(
                            "Overlay positional join length mismatch: base has more "
                            f"rows than the filtered overlay at position {position}"
                        )
                    overlay_row = t.cast(dict[str, Any], pickle.loads(record[0]))
                else:
                    base_join_column = self.base_join_column
                    if base_join_column is None:
                        raise ValueError(
                            "Keyed overlay is missing its base join column"
                        )
                    key = row[base_join_column]
                    _validate_join_key(key=key, expected_type=None, side="base")
                    if key is None:
                        raise ValueError("Overlay base join key must not be null")
                    key_blob = sqlite3.Binary(pickle.dumps(key, protocol=5))
                    try:
                        connection.execute(
                            "INSERT INTO seen_base_keys VALUES (?)", (key_blob,)
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate base join key: {key!r}") from error
                    record = connection.execute(
                        "SELECT position, key_type, row_blob FROM overlay_rows "
                        "WHERE key_blob = ?",
                        (key_blob,),
                    ).fetchone()
                    if record is None:
                        overlay_types = {
                            str(value[0])
                            for value in connection.execute(
                                "SELECT DISTINCT key_type FROM overlay_rows"
                            )
                        }
                        if (
                            len(overlay_types) == 1
                            and type(key).__name__ not in overlay_types
                        ):
                            raise ValueError(
                                "Overlay keyed join key type mismatch: base keys are "
                                f"{type(key).__name__}, overlay keys are "
                                f"{next(iter(overlay_types))}"
                            )
                        raise ValueError(
                            f"Overlay keyed join is missing base key {key!r}"
                        )
                    overlay_row = t.cast(dict[str, Any], pickle.loads(record[2]))
                    overlay_key_type = str(record[1])
                    if type(key).__name__ != overlay_key_type:
                        raise ValueError(
                            "Overlay keyed join key type mismatch for "
                            f"{key!r}: base is {type(key).__name__}, overlay is "
                            f"{overlay_key_type}"
                        )
                _check_overlay_equalities(
                    base_row=row,
                    overlay_row=overlay_row,
                    checks=self.equality_checks,
                    position=position,
                )
                text = _select_overlay_text(
                    row=overlay_row,
                    action=overlay_row[self.action_column],
                    allowed_actions=self.allowed_actions,
                    candidates=self.candidates,
                )
                position += 1
                if text is None:
                    continue
                row = dict(row)
                row[self.output_column] = text
                yield row
            total = int(
                connection.execute("SELECT count(*) FROM overlay_rows").fetchone()[0]
            )
            if position != total:
                join_name = "positional" if self.strategy == "positional" else "keyed"
                raise ValueError(
                    f"Overlay {join_name} join length mismatch: base has "
                    f"{position} rows but filtered overlay has {total}"
                )
        finally:
            connection.close()


def _check_overlay_equalities(
    base_row: dict[str, Any],
    overlay_row: dict[str, Any],
    checks: list[tuple[str, str]],
    position: int,
) -> None:
    """Check configured metadata equality without touching the audio field.

    Raises:
        ValueError:
            If a configured value differs or has a different type.
    """
    for base_column, overlay_column in checks:
        base_value = base_row[base_column]
        overlay_value = overlay_row[overlay_column]
        if type(base_value) is not type(overlay_value):
            raise ValueError(
                f"Overlay equality type mismatch at row {position}: base "
                f"{base_column} is {type(base_value).__name__}, overlay "
                f"{overlay_column} is {type(overlay_value).__name__}"
            )
        if base_value != overlay_value:
            raise ValueError(
                f"Overlay equality mismatch at row {position}: base "
                f"{base_column}={base_value!r}, overlay {overlay_column}="
                f"{overlay_value!r}"
            )


def _select_overlay_text(
    row: dict[str, Any],
    action: object,
    allowed_actions: set[str],
    candidates: list[tuple[frozenset[str] | None, str, str | None]],
) -> str | None:
    """Select the first non-blank candidate permitted for an overlay action.

    Returns:
        The selected transcript, or ``None`` for a disallowed/empty action.
    """
    action_name = str(action) if action is not None else ""
    if action_name not in allowed_actions:
        return None
    for actions, column, _ in candidates:
        if actions is not None and action_name not in actions:
            continue
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value
    return None


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


def _build_overlay_index(
    overlay_dataset: Dataset,
    strategy: str,
    overlay_join_column: str | None,
    action_column: str,
    allowed_actions: set[str],
    recognised_actions: set[str],
    candidates: list[tuple[frozenset[str] | None, str, str | None]],
) -> str:
    """Create the bounded-memory SQLite index used by ``apply_dataset_overlay``.

    Returns:
        Temporary SQLite database path.

    Raises:
        ValueError:
            If a key is invalid, duplicated, or no usable rows remain.
    """
    descriptor, index_path = tempfile.mkstemp(
        prefix="hviske-overlay-", suffix=".sqlite"
    )
    os.close(descriptor)
    connection = sqlite3.connect(index_path)
    connection.execute(
        "CREATE TABLE overlay_rows (position INTEGER PRIMARY KEY, key_blob BLOB, "
        "key_type TEXT, row_blob BLOB)"
    )
    connection.execute("CREATE UNIQUE INDEX overlay_key ON overlay_rows(key_blob)")
    usable = 0
    expected_key_type: str | None = None
    try:
        for position, raw_row in enumerate(overlay_dataset):
            row = t.cast(dict[str, Any], raw_row)
            key_blob = None
            key_type = None
            if strategy == "keyed":
                key = row[overlay_join_column or ""]
                _validate_join_key(key=key, expected_type=None, side="overlay")
                if key is None:
                    raise ValueError("Overlay join key must not be null")
                key_blob = sqlite3.Binary(pickle.dumps(key, protocol=5))
                key_type = type(key).__name__
                if expected_key_type is None:
                    expected_key_type = key_type
                elif key_type != expected_key_type:
                    raise ValueError(
                        "Overlay keyed join key type mismatch: row "
                        f"{position} is {key_type}, expected {expected_key_type}"
                    )
            try:
                connection.execute(
                    "INSERT INTO overlay_rows VALUES (?, ?, ?, ?)",
                    (position, key_blob, key_type, pickle.dumps(row, protocol=5)),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"Duplicate overlay join key at filtered overlay row {position}"
                ) from error
            action = _validate_overlay_action(
                action=row[action_column],
                recognised_actions=recognised_actions,
                position=position,
            )
            if action not in allowed_actions:
                continue
            if (
                _select_overlay_text(
                    row=row,
                    action=action,
                    allowed_actions=allowed_actions,
                    candidates=candidates,
                )
                is None
            ):
                raise ValueError(
                    f"Overlay retained action {action!r} at row {position} has no "
                    "usable text candidate (no usable rows remain for this row)"
                )
            usable += 1
        if usable == 0:
            raise ValueError(
                "Overlay contains no usable rows after allowed actions and text "
                "fallbacks"
            )
        connection.commit()
    except Exception:
        connection.close()
        Path(index_path).unlink(missing_ok=True)
        raise
    connection.close()
    return index_path


def _validate_overlay_action(
    action: object, recognised_actions: set[str], position: int
) -> str:
    """Validate one overlay action against the configured action domain.

    Returns:
        The validated action name.

    Raises:
        ValueError:
            If the action is null or outside the recognised action domain.
    """
    if not isinstance(action, str) or action not in recognised_actions:
        value = "null" if action is None else repr(action)
        raise ValueError(
            f"Overlay row {position} has action {value}; expected one of "
            + ", ".join(sorted(recognised_actions))
        )
    return action


def _effective_overlay_strategy(overlay_config: Mapping[str, object]) -> str:
    """Resolve the overlay strategy using the overlay implementation's precedence.

    Returns:
        The normalised configured strategy.
    """
    nested_join = overlay_config.get("join")
    join_config = nested_join if isinstance(nested_join, Mapping) else {}
    return str(
        overlay_config.get(
            "strategy",
            overlay_config.get("join_strategy", join_config.get("strategy", "keyed")),
        )
    ).lower()


def _overlay_equality_checks(value: object) -> list[tuple[str, str]]:
    """Normalise equality-check mappings and list forms.

    Returns:
        Pairs of base and overlay column names.

    Raises:
        ValueError:
            If the configured checks are not mappings with both columns.
    """
    if isinstance(value, Mapping):
        return [(str(base), str(overlay)) for base, overlay in value.items()]
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError("Overlay equality_checks must be a mapping or list")
    checks: list[tuple[str, str]] = []
    for check in value:
        if not isinstance(check, Mapping):
            raise ValueError("Each overlay equality check must be a mapping")
        base = check.get("base_column", check.get("base"))
        overlay = check.get("overlay_column", check.get("overlay"))
        if base is None or overlay is None:
            raise ValueError("Overlay equality checks need base and overlay columns")
        checks.append((str(base), str(overlay)))
    return checks


def _overlay_text_candidates(
    text_policy: Mapping[str, object],
) -> list[tuple[frozenset[str] | None, str, str | None]]:
    """Normalise ordered conditional text candidates.

    Returns:
        Conditional candidates in their configured order.

    Raises:
        ValueError:
            If candidates are missing, empty, or malformed.
    """
    raw_candidates = text_policy.get("candidates", text_policy.get("text_candidates"))
    if raw_candidates is None:
        raw_candidates = [
            {"column": "new_text", "actions": ["relabel", "strip"]},
            {"column": "reference_text"},
        ]
    if isinstance(raw_candidates, str) or not isinstance(raw_candidates, Iterable):
        raise ValueError("Overlay text candidates must be an ordered list")
    candidates: list[tuple[frozenset[str] | None, str, str | None]] = []
    for candidate in raw_candidates:
        if isinstance(candidate, str):
            candidates.append((None, candidate, None))
            continue
        if not isinstance(candidate, Mapping) or candidate.get("column") is None:
            raise ValueError("Each overlay text candidate needs a column")
        actions = candidate.get("actions")
        if actions is None and candidate.get("action") is not None:
            actions = [candidate["action"]]
        when = candidate.get("when")
        if actions is None and isinstance(when, Mapping):
            actions = when.get("actions", when.get("action"))
        if isinstance(actions, str):
            action_set = frozenset({actions})
        elif actions is None:
            action_set = None
        else:
            action_set = frozenset(str(action) for action in actions)
        candidates.append((action_set, str(candidate["column"]), None))
    if not candidates:
        raise ValueError("Overlay text candidates must not be empty")
    return candidates


def _require_columns(
    dataset: Dataset | IterableDataset, columns: list[str], dataset_name: str
) -> None:
    available = set(dataset.column_names or [])
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Missing {dataset_name} dataset columns: {missing}")


def join_audio_and_transcripts(
    audio_dataset: Dataset | IterableDataset,
    transcript_dataset: Dataset,
    audio_join_column: str,
    transcript_join_column: str,
    transcript_text_column: str,
) -> Dataset | IterableDataset:
    """Join a streaming audio dataset to an indexed transcript dataset.

    The audio side is never materialised. The compact transcript side is indexed in
    memory, then the audio stream is filtered lazily to keys with usable transcripts.
    Empty transcript rows and audio rows without a transcript are skipped.

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
            If a configured column is absent, transcript keys are duplicated, or no
            usable transcripts remain.
    """
    audio_dataset = _resolve_streaming_features(dataset=audio_dataset)
    _require_columns(
        dataset=audio_dataset, columns=[audio_join_column], dataset_name="audio"
    )
    _require_columns(
        dataset=transcript_dataset,
        columns=[transcript_join_column, transcript_text_column],
        dataset_name="transcript",
    )
    transcript_by_key: dict[object, str] = {}
    seen_transcript_keys: set[object] = set()
    key_type: type[object] | None = None
    for raw_row in transcript_dataset:
        row = t.cast(dict[str, Any], raw_row)
        key = row[transcript_join_column]
        _validate_join_key(key=key, expected_type=key_type, side="transcript")
        if key_type is None:
            key_type = type(key)
        if key in seen_transcript_keys:
            raise ValueError(f"Duplicate transcript key: {key!r}")
        seen_transcript_keys.add(key)
        text = row[transcript_text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        transcript_by_key[key] = text

    if not transcript_by_key:
        raise ValueError("Transcript dataset contains no usable transcripts")

    if audio_dataset.features is None:
        raise ValueError("Audio dataset must declare features")
    joined_features = audio_dataset.features.copy()
    joined_features["text"] = Value("string")
    has_transcript = partial(
        _has_transcript,
        audio_join_column=audio_join_column,
        transcript_by_key=transcript_by_key,
        key_type=key_type,
    )
    add_transcript = partial(
        _add_transcript,
        audio_join_column=audio_join_column,
        transcript_by_key=transcript_by_key,
    )
    if isinstance(audio_dataset, IterableDataset):
        filtered_audio = _filter_streaming_dataset(
            dataset=audio_dataset, function=has_transcript
        )
    else:
        filtered_audio = audio_dataset.filter(has_transcript)
    return t.cast(
        Dataset | IterableDataset,
        filtered_audio.map(add_transcript, features=joined_features),
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


def _add_transcript(
    example: dict[str, Any],
    audio_join_column: str,
    transcript_by_key: dict[object, str],
) -> dict[str, Any]:
    """Add the indexed transcript to an audio example.

    Returns:
        The audio example with its transcript.
    """
    example["text"] = transcript_by_key[example[audio_join_column]]
    return example


def _has_transcript(
    example: dict[str, Any],
    audio_join_column: str,
    transcript_by_key: dict[object, str],
    key_type: type[object] | None,
) -> bool:
    """Return whether an audio example has a usable indexed transcript."""
    key = example[audio_join_column]
    _validate_join_key(key=key, expected_type=key_type, side="audio")
    return key in transcript_by_key


def _row_matches_filters(
    row: Mapping[str, object], filters: Mapping[str, object]
) -> bool:
    """Return whether a row exactly matches every configured filter."""
    return all(row[column] == expected for column, expected in filters.items())


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
    configure_hub_streaming_retries(
        retry_config=t.cast(
            Mapping[str, object] | None, config.get("hub_streaming_retries")
        )
    )

    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    probabilities = config.dataset_probabilities
    if probabilities is not None:
        probabilities = _validate_dataset_probabilities(
            probabilities=probabilities, dataset_count=len(config.datasets)
        )

    # Import locally because the materialiser deliberately reuses this module's strict
    # overlay implementation.
    from .materialised_overlays import (
        configured_materialised_overlay_root,
        materialised_source_files,
        validate_materialised_overlay_root,
    )

    materialised_root = configured_materialised_overlay_root(config=config)
    materialised_files: dict[str, list[str]] = {}
    if materialised_root is not None:
        materialised_manifest = validate_materialised_overlay_root(
            config=config, root=materialised_root
        )
        materialised_files = materialised_source_files(
            manifest=materialised_manifest, root=materialised_root
        )

    all_datasets: list[Dataset | IterableDataset] = []
    for dataset_name, dataset_config in config.datasets.items():
        if is_main_process:
            logger.info(f"Loading dataset {dataset_name!r}")

        is_local_vtt = dataset_config.get("type") == "local_vtt"
        is_materialised = dataset_name in materialised_files
        immutable_revision_env = dataset_config.get("immutable_revision_env")
        if immutable_revision_env is not None:
            validate_immutable_source_revision(
                str(dataset_config.get("revision") or ""),
                revision_label=str(immutable_revision_env),
            )
        transcript_dataset_id = dataset_config.get("transcript_dataset_id")
        transcript_revision = None
        if transcript_dataset_id is not None:
            transcript_revision = validate_transcript_revision(
                str(dataset_config.transcript_revision)
            )
        overlay_config = dataset_config.get("overlay")
        overlay_revision = None
        if overlay_config is not None:
            overlay_revision = validate_overlay_revision(
                str(overlay_config.get("revision") or "")
            )
        base_data_files: list[str] | None = None
        overlay_data_files: list[str] | None = None
        is_hub_dataset = (
            not is_local_vtt
            and not is_materialised
            and not Path(dataset_config.id).exists()
        )
        if is_hub_dataset:
            base_data_files = _resolve_hub_data_files(
                dataset_id=str(dataset_config.id),
                revision=dataset_config.get("revision"),
                selection=t.cast(
                    Mapping[str, object] | None, dataset_config.get("data_file_shards")
                ),
            )
            if overlay_config is not None:
                overlay_data_files = _resolve_hub_data_files(
                    dataset_id=str(overlay_config.id),
                    revision=overlay_revision,
                    selection=t.cast(
                        Mapping[str, object] | None,
                        overlay_config.get("data_file_shards"),
                    ),
                )
                if (
                    _effective_overlay_strategy(
                        overlay_config=t.cast(Mapping[str, object], overlay_config)
                    )
                    == "positional"
                ):
                    _validate_positional_shard_parity(
                        base_files=base_data_files, overlay_files=overlay_data_files
                    )

        if is_materialised:
            with no_datasets_progress_bars():
                ds = load_dataset(
                    "parquet",
                    data_files=materialised_files[dataset_name],
                    split="train",
                    streaming=True,
                    cache_dir=config.cache_dir,
                )
        elif is_local_vtt:
            ds = load_vtt_manifest(
                manifest_path=Path(dataset_config.manifest_path),
                min_seconds=config.min_seconds_per_example,
                max_seconds=config.max_seconds_per_example,
                num_shards=dataset_config.get("local_vtt_num_shards", 1),
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
            if base_data_files is not None:
                kwargs["data_files"] = base_data_files
            with no_datasets_progress_bars():
                ds = load_dataset(**kwargs)

        if not isinstance(ds, Dataset | IterableDataset):
            raise ValueError(f"Unsupported dataset type: {type(ds)}")
        ds = _resolve_streaming_features(dataset=ds)

        if not is_local_vtt:
            audio_column = str(dataset_config.audio_column)
            if audio_column in (ds.column_names or []):
                ds = ds.cast_column(column=audio_column, feature=Audio(decode=False))

        row_filters = dataset_config.get("filters")
        if row_filters is not None:
            ds = _filter_dataset_rows(
                dataset=ds, filters=t.cast(Mapping[str, object], row_filters)
            )

        if overlay_config is not None and not is_materialised:
            overlay = _load_transcript_dataset(
                dataset_id=str(overlay_config.id),
                subset=overlay_config.get("subset"),
                split=str(overlay_config.get("split", "train")),
                revision=overlay_revision,
                cache_dir=config.cache_dir,
                trust_remote_code=overlay_config.get("trust_remote_code", False),
                data_files=overlay_data_files,
            )
            ds = apply_dataset_overlay(
                base_dataset=ds,
                overlay_dataset=overlay,
                overlay_config=t.cast(Mapping[str, object], overlay_config),
            )

        if not is_local_vtt and dataset_config.text_column != "text":
            ds = ds.rename_column(dataset_config.text_column, "text")
        if not is_local_vtt and dataset_config.audio_column != "audio":
            ds = ds.rename_column(dataset_config.audio_column, "audio")

        if transcript_dataset_id is not None:
            transcript = _load_transcript_dataset(
                dataset_id=transcript_dataset_id,
                subset=dataset_config.get("transcript_subset"),
                split=dataset_config.get("transcript_split", "train"),
                revision=transcript_revision,
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

        shuffle_buffer_size = dataset_config.get(
            "shuffle_buffer_size", config.get("shuffle_buffer_size", 1000)
        )
        if (
            not isinstance(shuffle_buffer_size, int)
            or isinstance(shuffle_buffer_size, bool)
            or shuffle_buffer_size <= 0
        ):
            raise ValueError("shuffle_buffer_size must be a positive integer")
        if isinstance(ds, IterableDataset):
            ds = ds.shuffle(seed=config.seed, buffer_size=shuffle_buffer_size)
        else:
            ds = ds.shuffle(seed=config.seed)

        if not is_local_vtt:
            ds = ds.cast_column(
                column="audio", feature=Audio(sampling_rate=config.model.sampling_rate)
            )
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
        )

        all_datasets.append(ds)

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
            datasets=t.cast(list[IterableDataset], all_datasets),
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
            If a filter column is missing.
    """
    dataset = _resolve_streaming_features(dataset=dataset)
    if dataset.features is None:
        raise ValueError("Cannot apply row filters without declared dataset features")
    missing_columns = [column for column in filters if column not in dataset.features]
    if missing_columns:
        raise ValueError(
            "Cannot apply row filters; missing dataset features: "
            + ", ".join(missing_columns)
        )

    original_features = dataset.features
    function = partial(_row_matches_filters, filters=dict(filters))
    if isinstance(dataset, IterableDataset):
        filtered = _filter_streaming_dataset(dataset=dataset, function=function)
    else:
        filtered = dataset.filter(function=function)
    filtered.info.features = original_features
    return filtered


def _filter_streaming_dataset(
    dataset: IterableDataset, function: Callable[..., bool]
) -> IterableDataset:
    """Filter a stream, working around repeated-filter failures in datasets 3.6.0.

    In datasets 3.6.0, ``IterableDataset.filter`` gives a formatting wrapper no
    features when its underlying iterable is already typed. The wrapper reports
    itself as typed, and the next filter then fails while expanding its missing
    features. This is the state produced by one ordinary filter over a stream with
    declared features. Try the public API first so later datasets releases use their
    native implementation; only reconstruct the failed operation through private
    iterable classes when that exact typed-underlying state triggers the defect.

    The compatibility path mirrors datasets 3.6.0's public implementation while
    supplying the underlying declared features. It retains the lazy iterable graph,
    formatting, shuffling, distributed sharding and repository tokens. If those
    private internals change or no usable underlying features exist, it fails with a
    clear ``RuntimeError`` rather than silently changing stream semantics.

    Args:
        dataset:
            Streaming dataset to filter.
        function:
            Predicate used to retain rows.

    Returns:
        A lazy filtered streaming dataset.

    Raises:
        RuntimeError:
            If the datasets compatibility internals are unavailable or incompatible.
        TypeError:
            If filtering fails for a reason other than the datasets 3.6.0 defect.
    """
    try:
        return dataset.filter(function=function)
    except TypeError as error:
        if not _is_datasets_360_filter_construction_defect(
            dataset=dataset, error=error
        ):
            raise

        ex_iterable = dataset._ex_iterable
        underlying_features = getattr(ex_iterable, "features", None)
        formatted_type = getattr(datasets_iterable, "FormattedExamplesIterable", None)
        filtered_type = getattr(datasets_iterable, "FilteredExamplesIterable", None)
        if (
            underlying_features is None
            or formatted_type is None
            or filtered_type is None
        ):
            raise RuntimeError(
                "Cannot apply repeated streaming filters safely: the datasets "
                "compatibility internals or typed features are unavailable"
            ) from error

        try:
            formatted = formatted_type(
                ex_iterable,
                formatting=dataset._formatting,
                features=underlying_features,
                token_per_repo_id=dataset._token_per_repo_id,
            )
            filtered = filtered_type(
                formatted, function=function, formatting=dataset._formatting
            )
            return IterableDataset(
                ex_iterable=filtered,
                info=dataset._info,
                split=dataset._split,
                formatting=dataset._formatting,
                shuffling=copy.deepcopy(dataset._shuffling),
                distributed=copy.deepcopy(dataset._distributed),
                token_per_repo_id=dataset._token_per_repo_id,
            )
        except (AttributeError, TypeError) as compatibility_error:
            raise RuntimeError(
                "Cannot apply repeated streaming filters safely: datasets private "
                "iterable APIs are incompatible"
            ) from compatibility_error


def _is_datasets_360_filter_construction_defect(
    dataset: IterableDataset, error: TypeError
) -> bool:
    """Return whether an error is datasets 3.6.0's repeated-filter defect."""
    if datasets.__version__ != "3.6.0" or error.args != (
        "'NoneType' object is not a mapping",
    ):
        return False

    ex_iterable = getattr(dataset, "_ex_iterable", None)
    if (
        dataset.features is None
        or ex_iterable is None
        or not getattr(ex_iterable, "is_typed", False)
        or getattr(ex_iterable, "features", None) is None
    ):
        return False

    formatted_type = getattr(datasets_iterable, "FormattedExamplesIterable", None)
    filtered_type = getattr(datasets_iterable, "FilteredExamplesIterable", None)
    if not isinstance(formatted_type, type) or not isinstance(filtered_type, type):
        return False

    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            frame.f_globals.get("__name__") == datasets_iterable.__name__
            and frame.f_code.co_name == "__init__"
            and isinstance(frame.f_locals.get("self"), filtered_type)
        ):
            formatted = frame.f_locals.get("ex_iterable")
            if (
                isinstance(formatted, formatted_type)
                and getattr(formatted, "is_typed", False)
                and getattr(formatted, "features", None) is None
            ):
                return True
        traceback = traceback.tb_next
    return False


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
    if isinstance(dataset, IterableDataset):
        dataset = t.cast(Data, _resolve_streaming_features(dataset=dataset))
    elif isinstance(dataset, IterableDatasetDict):
        dataset = t.cast(
            Data,
            IterableDatasetDict(
                {
                    split_name: _resolve_streaming_features(dataset=split_dataset)
                    for split_name, split_dataset in dataset.items()
                }
            ),
        )

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
            num_proc=None if num_proc == 1 else num_proc,
            desc="Filtering dataset",
            keep_in_memory=True,
        )
    elif isinstance(dataset, IterableDataset):
        filtered = _filter_streaming_dataset(dataset=dataset, function=filter_fn)
    elif isinstance(dataset, IterableDatasetDict):
        filtered = IterableDatasetDict(
            {
                split_name: _filter_streaming_dataset(
                    dataset=split_dataset, function=filter_fn
                )
                for split_name, split_dataset in dataset.items()
            }
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

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
    if isinstance(dataset, IterableDataset):
        dataset = t.cast(Data, _resolve_streaming_features(dataset=dataset))
    elif isinstance(dataset, IterableDatasetDict):
        dataset = t.cast(
            Data,
            IterableDatasetDict(
                {
                    split_name: _resolve_streaming_features(dataset=split_dataset)
                    for split_name, split_dataset in dataset.items()
                }
            ),
        )

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
            num_proc=None if num_proc == 1 else num_proc,
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
    model_input_names = getattr(processor, "model_input_names", [])
    if (
        "input_features" in model_input_names
        and not hasattr(processor, "get_decoder_prompt_ids")
        and not _is_parakeet_rnnt_processor(processor)
    ):
        return Features(
            input_features=Sequence(Sequence(Value("float64"))),
            attention_mask=Sequence(Value("int64")),
            labels=Sequence(Value("int64")),
            input_length=Value("int64"),
            num_seconds=Value("float64"),
        )
    if not hasattr(
        processor, "get_decoder_prompt_ids"
    ) and not _is_parakeet_rnnt_processor(processor):
        return Features(
            input_values=Sequence(Value("float64")),
            labels=Sequence(Value("int64")),
            input_length=Value("int64"),
            num_seconds=Value("float64"),
        )
    feature_schema: dict[str, object] = {
        "input_features": Sequence(Sequence(Value("float64"))),
        "decoder_input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "input_length": Value("int64"),
        "num_seconds": Value("float64"),
    }
    if getattr(processor, "uses_length", False):
        feature_schema["length"] = Value("int64")
    else:
        feature_schema["attention_mask"] = Sequence(Value("int64"))
    return Features(**feature_schema)


def _is_parakeet_rnnt_processor(processor: Callable | None) -> bool:
    """Return whether a processor follows the Parakeet RNNT contract."""
    return (
        processor is not None
        and hasattr(processor, "blank_token")
        and str(getattr(processor, "decoder_type", "")).lower() == "rnnt"
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

    Raises:
        ValueError:
            If Whisper processing has no language for the example.
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
    sampling_rate = int(audio["sampling_rate"])
    num_seconds = len(audio_array) / sampling_rate

    # Normalise and augment audio
    normalise = ta.PeakNormalization(p=1.0) if normalise_audio else ta.Identity()
    if augment_audio:
        download_background_noises()
        background_noise = ta.AddBackgroundNoise(
            background_paths=Path("background-noises"), p=0.7, sample_rate=sampling_rate
        )
        background_noise.audio = SoundfileAudio(sample_rate=sampling_rate, mono=True)
        augment = ta.Compose(
            [
                ta.PeakNormalization(p=1.0),
                ta.Gain(p=1.0),
                background_noise,
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
    else:
        augment = ta.Identity()
    normalise_and_augment = ta.Compose([normalise, augment], p=1.0)
    audio_array = normalise_and_augment(
        torch.tensor(audio_array).unsqueeze(0).unsqueeze(0), sample_rate=sampling_rate
    )[0, 0]

    # If we don't have a processor then we just re-assign the normalised audio and
    # return the processed example
    if processor is None:
        example[audio_column]["array"] = audio_array
        return example

    example_language = (
        example.get(language_column) if language_column is not None else None
    ) or language
    tokenizer = getattr(processor, "tokenizer", None)
    set_prefix_tokens = getattr(tokenizer, "set_prefix_tokens", None)
    is_whisper = callable(set_prefix_tokens)
    if is_whisper:
        if example_language is None:
            raise ValueError(
                "Whisper processing requires an ISO language for every example."
            )
        # Whisper's prefix is mutable processor state. Set it immediately before each
        # example so a multilingual stream cannot inherit the previous example's prompt.
        set_prefix_tokens(language=example_language, task="transcribe")
    elif _is_parakeet_rnnt_processor(processor) or (
        example_language is not None and hasattr(processor, "get_decoder_prompt_ids")
    ):
        if _is_parakeet_rnnt_processor(processor):
            # RNNT processors must create decoder inputs from audio and text together.
            processed = processor(
                audio_array, text=example[text_column], sampling_rate=sampling_rate
            )
        else:
            # Cohere needs the language prompt and transcript in one processor call.
            processed = processor(
                audio_array,
                language=example_language,
                text=example[text_column],
                punctuation=punctuation,
                sampling_rate=sampling_rate,
            )
        example["input_features"] = _to_python(processed["input_features"][0])
        if "attention_mask" in processed:
            example["attention_mask"] = _to_python(processed["attention_mask"][0])
        elif "length" in processed:
            example["length"] = _to_python(processed["length"][0])
        else:
            raise ValueError(
                "Prompt-aware processor must return attention_mask or length."
            )
        example["decoder_input_ids"] = _to_python(processed["decoder_input_ids"][0])
        example["labels"] = _to_python(processed["labels"][0])
        labels = t.cast(Sized, example["labels"])
        example["input_length"] = len(labels)
        example["num_seconds"] = num_seconds
        return example

    # Process the audio for Whisper and Wav2Vec2.
    processed = processor(audio_array, sampling_rate=sampling_rate)
    audio_feature_name = (
        "input_values" if "input_values" in processed else "input_features"
    )
    audio_array = processed[audio_feature_name][0]
    example[audio_feature_name] = audio_array
    if audio_feature_name == "input_features" and "attention_mask" in processed:
        example["attention_mask"] = _to_python(processed["attention_mask"][0])
    example["num_seconds"] = num_seconds

    # Some remote processors require audio for every call, so tokenise labels through
    # their tokenizer rather than invoking the processor with text alone.
    label_processor = tokenizer if tokenizer is not None else processor
    example["labels"] = label_processor(
        text=example[text_column], truncation=True
    ).input_ids
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
