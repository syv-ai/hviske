"""Build and validate durable local positional-overlay datasets."""

import hashlib
import json
import logging
import os
import shutil
import typing as t
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset, IterableDataset, load_dataset
from omegaconf import DictConfig, OmegaConf

from p1_dataset.source import harden_p1_logging

from .data import (
    _effective_overlay_strategy,
    _filter_dataset_rows,
    _load_transcript_dataset,
    _resolve_hub_data_files,
    _resolve_streaming_features,
    _validate_positional_shard_parity,
    apply_dataset_overlay,
)
from .utils import validate_immutable_source_revision, validate_overlay_revision

logger = logging.getLogger(__package__)

_FORMAT = "hviske-materialised-overlays-v1"
_MANIFEST = "manifest.json"
_COMPLETE = "COMPLETE"
_WRITE_BATCH_SIZE = 512
_DEFAULT_MINIMUM_FREE_BYTES = 10 * 1024**3
_SCHEMA = pa.schema(
    [
        pa.field(
            "audio",
            pa.struct(
                [
                    pa.field("bytes", pa.binary(), nullable=False),
                    pa.field("path", pa.string()),
                ]
            ),
            nullable=False,
        ),
        pa.field("text", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

JsonValue: t.TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
DatasetLoader: t.TypeAlias = Callable[..., object]


def configured_materialised_overlay_root(config: DictConfig) -> Path | None:
    """Resolve and enforce the configured materialised-overlay root policy.

    Args:
        config:
            Resolved finetuning configuration.

    Returns:
        The configured root, or ``None`` for configurations that permit remote joins.

    Raises:
        MaterialisedOverlayError:
            If production requires an artefact but no root is configured.
    """
    raw_root = config.get("materialised_overlay_root")
    root = (
        None
        if raw_root is None or str(raw_root).strip() in {"", "null"}
        else Path(str(raw_root))
    )
    if bool(config.get("require_materialised_overlays", False)) and root is None:
        raise MaterialisedOverlayError(
            "This configuration requires HVISKE_MATERIALISED_OVERLAYS_ROOT to point "
            "to a complete validated positional-overlay artefact"
        )
    return root


class MaterialisedOverlayError(ValueError):
    """Raised when a local overlay artefact is unsafe or inconsistent."""


def materialise_finetuning_overlays(
    config: DictConfig,
    output_root: Path,
    *,
    minimum_free_bytes: int = _DEFAULT_MINIMUM_FREE_BYTES,
    dataset_loader: DatasetLoader = load_dataset,
    hub_api: object | None = None,
) -> dict[str, JsonValue]:
    """Materialise configured positional overlays one physical shard at a time.

    Args:
        config:
            Resolved finetuning configuration.
        output_root:
            Destination directory for the immutable artefact.
        minimum_free_bytes (optional):
            Free-space reserve that must remain while writing. Defaults to 10 GiB.
        dataset_loader (optional):
            Dataset loader used for base shards. Defaults to ``load_dataset``.
        hub_api (optional):
            Hub API-compatible file lister. Defaults to an authenticated client.

    Returns:
        The completed build manifest.

    Raises:
        MaterialisedOverlayError:
            If source data or an existing partial artefact is unsafe or inconsistent.
    """
    harden_p1_logging()
    if minimum_free_bytes < 0:
        raise MaterialisedOverlayError("minimum_free_bytes must be non-negative")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / _MANIFEST
    complete_path = output_root / _COMPLETE
    if complete_path.exists():
        return validate_materialised_overlay_root(config=config, root=output_root)
    if manifest_path.exists():
        manifest = _read_json_mapping(path=manifest_path, label="manifest")
        _validate_manifest(config=config, root=output_root, manifest=manifest)
        _write_complete_marker(root=output_root, manifest_path=manifest_path)
        return t.cast(dict[str, JsonValue], manifest)

    source_manifests: list[JsonValue] = []
    for source_name, source_config in _positional_sources(config=config):
        source_manifests.append(
            _materialise_source(
                source_name=source_name,
                source_config=source_config,
                output_root=output_root,
                minimum_free_bytes=minimum_free_bytes,
                dataset_loader=dataset_loader,
                hub_api=hub_api,
                cache_dir=config.get("cache_dir"),
            )
        )
    if not source_manifests:
        raise MaterialisedOverlayError("No positional overlay sources are configured")

    manifest: dict[str, JsonValue] = {"format": _FORMAT, "sources": source_manifests}
    _atomic_json_write(path=manifest_path, value=manifest)
    _write_complete_marker(root=output_root, manifest_path=manifest_path)
    return manifest


def _atomic_json_write(path: Path, value: Mapping[str, JsonValue]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    with partial.open("w", encoding="utf-8") as file:
        file.write(payload)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def _materialise_source(
    source_name: str,
    source_config: Mapping[str, object],
    output_root: Path,
    minimum_free_bytes: int,
    dataset_loader: DatasetLoader,
    hub_api: object | None,
    cache_dir: str | None,
) -> dict[str, JsonValue]:
    overlay_config = t.cast(Mapping[str, object], source_config["overlay"])
    base_revision = validate_immutable_source_revision(
        str(source_config.get("revision") or ""),
        revision_label=f"{source_name} base revision",
    )
    overlay_revision = validate_overlay_revision(
        str(overlay_config.get("revision") or "")
    )
    _validate_output_contract(source_name=source_name, source_config=source_config)
    base_files = _resolve_hub_data_files(
        dataset_id=str(source_config["id"]),
        revision=base_revision,
        selection=t.cast(Mapping[str, object], source_config.get("data_file_shards")),
        hub_api=hub_api,
    )
    overlay_files = _resolve_hub_data_files(
        dataset_id=str(overlay_config["id"]),
        revision=overlay_revision,
        selection=t.cast(Mapping[str, object], overlay_config.get("data_file_shards")),
        hub_api=hub_api,
    )
    _validate_positional_shard_parity(
        base_files=base_files, overlay_files=overlay_files
    )
    if base_files is None or overlay_files is None:
        raise MaterialisedOverlayError(
            f"{source_name} must select mirrored physical shards"
        )

    provenance = _source_provenance(
        source_name=source_name,
        source_config=source_config,
        base_files=base_files,
        overlay_files=overlay_files,
    )
    shards: list[JsonValue] = []
    for base_file, overlay_file in zip(base_files, overlay_files, strict=True):
        shards.append(
            _materialise_shard(
                source_name=source_name,
                source_config=source_config,
                overlay_config=overlay_config,
                base_file=base_file,
                overlay_file=overlay_file,
                base_revision=base_revision,
                overlay_revision=overlay_revision,
                output_root=output_root,
                minimum_free_bytes=minimum_free_bytes,
                dataset_loader=dataset_loader,
                cache_dir=cache_dir,
                provenance_sha256=_canonical_sha256(provenance),
            )
        )
    return {**provenance, "shards": shards}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _normalise(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalise(value: object) -> JsonValue:
    if isinstance(value, DictConfig):
        return _normalise(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalise(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise MaterialisedOverlayError(f"Unsupported configuration value: {type(value)}")


def _materialise_shard(
    source_name: str,
    source_config: Mapping[str, object],
    overlay_config: Mapping[str, object],
    base_file: str,
    overlay_file: str,
    base_revision: str,
    overlay_revision: str,
    output_root: Path,
    minimum_free_bytes: int,
    dataset_loader: DatasetLoader,
    cache_dir: str | None,
    provenance_sha256: str,
) -> dict[str, JsonValue]:
    shard_name = Path(base_file).name
    relative_path = Path(source_name) / shard_name
    output_path = _safe_child(root=output_root, relative=relative_path.as_posix())
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(output_path.suffix + ".partial").unlink(missing_ok=True)
    receipt_path.with_suffix(receipt_path.suffix + ".partial").unlink(missing_ok=True)

    expected = {
        "format": _FORMAT,
        "source": source_name,
        "path": relative_path.as_posix(),
        "base_file": base_file,
        "overlay_file": overlay_file,
        "provenance_sha256": provenance_sha256,
    }
    if output_path.exists():
        return _resume_existing_shard(
            output_root=output_root,
            output_path=output_path,
            receipt_path=receipt_path,
            expected=expected,
        )
    if receipt_path.exists():
        raise MaterialisedOverlayError(
            f"Receipt exists without its Parquet output: {receipt_path}"
        )
    _require_free_space(path=output_root, reserve=minimum_free_bytes)

    base = dataset_loader(
        path=str(source_config["id"]),
        name=source_config.get("subset"),
        split=str(source_config.get("train_name", "train")),
        revision=base_revision,
        data_files=[base_file],
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or True,
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=bool(source_config.get("trust_remote_code", False)),
    )
    if not isinstance(base, Dataset | IterableDataset):
        raise MaterialisedOverlayError("Base shard loader returned an unsupported type")
    base = _resolve_streaming_features(dataset=base)
    audio_column = str(source_config.get("audio_column", "audio"))
    if audio_column not in (base.column_names or []):
        raise MaterialisedOverlayError(f"{source_name} base shard has no audio column")
    base = base.cast_column(column=audio_column, feature=Audio(decode=False))
    row_filters = source_config.get("filters")
    if row_filters is not None:
        if not isinstance(row_filters, Mapping):
            raise MaterialisedOverlayError("Source filters must be a mapping")
        base = _filter_dataset_rows(dataset=base, filters=row_filters)

    overlay = _load_transcript_dataset(
        dataset_id=str(overlay_config["id"]),
        subset=t.cast(str | None, overlay_config.get("subset")),
        split=str(overlay_config.get("split", "train")),
        revision=overlay_revision,
        cache_dir=cache_dir,
        trust_remote_code=bool(overlay_config.get("trust_remote_code", False)),
        dataset_loader=dataset_loader,
        data_files=[overlay_file],
    )
    joined = apply_dataset_overlay(
        base_dataset=base, overlay_dataset=overlay, overlay_config=overlay_config
    )
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    rows = _write_rows(
        rows=t.cast(Iterable[Mapping[str, object]], joined),
        path=partial_path,
        source_name=source_name,
        audio_column=audio_column,
        minimum_free_bytes=minimum_free_bytes,
        output_root=output_root,
    )
    _validate_parquet(
        path=partial_path, expected_rows=rows, expected_source=source_name
    )
    os.replace(partial_path, output_path)
    receipt: dict[str, JsonValue] = {
        **expected,
        "rows": rows,
        "sha256": _sha256_file(output_path),
    }
    _atomic_json_write(path=receipt_path, value=receipt)
    logger.info("Materialised %s rows for %s/%s", rows, source_name, shard_name)
    return receipt


def _require_free_space(path: Path, reserve: int) -> None:
    if shutil.disk_usage(path).free < reserve:
        raise MaterialisedOverlayError(
            f"Free disk space fell below the required {reserve} byte reserve"
        )


def _resume_existing_shard(
    output_root: Path,
    output_path: Path,
    receipt_path: Path,
    expected: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    rows = _validate_parquet(path=output_path, expected_source=str(expected["source"]))
    digest = _sha256_file(output_path)
    if receipt_path.exists():
        receipt = _read_json_mapping(path=receipt_path, label="shard receipt")
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise MaterialisedOverlayError(
                f"Shard receipt provenance mismatch: {receipt_path}"
            )
        if receipt.get("rows") != rows or receipt.get("sha256") != digest:
            raise MaterialisedOverlayError(
                f"Shard receipt checksum/count mismatch: {receipt_path}"
            )
        return t.cast(dict[str, JsonValue], receipt)

    receipt: dict[str, JsonValue] = {**expected, "rows": rows, "sha256": digest}
    _atomic_json_write(path=receipt_path, value=receipt)
    logger.info(
        "Recovered interrupted receipt for %s", output_path.relative_to(output_root)
    )
    return receipt


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterialisedOverlayError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise MaterialisedOverlayError(f"{label.capitalize()} must be a JSON object")
    return t.cast(Mapping[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MaterialisedOverlayError(
            f"Cannot checksum materialised file: {path}"
        ) from error
    return digest.hexdigest()


def _validate_parquet(
    path: Path, expected_rows: int | None = None, expected_source: str | None = None
) -> int:
    try:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        rows = parquet.metadata.num_rows
        if not schema.equals(_SCHEMA, check_metadata=False):
            raise MaterialisedOverlayError(
                f"Unexpected materialised Parquet schema: {path}"
            )
        for batch in parquet.iter_batches(batch_size=_WRITE_BATCH_SIZE):
            for row in batch.to_pylist():
                audio = row.get("audio")
                source = row.get("source")
                if (
                    not isinstance(audio, dict)
                    or not isinstance(audio.get("bytes"), bytes)
                    or not audio["bytes"]
                    or audio.get("path") is not None
                    or not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                    or (
                        expected_source is not None
                        and source != expected_source
                        and not (expected_source == "nst" and source == "nst_da")
                    )
                ):
                    raise MaterialisedOverlayError(
                        f"Unsafe materialised Parquet row: {path}"
                    )
    except (OSError, pa.ArrowException) as error:
        raise MaterialisedOverlayError(
            f"Cannot read materialised Parquet file: {path}"
        ) from error
    if expected_rows is not None and rows != expected_rows:
        raise MaterialisedOverlayError(
            f"Materialised Parquet row count changed: {path}"
        )
    return rows


def _safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or _is_url(relative):
        raise MaterialisedOverlayError(f"Unsafe materialised path: {relative!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise MaterialisedOverlayError(
            f"Materialised path escapes its root: {relative!r}"
        )
    return resolved


def _is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https"}


def _write_rows(
    rows: Iterable[Mapping[str, object]],
    path: Path,
    source_name: str,
    audio_column: str,
    minimum_free_bytes: int,
    output_root: Path,
) -> int:
    writer = pq.ParquetWriter(path, _SCHEMA, compression="zstd")
    batch: list[dict[str, object]] = []
    count = 0
    try:
        for row in rows:
            audio = row.get(audio_column)
            if not isinstance(audio, Mapping):
                raise MaterialisedOverlayError(
                    "Materialised audio must be an encoded mapping"
                )
            audio_bytes = audio.get("bytes")
            audio_path = audio.get("path")
            if audio_path is not None and _is_url(str(audio_path)):
                raise MaterialisedOverlayError(
                    "HTTP audio paths cannot be materialised"
                )
            if not isinstance(audio_bytes, bytes) or not audio_bytes:
                raise MaterialisedOverlayError(
                    "Every materialised row requires non-empty embedded compressed "
                    "audio bytes"
                )
            text = row.get("text")
            source = row.get("source")
            if not isinstance(text, str) or not text.strip():
                raise MaterialisedOverlayError(
                    "Every materialised row requires final text"
                )
            if source != source_name and not (
                source_name == "nst" and source == "nst_da"
            ):
                raise MaterialisedOverlayError(
                    f"Unexpected source discriminator {source!r} for {source_name}"
                )
            batch.append(
                {
                    "audio": {"bytes": audio_bytes, "path": None},
                    "text": text,
                    "source": source,
                }
            )
            if len(batch) >= _WRITE_BATCH_SIZE:
                _require_free_space(path=output_root, reserve=minimum_free_bytes)
                writer.write_table(pa.Table.from_pylist(batch, schema=_SCHEMA))
                count += len(batch)
                batch.clear()
        if batch:
            _require_free_space(path=output_root, reserve=minimum_free_bytes)
            writer.write_table(pa.Table.from_pylist(batch, schema=_SCHEMA))
            count += len(batch)
    finally:
        writer.close()
    _require_free_space(path=output_root, reserve=minimum_free_bytes)
    return count


def _source_provenance(
    source_name: str,
    source_config: Mapping[str, object],
    base_files: list[str],
    overlay_files: list[str],
) -> dict[str, JsonValue]:
    overlay = t.cast(Mapping[str, object], source_config["overlay"])
    base_revision = validate_immutable_source_revision(
        str(source_config.get("revision") or ""),
        revision_label=f"{source_name} base revision",
    )
    overlay_revision = validate_overlay_revision(str(overlay.get("revision") or ""))
    configuration = _normalise(source_config)
    return {
        "name": source_name,
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "base": {
            "id": str(source_config["id"]),
            "subset": _string_or_none(source_config.get("subset")),
            "split": str(source_config.get("train_name", "train")),
            "revision": base_revision,
            "files": base_files,
        },
        "overlay": {
            "id": str(overlay["id"]),
            "subset": _string_or_none(overlay.get("subset")),
            "split": str(overlay.get("split", "train")),
            "revision": overlay_revision,
            "files": overlay_files,
        },
    }


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _validate_output_contract(
    source_name: str, source_config: Mapping[str, object]
) -> None:
    overlay = t.cast(Mapping[str, object], source_config["overlay"])
    if str(source_config.get("audio_column", "audio")) != "audio":
        raise MaterialisedOverlayError(
            f"{source_name} must use the audio output column"
        )
    if str(overlay.get("output_text_column", "text")) != "text":
        raise MaterialisedOverlayError(
            f"{source_name} overlay must output final text to text"
        )
    equality = overlay.get("equality_checks")
    if not isinstance(equality, Mapping) or "source" not in equality:
        raise MaterialisedOverlayError(f"{source_name} must strictly check its source")


def _positional_sources(config: DictConfig) -> list[tuple[str, Mapping[str, object]]]:
    sources: list[tuple[str, Mapping[str, object]]] = []
    for name, raw_config in config.datasets.items():
        source_config = t.cast(Mapping[str, object], raw_config)
        overlay = source_config.get("overlay")
        if (
            isinstance(overlay, Mapping)
            and _effective_overlay_strategy(overlay) == "positional"
        ):
            sources.append((str(name), source_config))
    return sources


def _validate_manifest(
    config: DictConfig, root: Path, manifest: Mapping[str, object]
) -> None:
    if set(manifest) != {"format", "sources"} or manifest.get("format") != _FORMAT:
        raise MaterialisedOverlayError(
            "Materialised overlay manifest format is invalid"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not all(
        isinstance(item, Mapping) for item in sources
    ):
        raise MaterialisedOverlayError("Materialised overlay sources must be a list")
    configured = dict(_positional_sources(config=config))
    by_name = {str(source.get("name")): source for source in sources}
    if len(by_name) != len(sources) or set(by_name) != set(configured):
        raise MaterialisedOverlayError(
            "Materialised overlay source set does not match config"
        )

    for source_name, source_config in configured.items():
        source = t.cast(Mapping[str, object], by_name[source_name])
        _validate_output_contract(source_name=source_name, source_config=source_config)
        base_files = _manifest_file_list(source=source, key="base_file")
        overlay_files = _manifest_file_list(source=source, key="overlay_file")
        _validate_positional_shard_parity(
            base_files=base_files, overlay_files=overlay_files
        )
        _validate_manifest_files(
            files=base_files,
            selection=t.cast(
                Mapping[str, object], source_config.get("data_file_shards")
            ),
            label=f"{source_name} base",
        )
        overlay_config = t.cast(Mapping[str, object], source_config.get("overlay"))
        _validate_manifest_files(
            files=overlay_files,
            selection=t.cast(
                Mapping[str, object], overlay_config.get("data_file_shards")
            ),
            label=f"{source_name} overlay",
        )
        expected = _source_provenance(
            source_name=source_name,
            source_config=source_config,
            base_files=base_files,
            overlay_files=overlay_files,
        )
        if any(source.get(key) != value for key, value in expected.items()):
            raise MaterialisedOverlayError(
                f"Materialised overlay provenance mismatch for {source_name}"
            )
        shards = t.cast(list[Mapping[str, object]], source.get("shards"))
        if not shards:
            raise MaterialisedOverlayError(f"No materialised shards for {source_name}")
        provenance_sha256 = _canonical_sha256(expected)
        for base_file, overlay_file, shard in zip(
            base_files, overlay_files, shards, strict=True
        ):
            expected_path = f"{source_name}/{Path(base_file).name}"
            expected_receipt = {
                "format": _FORMAT,
                "source": source_name,
                "path": expected_path,
                "base_file": base_file,
                "overlay_file": overlay_file,
                "provenance_sha256": provenance_sha256,
            }
            if any(shard.get(key) != value for key, value in expected_receipt.items()):
                raise MaterialisedOverlayError(
                    f"Materialised shard provenance mismatch for {source_name}"
                )
            if set(shard) != {*expected_receipt, "rows", "sha256"}:
                raise MaterialisedOverlayError(
                    f"Materialised shard receipt fields are invalid for {source_name}"
                )
            path = _safe_child(root=root, relative=str(shard.get("path", "")))
            receipt_path = path.with_suffix(path.suffix + ".receipt.json")
            receipt = _read_json_mapping(path=receipt_path, label="shard receipt")
            if dict(receipt) != dict(shard):
                raise MaterialisedOverlayError(f"Manifest/receipt mismatch for {path}")
            rows = _validate_parquet(path=path, expected_source=source_name)
            if shard.get("rows") != rows or shard.get("sha256") != _sha256_file(path):
                raise MaterialisedOverlayError(f"Materialised shard is corrupt: {path}")


def _manifest_file_list(source: Mapping[str, object], key: str) -> list[str]:
    shards = source.get("shards")
    if not isinstance(shards, list) or not all(
        isinstance(item, Mapping) for item in shards
    ):
        raise MaterialisedOverlayError("Manifest shards must be a non-empty list")
    files = [str(item.get(key, "")) for item in shards]
    if not files or any(
        not path or _is_url(path) or Path(path).is_absolute() for path in files
    ):
        raise MaterialisedOverlayError(
            "Manifest Hub files must be relative repository paths"
        )
    provenance_key = "base" if key == "base_file" else "overlay"
    provenance = source.get(provenance_key)
    if not isinstance(provenance, Mapping) or provenance.get("files") != files:
        raise MaterialisedOverlayError("Manifest shard files disagree with provenance")
    return files


def _validate_manifest_files(
    files: list[str], selection: Mapping[str, object], label: str
) -> None:
    if not isinstance(selection, Mapping):
        raise MaterialisedOverlayError(f"{label} shard selection is missing")
    try:
        template = str(selection["template"])
        start = int(t.cast(int, selection["start"]))
        end = int(t.cast(int, selection["end"]))
        candidates = [template.format(shard=index) for index in range(start, end + 1)]
    except (KeyError, TypeError, ValueError) as error:
        raise MaterialisedOverlayError(f"{label} shard selection is invalid") from error
    positions = {candidate: index for index, candidate in enumerate(candidates)}
    if (
        len(set(files)) != len(files)
        or any(path not in positions for path in files)
        or [positions[path] for path in files]
        != sorted(positions[path] for path in files)
    ):
        raise MaterialisedOverlayError(
            f"{label} manifest files do not match the configured shard selection"
        )


def _write_complete_marker(root: Path, manifest_path: Path) -> None:
    _atomic_json_write(
        path=root / _COMPLETE,
        value={"format": _FORMAT, "manifest_sha256": _sha256_file(manifest_path)},
    )


def validate_materialised_overlay_root(
    config: DictConfig, root: Path
) -> dict[str, JsonValue]:
    """Validate a complete local overlay artefact before any data is loaded.

    Args:
        config:
            Resolved finetuning configuration.
        root:
            Materialised artefact root.

    Returns:
        The validated manifest.

    Raises:
        MaterialisedOverlayError:
            If the marker, manifest, provenance, files, checksums, counts, or schemas
            do not match.
    """
    root = root.expanduser().resolve()
    manifest_path = root / _MANIFEST
    complete_path = root / _COMPLETE
    if not root.is_dir():
        raise MaterialisedOverlayError(
            f"Materialised overlay root does not exist: {root}"
        )
    manifest = _read_json_mapping(path=manifest_path, label="manifest")
    complete = _read_json_mapping(path=complete_path, label="complete marker")
    if (
        set(complete) != {"format", "manifest_sha256"}
        or complete.get("format") != _FORMAT
    ):
        raise MaterialisedOverlayError(
            "Materialised overlay complete marker is invalid"
        )
    if complete.get("manifest_sha256") != _sha256_file(manifest_path):
        raise MaterialisedOverlayError(
            "Materialised overlay manifest checksum mismatch"
        )
    _validate_manifest(config=config, root=root, manifest=manifest)
    return t.cast(dict[str, JsonValue], manifest)


def materialised_source_files(
    manifest: Mapping[str, object], root: Path
) -> dict[str, list[str]]:
    """Return manifest-listed local Parquet files by source.

    Args:
        manifest:
            A manifest returned by :func:`validate_materialised_overlay_root`.
        root:
            Materialised artefact root.

    Returns:
        Ordered absolute Parquet paths keyed by source name.
    """
    result: dict[str, list[str]] = {}
    for source in t.cast(list[Mapping[str, object]], manifest["sources"]):
        result[str(source["name"])] = [
            str(_safe_child(root=root, relative=str(shard["path"])))
            for shard in t.cast(list[Mapping[str, object]], source["shards"])
        ]
    return result
