"""Focused tests for safe local positional-overlay materialisation."""

import io
import json
import sys
import typing as t
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Audio, Dataset, Features, IterableDataset, Value, load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

import hviske.data as data_module
import hviske.materialised_overlays as materialised_module
from hviske.data import load_data_for_finetuning
from hviske.materialised_overlays import (
    MaterialisedOverlayError,
    materialise_finetuning_overlays,
    validate_materialised_overlay_root,
)

BASE_REVISION = "a" * 40
OVERLAY_REVISION = "b" * 40
SHARD = "data/train-00001.parquet"


def _first_worker_row(batch: list[dict[str, object]]) -> dict[str, object]:
    """Return one worker row without tensor conversion."""
    return batch[0]


def test_complete_marker_detects_manifest_mutation(tmp_path: Path) -> None:
    """The complete marker makes the deterministic manifest immutable."""
    materialise_finetuning_overlays(
        config=_config(),
        output_root=tmp_path,
        minimum_free_bytes=0,
        dataset_loader=_loader(audio_bytes=[b"encoded"]),
        hub_api=FakeHubApi(),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][0]["name"] = "changed"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(MaterialisedOverlayError, match="manifest checksum"):
        validate_materialised_overlay_root(config=_config(), root=tmp_path)


class FakeHubApi:
    """Resolve one mirrored physical shard without network access."""

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[str]:
        """Return the configured fake shard."""
        assert repo_type == "dataset"
        assert repo_id in {"example/base", "example/overlay"}
        assert revision in {BASE_REVISION, OVERLAY_REVISION}
        return [SHARD]


def _config() -> DictConfig:
    return OmegaConf.create(
        {
            "cache_dir": None,
            "datasets": {
                "demo": {
                    "id": "example/base",
                    "subset": "default",
                    "train_name": "train",
                    "revision": BASE_REVISION,
                    "audio_column": "audio",
                    "text_column": "text",
                    "filter_dataset": False,
                    "language": "da",
                    "shuffle_buffer_size": 1,
                    "filters": {"source": "demo"},
                    "data_file_shards": {
                        "template": "data/train-{shard:05d}.parquet",
                        "start": 1,
                        "end": 1,
                    },
                    "overlay": {
                        "id": "example/overlay",
                        "split": "train",
                        "revision": OVERLAY_REVISION,
                        "data_file_shards": {
                            "template": "data/train-{shard:05d}.parquet",
                            "start": 1,
                            "end": 1,
                        },
                        "strategy": "positional",
                        "base_filters": {"source": "demo"},
                        "filters": {"source": "demo"},
                        "equality_checks": {
                            "source": "source",
                            "text": "reference_text",
                        },
                        "action_column": "action",
                        "allowed_actions": ["keep", "relabel", "strip"],
                        "recognised_actions": ["keep", "relabel", "strip", "drop"],
                        "text_policy": {
                            "candidates": [
                                {"column": "new_text", "actions": ["relabel", "strip"]},
                                {"column": "reference_text"},
                            ]
                        },
                    },
                }
            },
        }
    )


def _loader(audio_bytes: list[bytes]) -> materialised_module.DatasetLoader:
    def loader(**kwargs: object) -> Dataset | IterableDataset:
        if kwargs["path"] == "example/base":
            return _base_dataset(audio_bytes=audio_bytes)
        return Dataset.from_list([_overlay_row(text="one", action="keep")])

    return loader


def _base_dataset(audio_bytes: list[bytes]) -> IterableDataset:
    rows = [
        {"audio": {"bytes": value, "path": None}, "text": text, "source": "demo"}
        for value, text in zip(audio_bytes, ["one", "two", "three"], strict=False)
    ]
    return IterableDataset.from_generator(
        lambda: iter(rows),
        features=Features(
            {
                "audio": Audio(decode=False),
                "text": Value("string"),
                "source": Value("string"),
            }
        ),
    )


def _overlay_row(
    *, text: str, action: str, new_text: str | None = None
) -> dict[str, str | None]:
    return {
        "source": "demo",
        "reference_text": text,
        "new_text": new_text,
        "action": action,
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Linux spawn-worker regression")
def test_four_spawn_workers_interleave_local_shards_restartably(tmp_path: Path) -> None:
    """Four spawned workers consume each local shard exactly once on every restart."""
    files: list[str] = []
    for index in range(8):
        path = tmp_path / f"part-{index:02d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "audio": {"bytes": f"audio-{index}".encode(), "path": None},
                        "text": f"text-{index}",
                        "source": "demo",
                    }
                ],
                schema=materialised_module._SCHEMA,
            ),
            path,
        )
        files.append(str(path))
    dataset = load_dataset("parquet", data_files=files, split="train", streaming=True)
    torch_dataset = t.cast(TorchIterableDataset[dict[str, object]], dataset)
    loader = DataLoader(
        torch_dataset,
        batch_size=1,
        num_workers=4,
        multiprocessing_context="spawn",
        collate_fn=_first_worker_row,
    )

    expected = [f"text-{index}" for index in range(8)]
    assert sorted(str(row["text"]) for row in loader) == expected
    assert sorted(str(row["text"]) for row in loader) == expected


def test_interrupted_atomic_write_is_restartable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed Parquet promotion leaves no accepted shard and can be resumed."""
    loader = _loader(audio_bytes=[b"encoded"])
    original_replace = materialised_module.os.replace
    interrupted = False

    def interrupt_once(source: Path, destination: Path) -> None:
        nonlocal interrupted
        if not interrupted and str(source).endswith(".parquet.partial"):
            interrupted = True
            raise OSError("simulated interruption")
        original_replace(source, destination)

    monkeypatch.setattr(materialised_module.os, "replace", interrupt_once)
    with pytest.raises(OSError, match="simulated interruption"):
        materialise_finetuning_overlays(
            config=_config(),
            output_root=tmp_path,
            minimum_free_bytes=0,
            dataset_loader=loader,
            hub_api=FakeHubApi(),
        )
    assert not (tmp_path / "demo" / "train-00001.parquet").exists()

    monkeypatch.setattr(materialised_module.os, "replace", original_replace)
    materialise_finetuning_overlays(
        config=_config(),
        output_root=tmp_path,
        minimum_free_bytes=0,
        dataset_loader=loader,
        hub_api=FakeHubApi(),
    )
    validate_materialised_overlay_root(config=_config(), root=tmp_path)


def test_local_loader_skips_remote_base_and_overlay_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validated local source never resolves or loads its remote join inputs."""
    config = _training_config(root=tmp_path)
    materialise_finetuning_overlays(
        config=config,
        output_root=tmp_path,
        minimum_free_bytes=0,
        dataset_loader=_loader(audio_bytes=[_wav_bytes()]),
        hub_api=FakeHubApi(),
    )

    def unexpected_call(**kwargs: object) -> t.NoReturn:
        raise AssertionError(f"unexpected remote call: {kwargs}")

    monkeypatch.setattr(data_module, "_resolve_hub_data_files", unexpected_call)
    monkeypatch.setattr(data_module, "_load_transcript_dataset", unexpected_call)
    dataset = load_data_for_finetuning(config=config)

    assert set(dataset) == {"train"}


def _training_config(root: Path) -> DictConfig:
    config = _config()
    config.materialised_overlay_root = str(root)
    config.require_materialised_overlays = True
    config.dataset_probabilities = None
    config.seed = 4242
    config.min_seconds_per_example = 0.1
    config.max_seconds_per_example = 2.0
    config.streaming = True
    config.dataset_num_workers = 1
    config.shuffle_buffer_size = 1
    config.evaluation_datasets = []
    config.evaluation_lower_case = True
    config.evaluation_characters_to_keep = None
    config.model = {
        "sampling_rate": 16_000,
        "lower_case": True,
        "characters_to_keep": None,
        "language": "da",
        "punctuation": True,
    }
    return config


def _wav_bytes() -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 16_000)
    return payload.getvalue()


def test_manifest_and_output_corruption_fail_closed(tmp_path: Path) -> None:
    """Config drift and modified Parquet bytes invalidate the complete artefact."""
    manifest = materialise_finetuning_overlays(
        config=_config(),
        output_root=tmp_path,
        minimum_free_bytes=0,
        dataset_loader=_loader(audio_bytes=[b"encoded"]),
        hub_api=FakeHubApi(),
    )
    changed = _config()
    changed.datasets.demo.filters.source = "other"
    with pytest.raises(MaterialisedOverlayError, match="provenance mismatch"):
        validate_materialised_overlay_root(config=changed, root=tmp_path)

    shard = tmp_path / _first_manifest_shard_path(manifest)
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(MaterialisedOverlayError, match="corrupt|Cannot read"):
        validate_materialised_overlay_root(config=_config(), root=tmp_path)


def _first_manifest_shard_path(
    manifest: dict[str, materialised_module.JsonValue],
) -> str:
    sources = t.cast(list[dict[str, object]], manifest["sources"])
    shards = t.cast(list[dict[str, object]], sources[0]["shards"])
    return str(shards[0]["path"])


def test_materialisation_preserves_bytes_and_overlay_semantics(tmp_path: Path) -> None:
    """Embedded bytes survive while drop and relabel actions match the live join."""
    audio_bytes = [b"encoded-one", b"encoded-two", b"encoded-three"]

    def loader(**kwargs: object) -> Dataset | IterableDataset:
        if kwargs["path"] == "example/base":
            return _base_dataset(audio_bytes=audio_bytes)
        return Dataset.from_list(
            [
                _overlay_row(text="one", action="keep"),
                _overlay_row(text="two", action="relabel", new_text="revised"),
                _overlay_row(text="three", action="drop"),
            ]
        )

    manifest = materialise_finetuning_overlays(
        config=_config(),
        output_root=tmp_path,
        minimum_free_bytes=0,
        dataset_loader=loader,
        hub_api=FakeHubApi(),
    )

    shard = tmp_path / _first_manifest_shard_path(manifest)
    rows = pq.read_table(shard).to_pylist()
    assert [row["audio"]["bytes"] for row in rows] == audio_bytes[:2]
    assert [row["audio"]["path"] for row in rows] == [None, None]
    assert [row["text"] for row in rows] == ["one", "revised"]
    assert {tuple(row) for row in rows} == {("audio", "text", "source")}
    validate_materialised_overlay_root(config=_config(), root=tmp_path)


@pytest.mark.parametrize(
    "audio",
    [
        {"bytes": None, "path": None},
        {"bytes": None, "path": "https://example.test/signed.wav?token=secret"},
    ],
)
def test_materialisation_rejects_missing_or_url_only_audio(
    tmp_path: Path, audio: dict[str, object]
) -> None:
    """A local artefact never depends on absent bytes or an HTTP path."""
    with pytest.raises(MaterialisedOverlayError, match="bytes|HTTP"):
        materialised_module._write_rows(
            rows=[{"audio": audio, "text": "text", "source": "demo"}],
            path=tmp_path / "bad.parquet.partial",
            source_name="demo",
            audio_column="audio",
            minimum_free_bytes=0,
            output_root=tmp_path,
        )
