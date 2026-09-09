"""Focused tests for private-only Hub publication."""

import typing as t
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.monkeypatch import MonkeyPatch
from transformers.trainer import Trainer

import hviske.utils as utils


class FakeRepositoryNotFoundError(Exception):
    """Hub repository-not-found error for the mocked API."""


def test_private_only_creates_missing_repository_as_private(
    monkeypatch: MonkeyPatch,
) -> None:
    """A missing destination is created privately and checked immediately."""
    api = FakeHubApi(private=None)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)
    monkeypatch.setattr(utils, "RepositoryNotFoundError", FakeRepositoryNotFoundError)

    utils.ensure_private_hub_repository(repo_id="syvai/hviske-v6", token="token")

    assert api.create_calls[0]["private"] is True
    assert api.info_calls == 2
    assert api.private is True


class FakeHubApi:
    """Stateful Hub API double for visibility checks."""

    def __init__(self, private: bool | None) -> None:
        """Initialise the fake repository visibility."""
        self.private = private
        self.create_calls: list[dict[str, object]] = []
        self.info_calls = 0

    def create_repo(self, **kwargs: object) -> None:
        """Record private repository creation."""
        self.create_calls.append(kwargs)
        self.private = True

    def repo_info(self, **_: object) -> SimpleNamespace:
        """Return repository metadata or the configured missing error.

        Raises:
            FakeRepositoryNotFoundError:
                If the fake repository is missing.
        """
        self.info_calls += 1
        if self.private is None:
            raise FakeRepositoryNotFoundError
        return SimpleNamespace(private=self.private)


def test_private_only_refuses_existing_public_repository(
    monkeypatch: MonkeyPatch,
) -> None:
    """An existing public destination is never changed or uploaded to."""
    api = FakeHubApi(private=False)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)

    with pytest.raises(PermissionError, match="public repository"):
        utils.ensure_private_hub_repository(repo_id="syvai/hviske-v6", token="token")

    assert api.create_calls == []


def test_private_only_refuses_private_false() -> None:
    """A private-only run cannot be configured as public."""
    with pytest.raises(ValueError, match="private=true"):
        utils.validate_private_only_config({"private_only": True, "private": False})


def test_publication_accepts_valid_cohere_package(tmp_path: Path) -> None:
    """A complete Cohere package passes the reloadability gate."""
    _minimal_cohere_package(tmp_path)
    utils._validate_model_package(tmp_path)


def _minimal_cohere_package(
    folder: Path, *, weights: bool = True, processor: bool = True
) -> None:
    """Create the smallest package accepted by the Cohere publication gate."""
    (folder / "config.json").write_text("{}", encoding="utf-8")
    if processor:
        for name in (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ):
            (folder / name).write_text("{}", encoding="utf-8")
    if weights:
        (folder / "model.safetensors").write_bytes(b"weights")


def test_publication_rejects_empty_or_card_only_package(tmp_path: Path) -> None:
    """A README or empty output cannot be uploaded as a model."""
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    with pytest.raises(ValueError, match="Cohere package is missing"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_malformed_sharded_index(tmp_path: Path) -> None:
    """A broken sharded index cannot masquerade as a complete weight set."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete sharded"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_missing_processor(tmp_path: Path) -> None:
    """A weight file without Cohere processor essentials is incomplete."""
    _minimal_cohere_package(tmp_path, processor=False)
    with pytest.raises(ValueError, match="preprocessor_config"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_missing_weights(tmp_path: Path) -> None:
    """Processor metadata alone is not a reloadable model."""
    _minimal_cohere_package(tmp_path, weights=False)
    with pytest.raises(ValueError, match="needs model"):
        utils._validate_model_package(tmp_path)


def test_publish_fails_if_visibility_changes_after_upload(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """The final visibility check rejects a repository made public mid-upload."""
    api = FakeHubApi(private=True)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)

    def upload(**_: object) -> SimpleNamespace:
        api.private = False
        return SimpleNamespace()

    monkeypatch.setattr(utils, "upload_folder", upload)
    _populate_model_output(tmp_path)
    with pytest.raises(PermissionError, match="public repository"):
        utils.publish_model_folder(
            folder_path=tmp_path,
            repo_id="syvai/hviske-v6",
            finetuned_from="org/base-model",
            private=True,
            model_card_languages=["da", "en"],
            training_dataset_ids=["org/dataset"],
            training_sources=[_source("org/dataset")],
        )


def _populate_model_output(folder: Path) -> None:
    """Create allowed and forbidden files in a trainer output directory."""
    for name in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "processor_config.json",
        "chat_template.jinja",
    ):
        (folder / name).write_text("model", encoding="utf-8")
    (folder / "config.json").write_text("{}", encoding="utf-8")
    for name in (
        "arbitrary.json",
        "manifest.jsonl",
        "recording.wav",
        "trainer_state.json",
        "metrics.csv",
        "run.log",
    ):
        (folder / name).write_text("forbidden", encoding="utf-8")
    (folder / "checkpoint-100").mkdir()
    (folder / "checkpoint-100" / "model.safetensors").write_text(
        "forbidden", encoding="utf-8"
    )
    (folder / "wandb").mkdir()
    (folder / "wandb" / "run.json").write_text("forbidden", encoding="utf-8")
    (folder / "nested").mkdir()
    (folder / "nested" / "config.json").write_text("forbidden", encoding="utf-8")
    (folder / "symlink.safetensors").symlink_to(folder / "model.safetensors")


def _source(dataset_id: str) -> dict[str, object]:
    """Return complete provenance metadata for publication tests."""
    return {
        "id": dataset_id,
        "source": dataset_id,
        "subset": "none",
        "split": "train",
        "revision": "sha256-test",
        "probability": 1.0,
        "language": "da",
    }


def test_publish_stages_exact_top_level_allowlist(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Only regular top-level model artefacts and the generated card are staged."""
    api = FakeHubApi(private=True)
    staged_contents: list[set[str]] = []
    staged_card: list[str] = []
    upload_calls = 0
    _populate_model_output(tmp_path)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)

    def upload(**kwargs: object) -> SimpleNamespace:
        nonlocal upload_calls
        upload_calls += 1
        staging = Path(str(kwargs["folder_path"]))
        staged_contents.append({item.name for item in staging.iterdir()})
        staged_card.append((staging / "README.md").read_text(encoding="utf-8"))
        return SimpleNamespace()

    monkeypatch.setattr(utils, "upload_folder", upload)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "token")

    utils.publish_model_folder(
        folder_path=tmp_path,
        repo_id="syvai/hviske-v6",
        finetuned_from="CohereLabs/cohere-transcribe-03-2026",
        private=True,
        model_card_languages=["da", "en"],
        training_dataset_ids=["CoRal-project/coral-v3"],
        training_sources=[_source("CoRal-project/coral-v3")],
        evaluation_status="Evaluation ran during training.",
    )

    assert upload_calls == 1
    assert staged_contents == [
        {
            "README.md",
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "processor_config.json",
            "chat_template.jinja",
        }
    ]
    assert "license: openrail" in staged_card[0]
    assert "CoRal-project/coral-v3" in staged_card[0]
    assert "Private internal Danish-English ASR checkpoint" in staged_card[0]
    assert str(tmp_path) not in staged_card[0]
    assert api.info_calls == 3


def test_push_stages_once_and_disables_trainer_push(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Trainer publication also uses the filtered staging directory once."""
    api = FakeHubApi(private=True)
    _populate_model_output(tmp_path)
    trainer = t.cast(
        Trainer,
        SimpleNamespace(
            hub_model_id="syvai/hviske-v6",
            args=SimpleNamespace(
                hub_model_id="syvai/hviske-v6",
                output_dir=str(tmp_path),
                push_to_hub=True,
            ),
            is_world_process_zero=lambda: True,
            _finish_current_push=lambda: None,
        ),
    )
    uploads: list[set[str]] = []
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)

    def upload(**kwargs: object) -> SimpleNamespace:
        staging = Path(str(kwargs["folder_path"]))
        uploads.append({item.name for item in staging.iterdir()})
        return SimpleNamespace()

    monkeypatch.setattr(utils, "upload_folder", upload)

    utils.push_model_to_hub(
        trainer=trainer,
        model_name="hviske-v6",
        finetuned_from="org/base-model",
        create_pr=False,
        private=True,
        private_only=True,
        training_dataset_ids=["org/private-dataset"],
    )

    assert len(uploads) == 1
    assert "README.md" in uploads[0]
    assert "recording.wav" not in uploads[0]
    assert trainer.args.push_to_hub is False
    assert api.info_calls == 3
