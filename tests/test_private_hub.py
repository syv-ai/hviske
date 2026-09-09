"""Focused tests for private-only Hub publication."""

import typing as t
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    CohereAsrConfig,
    CohereAsrFeatureExtractor,
    CohereAsrForConditionalGeneration,
    CohereAsrProcessor,
    TokenizersBackend,
)
from transformers.trainer import Trainer

import hviske.utils as utils


class FakeRepositoryNotFoundError(Exception):
    """Hub repository-not-found error for the mocked API."""


def test_model_card_requires_pinned_base_metadata(tmp_path: Path) -> None:
    """Generated cards cannot omit the exact base model revision."""
    with pytest.raises(ValueError, match="pinned base model revision"):
        utils._stage_model_card(
            destination=tmp_path / "README.md",
            finetuned_from="org/base-model",
            model_card_languages=["da", "en"],
            training_dataset_ids=["org/dataset"],
            training_sources=[_source("org/dataset")],
            evaluation_status="Not evaluated.",
            reviewed_model_card=None,
            finetuned_from_revision=None,
        )


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


def test_publication_accepts_native_cohere_save_and_reload(tmp_path: Path) -> None:
    """A native Transformers save passes validation and local reload."""
    _minimal_cohere_package(tmp_path)
    utils._validate_model_package(tmp_path)

    processor = CohereAsrProcessor.from_pretrained(tmp_path, local_files_only=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        tmp_path, local_files_only=True
    )
    assert isinstance(processor, CohereAsrProcessor)
    assert isinstance(model, CohereAsrForConditionalGeneration)
    assert processor.tokenizer.__class__.__name__ == "TokenizersBackend"
    assert not (tmp_path / "preprocessor_config.json").exists()


def _minimal_cohere_package(
    folder: Path, *, weights: bool = True, processor: bool = True
) -> None:
    """Save a tiny native Transformers Cohere package for publication tests."""
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "<bos>": 3,
        "<|da|>": 4,
        "hello": 5,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    config_factory = t.cast(t.Callable[..., CohereAsrConfig], CohereAsrConfig)
    config = config_factory(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=16,
        max_position_embeddings=16,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=3,
        encoder_config={
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 16,
        },
    )
    model = CohereAsrForConditionalGeneration(config)
    if weights:
        model.save_pretrained(folder, safe_serialization=True)
    else:
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(config.to_json_string(), encoding="utf-8")
    if processor:
        cohere_processor = CohereAsrProcessor(CohereAsrFeatureExtractor(), tokenizer)
        cohere_processor.save_pretrained(folder)


def test_publication_rejects_corrupt_weights(tmp_path: Path) -> None:
    """Corrupt single-file weights cannot pass the publication gate."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors").write_bytes(b"not safetensors")
    with pytest.raises(ValueError, match="Invalid safetensors"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_empty_or_card_only_package(tmp_path: Path) -> None:
    """A README or empty output cannot be uploaded as a model."""
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    with pytest.raises(ValueError, match="Cohere package is missing"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_expected_tensor_duplicated_into_another_shard(
    tmp_path: Path,
) -> None:
    """A tensor assigned to one shard cannot also appear in another shard."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"encoder.weight": "model-00001-of-00002.safetensors", '
        '"decoder.weight": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    save_file(
        {
            "encoder.weight": np.ones((1, 1), dtype=np.float32),
            "decoder.weight": np.ones((1, 1), dtype=np.float32),
        },
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"decoder.weight": np.ones((1, 1), dtype=np.float32)},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    with pytest.raises(ValueError, match="tensor set does not match"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_malformed_sharded_index(tmp_path: Path) -> None:
    """A broken sharded index cannot masquerade as a complete weight set."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed sharded"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_missing_processor(tmp_path: Path) -> None:
    """A weight file without Cohere processor essentials is incomplete."""
    _minimal_cohere_package(tmp_path, processor=False)
    with pytest.raises(ValueError, match="processor_config"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_missing_shard(tmp_path: Path) -> None:
    """Every shard named by an index must exist and be valid."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"encoder.weight": "model-00001-of-00002.safetensors", '
        '"decoder.weight": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    save_file(
        {"encoder.weight": np.ones((1, 1), dtype=np.float32)},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    with pytest.raises(ValueError, match="Incomplete sharded"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_missing_weights(tmp_path: Path) -> None:
    """Processor metadata alone is not a reloadable model."""
    _minimal_cohere_package(tmp_path, weights=False)
    with pytest.raises(ValueError, match="needs model"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_per_shard_tensor_set_mismatch(tmp_path: Path) -> None:
    """Each shard must contain exactly its assigned tensor set."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"encoder.weight": "model-00001-of-00002.safetensors", '
        '"decoder.weight": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    save_file(
        {"decoder.weight": np.ones((1, 1), dtype=np.float32)},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"encoder.weight": np.ones((1, 1), dtype=np.float32)},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    with pytest.raises(ValueError, match="shard without that tensor"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_tensor_in_wrong_shard(tmp_path: Path) -> None:
    """An index entry must point to the shard that stores its tensor."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"encoder.weight": "model-00001-of-00002.safetensors", '
        '"decoder.weight": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        save_file(
            {"decoder.weight": np.ones((1, 1), dtype=np.float32)}, tmp_path / name
        )
    with pytest.raises(ValueError, match="shard without that tensor"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_unindexed_shard_tensor(tmp_path: Path) -> None:
    """A shard cannot contain tensors omitted from the weight map."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"encoder.weight": "model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    save_file(
        {
            "encoder.weight": np.ones((1, 1), dtype=np.float32),
            "extra.weight": np.ones((1, 1), dtype=np.float32),
        },
        tmp_path / "model-00001-of-00001.safetensors",
    )
    with pytest.raises(ValueError, match="unindexed tensors"):
        utils._validate_model_package(tmp_path)


def test_publication_rejects_wrong_model_type(tmp_path: Path) -> None:
    """A package for a different model family cannot be published as Cohere."""
    _minimal_cohere_package(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
    with pytest.raises(ValueError, match="wrong model_type"):
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
            finetuned_from_revision="base-revision",
            private=True,
            model_card_languages=["da", "en"],
            training_dataset_ids=["org/dataset"],
            training_sources=[_source("org/dataset")],
        )


def _populate_model_output(folder: Path) -> None:
    """Create allowed and forbidden files in a trainer output directory."""
    _minimal_cohere_package(folder)
    for name in ("vocab.json", "merges.txt", "chat_template.jinja"):
        (folder / name).write_text("model", encoding="utf-8")
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
        finetuned_from_revision="b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
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
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
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
        finetuned_from_revision="base-revision",
    )

    assert len(uploads) == 1
    assert "README.md" in uploads[0]
    assert "recording.wav" not in uploads[0]
    assert trainer.args.push_to_hub is False
    assert api.info_calls == 3


def test_reviewed_model_card_requires_exact_base_metadata(tmp_path: Path) -> None:
    """Reviewed cards must carry matching base model and revision frontmatter."""
    for name, metadata in (
        ("missing", "base_model: org/base-model\n"),
        ("wrong", "base_model: org/other-model\nbase_model_revision: base-revision\n"),
        ("correct", "base_model: org/base-model\nbase_model_revision: base-revision\n"),
    ):
        card = tmp_path / f"{name}.md"
        card.write_text(
            f"---\nlicense: openrail\n{metadata}---\n\n"
            "# Private internal checkpoint\n\n"
            "org/dataset none train sha256-test 1.0 da\n",
            encoding="utf-8",
        )
        destination = tmp_path / f"staged-{name}.md"
        if name == "correct":
            utils._stage_model_card(
                destination=destination,
                finetuned_from="org/base-model",
                model_card_languages=["da", "en"],
                training_dataset_ids=["org/dataset"],
                training_sources=[_source("org/dataset")],
                evaluation_status="Not evaluated.",
                reviewed_model_card=card,
                finetuned_from_revision="base-revision",
            )
            assert destination.read_text(encoding="utf-8") == card.read_text(
                encoding="utf-8"
            )
        else:
            with pytest.raises(ValueError, match="safe complete provenance"):
                utils._stage_model_card(
                    destination=destination,
                    finetuned_from="org/base-model",
                    model_card_languages=["da", "en"],
                    training_dataset_ids=["org/dataset"],
                    training_sources=[_source("org/dataset")],
                    evaluation_status="Not evaluated.",
                    reviewed_model_card=card,
                    finetuned_from_revision="base-revision",
                )
