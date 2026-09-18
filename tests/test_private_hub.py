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
    ParakeetFeatureExtractor,
    ParakeetForTDT,
    ParakeetProcessor,
    ParakeetTDTConfig,
    ParakeetTokenizer,
    TokenizersBackend,
)
from transformers.trainer import Trainer

import hviske.model_publication as publication


class FakeRepositoryNotFoundError(Exception):
    """Hub repository-not-found error for the mocked API."""


def test_generated_tdt_model_card_is_family_neutral(tmp_path: Path) -> None:
    """Generated TDT provenance never labels the package as Cohere."""
    destination = tmp_path / "README.md"
    publication._stage_model_card(
        destination=destination,
        finetuned_from="nvidia/parakeet-tdt-0.6b-v3",
        finetuned_from_revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        model_card_languages=["da", "en"],
        training_dataset_ids=["org/dataset"],
        evaluation_status="Reviewed.",
    )

    card = destination.read_text(encoding="utf-8")
    assert "This private ASR checkpoint" in card
    assert "Cohere checkpoint" not in card
    assert "base_model: nvidia/parakeet-tdt-0.6b-v3" in card


def test_model_card_requires_pinned_base_metadata(tmp_path: Path) -> None:
    """Generated cards cannot omit the exact base model revision."""
    with pytest.raises(ValueError, match="pinned base model revision"):
        publication._stage_model_card(
            destination=tmp_path / "README.md",
            finetuned_from="org/base-model",
            model_card_languages=["da", "en"],
            training_dataset_ids=["org/dataset"],
            evaluation_status="Not evaluated.",
            finetuned_from_revision=None,
        )


def test_private_only_creates_missing_repository_as_private(
    monkeypatch: MonkeyPatch,
) -> None:
    """A missing destination is created privately and checked immediately."""
    api = FakeHubApi(private=None)
    monkeypatch.setattr(publication, "HfApi", lambda **_: api)
    monkeypatch.setattr(
        publication, "RepositoryNotFoundError", FakeRepositoryNotFoundError
    )

    publication.ensure_private_hub_repository(repo_id="syvai/hviske-v6", token="token")

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
    monkeypatch.setattr(publication, "HfApi", lambda **_: api)

    with pytest.raises(PermissionError, match="public repository"):
        publication.ensure_private_hub_repository(
            repo_id="syvai/hviske-v6", token="token"
        )

    assert api.create_calls == []


def test_private_only_refuses_private_false() -> None:
    """A private-only run cannot be configured as public."""
    with pytest.raises(ValueError, match="private=true"):
        publication.validate_private_only_config(
            {"private_only": True, "private": False}
        )


def test_publication_accepts_native_cohere_save_and_reload(tmp_path: Path) -> None:
    """A native Transformers save passes validation and local reload."""
    _minimal_cohere_package(tmp_path)
    assert publication._validate_model_package(tmp_path) == "cohere_asr"

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


def test_publication_accepts_native_parakeet_tdt_save(tmp_path: Path) -> None:
    """A saved TDT package follows the pinned Transformers artefact contract."""
    _minimal_tdt_package(tmp_path)

    assert publication._validate_model_package(tmp_path) == "parakeet_tdt"


def _minimal_tdt_package(folder: Path) -> None:
    """Save a tiny native Transformers Parakeet TDT package."""
    tokenizer = ParakeetTokenizer(
        vocab={"<pad>": 0, "a": 1, "<unk>": 2, "<s>": 3, "</s>": 4, "<blank>": 5},
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        blank_token="<blank>",
    )
    processor = ParakeetProcessor(
        feature_extractor=ParakeetFeatureExtractor(feature_size=1),
        tokenizer=tokenizer,
        blank_token="<blank>",
        decoder_type="tdt",
    )
    model = ParakeetForTDT(
        ParakeetTDTConfig(
            encoder_config={
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": 8,
            },
            vocab_size=6,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=0,
            blank_token_id=5,
            durations=(0, 1, 2),
        )
    )
    processor.save_pretrained(folder)
    model.save_pretrained(folder)


def test_publication_rejects_corrupt_weights(tmp_path: Path) -> None:
    """Corrupt single-file weights cannot pass the publication gate."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors").write_bytes(b"not safetensors")
    with pytest.raises(ValueError, match="Invalid safetensors"):
        publication._validate_model_package(tmp_path)


def test_publication_rejects_empty_or_card_only_package(tmp_path: Path) -> None:
    """A README or empty output cannot be uploaded as a model."""
    (tmp_path / "README.md").write_text("card", encoding="utf-8")
    with pytest.raises(ValueError, match="Cohere package is missing"):
        publication._validate_model_package(tmp_path)


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
        publication._validate_model_package(tmp_path)


def test_publication_rejects_malformed_sharded_index(tmp_path: Path) -> None:
    """A broken sharded index cannot masquerade as a complete weight set."""
    _minimal_cohere_package(tmp_path, weights=False)
    (tmp_path / "model.safetensors.index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed sharded"):
        publication._validate_model_package(tmp_path)


def test_publication_rejects_malformed_tdt_processor(tmp_path: Path) -> None:
    """A TDT package with a non-TDT processor cannot pass validation."""
    _minimal_tdt_package(tmp_path)
    processor_config = tmp_path / "processor_config.json"
    processor_config.write_text(
        processor_config.read_text(encoding="utf-8").replace(
            '"decoder_type": "tdt"', '"decoder_type": "rnnt"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong decoder_type"):
        publication._validate_model_package(tmp_path)


def test_publication_rejects_missing_processor(tmp_path: Path) -> None:
    """A weight file without Cohere processor essentials is incomplete."""
    _minimal_cohere_package(tmp_path, processor=False)
    with pytest.raises(ValueError, match="processor_config"):
        publication._validate_model_package(tmp_path)


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
        publication._validate_model_package(tmp_path)


def test_publication_rejects_missing_weights(tmp_path: Path) -> None:
    """Processor metadata alone is not a reloadable model."""
    _minimal_cohere_package(tmp_path, weights=False)
    with pytest.raises(ValueError, match="needs model"):
        publication._validate_model_package(tmp_path)


def test_publication_rejects_package_family_provenance_mismatch(tmp_path: Path) -> None:
    """A valid package cannot be published with another model family's metadata."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _minimal_cohere_package(source)

    with pytest.raises(ValueError, match="does not match publication provenance"):
        publication._copy_model_artefacts(
            source=source, destination=destination, expected_model_type="parakeet_tdt"
        )


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
        publication._validate_model_package(tmp_path)


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
        publication._validate_model_package(tmp_path)


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
        publication._validate_model_package(tmp_path)


def test_publication_rejects_wrong_model_type(tmp_path: Path) -> None:
    """A package for a different model family cannot be published as Cohere."""
    _minimal_cohere_package(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
    with pytest.raises(ValueError, match="wrong model_type"):
        publication._validate_model_package(tmp_path)


def test_publication_rejects_wrong_parakeet_family(tmp_path: Path) -> None:
    """A TDT package relabelled as another Parakeet family is rejected."""
    _minimal_tdt_package(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            '"model_type": "parakeet_tdt"', '"model_type": "parakeet_rnnt"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported model family"):
        publication._validate_model_package(tmp_path)


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
    monkeypatch.setattr(publication, "HfApi", lambda **_: api)

    def upload(**kwargs: object) -> SimpleNamespace:
        staging = Path(str(kwargs["folder_path"]))
        uploads.append({item.name for item in staging.iterdir()})
        return SimpleNamespace()

    monkeypatch.setattr(publication, "upload_folder", upload)

    publication.push_model_to_hub(
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
