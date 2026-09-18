"""Tests for the publication-only model command."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from hydra import compose
from omegaconf import OmegaConf

import hviske.finetune as finetune_module
from scripts import publish_model


def test_cohere_publication_preset_has_matching_package_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preserved Cohere baseline has an explicit publication provenance path."""
    monkeypatch.setenv("P1_SEGMENTS_REVISION", "1" * 40)

    config = compose(config_name="cohere_publication")

    assert config.model.pretrained_model_id == "CohereLabs/cohere-transcribe-03-2026"
    assert publish_model._expected_model_type(config=config) == "cohere_asr"


def test_publish_command_does_not_invoke_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Publishing a reviewed package never starts the finetuning workflow."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "nvidia/parakeet-tdt-0.6b-v3",
                "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
            },
            "model_card_languages": ["da", "en"],
            "training_dataset_ids": ["syvai/p1-segments"],
        }
    )
    published: dict[str, object] = {}

    monkeypatch.setattr(publish_model, "_load_config", lambda config_name: config)
    monkeypatch.setattr(
        publish_model,
        "training_sources_from_config",
        lambda config: [{"id": "syvai/p1-segments"}],
    )
    monkeypatch.setattr(
        publish_model, "publish_model_folder", lambda **kwargs: published.update(kwargs)
    )
    monkeypatch.setattr(
        finetune_module,
        "finetune",
        lambda config: pytest.fail("publication must not invoke training"),
    )

    result = CliRunner().invoke(
        publish_model.main,
        [
            str(model_dir),
            "syvai/hviske-v6.0",
            "--private",
            "--evaluation-status",
            "Reviewed.",
        ],
    )

    assert result.exit_code == 0, result.output
    assert published["folder_path"] == model_dir
    assert published["repo_id"] == "syvai/hviske-v6.0"
    assert published["private"] is True
    assert published["evaluation_status"] == "Reviewed."
    assert published["finetuned_from"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert published["expected_model_type"] == "parakeet_tdt"
