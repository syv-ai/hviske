"""Focused tests for the Olmix calibration matrix and launcher."""

from pathlib import Path

import pytest
from hydra import compose
from omegaconf import OmegaConf

from src.scripts.run_olmix_benchmark import (
    ANCHORS,
    CHECKPOINTS,
    MODELS,
    _make_run_id,
    build_command,
)

SOURCE_ORDER = [
    "p1",
    "drtv_local",
    "youtube_local",
    "coral_read_aloud",
    "coral_conversation",
    "ftspeech",
    "nota",
    "nst",
    "voxpopuli_da",
    "peoples_speech_clean",
    "ami_sdm",
    "ami_ihm",
    "voxpopuli_en",
    "librispeech_clean_train_100",
    "librispeech_clean_train_360",
    "librispeech_other_train_500",
]


def test_hviske_v5_tiny_model_config_is_native_and_trainable() -> None:
    """Expose the trainable native Cohere checkpoint with Danish prompts."""
    config = compose(
        config_name="asr_finetuning",
        overrides=["model=hviske-v5-tiny", "enable_experiment_tracking=false"],
    )

    assert config.model.type == "cohere"
    assert config.model.pretrained_model_id == "syvai/hviske-v5-tiny"
    assert config.model.freeze_feature_encoder is False
    assert config.model.sampling_rate == 16_000
    assert config.model.language == "da"
    assert config.model.punctuation is True
    assert config.model.learning_rate == pytest.approx(1.0e-5)


def test_launcher_command_uses_non_uniform_schedule_and_safe_overrides() -> None:
    """Construct a credential-free command with the calibration schedule."""
    command = build_command(
        model="hviske-v5-tiny",
        anchor="olmix_read_speech_heavy",
        model_id="run-model",
        models_dir=Path("runs/test/models"),
        hydra_run_dir=Path("runs/test/hydra"),
    )

    assert command[:6] == [
        "uv",
        "run",
        "python",
        "src/scripts/finetune_asr_model.py",
        "--config-name",
        "sparkie_bilingual",
    ]
    assert "+anchors=olmix_read_speech_heavy" in command
    assert "+evaluation_steps=[250,500,1000,2000,3000]" in command
    assert "eval_steps=1" in command
    assert "max_steps=3000" in command
    assert "max_validation_samples_per_dataset=500" in command
    assert "push_to_hub=false" in command
    assert "enable_experiment_tracking=false" in command
    assert not any("token" in value.lower() for value in command)


def test_matrix_model_and_anchor_sets_are_explicit() -> None:
    """Keep the six serial matrix cells and requested checkpoints explicit."""
    assert MODELS == ("whisper-xxsmall", "hviske-v5-tiny")
    assert CHECKPOINTS == (250, 500, 1000, 2000, 3000)
    assert len(MODELS) * len(ANCHORS) == 6


@pytest.mark.parametrize("anchor", ANCHORS)
def test_olmix_anchor_composition_preserves_sources_and_language_totals(
    anchor: str,
) -> None:
    """Compose each anchor without changing dataset order or language mix."""
    config = compose(config_name="sparkie_bilingual", overrides=[f"+anchors={anchor}"])

    assert list(config.datasets) == SOURCE_ORDER
    assert len(config.dataset_probabilities) == len(SOURCE_ORDER)
    assert sum(config.dataset_probabilities) == pytest.approx(1.0)
    assert sum(config.dataset_probabilities[:9]) == pytest.approx(0.60)
    assert sum(config.dataset_probabilities[9:]) == pytest.approx(0.40)
    assert all(probability > 0 for probability in config.dataset_probabilities)


def test_olmix_anchors_are_distinct_and_only_override_probabilities() -> None:
    """Keep anchor files limited to a global probability override."""
    baseline = OmegaConf.load("config/anchors/olmix_baseline.yaml")
    read_heavy = OmegaConf.load("config/anchors/olmix_read_speech_heavy.yaml")
    spontaneous_heavy = OmegaConf.load("config/anchors/olmix_spontaneous_heavy.yaml")

    assert list(baseline) == ["dataset_probabilities"]
    assert list(read_heavy) == ["dataset_probabilities"]
    assert list(spontaneous_heavy) == ["dataset_probabilities"]
    assert baseline.dataset_probabilities != read_heavy.dataset_probabilities
    assert read_heavy.dataset_probabilities != spontaneous_heavy.dataset_probabilities


def test_run_ids_are_unique_for_same_job() -> None:
    """Avoid collisions when two jobs start within the same second."""
    first = _make_run_id(
        model="whisper-xxsmall", anchor="olmix_baseline", run_kind="matrix"
    )
    second = _make_run_id(
        model="whisper-xxsmall", anchor="olmix_baseline", run_kind="matrix"
    )

    assert first != second
