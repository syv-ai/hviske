"""Regression tests for the production Sparkie bilingual preset."""

from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from hydra import compose
from omegaconf import DictConfig

TRAINING_IDS = [
    "syvai/p1",
    "local_vtt",
    "local_vtt",
    "CoRal-project/coral-v3",
    "CoRal-project/coral-v3",
    "alexandrainst/ftspeech",
    "alexandrainst/nota",
    "alexandrainst/nst-da",
    "openslr/librispeech_asr",
    "openslr/librispeech_asr",
    "openslr/librispeech_asr",
    "fsicoli/common_voice_19_0",
]


def _preset(monkeypatch: MonkeyPatch) -> DictConfig:
    """Resolve the preset with safe placeholders for private p1 columns.

    Returns:
        The resolved Sparkie preset.
    """
    monkeypatch.setenv("P1_AUDIO_JOIN_COLUMN", "audio_id")
    monkeypatch.setenv("P1_TRANSCRIPT_JOIN_COLUMN", "audio_id")
    monkeypatch.setenv("P1_TRANSCRIPT_TEXT_COLUMN", "text")
    return compose(config_name="sparkie_bilingual")


def test_sparkie_training_order_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preset keeps source order aligned with its explicit probabilities."""
    config = _preset(monkeypatch)

    assert config.push_to_hub is False
    assert config.enable_experiment_tracking is False
    assert config.private is True
    assert config.private_only is True
    assert config.save_total_limit == 3
    assert list(config.model_card_languages) == ["da", "en"]
    assert list(config.datasets) == [
        "p1",
        "drtv_local",
        "youtube_local",
        "coral_read_aloud",
        "coral_conversation",
        "ftspeech",
        "nota",
        "nst",
        "librispeech_clean_train_100",
        "librispeech_clean_train_360",
        "librispeech_other_train_500",
        "common_voice_19_en",
    ]
    assert [
        dataset.get("id", dataset.get("type")) for dataset in config.datasets.values()
    ] == TRAINING_IDS
    assert sum(config.dataset_probabilities) == 1
    assert sum(config.dataset_probabilities[:8]) == 0.6
    assert sum(config.dataset_probabilities[8:]) == 0.4


def test_sparkie_languages_and_coral_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every bilingual source and validation split carries its prompt language."""
    config = _preset(monkeypatch)

    assert [dataset.language for dataset in config.datasets.values()][8:] == [
        "en",
        "en",
        "en",
        "en",
    ]
    assert [dataset.language for dataset in config.evaluation_datasets] == [
        "da",
        "da",
        "en",
        "en",
        "en",
    ]
    assert [dataset.subset for dataset in config.evaluation_datasets[:2]] == [
        "read_aloud",
        "conversation",
    ]
    assert all(
        dataset.id == "CoRal-project/coral-v3"
        for dataset in config.evaluation_datasets[:2]
    )
    assert "coral_tts" not in str(config)
    assert all("coral-v2" not in str(dataset) for dataset in config.datasets.values())


def test_youtube_local_manifest_is_danish() -> None:
    """The Sparkie YouTube manifest uses Danish VTT transcripts."""
    contents = Path("config/datasets/youtube_local.yaml").read_text()

    assert "language: da" in contents
    assert "language: en" not in contents
