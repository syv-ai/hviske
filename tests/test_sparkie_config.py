"""Regression tests for the production Sparkie bilingual preset."""

from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from hydra import compose
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

TRAINING_NAMES = [
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
TRAINING_IDS = [
    "syvai/p1",
    "local_vtt",
    "local_vtt",
    "CoRal-project/coral-v3",
    "CoRal-project/coral-v3",
    "alexandrainst/ftspeech",
    "alexandrainst/nota",
    "alexandrainst/nst-da",
    "syvai/danish-asr-unified",
    "MLCommons/peoples_speech",
    "edinburghcstr/ami",
    "edinburghcstr/ami",
    "facebook/voxpopuli",
    "openslr/librispeech_asr",
    "openslr/librispeech_asr",
    "openslr/librispeech_asr",
]
TRAINING_PROBABILITIES = [
    0.08,
    0.10,
    0.07,
    0.04,
    0.10,
    0.05,
    0.05,
    0.05,
    0.06,
    0.16,
    0.04,
    0.03,
    0.09,
    0.01,
    0.025,
    0.045,
]


def test_p1_transcript_revision_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private transcript revision cannot silently follow a mutable branch."""
    monkeypatch.delenv("P1_TRANSCRIPT_REVISION", raising=False)
    monkeypatch.setenv("P1_AUDIO_JOIN_COLUMN", "audio_id")
    monkeypatch.setenv("P1_TRANSCRIPT_JOIN_COLUMN", "audio_id")
    monkeypatch.setenv("P1_TRANSCRIPT_TEXT_COLUMN", "text")
    config = compose(config_name="sparkie_bilingual")

    with pytest.raises(InterpolationResolutionError, match="P1_TRANSCRIPT_REVISION"):
        OmegaConf.resolve(config)


def test_sparkie_dataset_coordinates_and_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every Hub source uses the approved subset, split, columns, and revision."""
    config = _preset(monkeypatch)
    datasets = config.datasets

    expected_coordinates = {
        "p1": ("syvai/p1", None, "train", "text", "audio"),
        "coral_read_aloud": (
            "CoRal-project/coral-v3",
            "read_aloud",
            "train",
            "text",
            "audio",
        ),
        "coral_conversation": (
            "CoRal-project/coral-v3",
            "conversation",
            "train",
            "text",
            "audio",
        ),
        "ftspeech": ("alexandrainst/ftspeech", None, "train", "sentence", "audio"),
        "nota": ("alexandrainst/nota", None, "train", "text", "audio"),
        "nst": ("alexandrainst/nst-da", None, "train", "text", "audio"),
        "voxpopuli_da": (
            "syvai/danish-asr-unified",
            "default",
            "train",
            "text",
            "audio",
        ),
        "peoples_speech_clean": (
            "MLCommons/peoples_speech",
            "clean",
            "train",
            "text",
            "audio",
        ),
        "ami_sdm": ("edinburghcstr/ami", "sdm", "train", "text", "audio"),
        "ami_ihm": ("edinburghcstr/ami", "ihm", "train", "text", "audio"),
        "voxpopuli_en": (
            "facebook/voxpopuli",
            "en",
            "train",
            "normalized_text",
            "audio",
        ),
        "librispeech_clean_train_100": (
            "openslr/librispeech_asr",
            "clean",
            "train.100",
            "text",
            "audio",
        ),
        "librispeech_clean_train_360": (
            "openslr/librispeech_asr",
            "clean",
            "train.360",
            "text",
            "audio",
        ),
        "librispeech_other_train_500": (
            "openslr/librispeech_asr",
            "other",
            "train.500",
            "text",
            "audio",
        ),
    }
    assert {
        name: (
            datasets[name].id,
            datasets[name].subset,
            datasets[name].train_name,
            datasets[name].text_column,
            datasets[name].audio_column,
        )
        for name in expected_coordinates
    } == expected_coordinates

    expected_revisions = {
        "p1": "449b9c2294026df6d0d37538f279fdec03f565ff",
        "coral_read_aloud": "01f7c93c21fc9dec87fe9f7149c79569cc433f08",
        "coral_conversation": "01f7c93c21fc9dec87fe9f7149c79569cc433f08",
        "ftspeech": "e1b7096db63c7d996a8220b13780a547a88a9af0",
        "nota": "acd1ad2389426f84fc3cb812c52d3d5dde5c7ce3",
        "nst": "0f14ad2005e0aab8f56cf3213b7689da1faf23c2",
        "voxpopuli_da": "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f",
        "peoples_speech_clean": "f10597c5d3d3a63f8b6827701297c3afdf178272",
        "ami_sdm": "46f28f2503e2ec48f8867a84eef356c70476beab",
        "ami_ihm": "46f28f2503e2ec48f8867a84eef356c70476beab",
        "voxpopuli_en": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
        "librispeech_clean_train_100": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        "librispeech_clean_train_360": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        "librispeech_other_train_500": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    }
    assert {
        name: datasets[name].revision for name in expected_revisions
    } == expected_revisions
    assert (
        datasets.p1.transcript_dataset_id,
        datasets.p1.transcript_subset,
        datasets.p1.transcript_split,
        datasets.p1.audio_join_column,
        datasets.p1.transcript_join_column,
        datasets.p1.transcript_text_column,
    ) == ("syvai/p1-transcripts", None, "train", "audio_id", "audio_id", "text")
    assert datasets.p1.transcript_revision == "transcript-revision"
    assert datasets.p1.transcript_trust_remote_code is False
    assert all(
        dataset.get("trust_remote_code", False) is False
        for dataset in datasets.values()
    )
    assert dict(datasets.voxpopuli_da.filters) == {"source": "voxpopuli"}
    assert [
        name
        for name, dataset in datasets.items()
        if dataset.get("id") == "syvai/danish-asr-unified"
    ] == ["voxpopuli_da"]


def _preset(monkeypatch: MonkeyPatch) -> DictConfig:
    """Resolve the preset with placeholders for private p1 schema fields.

    Returns:
        The resolved Sparkie preset.
    """
    monkeypatch.setenv("P1_TRANSCRIPT_REVISION", "transcript-revision")
    monkeypatch.setenv("P1_AUDIO_JOIN_COLUMN", "audio_id")
    monkeypatch.setenv("P1_TRANSCRIPT_JOIN_COLUMN", "audio_id")
    monkeypatch.setenv("P1_TRANSCRIPT_TEXT_COLUMN", "text")
    return compose(config_name="sparkie_bilingual")


def test_sparkie_evaluation_and_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    """FLEURS remains evaluation-only and superseded English sources stay absent."""
    config = _preset(monkeypatch)
    evaluations = config.evaluation_datasets

    assert [
        (
            source.id,
            source.subset,
            source.val_name,
            source.text_column,
            source.audio_column,
        )
        for source in evaluations
    ] == [
        ("CoRal-project/coral-v3", "read_aloud", "val", "text", "audio"),
        ("CoRal-project/coral-v3", "conversation", "val", "text", "audio"),
        ("openslr/librispeech_asr", "clean", "validation", "text", "audio"),
        ("openslr/librispeech_asr", "other", "validation", "text", "audio"),
        ("google/fleurs", "en_us", "validation", "raw_transcription", "audio"),
    ]
    assert [source.language for source in evaluations] == ["da", "da", "en", "en", "en"]
    assert all(source.trust_remote_code is False for source in evaluations)
    assert evaluations[-1].revision == "70bb2e84b976b7e960aa89f1c648e09c59f894dd"

    serialised = OmegaConf.to_yaml(config).lower()
    for excluded in (
        "coral-v2",
        "coral_tts",
        "common_voice",
        "gigaspeech",
        "spgispeech",
        "fleurs_en_us",
    ):
        assert excluded not in serialised
    assert not Path("config/datasets/common_voice_19_en.yaml").exists()
    assert not Path("config/datasets/fleurs_en_us.yaml").exists()


def test_sparkie_private_publication_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training stays local and publication metadata names every Hub source."""
    config = _preset(monkeypatch)

    assert config.push_to_hub is False
    assert config.enable_experiment_tracking is False
    assert config.private is True
    assert config.private_only is True
    assert config.save_total_limit == 3
    assert list(config.model_card_languages) == ["da", "en"]
    assert list(config.training_dataset_ids) == [
        "syvai/p1",
        "syvai/p1-transcripts",
        "CoRal-project/coral-v3",
        "alexandrainst/ftspeech",
        "alexandrainst/nota",
        "alexandrainst/nst-da",
        "syvai/danish-asr-unified",
        "MLCommons/peoples_speech",
        "edinburghcstr/ami",
        "facebook/voxpopuli",
        "openslr/librispeech_asr",
    ]


def test_sparkie_training_order_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preset keeps source order aligned with the approved probabilities."""
    config = _preset(monkeypatch)

    assert list(config.datasets) == TRAINING_NAMES
    assert [
        dataset.get("id", dataset.get("type")) for dataset in config.datasets.values()
    ] == TRAINING_IDS
    assert list(config.dataset_probabilities) == TRAINING_PROBABILITIES
    assert sum(config.dataset_probabilities) == pytest.approx(1.0)
    assert sum(config.dataset_probabilities[:9]) == pytest.approx(0.6)
    assert sum(config.dataset_probabilities[9:]) == pytest.approx(0.4)


def test_youtube_local_manifest_is_danish() -> None:
    """The Sparkie YouTube manifest uses Danish VTT transcripts."""
    contents = Path("config/datasets/youtube_local.yaml").read_text()

    assert "language: da" in contents
    assert "language: en" not in contents
