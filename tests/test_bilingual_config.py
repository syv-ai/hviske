"""Regression tests for the production bilingual preset."""

import typing as t
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from hydra import compose
from omegaconf import DictConfig, OmegaConf

import hviske.experiment_tracking.wandb_setup as wandb_module

P1_SEGMENTS_SHA = "44284e5849b6b1d96b874891c579654a644e0e2f"


TRAINING_NAMES = [
    "p1",
    "drtv_local",
    "youtube_local",
    "coral_read_aloud",
    "coral_conversation",
    "ftspeech",
    "nota",
    "nst",
    "peoples_speech_clean",
    "ami_sdm",
    "ami_ihm",
    "voxpopuli_en",
    "librispeech_clean_train_100",
    "librispeech_clean_train_360",
    "librispeech_other_train_500",
]
TRAINING_IDS = [
    "syvai/p1-segments",
    "local_vtt",
    "local_vtt",
    "syvai/danish-asr-unified",
    "syvai/danish-asr-unified",
    "syvai/danish-asr-unified",
    "syvai/danish-asr-unified",
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
    0.102857,
    0.128571,
    0.09,
    0.017143,
    0.128572,
    0.09,
    0.021428,
    0.021429,
    0.16,
    0.04,
    0.03,
    0.09,
    0.01,
    0.025,
    0.045,
]
V2_TRAINING_NAMES = [*TRAINING_NAMES, "common_voice_19_en"]
V2_TRAINING_IDS = [*TRAINING_IDS, "fsicoli/common_voice_19_0"]
V2_TRAINING_PROBABILITIES = [
    0.208896,
    0.225365,
    0.140233,
    0.013111,
    0.006451,
    0.043623,
    0.004320,
    0.008001,
    0.160100,
    0.011445,
    0.011571,
    0.019460,
    0.003043,
    0.011092,
    0.015857,
    0.117432,
]
V2_NOMINAL_RAW_ROWS = [
    4_767_938,
    5_143_820,
    3_200_734,
    299_255,
    147_249,
    995_677,
    98_600,
    182_605,
    1_501_271,
    107_319,
    108_502,
    182_482,
    28_539,
    104_014,
    148_688,
    1_101_170,
]


def test_bilingual_dataset_coordinates_and_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every Hub source uses the approved subset, split, columns, and revision."""
    config = _preset(monkeypatch)
    datasets = config.datasets

    expected_coordinates = {
        "p1": ("syvai/p1-segments", None, "train", "text", "audio"),
        "coral_read_aloud": (
            "syvai/danish-asr-unified",
            "default",
            "train",
            "text",
            "audio",
        ),
        "coral_conversation": (
            "syvai/danish-asr-unified",
            "default",
            "train",
            "text",
            "audio",
        ),
        "ftspeech": ("syvai/danish-asr-unified", "default", "train", "text", "audio"),
        "nota": ("syvai/danish-asr-unified", "default", "train", "text", "audio"),
        "nst": ("syvai/danish-asr-unified", "default", "train", "text", "audio"),
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
        "p1": P1_SEGMENTS_SHA,
        "coral_read_aloud": "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f",
        "coral_conversation": "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f",
        "ftspeech": "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f",
        "nota": "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f",
        "nst": "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f",
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
    assert "immutable_revision_env" not in datasets.p1
    assert datasets.p1.revision == P1_SEGMENTS_SHA
    assert datasets.p1.trust_remote_code is False
    assert all(
        dataset.get("trust_remote_code", False) is False
        for dataset in datasets.values()
    )
    assert all(
        dict(datasets[name].filters) == {"source": source}
        for name, source in {
            "coral_read_aloud": "coral_read_aloud",
            "coral_conversation": "coral_conversation",
            "ftspeech": "ftspeech",
            "nota": "nota",
            "nst": "nst_da",
        }.items()
    )
    assert dict(datasets.nst.filters) == {"source": "nst_da"}
    assert [
        name
        for name, dataset in datasets.items()
        if dataset.get("id") == "syvai/danish-asr-unified"
    ] == ["coral_read_aloud", "coral_conversation", "ftspeech", "nota", "nst"]
    expected_shard_ranges = {
        "nota": (354, 367),
        "ftspeech": (367, 563),
        "coral_read_aloud": (563, 623),
        "coral_conversation": (623, 653),
        "nst": (653, 689),
    }
    assert {
        name: (dataset.data_file_shards.start, dataset.data_file_shards.end)
        for name, dataset in datasets.items()
        if dataset.get("data_file_shards") is not None
    } == expected_shard_ranges
    for name, bounds in expected_shard_ranges.items():
        dataset = datasets[name]
        assert dataset.data_file_shards.template == "data/train-{shard:05d}.parquet"


def _preset(monkeypatch: MonkeyPatch) -> DictConfig:
    """Resolve the original preset without a launch-time P1 revision override.

    Returns:
        The resolved Sparkie preset.
    """
    monkeypatch.delenv("P1_SEGMENTS_REVISION", raising=False)
    return compose(config_name="bilingual")


def test_bilingual_evaluation_and_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert "common_voice_19_en" not in config.datasets
    assert not Path("config/datasets/fleurs_en_us.yaml").exists()


def test_bilingual_private_publication_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training stays local and publication metadata names every Hub source."""
    config = _preset(monkeypatch)

    assert config.push_to_hub is False
    assert config.enable_experiment_tracking is True
    assert config.experiment_tracking.type == "wandb"
    assert config.experiment_tracking.name_experiment == "hviske"
    assert config.experiment_tracking.name_group == "default"
    assert config.experiment_tracking.name_run == config.model_id
    assert config.experiment_tracking.entity is None
    assert config.experiment_tracking.job_type == "train"
    assert list(config.experiment_tracking.tags) == [
        "bilingual",
        "production",
        "parakeet-tdt",
        "max8s",
    ]
    assert config.experiment_tracking.mode == "online"
    assert config.experiment_tracking.log_model is False
    assert config.experiment_tracking.watch is False
    assert config.model.name == "parakeet-tdt"
    assert config.model.revision == "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    assert config.model.learning_rate == 5e-6
    assert str(config.model_id).startswith("parakeet-tdt-")
    assert config.private is True
    assert config.private_only is True
    assert config.save_total_limit == 3
    assert config.max_validation_samples_per_dataset == 1000
    assert config.max_seconds_per_example == 8.0
    assert config.shuffle_buffer_size == 1
    assert config.total_batch_size == 60
    assert config.per_device_batch_size == 6
    assert config.max_steps == 200_000
    assert config.stop_after_steps is None
    assert config.warmup_steps == 1_000
    assert config.logging_steps == 10
    assert config.eval_steps == 2_000
    assert config.save_steps == 500
    assert list(config.model_card_languages) == ["da", "en"]
    assert list(config.training_dataset_ids) == [
        "syvai/p1-segments",
        "syvai/danish-asr-unified",
        "MLCommons/peoples_speech",
        "edinburghcstr/ami",
        "facebook/voxpopuli",
        "openslr/librispeech_asr",
    ]


def test_bilingual_shuffle_buffers_are_source_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production preset keeps exact effective shuffle values per source."""
    config = _preset(monkeypatch)

    expected_buffers = {
        "p1": 1,
        "drtv_local": 128,
        "youtube_local": 128,
        "coral_read_aloud": 16,
        "coral_conversation": 16,
        "ftspeech": 16,
        "nota": 16,
        "nst": 16,
        "peoples_speech_clean": 1,
        "ami_sdm": 1,
        "ami_ihm": 1,
        "voxpopuli_en": 1,
        "librispeech_clean_train_100": 1,
        "librispeech_clean_train_360": 1,
        "librispeech_other_train_500": 1,
    }
    assert config.shuffle_buffer_size == 1
    assert {
        name: dataset.get("shuffle_buffer_size", config.shuffle_buffer_size)
        for name, dataset in config.datasets.items()
    } == expected_buffers


def test_bilingual_training_order_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preset keeps source order aligned with the approved probabilities."""
    config = _preset(monkeypatch)

    assert len(config.datasets) == 15
    assert list(config.datasets) == TRAINING_NAMES
    assert [
        dataset.get("id", dataset.get("type")) for dataset in config.datasets.values()
    ] == TRAINING_IDS
    assert list(config.dataset_probabilities) == TRAINING_PROBABILITIES
    assert sum(config.dataset_probabilities) == pytest.approx(1.0)
    assert sum(config.dataset_probabilities[:8]) == pytest.approx(0.6)
    assert sum(config.dataset_probabilities[8:]) == pytest.approx(0.4)


def test_bilingual_v2_common_voice_coordinates_and_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common Voice uses the pinned English Hub source and standard filtering."""
    dataset = _preset_v2(monkeypatch).datasets.common_voice_19_en

    assert (
        dataset.id,
        dataset.subset,
        dataset.train_name,
        dataset.text_column,
        dataset.audio_column,
        dataset.language,
    ) == ("fsicoli/common_voice_19_0", "en", "train", "sentence", "audio", "en")
    assert dataset.filter_dataset is True
    assert "filters" not in dataset
    assert dataset.trust_remote_code is False
    assert dataset.revision == "590c8abec6cf7c8d06e650f1438e60332a796e11"


def _preset_v2(monkeypatch: MonkeyPatch) -> DictConfig:
    """Resolve the bilingual v2 preset without a launch-time P1 revision override.

    Returns:
        The resolved bilingual v2 preset.
    """
    monkeypatch.delenv("P1_SEGMENTS_REVISION", raising=False)
    return compose(config_name="bilingual_v2")


def test_bilingual_v2_preserves_original_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v2 preset is additive and leaves the original mix untouched."""
    config = _preset(monkeypatch)

    assert list(config.datasets) == TRAINING_NAMES
    assert list(config.dataset_probabilities) == TRAINING_PROBABILITIES
    assert config.max_steps == 200_000
    assert config.total_batch_size == 60
    assert "common_voice_19_en" not in config.datasets


def test_bilingual_v2_probabilities_provide_balanced_nominal_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v2 mix gives each language's source its intended nominal exposure."""
    config = _preset_v2(monkeypatch)
    total_sampled_rows = config.max_steps * config.total_batch_size
    nominal_exposures = [
        probability * total_sampled_rows / nominal_rows
        for probability, nominal_rows in zip(
            config.dataset_probabilities, V2_NOMINAL_RAW_ROWS, strict=True
        )
    ]

    assert nominal_exposures[:8] == pytest.approx([1.002] * 8, abs=0.001)
    assert nominal_exposures[8:] == pytest.approx([2.438] * 8, abs=0.001)
    assert nominal_exposures[V2_TRAINING_NAMES.index("coral_conversation")] == (
        pytest.approx(1.002, abs=0.001)
    )


def test_bilingual_v2_training_horizon_preserves_effective_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v2 run uses the row-balanced horizon without changing batch size."""
    config = _preset_v2(monkeypatch)

    assert config.max_steps == 381_000
    assert config.total_batch_size == 60


def test_bilingual_v2_training_order_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v2 preset keeps order aligned with its row-balanced probabilities."""
    config = _preset_v2(monkeypatch)

    assert list(config.datasets) == V2_TRAINING_NAMES
    assert [
        dataset.get("id", dataset.get("type")) for dataset in config.datasets.values()
    ] == V2_TRAINING_IDS
    assert list(config.dataset_probabilities) == V2_TRAINING_PROBABILITIES
    assert sum(config.dataset_probabilities) == pytest.approx(1.0)
    assert sum(config.dataset_probabilities[:8]) == pytest.approx(0.65)
    assert sum(config.dataset_probabilities[8:]) == pytest.approx(0.35)


def test_bilingual_wandb_payload_has_published_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved production payload contains configured dataset sources."""
    payload = wandb_module._resolved_config_payload(config=_preset(monkeypatch))

    assert payload["model_dir"] == "[REDACTED]"
    assert payload["cache_dir"] is None
    datasets = t.cast(dict[str, object], payload["datasets"])
    drtv = t.cast(dict[str, object], datasets["drtv_local"])
    youtube = t.cast(dict[str, object], datasets["youtube_local"])
    assert drtv["manifest_path"] == "[REDACTED]"
    assert youtube["manifest_path"] == "[REDACTED]"
    assert "p1" in datasets
    assert all(isinstance(dataset, dict) for dataset in datasets.values())


def test_p1_segments_revision_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The P1 dataset always resolves to its frozen published commit."""
    monkeypatch.setenv("P1_SEGMENTS_REVISION", "main")
    config = compose(config_name="bilingual")
    OmegaConf.resolve(config)

    from hviske.utils import validate_immutable_source_revision

    assert config.datasets.p1.revision == P1_SEGMENTS_SHA
    assert "immutable_revision_env" not in config.datasets.p1
    assert (
        validate_immutable_source_revision(
            str(config.datasets.p1.revision), revision_label="P1 dataset revision"
        )
        == P1_SEGMENTS_SHA
    )
