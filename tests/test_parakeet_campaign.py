"""Regression tests for the independent Parakeet-TDT campaign presets."""

from collections.abc import Generator, Sequence
from typing import cast

import pytest
from datasets import Dataset, IterableDataset
from hydra import compose
from omegaconf import DictConfig

from hviske.data import limit_training_dataset, validate_max_train_samples

DANISH_NAMES = [
    "coral_read_aloud",
    "coral_conversation",
    "ftspeech",
    "fleurs",
    "common_voice_19",
]
DANISH_CAPS = [299255, 147249, 995677, 1032, 3484]
DANISH_ADDITIONS = ["p1", "drtv_local", "youtube_local", "nota", "nst"]
ENGLISH_NAMES = [
    "peoples_speech_clean",
    "ami_sdm",
    "ami_ihm",
    "voxpopuli_en",
    "librispeech_clean_train_100",
    "librispeech_clean_train_360",
    "librispeech_other_train_500",
    "common_voice_19_en",
]
CAMPAIGN_NAMES = [
    "parakeet_tdt_danish_leaderboard",
    "parakeet_tdt_danish_05",
    "parakeet_tdt_danish_20",
    "parakeet_tdt_danish_50",
    "parakeet_tdt_danish_100",
    "parakeet_tdt_bilingual_10",
    "parakeet_tdt_bilingual_50",
    "parakeet_tdt_bilingual_100",
    "parakeet_tdt_bilingual_65_35",
]


def test_campaign_revisions_and_immutable_final_horizon() -> None:
    """The CV19 stand-in and final 65/35 campaign are pinned and fresh."""
    config = _config("parakeet_tdt_danish_leaderboard")
    assert config.datasets.coral_read_aloud.revision == (
        "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f"
    )
    assert config.datasets.coral_conversation.revision == (
        "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f"
    )
    assert config.datasets.ftspeech.revision == (
        "5a3a49ee981baab6e1e37ddd2c45f9943c27d08f"
    )
    assert config.datasets.fleurs.revision == (
        "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    )
    assert (
        config.datasets.common_voice_19.revision
        == "590c8abec6cf7c8d06e650f1438e60332a796e11"
    )
    final = _config("parakeet_tdt_bilingual_65_35")
    assert final.max_steps == 381000
    assert final.model_id == "parakeet-tdt-bilingual-65-35-v1"
    assert final.resume_from_checkpoint is False
    assert final.push_to_hub is False
    assert sum(final.dataset_probabilities[:10]) == pytest.approx(0.65)
    assert sum(final.dataset_probabilities[10:]) == pytest.approx(0.35)


def _config(name: str) -> DictConfig:
    return compose(config_name=name)


@pytest.mark.parametrize("name", CAMPAIGN_NAMES)
def test_campaign_safety_defaults_resolve(name: str) -> None:
    """Every campaign keeps the conservative rate and upload policy."""
    config = _config(name)

    assert config.model.learning_rate == pytest.approx(5e-6)
    assert config.dataset_num_workers == 1
    assert config.dataloader_num_workers == 0
    assert config.experiment_tracking.log_model is False
    assert config.push_to_hub is False
    assert config.resume_from_checkpoint is False


def test_cap_is_after_filter_and_stream_cap_is_lazy() -> None:
    """Filtering first means rejected rows do not consume a source cap."""
    dataset = Dataset.from_list([{"value": value} for value in range(8)])
    filtered = dataset.filter(lambda row: row["value"] % 2 == 0)
    capped = limit_training_dataset(filtered, 2)
    assert cast(Dataset, capped)["value"] == [0, 2]

    seen: list[int] = []

    def rows() -> Generator[dict[str, int], None, None]:
        for value in range(100):
            seen.append(value)
            yield {"value": value}

    stream = IterableDataset.from_generator(rows)
    limited = limit_training_dataset(stream, 3)
    assert isinstance(limited, IterableDataset)
    assert [row["value"] for row in limited] == [0, 1, 2]
    assert len(seen) == 3


def test_danish_leaderboard_source_order_caps_and_fresh_run_settings() -> None:
    """Stage 1 uses only the five documented Danish train sources."""
    config = _config("parakeet_tdt_danish_leaderboard")
    assert list(config.datasets) == DANISH_NAMES
    assert [
        config.datasets[name].max_train_samples for name in DANISH_NAMES
    ] == DANISH_CAPS
    assert sum(config.dataset_probabilities) == pytest.approx(1)
    for probability, cap in zip(config.dataset_probabilities, DANISH_CAPS):
        assert probability == pytest.approx(cap / sum(DANISH_CAPS))
    assert config.max_steps == 24112
    assert config.total_batch_size == 60
    assert config.per_device_batch_size == 4
    assert config.dataloader_num_workers == 0
    assert config.max_seconds_per_example == 8
    assert config.resume_from_checkpoint is False
    assert config.push_to_hub is False
    assert all(item.language == "da" for item in config.evaluation_datasets)


@pytest.mark.parametrize(
    ("name", "steps", "english_caps"),
    [
        (
            "parakeet_tdt_bilingual_10",
            252810,
            [150127, 10732, 10850, 18248, 2854, 10401, 14869, 110117],
        ),
        (
            "parakeet_tdt_bilingual_50",
            274690,
            [750636, 53660, 54251, 91241, 14270, 52007, 74344, 550585],
        ),
        (
            "parakeet_tdt_bilingual_100",
            302040,
            [1501271, 107319, 108502, 182482, 28539, 104014, 148688, 1101170],
        ),
    ],
)
def test_english_is_introduced_after_full_danish_and_adds_eval(
    name: str, steps: int, english_caps: Sequence[int]
) -> None:
    """English appears only in bilingual stages and is evaluated then too."""
    config = _config(name)
    names = list(config.datasets)
    assert names[:5] == DANISH_NAMES
    assert names[5:10] == DANISH_ADDITIONS
    assert names[10:] == ENGLISH_NAMES
    assert [config.datasets[item].max_train_samples for item in names[10:]] == list(
        english_caps
    )
    assert config.max_steps == steps
    assert [item.language for item in config.evaluation_datasets].count("en") == 3
    assert sum(config.dataset_probabilities) == pytest.approx(1)


def test_max_train_samples_validation_rejects_bool_and_non_positive() -> None:
    """Caps are positive integers, not truthy flags or arbitrary numbers."""
    assert validate_max_train_samples(None) is None
    assert validate_max_train_samples(3) == 3
    for invalid in (True, False, 0, -1, 1.5, "3"):
        with pytest.raises(ValueError, match="max_train_samples"):
            validate_max_train_samples(invalid)


@pytest.mark.parametrize(
    ("name", "steps", "fraction_caps"),
    [
        ("parakeet_tdt_danish_05", 35274, [238397, 257191, 160037, 4930, 9130]),
        ("parakeet_tdt_danish_20", 68758, [953588, 1028764, 640147, 19720, 36521]),
        ("parakeet_tdt_danish_50", 135726, [2383969, 2571910, 1600367, 49300, 91302]),
        ("parakeet_tdt_danish_100", 247340, [4767938, 5143820, 3200734, 98600, 182605]),
    ],
)
def test_progressive_danish_caps_and_horizons(
    name: str, steps: int, fraction_caps: Sequence[int]
) -> None:
    """Danish stages preserve the leaderboard prefix and add sources progressively."""
    config = _config(name)
    names = list(config.datasets)
    assert names[:5] == DANISH_NAMES
    assert names[5:] == DANISH_ADDITIONS
    assert [
        config.datasets[item].max_train_samples for item in names[:5]
    ] == DANISH_CAPS
    assert [config.datasets[item].max_train_samples for item in names[5:]] == list(
        fraction_caps
    )
    assert config.max_steps == steps
    assert all(item.language == "da" for item in config.evaluation_datasets)
