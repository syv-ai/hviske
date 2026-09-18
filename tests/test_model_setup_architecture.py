"""Focused contracts for shared model setup and decoder dispatch."""

import typing as t

import pytest
import torch
from omegaconf import OmegaConf
from transformers import (
    CohereAsrProcessor,
    ParakeetFeatureExtractor,
    ParakeetProcessor,
    ParakeetTokenizer,
    Trainer,
)

from hviske.base_model_setup import BaseModelSetup
from hviske.cohere import CohereModelSetup, CohereSeq2SeqTrainer
from hviske.parakeet import (
    DataCollatorParakeetWithPadding,
    ParakeetCTCStrategy,
    ParakeetDecoderStrategy,
    ParakeetGenerationTrainer,
    ParakeetModelSetup,
    ParakeetRNNTStrategy,
    ParakeetTDTStrategy,
    parakeet_decoder_strategy,
)
from hviske.wav2vec2 import Wav2Vec2ModelSetup
from hviske.whisper import WhisperModelSetup


def test_cohere_keeps_prompt_aware_seq2seq_trainer() -> None:
    """Cohere remains on its prompt-aware seq2seq path, not Parakeet's collator."""
    setup = CohereModelSetup(config=OmegaConf.create({}))
    assert setup.load_trainer_class() is CohereSeq2SeqTrainer
    assert not isinstance(setup, ParakeetModelSetup)
    assert CohereAsrProcessor is not ParakeetProcessor


@pytest.mark.parametrize(
    "setup_class",
    [Wav2Vec2ModelSetup, WhisperModelSetup, CohereModelSetup, ParakeetModelSetup],
)
def test_model_setups_share_only_generic_base(
    setup_class: type[BaseModelSetup],
) -> None:
    """Every setup uses generic state without changing model-specific contracts."""
    assert setup_class.__bases__ == (BaseModelSetup,)


@pytest.mark.parametrize("family", ["ctc", "rnnt", "tdt"])
def test_parakeet_label_and_decoder_padding_contract(family: str) -> None:
    """RNNT labels use blank padding while decoder inputs retain tokenizer padding."""
    processor = _parakeet_processor(family=family)
    collator = DataCollatorParakeetWithPadding(
        processor=processor, sample_rate=16_000, padding="longest"
    )
    first = {"input_features": torch.zeros(2, 1), "labels": [1]}
    second = {"input_features": torch.zeros(2, 1), "labels": [1, 2]}
    if family != "ctc":
        first["decoder_input_ids"] = [5, 1]
        second["decoder_input_ids"] = [5, 1, 2]

    batch = collator([first, second])
    expected_label_padding = 5 if family == "rnnt" else 0
    assert batch["labels"].tolist() == [[1, expected_label_padding], [1, 2]]
    if family != "ctc":
        assert batch["decoder_input_ids"].tolist() == [[5, 1, 0], [5, 1, 2]]


def _parakeet_processor(family: str) -> ParakeetProcessor:
    tokenizer = ParakeetTokenizer(
        vocab={"<pad>": 0, "a": 1, "<unk>": 2, "<blank>": 5},
        pad_token="<pad>",
        unk_token="<unk>",
        blank_token="<blank>",
    )
    return ParakeetProcessor(
        feature_extractor=ParakeetFeatureExtractor(feature_size=1),
        tokenizer=tokenizer,
        blank_token="<blank>",
        decoder_type=family,
    )


@pytest.mark.parametrize(
    ("family", "strategy_class", "trainer_class"),
    [
        ("ctc", ParakeetCTCStrategy, Trainer),
        ("rnnt", ParakeetRNNTStrategy, ParakeetGenerationTrainer),
        ("tdt", ParakeetTDTStrategy, ParakeetGenerationTrainer),
    ],
)
def test_parakeet_strategy_matrix(
    family: str,
    strategy_class: type[ParakeetDecoderStrategy],
    trainer_class: type[Trainer],
) -> None:
    """Each decoder family has an explicit loader/trainer contract."""
    strategy = parakeet_decoder_strategy(
        t.cast(t.Literal["ctc", "rnnt", "tdt"], family)
    )
    assert isinstance(strategy, strategy_class)
    assert strategy.trainer_class() is trainer_class
