"""Focused tests for native Cohere ASR support."""

import contextlib
import typing as t
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from hviske.cohere import (
    CohereAsrForConditionalGeneration,
    CohereAsrProcessor,
    CohereModelSetup,
    CohereSeq2SeqTrainer,
)
from hviske.data_collators import DataCollatorCohereWithPadding
from hviske.data_models import Processor
from hviske.model_setup import load_model_setup


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0
    unk_token_id = 1

    def get_vocab(self) -> dict[str, int]:
        return {
            "▁": 2,
            "<|startofcontext|>": 3,
            "<|startoftranscript|>": 4,
            "<|emo:undefined|>": 5,
            "<|da|>": 6,
            "<|pnc|>": 7,
            "<|nopnc|>": 8,
            "<|noitn|>": 9,
            "<|notimestamp|>": 10,
            "<|nodiarize|>": 11,
        }

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        vocabulary = self.get_vocab()
        if isinstance(tokens, str):
            return vocabulary.get(tokens, self.unk_token_id)
        return [int(t.cast(int, self.convert_tokens_to_ids(token))) for token in tokens]

    def pad(
        self, features: list[dict[str, list[int]]], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        del kwargs
        width = max(len(feature["input_ids"]) for feature in features)
        ids = [
            feature["input_ids"]
            + [self.pad_token_id] * (width - len(feature["input_ids"]))
            for feature in features
        ]
        masks = [[int(token != self.pad_token_id) for token in row] for row in ids]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class _FeatureExtractor:
    def pad(self, features: list[dict], **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        width = max(feature["input_features"].shape[0] for feature in features)
        values = []
        masks = []
        for feature in features:
            current = feature["input_features"]
            values.append(
                torch.nn.functional.pad(current, (0, 0, 0, width - len(current)))
            )
            mask = feature.get("attention_mask")
            if mask is None:
                mask = torch.ones(len(current), dtype=torch.bool)
            masks.append(torch.nn.functional.pad(mask, (0, width - len(mask))))
        return {
            "input_features": torch.stack(values),
            "attention_mask": torch.stack(masks),
        }


class _Processor:
    tokenizer = _Tokenizer()
    feature_extractor = _FeatureExtractor()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_config = SimpleNamespace(max_length=None, max_new_tokens=None)
        self.generated_inputs: dict[str, torch.Tensor] | None = None

    def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
        self.generated_inputs = kwargs
        return torch.tensor([[2, 3, 4]])

    def forward(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(loss=torch.tensor(0.5))


def test_model_factory_dispatches_cohere() -> None:
    """The model factory selects the dedicated Cohere setup."""
    config = OmegaConf.create({"model": {"type": "cohere"}})
    assert isinstance(load_model_setup(config), CohereModelSetup)


def test_native_loading_disables_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both checkpoint loaders use native Transformers classes only."""
    processor = object.__new__(CohereAsrProcessor)
    model = MagicMock(spec=CohereAsrForConditionalGeneration)
    model.config = SimpleNamespace(use_cache=True)
    processor_loader = MagicMock(return_value=processor)
    model_loader = MagicMock(return_value=model)
    monkeypatch.setattr(CohereAsrProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        CohereAsrForConditionalGeneration, "from_pretrained", model_loader
    )
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_model_id": "test/checkpoint",
                "freeze_feature_encoder": False,
            }
        }
    )
    setup = CohereModelSetup(config=config)
    setup.load_processor()
    setup.load_model()
    assert processor_loader.call_args.kwargs["trust_remote_code"] is False
    assert model_loader.call_args.kwargs["trust_remote_code"] is False


def test_language_validation_uses_checkpoint_vocabulary() -> None:
    """A tokenizer-added language is accepted, while an unknown one is rejected."""
    processor = object.__new__(CohereAsrProcessor)
    processor.tokenizer = _Tokenizer()
    assert len(processor.get_decoder_prompt_ids(language="da")) == 10
    with pytest.raises(ValueError, match="not represented"):
        processor.get_decoder_prompt_ids(language="xx")


def test_cohere_collator_aligns_prompt_transcript_and_mask() -> None:
    """The final prompt position starts the transcript loss."""
    processor = t.cast(Processor, _Processor())
    collator = DataCollatorCohereWithPadding(processor=processor, padding="longest")
    batch = collator(
        [
            {
                "input_features": torch.zeros(2, 128),
                "attention_mask": torch.tensor([True, True]),
                "decoder_input_ids": [10, 11],
                "labels": [20, 21],
            }
        ]
    )
    assert batch["decoder_input_ids"].tolist() == [[10, 11, 20, 21]]
    assert batch["labels"].tolist() == [[-100, 20, 21, 99]]
    assert batch["attention_mask"].tolist() == [[True, True]]


def test_cohere_trainer_generates_from_prompt_ids() -> None:
    """Evaluation generation receives the prompt even when its shape matches labels."""
    trainer = object.__new__(CohereSeq2SeqTrainer)
    trainer.args = t.cast(  # pyrefly: ignore[bad-assignment]
        object, SimpleNamespace(predict_with_generate=True, prediction_loss_only=False)
    )
    trainer.model = _Model()
    trainer._gen_kwargs = {}
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = t.cast(  # pyrefly: ignore[bad-assignment]
        object, contextlib.nullcontext
    )
    inputs = {
        "input_features": torch.zeros(1, 2, 128),
        "decoder_input_ids": torch.tensor([[10, 11, 20, 21]]),
        "labels": torch.tensor([[-100, 20, 21, 99]]),
    }
    trainer.prediction_step(
        model=trainer.model, inputs=inputs, prediction_loss_only=False
    )
    assert trainer.model.generated_inputs is not None
    assert torch.equal(
        trainer.model.generated_inputs["decoder_input_ids"], inputs["decoder_input_ids"]
    )
    assert "labels" not in trainer.model.generated_inputs
