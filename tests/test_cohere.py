"""Focused tests for native Cohere ASR support."""

import contextlib
import typing as t
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from hviske.cohere import (
    CohereAsrForConditionalGeneration,
    CohereAsrProcessor,
    CohereASRTranscriber,
    CohereModelSetup,
    CohereSeq2SeqTrainer,
    get_asr_call_kwargs,
    load_asr_transcriber,
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
        self.forward_inputs: dict[str, torch.Tensor] | None = None

    def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
        self.generated_inputs = kwargs
        return torch.tensor([[2, 3, 4]])

    def forward(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        self.forward_inputs = kwargs
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


def test_native_cohere_loader_dispatches_model_aware_transcriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared loader selects the prompt-aware path for native Cohere."""
    monkeypatch.setattr(
        "hviske.cohere.AutoConfig.from_pretrained",
        MagicMock(return_value=SimpleNamespace(model_type="cohere_asr")),
    )
    processor = MagicMock(spec=CohereAsrProcessor)
    model = MagicMock(spec=CohereAsrForConditionalGeneration)
    monkeypatch.setattr(
        CohereAsrProcessor, "from_pretrained", MagicMock(return_value=processor)
    )
    monkeypatch.setattr(
        CohereAsrForConditionalGeneration,
        "from_pretrained",
        MagicMock(return_value=model),
    )
    transcriber = load_asr_transcriber(
        model_id="test/cohere",
        no_lm=False,
        device=torch.device("cpu"),
        language="sv",
        punctuation=False,
    )
    assert isinstance(transcriber, CohereASRTranscriber)
    assert transcriber.language == "sv"
    assert transcriber.punctuation is False


def test_native_cohere_dispatch_omits_whisper_generation_kwargs() -> None:
    """Native Cohere entry points must not pass Whisper language/task kwargs."""
    transcriber = object.__new__(CohereASRTranscriber)
    assert get_asr_call_kwargs(transcriber) == {}


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
    assert batch["prompt_length"].tolist() == [2]


def test_cohere_collator_truncates_transcripts_to_model_limit() -> None:
    """Transcript tokens are truncated after reserving prompt and EOS space."""
    processor = t.cast(Processor, _Processor())
    collator = DataCollatorCohereWithPadding(
        processor=processor, padding="longest", max_length=4
    )
    batch = collator(
        [
            {
                "input_features": torch.zeros(2, 128),
                "decoder_input_ids": [10, 11],
                "labels": [20, 21, 22],
            }
        ]
    )
    assert batch["decoder_input_ids"].tolist() == [[10, 11, 20, 21]]
    assert batch["labels"].tolist() == [[-100, 20, 21, 99]]

    boundary_batch = DataCollatorCohereWithPadding(
        processor=processor, padding="longest", max_length=3
    )(
        [
            {
                "input_features": torch.zeros(2, 128),
                "decoder_input_ids": [10, 11],
                "labels": [20, 21],
            }
        ]
    )
    assert boundary_batch["decoder_input_ids"].tolist() == [[10, 11, 20]]
    assert boundary_batch["labels"].tolist() == [[-100, 20, 99]]

    with pytest.raises(ValueError, match="prompt and EOS"):
        DataCollatorCohereWithPadding(
            processor=processor, padding="longest", max_length=2
        )(
            [
                {
                    "input_features": torch.zeros(2, 128),
                    "decoder_input_ids": [10, 11],
                    "labels": [20],
                }
            ]
        )


def test_native_cohere_transcriber_builds_prompt_and_reassembles_chunks() -> None:
    """Native inference passes prompt IDs and reassembles processor chunks."""

    class _InferenceProcessor:
        feature_extractor = SimpleNamespace(sampling_rate=16_000)

        def __call__(self, audio: object, **kwargs: object) -> dict[str, object]:
            self.call = (audio, kwargs)
            return {
                "input_features": torch.zeros(3, 2, 128),
                "attention_mask": torch.ones(3, 2),
                "decoder_input_ids": torch.tensor([[10, 11]] * 3),
                "audio_chunk_index": [(0, 0), (0, 1), (1, None)],
            }

        def decode(self, sequences: torch.Tensor, **kwargs: object) -> list[str]:
            self.decode_call = (sequences, kwargs)
            return ["first", "second"]

    class _InferenceModel:
        def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
            self.inputs = kwargs
            return torch.tensor([[10, 11, 20], [10, 11, 21], [10, 11, 22]])

    processor = _InferenceProcessor()
    model = _InferenceModel()
    transcriber = CohereASRTranscriber(
        model=t.cast(CohereAsrForConditionalGeneration, model),
        processor=t.cast(CohereAsrProcessor, processor),
        device=torch.device("cpu"),
    )
    outputs = list(
        transcriber(
            [
                {"array": np.zeros(3), "sampling_rate": 16_000},
                {"array": np.zeros(3), "sampling_rate": 16_000},
            ],
            batch_size=2,
        )
    )
    assert outputs == [{"text": "first"}, {"text": "second"}]
    assert processor.call[1]["language"] == "da"
    assert processor.call[1]["punctuation"] is True
    assert processor.decode_call[1]["audio_chunk_index"] == [(0, 0), (0, 1), (1, None)]
    assert processor.decode_call[1]["language"] == "da"
    assert torch.equal(model.inputs["decoder_input_ids"], torch.tensor([[10, 11]] * 3))


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
        "decoder_attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "prompt_length": torch.tensor([2]),
        "labels": torch.tensor([[-100, 20, 21, 99]]),
    }
    trainer.prediction_step(
        model=trainer.model, inputs=inputs, prediction_loss_only=False
    )
    assert trainer.model.generated_inputs is not None
    assert torch.equal(
        trainer.model.generated_inputs["decoder_input_ids"], torch.tensor([[10, 11]])
    )
    assert torch.equal(
        trainer.model.generated_inputs["decoder_attention_mask"], torch.tensor([[1, 1]])
    )
    assert "labels" not in trainer.model.generated_inputs
    assert trainer.model.forward_inputs is not None
    assert torch.equal(
        trainer.model.forward_inputs["decoder_input_ids"], inputs["decoder_input_ids"]
    )
    assert torch.equal(trainer.model.forward_inputs["labels"], inputs["labels"])
    assert "prompt_length" not in trainer.model.forward_inputs
