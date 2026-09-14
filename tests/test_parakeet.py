"""Focused tests for Transformers-native Parakeet support."""

import typing as t
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from transformers import EvalPrediction, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.parakeet.feature_extraction_parakeet import (
    ParakeetFeatureExtractor,
)
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)

from hviske.cohere import get_asr_call_kwargs
from hviske.compute_metrics import compute_error_rate_metrics
from hviske.data_collators import DataCollatorParakeetWithPadding
from hviske.data_models import Processor
from hviske.model_setup import load_model_setup
from hviske.parakeet import (
    ParakeetGenerationTrainer,
    ParakeetModelSetup,
    parakeet_family,
)


def test_parakeet_collator_pads_frames_masks_and_labels() -> None:
    """Feature width is inferred by the feature extractor rather than hard-coded."""
    processor = SimpleNamespace(
        feature_extractor=ParakeetFeatureExtractor(feature_size=3),
        tokenizer=MagicMock(),
    )
    processor.tokenizer.pad.return_value = {"input_ids": torch.tensor([[4, 5], [6, 0]])}
    collator = DataCollatorParakeetWithPadding(
        processor=t.cast(Processor, processor), sample_rate=16_000, padding="longest"
    )

    batch = collator(
        [
            {
                "input_features": torch.zeros(4, 3),
                "attention_mask": torch.ones(4),
                "labels": [4, 5],
            },
            {
                "input_features": torch.zeros(6, 3),
                "attention_mask": torch.ones(6),
                "labels": [6],
            },
        ]
    )

    assert batch["input_features"].shape == (2, 6, 3)
    assert batch["attention_mask"].shape == (2, 6)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert batch["labels"].tolist() == [[4, 5], [6, 0]]


def test_parakeet_dispatch() -> None:
    """The model factory selects the Parakeet setup."""
    config = OmegaConf.create({"model": {"type": "parakeet"}})
    assert isinstance(load_model_setup(config), ParakeetModelSetup)


def test_parakeet_family_rejects_collection_checkpoint() -> None:
    """NeMo collection configs fail before an auto model is selected."""
    with pytest.raises(ValueError, match="Unsupported Parakeet checkpoint"):
        parakeet_family(SimpleNamespace(model_type="nemo"))


def test_parakeet_family_uses_checkpoint_architecture() -> None:
    """TDT and RNNT are distinguished using native Transformers config data."""
    assert parakeet_family(SimpleNamespace(model_type="parakeet_tdt")) == "tdt"
    assert (
        parakeet_family(
            SimpleNamespace(model_type="unknown", architectures=["ParakeetForRNNT"])
        )
        == "rnnt"
    )


def test_parakeet_generation_trainer_unwraps_transducer_output(tmp_path: Path) -> None:
    """Trainer metrics receive sequences, not the RNNT generation wrapper."""

    class TinyModel(torch.nn.Module):
        config = SimpleNamespace(
            model_type="parakeet_rnnt", keys_to_ignore_at_inference=[]
        )

        def forward(
            self,
            input_features: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
        ) -> CausalLMOutput:
            return CausalLMOutput(
                loss=t.cast(torch.FloatTensor, torch.tensor(0.5)),
                logits=t.cast(torch.FloatTensor, input_features),
            )

        def generate(
            self,
            input_features: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            return SimpleNamespace(sequences=torch.tensor([[1, 2]]))

    trainer = ParakeetGenerationTrainer(
        model=TinyModel(),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    loss, predictions, labels = trainer.prediction_step(
        model=t.cast(torch.nn.Module, trainer.model),
        inputs={
            "input_features": torch.zeros(1, 2, 3),
            "attention_mask": torch.ones(1, 2),
            "labels": torch.tensor([[1, 2]]),
        },
        prediction_loss_only=False,
    )

    assert loss is not None
    assert torch.equal(t.cast(torch.Tensor, predictions), torch.tensor([[1, 2]]))
    assert torch.equal(t.cast(torch.Tensor, labels), torch.tensor([[1, 2]]))


def test_parakeet_inference_omits_whisper_generation_kwargs() -> None:
    """Shared pipelines must not send language/task kwargs to Parakeet."""
    transcriber = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="parakeet_tdt"))
    )
    assert (
        get_asr_call_kwargs(t.cast(AutomaticSpeechRecognitionPipeline, transcriber))
        == {}
    )


def test_parakeet_load_model_selects_native_auto_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CTC uses AutoModelForCTC while transducers use AutoModel."""
    config = OmegaConf.create(
        {
            "model": {
                "type": "parakeet",
                "pretrained_model_id": "checkpoint",
                "freeze_feature_encoder": False,
            },
            "gradient_checkpointing": False,
        }
    )
    setup = ParakeetModelSetup(config=config)
    ctc_model = SimpleNamespace(config=SimpleNamespace())
    ctc_loader = MagicMock(return_value=ctc_model)
    general_loader = MagicMock()
    monkeypatch.setattr(
        "hviske.parakeet.AutoConfig.from_pretrained",
        MagicMock(return_value=SimpleNamespace(model_type="parakeet_ctc")),
    )
    monkeypatch.setattr("hviske.parakeet.AutoModelForCTC.from_pretrained", ctc_loader)
    monkeypatch.setattr("hviske.parakeet.AutoModel.from_pretrained", general_loader)

    assert setup.load_model() is ctc_model
    ctc_loader.assert_called_once()
    general_loader.assert_not_called()


def test_parakeet_metric_decodes_generated_sequences() -> None:
    """RNNT/TDT generated IDs use the same tokenizer path as CTC labels."""
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.batch_decode.side_effect = [["hej"], ["hej"]]
    processor = SimpleNamespace(
        tokenizer=tokenizer, batch_decode=tokenizer.batch_decode
    )
    prediction = EvalPrediction(
        predictions=torch.tensor([[1, 2]]).numpy(),
        label_ids=torch.tensor([[1, 2]]).numpy(),
    )

    metrics = compute_error_rate_metrics(
        pred=prediction, processor=t.cast(Processor, processor), log_examples=False
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}
    assert tokenizer.batch_decode.call_count == 2
