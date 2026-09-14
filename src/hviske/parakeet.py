"""Model setup for Transformers-native NVIDIA Parakeet models."""

import logging
import os
import typing as t
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCTC,
    AutoProcessor,
    PreTrainedConfig,
    PreTrainedModel,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction

from .compute_metrics import compute_error_rate_metrics
from .data_collators import DataCollatorParakeetWithPadding
from .data_models import PreTrainedModelData, Processor
from .wav2vec2 import Wav2Vec2ModelSetup

logger = logging.getLogger(__package__)


ParakeetFamily: t.TypeAlias = t.Literal["ctc", "rnnt"]


class ParakeetGenerationTrainer(Trainer):
    """Trainer prediction step for Parakeet RNNT generation outputs."""

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Generate token sequences while retaining the normal Trainer loss.

        Returns:
            The loss, generated token IDs, and labels expected by Trainer.
        """
        if prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
            )

        prepared_inputs = self._prepare_inputs(inputs)
        labels = prepared_inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach()
        loss, _, _ = super().prediction_step(
            model=model,
            inputs=prepared_inputs,
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
        )
        generation_inputs = {
            key: value
            for key, value in prepared_inputs.items()
            if key in {"input_features", "attention_mask"}
        }
        generate = t.cast(t.Callable[..., object], model.generate)
        generated = generate(**generation_inputs)
        sequences = t.cast(torch.Tensor, getattr(generated, "sequences", generated))
        return loss, sequences.detach(), t.cast(torch.Tensor | None, labels)


class ParakeetModelSetup(Wav2Vec2ModelSetup):
    """Model setup for Transformers-native NVIDIA Parakeet checkpoints.

    CTC checkpoints are loaded through ``AutoModelForCTC``.  RNNT is registered
    as a general ``AutoModel`` architecture in Transformers 5.17 and therefore
    deliberately does not use a speech-to-sequence auto class.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        super().__init__(config=config)
        self.processor: Processor

    def load_model(self) -> PreTrainedModel:
        """Load the model class selected from the Transformers checkpoint config.

        Returns:
            A Parakeet CTC or RNNT model.

        """
        model_id = str(self.config.model.pretrained_model_id)
        family = self._load_family(model_id=model_id)
        kwargs = self._hub_kwargs()
        if family == "ctc":
            model = AutoModelForCTC.from_pretrained(model_id, **kwargs)
        else:
            model = AutoModel.from_pretrained(model_id, **kwargs)

        if self.config.model.get("freeze_feature_encoder", False):
            encoder = getattr(model, "encoder", None)
            if encoder is not None:
                for parameter in encoder.parameters():
                    parameter.requires_grad = False

        if self.config.gradient_checkpointing and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model

    def _hub_kwargs(self) -> dict[str, str | bool]:
        kwargs: dict[str, str | bool] = {
            "token": os.getenv("HUGGINGFACE_HUB_TOKEN") or True,
            "revision": self._revision(),
        }
        return kwargs

    def _revision(self) -> str:
        revision = self.config.model.get("revision")
        return str(revision) if revision is not None else "main"

    def _load_family(self, model_id: str) -> ParakeetFamily:
        try:
            config = AutoConfig.from_pretrained(model_id, **self._hub_kwargs())
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_id!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported; use a "
                "checkpoint with config.json."
            ) from error
        return parakeet_family(config=config)

    def load_processor(self) -> Processor:
        """Load the Transformers-native Parakeet processor.

        Returns:
            The Parakeet processor.

        Raises:
            ValueError:
                If the checkpoint is not packaged for Transformers.
        """
        model_id = str(self.config.model.pretrained_model_id)
        try:
            processor = AutoProcessor.from_pretrained(model_id, **self._hub_kwargs())
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_id!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported; use a "
                "checkpoint with config.json and processor files."
            ) from error
        if not hasattr(processor, "feature_extractor") or not hasattr(
            processor, "tokenizer"
        ):
            raise ValueError(
                f"Parakeet checkpoint {model_id!r} does not provide a "
                "Transformers feature extractor and tokenizer."
            )
        if self._load_family(model_id=model_id) == "rnnt":
            setattr(processor, "decoder_type", "rnnt")
        self.processor = processor
        return processor

    def load_saved(self) -> PreTrainedModelData:
        """Load a saved Parakeet model, processor, collator, and metrics.

        Returns:
            The saved model data.

        Raises:
            ValueError:
                If the saved checkpoint is not Transformers-native.
        """
        if Path(self.config.model_dir).exists():
            model_path = str(self.config.model_dir)
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        try:
            processor = AutoProcessor.from_pretrained(model_path, **self._hub_kwargs())
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_path!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported."
            ) from error

        family = self._load_family(model_id=model_path)
        if family == "rnnt":
            setattr(processor, "decoder_type", "rnnt")
        try:
            if family == "ctc":
                model = AutoModelForCTC.from_pretrained(
                    model_path, **self._hub_kwargs()
                )
            else:
                model = AutoModel.from_pretrained(model_path, **self._hub_kwargs())
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_path!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported."
            ) from error

        self.processor = processor
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=self.load_data_collator(),
            compute_metrics=self.load_compute_metrics(),
        )

    def load_compute_metrics(self) -> t.Callable[[EvalPrediction], dict]:
        """Return metrics that decode Parakeet CTC or generated RNNT IDs."""
        return partial(_compute_parakeet_metrics, processor=self.processor)

    def load_data_collator(self) -> DataCollatorParakeetWithPadding:
        """Return the feature-aware Parakeet collator."""
        return DataCollatorParakeetWithPadding(
            processor=self.processor,
            sample_rate=self.config.model.sampling_rate,
            padding=self.config.padding,
        )

    def load_trainer_class(self) -> t.Type[Trainer]:
        """Return Trainer or the generation-compatible RNNT Trainer."""
        family = self._load_family(model_id=str(self.config.model.pretrained_model_id))
        return Trainer if family == "ctc" else ParakeetGenerationTrainer


def parakeet_family(config: PreTrainedConfig) -> ParakeetFamily:
    """Identify a Parakeet family from its model type or architecture.

    Args:
        config:
            Transformers checkpoint configuration.

    Returns:
        The Parakeet family.

    Raises:
        ValueError:
            If the config is not a supported CTC or RNNT checkpoint, or if it
            is a TDT checkpoint.
    """
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(
        str(architecture).lower()
        for architecture in getattr(config, "architectures", []) or []
    )
    descriptor = f"{model_type} {architectures}"
    if "tdt" in descriptor:
        raise ValueError(
            "Parakeet TDT fine-tuning is unsupported: native TDT loss is broken "
            "on the supported Transformers stack. Use NVIDIA NeMo for TDT "
            "fine-tuning."
        )
    if "rnnt" in descriptor or "transducer" in descriptor:
        return "rnnt"
    if "ctc" in descriptor:
        return "ctc"
    raise ValueError(
        "Unsupported Parakeet checkpoint architecture. Expected a "
        "Transformers-native ParakeetForCTC or ParakeetForRNNT "
        f"config, got model_type={model_type!r}, architectures={architectures!r}."
    )


def _compute_parakeet_metrics(
    pred: EvalPrediction, processor: Processor
) -> dict[str, float]:
    """Decode Parakeet predictions without Trainer's concatenation sentinel.

    Returns:
        Character and word error rates for the predictions.
    """
    predictions = np.asarray(pred.predictions).copy()
    if predictions.ndim == 2:
        predictions[predictions == -100] = processor.tokenizer.pad_token_id
    is_rnnt = str(getattr(processor, "decoder_type", "")).lower() == "rnnt"
    return compute_error_rate_metrics(
        pred=EvalPrediction(predictions=predictions, label_ids=pred.label_ids),
        processor=processor,
        log_examples=False,
        label_group_tokens=False if is_rnnt else None,
    )
