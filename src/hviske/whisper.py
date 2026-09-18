"""Model setup for Whisper models."""

import logging
import os
from functools import partial
from pathlib import Path
from typing import Type

from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from .base_model_setup import BaseModelSetup
from .compute_metrics import compute_error_rate_metrics
from .data_collators import DataCollatorSpeechSeq2SeqWithPadding
from .data_models import PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


class WhisperModelSetup(BaseModelSetup):
    """Model setup for Whisper models."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        super().__init__(config=config)
        self.processor: WhisperProcessor

    def load_data_collator(self) -> DataCollatorSpeechSeq2SeqWithPadding:
        """Return the data collator for the model.

        Returns:
            The data collator.
        """
        return DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            sample_rate=self.config.model.sampling_rate,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )

    def load_model(self) -> WhisperForConditionalGeneration:
        """Return the model for the setup."""
        with transformers_output_ignored():
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model.pretrained_model_id,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
                revision=self._revision(),
                dropout=self.config.model.dropout,
                activation_dropout=self.config.model.activation_dropout,
                attention_dropout=self.config.model.attention_dropout,
                pad_token_id=self.processor.tokenizer.pad_token_id,  # type: ignore[attr-defined]
                bos_token_id=self.processor.tokenizer.bos_token_id,  # type: ignore[attr-defined]
                eos_token_id=self.processor.tokenizer.eos_token_id,  # type: ignore[attr-defined]
                apply_spec_augment=True,
                mask_time_prob=self.config.model.mask_time_prob,
                mask_time_length=self.config.model.mask_time_length,
                mask_feature_prob=self.config.model.mask_feature_prob,
                mask_feature_length=self.config.model.mask_feature_length,
                encoder_layerdrop=self.config.model.layerdrop,
                decoder_layerdrop=self.config.model.layerdrop,
            )
            assert isinstance(model, WhisperForConditionalGeneration)

        if self.config.model.freeze_feature_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.proj_out.parameters():
                param.requires_grad = True

        # The Whisper model has token ids that are forced as model outputs before
        # autoregressive generation is started (forced_decoder_ids). These token ids
        # control the transcription language and task for zero-shot ASR. For
        # fine-tuning, we'll set these ids to None, as we'll train the model to predict
        # the correct language and task. There are also tokens that are completely
        # suppressed during generation (suppress_tokens). These tokens have their log
        # probabilities set to -inf, such that they are never sampled. We'll override
        # these tokens to an empty list, meaning no tokens are suppressed.
        # Source: https://hf.co/blog/fine-tune-whisper#load-a-pre-trained-checkpoint
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        # Disabling cache as this is incompatible with gradient checkpointing
        model.config.use_cache = False

        return model

    def _revision(self) -> str:
        revision = self.config.model.get("revision")
        return str(revision) if revision is not None else "main"

    def load_processor(self) -> WhisperProcessor:
        """Return the processor for the model."""
        processor_or_tup = WhisperProcessor.from_pretrained(
            self.config.model.pretrained_model_id,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            revision=self._revision(),
        )
        assert isinstance(processor_or_tup, WhisperProcessor)
        self.processor = processor_or_tup

        # Whisper tokenizers are misconfigured with a max_length that is too high, but
        # the correct max_length is stored in the generation config, so update it here.
        model_id = self.config.model.pretrained_model_id
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            revision=self._revision(),
        )
        max_length = generation_config.max_length
        if max_length is None:
            hf_config = AutoConfig.from_pretrained(
                model_id,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
                revision=self._revision(),
            )
            max_length = int(hf_config.max_target_positions)
        self.processor.tokenizer.model_max_length = min(  # type: ignore[attr-defined]
            self.processor.tokenizer.model_max_length,  # type: ignore[attr-defined]
            max_length,
        )

        return self.processor

    def load_saved(self) -> PreTrainedModelData:
        """Load the model setup.

        Returns:
            The model setup.
        """
        if Path(self.config.model_dir).exists():
            model_path = self.config.model_dir
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        processor: Processor
        processor_or_tup = WhisperProcessor.from_pretrained(
            model_path,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            revision=self._revision(),
        )
        assert isinstance(processor_or_tup, WhisperProcessor)
        processor = processor_or_tup

        model_or_tup = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            revision=self._revision(),
        )
        assert isinstance(model_or_tup, WhisperForConditionalGeneration)
        model = model_or_tup

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            sample_rate=self.config.model.sampling_rate,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=data_collator,
            compute_metrics=partial(compute_error_rate_metrics, processor=processor),
        )

    def load_trainer_class(self) -> Type[Trainer]:
        """Return the trainer class used to train the model."""
        return Seq2SeqTrainer

    def load_training_arguments(self) -> TrainingArguments:
        """Return the training arguments for the model."""
        return Seq2SeqTrainingArguments(
            **self._training_arguments_kwargs(sequence_to_sequence=True)
        )
