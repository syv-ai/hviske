"""Model setup for Wav2Vec 2.0 models."""

import json
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Type

from omegaconf import DictConfig
from transformers import (
    PreTrainedModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from .base_model_setup import BaseModelSetup
from .compute_metrics import compute_error_rate_metrics
from .data_collators import DataCollatorCTCWithPadding
from .data_models import PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


class Wav2Vec2ModelSetup(BaseModelSetup):
    """Model setup for Wav2Vec 2.0 models."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        super().__init__(config=config)
        self.processor: Processor

    def load_data_collator(self) -> DataCollatorMixin:
        """Return the data collator for the model.

        Returns:
            The data collator.
        """
        return DataCollatorCTCWithPadding(
            processor=self.processor,
            sample_rate=self.config.model.sampling_rate,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )

    def load_model(self) -> PreTrainedModel:
        """Return the model for the model."""
        with transformers_output_ignored():
            model = Wav2Vec2ForCTC.from_pretrained(
                self.config.model.pretrained_model_id,
                activation_dropout=self.config.model.activation_dropout,
                attention_dropout=self.config.model.attention_dropout,
                hidden_dropout=self.config.model.hidden_dropout,
                feat_proj_dropout=self.config.model.feat_proj_dropout,
                final_dropout=self.config.model.final_dropout,
                apply_spec_augment=True,
                mask_time_prob=self.config.model.mask_time_prob,
                mask_time_length=self.config.model.mask_time_length,
                mask_feature_prob=self.config.model.mask_feature_prob,
                mask_feature_length=self.config.model.mask_feature_length,
                layerdrop=self.config.model.layerdrop,
                ctc_loss_reduction=self.config.model.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,  # type: ignore[missing-attribute]
                bos_token_id=self.processor.tokenizer.bos_token_id,  # type: ignore[missing-attribute]
                eos_token_id=self.processor.tokenizer.eos_token_id,  # type: ignore[missing-attribute]
                vocab_size=len(self.processor.tokenizer.get_vocab()),  # type: ignore[missing-attribute]
                ctc_zero_infinity=True,
            )
        assert isinstance(model, Wav2Vec2ForCTC)

        if self.config.model.freeze_feature_encoder:
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

        return model

    def load_processor(self) -> Processor:
        """Return the processor for the model.

        Returns:
            The processor for the model.

        Raises:
            ValueError:
                If the tokeniser could not be loaded.
        """
        # We dump the vocabulary to a file since the tokenizer uses this file during
        # initialisation
        while True:
            try:
                dump_vocabulary(self.config)
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                    self.config.model_dir,
                    pad_token="<pad>",
                    unk_token="<unk>",
                    bos_token="<s>",
                    eos_token="</s>",
                    word_delimiter_token="|",
                    replace_word_delimiter_char=" ",
                )
                if not tokenizer:
                    raise ValueError("Tokeniser could not be loaded.")
                break
            except json.decoder.JSONDecodeError:
                log_message = "JSONDecodeError while loading tokeniser"
                process_id = os.getenv("RANK")
                if process_id is not None:
                    log_message += f" in process {process_id}"
                log_message += ". Retrying in a second."
                if self.is_main_process:
                    logger.warning(log_message)
                time.sleep(1)

        # Set the `model_max_length` attribute of the tokenizer, if it hasn't been set,
        # to ensure that truncation is done correctly
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
            tokenizer.model_max_length = 512

        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.config.model.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=extractor, tokenizer=tokenizer
        )

        return self.processor

    def load_saved(self) -> PreTrainedModelData:
        """Return the saved model data for the model.

        Returns:
            The model setup.

        Raises:
            FileNotFoundError:
                If the model was trained with a language model decoder, but the language
                model decoder was not found.
        """
        if Path(self.config.model_dir).exists():
            model_path = self.config.model_dir
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        processor: Wav2Vec2Processor | Wav2Vec2ProcessorWithLM
        if self.config.model.decoder is not None:
            try:
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                    model_path, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
                )
            except (FileNotFoundError, ValueError):
                raise FileNotFoundError(
                    "The model was trained with a language model decoder, but the "
                    "language model decoder was not found."
                )
        else:
            processor_or_tup = Wav2Vec2Processor.from_pretrained(
                model_path, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
            )
            assert not isinstance(processor_or_tup, tuple)
            processor = processor_or_tup

        model_or_tup = Wav2Vec2ForCTC.from_pretrained(
            model_path, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        assert isinstance(model_or_tup, Wav2Vec2ForCTC)
        model = model_or_tup

        data_collator = DataCollatorCTCWithPadding(
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
        """Return the trainer class for the model."""
        return Trainer

    def load_training_arguments(self) -> TrainingArguments:
        """Return the training arguments for the model."""
        return TrainingArguments(
            **self._training_arguments_kwargs(
                sequence_to_sequence=False, include_max_grad_norm=True
            )
        )


def dump_vocabulary(config: DictConfig) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    It will dump the file to `${config.model_dir}/vocab.json`.

    Args:
        config:
            The Hydra configuration object.
    """
    # Build the set of all unique characters in the dataset
    unique_characters: set[str] = set(config.model.characters_to_keep + "|")
    sorted_unique_characters: list[str] = sorted(unique_characters)

    # Build vocabulary
    vocab = {char: idx for idx, char in enumerate(sorted_unique_characters)}

    # Dump the vocabulary to a json file
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f)
