"""Model setup and training utilities for Cohere ASR models."""

import logging
import os
import sys
import typing as t
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Type

import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import CohereAsrForConditionalGeneration
from transformers import CohereAsrProcessor as TransformersCohereAsrProcessor
from transformers.integrations import is_deepspeed_zero3_enabled, is_fsdp_managed_module
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import EvalPrediction, SchedulerType
from transformers.training_args import OptimizerNames, TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from .compute_metrics import compute_error_rate_metrics
from .data_collators import DataCollatorCohereWithPadding
from .data_models import ModelSetup, PreTrainedModelData
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


class CohereAsrProcessor(TransformersCohereAsrProcessor):
    """Native Cohere processor with checkpoint-aware language validation.

    Transformers currently keeps its list of languages in a module constant. Some
    fine-tuned checkpoints add a language token without changing that constant, so
    validation is intentionally based on the tokenizer actually loaded with the
    checkpoint instead.
    """

    def get_decoder_prompt_ids(
        self, language: str, punctuation: bool = True
    ) -> list[int]:
        """Build a prompt when the checkpoint has a real token for ``language``.

        Args:
            language:
                The ISO language code represented by the checkpoint.
            punctuation (optional):
                Whether punctuation should be enabled. Defaults to ``True``.

        Returns:
            The decoder prompt token IDs.

        Raises:
            ValueError:
                If the tokenizer does not contain the requested language token.
        """
        language_token = f"<|{language}|>"
        vocabulary = self.tokenizer.get_vocab()
        language_id = self.tokenizer.convert_tokens_to_ids(language_token)
        unknown_id = self.tokenizer.unk_token_id
        if (
            language_token not in vocabulary
            or language_id is None
            or language_id == unknown_id
        ):
            raise ValueError(
                f"Language {language!r} is not represented by this Cohere ASR "
                "checkpoint's tokenizer."
            )

        punctuation_token = "<|pnc|>" if punctuation else "<|nopnc|>"
        tokens = [
            "▁",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            language_token,
            language_token,
            punctuation_token,
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        return [
            int(token_id) for token_id in self.tokenizer.convert_tokens_to_ids(tokens)
        ]


class CohereSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2Seq trainer that keeps the Cohere prompt during evaluation."""

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        **gen_kwargs: object,
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
        """Generate from prompt IDs while retaining labels for the loss.

        Returns:
            The loss, generated tokens and labels, respectively.
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )

        has_labels = "labels" in inputs
        prepared_inputs = t.cast(dict[str, torch.Tensor], self._prepare_inputs(inputs))
        cohere_model = t.cast(CohereAsrForConditionalGeneration, model)
        generation_inputs = {
            key: value for key, value in prepared_inputs.items() if key != "labels"
        }
        if hasattr(self, "_gen_kwargs") and not gen_kwargs:
            gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("num_beams") is None:
            gen_kwargs.pop("num_beams", None)
        if gen_kwargs.get("max_length") is None:
            gen_kwargs.pop("max_length", None)
        gen_kwargs["synced_gpus"] = gen_kwargs.get(
            "synced_gpus",
            is_deepspeed_zero3_enabled()
            or is_fsdp_managed_module(t.cast(torch.nn.Module, self.model)),
        )

        # Unlike the generic trainer, never remove decoder_input_ids when their shape
        # happens to match labels: those IDs are the Cohere language/punctuation prompt.
        generate = t.cast(t.Callable[..., object], cohere_model.generate)
        generated_tokens = t.cast(
            torch.Tensor, generate(**generation_inputs, **gen_kwargs)
        )

        generation_config = cohere_model.generation_config
        if generation_config.max_length is not None:
            if generated_tokens.shape[-1] < generation_config.max_length:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, generation_config.max_length
                )
            elif (
                generation_config.max_new_tokens is not None
                and generated_tokens.shape[-1] < generation_config.max_new_tokens + 1
            ):
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, generation_config.max_new_tokens + 1
                )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = cohere_model(**prepared_inputs)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    loss = outputs[0]
                loss = loss.detach().mean()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return t.cast(float | None, loss), None, None

        labels = prepared_inputs.get("labels") if has_labels else None
        if labels is not None and generation_config.max_length is not None:
            if labels.shape[-1] < generation_config.max_length:
                labels = self._pad_tensors_to_max_len(
                    labels, generation_config.max_length
                )
            elif (
                generation_config.max_new_tokens is not None
                and labels.shape[-1] < generation_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, generation_config.max_new_tokens + 1
                )
        return t.cast(float | None, loss), generated_tokens, labels


class CohereModelSetup(ModelSetup):
    """Model setup for native Transformers Cohere ASR models."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        self.config = config
        self.processor: CohereAsrProcessor
        self.is_main_process = os.getenv("RANK", "0") == "0"

    def load_processor(self) -> CohereAsrProcessor:
        """Load the native processor without remote Python code.

        Returns:
            The checkpoint processor.

        Raises:
            TypeError:
                If the checkpoint is not a native Cohere processor.
        """
        processor = CohereAsrProcessor.from_pretrained(
            self.config.model.pretrained_model_id,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            trust_remote_code=False,
        )
        if not isinstance(processor, CohereAsrProcessor):
            raise TypeError(
                "The checkpoint did not load as a native Cohere ASR processor."
            )
        self.processor = processor
        return processor

    def load_model(self) -> CohereAsrForConditionalGeneration:
        """Load the native Cohere ASR model.

        Returns:
            The native Cohere model.
        """
        with transformers_output_ignored():
            model = CohereAsrForConditionalGeneration.from_pretrained(
                self.config.model.pretrained_model_id,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                trust_remote_code=False,
            )
        if self.config.model.freeze_feature_encoder:
            encoder = model.model.encoder
            for parameter in encoder.parameters():
                parameter.requires_grad = False

        # Gradient checkpointing and the decoder cache are incompatible.
        model.config.use_cache = False
        return model

    def load_data_collator(self) -> DataCollatorCohereWithPadding:
        """Return the Cohere prompt-aware data collator."""
        return DataCollatorCohereWithPadding(
            processor=self.processor,
            padding=self.config.padding,
            max_length=self.config.model.max_length,
        )

    def load_trainer_class(self) -> Type[Trainer]:
        """Return the Cohere evaluation trainer."""
        return CohereSeq2SeqTrainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        """Return the error-rate metric function."""
        return partial(compute_error_rate_metrics, processor=self.processor)

    def load_training_arguments(self) -> TrainingArguments:
        """Build the common training configuration for Cohere.

        Returns:
            The sequence-to-sequence training arguments.
        """
        num_devices = max(torch.cuda.device_count(), 1)
        per_device_total_batch_size = self.config.total_batch_size // num_devices
        gradient_accumulation_steps = (
            per_device_total_batch_size // self.config.per_device_batch_size
        )
        if gradient_accumulation_steps == 0:
            gradient_accumulation_steps = 1

        fp16 = False
        bf16 = False
        if not mps_is_available():
            if self.config.bf16_allowed and torch.cuda.is_bf16_supported():
                bf16 = True
            elif self.config.fp16_allowed and torch.cuda.is_available():
                fp16 = True

        if self.config.early_stopping:
            self.config.save_total_limit = max(self.config.save_total_limit, 1)
        metric_name = (
            (
                f"val_{self.config.evaluation_datasets[0].id.split('/')[-1]}_"
                f"{self.config.evaluation_datasets[0].subset}_cer"
            )
            .lower()
            .replace("-", "_")
        )

        return Seq2SeqTrainingArguments(
            output_dir=self.config.model_dir,
            hub_model_id=f"{self.config.hub_organisation}/{self.config.model_id}",
            hub_private_repo=self.config.private,
            per_device_train_batch_size=self.config.per_device_batch_size,
            per_device_eval_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.config.model.learning_rate,
            lr_scheduler_type=SchedulerType.COSINE,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            fp16=fp16,
            bf16=bf16,
            push_to_hub=False,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_strategy="no" if self.config.save_total_limit == 0 else "steps",
            logging_steps=self.config.logging_steps,
            length_column_name="input_length",
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs=dict(use_reentrant=False),
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.early_stopping,
            metric_for_best_model=metric_name,
            greater_is_better=False,
            seed=self.config.seed,
            remove_unused_columns=False,
            optim=OptimizerNames.ADAMW_TORCH,
            adam_beta1=self.config.adam_first_momentum,
            adam_beta2=self.config.adam_second_momentum,
            report_to=[self.config.experiment_tracking.type]
            if self.config.enable_experiment_tracking
            else [],
            ignore_data_skip=self.config.ignore_data_skip,
            predict_with_generate=True,
            generation_max_length=self.config.model.max_length,
            use_cpu=hasattr(sys, "_called_from_test"),
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_drop_last=True,
            ddp_find_unused_parameters=False,
            accelerator_config=AcceleratorConfig(dispatch_batches=False).to_dict(),
        )

    def load_saved(self) -> PreTrainedModelData:
        """Load a saved native Cohere model and its processing objects.

        Returns:
            The saved model, processor, collator and metric function.

        Raises:
            TypeError:
                If the saved checkpoint contains non-native model objects.
        """
        if Path(self.config.model_dir).exists():
            model_path = self.config.model_dir
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        processor = CohereAsrProcessor.from_pretrained(
            model_path,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            trust_remote_code=False,
        )
        model = CohereAsrForConditionalGeneration.from_pretrained(
            model_path,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            trust_remote_code=False,
        )
        if not isinstance(processor, CohereAsrProcessor):
            raise TypeError(
                "The saved checkpoint did not contain a native Cohere processor."
            )
        if not isinstance(model, CohereAsrForConditionalGeneration):
            raise TypeError(
                "The saved checkpoint did not contain a native Cohere model."
            )
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=DataCollatorCohereWithPadding(
                processor=processor,
                padding=self.config.padding,
                max_length=self.config.model.max_length,
            ),
            compute_metrics=partial(compute_error_rate_metrics, processor=processor),
        )
