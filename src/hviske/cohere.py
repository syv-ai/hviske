"""Model setup and training utilities for Cohere ASR models."""

import logging
import os
import sys
import typing as t
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Type

import numpy as np
import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    AutoConfig,
    CohereAsrForConditionalGeneration,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers import CohereAsrProcessor as TransformersCohereAsrProcessor
from transformers.integrations import is_deepspeed_zero3_enabled, is_fsdp_managed_module
from transformers.pipelines import pipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
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


class CohereASRTranscriber:
    """Run native Cohere ASR with its prompt-aware processor."""

    def __init__(
        self,
        model: CohereAsrForConditionalGeneration,
        processor: TransformersCohereAsrProcessor,
        device: torch.device,
        language: str = "da",
        punctuation: bool = True,
    ) -> None:
        """Initialise a native Cohere transcriber.

        Args:
            model:
                The native Cohere ASR model.
            processor:
                The native Cohere ASR processor.
            device:
                Device on which inference should run.
            language (optional):
                Language code used to construct the decoder prompt. Defaults to ``da``.
            punctuation (optional):
                Whether punctuation should be enabled. Defaults to ``True``.
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.language = language
        self.punctuation = punctuation

    def __call__(
        self, inputs: object, batch_size: int = 1, **kwargs: object
    ) -> dict[str, str] | Iterator[dict[str, str]]:
        """Transcribe one audio input or an iterable of audio inputs.

        ``AutomaticSpeechRecognitionPipeline`` cannot construct Cohere's decoder
        prompt because its generic preprocessing call does not provide a language.
        This small adapter keeps that model-specific contract in one place.

        Returns:
            A transcription dictionary for one input or an iterator of dictionaries.

        Raises:
            TypeError:
                If inputs are neither audio nor an iterable of audio inputs.
            ValueError:
                If batch_size is less than one.
        """
        del kwargs
        if isinstance(inputs, (dict, np.ndarray, torch.Tensor)) or (
            isinstance(inputs, list)
            and (not inputs or isinstance(inputs[0], (float, int)))
        ):
            return self._transcribe_batch(batch=[inputs])[0]
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Iterable):
            raise TypeError("Cohere ASR inputs must be audio or an iterable of audio.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        return self._transcribe_iter(inputs=inputs, batch_size=batch_size)

    def _transcribe_iter(
        self, inputs: Iterable[object], batch_size: int
    ) -> Iterator[dict[str, str]]:
        batch: list[object] = []
        for audio in inputs:
            batch.append(audio)
            if len(batch) == batch_size:
                yield from self._transcribe_batch(batch=batch)
                batch = []
        if batch:
            yield from self._transcribe_batch(batch=batch)

    def _transcribe_batch(self, batch: list[object]) -> list[dict[str, str]]:
        audio_values: list[np.ndarray | torch.Tensor | list[float]] = []
        sampling_rates: set[int] = set()
        feature_extractor = getattr(self.processor, "feature_extractor")
        expected_rate = int(getattr(feature_extractor, "sampling_rate"))
        for item in batch:
            if isinstance(item, dict):
                raw_audio = item.get("array", item.get("raw"))
                sampling_rate = item.get("sampling_rate", expected_rate)
            else:
                raw_audio = item
                sampling_rate = expected_rate
            if raw_audio is None or not isinstance(sampling_rate, (int, np.integer)):
                raise ValueError(
                    "Each Cohere audio input needs audio and sampling_rate."
                )
            audio_values.append(
                t.cast(np.ndarray | torch.Tensor | list[float], raw_audio)
            )
            sampling_rates.add(int(sampling_rate))
        if len(sampling_rates) != 1:
            raise ValueError(
                "All audio inputs in a Cohere batch must share a sampling rate."
            )
        sampling_rate = sampling_rates.pop()
        processed = self.processor(
            audio_values,
            language=self.language,
            punctuation=self.punctuation,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        chunk_index = t.cast(
            list[tuple[int, int | None]], processed["audio_chunk_index"]
        )
        generation_inputs = {
            "input_features": processed["input_features"].to(self.device),
            "attention_mask": processed["attention_mask"].to(self.device),
            "decoder_input_ids": processed["decoder_input_ids"].to(self.device),
        }
        generate = t.cast(Callable[..., object], self.model.generate)
        with torch.no_grad():
            generated = generate(**generation_inputs)
        sequences = (
            generated
            if isinstance(generated, torch.Tensor)
            else t.cast(torch.Tensor, getattr(t.cast(object, generated), "sequences"))
        )
        decoded = self.processor.decode(
            sequences,
            skip_special_tokens=True,
            audio_chunk_index=chunk_index,
            language=self.language,
        )
        texts = [decoded] if isinstance(decoded, str) else list(decoded)
        if len(texts) != len(batch):
            raise RuntimeError(
                "Native Cohere decoding returned a different number of outputs than "
                "the input batch."
            )
        return [dict(text=text) for text in texts]


def load_asr_transcriber(
    model_id: str,
    no_lm: bool,
    device: torch.device,
    language: str = "da",
    punctuation: bool = True,
) -> AutomaticSpeechRecognitionPipeline | CohereASRTranscriber:
    """Load a model-aware ASR transcriber.

    Native Cohere checkpoints use :class:`CohereASRTranscriber`; Whisper and Wav2Vec2
    retain the standard Transformers pipeline and its existing generation arguments.

    Args:
        model_id:
            The model ID to load.
        no_lm:
            Whether to load the Wav2Vec2 pipeline without a language model.
        device:
            Device on which to run inference.
        language (optional):
            Language code for native Cohere prompts. Defaults to ``da``.
        punctuation (optional):
            Whether native Cohere should produce punctuation. Defaults to ``True``.

    Returns:
        A native Cohere adapter or a standard Transformers ASR pipeline.
    """
    if not no_lm:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        if getattr(config, "model_type", None) == "cohere_asr":
            processor = CohereAsrProcessor.from_pretrained(
                model_id, trust_remote_code=False
            )
            model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, trust_remote_code=False
            )
            t.cast(Callable[..., object], model.to)(device)
            return CohereASRTranscriber(
                model=model,
                processor=processor,
                device=device,
                language=language,
                punctuation=punctuation,
            )
    if no_lm:
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer")
        feature_extractor = getattr(processor, "feature_extractor")
        return pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
        )
    return pipeline(task="automatic-speech-recognition", model=model_id, device=device)


def get_asr_call_kwargs(
    transcriber: AutomaticSpeechRecognitionPipeline | CohereASRTranscriber,
) -> dict[str, object]:
    """Return generation arguments compatible with the selected ASR model."""
    if isinstance(transcriber, CohereASRTranscriber):
        return {}
    return {"generate_kwargs": {"language": "danish", "task": "transcribe"}}


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
        tokenizer = getattr(self, "tokenizer")
        vocabulary = tokenizer.get_vocab()
        language_id = tokenizer.convert_tokens_to_ids(language_token)
        unknown_id = tokenizer.unk_token_id
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
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return [int(token_id) for token_id in token_ids]


class CohereSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2Seq trainer that keeps the Cohere prompt during evaluation."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        """Compute loss without forwarding the collator-only prompt length.

        Returns:
            The model loss, optionally paired with model outputs.
        """
        model_inputs = {
            key: value for key, value in inputs.items() if key != "prompt_length"
        }
        return super().compute_loss(
            model=model,
            inputs=model_inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

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

        Raises:
            ValueError:
                If the collated prompt contract is missing or inconsistent.
        """
        if (
            not getattr(self.args, "predict_with_generate", False)
            or prediction_loss_only
        ):
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
        prompt_lengths = prepared_inputs.get("prompt_length")
        decoder_input_ids = prepared_inputs.get("decoder_input_ids")
        decoder_attention_mask = prepared_inputs.get("decoder_attention_mask")
        if prompt_lengths is None or decoder_input_ids is None:
            raise ValueError(
                "Cohere evaluation inputs must include decoder_input_ids and "
                "the collator's prompt_length contract."
            )
        if decoder_attention_mask is None:
            raise ValueError(
                "Cohere evaluation inputs must include decoder_attention_mask."
            )
        if prompt_lengths.ndim != 1 or not torch.all(
            prompt_lengths == prompt_lengths[0]
        ):
            raise ValueError(
                "Cohere prompts must have one shared prompt length per batch."
            )
        prompt_length = int(prompt_lengths[0].item())
        if prompt_length <= 0 or prompt_length > decoder_input_ids.shape[1]:
            raise ValueError(
                "Cohere prompt_length does not describe decoder_input_ids: "
                f"{prompt_length} for width {decoder_input_ids.shape[1]}."
            )
        generation_inputs = {
            key: value
            for key, value in prepared_inputs.items()
            if key
            not in {
                "labels",
                "prompt_length",
                "decoder_input_ids",
                "decoder_attention_mask",
            }
        }
        generation_inputs["decoder_input_ids"] = decoder_input_ids[:, :prompt_length]
        generation_inputs["decoder_attention_mask"] = decoder_attention_mask[
            :, :prompt_length
        ]
        model_inputs = {
            key: value
            for key, value in prepared_inputs.items()
            if key != "prompt_length"
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
                    outputs = cohere_model(**model_inputs)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    loss = outputs[0]
                loss = loss.detach().mean()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return t.cast(float | None, loss), None, None

        labels = model_inputs.get("labels") if has_labels else None
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
