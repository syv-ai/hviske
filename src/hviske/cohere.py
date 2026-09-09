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
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
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
from .data_collators import (
    DataCollatorCohereWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from .data_models import ModelSetup, PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
COHERE_MODEL_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"


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


class CohereModelSetup(ModelSetup):
    """Model setup for native and remote-code Cohere ASR models."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        self.config = config
        self.processor: CohereAsrProcessor
        self.is_main_process = os.getenv("RANK", "0") == "0"

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        """Return the error-rate metric function."""
        return partial(compute_error_rate_metrics, processor=self.processor)

    def load_data_collator(
        self,
    ) -> DataCollatorCohereWithPadding | DataCollatorSpeechSeq2SeqWithPadding:
        """Return the data collator for the selected Cohere implementation."""
        return DataCollatorCohereWithPadding(
            processor=self.processor,
            padding=self.config.padding,
            max_length=self.config.model.max_length,
        )

    def load_model(self) -> CohereAsrForConditionalGeneration:
        """Load the configured Cohere ASR model.

        Returns:
            The loaded Cohere model.

        Raises:
            TypeError:
                If remote code does not expose the required model capabilities.
        """
        if self._uses_remote_code():
            with transformers_output_ignored():
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.model.pretrained_model_id, **self._pretrained_kwargs()
                )
            if not all(
                callable(getattr(model, name, None))
                for name in ("forward", "generate", "save_pretrained")
            ):
                raise TypeError(
                    "The remote Cohere model must support forward, generate and "
                    "save_pretrained."
                )
            if self.config.model.freeze_feature_encoder:
                self._freeze_encoder(model)
        else:
            with transformers_output_ignored():
                model = CohereAsrForConditionalGeneration.from_pretrained(
                    self.config.model.pretrained_model_id,
                    token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                    trust_remote_code=False,
                    revision=str(self.config.model.revision),
                )
            if self.config.model.freeze_feature_encoder:
                encoder = model.model.encoder
                for parameter in encoder.parameters():
                    parameter.requires_grad = False

        # Gradient checkpointing and the decoder cache are incompatible.
        model.config.use_cache = False
        return t.cast(CohereAsrForConditionalGeneration, model)

    @staticmethod
    def _freeze_encoder(model: object) -> None:
        get_encoder = getattr(model, "get_encoder", None)
        encoder = None
        if callable(get_encoder):
            try:
                encoder = get_encoder()
            except (AttributeError, NotImplementedError):
                encoder = None
        if encoder is None:
            encoder = getattr(model, "encoder", None)
        if encoder is None:
            model_body = getattr(model, "model", None)
            encoder = getattr(model_body, "encoder", None)
        parameters = getattr(encoder, "parameters", None)
        if not callable(parameters):
            raise TypeError("The remote Cohere model has no accessible encoder.")
        for parameter in parameters():
            parameter.requires_grad = False

    def _pretrained_kwargs(self) -> dict[str, str | bool]:
        kwargs: dict[str, str | bool] = {
            "token": os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            "trust_remote_code": self._uses_remote_code(),
        }
        revision = self.config.model.get("revision")
        if revision is not None:
            kwargs["revision"] = str(revision)
        return kwargs

    def _uses_remote_code(self) -> bool:
        return bool(self.config.model.get("trust_remote_code", False))

    def load_processor(self) -> CohereAsrProcessor:
        """Load the configured Cohere processor.

        Returns:
            The checkpoint processor.

        Raises:
            TypeError:
                If the checkpoint does not expose the required processing API.
        """
        if self._uses_remote_code():
            processor = AutoProcessor.from_pretrained(
                self.config.model.pretrained_model_id, **self._pretrained_kwargs()
            )
            self._validate_remote_processor(processor)
            processor = RemoteCohereAsrProcessor(processor=processor)
        else:
            processor = CohereAsrProcessor.from_pretrained(
                self.config.model.pretrained_model_id,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                trust_remote_code=False,
                revision=str(self.config.model.revision),
            )
            if not isinstance(processor, CohereAsrProcessor):
                raise TypeError(
                    "The checkpoint did not load as a native Cohere ASR processor."
                )
        self.processor = t.cast(CohereAsrProcessor, processor)
        return self.processor

    @staticmethod
    def _validate_remote_processor(processor: object) -> None:
        required = ("feature_extractor", "tokenizer", "batch_decode")
        missing = [name for name in required if not hasattr(processor, name)]
        if not callable(processor):
            missing.append("__call__")
        if missing:
            raise TypeError(
                "The remote Cohere processor is missing required capabilities: "
                + ", ".join(missing)
            )

    def load_saved(
        self, revision: str | None = None, saved_model_revision: str | None = None
    ) -> PreTrainedModelData:
        """Load a saved Cohere model and its processing objects.

        Args:
            revision (optional):
                Immutable revision of the saved Hub checkpoint.
            saved_model_revision (optional):
                Explicit alias for ``revision``. Defaults to the saved-model revision
                in the configuration when present.

        Returns:
            The saved model, processor, collator and metric function.

        Raises:
            TypeError:
                If the saved checkpoint does not expose the required API.
            ValueError:
                If both saved-checkpoint revision arguments disagree.
        """
        if (
            revision is not None
            and saved_model_revision is not None
            and revision != saved_model_revision
        ):
            raise ValueError("Saved model revisions must agree")
        requested_revision = saved_model_revision or revision
        if requested_revision is None:
            requested_revision = self.config.get("saved_model_revision")
        if requested_revision is None:
            requested_revision = self.config.get("model_revision")
        if requested_revision is None:
            requested_revision = self.config.model.get("saved_model_revision")

        local_path = Path(self.config.model_dir).exists()
        if local_path:
            model_path = self.config.model_dir
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        pretrained_kwargs: dict[str, str | bool] = {
            "token": os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            "trust_remote_code": self._uses_remote_code(),
        }
        if not local_path and requested_revision is not None:
            pretrained_kwargs["revision"] = str(requested_revision)

        if self._uses_remote_code():
            processor = AutoProcessor.from_pretrained(model_path, **pretrained_kwargs)
            self._validate_remote_processor(processor)
            processor = RemoteCohereAsrProcessor(processor=processor)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path, **pretrained_kwargs
            )
            if not all(
                callable(getattr(model, name, None))
                for name in ("forward", "generate", "save_pretrained")
            ):
                raise TypeError(
                    "The saved remote Cohere model has an incomplete model API."
                )
        else:
            token = os.getenv("HUGGINGFACE_HUB_TOKEN", True)
            if not local_path and requested_revision is not None:
                processor = CohereAsrProcessor.from_pretrained(
                    model_path,
                    token=token,
                    trust_remote_code=False,
                    revision=str(requested_revision),
                )
                model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_path,
                    token=token,
                    trust_remote_code=False,
                    revision=str(requested_revision),
                )
            else:
                processor = CohereAsrProcessor.from_pretrained(
                    model_path, token=token, trust_remote_code=False
                )
                model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_path, token=token, trust_remote_code=False
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
            processor=t.cast(Processor, processor),
            model=model,
            data_collator=DataCollatorCohereWithPadding(
                processor=t.cast(Processor, processor),
                padding=self.config.padding,
                max_length=self.config.model.max_length,
            ),
            compute_metrics=partial(
                compute_error_rate_metrics, processor=t.cast(Processor, processor)
            ),
        )

    def load_trainer_class(self) -> Type[Trainer]:
        """Return the prompt-aware Cohere trainer."""
        return CohereSeq2SeqTrainer

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
            eval_steps=(
                1 if self.config.get("evaluation_steps") else self.config.eval_steps
            ),
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


class RemoteCohereAsrProcessor:
    """Adapt the revision-pinned remote processor to the Cohere training API.

    The checkpoint's remote processor only extracts audio features and tokenises
    text.  Its model nevertheless expects the Cohere decoder prompt, so keeping
    that prompt construction here avoids relying on remote processor behaviour or
    tokenising a concatenated string (which splits special tokens incorrectly).
    """

    uses_length = True

    def __init__(self, processor: object) -> None:
        """Wrap a remote processor.

        Args:
            processor:
                The processor loaded from the checkpoint.
        """
        self._processor = processor
        self.feature_extractor = getattr(processor, "feature_extractor")
        self.tokenizer = getattr(processor, "tokenizer")

    def __call__(
        self,
        audio: object,
        language: str,
        text: str | None = None,
        punctuation: bool = True,
        sampling_rate: int | None = None,
        **kwargs: object,
    ) -> t.MutableMapping[str, object]:
        """Extract remote features and add native-style prompt and labels.

        Args:
            audio:
                Audio waveform accepted by the remote processor.
            language:
                ISO language code used to construct the decoder prompt.
            text (optional):
                Transcript to tokenise as labels. Defaults to ``None``.
            punctuation (optional):
                Whether punctuation should be enabled. Defaults to ``True``.
            sampling_rate (optional):
                Audio sampling rate. Defaults to ``None``.
            **kwargs:
                Additional feature-extractor arguments.

        Returns:
            Remote audio features with Cohere prompt IDs and optional labels.
        """
        process = t.cast(Callable[..., object], self._processor)
        processed = t.cast(
            t.MutableMapping[str, object],
            process(audio=audio, sampling_rate=sampling_rate, **kwargs),
        )
        prompt_ids = self.get_decoder_prompt_ids(
            language=language, punctuation=punctuation
        )
        batch_size = len(t.cast(t.Sized, processed["input_features"]))
        processed["decoder_input_ids"] = torch.tensor(
            [prompt_ids] * batch_size, dtype=torch.long
        )
        if text is not None:
            tokenise = t.cast(
                Callable[..., t.MutableMapping[str, object]], self.tokenizer
            )
            tokenised = tokenise(text=text, truncation=True)
            token_ids = t.cast(
                list[int] | list[list[int]] | torch.Tensor, tokenised["input_ids"]
            )
            if isinstance(token_ids, torch.Tensor):
                if token_ids.ndim == 1:
                    token_ids = token_ids.unsqueeze(0)
            elif token_ids and isinstance(token_ids[0], int):
                token_ids = [token_ids]
            processed["labels"] = token_ids
        return processed

    def get_decoder_prompt_ids(
        self, language: str, punctuation: bool = True
    ) -> list[int]:
        """Build the exact prompt expected by the Cohere decoder.

        Returns:
            The decoder prompt token IDs.
        """
        return CohereAsrProcessor.get_decoder_prompt_ids(
            t.cast(CohereAsrProcessor, self), language=language, punctuation=punctuation
        )

    def __getattr__(self, name: str) -> object:
        """Delegate decoding and persistence helpers to the remote processor.

        Returns:
            The delegated remote processor attribute.
        """
        return getattr(self._processor, name)


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


class CohereASRTranscriber:
    """Run native Cohere ASR with its prompt-aware processor."""

    def __init__(
        self,
        model: CohereAsrForConditionalGeneration,
        processor: TransformersCohereAsrProcessor,
        device: torch.device,
        language: str = "da",
        punctuation: bool = True,
        max_new_tokens: int = 256,
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
            max_new_tokens (optional):
                Maximum number of tokens generated per audio input. Defaults to ``256``.

        Raises:
            ValueError:
                If ``max_new_tokens`` is less than one.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one.")
        self.model = model
        self.processor = processor
        object.__setattr__(self, "_device", device)
        self.language = language
        self.punctuation = punctuation
        self.max_new_tokens = max_new_tokens

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
        process = t.cast(Callable[..., object], self.processor)
        processed = t.cast(
            dict[str, object],
            process(
                audio_values,
                language=self.language,
                punctuation=self.punctuation,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ),
        )
        chunk_index = t.cast(
            list[tuple[int, int | None]], processed["audio_chunk_index"]
        )
        generation_inputs = {
            "input_features": t.cast(torch.Tensor, processed["input_features"]).to(
                self._device
            ),
            "attention_mask": t.cast(torch.Tensor, processed["attention_mask"]).to(
                self._device
            ),
            "decoder_input_ids": t.cast(
                torch.Tensor, processed["decoder_input_ids"]
            ).to(self._device),
        }
        generate = t.cast(Callable[..., object], self.model.generate)
        with torch.no_grad():
            generated = generate(
                **generation_inputs, max_new_tokens=self.max_new_tokens
            )
        sequences = (
            generated
            if isinstance(generated, torch.Tensor)
            else t.cast(torch.Tensor, getattr(generated, "sequences"))
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


def get_asr_call_kwargs(
    transcriber: AutomaticSpeechRecognitionPipeline | CohereASRTranscriber,
) -> dict[str, object]:
    """Return generation arguments compatible with the selected ASR model."""
    if isinstance(transcriber, CohereASRTranscriber):
        return {}
    return {"generate_kwargs": {"language": "danish", "task": "transcribe"}}


def load_asr_transcriber(
    model_id: str,
    no_lm: bool,
    device: torch.device,
    language: str = "da",
    punctuation: bool = True,
    max_new_tokens: int = 256,
    revision: str | None = None,
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
        max_new_tokens (optional):
            Maximum number of tokens generated per audio input. Defaults to ``256``.
        revision (optional):
            Immutable Hub revision for a native Cohere checkpoint. Defaults to the
            pinned official Cohere checkpoint revision when ``model_id`` is that base
            checkpoint. Other Hub models are loaded without a revision unless one is
            explicitly supplied.

    Returns:
        A native Cohere adapter or a standard Transformers ASR pipeline.

    Raises:
        ValueError:
            If ``max_new_tokens`` is less than one.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least one.")
    if not no_lm:
        if revision is None and model_id == COHERE_MODEL_ID:
            revision = COHERE_MODEL_REVISION
        if revision is None:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        else:
            config = AutoConfig.from_pretrained(
                model_id, trust_remote_code=False, revision=revision
            )
        if getattr(config, "model_type", None) == "cohere_asr":
            if revision is None:
                processor = CohereAsrProcessor.from_pretrained(
                    model_id, trust_remote_code=False
                )
                model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_id, trust_remote_code=False
                )
            else:
                processor = CohereAsrProcessor.from_pretrained(
                    model_id, trust_remote_code=False, revision=revision
                )
                model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_id, trust_remote_code=False, revision=revision
                )
            t.cast(Callable[..., object], model.to)(device)
            return CohereASRTranscriber(
                model=model,
                processor=processor,
                device=device,
                language=language,
                punctuation=punctuation,
                max_new_tokens=max_new_tokens,
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
