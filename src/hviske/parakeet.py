"""Model setup for Transformers-native NVIDIA Parakeet models."""

import collections.abc as c
import logging
import os
import typing as t
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import extract_model_from_parallel
from omegaconf import DictConfig
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCTC,
    AutoModelForTDT,
    AutoProcessor,
    PreTrainedConfig,
    PreTrainedModel,
    Trainer,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.parakeet.modeling_parakeet import (
    ParakeetRNNTDecoder,
    ParakeetRNNTJointNetwork,
)
from transformers.trainer_utils import EvalPrediction

from .compute_metrics import compute_error_rate_metrics
from .data_models import PreTrainedModelData, Processor
from .dataloader_shutdown import start_worker_shutdown_watcher
from .wav2vec2 import Wav2Vec2ModelSetup

logger = logging.getLogger(__package__)


ParakeetFamily: t.TypeAlias = t.Literal["ctc", "rnnt", "tdt"]


_TRANSDUCER_FAMILIES = frozenset(("rnnt", "tdt"))


class ParakeetGenerationTrainer(Trainer):
    """Trainer prediction step for Parakeet transducer generation outputs."""

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
        generation_model = extract_model_from_parallel(model)
        generate = t.cast(t.Callable[..., object], generation_model.generate)
        generated = generate(**generation_inputs)
        sequences = t.cast(torch.Tensor, getattr(generated, "sequences", generated))
        return loss, sequences.detach(), t.cast(torch.Tensor | None, labels)


@dataclass
class DataCollatorParakeetWithPadding(DataCollatorMixin):
    """Pad Parakeet frame features, decoder inputs, and labels.

    Parakeet receives log-mel features shaped ``(frames, feature_size)``.  The
    feature extractor knows the feature size and frame padding value, so this
    collator intentionally delegates padding rather than assuming a mel width.
    Native RNNT and TDT decoder inputs are validated against the model's blank ID
    before padding; tokenizer padding semantics are otherwise preserved.
    """

    processor: Processor
    sample_rate: int
    padding: bool | str
    return_tensors: str = "pt"
    model_config: object | None = None

    def __post_init__(self) -> None:
        """Reject frame padding without an explicit Parakeet frame length.

        Raises:
            ValueError:
                If ``padding`` is ``"max_length"``.
        """
        if self.padding == "max_length":
            raise ValueError(
                "Parakeet does not support padding='max_length': a frame max_length "
                "is required, but this configuration does not provide one. Use "
                "padding='longest' instead."
            )

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate preprocessed Parakeet features and padded labels.

        Args:
            features:
                Examples containing ``input_features`` and optionally an
                ``attention_mask``, plus token ID ``labels``. Transducer examples
                also contain ``decoder_input_ids``.

        Returns:
            A batch suitable for a native Parakeet model.

        Raises:
            ValueError:
                If examples do not contain preprocessed features or raw audio.
        """
        start_worker_shutdown_watcher()
        has_decoder_inputs = any("decoder_input_ids" in feature for feature in features)
        if has_decoder_inputs:
            if any("decoder_input_ids" not in feature for feature in features):
                raise ValueError(
                    "Every Parakeet transducer feature must contain decoder_input_ids."
                )
            for feature in features:
                validate_parakeet_transducer_inputs(
                    decoder_input_ids=feature["decoder_input_ids"],
                    labels=feature["labels"],
                    processor=self.processor,
                    model_config=self.model_config,
                )

        if "input_features" in features[0]:
            audio_features = [
                {
                    key: feature[key]
                    for key in ("input_features", "attention_mask")
                    if key in feature
                }
                for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                audio_features,
                padding=self.padding,
                return_attention_mask=True,
                return_tensors=self.return_tensors,
            )
        elif "audio" in features[0]:
            batch = self.processor.feature_extractor(
                [feature["audio"]["array"] for feature in features],
                sampling_rate=self.sample_rate,
                padding=self.padding,
                return_attention_mask=True,
                return_tensors=self.return_tensors,
            )
        else:
            raise ValueError(
                "Parakeet features must contain either 'input_features' or 'audio'."
            )

        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].long()

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors=self.return_tensors
        )
        batch["labels"] = labels_batch["input_ids"]

        if has_decoder_inputs:
            decoder_features = [
                {"input_ids": feature["decoder_input_ids"]} for feature in features
            ]
            decoder_batch = self.processor.tokenizer.pad(
                decoder_features,
                padding=self.padding,
                return_tensors=self.return_tensors,
            )
            batch["decoder_input_ids"] = decoder_batch["input_ids"]
        return batch


class ParakeetModelSetup(Wav2Vec2ModelSetup):
    """Model setup for Transformers-native NVIDIA Parakeet checkpoints.

    CTC checkpoints are loaded through ``AutoModelForCTC``.  RNNT and TDT use
    their native Transformers auto classes and share the transducer data path.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        super().__init__(config=config)
        self.processor: Processor
        self.model_config: PreTrainedConfig | None = None

    @staticmethod
    def _resize_rnnt_heads(model: PreTrainedModel, vocabulary_size: int) -> None:
        """Resize RNNT heads for backwards-compatible callers."""
        ParakeetModelSetup._resize_transducer_heads(
            model=model, vocabulary_size=vocabulary_size, family="rnnt"
        )

    @staticmethod
    def _resize_transducer_heads(
        model: PreTrainedModel, vocabulary_size: int, family: ParakeetFamily
    ) -> None:
        """Resize transducer heads while retaining token and duration rows.

        TDT's joint head contains token logits followed by one row per duration.
        Only the token portion is vocabulary-dependent; duration rows and the blank
        token ID must remain unchanged.

        Raises:
            ValueError:
                If the model does not expose the expected heads or the target
                vocabulary would discard existing rows.
        """
        decoder = t.cast(ParakeetRNNTDecoder | None, getattr(model, "decoder", None))
        joint = t.cast(ParakeetRNNTJointNetwork | None, getattr(model, "joint", None))
        if decoder is None or joint is None:
            raise ValueError(
                f"Native Parakeet {family.upper()} model does not provide "
                "decoder and joint heads."
            )
        embedding = decoder.embedding
        head = joint.head
        if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear):
            raise ValueError(
                f"Native Parakeet {family.upper()} model does not provide "
                "decoder and joint heads."
            )

        old_vocabulary_size = int(getattr(joint, "vocab_size", model.config.vocab_size))
        duration_count = (
            len(getattr(model.config, "durations", ())) if family == "tdt" else 0
        )
        expected_head_size = old_vocabulary_size + duration_count
        if head.out_features != expected_head_size:
            raise ValueError(
                f"Native Parakeet {family.upper()} joint head has an unexpected "
                f"size: {head.out_features} != {expected_head_size}."
            )
        if (
            embedding.num_embeddings > vocabulary_size
            or old_vocabulary_size > vocabulary_size
        ):
            raise ValueError(
                f"The adapted tokenizer is smaller than a Parakeet "
                f"{family.upper()} head."
            )

        target_head_size = vocabulary_size + duration_count
        if embedding.num_embeddings < vocabulary_size:
            new_embedding = nn.Embedding(
                num_embeddings=vocabulary_size,
                embedding_dim=embedding.embedding_dim,
                padding_idx=embedding.padding_idx,
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
            model._init_weights(new_embedding)
            with torch.no_grad():
                new_embedding.weight[: embedding.num_embeddings].copy_(embedding.weight)
            decoder.embedding = new_embedding

        if head.out_features < target_head_size:
            new_head = nn.Linear(
                in_features=head.in_features,
                out_features=target_head_size,
                bias=head.bias is not None,
                device=head.weight.device,
                dtype=head.weight.dtype,
            )
            model._init_weights(new_head)
            with torch.no_grad():
                new_head.weight[:old_vocabulary_size].copy_(
                    head.weight[:old_vocabulary_size]
                )
                if duration_count:
                    new_head.weight[vocabulary_size:].copy_(
                        head.weight[old_vocabulary_size:]
                    )
                if head.bias is not None and new_head.bias is not None:
                    new_head.bias[:old_vocabulary_size].copy_(
                        head.bias[:old_vocabulary_size]
                    )
                    if duration_count:
                        new_head.bias[vocabulary_size:].copy_(
                            head.bias[old_vocabulary_size:]
                        )
            joint.head = new_head

        joint.vocab_size = vocabulary_size
        model.config.vocab_size = vocabulary_size
        generation_config = getattr(model, "generation_config", None)
        blank_token_id = getattr(model.config, "blank_token_id", None)
        if generation_config is not None and blank_token_id is not None:
            generation_config.decoder_start_token_id = blank_token_id

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
        elif family == "tdt":
            model = AutoModelForTDT.from_pretrained(model_id, **kwargs)
        else:
            model = AutoModel.from_pretrained(model_id, **kwargs)
        self._resize_model_vocabulary(model=model, family=family)

        if self.config.model.get("freeze_feature_encoder", False):
            encoder = getattr(model, "encoder", None)
            if encoder is not None:
                for parameter in encoder.parameters():
                    parameter.requires_grad = False

        if self.config.gradient_checkpointing and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        self.model_config = model.config
        return model

    def _hub_kwargs(self, model_id: str | None = None) -> dict[str, str | bool]:
        """Return Hub arguments without applying a base revision to local paths.

        Args:
            model_id (optional):
                Checkpoint identifier or local directory. Defaults to the configured
                pretrained model.

        Returns:
            Arguments accepted by ``from_pretrained``.
        """
        checkpoint = model_id or str(self.config.model.pretrained_model_id)
        if Path(checkpoint).exists():
            return {}
        return {
            "token": os.getenv("HUGGINGFACE_HUB_TOKEN") or True,
            "revision": self._revision(),
        }

    def _revision(self) -> str:
        revision = self.config.model.get("revision")
        return str(revision) if revision is not None else "main"

    def _load_family(self, model_id: str) -> ParakeetFamily:
        try:
            config = AutoConfig.from_pretrained(
                model_id, **self._hub_kwargs(model_id=model_id)
            )
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_id!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported; use a "
                "checkpoint with config.json."
            ) from error
        return parakeet_family(config=config)

    def _resize_model_vocabulary(
        self, model: PreTrainedModel, family: ParakeetFamily
    ) -> None:
        """Resize Parakeet vocabulary-dependent modules after processor adaptation."""
        processor = getattr(self, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            return

        vocabulary_size = len(tokenizer)
        if family == "ctc":
            self._resize_ctc_head(model=model, vocabulary_size=vocabulary_size)
        else:
            self._resize_transducer_heads(
                model=model, vocabulary_size=vocabulary_size, family=family
            )

    @staticmethod
    def _resize_ctc_head(model: PreTrainedModel, vocabulary_size: int) -> None:
        """Resize a CTC head while retaining its pretrained rows.

        Raises:
            ValueError:
                If the model does not expose a CTC head or the target vocabulary
                would discard existing output rows.
        """
        head = getattr(model, "ctc_head", None)
        if not isinstance(head, nn.Conv1d):
            raise ValueError("Native Parakeet CTC model does not provide ctc_head.")
        if head.out_channels > vocabulary_size:
            raise ValueError("The adapted tokenizer is smaller than the CTC head.")
        if head.out_channels < vocabulary_size:
            new_head = type(head)(
                in_channels=head.in_channels,
                out_channels=vocabulary_size,
                kernel_size=head.kernel_size,
                stride=head.stride,
                padding=head.padding,
                dilation=head.dilation,
                groups=head.groups,
                bias=head.bias is not None,
                padding_mode=head.padding_mode,
                device=head.weight.device,
                dtype=head.weight.dtype,
            )
            model._init_weights(new_head)
            with torch.no_grad():
                new_head.weight[: head.out_channels].copy_(head.weight)
                if head.bias is not None and new_head.bias is not None:
                    new_head.bias[: head.out_channels].copy_(head.bias)
            model.ctc_head = new_head
        model.config.vocab_size = vocabulary_size

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
            processor = AutoProcessor.from_pretrained(
                model_id, **self._hub_kwargs(model_id=model_id)
            )
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
        family = self._load_family(model_id=model_id)
        if family in _TRANSDUCER_FAMILIES:
            setattr(processor, "decoder_type", family)
        self.processor = self._adapt_processor(processor=processor)
        return self.processor

    def _adapt_processor(self, processor: Processor) -> Processor:
        """Add retained characters that the native tokenizer cannot encode.

        Added tokens are deliberately ordinary tokens.  This leaves the native BPE
        vocabulary and every existing special-token ID untouched.

        Returns:
            The processor with the adapted tokenizer.
        """
        characters_to_keep = self.config.model.get("characters_to_keep")
        if characters_to_keep is None:
            return processor

        tokenizer = processor.tokenizer
        unknown_token_id = tokenizer.unk_token_id
        if unknown_token_id is None:
            return processor

        for character in dict.fromkeys(str(char) for char in characters_to_keep):
            token_ids = tokenizer(character, add_special_tokens=False)["input_ids"]
            if (
                tokenizer.convert_tokens_to_ids(character) == unknown_token_id
                or unknown_token_id in token_ids
            ):
                tokenizer.add_tokens(character)
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
            processor = AutoProcessor.from_pretrained(
                model_path, **self._hub_kwargs(model_id=model_path)
            )
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_path!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported."
            ) from error

        family = self._load_family(model_id=model_path)
        if family in _TRANSDUCER_FAMILIES:
            setattr(processor, "decoder_type", family)
        processor = self._adapt_processor(processor=processor)
        try:
            if family == "ctc":
                model = AutoModelForCTC.from_pretrained(
                    model_path, **self._hub_kwargs(model_id=model_path)
                )
            elif family == "tdt":
                model = AutoModelForTDT.from_pretrained(
                    model_path, **self._hub_kwargs(model_id=model_path)
                )
            else:
                model = AutoModel.from_pretrained(
                    model_path, **self._hub_kwargs(model_id=model_path)
                )
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Parakeet checkpoint {model_path!r} is not a Transformers-native "
                "checkpoint. NeMo collection .nemo files are unsupported."
            ) from error

        self.processor = processor
        self._resize_model_vocabulary(model=model, family=family)
        self.model_config = model.config
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=self.load_data_collator(),
            compute_metrics=self.load_compute_metrics(),
        )

    def load_compute_metrics(self) -> t.Callable[[EvalPrediction], dict]:
        """Return metrics that decode Parakeet CTC or transducer IDs."""
        return partial(_compute_parakeet_metrics, processor=self.processor)

    def load_data_collator(self) -> DataCollatorParakeetWithPadding:
        """Return the feature-aware Parakeet collator."""
        return DataCollatorParakeetWithPadding(
            processor=self.processor,
            sample_rate=self.config.model.sampling_rate,
            padding=self.config.padding,
            model_config=self.model_config,
        )

    def load_trainer_class(self) -> t.Type[Trainer]:
        """Return Trainer or the generation-compatible RNNT Trainer."""
        family = self._load_family(model_id=str(self.config.model.pretrained_model_id))
        return Trainer if family == "ctc" else ParakeetGenerationTrainer


def validate_parakeet_transducer_inputs(
    decoder_input_ids: object,
    labels: object,
    processor: object,
    model_config: object | None = None,
) -> None:
    """Require decoder IDs to be the unpadded blank-prefixed labels.

    Raises:
        ValueError:
            If the IDs do not equal ``[blank_token_id, *labels]``.
    """
    resolved_blank_token_id = get_parakeet_blank_token_id(
        processor=processor, model_config=model_config
    )
    actual_ids = _as_int_list(decoder_input_ids)
    label_ids = _as_int_list(labels)
    if actual_ids != [resolved_blank_token_id, *label_ids]:
        raise ValueError(
            "Parakeet transducer decoder_input_ids must contain exactly one more "
            "token and equal [blank_token_id, *labels] before padding."
        )


def _as_int_list(values: object) -> list[int]:
    """Convert tensor-like or iterable token IDs to a Python list.

    Returns:
        The token IDs as integers.

    Raises:
        TypeError:
            If ``values`` is neither an integer nor an iterable.
    """
    if hasattr(values, "tolist"):
        tolist = getattr(values, "tolist")
        if callable(tolist):
            values = tolist()
    if isinstance(values, Integral):
        return [int(values)]
    if not isinstance(values, c.Iterable):
        raise TypeError("Expected an integer or iterable of integers")
    return [int(value) for value in t.cast(c.Iterable[int], values)]


def get_parakeet_blank_token_id(
    processor: object, model_config: object | None = None
) -> int:
    """Resolve a Parakeet blank ID without falling back to the pad ID.

    Returns:
        The configured blank token ID.

    Raises:
        ValueError:
            If no usable blank token ID is exposed by the processor or configuration.
    """
    for source in (model_config, processor, getattr(processor, "tokenizer", None)):
        blank_token_id = _configured_int(source, "blank_token_id")
        if blank_token_id is not None:
            return blank_token_id

    tokenizer = getattr(processor, "tokenizer", None)
    blank_token = getattr(processor, "blank_token", None)
    if blank_token is None:
        blank_token = getattr(tokenizer, "blank_token", None)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if blank_token is not None and callable(convert_tokens_to_ids):
        converted_id = convert_tokens_to_ids(blank_token)
        if isinstance(converted_id, Integral):
            return int(converted_id)

    raise ValueError(
        "Parakeet transducer processor does not expose a usable blank_token_id."
    )


def _configured_int(source: object | None, name: str) -> int | None:
    """Return an integer configuration value from an object or mapping."""
    if source is None:
        return None
    if isinstance(source, c.Mapping):
        value = source.get(name)
    else:
        value = getattr(source, name, None)
    return int(value) if isinstance(value, Integral) else None


def parakeet_family(config: PreTrainedConfig) -> ParakeetFamily:
    """Identify a Parakeet family from its model type or architecture.

    Args:
        config:
            Transformers checkpoint configuration.

    Returns:
        The Parakeet family.

    Raises:
        ValueError:
            If the config is not a supported CTC, RNNT, or TDT checkpoint.
    """
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(
        str(architecture).lower()
        for architecture in getattr(config, "architectures", []) or []
    )
    descriptor = f"{model_type} {architectures}"
    if "tdt" in descriptor:
        return "tdt"
    if "rnnt" in descriptor or "transducer" in descriptor:
        return "rnnt"
    if "ctc" in descriptor:
        return "ctc"
    raise ValueError(
        "Unsupported Parakeet checkpoint architecture. Expected a "
        "Transformers-native ParakeetForCTC, ParakeetForRNNT, or ParakeetForTDT "
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
    is_transducer = str(getattr(processor, "decoder_type", "")).lower() in {
        "rnnt",
        "tdt",
    }
    return compute_error_rate_metrics(
        pred=EvalPrediction(predictions=predictions, label_ids=pred.label_ids),
        processor=processor,
        log_examples=False,
        label_group_tokens=False if is_transducer else None,
    )
