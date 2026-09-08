"""Data collators for the models."""

import logging
from dataclasses import dataclass

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

from .data_models import Processor

logger = logging.getLogger(__package__)


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor:
            The processor used for proccessing the data.
        sample_rate:
            The sample rate that the audio is in.
        max_seconds_per_example:
            The maximum number of seconds per example.
        padding:
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only a
                single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length or to
                the maximum acceptable input length for the model if that argument is
                not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of different
                lengths).
    """

    processor: Processor
    sample_rate: int
    max_seconds_per_example: float
    padding: bool | str
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features:
                A list of feature dicts.

        Returns:
            A dictionary of the collated features.

        Raises:
            ValueError:
                If the features do not contain either 'input_features' or 'audio' key.
        """
        if "input_values" in features[0]:
            audio_features = [dict(input_values=f["input_values"]) for f in features]
        elif "audio" in features[0]:
            audio_features = [dict(input_values=f["audio"]["array"]) for f in features]
        else:
            raise ValueError(
                "Features must contain either 'input_values' or 'audio' key."
            )

        # Get the batch
        batch: BatchFeature = self.processor.pad(  # type: ignore[union-attr]
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=int(self.sample_rate * self.max_seconds_per_example),
        )

        # Get the tokenized label sequences
        label_features = [dict(input_ids=feature["labels"]) for feature in features]

        # Pad the labels to max length
        labels_batch: BatchEncoding = self.processor.pad(  # type: ignore[union-attr]
            labels=label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=min(self.processor.tokenizer.model_max_length, 512),  # type: ignore[union-attr]
        )

        # Replace padding with -100 to ignore loss correctly
        non_one_entries: torch.Tensor = labels_batch.attention_mask.ne(1)
        labels: torch.Tensor = labels_batch.input_ids.masked_fill(non_one_entries, -100)

        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor:
            The processor used for proccessing the data.
        sample_rate:
            The sample rate that the audio is in.
        max_seconds_per_example:
            The maximum number of seconds per example.
        padding:
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only a
                single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length or to
                the maximum acceptable input length for the model if that argument is
                not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of different
                lengths).
    """

    processor: Processor
    sample_rate: int
    max_seconds_per_example: float
    padding: bool | str
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features:
                A list of feature dicts.

        Returns:
            BatchFeature:
                A dictionary of the collated features.

        Raises:
            ValueError:
                If the features do not contain either 'input_features' or 'audio' key.
        """
        if "input_features" in features[0]:
            audio_features = [
                dict(input_features=f["input_features"]) for f in features
            ]
        elif "audio" in features[0]:
            audio_features = [dict(audio=f["audio"]["array"]) for f in features]
        else:
            raise ValueError(
                "Features must contain either 'input_features' or 'audio' key."
            )

        # Get the batch
        batch = self.processor.feature_extractor.pad(  # type: ignore[union-attr]
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=int(self.sample_rate * self.max_seconds_per_example),
        )

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(  # type: ignore[union-attr]
            label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=min(self.processor.tokenizer.model_max_length, 512),  # type: ignore[union-attr]
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step, cut BOS token here as
        # it's appended later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():  # type: ignore[union-attr]
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorCohereWithPadding(DataCollatorMixin):
    """Collate Cohere features while retaining the decoder prompt.

    Cohere's prompt is part of the decoder input. Only the final prompt position
    predicts the first transcript token; all earlier prompt positions are masked from
    the loss.
    """

    processor: Processor
    padding: bool | str
    max_length: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate audio features, prompt-aware decoder inputs and labels.

        Args:
            features:
                Processed examples containing features, prompts and transcript IDs.

        Returns:
            A padded batch suitable for Cohere training.

        Raises:
            ValueError:
                If features do not contain the Cohere inputs.
        """
        if "input_features" not in features[0]:
            raise ValueError("Cohere features must contain 'input_features'.")
        if any("decoder_input_ids" not in feature for feature in features):
            raise ValueError("Cohere features must contain decoder prompt IDs.")

        audio_features = []
        for feature in features:
            audio_feature = {"input_features": feature["input_features"]}
            if "attention_mask" in feature:
                audio_feature["attention_mask"] = feature["attention_mask"]
            audio_features.append(audio_feature)
        batch = self.processor.feature_extractor.pad(
            audio_features, padding=self.padding, return_tensors=self.return_tensors
        )

        eos_token_id = self.processor.tokenizer.eos_token_id
        decoder_features: list[dict[str, list[int]]] = []
        label_features: list[dict[str, list[int]]] = []
        prompt_lengths: list[int] = []
        for feature in features:
            prompt = _as_int_list(feature["decoder_input_ids"])
            transcript = _as_int_list(feature["labels"])
            if self.max_length is not None:
                if self.max_length <= len(prompt):
                    raise ValueError(
                        "Cohere model.max_length must leave room for the decoder "
                        f"prompt and EOS (got max_length={self.max_length}, "
                        f"prompt_length={len(prompt)})."
                    )
                transcript = transcript[: self.max_length - len(prompt)]
            labels = [-100] * max(len(prompt) - 1, 0) + transcript + [eos_token_id]
            decoder_features.append({"input_ids": prompt + transcript})
            label_features.append({"input_ids": labels})
            prompt_lengths.append(len(prompt))

        decoder_batch = self.processor.tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        batch["decoder_input_ids"] = decoder_batch["input_ids"]
        batch["decoder_attention_mask"] = decoder_batch["attention_mask"]
        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        batch["prompt_length"] = torch.tensor(prompt_lengths, dtype=torch.long)
        return batch


def _as_int_list(values: object) -> list[int]:
    """Convert a tensor, array or sequence of IDs into a Python list.

    Args:
        values:
            IDs represented as a tensor, array or sequence.

    Returns:
        A list of integer IDs.
    """
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, int):
        return [values]
    return [int(value) for value in values]  # type: ignore[union-attr]
