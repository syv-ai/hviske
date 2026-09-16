"""Tests for ASR metric computation."""

import typing as t
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from transformers import EvalPrediction

from hviske.compute_metrics import compute_error_rate_metrics
from hviske.data_models import Processor


def test_compute_metrics_decodes_regular_two_dimensional_ids() -> None:
    """Regular two-dimensional token IDs continue through the normal decoder."""
    processor = _make_processor()
    processor.batch_decode.return_value = ["hej"]
    processor.tokenizer.batch_decode.return_value = ["hej"]
    predictions = np.array([[1, 2]])
    labels = np.array([[1, 2]])

    metrics = compute_error_rate_metrics(
        pred=EvalPrediction(predictions=predictions, label_ids=labels),
        processor=t.cast(Processor, processor),
        log_examples=False,
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}
    np.testing.assert_array_equal(processor.batch_decode.call_args.args[0], predictions)
    np.testing.assert_array_equal(
        processor.tokenizer.batch_decode.call_args.kwargs["sequences"], labels
    )


def _make_processor(pad_token_id: int = 0) -> SimpleNamespace:
    """Create a processor double with independent decoding methods.

    Returns:
        Processor double configured with the requested padding token ID.
    """
    tokenizer = MagicMock(pad_token_id=pad_token_id)
    processor = SimpleNamespace(tokenizer=tokenizer, batch_decode=MagicMock())
    return processor


def test_compute_metrics_decodes_three_dimensional_logits() -> None:
    """Three-dimensional logits still use argmax decoding."""
    processor = _make_processor()
    processor.batch_decode.return_value = ["hej"]
    processor.tokenizer.batch_decode.return_value = ["hej"]
    logits = np.array([[[0.1, 0.9], [0.8, 0.2]]])
    labels = np.array([[1, 0]])

    metrics = compute_error_rate_metrics(
        pred=EvalPrediction(predictions=logits, label_ids=labels),
        processor=t.cast(Processor, processor),
        log_examples=False,
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}
    np.testing.assert_array_equal(
        processor.batch_decode.call_args.args[0], np.array([[1, 0]])
    )


def test_compute_metrics_replaces_prediction_padding_without_mutation() -> None:
    """Trainer padding is removed from decoded IDs while inputs stay unchanged."""
    processor = _make_processor()
    processor.batch_decode.return_value = ["hej", "du"]
    processor.tokenizer.batch_decode.return_value = ["hej", "du"]
    predictions = np.array([[1, -100], [3, 4]])
    labels = np.array([[1, -100], [3, 4]])
    original_predictions = predictions.copy()
    original_labels = labels.copy()

    metrics = compute_error_rate_metrics(
        pred=EvalPrediction(predictions=predictions, label_ids=labels),
        processor=t.cast(Processor, processor),
        log_examples=False,
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}
    np.testing.assert_array_equal(predictions, original_predictions)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(
        processor.batch_decode.call_args.args[0], np.array([[1, 0], [3, 4]])
    )
    np.testing.assert_array_equal(
        processor.tokenizer.batch_decode.call_args.kwargs["sequences"],
        np.array([[1, 0], [3, 4]]),
    )
