"""Validation helpers for native Parakeet transducer inputs."""

import collections.abc as c
import typing as t
from numbers import Integral


def validate_parakeet_transducer_inputs(
    decoder_input_ids: object,
    labels: object,
    processor: object,
    model_config: object | None = None,
    blank_token_id: int | None = None,
) -> None:
    """Require decoder IDs to be the unpadded blank-prefixed labels.

    Args:
        decoder_input_ids:
            Unpadded decoder input IDs returned by the processor.
        labels:
            Unpadded transcript labels returned by the processor.
        processor:
            The Parakeet processor used to produce the IDs.
        model_config (optional):
            A model-facing configuration, when available. Defaults to ``None``.
        blank_token_id (optional):
            An explicit blank ID for callers that have already resolved model
            configuration. Defaults to ``None``.

    Raises:
        ValueError:
            If the IDs do not equal ``[blank_token_id, *labels]``.
    """
    resolved_blank_token_id = blank_token_id
    if resolved_blank_token_id is None:
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

    Args:
        values:
            Token IDs represented as a tensor, array or iterable.

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

    Args:
        processor:
            The Parakeet processor.
        model_config (optional):
            A model-facing configuration, when available. Defaults to ``None``.

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
