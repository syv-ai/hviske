"""Unit tests for the `utils` module."""

import multiprocessing
import os
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import datasets.utils.logging as ds_logging
import pytest
import transformers.utils.logging as hf_logging
from datasets import Features, IterableDataset, Value
from datasets import config as datasets_config

from hviske.utils import (
    block_terminal_output,
    convert_iterable_dataset_to_dataset,
    convert_numeral_to_words,
    transformers_output_ignored,
)


class TestBlockTerminalOutput:
    """Tests for the `block_terminal_output` function."""

    def test_datasets_logging_level_is_error(self) -> None:
        """Test that the datasets logging level is set to error."""
        ds_logging.set_verbosity_warning()
        assert ds_logging.get_verbosity() == ds_logging.WARNING
        with output_blocked():
            assert ds_logging.get_verbosity() == ds_logging.ERROR


class output_blocked:
    """Convenience context manager to block terminal output."""

    def __enter__(self) -> None:
        """Block terminal output."""
        block_terminal_output()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Unblock terminal output."""
        ds_logging.set_verbosity_warning()


def test_convert_iterable_dataset_preserves_explicit_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit cache directory takes precedence over the library default."""
    canonical_cache = tmp_path / "canonical"
    explicit_cache = tmp_path / "explicit"
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(canonical_cache))
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
    dataset_id = "campaign-dataset"

    converted = convert_iterable_dataset_to_dataset(
        iterable_dataset=_iterable_dataset("explicit"),
        cache_dir=explicit_cache,
        dataset_id=dataset_id,
    )

    assert converted["text"] == ["explicit"]
    assert (explicit_cache / dataset_id).is_dir()
    assert not (canonical_cache / dataset_id).exists()

    loaded = convert_iterable_dataset_to_dataset(
        iterable_dataset=_iterable_dataset("should not be used"),
        cache_dir=explicit_cache,
        dataset_id=dataset_id,
    )

    assert loaded["text"] == ["explicit"]


def _iterable_dataset(text: str) -> IterableDataset:
    """Create a small typed iterable dataset for conversion tests.

    Returns:
        A one-row iterable dataset containing ``text``.
    """

    def rows() -> Iterator[dict[str, str]]:
        yield {"text": text}

    return IterableDataset.from_generator(
        rows, features=Features(text=Value(dtype="string"))
    )


def test_convert_iterable_dataset_uses_datasets_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing cache directory uses the path resolved by `datasets`."""
    canonical_cache = tmp_path / "canonical"
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(canonical_cache))
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "environment"))
    environment_cache = os.environ["HF_DATASETS_CACHE"]
    dataset_id = "campaign-dataset"

    converted = convert_iterable_dataset_to_dataset(
        iterable_dataset=_iterable_dataset("canonical"), dataset_id=dataset_id
    )

    assert converted["text"] == ["canonical"]
    assert (canonical_cache / dataset_id).is_dir()
    assert not (Path(environment_cache) / dataset_id).exists()

    loaded = convert_iterable_dataset_to_dataset(
        iterable_dataset=_iterable_dataset("should not be used"), dataset_id=dataset_id
    )

    assert loaded["text"] == ["canonical"]
    assert os.environ["HF_DATASETS_CACHE"] == environment_cache


@pytest.mark.parametrize(
    argnames=["numeral", "expected"],
    argvalues=[
        ("0", "nul"),
        ("1", "en"),
        ("2", "to"),
        ("3", "tre"),
        ("4", "fire"),
        ("5", "fem"),
        ("6", "seks"),
        ("7", "syv"),
        ("8", "otte"),
        ("9", "ni"),
        ("10", "ti"),
        ("11", "elleve"),
        ("12", "tolv"),
        ("13", "tretten"),
        ("14", "fjorten"),
        ("15", "femten"),
        ("16", "seksten"),
        ("17", "sytten"),
        ("18", "atten"),
        ("19", "nitten"),
        ("20", "tyve"),
        ("21", "enogtyve"),
        ("22", "toogtyve"),
        ("23", "treogtyve"),
        ("24", "fireogtyve"),
        ("25", "femogtyve"),
        ("26", "seksogtyve"),
        ("27", "syvogtyve"),
        ("28", "otteogtyve"),
        ("29", "niogtyve"),
        ("30", "tredive"),
        ("40", "fyrre"),
        ("50", "halvtreds"),
        ("60", "tres"),
        ("70", "halvfjerds"),
        ("80", "firs"),
        ("90", "halvfems"),
        ("100", "hundrede"),
        ("101", "et hundrede og en"),
        ("110", "et hundrede og ti"),
        ("121", "et hundrede og enogtyve"),
        ("200", "to hundrede"),
        ("999", "ni hundrede og nioghalvfems"),
        ("1000", "tusind"),
        ("1001", "et tusind og en"),
        ("1010", "et tusind og ti"),
        ("1100", "et tusind et hundrede"),
        ("1121", "et tusind et hundrede og enogtyve"),
        ("2000", "to tusind"),
        ("10.000", "ti tusind"),
        ("100.000", "et hundrede tusind"),
        ("100000", "et hundrede tusind"),
        ("999.999", "ni hundrede og nioghalvfems tusind ni hundrede og nioghalvfems"),
        ("999999", "ni hundrede og nioghalvfems tusind ni hundrede og nioghalvfems"),
        ("1.000.000", "en million"),
        ("1.000000", "1.000000"),
        ("1.0.00000", "1.0.00000"),
        ("1.000.001", "en million og en"),
        ("10.000.000", "ti millioner"),
        ("100.000.000", "et hundrede millioner"),
        (
            "999.999.999",
            "ni hundrede og nioghalvfems millioner ni hundrede og nioghalvfems tusind "
            "ni hundrede og nioghalvfems",
        ),
        ("10,123", "ti komma et to tre"),
        ("10.102,92", "ti tusind et hundrede og to komma ni to"),
    ],
)
def test_convert_numeral_to_words(numeral: str, expected: str) -> None:
    """Test that the `convert_numeral_to_words` function works as expected."""
    assert convert_numeral_to_words(numeral=numeral) == expected


def test_transformers_output_ignored() -> None:
    """Test that the transformers output is ignored."""
    hf_logging.set_verbosity_info()
    assert hf_logging.get_verbosity() == hf_logging.INFO
    with transformers_output_ignored():
        assert hf_logging.get_verbosity() == hf_logging.ERROR
