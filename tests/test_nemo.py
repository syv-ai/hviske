"""Focused tests for optional NeMo ASR inference."""

import collections.abc as c
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers.pipelines.pt_utils import KeyDataset

import hviske.evaluate as evaluation
import hviske.nemo as nemo
from hviske.cohere import get_asr_call_kwargs


class _Model:
    def __init__(self, sampling_rate: int = 16_000) -> None:
        self.cfg = SimpleNamespace(sample_rate=sampling_rate)
        self.calls: list[dict[str, object]] = []
        self.device = torch.device("cpu")
        self.evaluated = False

    def eval(self) -> "_Model":
        self.evaluated = True
        return self

    def to(self, device: torch.device) -> "_Model":
        self.device = device
        return self

    def transcribe(
        self, audio: list[object], batch_size: int, **kwargs: object
    ) -> object:
        self.calls.append(dict(audio=audio, batch_size=batch_size, kwargs=kwargs))
        return [f"text-{index}" for index in range(len(audio))]


class _Factory:
    restored: list[tuple[str, torch.device]] = []
    pretrained: list[tuple[str, torch.device]] = []
    model = _Model()

    @classmethod
    def from_pretrained(cls, model_name: str, map_location: torch.device) -> _Model:
        cls.pretrained.append((model_name, map_location))
        return cls.model

    @classmethod
    def restore_from(cls, restore_path: str, map_location: torch.device) -> _Model:
        cls.restored.append((restore_path, map_location))
        return cls.model


def test_evaluation_config_defaults_to_transformers() -> None:
    """The evaluation configuration keeps Transformers as its default backend."""
    config = OmegaConf.load("config/evaluation.yaml")
    assert config.inference_backend == "transformers"


def test_evaluation_dispatches_nemo_without_transformers_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit backend selects NeMo and does not add Whisper kwargs."""
    transcriber = nemo.NemoASRTranscriber(model=_Model())
    calls: list[dict[str, object]] = []

    def fake_loader(source: str, device: torch.device) -> nemo.NemoASRTranscriber:
        calls.append(dict(source=source, device=device))
        return transcriber

    monkeypatch.setattr(evaluation, "load_nemo_asr_transcriber", fake_loader)
    result = evaluation.load_asr_pipeline(
        model_id="nvidia/parakeet-rnnt-110m-da-dk",
        no_lm=False,
        inference_backend="nemo",
    )

    assert result is transcriber
    assert calls[0]["source"] == "nvidia/parakeet-rnnt-110m-da-dk"
    assert get_asr_call_kwargs(transcriber) == {}


def test_loader_rejects_missing_local_archive(tmp_path: Path) -> None:
    """A missing local-looking archive is not interpreted as a registry ID."""
    with pytest.raises(FileNotFoundError):
        nemo.load_nemo_asr_model(source=tmp_path / "missing.nemo")


def test_loader_reports_missing_nemo_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional dependency error explains how to install the backend."""

    def raise_missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(nemo.importlib, "import_module", raise_missing)

    with pytest.raises(ImportError, match="uv sync --extra nemo"):
        nemo.load_nemo_asr_model(source="registry-id")


def test_loader_selects_cpu_and_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit CPU selection and explicit CUDA validation are deterministic."""
    _patch_nemo(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    nemo.load_nemo_asr_model(source="registry-id")
    assert _Factory.model.device == torch.device("cpu")

    with pytest.raises(RuntimeError, match="CUDA device"):
        nemo.load_nemo_asr_model(source="registry-id", device="cuda")


def _patch_nemo(monkeypatch: pytest.MonkeyPatch) -> None:
    _Factory.restored.clear()
    _Factory.pretrained.clear()
    monkeypatch.setattr(
        nemo.importlib, "import_module", lambda name: SimpleNamespace(ASRModel=_Factory)
    )


def test_loader_uses_pretrained_for_registry_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry IDs use NeMo's generic pretrained API."""
    _patch_nemo(monkeypatch)

    nemo.load_nemo_asr_model(source="nvidia/parakeet-rnnt-110m-da-dk", device="cpu")

    assert _Factory.pretrained == [
        ("nvidia/parakeet-rnnt-110m-da-dk", torch.device("cpu"))
    ]


def test_loader_uses_restore_for_local_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local archives use NeMo's generic restore API."""
    _patch_nemo(monkeypatch)
    archive = tmp_path / "model.nemo"
    archive.touch()

    model = nemo.load_nemo_asr_model(source=archive, device="cpu")

    assert model is _Factory.model
    assert _Factory.restored == [(str(archive), torch.device("cpu"))]
    assert model.device == torch.device("cpu")
    assert model.evaluated is True


def test_nemo_module_has_no_eager_nemo_import() -> None:
    """The module itself remains importable when NeMo is absent."""
    assert "nemo.collections.asr.models" not in sys.modules


def test_transcriber_accepts_one_path_and_hypothesis_output(tmp_path: Path) -> None:
    """A single path returns a dictionary and hypothesis objects are supported."""
    path = tmp_path / "audio.wav"
    path.touch()

    output = nemo.NemoASRTranscriber(model=_HypothesisModel())(path)

    assert output == {"text": "hej"}


class _HypothesisModel(_Model):
    def transcribe(
        self, audio: list[object], batch_size: int, **kwargs: object
    ) -> object:
        del audio, batch_size, kwargs
        return [_Hypothesis("hej")]


class _Hypothesis:
    def __init__(self, text: str) -> None:
        self.text = text


def test_transcriber_accepts_transformers_key_dataset() -> None:
    """A Transformers KeyDataset follows the evaluation input contract."""
    model = _Model()
    transcriber = nemo.NemoASRTranscriber(model=model)
    inputs = KeyDataset(dataset=_AudioDataset(), key="audio")

    assert not isinstance(inputs, c.Iterable)
    assert list(transcriber(inputs, batch_size=2)) == [
        {"text": "text-0"},
        {"text": "text-1"},
    ]


class _AudioDataset(torch.utils.data.Dataset[dict[str, object]]):
    def __getitem__(self, index: int) -> dict[str, object]:
        if index >= len(self):
            raise IndexError(index)
        return {
            "audio": {
                "array": np.full(2, index, dtype=np.float32),
                "sampling_rate": 16_000,
            }
        }

    def __len__(self) -> int:
        return 2


def test_transcriber_batches_and_normalises_inputs() -> None:
    """Arrays, tensors and dictionaries are sent in requested batch sizes."""
    model = _Model()
    transcriber = nemo.NemoASRTranscriber(model=model)
    inputs = [
        np.zeros(4, dtype=np.float32),
        {"array": np.ones(3), "sampling_rate": 16_000},
        torch.zeros(2),
    ]

    outputs = list(transcriber(inputs, batch_size=2))

    assert outputs == [{"text": "text-0"}, {"text": "text-1"}, {"text": "text-0"}]
    assert [call["batch_size"] for call in model.calls] == [2, 1]
    assert model.calls[0]["kwargs"] == {"return_hypotheses": False}


def test_transcriber_checks_sampling_rate() -> None:
    """Audio dictionaries must use the model's configured sampling rate."""
    transcriber = nemo.NemoASRTranscriber(model=_Model(sampling_rate=8_000))

    with pytest.raises(ValueError, match="8000 Hz"):
        transcriber({"raw": np.zeros(2), "sampling_rate": 16_000})


def test_transcriber_detects_output_count_mismatch() -> None:
    """A broken NeMo result cannot silently shift evaluation predictions."""
    with pytest.raises(RuntimeError, match="different number of outputs"):
        list(nemo.NemoASRTranscriber(model=_MismatchModel())([np.zeros(2)]))


class _MismatchModel(_Model):
    def transcribe(
        self, audio: list[object], batch_size: int, **kwargs: object
    ) -> object:
        del audio, batch_size, kwargs
        return []
