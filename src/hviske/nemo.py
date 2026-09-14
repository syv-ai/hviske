"""Optional NeMo inference support for ASR checkpoints.

NeMo is deliberately imported inside the model loader.  This keeps the normal
Hviske installation usable without the optional NeMo dependency.
"""

import collections.abc as c
import importlib
import pathlib
import typing as t

import numpy as np
import torch


class _NemoASRModel(t.Protocol):
    """The small part of NeMo's ASR model interface used by this module."""

    @property
    def cfg(self) -> object:
        """NeMo model configuration."""

    def eval(self) -> object:
        """Set the model to evaluation mode."""

    def to(self, device: torch.device) -> object:
        """Move the model to a device."""

    def transcribe(
        self, audio: list[object], batch_size: int, **kwargs: object
    ) -> object:
        """Transcribe a batch of audio inputs."""


class _NemoASRModelFactory(t.Protocol):
    """The class-level part of NeMo's generic ASR model API."""

    @classmethod
    def from_pretrained(cls, model_name: str) -> object:
        """Load a NeMo registry model."""

    @classmethod
    def restore_from(cls, restore_path: str) -> object:
        """Restore a local NeMo archive."""


class NemoASRTranscriber:
    """Adapt NeMo's batch transcription API to the Hviske ASR contract."""

    def __init__(self, model: _NemoASRModel) -> None:
        """Initialise a NeMo transcriber.

        Args:
            model:
                A loaded NeMo ASR model.
        """
        self.model = model
        self.sampling_rate = _get_model_sampling_rate(model)

    def __call__(
        self, inputs: object, batch_size: int = 1, **kwargs: object
    ) -> dict[str, str] | c.Iterator[dict[str, str]]:
        """Transcribe one audio input or an iterable of audio inputs.

        Args:
            inputs:
                A local audio path, an audio array or tensor, an audio dictionary,
                or an iterable containing those values.
            batch_size (optional):
                Number of inputs sent to NeMo per call. Defaults to ``1``.
            kwargs (optional):
                Ignored for compatibility with Transformers ASR callables. NeMo does
                not receive Whisper generation arguments.

        Returns:
            A transcription dictionary for one input, or a lazy iterator of them.

        Raises:
            TypeError:
                If an input is not a supported audio value.
            ValueError:
                If ``batch_size`` is less than one or a sampling rate is invalid.
        """
        del kwargs
        if _is_single_audio(inputs):
            return self._transcribe_batch(batch=[inputs])[0]
        if isinstance(inputs, (str, bytes, pathlib.Path)) or not isinstance(
            inputs, c.Iterable
        ):
            raise TypeError(
                "NeMo ASR inputs must be audio or an iterable of audio inputs."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        return self._transcribe_iter(inputs=inputs, batch_size=batch_size)

    def _transcribe_batch(self, batch: list[object]) -> list[dict[str, str]]:
        prepared = [self._prepare_audio(item=item) for item in batch]
        outputs = self.model.transcribe(
            audio=prepared, batch_size=len(prepared), return_hypotheses=False
        )
        texts = _normalise_outputs(outputs=outputs, expected_count=len(batch))
        return [dict(text=text) for text in texts]

    def _prepare_audio(self, item: object) -> object:
        if isinstance(item, dict):
            raw_audio = item.get("array")
            if raw_audio is None:
                raw_audio = item.get("raw")
            if raw_audio is None:
                raise ValueError("Each NeMo audio dictionary needs 'array' or 'raw'.")
            supplied_rate = item.get("sampling_rate", self.sampling_rate)
            if not isinstance(supplied_rate, (int, np.integer)) or isinstance(
                supplied_rate, bool
            ):
                raise ValueError("NeMo audio sampling_rate must be an integer.")
            if int(supplied_rate) != self.sampling_rate:
                raise ValueError(
                    f"NeMo audio sampling rate must be {self.sampling_rate} Hz, "
                    f"not {int(supplied_rate)} Hz."
                )
            return _normalise_audio_value(raw_audio)
        if isinstance(item, (str, pathlib.Path)):
            return str(item)
        return _normalise_audio_value(item)

    def _transcribe_iter(
        self, inputs: c.Iterable[object], batch_size: int
    ) -> c.Iterator[dict[str, str]]:
        batch: list[object] = []
        for audio in inputs:
            batch.append(audio)
            if len(batch) == batch_size:
                yield from self._transcribe_batch(batch=batch)
                batch = []
        if batch:
            yield from self._transcribe_batch(batch=batch)


def load_nemo_asr_transcriber(
    source: str | pathlib.Path, device: str | torch.device | None = None
) -> NemoASRTranscriber:
    """Load a NeMo ASR model and wrap it as a Hviske transcriber.

    Args:
        source:
            A local ``.nemo`` archive or a NeMo Hub/registry model ID.
        device (optional):
            Inference device. Defaults to CUDA when available, otherwise CPU.

    Returns:
        A NeMo transcriber compatible with Hviske's ASR callable contract.
    """
    return NemoASRTranscriber(model=load_nemo_asr_model(source=source, device=device))


def _get_model_sampling_rate(model: _NemoASRModel) -> int:
    candidates: list[object] = [getattr(model, "sample_rate", None)]
    config = getattr(model, "cfg", None)
    candidates.extend(
        [
            _get_config_value(config, "sample_rate"),
            _get_config_value(_get_config_value(config, "preprocessor"), "sample_rate"),
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, (int, np.integer)) and not isinstance(candidate, bool):
            if int(candidate) > 0:
                return int(candidate)
    return 16_000


def _get_config_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _is_single_audio(inputs: object) -> bool:
    if isinstance(inputs, (dict, np.ndarray, torch.Tensor, pathlib.Path)):
        return True
    if isinstance(inputs, (str, bytes)):
        return True
    return (
        bool(inputs)
        and isinstance(inputs, (list, tuple))
        and all(
            isinstance(value, (float, int, np.number)) and not isinstance(value, bool)
            for value in inputs
        )
    )


def _normalise_audio_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise TypeError("NeMo audio tensors must contain at least one sample.")
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise TypeError("NeMo audio arrays must contain at least one sample.")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("NeMo audio arrays must contain numeric samples.")
        return value
    if isinstance(value, (list, tuple)):
        if not value or not all(
            isinstance(sample, (float, int, np.number)) and not isinstance(sample, bool)
            for sample in value
        ):
            raise TypeError("NeMo audio arrays must contain numeric samples.")
        return np.asarray(value)
    raise TypeError("NeMo audio must be a local path, array, tensor, or audio dict.")


def _normalise_outputs(outputs: object, expected_count: int) -> list[str]:
    if isinstance(outputs, (str, bytes)) or hasattr(outputs, "text"):
        raw_outputs: list[object] = [outputs]
    elif (
        isinstance(outputs, tuple)
        and len(outputs) == 2
        and isinstance(outputs[0], c.Iterable)
    ):
        raw_outputs = list(t.cast(c.Iterable[object], outputs[0]))
    elif isinstance(outputs, c.Iterable) and not isinstance(outputs, dict):
        raw_outputs = list(t.cast(c.Iterable[object], outputs))
    else:
        raise TypeError("NeMo transcribe returned an invalid output value.")
    if len(raw_outputs) != expected_count:
        raise RuntimeError(
            "NeMo transcription returned a different number of outputs than the "
            "input batch."
        )
    texts: list[str] = []
    for output in raw_outputs:
        text = output if isinstance(output, str) else getattr(output, "text", None)
        if not isinstance(text, str):
            raise TypeError("NeMo transcription outputs must be strings or hypotheses.")
        texts.append(text)
    return texts


def load_nemo_asr_model(
    source: str | pathlib.Path, device: str | torch.device | None = None
) -> _NemoASRModel:
    """Load a NeMo ASR model from a local archive or a model registry ID.

    Args:
        source:
            A local ``.nemo`` archive or a NeMo Hub/registry model ID.
        device (optional):
            Inference device. Defaults to CUDA when available, otherwise CPU.

    Returns:
        An evaluation-mode NeMo ASR model on the requested device.

    Raises:
        FileNotFoundError:
            If a local-looking ``.nemo`` source does not exist.
    """
    resolved_device = _resolve_device(device=device)
    local_path = _local_nemo_path(source=source)
    if local_path is not None and not local_path.is_file():
        raise FileNotFoundError(f"NeMo checkpoint does not exist: {local_path}")

    asr_model = _load_asr_model_class()
    if local_path is not None:
        model = asr_model.restore_from(restore_path=str(local_path))
    else:
        model = asr_model.from_pretrained(model_name=str(source))
    loaded_model = t.cast(_NemoASRModel, model)
    loaded_model.to(resolved_device)
    loaded_model.eval()
    return loaded_model


def _load_asr_model_class() -> _NemoASRModelFactory:
    try:
        module = importlib.import_module("nemo.collections.asr.models")
    except ImportError as error:
        raise ImportError(
            "NeMo inference support is not installed. Install it with "
            "`uv sync --extra nemo` (or install nemo-toolkit[asr]) before using "
            "inference_backend=nemo."
        ) from error
    return t.cast(_NemoASRModelFactory, getattr(module, "ASRModel"))


def _local_nemo_path(source: str | pathlib.Path) -> pathlib.Path | None:
    if isinstance(source, pathlib.Path):
        return source
    if source.lower().endswith(".nemo"):
        return pathlib.Path(source)
    return None


def _resolve_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved} was requested, but CUDA is not available."
        )
    return resolved
