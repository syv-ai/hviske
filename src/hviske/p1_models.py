"""Pinned offline model adapters for the P1 segmentation pipeline."""

from __future__ import annotations

import collections.abc as c
import hashlib
import json
import logging
import os
import typing as t
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .p1_segments import AlignmentResult, CTCBackend, VADBackend, VADSignal

logger = logging.getLogger(__name__)

SILERO_REPOSITORY = "snakers4/silero-vad"
SILERO_REVISION = "867c2aa692646a1f1de3e94a15c9dd9f614c0acb"
SILERO_MODEL_PATH = "src/silero_vad/data/silero_vad.jit"
SILERO_MODEL_BLOB = "5c6988d663950a93a5f0d6c38c2fe024653ec552"
SILERO_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"


class HuggingFaceCTCBackend(CTCBackend):
    """Pinned Danish wav2vec2 emissions plus real ctc-segmentation alignment."""

    def __init__(
        self,
        repository: str,
        revision: str,
        *,
        device: str = "cpu",
        frame_duration_ms: float = 20.0,
    ) -> None:
        """Load the processor and model at one immutable Hub revision.

        Raises:
            ValueError:
                If the processor does not expose a CTC blank token.
        """
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(repository, revision=revision)
        self._model = AutoModelForCTC.from_pretrained(repository, revision=revision)
        self._model.eval()
        self._device = device
        self._model.to(device)
        self._frame_duration_ms = frame_duration_ms
        blank_id = self._processor.tokenizer.pad_token_id
        if blank_id is None:
            raise ValueError("the CTC tokenizer must declare its blank/pad token")
        self._blank_id = int(blank_id)

    def align(
        self,
        audio: np.ndarray,
        alignment_text: str,
        word_map: tuple[str, ...],
        start_ms: int,
        end_ms: int,
        sampling_rate: int,
    ) -> AlignmentResult:
        """Produce word boundaries with the reference CTC segmentation algorithm.

        Returns:
            Alignment boundaries and the package's minimum mean log-probability
            scores.

        Raises:
            ValueError:
                If tokenisation does not produce one alignment for each word.
        """
        del end_ms
        inputs = self._processor(
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        )
        inputs = inputs.to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(0)
            emissions = self._torch.log_softmax(logits, dim=-1).cpu().numpy()
        words = alignment_text.split()
        tokenised_words = [
            tuple(
                int(token)
                for token in self._processor.tokenizer(
                    word, add_special_tokens=False
                ).input_ids
            )
            for word in words
        ]
        from .p1_segments import align_ctc_word_tokens

        result = align_ctc_word_tokens(
            emissions=emissions,
            tokenised_words=tokenised_words,
            start_ms=start_ms,
            frame_duration_ms=self._frame_duration_ms,
            blank_id=self._blank_id,
        )
        if len(word_map) != len(result.word_boundaries):
            raise ValueError("word map does not match tokenised alignment words")
        return result


@dataclass
class SileroVADBackend(VADBackend):
    """Small offline VAD adapter around the pinned Silero JIT model."""

    model: object
    device: str = "cpu"
    threshold: float = 0.5
    frame_size: int = 512

    def analyse(self, audio: np.ndarray, sampling_rate: int) -> VADSignal:
        """Run the model once and return thresholded speech intervals.

        Returns:
            Frame intervals classified as speech.

        Raises:
            ValueError:
                If the input sampling rate is unsupported.
        """
        if sampling_rate not in {8000, 16000}:
            raise ValueError("Silero VAD accepts 8 kHz or 16 kHz audio")
        import torch

        values = np.asarray(audio, dtype=np.float32)
        probabilities: list[float] = []
        with torch.no_grad():
            for start in range(0, len(values), self.frame_size):
                frame = torch.from_numpy(values[start : start + self.frame_size]).to(
                    self.device
                )
                if not len(frame):
                    continue
                model_call = t.cast(c.Callable[[object, int], object], self.model)
                result = model_call(frame, sampling_rate)
                probability = result[0] if isinstance(result, tuple) else result
                tensor_probability = t.cast(torch.Tensor, probability)
                if tensor_probability.ndim:
                    tensor_probability = tensor_probability.reshape(-1)[0]
                probabilities.append(float(tensor_probability.item()))
        intervals: list[tuple[int, int]] = []
        start_frame: int | None = None
        for index, probability in enumerate(probabilities):
            if probability >= self.threshold and start_frame is None:
                start_frame = index
            if probability < self.threshold and start_frame is not None:
                intervals.append(
                    (
                        start_frame * self.frame_size * 1000 // sampling_rate,
                        index * self.frame_size * 1000 // sampling_rate,
                    )
                )
                start_frame = None
        if start_frame is not None:
            intervals.append(
                (
                    start_frame * self.frame_size * 1000 // sampling_rate,
                    len(values) * 1000 // sampling_rate,
                )
            )
        return VADSignal(tuple(intervals), len(values) * 1000 // sampling_rate)


def make_silero_vad(
    cache_dir: Path,
    *,
    device: str = "cpu",
    opener: c.Callable[[str], c.BinaryIO] | None = None,
) -> SileroVADBackend:
    """Load the pinned Silero JIT model and never consult a model package cache.

    Returns:
        An offline VAD backend using the verified JIT asset.
    """
    path = download_silero_vad(cache_dir=cache_dir, opener=opener)
    verify_silero_vad_asset(path, require_provenance=True)
    import torch

    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return SileroVADBackend(model=model, device=device)


def download_silero_vad(
    cache_dir: Path, *, opener: c.Callable[[str], c.BinaryIO] | None = None
) -> Path:
    """Download and verify the immutable Silero JIT asset once.

    Args:
        cache_dir:
            Bounded directory in which the asset and provenance sidecar are stored.
        opener (optional):
            Injectable byte-stream opener for offline tests.

    Returns:
        Verified local JIT path.
    """
    destination = (
        Path(cache_dir) / "silero-vad" / SILERO_REVISION / Path(SILERO_MODEL_PATH).name
    )
    sidecar = destination.with_suffix(destination.suffix + ".provenance.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sidecar.exists():
        verify_silero_vad_asset(destination, require_provenance=True)
        return destination
    url = f"https://github.com/{SILERO_REPOSITORY}/raw/{SILERO_REVISION}/{SILERO_MODEL_PATH}"
    stream_opener = opener or urllib.request.urlopen
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with stream_opener(url) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        verify_silero_vad_asset(temporary, provenance_path=None)
        os.replace(temporary, destination)
        coordinates = {
            "repository": SILERO_REPOSITORY,
            "revision": SILERO_REVISION,
            "path": SILERO_MODEL_PATH,
            "git_blob": SILERO_MODEL_BLOB,
            "sha256": SILERO_MODEL_SHA256,
        }
        sidecar.write_text(json.dumps(coordinates, sort_keys=True), encoding="utf-8")
        verify_silero_vad_asset(destination, require_provenance=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def verify_silero_vad_asset(
    path: Path,
    *,
    repository: str = SILERO_REPOSITORY,
    revision: str = SILERO_REVISION,
    model_path: str = SILERO_MODEL_PATH,
    expected_blob: str = SILERO_MODEL_BLOB,
    expected_sha256: str = SILERO_MODEL_SHA256,
    provenance_path: Path | None = None,
    require_provenance: bool = False,
) -> None:
    """Verify the exact Silero JIT file and its source coordinates.

    Args:
        path:
            Local JIT asset to verify.
        repository:
            Git repository which owns the asset.
        revision:
            Immutable commit containing the asset.
        model_path:
            Repository-relative asset path.
        expected_blob:
            Git blob object ID for the asset.
        expected_sha256:
            SHA-256 digest of the asset bytes.
        provenance_path (optional):
            Sidecar containing the download coordinates. Defaults to the asset's
            ``.provenance.json`` sidecar when it exists.
        require_provenance (optional):
            Require the sidecar to exist. Defaults to False for byte-level test
            fixtures; production loading always enables it.

    Raises:
        ModelPinError:
            If any coordinate, sidecar, Git blob ID, or content digest differs.
    """
    if repository != SILERO_REPOSITORY or revision != SILERO_REVISION:
        raise ModelPinError("Silero repository revision is not pinned")
    if model_path != SILERO_MODEL_PATH:
        raise ModelPinError("Silero asset path is not pinned")
    if not path.is_file():
        raise ModelPinError(f"missing Silero JIT asset: {path}")
    payload = path.read_bytes()
    content_sha256 = hashlib.sha256(payload).hexdigest()
    if content_sha256 != expected_sha256:
        raise ModelPinError(
            f"Silero SHA-256 mismatch: expected {expected_sha256}, got {content_sha256}"
        )
    git_header = f"blob {len(payload)}\0".encode("ascii")
    blob = hashlib.sha1(git_header + payload).hexdigest()
    if blob != expected_blob:
        raise ModelPinError(
            f"Silero Git blob mismatch: expected {expected_blob}, got {blob}"
        )

    sidecar = provenance_path or path.with_suffix(path.suffix + ".provenance.json")
    if not sidecar.exists():
        if require_provenance:
            raise ModelPinError("Silero provenance sidecar is missing")
        return
    try:
        coordinates = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPinError("invalid Silero provenance sidecar") from exc
    expected = {
        "repository": repository,
        "revision": revision,
        "path": model_path,
        "git_blob": expected_blob,
        "sha256": expected_sha256,
    }
    if coordinates != expected:
        raise ModelPinError("Silero provenance sidecar does not match the pin")


class ModelPinError(ValueError):
    """Raised when a pinned model asset is absent or has changed."""
