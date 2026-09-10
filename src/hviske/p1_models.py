"""Pinned offline model adapters for the P1 segmentation pipeline."""

from __future__ import annotations

import collections.abc as c
import hashlib
import json
import logging
import numbers
import os
import typing as t
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .p1_contracts import P1_RUNTIME_CONTRACT
from .p1_segments import (
    CTCEmissionsAlignmentAdapter,
    UnsupportedAlignmentText,
    VADBackend,
    VADSignal,
)

logger = logging.getLogger(__name__)

SILERO_REPOSITORY = "snakers4/silero-vad"
SILERO_REVISION = "867c2aa692646a1f1de3e94a15c9dd9f614c0acb"
SILERO_MODEL_PATH = "src/silero_vad/data/silero_vad.jit"
SILERO_MODEL_BLOB = "5c6988d663950a93a5f0d6c38c2fe024653ec552"
SILERO_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"

# Compatibility aliases keep the adapter's public constants sourced from one spec.
ROEST_REPOSITORY = P1_RUNTIME_CONTRACT.roest_repository
ROEST_REVISION = P1_RUNTIME_CONTRACT.roest_revision
ROEST_LICENSE = P1_RUNTIME_CONTRACT.roest_license
ROEST_LICENSE_URL = P1_RUNTIME_CONTRACT.roest_license_url
ROEST_LICENSE_REPOSITORY = P1_RUNTIME_CONTRACT.roest_license_repository
ROEST_LICENSE_REVISION = P1_RUNTIME_CONTRACT.roest_license_revision
ROEST_LICENSE_SHA256 = P1_RUNTIME_CONTRACT.roest_license_sha256
ROEST_MODEL_CARD_URL = P1_RUNTIME_CONTRACT.roest_model_card_url
ROEST_MODEL_CARD_SHA256 = P1_RUNTIME_CONTRACT.roest_model_card_sha256
ROEST_SAMPLING_RATE = P1_RUNTIME_CONTRACT.roest_sampling_rate
ROEST_FRAME_STRIDE_SAMPLES = P1_RUNTIME_CONTRACT.roest_frame_stride_samples
ROEST_FRAME_DURATION_MS = P1_RUNTIME_CONTRACT.roest_frame_duration_ms
ROEST_VOCAB_SIZE = P1_RUNTIME_CONTRACT.roest_vocab_size
ROEST_BLANK_TOKEN_ID = P1_RUNTIME_CONTRACT.roest_blank_token_id
ROEST_WORD_DELIMITER_TOKEN_ID = P1_RUNTIME_CONTRACT.roest_word_delimiter_token_id
ROEST_REQUIRED_TOKENS = P1_RUNTIME_CONTRACT.roest_required_tokens
ROEST_TOKENIZER_CASE = P1_RUNTIME_CONTRACT.roest_tokenizer_case


class HuggingFaceCTCBackend(CTCEmissionsAlignmentAdapter):
    """Pinned Roest Danish wav2vec2 emissions and CTC alignment."""

    def __init__(
        self,
        repository: str,
        revision: str,
        *,
        device: str = "cpu",
        frame_duration_ms: float | None = None,
        case_folding: bool = True,
    ) -> None:
        """Load the processor and model at one immutable Hub revision.

        Raises:
            ModelPinError:
                If the repository or revision is not the pinned Roest checkpoint.
        """
        import torch
        from transformers import (
            AutoModelForCTC,
            Wav2Vec2CTCTokenizer,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Processor,
        )

        if (repository, revision) != (ROEST_REPOSITORY, ROEST_REVISION):
            raise ModelPinError("P1 CTC backend requires the pinned Roest checkpoint")
        validate_ctc_normalisation_compatibility(
            repository=repository, case_folding=case_folding
        )
        verify_hub_model_revision(repository=repository, revision=revision)
        self._torch = torch
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(repository, revision=revision)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            repository, revision=revision
        )
        self._processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )
        self._model = AutoModelForCTC.from_pretrained(repository, revision=revision)
        self._model.eval()
        self._device = device
        self._model.to(device)
        self._model_dtype = self._model_floating_dtype()
        self._frame_duration_ms = validate_ctc_model_contract(
            model_config=self._model.config,
            processor=self._processor,
            expected_frame_duration_ms=frame_duration_ms,
        )
        self._sampling_rate = ROEST_SAMPLING_RATE
        self._frame_stride_samples = ROEST_FRAME_STRIDE_SAMPLES
        self._blank_id = ROEST_BLANK_TOKEN_ID
        super().__init__(
            emissions_provider=self._compute_emissions,
            tokeniser=self._tokenise_word,
            frame_duration_ms=self._frame_duration_ms,
            blank_id=self._blank_id,
            validate_word_map=True,
        )

    def _model_floating_dtype(self) -> object:
        """Return the first floating-point parameter dtype, or float32."""
        parameters = getattr(self._model, "parameters", None)
        if not callable(parameters):
            return self._torch.float32
        try:
            parameter_iterator = iter(parameters())
        except (AttributeError, TypeError):
            return self._torch.float32
        for parameter in parameter_iterator:
            is_floating_point = getattr(parameter, "is_floating_point", None)
            if callable(is_floating_point) and is_floating_point():
                return parameter.dtype
        return self._torch.float32

    def _compute_emissions(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Run the pinned model and return frame-by-class log probabilities.

        Returns:
            Frame-by-class CTC log probabilities.

        Raises:
            ValueError:
                If audio uses a sampling rate other than the pinned processor rate.
        """
        if sampling_rate != self._sampling_rate:
            raise ValueError(
                f"CTC audio must be sampled at {self._sampling_rate} Hz, "
                f"not {sampling_rate} Hz"
            )
        inputs = self._processor(
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        )
        inputs = inputs.to(self._device)
        model_dtype = self._model_floating_dtype()
        self._model_dtype = model_dtype
        inputs["input_values"] = inputs["input_values"].to(
            self._device, dtype=model_dtype
        )
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits.squeeze(0).float()
            return self._torch.log_softmax(logits, dim=-1).cpu().numpy()

    def _tokenise_word(self, word: str) -> c.Sequence[int]:
        """Tokenise one canonical alignment word without special tokens.

        Returns:
            Token IDs for the word.

        Raises:
            UnsupportedAlignmentText:
                If the word contains a character outside the pinned vocabulary or
                the tokenizer returns an invalid token ID.
        """
        token_ids = self._processor.tokenizer(word, add_special_tokens=False).input_ids
        unk_id = getattr(self._processor.tokenizer, "unk_token_id", None)
        if token_ids is None or isinstance(token_ids, (bool, str, bytes)):
            raise UnsupportedAlignmentText(
                "alignment text contains no usable CTC token IDs"
            )
        if not token_ids:
            raise UnsupportedAlignmentText(
                "alignment text contains a token outside the CTC vocabulary"
            )
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, numbers.Integral):
                raise UnsupportedAlignmentText(
                    "tokeniser returned a non-integral CTC token ID"
                )
            if unk_id is not None and token_id == unk_id:
                raise UnsupportedAlignmentText(
                    "alignment text contains a token outside the CTC vocabulary"
                )
            if token_id < 0 or token_id >= ROEST_VOCAB_SIZE:
                raise UnsupportedAlignmentText(
                    "tokeniser returned an invalid CTC token ID"
                )
            if token_id == self._blank_id:
                raise UnsupportedAlignmentText(
                    "tokeniser returned the CTC blank token ID"
                )
        return token_ids


class ModelPinError(ValueError):
    """Raised when a pinned model asset is absent or has changed."""


def validate_ctc_model_contract(
    *,
    model_config: object,
    processor: object,
    expected_frame_duration_ms: float | None = None,
) -> float:
    """Validate the pinned Roest CTC architecture and return its frame duration.

    The alignment clock is derived from the model's convolutional feature extractor,
    rather than from a separately maintained default.  This keeps a changed model
    architecture from silently shifting every boundary in the P1 output.

    Args:
        model_config:
            Transformers model configuration to validate.
        processor:
            Transformers processor exposing a feature extractor and tokenizer.
        expected_frame_duration_ms (optional):
            Optional caller expectation retained for explicit compatibility checks.

    Returns:
        Duration represented by one emission frame in milliseconds.

    Raises:
        ValueError:
            If the model is not the pinned CTC shape or its tokenizer is incomplete.
    """
    architecture = getattr(model_config, "architectures", None)
    if architecture != [P1_RUNTIME_CONTRACT.roest_architecture]:
        raise ValueError(
            "P1 CTC model must declare "
            f"{P1_RUNTIME_CONTRACT.roest_architecture} architecture"
        )
    if (
        getattr(model_config, "model_type", None)
        != P1_RUNTIME_CONTRACT.roest_model_type
    ):
        raise ValueError(
            "P1 CTC model must declare model_type "
            f"{P1_RUNTIME_CONTRACT.roest_model_type}"
        )

    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    if sampling_rate != ROEST_SAMPLING_RATE:
        raise ValueError(
            f"P1 CTC processor must use {ROEST_SAMPLING_RATE} Hz sampling, "
            f"not {sampling_rate!r}"
        )
    # Roest's pinned config.json does not declare a sampling rate. The processor
    # preprocessor metadata is the authoritative input clock; validate the model
    # declaration only when a future checkpoint supplies one.
    configured_sampling_rate = getattr(model_config, "sampling_rate", None)
    if configured_sampling_rate is not None and (
        configured_sampling_rate != P1_RUNTIME_CONTRACT.roest_sampling_rate
    ):
        raise ValueError("CTC model sampling rate is not the P1 contract")

    strides = getattr(model_config, "conv_stride", None)
    if not isinstance(strides, c.Sequence) or isinstance(strides, (str, bytes)):
        raise ValueError("CTC model must declare convolutional strides")
    if not strides or any(
        not isinstance(value, int) or value <= 0 for value in strides
    ):
        raise ValueError("CTC convolutional strides must be positive integers")
    stride_samples = int(np.prod(strides))
    if stride_samples != P1_RUNTIME_CONTRACT.roest_frame_stride_samples:
        raise ValueError(
            "unsupported CTC frame stride: "
            f"expected {P1_RUNTIME_CONTRACT.roest_frame_stride_samples} samples, "
            f"got {stride_samples}"
        )
    ratio = getattr(model_config, "inputs_to_logits_ratio", None)
    if ratio is not None and ratio != stride_samples:
        raise ValueError("CTC inputs_to_logits_ratio disagrees with conv_stride")
    frame_duration_ms = 1000.0 * stride_samples / sampling_rate
    if frame_duration_ms != P1_RUNTIME_CONTRACT.roest_frame_duration_ms:
        raise ValueError("CTC frame duration is not the P1 20 ms contract")
    if expected_frame_duration_ms is not None and (
        expected_frame_duration_ms != frame_duration_ms
    ):
        raise ValueError(
            "configured frame duration does not match the loaded CTC model: "
            f"expected {expected_frame_duration_ms}, got {frame_duration_ms}"
        )

    vocab_size = getattr(model_config, "vocab_size", None)
    tokenizer = getattr(processor, "tokenizer", None)
    vocabulary = getattr(tokenizer, "get_vocab", lambda: {})()
    if any(vocabulary.get(token) is None for token in ROEST_REQUIRED_TOKENS):
        raise ValueError("P1 CTC tokenizer does not cover Danish letters and numbers")
    if (
        vocab_size != P1_RUNTIME_CONTRACT.roest_vocab_size
        or len(vocabulary) != P1_RUNTIME_CONTRACT.roest_vocab_size
    ):
        raise ValueError(
            "P1 CTC tokenizer must expose exactly "
            f"{P1_RUNTIME_CONTRACT.roest_vocab_size} vocabulary entries"
        )
    if (
        getattr(model_config, "pad_token_id", None)
        != P1_RUNTIME_CONTRACT.roest_blank_token_id
    ):
        raise ValueError(
            "P1 CTC model must use token "
            f"{P1_RUNTIME_CONTRACT.roest_blank_token_id} as its CTC blank"
        )
    if (
        getattr(tokenizer, "pad_token_id", None)
        != P1_RUNTIME_CONTRACT.roest_blank_token_id
    ):
        raise ValueError(
            "P1 CTC tokenizer must use token "
            f"{P1_RUNTIME_CONTRACT.roest_blank_token_id} as its CTC blank"
        )
    if getattr(tokenizer, "pad_token", None) != "<pad>":
        raise ValueError("P1 CTC tokenizer blank must be the <pad> token")
    if getattr(tokenizer, "word_delimiter_token", None) != "|":
        raise ValueError("P1 CTC tokenizer must use | as its word delimiter")
    if vocabulary.get("|") != P1_RUNTIME_CONTRACT.roest_word_delimiter_token_id:
        raise ValueError(
            "P1 CTC tokenizer must use token "
            f"{P1_RUNTIME_CONTRACT.roest_word_delimiter_token_id} "
            "as its word delimiter"
        )
    return frame_duration_ms


def validate_ctc_normalisation_compatibility(
    *, repository: str, case_folding: bool
) -> None:
    """Validate text normalisation against the pinned CTC tokenizer.

    The pinned Roest vocabulary contains lowercase letters only.  Case folding is
    therefore part of the model/configuration contract, rather than an optional
    presentation choice.

    Args:
        repository:
            CTC model repository to validate.
        case_folding:
            Whether canonical alignment text is case-folded.

    Raises:
        ValueError:
            If the pinned Roest model is configured without case folding.
    """
    if (
        repository == P1_RUNTIME_CONTRACT.roest_repository
        and case_folding != P1_RUNTIME_CONTRACT.normalisation_case_folding
    ):
        raise ValueError(
            f"Roest's {P1_RUNTIME_CONTRACT.roest_tokenizer_case} tokenizer requires "
            "normalisation.case_folding=true"
        )


def verify_hub_model_revision(
    *, repository: str, revision: str, api: object | None = None
) -> None:
    """Verify a Hub model through Hugging Face's repository API.

    Silero is deliberately not routed through this function: it is a GitHub asset,
    and its commit/blob identity is checked by :func:`verify_silero_vad_revision`.

    Raises:
        ModelPinError:
            If the revision is incomplete or resolves to another revision.
    """
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ModelPinError("Hub model revision must be a complete 40-character SHA")
    client = api
    if client is None:
        from huggingface_hub import HfApi

        client = HfApi()
    repo_info = client.repo_info(repository, revision=revision)
    resolved = (
        repo_info.get("sha")
        if isinstance(repo_info, c.Mapping)
        else getattr(repo_info, "sha", None)
    )
    if resolved is not None and resolved != revision:
        raise ModelPinError("Hugging Face did not resolve the pinned model revision")


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
        reset_states = getattr(self.model, "reset_states", None)
        if callable(reset_states):
            reset_states()
        probabilities: list[float] = []
        with torch.no_grad():
            for start in range(0, len(values), self.frame_size):
                frame_values = values[start : start + self.frame_size]
                if len(frame_values) < self.frame_size:
                    frame_values = np.pad(
                        frame_values, (0, self.frame_size - len(frame_values))
                    )
                frame = torch.from_numpy(frame_values).to(self.device)
                model_call = t.cast(c.Callable[[object, int], object], self.model)
                result = model_call(frame, sampling_rate)
                probability = result[0] if isinstance(result, tuple) else result
                tensor_probability = t.cast(torch.Tensor, probability)
                if tensor_probability.ndim:
                    tensor_probability = tensor_probability.reshape(-1)[0]
                probabilities.append(float(tensor_probability.item()))
        duration_ms = len(values) * 1000 // sampling_rate
        intervals: list[tuple[int, int]] = []
        start_frame: int | None = None
        for index, probability in enumerate(probabilities):
            if probability >= self.threshold and start_frame is None:
                start_frame = index
            if probability < self.threshold and start_frame is not None:
                start_ms = min(
                    duration_ms, start_frame * self.frame_size * 1000 // sampling_rate
                )
                end_ms = min(
                    duration_ms, index * self.frame_size * 1000 // sampling_rate
                )
                if end_ms > start_ms:
                    intervals.append((start_ms, end_ms))
                start_frame = None
        if start_frame is not None:
            start_ms = min(
                duration_ms, start_frame * self.frame_size * 1000 // sampling_rate
            )
            if duration_ms > start_ms:
                intervals.append((start_ms, duration_ms))
        return VADSignal(tuple(intervals), duration_ms)


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
    if opener is None:
        url = verify_silero_vad_revision()
    else:
        url = (
            f"https://raw.githubusercontent.com/{SILERO_REPOSITORY}/"
            f"{SILERO_REVISION}/{SILERO_MODEL_PATH}"
        )
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
    if expected_blob != SILERO_MODEL_BLOB or expected_sha256 != SILERO_MODEL_SHA256:
        raise ModelPinError("Silero asset digests are not pinned")
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


def verify_silero_vad_revision(
    *,
    repository: str = SILERO_REPOSITORY,
    revision: str = SILERO_REVISION,
    model_path: str = SILERO_MODEL_PATH,
    expected_blob: str = SILERO_MODEL_BLOB,
    expected_sha256: str = SILERO_MODEL_SHA256,
    path: Path | None = None,
    opener: c.Callable[[str], c.BinaryIO] | None = None,
) -> str:
    """Verify Silero's exact GitHub commit and blob before downloading it.

    Returns:
        The immutable raw-file URL for the verified GitHub revision.

    Raises:
        ModelPinError:
            If GitHub resolves either coordinate to a different object.
    """
    if repository != SILERO_REPOSITORY:
        raise ModelPinError("Silero repository is not pinned")
    if revision != SILERO_REVISION:
        raise ModelPinError("Silero revision is not pinned")
    if (
        model_path != SILERO_MODEL_PATH
        or expected_blob != SILERO_MODEL_BLOB
        or expected_sha256 != SILERO_MODEL_SHA256
    ):
        raise ModelPinError("Silero asset coordinates are not pinned")
    stream_opener = opener or urllib.request.urlopen
    api_root = "https://api.github.com/repos/"
    encoded_repository = urllib.parse.quote(repository, safe="/")
    commit_url = f"{api_root}{encoded_repository}/commits/{revision}"
    commit = _read_json_url(commit_url, stream_opener)
    if commit.get("sha") != revision:
        raise ModelPinError("GitHub did not resolve the pinned Silero commit")
    encoded_path = urllib.parse.quote(model_path, safe="/")
    contents_url = (
        f"{api_root}{encoded_repository}/contents/{encoded_path}?ref={revision}"
    )
    contents = _read_json_url(contents_url, stream_opener)
    if contents.get("sha") != expected_blob:
        raise ModelPinError("GitHub did not resolve the pinned Silero blob")
    if path is not None:
        verify_silero_vad_asset(
            path,
            repository=repository,
            revision=revision,
            model_path=model_path,
            expected_blob=expected_blob,
            expected_sha256=expected_sha256,
        )
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{model_path}"


def _read_json_url(
    url: str, opener: c.Callable[[str], c.BinaryIO]
) -> dict[str, object]:
    """Read one GitHub API response without introducing a Hub client.

    Returns:
        The decoded JSON object.

    Raises:
        ModelPinError:
            If the response cannot be decoded as an object.
    """
    try:
        with opener(url) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPinError(f"could not verify GitHub model metadata: {url}") from exc
    if not isinstance(value, dict):
        raise ModelPinError("GitHub model metadata is not an object")
    return t.cast(dict[str, object], value)
