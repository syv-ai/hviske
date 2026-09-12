"""Offline tests for pinned P1 model adapters."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from p1_dataset.models import (
    ROEST_REPOSITORY,
    ROEST_VOCAB_SIZE,
    SILERO_MODEL_BLOB,
    SILERO_MODEL_PATH,
    SILERO_MODEL_SHA256,
    SILERO_REPOSITORY,
    SILERO_REVISION,
    HuggingFaceCTCBackend,
    ModelPinError,
    SileroVADBackend,
    UnsupportedAlignmentText,
    download_silero_vad,
    validate_ctc_model_contract,
    validate_ctc_normalisation_compatibility,
    verify_hub_model_revision,
    verify_silero_vad_asset,
    verify_silero_vad_revision,
)


def test_hub_model_revision_uses_hf_api() -> None:
    """Hub revisions are resolved by the injected HfApi-compatible client."""

    class FakeApi:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def repo_info(self, repository: str, revision: str) -> object:
            self.calls.append((repository, revision))
            return type("Info", (), {"sha": revision})()

    api = FakeApi()
    verify_hub_model_revision(repository="org/model", revision="a" * 40, api=api)
    assert api.calls == [("org/model", "a" * 40)]


@pytest.mark.parametrize(
    "model_dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["float32", "float16", "bfloat16"],
)
def test_roest_ctc_backend_casts_inputs_and_normalises_emissions(
    model_dtype: torch.dtype,
) -> None:
    """Inputs follow the model dtype and device while masks stay integer."""
    requested_device = torch.device("meta")

    class FakeBatchEncoding(dict[str, torch.Tensor]):
        def to(self, device: torch.device) -> "FakeBatchEncoding":
            return FakeBatchEncoding(
                {key: value.to(device) for key, value in self.items()}
            )

    class FakeProcessor:
        def __call__(
            self, audio: np.ndarray, *, sampling_rate: int, return_tensors: str
        ) -> FakeBatchEncoding:
            assert sampling_rate == 16_000
            assert return_tensors == "pt"
            return FakeBatchEncoding(
                {
                    "input_values": torch.from_numpy(audio).unsqueeze(0),
                    "attention_mask": torch.ones((1, len(audio)), dtype=torch.long),
                }
            )

    class FakeCTC(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype_marker = torch.nn.Parameter(
                torch.ones(1, dtype=model_dtype), requires_grad=False
            )
            self.input_values_device: torch.device | None = None
            self.attention_mask_device: torch.device | None = None
            self.input_values_dtype: torch.dtype | None = None
            self.attention_mask_dtype: torch.dtype | None = None

        def forward(
            self, input_values: torch.Tensor, attention_mask: torch.Tensor
        ) -> SimpleNamespace:
            self.input_values_device = input_values.device
            self.attention_mask_device = attention_mask.device
            self.input_values_dtype = input_values.dtype
            self.attention_mask_dtype = attention_mask.dtype
            assert not torch.is_grad_enabled()
            logits = torch.tensor(
                [[[-2.0, 0.0], [0.5, -0.5], [1.0, 2.0], [0.0, 0.0]]], dtype=model_dtype
            )
            return SimpleNamespace(logits=logits)

    model = FakeCTC()
    backend = object.__new__(HuggingFaceCTCBackend)
    object.__setattr__(backend, "_torch", torch)
    object.__setattr__(backend, "_processor", FakeProcessor())
    object.__setattr__(backend, "_model", model)
    object.__setattr__(backend, "_device", requested_device)
    object.__setattr__(backend, "_sampling_rate", 16_000)

    emissions = backend._compute_emissions(
        np.ones(4, dtype=np.float32), sampling_rate=16_000
    )

    assert model.input_values_device == requested_device
    assert model.attention_mask_device == requested_device
    assert model.input_values_dtype == model_dtype
    assert model.attention_mask_dtype == torch.long
    assert isinstance(emissions, np.ndarray)
    assert emissions.dtype == np.float32
    assert emissions.shape == (4, 2)
    assert np.isfinite(emissions).all()
    np.testing.assert_allclose(np.exp(emissions).sum(axis=-1), np.ones(4))


def test_roest_ctc_contract_derives_20ms_frames_and_token_coverage() -> None:
    """The pinned architecture derives the expected alignment clock and coverage."""
    model_config, processor = _roest_contract_objects()

    assert (
        validate_ctc_model_contract(model_config=model_config, processor=processor)
        == 20.0
    )


def _roest_contract_objects() -> tuple[SimpleNamespace, SimpleNamespace]:
    vocabulary = {
        token: index
        for index, token in enumerate("0123456789abcdefghijklmnopqrstuvwxyz|åæéøü")
    }
    vocabulary.update({"<s>": 42, "</s>": 43, "<unk>": 44, "<pad>": 45})
    tokenizer = SimpleNamespace(
        get_vocab=lambda: vocabulary,
        pad_token_id=45,
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000), tokenizer=tokenizer
    )
    model_config = SimpleNamespace(
        architectures=["Wav2Vec2ForCTC"],
        model_type="wav2vec2",
        # The pinned Roest config.json omits sampling_rate; the processor owns it.
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        inputs_to_logits_ratio=320,
        vocab_size=46,
        pad_token_id=45,
    )
    return model_config, processor


def test_roest_ctc_contract_rejects_missing_danish_token() -> None:
    """The tokenizer must retain the Danish letters used by alignment text."""
    model_config, processor = _roest_contract_objects()
    vocabulary = processor.tokenizer.get_vocab()
    del vocabulary["å"]

    with pytest.raises(ValueError, match="does not cover Danish letters"):
        validate_ctc_model_contract(model_config=model_config, processor=processor)


def test_roest_ctc_contract_rejects_sampling_rate_mismatch() -> None:
    """The processor and model must use the P1 16 kHz input clock."""
    model_config, processor = _roest_contract_objects()
    processor.feature_extractor.sampling_rate = 8_000

    with pytest.raises(ValueError, match="must use 16000 Hz sampling"):
        validate_ctc_model_contract(model_config=model_config, processor=processor)


def test_roest_ctc_contract_rejects_unsupported_frame_stride() -> None:
    """A changed convolutional stride cannot silently corrupt boundaries."""
    model_config, processor = _roest_contract_objects()
    model_config.conv_stride = [5, 2, 2, 2, 2, 2]

    with pytest.raises(ValueError, match="unsupported CTC frame stride"):
        validate_ctc_model_contract(model_config=model_config, processor=processor)


def test_roest_lowercase_tokenizer_requires_case_folding() -> None:
    """The model/config invariant rejects canonical text that retains uppercase."""
    with pytest.raises(ValueError, match="case_folding=true"):
        validate_ctc_normalisation_compatibility(
            repository=ROEST_REPOSITORY, case_folding=False
        )

    validate_ctc_normalisation_compatibility(
        repository=ROEST_REPOSITORY, case_folding=True
    )


def test_roest_tokeniser_rejects_unknown_and_blank_token_ids() -> None:
    """Unknown and blank labels cannot be forced through the CTC aligner."""

    class Tokenizer:
        def __init__(self, unk_token_id: int | None, input_ids: list[int]) -> None:
            self.unk_token_id = unk_token_id
            self.input_ids = input_ids

        def __call__(self, word: str, *, add_special_tokens: bool) -> SimpleNamespace:
            del word
            assert not add_special_tokens
            return SimpleNamespace(input_ids=self.input_ids)

    backend = object.__new__(HuggingFaceCTCBackend)
    object.__setattr__(
        backend,
        "_processor",
        SimpleNamespace(tokenizer=Tokenizer(unk_token_id=7, input_ids=[7])),
    )
    object.__setattr__(backend, "_blank_id", 0)

    with pytest.raises(UnsupportedAlignmentText):
        backend._tokenise_word("hej")

    object.__setattr__(
        backend,
        "_processor",
        SimpleNamespace(tokenizer=Tokenizer(unk_token_id=None, input_ids=[0])),
    )
    with pytest.raises(UnsupportedAlignmentText):
        backend._tokenise_word("hej")


@pytest.mark.parametrize(
    "input_ids",
    [[], [None], [True], [1.5], [-1], [ROEST_VOCAB_SIZE]],
    ids=["empty", "none", "bool", "non-integral", "negative", "out-of-range"],
)
def test_roest_tokeniser_rejects_unsupported_token_ids(input_ids: list[object]) -> None:
    """Malformed tokenizer output becomes a typed unsupported-text rejection."""

    class Tokenizer:
        unk_token_id = None

        def __call__(self, word: str, *, add_special_tokens: bool) -> SimpleNamespace:
            del word
            assert not add_special_tokens
            return SimpleNamespace(input_ids=input_ids)

    backend = object.__new__(HuggingFaceCTCBackend)
    object.__setattr__(backend, "_processor", SimpleNamespace(tokenizer=Tokenizer()))
    object.__setattr__(backend, "_blank_id", 0)

    with pytest.raises(UnsupportedAlignmentText):
        backend._tokenise_word("hej")


def test_silero_asset_rejects_content_checksum(tmp_path: Path) -> None:
    """A file with the wrong content cannot be loaded as the pinned model."""
    asset = tmp_path / "silero_vad.jit"
    asset.write_bytes(b"not a model")
    with pytest.raises(ModelPinError, match="SHA-256"):
        verify_silero_vad_asset(asset)


def test_silero_download_verifies_git_blob_and_provenance(tmp_path: Path) -> None:
    """The downloader records immutable source coordinates beside the bytes."""
    payload = b"synthetic model"
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    digest = hashlib.sha256(payload).hexdigest()

    def opener(_: str) -> io.BytesIO:
        return io.BytesIO(payload)

    with pytest.raises(ModelPinError):
        verify_silero_vad_asset(
            tmp_path / "missing.jit", expected_blob=blob, expected_sha256=digest
        )

    # The real constants are intentionally not replaced in production; this
    # verifies the downloader's immutable URL/provenance protocol without a fetch.
    with pytest.raises(ModelPinError, match="SHA-256"):
        download_silero_vad(tmp_path, opener=opener)


def test_silero_pin_constants_are_complete() -> None:
    """The production pin names the exact repository, revision, path and digest."""
    assert SILERO_REPOSITORY == "snakers4/silero-vad"
    assert len(SILERO_REVISION) == 40
    assert SILERO_MODEL_PATH == "src/silero_vad/data/silero_vad.jit"
    assert len(SILERO_MODEL_SHA256) == 64


def test_silero_revision_verifier_checks_github_commit_and_blob() -> None:
    """GitHub coordinates are checked without constructing an HfApi client."""
    responses = {
        f"https://api.github.com/repos/{SILERO_REPOSITORY}/commits/{SILERO_REVISION}": {
            "sha": SILERO_REVISION
        },
        (
            "https://api.github.com/repos/"
            f"{SILERO_REPOSITORY}/contents/{SILERO_MODEL_PATH}?ref={SILERO_REVISION}"
        ): {"sha": SILERO_MODEL_BLOB},
    }

    def opener(url: str) -> io.BytesIO:
        return io.BytesIO(json.dumps(responses[url]).encode())

    assert verify_silero_vad_revision(opener=opener).startswith(
        "https://raw.githubusercontent.com/"
    )


def test_silero_uses_padded_frames_and_resets_between_programmes() -> None:
    """Silero receives fixed frames and cannot carry state between programmes."""

    class FakeModel:
        def __init__(self) -> None:
            self.reset_count = 0
            self.frames: list[np.ndarray] = []
            self.frame_in_programme = 0

        def __call__(self, frame: torch.Tensor, sampling_rate: int) -> torch.Tensor:
            assert sampling_rate == 16_000
            self.frames.append(frame.detach().cpu().numpy())
            self.frame_in_programme += 1
            return torch.tensor(1.0 if self.frame_in_programme == 1 else 0.0)

        def reset_states(self) -> None:
            self.reset_count += 1
            self.frame_in_programme = 0

    model = FakeModel()
    backend = SileroVADBackend(model=model)
    first = backend.analyse(np.ones(513, dtype=np.float32), 16_000)
    second = backend.analyse(np.ones(512, dtype=np.float32), 16_000)

    assert model.reset_count == 2
    assert [len(frame) for frame in model.frames] == [512, 512, 512]
    assert np.all(model.frames[1][1:] == 0)
    assert first.programme_duration_ms == 32
    assert first.speech_intervals == ((0, 32),)
    assert second.speech_intervals == ((0, 32),)
