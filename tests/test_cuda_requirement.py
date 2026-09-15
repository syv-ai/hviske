"""Tests for the optional CUDA requirement used by training presets."""

import pytest
from _pytest.monkeypatch import MonkeyPatch
from hydra import compose
from omegaconf import OmegaConf

import hviske.finetune as finetune_module


def test_cuda_requirement_check_fails_with_runtime_details(
    monkeypatch: MonkeyPatch,
) -> None:
    """A missing CUDA device produces an actionable, versioned error."""
    config = OmegaConf.create({"require_cuda": True})
    monkeypatch.setattr(finetune_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required") as error:
        finetune_module.check_cuda_requirement(config=config)

    message = str(error.value)
    assert "torch" in message
    assert "CUDA build" in message


def test_cuda_requirement_check_passes(monkeypatch: MonkeyPatch) -> None:
    """A required CUDA device allows training to continue."""
    config = OmegaConf.create({"require_cuda": True})
    monkeypatch.setattr(finetune_module.torch.cuda, "is_available", lambda: True)

    finetune_module.check_cuda_requirement(config=config)


def test_cuda_requirement_is_checked_before_tracking(monkeypatch: MonkeyPatch) -> None:
    """CUDA failure precedes experiment tracking initialisation."""
    config = OmegaConf.create(
        {
            "require_cuda": True,
            "dataloader_num_workers": 0,
            "enable_experiment_tracking": True,
        }
    )
    monkeypatch.setattr(finetune_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        finetune_module,
        "load_extracking_setup",
        lambda config: pytest.fail("tracking must not initialise without CUDA"),
    )

    with pytest.raises(RuntimeError, match="CUDA is required"):
        finetune_module.finetune(config=config)


def test_generic_preset_does_not_require_cuda(monkeypatch: MonkeyPatch) -> None:
    """The generic preset remains usable when CUDA is unavailable."""
    config = compose(config_name="asr_finetuning")
    monkeypatch.setattr(finetune_module.torch.cuda, "is_available", lambda: False)

    assert config.require_cuda is False
    finetune_module.check_cuda_requirement(config=config)


def test_sparkie_preset_requires_cuda() -> None:
    """The production Sparkie preset must not silently run on CPU."""
    config = compose(config_name="sparkie_bilingual")

    assert config.require_cuda is True
