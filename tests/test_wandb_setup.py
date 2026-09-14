"""Focused tests for production W&B setup."""

import pytest
from omegaconf import DictConfig, OmegaConf

import hviske.experiment_tracking.wandb_setup as wandb_module
import hviske.finetune as finetune_module
from hviske.experiment_tracking.wandb_setup import WandbSetup, preflight_wandb_access


def test_production_wandb_rejects_offline_mode() -> None:
    """The production preset cannot silently become offline."""
    with pytest.raises(ValueError, match="mode=online"):
        WandbSetup(config=_config(mode="offline"))


def _config(**tracking_overrides: object) -> DictConfig:
    """Build a minimal W&B configuration for unit tests.

    Returns:
        A production-like W&B configuration.
    """
    tracking = {
        "type": "wandb",
        "name_experiment": "hviske",
        "name_run": "run",
        "name_group": "v6.0",
        "entity": None,
        "id": "run-id",
        "resume": "allow",
        "job_type": "train",
        "tags": ["v6.0"],
        "mode": "online",
        "save_code": True,
        "log_model": False,
        "watch": False,
        "production": True,
    }
    tracking.update(tracking_overrides)
    return OmegaConf.create(
        {
            "enable_experiment_tracking": True,
            "experiment_tracking": tracking,
            "resolved_value": "${experiment_tracking.name_run}",
            "api_key": "must-not-be-logged",
        }
    )


def test_tracking_initialises_before_expensive_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authentication failure happens before noise, model, or data loading."""
    events: list[str] = []

    class Setup:
        def run_initialization(self) -> None:
            events.append("tracking")
            raise RuntimeError("authentication failed")

    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(
        finetune_module, "load_extracking_setup", lambda config: Setup()
    )
    monkeypatch.setattr(
        finetune_module, "download_background_noises", lambda: events.append("noise")
    )
    monkeypatch.setattr(
        finetune_module, "load_model_setup", lambda config: events.append("model")
    )

    with pytest.raises(RuntimeError, match="authentication failed"):
        finetune_module.finetune(config=_config())

    assert events == ["tracking"]


def test_wandb_init_uses_plain_resolved_safe_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialisation passes production fields without credentials."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        wandb_module.wandb, "init", lambda **kwargs: calls.append(kwargs)
    )

    WandbSetup(config=_config()).run_initialization()

    assert len(calls) == 1
    init_kwargs = calls[0]
    assert init_kwargs["project"] == "hviske"
    assert init_kwargs["group"] == "v6.0"
    assert init_kwargs["id"] == "run-id"
    assert init_kwargs["resume"] == "allow"
    assert init_kwargs["mode"] == "online"
    payload = init_kwargs["config"]
    assert isinstance(payload, dict)
    assert payload["resolved_value"] == "run"
    assert payload["api_key"] == "[REDACTED]"


def test_wandb_preflight_verifies_login_and_online_api() -> None:
    """Credential verification and API access are independently exercised."""
    login_calls: list[dict[str, object]] = []

    def login(**kwargs: object) -> bool:
        login_calls.append(kwargs)
        return True

    class Api:
        viewer = {"username": "personal"}

    preflight_wandb_access(config=_config(), login=login, api_factory=Api)

    assert login_calls == [{"verify": True}]
