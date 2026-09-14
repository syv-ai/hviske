"""Weights & Biases experiment tracking setup and preflight helpers."""

import collections.abc as c
import os
import typing as t

import wandb
from omegaconf import DictConfig, OmegaConf

from .extracking_setup import ExTrackingSetup


class WandbSetup(ExTrackingSetup):
    """Configure an authenticated, online W&B run."""

    def __init__(self, config: DictConfig) -> None:
        """Create a setup after validating its W&B configuration.

        Args:
            config:
                The configuration object.
        """
        super().__init__(config=config)
        validate_wandb_config(config=config)

    def run_finalization(self) -> None:
        """Finish the W&B run."""
        wandb.finish()  # type: ignore[attr-defined]

    def run_initialization(self) -> None:
        """Initialise the W&B run before loading training data or models."""
        tracking = self.config.experiment_tracking
        _set_artifact_logging_environment(tracking=tracking)
        init_kwargs: dict[str, object] = {
            "project": str(tracking.name_experiment),
            "name": str(tracking.name_run),
            "group": str(tracking.name_group),
            "config": _resolved_config_payload(config=self.config),
        }
        for field in ("entity", "id", "resume", "job_type", "mode", "save_code"):
            value = tracking.get(field)
            if value is not None:
                init_kwargs[field] = value
        tags = tracking.get("tags")
        if tags:
            init_kwargs["tags"] = [str(tag) for tag in tags]
        init_fn = t.cast(c.Callable[..., object], wandb.init)
        init_fn(**init_kwargs)


def _resolved_config_payload(config: DictConfig) -> dict[str, object]:
    """Return a resolved, credential-safe plain configuration mapping.

    Args:
        config:
            The complete Hydra configuration.

    Raises:
        ValueError:
            If the resolved configuration is not a mapping.
    """
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(resolved, dict):
        raise ValueError("W&B configuration payload must be a mapping")
    return t.cast(dict[str, object], _remove_sensitive_values(resolved))


def _remove_sensitive_values(value: object, key: str = "") -> object:
    """Remove credential-shaped fields from a recursively plain value.

    Args:
        value:
            The value to inspect.
        key (optional):
            The parent mapping key. Defaults to an empty string.

    Returns:
        A recursively redacted plain value.
    """
    lowered_key = key.lower()
    if any(
        marker in lowered_key for marker in ("token", "secret", "password", "api_key")
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(child_key): _remove_sensitive_values(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_remove_sensitive_values(item) for item in value]
    return value


def _set_artifact_logging_environment(tracking: DictConfig) -> None:
    """Apply the configured W&B artefact and parameter logging policy."""
    os.environ["WANDB_LOG_MODEL"] = str(tracking.get("log_model", False)).lower()
    os.environ["WANDB_WATCH"] = str(tracking.get("watch", False)).lower()


def validate_wandb_config(config: DictConfig) -> None:
    """Reject W&B settings that could silently lose production tracking.

    Args:
        config:
            The complete Hydra configuration.

    Raises:
        ValueError:
            If W&B settings are incomplete or inconsistent.
    """
    tracking = config.get("experiment_tracking")
    if tracking is None or tracking.get("type") != "wandb":
        raise ValueError("W&B setup requires experiment_tracking.type=wandb")

    project = str(tracking.get("name_experiment") or "").strip()
    if not project:
        raise ValueError("W&B project must not be empty")
    entity = tracking.get("entity")
    if entity is not None and not str(entity).strip():
        raise ValueError("W&B entity must be unset or a non-empty workspace")
    if not str(tracking.get("name_run") or "").strip():
        raise ValueError("W&B run name must not be empty")
    if not str(tracking.get("name_group") or "").strip():
        raise ValueError("W&B run group must not be empty")

    mode = tracking.get("mode")
    if mode is not None and mode not in {"online", "offline", "disabled", "shared"}:
        raise ValueError(f"Unsupported W&B mode: {mode}")
    resume = tracking.get("resume")
    if resume is not None and resume not in {
        True,
        False,
        "allow",
        "never",
        "must",
        "auto",
    }:
        raise ValueError(f"Unsupported W&B resume policy: {resume}")
    run_id = tracking.get("id")
    if resume in {True, "must"} and not str(run_id or "").strip():
        raise ValueError("W&B resume policy requires an explicit run ID")

    tags = tracking.get("tags")
    if tags is not None and not OmegaConf.is_list(tags):
        raise ValueError("W&B tags must be a list")

    if tracking.get("production"):
        if mode != "online":
            raise ValueError("Production W&B tracking must use mode=online")
        if tracking.get("log_model") is not False:
            raise ValueError("Production W&B model artefact logging must be disabled")
        if tracking.get("watch") is not False:
            raise ValueError("Production W&B parameter watching must be disabled")


def preflight_wandb_access(
    config: DictConfig,
    login: c.Callable[..., bool] | None = None,
    api_factory: c.Callable[[], object] | None = None,
) -> None:
    """Verify stored W&B credentials and online API access without logging a key.

    Args:
        config:
            The complete Hydra configuration.
        login (optional):
            Credential verification callable, injectable for unit tests.
        api_factory (optional):
            Authenticated API client factory, injectable for unit tests.

    Raises:
        RuntimeError:
            If credentials or online W&B access are unavailable.
    """
    validate_wandb_config(config=config)
    login_fn = login or wandb.login
    api_factory_fn = api_factory or wandb.Api
    if not login_fn(verify=True):
        raise RuntimeError("W&B authentication verification failed")
    try:
        viewer = getattr(api_factory_fn(), "viewer")
    except Exception as error:
        del error
        raise RuntimeError("W&B online API access failed") from None
    if viewer is None:
        raise RuntimeError("W&B authentication verification returned no user")
