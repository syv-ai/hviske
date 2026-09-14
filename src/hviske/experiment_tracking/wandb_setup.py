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

    def run_finalization(self, exit_code: int = 0) -> None:
        """Finish the W&B run and report its process exit status.

        Args:
            exit_code (optional):
                The process exit code to report. Defaults to ``0``.
        """
        wandb.finish(exit_code=exit_code)  # type: ignore[attr-defined]

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


_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "client_secret",
    "cookie",
    "credentials",
    "hf_token",
    "huggingface_hub_token",
    "huggingface_token",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "token",
    "wandb_api_key",
}
_PATH_KEY_SUFFIXES = ("_dir", "_path")
_PATH_KEYS = {
    "cache",
    "cache_dir",
    "checkpoint_dir",
    "logging_dir",
    "manifest",
    "manifest_path",
    "model_dir",
    "output",
    "path",
    "models_dir",
    "output_dir",
    "source_dir",
    "source_wav_path",
}


def _remove_sensitive_values(value: object, key: str = "") -> object:
    """Redact credentials and local filesystem locations in a plain value.

    Credential matching is deliberately exact: a benign hyperparameter such as
    ``tokenizer_token`` must remain useful in the W&B configuration.

    Args:
        value:
            The value to inspect.
        key (optional):
            The parent mapping key. Defaults to an empty string.

    Returns:
        A recursively redacted plain value.
    """
    lowered_key = key.lower().replace("-", "_")
    if value is not None and lowered_key in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if value is not None and (
        lowered_key in _PATH_KEYS or lowered_key.endswith(_PATH_KEY_SUFFIXES)
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
    """Apply explicitly configured W&B artefact and parameter policies."""
    for config_key, environment_key in (
        ("log_model", "WANDB_LOG_MODEL"),
        ("watch", "WANDB_WATCH"),
    ):
        value = tracking.get(config_key)
        if value is not None:
            os.environ[environment_key] = str(value).lower()


def preflight_wandb_access(
    config: DictConfig, api_factory: c.Callable[[], object] | None = None
) -> None:
    """Verify stored W&B credentials and online API access non-interactively.

    This deliberately does not call :func:`wandb.login`: the explicit CLI login is
    the only allowed credential prompt, and preflight must be safe for automation.

    Args:
        config:
            The complete Hydra configuration.
        api_factory (optional):
            Authenticated API client factory, injectable for unit tests.

    Raises:
        RuntimeError:
            If stored credentials are unavailable or the online API cannot be read.
    """
    validate_wandb_config(config=config)
    api_factory_fn = api_factory or wandb.Api
    try:
        api = api_factory_fn()
        viewer = getattr(api, "viewer")
    except Exception as error:
        raise RuntimeError(
            "W&B preflight could not use stored credentials or reach the online API; "
            "run `uv run wandb login --verify` first"
        ) from error
    if viewer is None:
        raise RuntimeError(
            "W&B preflight found no authenticated viewer; run "
            "`uv run wandb login --verify` first"
        )


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
    if resume in {"allow", "auto"} and not str(run_id or "").strip():
        raise ValueError(
            "W&B resume=allow/auto requires an explicit run ID; use resume=never "
            "for a fresh run"
        )

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
