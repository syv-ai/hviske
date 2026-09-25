"""Weights & Biases experiment tracking setup and preflight helpers."""

import collections.abc as c
import logging
import math
import os
import re
import typing as t

import wandb
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib import auth as wandb_auth

from ..hub_access_health import HubAccessHealthMonitor
from .extracking_setup import ExTrackingSetup

logger = logging.getLogger(__name__)

_ALERT_MESSAGE_LIMIT = 512
_ALERT_IDENTITY_LIMIT = 128
_ALERT_DIAGNOSTIC_LIMIT = 256
_ALERT_REDACTED = "[REDACTED]"
_URL_PATTERN = re.compile(r"(?i)\b(?:https?|s3|gs|file)://[^\s<>\"']+")
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?P<key>api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
    r"client[_ -]?secret|password|passwd|refresh[_ -]?token|secret|token)"
    r"\s*(?:=|:)\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\b(?:basic|bearer)\s+[^\s,;]+")
_POSIX_PATH_PATTERN = re.compile(
    r"(?<![\w])(?:~|/)(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+"
)
_WINDOWS_PATH_PATTERN = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\)[^\s,;]+")
_RELATIVE_PATH_PATTERN = re.compile(r"(?<![\w])(?:\.\.?/)[^\s,;]+")


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
        self._hub_access_monitor: HubAccessHealthMonitor | None = None

    @staticmethod
    def _log_hub_access_health(metrics: c.Mapping[str, float]) -> None:
        log_fn = t.cast(c.Callable[[dict[str, float]], None], wandb.log)
        log_fn(dict(metrics))

    def report_failure(self, error: BaseException) -> None:
        """Send a best-effort alert for an unhandled online-run failure.

        Args:
            error:
                The exception that will be re-raised by the training workflow.
        """
        if isinstance(error, KeyboardInterrupt):
            return

        run = getattr(wandb, "run", None)
        if run is None or getattr(run, "disabled", False):
            return
        mode = getattr(run, "mode", None)
        if mode is None:
            settings = getattr(run, "settings", None)
            mode = getattr(settings, "mode", None)
        if mode is None:
            mode = os.environ.get("WANDB_MODE")
        if mode is not None and str(mode).lower() != "online":
            return

        exception_type = _sanitise_alert_value(
            type(error).__name__, limit=_ALERT_IDENTITY_LIMIT
        )
        identity_fields: list[str] = []
        for field in ("entity", "project", "name", "id"):
            identity_value = _sanitise_alert_value(
                getattr(run, field, None), limit=_ALERT_IDENTITY_LIMIT
            )
            identity_fields.append(f"{field}={identity_value}")
        run_identity = ", ".join(identity_fields)
        message = _sanitise_alert_message(error)
        alert_levels = getattr(wandb, "AlertLevel", None)
        error_level = getattr(alert_levels, "ERROR", "error")
        try:
            wandb.alert(  # type: ignore[attr-defined]
                title=f"Training failed: {exception_type}",
                text=(
                    f"W&B run identity: {run_identity}\n"
                    f"Exception: {exception_type}\n"
                    f"Message: {message}"
                ),
                level=error_level,
            )
        except BaseException:
            logger.exception("W&B failure alert could not be delivered")

    def run_finalization(self, exit_code: int = 0) -> None:
        """Finish the W&B run and report its process exit status.

        Args:
            exit_code (optional):
                The process exit code to report. Defaults to ``0``.
        """
        if self._hub_access_monitor is not None:
            try:
                self._hub_access_monitor.stop()
            except BaseException:
                logger.exception(
                    "Hugging Face Hub health monitor failed to stop; "
                    "tracking finalisation will continue"
                )
            finally:
                self._hub_access_monitor = None
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
        monitor = HubAccessHealthMonitor(
            report=self._log_hub_access_health,
            heartbeat_seconds=float(tracking.get("hub_access_heartbeat_seconds", 30.0)),
        )
        try:
            monitor.start()
        except BaseException:
            monitor.stop()
            logger.exception(
                "Hugging Face Hub health monitor failed to start; "
                "training will continue"
            )
        else:
            self._hub_access_monitor = monitor


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


def _sanitise_alert_message(value: object) -> str:
    """Return useful, bounded failure context without exposing private data.

    URLs, credentials, and local paths are removed before the message length is
    considered. Longer messages are replaced entirely because a bounded prefix
    could still contain private transcript or sample content.

    Returns:
        A safe, single-line diagnostic or a redaction marker.
    """
    try:
        text = str(value)
    except BaseException:
        return "<unavailable>"
    text = "".join(char if char.isprintable() else " " for char in text)
    text = " ".join(text.split())
    text = _URL_PATTERN.sub(_ALERT_REDACTED, text)
    text = _CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group('key')}={_ALERT_REDACTED}", text
    )
    text = _BEARER_PATTERN.sub(_ALERT_REDACTED, text)
    text = _POSIX_PATH_PATTERN.sub(_ALERT_REDACTED, text)
    text = _WINDOWS_PATH_PATTERN.sub(_ALERT_REDACTED, text)
    text = _RELATIVE_PATH_PATTERN.sub(_ALERT_REDACTED, text)
    if len(text) > _ALERT_DIAGNOSTIC_LIMIT:
        return f"[REDACTED: diagnostic exceeded {_ALERT_DIAGNOSTIC_LIMIT} characters]"
    return text[:_ALERT_MESSAGE_LIMIT] or "<empty>"


def _sanitise_alert_value(value: object, *, limit: int) -> str:
    """Convert an alert value to bounded, single-line printable text.

    Returns:
        A printable, whitespace-normalised value no longer than ``limit``.
    """
    try:
        text = str(value)
    except BaseException:
        return "<unavailable>"
    text = "".join(char if char.isprintable() else " " for char in text)
    text = " ".join(text.split())
    return text[:limit] or "<empty>"


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
    "resume_from_checkpoint",
    "source_dir",
    "source_wav_path",
}


def _read_stored_wandb_api_key() -> str | None:
    """Read a W&B key from settings, the environment, or netrc without prompting.

    Returns:
        The stored W&B API key, or ``None`` when no key is configured.
    """
    settings = wandb.Settings()
    api_key = getattr(settings, "api_key", None) or os.environ.get("WANDB_API_KEY")
    if api_key:
        return str(api_key)

    base_url = os.environ.get("WANDB_BASE_URL") or str(settings.base_url)
    return wandb_auth.read_netrc_auth(host=base_url)


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
    if lowered_key == "resume_from_checkpoint":
        return value if value is None or isinstance(value, bool) else "[REDACTED]"
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
    config: DictConfig,
    api_factory: c.Callable[..., object] | None = None,
    credential_lookup: c.Callable[[], str | None] | None = None,
) -> None:
    """Verify stored W&B credentials and online API access non-interactively.

    This deliberately does not call :func:`wandb.login`: the explicit CLI login is
    the only allowed credential prompt, and preflight must be safe for automation.

    Args:
        config:
            The complete Hydra configuration.
        api_factory (optional):
            Authenticated API client factory, injectable for unit tests.
        credential_lookup (optional):
            Non-interactive stored-key lookup, injectable for unit tests.

    Raises:
        RuntimeError:
            If stored credentials are unavailable or the online API cannot be read.
    """
    validate_wandb_config(config=config)
    credential_lookup_fn = credential_lookup or _read_stored_wandb_api_key
    try:
        api_key = credential_lookup_fn()
    except Exception as error:
        raise RuntimeError(
            "W&B preflight could not use stored credentials or reach the online API; "
            "run `uv run wandb login --verify` first"
        ) from error
    if not api_key:
        raise RuntimeError(
            "W&B preflight could not use stored credentials or reach the online API; "
            "run `uv run wandb login --verify` first"
        )

    api_factory_fn = api_factory or wandb.Api
    try:
        api = api_factory_fn(api_key=api_key)
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

    heartbeat_seconds = tracking.get("hub_access_heartbeat_seconds", 30.0)
    if (
        not isinstance(heartbeat_seconds, int | float)
        or isinstance(heartbeat_seconds, bool)
        or not math.isfinite(heartbeat_seconds)
        or heartbeat_seconds <= 0
    ):
        raise ValueError(
            "W&B Hub access heartbeat interval must be positive and finite"
        )

    if tracking.get("production"):
        if mode != "online":
            raise ValueError("Production W&B tracking must use mode=online")
        if tracking.get("log_model") is not False:
            raise ValueError("Production W&B model artefact logging must be disabled")
        if tracking.get("watch") is not False:
            raise ValueError("Production W&B parameter watching must be disabled")
