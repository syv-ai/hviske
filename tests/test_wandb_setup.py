"""Focused tests for production W&B setup."""

import os
import typing as t
from contextlib import nullcontext
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

import hviske.experiment_tracking.mlflow_setup as mlflow_module
import hviske.experiment_tracking.wandb_setup as wandb_module
import hviske.finetune as finetune_module
from hviske.experiment_tracking.wandb_setup import WandbSetup, preflight_wandb_access


def test_generic_wandb_policy_preserves_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset generic policy values do not overwrite caller environment controls."""
    monkeypatch.setenv("WANDB_LOG_MODEL", "true")
    monkeypatch.setenv("WANDB_WATCH", "all")
    monkeypatch.setattr(wandb_module.wandb, "init", lambda **kwargs: None)

    WandbSetup(
        config=_config(production=False, log_model=None, watch=None)
    ).run_initialization()

    assert os.environ["WANDB_LOG_MODEL"] == "true"
    assert os.environ["WANDB_WATCH"] == "all"


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


def test_health_monitor_start_failure_does_not_block_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health telemetry startup cannot fail W&B run initialisation."""

    class BrokenMonitor:
        def __init__(self, **_: object) -> None:
            pass

        @staticmethod
        def start() -> None:
            raise RuntimeError("monitor startup failed")

        @staticmethod
        def stop() -> None:
            pass

    monkeypatch.setattr(wandb_module, "HubAccessHealthMonitor", BrokenMonitor)
    monkeypatch.setattr(wandb_module.wandb, "init", lambda **_: None)

    WandbSetup(config=_config()).run_initialization()


def test_health_monitor_stop_failure_does_not_block_wandb_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health telemetry cleanup cannot terminate tracking finalisation."""
    exit_codes: list[int] = []

    class BrokenMonitor:
        @staticmethod
        def stop() -> None:
            raise RuntimeError("monitor cleanup failed")

    monkeypatch.setattr(
        wandb_module.wandb,
        "finish",
        lambda **kwargs: exit_codes.append(int(kwargs["exit_code"])),
    )
    setup = WandbSetup(config=_config())
    monkeypatch.setattr(setup, "_hub_access_monitor", BrokenMonitor())

    setup.run_finalization(exit_code=1)

    assert exit_codes == [1]
    assert setup._hub_access_monitor is None


def test_mlflow_finalization_marks_failed_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLflow retains its backend while recording failed training."""
    statuses: list[str] = []
    monkeypatch.setattr(
        mlflow_module.mlflow,
        "end_run",
        lambda **kwargs: statuses.append(str(kwargs["status"])),
    )

    mlflow_module.MLFlowSetup(config=_config()).run_finalization(exit_code=1)

    assert statuses == ["FAILED"]


def test_production_wandb_rejects_offline_mode() -> None:
    """The production preset cannot silently become offline."""
    with pytest.raises(ValueError, match="mode=online"):
        WandbSetup(config=_config(mode="offline"))


def test_tracking_failure_preserves_original_exception_when_finish_fails() -> None:
    """A finalisation failure cannot mask the original training exception.

    Raises:
        ValueError:
            The original training exception raised by the test body.
    """

    class Setup:
        def run_finalization(self, exit_code: int = 0) -> None:
            del exit_code
            raise RuntimeError("tracking unavailable")

    with pytest.raises(ValueError, match="training failed"):
        try:
            raise ValueError("training failed")
        except ValueError:
            finetune_module._finalize_tracking_after_failure(
                t.cast(finetune_module.ExTrackingSetup, Setup())
            )
            raise


def test_tracking_finalizes_after_publication_inputs_are_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful tracking finalisation is ordered after the final model save."""
    events: list[str] = []

    class Setup:
        def run_finalization(self, exit_code: int = 0) -> None:
            events.append(f"finish:{exit_code}")

        def run_initialization(self) -> None:
            events.append("initialise")

    class Processor:
        tokenizer = object()

        def save_pretrained(self, save_directory: str) -> None:
            del save_directory
            events.append("processor")

    class Model:
        def save_pretrained(self, save_directory: str) -> None:
            del save_directory
            events.append("model")

    class Trainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def train(self, **kwargs: object) -> None:
            del kwargs
            events.append("train")

    class ModelSetup:
        def load_compute_metrics(self) -> None:
            return None

        def load_data_collator(self) -> None:
            return None

        def load_model(self) -> Model:
            return Model()

        def load_processor(self) -> Processor:
            return Processor()

        def load_trainer_class(self) -> type[Trainer]:
            return Trainer

        def load_training_arguments(self) -> object:
            return object()

    config = OmegaConf.merge(
        _config(),
        OmegaConf.create(
            {
                "model_dir": str(tmp_path),
                "model": {"use_decoder": False},
                "early_stopping": False,
                "resume_from_checkpoint": False,
                "push_to_hub": False,
            }
        ),
    )
    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(
        finetune_module, "load_extracking_setup", lambda config: Setup()
    )
    monkeypatch.setattr(finetune_module, "download_background_noises", lambda: None)
    monkeypatch.setattr(
        finetune_module, "load_model_setup", lambda config: ModelSetup()
    )
    monkeypatch.setattr(
        finetune_module,
        "load_data_for_finetuning",
        lambda config, processor: {"train": [], "val": []},
    )
    monkeypatch.setattr(finetune_module, "disable_tqdm", lambda: nullcontext())

    finetune_module.finetune(config=t.cast(DictConfig, config))

    assert events[-1] == "finish:0"
    assert events.index("finish:0") > events.index("model")


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


def test_training_failure_alert_precedes_tracking_finalisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure alerting runs while the tracking run is still active."""
    events: list[str] = []

    class Setup:
        def report_failure(self, error: BaseException) -> None:
            events.append(f"alert:{type(error).__name__}")

        def run_finalization(self, exit_code: int = 0) -> None:
            events.append(f"finish:{exit_code}")

        def run_initialization(self) -> None:
            events.append("initialise")

    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(
        finetune_module, "load_extracking_setup", lambda config: Setup()
    )
    monkeypatch.setattr(
        finetune_module,
        "download_background_noises",
        lambda: (_ for _ in ()).throw(RuntimeError("training failed")),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        finetune_module.finetune(config=_config())

    assert events == ["initialise", "alert:RuntimeError", "finish:1"]


def test_wandb_alert_delivery_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alert transport error cannot replace the training exception."""
    run = type("Run", (), {"mode": "online", "id": "run-id"})()
    monkeypatch.setattr(wandb_module.wandb, "run", run, raising=False)
    monkeypatch.setattr(
        wandb_module.wandb,
        "alert",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
        raising=False,
    )

    WandbSetup(config=_config()).report_failure(RuntimeError("training failed"))


def test_wandb_alert_includes_bounded_sanitised_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Online W&B alerts identify the run without unbounded control characters."""
    alerts: list[dict[str, object]] = []
    run = type(
        "Run",
        (),
        {
            "entity": "team",
            "project": "hviske",
            "name": "v6.0",
            "id": "gkxzy9dn",
            "mode": "online",
        },
    )()
    monkeypatch.setattr(wandb_module.wandb, "run", run, raising=False)
    monkeypatch.setattr(
        wandb_module.wandb,
        "alert",
        lambda **kwargs: alerts.append(dict(kwargs)),
        raising=False,
    )

    setup = WandbSetup(config=_config())
    setup.report_failure(RuntimeError("bad\n\x00" + "x" * 600))

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["title"] == "Training failed: RuntimeError"
    text = str(alert["text"])
    assert "gkxzy9dn" in text
    assert "Exception: RuntimeError" in text
    assert "\\n" not in text
    assert len(text) < 900


def test_wandb_alert_redacts_credentials_urls_and_local_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure alerts do not expose credentials or filesystem and URL details."""
    alerts: list[dict[str, object]] = []
    run = type("Run", (), {"mode": "online", "id": "run-id"})()
    monkeypatch.setattr(wandb_module.wandb, "run", run, raising=False)
    monkeypatch.setattr(
        wandb_module.wandb,
        "alert",
        lambda **kwargs: alerts.append(dict(kwargs)),
        raising=False,
    )

    secret = "not-for-wandb"
    message = (
        f"request failed token={secret} "
        f"https://example.test/audio?X-Amz-Signature={secret} "
        f"/private/training/{secret}.json"
    )
    WandbSetup(config=_config()).report_failure(RuntimeError(message))

    text = str(alerts[0]["text"])
    assert secret not in text
    assert "X-Amz-Signature" not in text
    assert "/private/training" not in text
    assert "Exception: RuntimeError" in text


def test_wandb_alert_redacts_long_private_sample_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long exception payloads are not retained as a private sample excerpt."""
    alerts: list[dict[str, object]] = []
    run = type("Run", (), {"mode": "online", "id": "run-id"})()
    monkeypatch.setattr(wandb_module.wandb, "run", run, raising=False)
    monkeypatch.setattr(
        wandb_module.wandb,
        "alert",
        lambda **kwargs: alerts.append(dict(kwargs)),
        raising=False,
    )

    private_sample = "private-danish-transcript-" + ("sample-word " * 80)
    WandbSetup(config=_config()).report_failure(RuntimeError(private_sample))

    text = str(alerts[0]["text"])
    assert private_sample not in text
    assert "sample-word" not in text
    assert "Exception: RuntimeError" in text
    assert len(text) < 900


def test_wandb_alert_skips_keyboard_interrupt_and_non_online_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline, disabled, and manually interrupted runs do not alert."""
    alerts: list[dict[str, object]] = []
    monkeypatch.setattr(
        wandb_module.wandb,
        "alert",
        lambda **kwargs: alerts.append(dict(kwargs)),
        raising=False,
    )
    setup = WandbSetup(config=_config())
    for mode in ("offline", "disabled"):
        run = type("Run", (), {"mode": mode, "id": "run-id"})()
        monkeypatch.setattr(wandb_module.wandb, "run", run, raising=False)
        setup.report_failure(RuntimeError("failed"))
    setup.report_failure(KeyboardInterrupt())

    assert alerts == []


def test_wandb_finalization_passes_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """W&B receives success and failure status from the training process."""
    exit_codes: list[int] = []
    monkeypatch.setattr(
        wandb_module.wandb,
        "finish",
        lambda **kwargs: exit_codes.append(int(kwargs["exit_code"])),
    )

    setup = WandbSetup(config=_config())
    setup.run_finalization(exit_code=0)
    setup.run_finalization(exit_code=1)

    assert exit_codes == [0, 1]


def test_wandb_init_uses_plain_resolved_safe_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialisation passes production fields and starts Hub health reporting."""
    calls: list[dict[str, object]] = []
    health: list[dict[str, float]] = []
    monkeypatch.setattr(
        wandb_module.wandb, "init", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        wandb_module.wandb, "log", lambda metrics: health.append(dict(metrics))
    )
    monkeypatch.setattr(wandb_module.wandb, "finish", lambda **_: None)
    monkeypatch.setenv("WANDB_LOG_MODEL", "true")
    monkeypatch.setenv("WANDB_WATCH", "all")

    setup = WandbSetup(config=_config())
    setup.run_initialization()
    setup.run_finalization()
    assert os.environ["WANDB_LOG_MODEL"] == "false"
    assert os.environ["WANDB_WATCH"] == "false"

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
    assert health[0]["health/hub_access_blocked"] == 0
    assert setup._hub_access_monitor is None


def test_wandb_payload_redacts_resolved_paths_without_redacting_hyperparameters() -> (
    None
):
    """Resolved local paths and credentials are redacted precisely."""
    config = OmegaConf.merge(
        _config(),
        OmegaConf.create(
            {
                "model_dir": "models/run",
                "cache_dir": "/private/cache",
                "tokenizer_token": "benign-value",
                "wandb_api_key": "secret-value",
                "resume_from_checkpoint": "/private/checkpoints/checkpoint-2000",
                "datasets": {"p1": {"data_path": "/private/audio/p1.parquet"}},
            }
        ),
    )

    payload = wandb_module._resolved_config_payload(config=t.cast(DictConfig, config))

    assert payload["model_dir"] == "[REDACTED]"
    assert payload["cache_dir"] == "[REDACTED]"
    datasets = t.cast(dict[str, object], payload["datasets"])
    p1 = t.cast(dict[str, object], datasets["p1"])
    assert p1["data_path"] == "[REDACTED]"
    assert payload["tokenizer_token"] == "benign-value"
    assert payload["wandb_api_key"] == "[REDACTED]"
    assert payload["resume_from_checkpoint"] == "[REDACTED]"

    false_config = OmegaConf.merge(
        _config(), OmegaConf.create({"resume_from_checkpoint": False})
    )
    false_payload = wandb_module._resolved_config_payload(
        config=t.cast(DictConfig, false_config)
    )
    assert false_payload["resume_from_checkpoint"] is False


def test_wandb_preflight_default_path_does_not_prompt_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path never constructs an API client without a stored key."""
    monkeypatch.setattr(wandb_module, "_read_stored_wandb_api_key", lambda: None)
    monkeypatch.setattr(
        wandb_module.wandb,
        "login",
        lambda **kwargs: pytest.fail("preflight must not call wandb.login"),
    )
    monkeypatch.setattr(
        wandb_module.wandb,
        "Api",
        lambda **kwargs: pytest.fail("preflight must not construct wandb.Api"),
    )

    with pytest.raises(RuntimeError, match="stored credentials"):
        preflight_wandb_access(config=_config())


def test_wandb_preflight_reports_missing_credentials() -> None:
    """Missing stored credentials produce an actionable error."""
    with pytest.raises(RuntimeError, match="stored credentials"):
        preflight_wandb_access(config=_config(), credential_lookup=lambda: None)


def test_wandb_preflight_uses_api_without_a_login_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight only reads stored credentials and the online viewer."""
    monkeypatch.setattr(
        wandb_module.wandb,
        "login",
        lambda **kwargs: pytest.fail("preflight must not call wandb.login"),
    )

    class Api:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "stored-key"

        viewer = {"username": "personal"}

    preflight_wandb_access(
        config=_config(), api_factory=Api, credential_lookup=lambda: "stored-key"
    )


def test_wandb_rejects_ambiguous_fresh_resume() -> None:
    """A fresh run cannot use an implicit resumable W&B ID."""
    with pytest.raises(ValueError, match="explicit run ID"):
        WandbSetup(config=_config(id=None, resume="allow"))


@pytest.mark.parametrize("interval", [0, -1, float("inf"), True])
def test_wandb_rejects_invalid_hub_health_interval(interval: object) -> None:
    """Hub access heartbeats require a positive finite interval."""
    with pytest.raises(ValueError, match="heartbeat interval"):
        WandbSetup(config=_config(hub_access_heartbeat_seconds=interval))
