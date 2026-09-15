"""Tests for the finetuning DataLoader multiprocessing policy."""

import io
import json
import logging
import multiprocessing
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import DictConfig, OmegaConf

import hviske.finetune as finetune_module


def _probe_finetune_spawn_import(script_path: str, result_path: str) -> None:
    """Execute the finetuning entry point as a multiprocessing worker import."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
    captured_stderr = io.StringIO()
    sys.stderr = captured_stderr

    root_logger.info("before finetuning entry-point import")
    runpy.run_path(script_path, run_name="__mp_main__")
    root_logger.info("after finetuning entry-point import")

    Path(result_path).write_text(
        json.dumps(
            {
                "level": root_logger.level,
                "handlers": len(root_logger.handlers),
                "stderr": captured_stderr.getvalue(),
            }
        ),
        encoding="utf-8",
    )


def test_conflicting_start_method_fails_clearly(monkeypatch: MonkeyPatch) -> None:
    """Fork cannot be silently retained when DataLoader workers are enabled."""
    multiprocessing = finetune_module.torch_mp
    monkeypatch.setattr(multiprocessing, "get_start_method", Mock(return_value="fork"))
    monkeypatch.setattr(multiprocessing, "set_start_method", Mock())

    with pytest.raises(RuntimeError, match="already set to 'fork'.*"):
        finetune_module._configure_dataloader_multiprocessing(
            config=_config(dataloader_num_workers=4)
        )

    multiprocessing.set_start_method.assert_not_called()


def _config(*, dataloader_num_workers: int) -> DictConfig:
    """Build the smallest config accepted by the worker policy.

    Returns:
        A minimal Hydra configuration.
    """
    return OmegaConf.create({"dataloader_num_workers": dataloader_num_workers})


def test_existing_spawn_is_idempotent(monkeypatch: MonkeyPatch) -> None:
    """An already selected spawn context does not call the setter again."""
    multiprocessing = finetune_module.torch_mp
    get_start_method = Mock(return_value="spawn")
    set_start_method = Mock()
    monkeypatch.setattr(multiprocessing, "get_start_method", get_start_method)
    monkeypatch.setattr(multiprocessing, "set_start_method", set_start_method)

    finetune_module._configure_dataloader_multiprocessing(
        config=_config(dataloader_num_workers=4)
    )

    get_start_method.assert_called_once_with(allow_none=True)
    set_start_method.assert_not_called()


def test_spawn_is_configured_before_tracking_and_training_work(
    monkeypatch: MonkeyPatch,
) -> None:
    """Spawn is selected before tracking, model, or dataset initialisation."""
    events: list[str] = []
    multiprocessing = finetune_module.torch_mp
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **kwargs: None)
    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda method: events.append(method)
    )

    class Setup:
        def run_finalization(self, exit_code: int) -> None:
            del exit_code

        def run_initialization(self) -> None:
            events.append("tracking")

    class Processor:
        tokenizer = object()

        def save_pretrained(self, save_directory: str) -> None:
            del save_directory

    class Model:
        def save_pretrained(self, save_directory: str) -> None:
            del save_directory

    class Trainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def train(self, resume_from_checkpoint: object) -> None:
            del resume_from_checkpoint

    class ModelSetup:
        def load_compute_metrics(self) -> None:
            return None

        def load_data_collator(self) -> None:
            return None

        def load_model(self) -> Model:
            events.append("model")
            return Model()

        def load_processor(self) -> Processor:
            events.append("processor")
            return Processor()

        def load_trainer_class(self) -> type[Trainer]:
            return Trainer

        def load_training_arguments(self) -> None:
            return None

    config = OmegaConf.create(
        {
            "dataloader_num_workers": 4,
            "enable_experiment_tracking": True,
            "model_dir": "models/test",
            "resume_from_checkpoint": False,
            "early_stopping": False,
            "push_to_hub": False,
            "model": {"use_decoder": False},
        }
    )
    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(
        finetune_module,
        "load_extracking_setup",
        lambda config: events.append("load_tracking") or Setup(),
    )
    monkeypatch.setattr(
        finetune_module, "download_background_noises", lambda: events.append("noise")
    )
    monkeypatch.setattr(
        finetune_module,
        "load_model_setup",
        lambda config: events.append("load_model_setup") or ModelSetup(),
    )
    monkeypatch.setattr(
        finetune_module,
        "load_data_for_finetuning",
        lambda config, processor: events.append("data") or {"train": object()},
    )
    monkeypatch.setattr(finetune_module, "block_terminal_output", lambda: None)
    monkeypatch.setattr(finetune_module, "disable_tqdm", nullcontext)

    finetune_module.finetune(config=config)

    assert events[:3] == ["spawn", "load_tracking", "tracking"]
    assert events.index("spawn") < events.index("noise")
    assert events.index("spawn") < events.index("model")
    assert events.index("spawn") < events.index("data")


def test_spawn_worker_import_does_not_configure_root_logging(tmp_path: Path) -> None:
    """A spawn import does not emit INFO markers or install root handlers."""
    script_path = Path(__file__).parents[1] / "src/scripts/finetune_asr_model.py"
    result_path = tmp_path / "spawn-import-result.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_probe_finetune_spawn_import, args=(str(script_path), str(result_path))
    )
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join()

    assert process.exitcode == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result == {"level": logging.WARNING, "handlers": 0, "stderr": ""}


def test_zero_workers_do_not_configure_multiprocessing(
    monkeypatch: MonkeyPatch,
) -> None:
    """A zero-worker DataLoader leaves the process start method untouched."""
    multiprocessing = finetune_module.torch_mp
    get_start_method = Mock()
    set_start_method = Mock()
    monkeypatch.setattr(multiprocessing, "get_start_method", get_start_method)
    monkeypatch.setattr(multiprocessing, "set_start_method", set_start_method)

    finetune_module._configure_dataloader_multiprocessing(
        config=_config(dataloader_num_workers=0)
    )

    get_start_method.assert_not_called()
    set_start_method.assert_not_called()
