"""Regression tests for benchmark evaluation callbacks."""

import json
import typing as t
from contextlib import nullcontext
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import hviske.finetune as finetune_module
from hviske.callbacks import EvaluationScheduleCallback, StopAfterStepCallback


@pytest.mark.parametrize(
    "early_stopping",
    [False, True],
    ids=["without_early_stopping", "with_early_stopping"],
)
def test_bounded_terminal_evaluation_runs_after_training_workers_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, early_stopping: bool
) -> None:
    """Defer terminal evaluation only when early stopping is disabled."""
    events: list[str] = []
    metrics_path = tmp_path / "metrics.jsonl"

    class Processor:
        tokenizer = object()

        def save_pretrained(self, save_directory: str) -> None:
            del save_directory

    class Model:
        def save_pretrained(self, save_directory: str) -> None:
            del save_directory

    class RecordingTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.args = TrainingArguments(output_dir=str(tmp_path))
            self.args.load_best_model_at_end = early_stopping
            self.callbacks = t.cast(list[TrainerCallback], kwargs["callbacks"])
            self.eval_dataset = t.cast(dict[str, object], kwargs["eval_dataset"])
            self.state = TrainerState(global_step=0)
            self.training_workers_alive = False
            self.evaluations: list[tuple[str, int, tuple[str, ...]]] = []
            trainer_instances.append(self)

        def evaluate(self) -> None:
            assert self.training_workers_alive is False
            self._record_evaluation(source="deferred")

        def _record_evaluation(self, source: str) -> None:
            step = self.state.global_step
            dataset_names = tuple(self.eval_dataset)
            events.append(f"evaluate:{source}:{step}")
            self.evaluations.append((source, step, dataset_names))
            for dataset_name in dataset_names:
                logs = {f"eval_{dataset_name}_cer": step / 100}
                for callback in self.callbacks:
                    callback.on_log(self.args, self.state, TrainerControl(), logs=logs)

        def train(self, resume_from_checkpoint: object) -> None:
            del resume_from_checkpoint
            self.training_workers_alive = True
            for step in range(1, 5):
                self.state.global_step = step
                control = TrainerControl(should_evaluate=True, should_save=step == 4)
                for callback in self.callbacks:
                    callback.on_step_end(self.args, self.state, control)
                if control.should_evaluate:
                    self._record_evaluation(source="in-loop")
                if control.should_save:
                    events.append(f"checkpoint:{step}")
                if control.should_training_stop:
                    break
            self.training_workers_alive = False
            events.append("train-return")

    trainer_instances: list[RecordingTrainer] = []

    class ModelSetup:
        def load_compute_metrics(self) -> None:
            return None

        def load_data_collator(self) -> None:
            return None

        def load_model(self) -> Model:
            return Model()

        def load_processor(self) -> Processor:
            return Processor()

        def load_trainer_class(self) -> type[RecordingTrainer]:
            return RecordingTrainer

        def load_training_arguments(self) -> object:
            return object()

    config = OmegaConf.create(
        {
            "dataloader_num_workers": 0,
            "enable_experiment_tracking": False,
            "model_dir": str(tmp_path / "model"),
            "resume_from_checkpoint": False,
            "early_stopping": early_stopping,
            "early_stopping_patience": 2,
            "push_to_hub": False,
            "model": {"use_decoder": False},
            "evaluation_steps": [2, 4],
            "stop_after_steps": 4,
            "evaluation_metrics_path": str(metrics_path),
        }
    )
    monkeypatch.setattr(
        finetune_module, "validate_private_only_config", lambda config: None
    )
    monkeypatch.setattr(finetune_module, "download_background_noises", lambda: None)
    monkeypatch.setattr(
        finetune_module, "load_model_setup", lambda config: ModelSetup()
    )
    monkeypatch.setattr(
        finetune_module,
        "load_data_for_finetuning",
        lambda config, processor: {
            "train": object(),
            "val_danish": object(),
            "val_english": object(),
        },
    )
    monkeypatch.setattr(finetune_module, "block_terminal_output", lambda: None)
    monkeypatch.setattr(finetune_module, "disable_tqdm", nullcontext)

    finetune_module.finetune(config=config)

    trainer = trainer_instances[0]
    assert trainer.state.global_step == 4
    expected_source = "in-loop" if early_stopping else "deferred"
    assert trainer.evaluations == [
        ("in-loop", 2, ("val_danish", "val_english")),
        (expected_source, 4, ("val_danish", "val_english")),
    ]
    assert events == (
        ["evaluate:in-loop:2", "evaluate:in-loop:4", "checkpoint:4", "train-return"]
        if early_stopping
        else [
            "evaluate:in-loop:2",
            "checkpoint:4",
            "train-return",
            "evaluate:deferred:4",
        ]
    )
    metric_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in metric_lines]
    assert records == [
        {"eval_val_danish_cer": 0.02, "step": 2},
        {"eval_val_english_cer": 0.02, "step": 2},
        {"eval_val_danish_cer": 0.04, "step": 4},
        {"eval_val_english_cer": 0.04, "step": 4},
    ]


def test_evaluation_schedule_suppresses_non_scheduled_steps(tmp_path: Path) -> None:
    """Only explicitly requested global steps trigger evaluation."""
    callback = EvaluationScheduleCallback(evaluation_steps=[250])
    args = TrainingArguments(output_dir=str(tmp_path))
    control = TrainerControl(should_evaluate=True)

    callback.on_step_end(args, TrainerState(global_step=249), control)
    assert control.should_evaluate is False
    control.should_evaluate = False
    callback.on_step_end(args, TrainerState(global_step=250), control)
    assert control.should_evaluate is True


def test_evaluation_schedule_writes_global_step_tagged_metrics(tmp_path: Path) -> None:
    """Evaluation logs retain the Trainer global step in JSONL output."""
    metrics_path = tmp_path / "metrics.jsonl"
    callback = EvaluationScheduleCallback(
        evaluation_steps=[250], metrics_path=metrics_path
    )
    callback.on_log(
        TrainingArguments(output_dir=str(tmp_path)),
        TrainerState(global_step=250),
        TrainerControl(),
        logs={"eval_cer": 0.12, "loss": 0.4, "epoch": 1.0},
    )

    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert record == {"eval_cer": 0.12, "loss": 0.4, "step": 250}


def test_stop_after_step_preserves_scheduler_horizon() -> None:
    """A bounded stop requests termination without changing Trainer arguments."""
    callback = StopAfterStepCallback(stop_after_steps=2_000)
    args = TrainingArguments(output_dir="test-output", max_steps=100_000)
    control = TrainerControl()

    callback.on_step_end(args, TrainerState(global_step=1_999), control)
    assert control.should_training_stop is False
    callback.on_step_end(args, TrainerState(global_step=2_000), control)
    assert control.should_training_stop is True
    assert args.max_steps == 100_000


def test_stop_after_step_rejects_non_positive_step() -> None:
    """A bounded run must have a reachable positive stopping step."""
    with pytest.raises(ValueError, match="positive"):
        StopAfterStepCallback(stop_after_steps=0)
