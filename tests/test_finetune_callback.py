"""Regression tests for benchmark evaluation callbacks."""

import json
from pathlib import Path

import pytest
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from hviske.finetune import EvaluationScheduleCallback, StopAfterStepCallback


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
