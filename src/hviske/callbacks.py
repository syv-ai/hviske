"""Training callbacks shared by fine-tuning workflows."""

import json
from pathlib import Path

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .dataloader_shutdown import DataLoaderShutdownController


class DataLoaderShutdownCallback(TrainerCallback):
    """Signal DataLoader workers once Transformers reaches a terminal state."""

    def __init__(self, controller: DataLoaderShutdownController) -> None:
        """Initialise the callback with the current run's controller.

        Args:
            controller:
                Controller publishing the inherited worker sentinel.
        """
        self.controller = controller

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Signal natural epoch completion when no evaluation remains in-loop.

        Returns:
            The unchanged trainer control object.
        """
        del args, state, kwargs
        if control.should_training_stop and not control.should_evaluate:
            self.controller.request_shutdown()
        return control

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Signal after terminal evaluation callbacks have completed.

        Returns:
            The unchanged trainer control object.
        """
        del args, state, kwargs
        if control.should_training_stop:
            self.controller.request_shutdown()
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Signal after terminal step callbacks, unless evaluation is pending.

        Returns:
            The unchanged trainer control object.
        """
        del args, state, kwargs
        if control.should_training_stop and not control.should_evaluate:
            self.controller.request_shutdown()
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Request cooperative worker shutdown as a final compatibility fallback."""
        del args, state, control, kwargs
        self.controller.request_shutdown()


class EvaluationScheduleCallback(TrainerCallback):
    """Restrict step-based evaluation to a finite, non-uniform schedule."""

    def __init__(
        self, evaluation_steps: list[int], metrics_path: str | Path | None = None
    ) -> None:
        """Initialise the callback with the permitted trainer steps.

        Args:
            evaluation_steps:
                Global steps at which validation should run.
            metrics_path (optional):
                JSONL destination for step-tagged metrics. Defaults to ``None``.
        """
        self.evaluation_steps = set(evaluation_steps)
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float | int | bool] | None = None,
        **kwargs: object,
    ) -> None:
        """Write machine-readable metrics with the Trainer's global step."""
        del args, control, kwargs
        if self.metrics_path is None or not logs:
            return
        metrics = {
            key: value
            for key, value in logs.items()
            if key.startswith("eval_") or key in {"loss", "learning_rate"}
        }
        if not metrics:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"step": state.global_step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Evaluate only at the configured global steps.

        Returns:
            The updated trainer control object.
        """
        del args, kwargs
        control.should_evaluate = state.global_step in self.evaluation_steps
        return control


class StopAfterStepCallback(TrainerCallback):
    """Stop training after a step without changing the scheduler horizon."""

    def __init__(self, stop_after_steps: int) -> None:
        """Initialise the callback with the final permitted global step.

        Args:
            stop_after_steps:
                Global step at which training should stop.

        Raises:
            ValueError:
                If ``stop_after_steps`` is not positive.
        """
        if stop_after_steps < 1:
            raise ValueError("stop_after_steps must be positive")
        self.stop_after_steps = stop_after_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Request termination once the configured global step is reached.

        Returns:
            The updated trainer control object.
        """
        del args, kwargs
        if state.global_step >= self.stop_after_steps:
            control.should_training_stop = True
        return control
