"""Finetuning ASR models."""

import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig
from transformers.trainer_callback import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments

from hviske.data import download_background_noises

from .data import load_data_for_finetuning
from .data_models import ModelSetup
from .experiment_tracking import ExTrackingSetup, load_extracking_setup
from .model_setup import load_model_setup
from .ngram import train_and_store_ngram_model
from .utils import (
    block_terminal_output,
    disable_tqdm,
    push_model_to_hub,
    validate_private_only_config,
)

logger = logging.getLogger(__package__)


def finetune(config: DictConfig) -> None:
    """Finetune a model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    validate_private_only_config(config=config)
    download_background_noises()

    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    model_setup: ModelSetup = load_model_setup(config=config)
    processor = model_setup.load_processor()
    dataset = load_data_for_finetuning(config=config, processor=processor)
    processor.save_pretrained(save_directory=config.model_dir)
    model = model_setup.load_model()

    extracking_setup: ExTrackingSetup | None = None
    if config.enable_experiment_tracking and is_main_process:
        extracking_setup = load_extracking_setup(config=config)
        extracking_setup.run_initialization()

    vals = {
        split_name: split
        for split_name, split in dataset.items()
        if split_name.startswith("val")
    }
    match len(vals):
        case 0:
            eval_dataset = None
        case 1:
            eval_dataset = list(vals.values())[0]
        case _:
            eval_dataset = vals

    if eval_dataset is None and is_main_process:
        logger.info("No validation set found. Disabling early stopping.")

    callbacks: list[TrainerCallback] = []
    evaluation_steps = config.get("evaluation_steps")
    if evaluation_steps:
        callbacks.append(
            EvaluationScheduleCallback(
                evaluation_steps=[int(step) for step in evaluation_steps],
                metrics_path=config.get("evaluation_metrics_path"),
            )
        )
    if eval_dataset is not None and config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )

    trainer = model_setup.load_trainer_class()(
        model=model,
        data_collator=model_setup.load_data_collator(),
        args=model_setup.load_training_arguments(),
        compute_metrics=model_setup.load_compute_metrics(),
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=getattr(processor, "tokenizer"),
        callbacks=callbacks or None,
    )

    block_terminal_output()
    with disable_tqdm():
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    if extracking_setup is not None and is_main_process:
        extracking_setup.run_finalization()

    model.save_pretrained(save_directory=config.model_dir)

    if hasattr(config.model, "use_decoder") and config.model.use_decoder:
        train_and_store_ngram_model(config=config)

    if config.push_to_hub:
        push_model_to_hub(
            trainer=trainer,
            model_name=config.model_id,
            finetuned_from=config.model.pretrained_model_id,
            create_pr=config.create_pr,
            private=config.private,
            private_only=config.get("private_only", False),
            model_card_languages=config.get("model_card_languages"),
            training_dataset_ids=list(
                config.get("training_dataset_ids")
                or [
                    str(dataset_config.id)
                    for dataset_config in config.datasets.values()
                ]
            ),
            evaluation_status=(
                config.get("evaluation_status")
                or (
                    "Evaluation ran during training."
                    if eval_dataset is not None
                    else "Not evaluated: no validation set was configured."
                )
            ),
        )


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
