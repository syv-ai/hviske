"""Finetuning ASR models."""

import gc
import json
import logging
import os
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as torch_mp
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
from .dataloader_shutdown import DataLoaderShutdownController
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
    check_cuda_requirement(config=config)
    _configure_dataloader_multiprocessing(config=config)
    validate_private_only_config(config=config)
    worker_shutdown = DataLoaderShutdownController(
        enabled=int(config.get("dataloader_num_workers") or 0) > 0
    )
    with worker_shutdown:
        extracking_setup: ExTrackingSetup | None = None
        try:
            # Note whether this is the main process in a distributed setting.
            is_main_process = os.getenv("RANK", "0") == "0"
            if config.enable_experiment_tracking and is_main_process:
                extracking_setup = load_extracking_setup(config=config)

            if extracking_setup is not None:
                extracking_setup.run_initialization()

            download_background_noises()
            model_setup: ModelSetup = load_model_setup(config=config)
            processor = model_setup.load_processor()
            dataset = load_data_for_finetuning(config=config, processor=processor)
            processor.save_pretrained(save_directory=config.model_dir)
            model = model_setup.load_model()

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
            evaluation_steps = [
                int(step) for step in (config.get("evaluation_steps") or [])
            ]
            configured_stop = config.get("stop_after_steps")
            stop_after_steps = (
                int(configured_stop) if configured_stop is not None else None
            )
            # In-loop validation keeps spawned training workers alive until it finishes.
            # Defer only when train() will not restore an earlier best model afterwards.
            deferred_evaluation_step = (
                stop_after_steps
                if (
                    eval_dataset is not None
                    and not config.early_stopping
                    and stop_after_steps in evaluation_steps
                )
                else None
            )
            if evaluation_steps:
                callbacks.append(
                    EvaluationScheduleCallback(
                        evaluation_steps=[
                            step
                            for step in evaluation_steps
                            if step != deferred_evaluation_step
                        ],
                        metrics_path=config.get("evaluation_metrics_path"),
                    )
                )
            if stop_after_steps is not None:
                callbacks.append(
                    StopAfterStepCallback(stop_after_steps=stop_after_steps)
                )
            if eval_dataset is not None and config.early_stopping:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=config.early_stopping_patience
                    )
                )
            callbacks.append(DataLoaderShutdownCallback(controller=worker_shutdown))

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
                try:
                    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
                except BaseException as error:
                    worker_shutdown.request_shutdown()
                    # Trainer frames retain iterators while traceback references live.
                    traceback.clear_frames(error.__traceback__)
                    trainer = None
                    gc.collect()
                    worker_shutdown.reset()
                    raise
                else:
                    # The callback normally signals while Trainer is unwinding. This
                    # also covers trainers that omit that callback.
                    worker_shutdown.request_shutdown()
                    worker_shutdown.reset()
                    completed_trainer = trainer
                if (
                    deferred_evaluation_step is not None
                    and completed_trainer.state.global_step == deferred_evaluation_step
                ):
                    completed_trainer.evaluate()

            model.save_pretrained(save_directory=config.model_dir)

            if hasattr(config.model, "use_decoder") and config.model.use_decoder:
                train_and_store_ngram_model(config=config)

            if config.push_to_hub:
                push_model_to_hub(
                    trainer=completed_trainer,
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
                    finetuned_from_revision=config.model.get("revision"),
                )
        except BaseException:
            if extracking_setup is not None:
                _finalize_tracking_after_failure(extracking_setup)
            raise
        else:
            if extracking_setup is not None:
                extracking_setup.run_finalization(exit_code=0)


class DataLoaderShutdownCallback(TrainerCallback):
    """Signal DataLoader workers during Transformers terminal callbacks."""

    def __init__(self, controller: DataLoaderShutdownController) -> None:
        """Initialise the callback with the current run's controller.

        Args:
            controller:
                Controller publishing the inherited worker sentinel.
        """
        self.controller = controller

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Request cooperative worker shutdown before Trainer returns."""
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


def _configure_dataloader_multiprocessing(config: DictConfig) -> None:
    """Configure worker creation before loading tracking or datasets.

    Args:
        config:
            The Hydra configuration object.

    Raises:
        RuntimeError:
            If another multiprocessing start method has already been selected.
    """
    dataloader_num_workers = int(config.get("dataloader_num_workers") or 0)
    if dataloader_num_workers <= 0:
        return

    start_method = torch_mp.get_start_method(allow_none=True)
    if start_method == "spawn":
        return
    if start_method is not None:
        raise RuntimeError(
            "PyTorch multiprocessing start method is already set to "
            f"{start_method!r}; finetuning with dataloader_num_workers > 0 requires "
            "'spawn'. Set the start method to 'spawn' before starting finetuning."
        )

    torch_mp.set_start_method("spawn")


def _finalize_tracking_after_failure(setup: ExTrackingSetup) -> None:
    """Best-effort finalisation that cannot replace the training exception."""
    try:
        setup.run_finalization(exit_code=1)
    except BaseException:
        logger.exception("Experiment tracking finalisation failed after training error")


def check_cuda_requirement(config: DictConfig) -> None:
    """Fail early when a configuration requires an unavailable CUDA device.

    Args:
        config:
            The Hydra configuration object.

    Raises:
        RuntimeError:
            If CUDA is required but PyTorch cannot access a CUDA device.
    """
    if not config.get("require_cuda", False):
        return

    cuda_available = False
    availability_error: str | None = None
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as error:
        availability_error = f"{type(error).__name__}: {error}"

    if cuda_available:
        return

    torch_version = getattr(torch, "__version__", "unknown")
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    torch_cuda_version = torch_cuda_version or "none"
    error_detail = (
        f" CUDA availability check raised {availability_error}."
        if availability_error is not None
        else ""
    )
    raise RuntimeError(
        "CUDA is required by this configuration, but no CUDA device is available "
        f"(torch {torch_version}; CUDA build {torch_cuda_version}).{error_detail} "
        "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver before "
        "starting training."
    )
