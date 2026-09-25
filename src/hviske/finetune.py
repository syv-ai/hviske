"""Finetuning ASR models."""

import gc
import logging
import os
import traceback

import torch
import torch.multiprocessing as torch_mp
from omegaconf import DictConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

from .audio import download_background_noises
from .callbacks import (
    DataLoaderShutdownCallback,
    EvaluationScheduleCallback,
    StopAfterStepCallback,
)
from .data import load_data_for_finetuning
from .data_models import ModelSetup
from .dataloader_shutdown import DataLoaderShutdownController
from .experiment_tracking import ExTrackingSetup, load_extracking_setup
from .hub_access_health import HubAccessHealthScope
from .hub_retries import configure_hub_streaming_retries, retry_hub_access
from .model_publication import push_model_to_hub, validate_private_only_config
from .model_setup import load_model_setup
from .ngram import train_and_store_ngram_model
from .utils import block_terminal_output, disable_tqdm

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
    health_scope = HubAccessHealthScope(
        run_key=str(config.get("model_dir") or config.get("model_id") or os.getpid())
    )
    with worker_shutdown, health_scope:
        extracking_setup: ExTrackingSetup | None = None
        try:
            # Note whether this is the main process in a distributed setting.
            is_main_process = os.getenv("RANK", "0") == "0"
            if config.enable_experiment_tracking and is_main_process:
                extracking_setup = load_extracking_setup(config=config)

            if extracking_setup is not None:
                extracking_setup.run_initialization()

            configure_hub_streaming_retries(
                retry_config=config.get("hub_streaming_retries")
            )
            download_background_noises()
            model_setup: ModelSetup = load_model_setup(config=config)
            processor = retry_hub_access(
                operation=model_setup.load_processor,
                url=str(config.model.get("pretrained_model_id") or "model processor"),
            )
            dataset = load_data_for_finetuning(config=config, processor=processor)
            processor.save_pretrained(save_directory=config.model_dir)
            model = retry_hub_access(
                operation=model_setup.load_model,
                url=str(config.model.get("pretrained_model_id") or "model"),
            )

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
        except BaseException as error:
            if extracking_setup is not None:
                _report_tracking_failure(extracking_setup, error)
                _finalize_tracking_after_failure(extracking_setup)
            raise
        else:
            if extracking_setup is not None:
                extracking_setup.run_finalization(exit_code=0)


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


def _report_tracking_failure(setup: ExTrackingSetup, error: BaseException) -> None:
    """Best-effort failure alerting that cannot replace the training exception."""
    report_failure = getattr(setup, "report_failure", None)
    if not callable(report_failure) or isinstance(error, KeyboardInterrupt):
        return
    try:
        report_failure(error)
    except BaseException:
        logger.exception("Experiment tracking failure alert could not be delivered")


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
