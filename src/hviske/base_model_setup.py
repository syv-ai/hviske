"""Shared mechanics for model setup implementations."""

import logging
import os
import sys
import typing as t
from functools import partial

import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer_utils import EvalPrediction, SchedulerType
from transformers.training_args import OptimizerNames

from .compute_metrics import compute_error_rate_metrics
from .data_models import ModelSetup, Processor

logger = logging.getLogger(__package__)


class TrainingArgumentsKwargs(t.TypedDict, total=False):
    """Common, constructor-compatible training argument values."""

    output_dir: str
    hub_model_id: str
    hub_private_repo: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: SchedulerType
    warmup_steps: int
    max_steps: int
    fp16: bool
    bf16: bool
    push_to_hub: bool
    eval_strategy: str
    eval_steps: int
    save_steps: int
    save_strategy: str
    logging_steps: int
    length_column_name: str
    max_grad_norm: float
    gradient_checkpointing: bool
    gradient_checkpointing_kwargs: dict[str, bool]
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    seed: int
    remove_unused_columns: bool
    optim: OptimizerNames
    adam_beta1: float
    adam_beta2: float
    report_to: list[str]
    ignore_data_skip: bool
    use_cpu: bool
    dataloader_num_workers: int
    dataloader_drop_last: bool
    ddp_find_unused_parameters: bool


class Seq2SeqTrainingArgumentsKwargs(TrainingArgumentsKwargs, total=False):
    """Additional keyword values accepted by seq2seq training arguments."""

    predict_with_generate: bool
    generation_max_length: int
    accelerator_config: dict[str, object]


class BaseModelSetup(ModelSetup):
    """Shared configuration, metrics, and training-argument mechanics.

    Model loading, processing, collation, trainer choice, and model-specific revision
    or vocabulary handling deliberately remain in concrete setup classes.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialise generic setup state.

        Args:
            config:
                The Hydra configuration object.
        """
        self.config = config
        self.processor: Processor
        self.is_main_process = os.getenv("RANK", "0") == "0"

    @t.overload
    def _training_arguments_kwargs(
        self,
        *,
        sequence_to_sequence: t.Literal[False],
        include_max_grad_norm: bool = False,
    ) -> TrainingArgumentsKwargs: ...

    @t.overload
    def _training_arguments_kwargs(
        self,
        *,
        sequence_to_sequence: t.Literal[True],
        include_max_grad_norm: bool = False,
    ) -> Seq2SeqTrainingArgumentsKwargs: ...

    def _training_arguments_kwargs(
        self, *, sequence_to_sequence: bool, include_max_grad_norm: bool = False
    ) -> TrainingArgumentsKwargs | Seq2SeqTrainingArgumentsKwargs:
        """Calculate values shared by Transformers training argument classes.

        Args:
            sequence_to_sequence:
                Whether to include generation-specific sequence-to-sequence values.
            include_max_grad_norm:
                Whether the caller's historical argument set includes this value.

        Returns:
            Keyword arguments accepted by ``TrainingArguments`` and its seq2seq
            subclass. Existing configuration values and their primitive types are kept
            unchanged.
        """
        num_devices = max(torch.cuda.device_count(), 1)
        per_device_total_batch_size = self.config.total_batch_size // num_devices
        gradient_accumulation_steps = (
            per_device_total_batch_size // self.config.per_device_batch_size
        )
        if gradient_accumulation_steps == 0:
            if self.is_main_process:
                logger.warning(
                    "Your `total_batch_size` is too small; using one accumulation step."
                )
            gradient_accumulation_steps = 1

        fp16 = False
        bf16 = False
        if not mps_is_available():
            if self.config.bf16_allowed and torch.cuda.is_bf16_supported():
                bf16 = True
            elif self.config.fp16_allowed and torch.cuda.is_available():
                fp16 = True

        if self.config.early_stopping:
            self.config.save_total_limit = max(self.config.save_total_limit, 1)

        metric_name = (
            (
                f"val_{self.config.evaluation_datasets[0].id.split('/')[-1]}_"
                f"{self.config.evaluation_datasets[0].subset}_cer"
            )
            .lower()
            .replace("-", "_")
        )
        kwargs: TrainingArgumentsKwargs | Seq2SeqTrainingArgumentsKwargs = {
            "output_dir": self.config.model_dir,
            "hub_model_id": f"{self.config.hub_organisation}/{self.config.model_id}",
            "hub_private_repo": self.config.private,
            "per_device_train_batch_size": self.config.per_device_batch_size,
            "per_device_eval_batch_size": self.config.per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": self.config.model.learning_rate,
            "lr_scheduler_type": SchedulerType.COSINE,
            "warmup_steps": self.config.warmup_steps,
            "max_steps": self.config.max_steps,
            "fp16": fp16,
            "bf16": bf16,
            "push_to_hub": False,
            "eval_strategy": "steps",
            "eval_steps": (
                1 if self.config.get("evaluation_steps") else self.config.eval_steps
            ),
            "save_steps": self.config.save_steps,
            "save_strategy": "no" if self.config.save_total_limit == 0 else "steps",
            "logging_steps": self.config.logging_steps,
            "length_column_name": "input_length",
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "save_total_limit": self.config.save_total_limit,
            "load_best_model_at_end": self.config.early_stopping,
            "metric_for_best_model": metric_name,
            "greater_is_better": False,
            "seed": self.config.seed,
            "remove_unused_columns": False,
            "optim": OptimizerNames.ADAMW_TORCH,
            "adam_beta1": self.config.adam_first_momentum,
            "adam_beta2": self.config.adam_second_momentum,
            "report_to": [self.config.experiment_tracking.type]
            if self.config.enable_experiment_tracking
            else [],
            "ignore_data_skip": self.config.ignore_data_skip,
            "use_cpu": hasattr(sys, "_called_from_test"),
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "dataloader_drop_last": True,
            "ddp_find_unused_parameters": False,
        }
        if include_max_grad_norm:
            kwargs["max_grad_norm"] = self.config.max_grad_norm
        if sequence_to_sequence:
            kwargs.update(
                t.cast(
                    Seq2SeqTrainingArgumentsKwargs,
                    {
                        "predict_with_generate": True,
                        "generation_max_length": self.config.model.max_length,
                        "accelerator_config": AcceleratorConfig(
                            dispatch_batches=False
                        ).to_dict(),
                    },
                )
            )
        return kwargs

    def load_compute_metrics(self) -> t.Callable[[EvalPrediction], dict]:
        """Return the generic error-rate metric bound to this setup's processor."""
        return t.cast(
            t.Callable[[EvalPrediction], dict],
            partial(compute_error_rate_metrics, processor=self.processor),
        )
