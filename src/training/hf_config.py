"""
Map the training YAML configs onto Hugging Face model and Trainer arguments.

The SFT and reward trainers both go through these helpers so that the values in
config/*.yaml are the values that actually run, instead of being shadowed by
hard-coded TrainingArguments.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

_PRECISION_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
}

_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# YAML `training:` keys that map 1:1 onto TrainingArguments fields.
_PASSTHROUGH_TRAINING_KEYS = (
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "warmup_steps",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "save_strategy",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "deepspeed",
    "seed",
    "data_seed",
)

# Mixed-precision flags are derived from model.precision; keeping a second copy
# under training: is how fp16 silently stayed on after switching to bf16.
_REJECTED_TRAINING_KEYS = ("fp16", "bf16", "fp16_opt_level")


def resolve_precision(model_config: Optional[Mapping[str, Any]]) -> str:
    """Return the canonical precision name ("bf16", "fp16" or "fp32")."""
    value = str((model_config or {}).get("precision", "bf16")).strip().lower()
    if value not in _PRECISION_ALIASES:
        raise ValueError(
            f"Unsupported model.precision {value!r}; expected one of bf16, fp16, fp32"
        )
    return _PRECISION_ALIASES[value]


def model_load_dtype(precision: str, full_fine_tune: bool) -> torch.dtype:
    """
    Return the dtype to load weights in.

    fp16 mixed precision needs fp32 master weights for full fine-tuning: the
    GradScaler refuses to unscale fp16 gradients. bf16 has no loss scaler, so
    the weights can stay in bf16.
    """
    if precision == "fp16" and full_fine_tune:
        return torch.float32
    return _PRECISION_DTYPES[precision]


def attention_implementation(model_config: Optional[Mapping[str, Any]]) -> str:
    """Pick the attention kernel requested by the model config."""
    model_config = model_config or {}
    explicit = model_config.get("attn_implementation")
    if explicit:
        return str(explicit)
    if model_config.get("use_flash_attention", False):
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        logger.warning(
            "model.use_flash_attention is true but flash-attn is not installed; "
            "falling back to sdpa attention"
        )
    return "sdpa"


def report_targets(tracking_config: Optional[Mapping[str, Any]]) -> List[str]:
    """Trainer report_to integrations. MLflow is handled by MLflowMetricsCallback."""
    tracking_config = tracking_config or {}
    targets = []
    if tracking_config.get("tensorboard_enabled", False):
        targets.append("tensorboard")
    if tracking_config.get("wandb_enabled", False):
        targets.append("wandb")
    return targets or ["none"]


def early_stopping_settings(
    raw_config: Mapping[str, Any],
    has_eval: bool,
) -> Optional[Dict[str, Any]]:
    """Return safety.early_stopping settings when they can be applied."""
    early_stopping = dict(((raw_config.get("safety") or {}).get("early_stopping")) or {})
    if not has_eval or not early_stopping.get("enabled", False):
        return None
    return {
        "patience": int(early_stopping.get("patience", 3)),
        "metric": str(early_stopping.get("metric", "eval_loss")),
        "greater_is_better": str(early_stopping.get("mode", "min")).lower() == "max",
    }


def build_training_arguments_kwargs(
    raw_config: Mapping[str, Any],
    output_dir: str,
    has_eval: bool,
) -> Dict[str, Any]:
    """Translate a training YAML mapping into TrainingArguments kwargs."""
    training = dict(raw_config.get("training") or {})
    model_config = dict(raw_config.get("model") or {})

    rejected = [key for key in _REJECTED_TRAINING_KEYS if key in training]
    if rejected:
        raise ValueError(
            "Set mixed precision with model.precision (bf16/fp16/fp32) instead of "
            f"training.{', training.'.join(rejected)}"
        )

    kwargs: Dict[str, Any] = {
        key: training[key]
        for key in _PASSTHROUGH_TRAINING_KEYS
        if training.get(key) is not None
    }
    if training.get("optimizer"):
        kwargs["optim"] = training["optimizer"]

    precision = resolve_precision(model_config)
    kwargs["bf16"] = precision == "bf16"
    kwargs["fp16"] = precision == "fp16"

    # transformers 4.46+ only accepts eval_strategy; accept the old YAML name.
    eval_strategy = training.get("eval_strategy", training.get("evaluation_strategy", "steps"))
    kwargs["eval_strategy"] = eval_strategy if has_eval else "no"
    if not has_eval:
        kwargs.pop("eval_steps", None)

    if model_config.get("gradient_checkpointing", False):
        kwargs["gradient_checkpointing"] = True
        # Non-reentrant checkpointing also works for LoRA, where the frozen
        # embedding output does not require grad.
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    early_stopping = early_stopping_settings(raw_config, has_eval)
    if early_stopping is not None:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = early_stopping["metric"]
        kwargs["greater_is_better"] = early_stopping["greater_is_better"]
        kwargs["save_strategy"] = kwargs["eval_strategy"]

    tracking = dict(raw_config.get("tracking") or {})
    kwargs["report_to"] = report_targets(tracking)
    kwargs["logging_dir"] = tracking.get("tensorboard_dir") or f"{output_dir}/logs"
    kwargs["output_dir"] = output_dir
    kwargs["remove_unused_columns"] = False
    return kwargs
