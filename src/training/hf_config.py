"""
Map the training YAML configs onto Hugging Face model and Trainer arguments.

The SFT and reward trainers both go through these helpers so that the values in
config/*.yaml are the values that actually run, instead of being shadowed by
hard-coded TrainingArguments.
"""

from __future__ import annotations

import importlib.util
import logging
import os
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
    "use_cpu",
    "seed",
    "data_seed",
)

# Mixed-precision flags are derived from model.precision; keeping a second copy
# under training: is how fp16 silently stayed on after switching to bf16.
_REJECTED_TRAINING_KEYS = ("fp16", "bf16", "fp16_opt_level")

# Llama attention + MLP projections; adapting only q/k/v/o leaves most of the
# trainable capacity on the table.
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_DISTRIBUTED_STRATEGIES = ("none", "ddp", "fsdp")
_FSDP_SHARDING = ("full_shard", "shard_grad_op", "hybrid_shard")
_FSDP_CHECKPOINT_STATE_DICT = {
    "sharded": "SHARDED_STATE_DICT",
    "full": "FULL_STATE_DICT",
}


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


def precision_dtype(precision: str) -> torch.dtype:
    """Dtype a finished checkpoint is saved in for this precision."""
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


def apply_lora(model: Any, lora_settings: Optional[Mapping[str, Any]], default_dropout: float = 0.05) -> Any:
    """Wrap a causal LM with a LoRA adapter configured from a `lora:` YAML section."""
    # peft is only needed for adapter runs, not full fine-tuning.
    from peft import LoraConfig, TaskType, get_peft_model

    lora_settings = lora_settings or {}
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_settings.get("r", 16)),
        lora_alpha=int(lora_settings.get("alpha", 32)),
        lora_dropout=float(lora_settings.get("dropout", default_dropout)),
        target_modules=list(lora_settings.get("target_modules") or DEFAULT_LORA_TARGET_MODULES),
        bias=lora_settings.get("bias", "none"),
        modules_to_save=lora_settings.get("modules_to_save"),
    )
    logger.info("Applying LoRA configuration: %s", lora_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


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


def distributed_settings(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the `distributed:` section with defaults filled in."""
    distributed = dict(raw_config.get("distributed") or {})
    distributed.setdefault("strategy", "none")
    distributed.setdefault("timeout_minutes", 60)
    distributed.setdefault("find_unused_parameters", False)
    distributed["fsdp"] = dict(distributed.get("fsdp") or {})
    return distributed


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def resolve_distributed_strategy(raw_config: Mapping[str, Any], world_size: int) -> str:
    """
    Return the strategy that will actually run.

    A single process falls back to "none": Trainer refuses FSDP without a
    process group, and CPU tests and single-GPU debug runs share these configs.
    """
    strategy = str(distributed_settings(raw_config)["strategy"]).strip().lower()
    if strategy not in _DISTRIBUTED_STRATEGIES:
        raise ValueError(
            f"Unsupported distributed.strategy {strategy!r}; expected one of "
            f"{', '.join(_DISTRIBUTED_STRATEGIES)}"
        )
    if strategy != "none" and world_size <= 1:
        logger.warning(
            "distributed.strategy=%s needs torchrun with more than one process; "
            "running single-process instead",
            strategy,
        )
        return "none"
    return strategy


def build_distributed_kwargs(
    raw_config: Mapping[str, Any],
    world_size: int,
    gradient_checkpointing: bool,
) -> Dict[str, Any]:
    """TrainingArguments kwargs for DDP or FSDP (empty for a single process)."""
    if (raw_config.get("training") or {}).get("deepspeed"):
        raise ValueError(
            "training.deepspeed is not supported; use distributed.strategy fsdp or ddp"
        )
    settings = distributed_settings(raw_config)
    strategy = resolve_distributed_strategy(raw_config, world_size)
    if strategy == "none":
        return {}

    kwargs: Dict[str, Any] = {"ddp_timeout": int(float(settings["timeout_minutes"]) * 60)}
    if strategy == "ddp":
        kwargs["ddp_find_unused_parameters"] = bool(settings["find_unused_parameters"])
        return kwargs

    fsdp = settings["fsdp"]
    version = int(fsdp.get("version", 1))
    if version != 1:
        raise ValueError(
            "distributed.fsdp.version must be 1: transformers 4.57 TrainingArguments "
            "cannot select FSDP2 or its state dict type"
        )
    sharding = str(fsdp.get("sharding", "full_shard"))
    if sharding not in _FSDP_SHARDING:
        raise ValueError(
            f"Unsupported distributed.fsdp.sharding {sharding!r}; expected one of "
            f"{', '.join(_FSDP_SHARDING)}"
        )
    sync_module_states = bool(fsdp.get("sync_module_states", True))
    cpu_ram_efficient_loading = bool(fsdp.get("cpu_ram_efficient_loading", False))
    if cpu_ram_efficient_loading and not sync_module_states:
        raise ValueError(
            "distributed.fsdp.cpu_ram_efficient_loading requires sync_module_states"
        )
    activation_checkpointing = fsdp.get("activation_checkpointing")
    if activation_checkpointing is None:
        activation_checkpointing = gradient_checkpointing

    # Without "auto_wrap" accelerate falls back to NO_WRAP and the whole model
    # becomes one FSDP unit, so every forward gathers all 8B parameters.
    options = [sharding, "auto_wrap"]
    if fsdp.get("offload", False):
        options.append("offload")
    layer_classes = fsdp.get("transformer_layer_cls_to_wrap") or ["LlamaDecoderLayer"]
    if isinstance(layer_classes, str):
        layer_classes = [layer_classes]
    kwargs["fsdp"] = " ".join(options)
    kwargs["fsdp_config"] = {
        "transformer_layer_cls_to_wrap": list(layer_classes),
        "backward_prefetch": str(fsdp.get("backward_prefetch", "backward_pre")),
        "forward_prefetch": bool(fsdp.get("forward_prefetch", False)),
        "limit_all_gathers": bool(fsdp.get("limit_all_gathers", True)),
        "use_orig_params": bool(fsdp.get("use_orig_params", True)),
        "sync_module_states": sync_module_states,
        "cpu_ram_efficient_loading": cpu_ram_efficient_loading,
        "activation_checkpointing": bool(activation_checkpointing),
    }
    return kwargs


def keep_fsdp_buffers_in_fp32(trainer: Any) -> None:
    """
    Keep module buffers (Llama's RoPE inv_freq) in fp32 under FSDP mixed precision.

    accelerate's FSDP1 policy casts buffers to the compute dtype unless
    buffer_autocast is set, so training would rotate positions with
    bf16-rounded frequencies while a plain from_pretrained (inference, the DPO
    reference pass) keeps them in fp32. Call after building the Trainer and
    before train()/evaluate(), which wrap the model with the plugin's policy.
    """
    if not getattr(trainer, "is_fsdp_enabled", False):
        return
    precision = str(trainer.accelerator.state.mixed_precision)
    if precision not in ("bf16", "fp16"):
        return
    trainer.accelerator.state.fsdp_plugin.set_mixed_precision(
        precision, buffer_autocast=True, override=True
    )


def fsdp_environment(raw_config: Mapping[str, Any], world_size: int) -> Dict[str, str]:
    """
    Environment accelerate reads for FSDP settings TrainingArguments cannot pass.

    Intermediate checkpoints default to sharded so rank 0 does not gather a full
    fp32 model plus optimizer state every save_steps. Apply before constructing
    TrainingArguments.
    """
    if resolve_distributed_strategy(raw_config, world_size) != "fsdp":
        return {}
    name = str(distributed_settings(raw_config)["fsdp"].get("checkpoint_state_dict_type", "sharded"))
    if name not in _FSDP_CHECKPOINT_STATE_DICT:
        raise ValueError(
            f"Unsupported distributed.fsdp.checkpoint_state_dict_type {name!r}; "
            "expected sharded or full"
        )
    return {"FSDP_STATE_DICT_TYPE": _FSDP_CHECKPOINT_STATE_DICT[name]}


def build_training_arguments_kwargs(
    raw_config: Mapping[str, Any],
    output_dir: str,
    has_eval: bool,
    world_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Translate a training YAML mapping into TrainingArguments kwargs."""
    training = dict(raw_config.get("training") or {})
    model_config = dict(raw_config.get("model") or {})
    if world_size is None:
        world_size = env_world_size()

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

    gradient_checkpointing = bool(model_config.get("gradient_checkpointing", False))
    distributed_kwargs = build_distributed_kwargs(raw_config, world_size, gradient_checkpointing)
    kwargs.update(distributed_kwargs)
    fsdp_activation_checkpointing = (distributed_kwargs.get("fsdp_config") or {}).get(
        "activation_checkpointing", False
    )
    # Trainer raises if both are on; FSDP checkpoints per wrapped decoder layer.
    if gradient_checkpointing and not fsdp_activation_checkpointing:
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
