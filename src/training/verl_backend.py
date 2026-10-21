"""
The pipeline's rlhf stage with verl (verl_ppo/) instead of rlhf_trainer.

verl runs in its own uv environment (verl_ppo/setup_env.sh), so this module
never imports it. It writes verl's prompt parquet, runs verl_ppo/run_ppo.sh as
one subprocess (Ray spreads it over the GPUs), then turns the result into what
the native backend produces: <output_dir>/checkpoint-final as a full Hugging
Face model with our tokenizer, final metrics merged into the promotion metrics
file, and an MLflow registration. training_eval, the artifact manifest and the
AWQ/serving handoffs read it unchanged.

Select it in training_pipeline.yaml:

    stages:
      rlhf:
        backend: verl          # native (default) | verl
        verl: {mode: full, dir: /workspace/verl}   # dir defaults to $VERL_DIR
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional

try:
    from .mlflow_lifecycle import MLflowTrainingLifecycle, PromotionConfig
    from .results import TrainingRunResult, merge_metrics_file
except ImportError:  # pragma: no cover - script execution fallback
    from mlflow_lifecycle import MLflowTrainingLifecycle, PromotionConfig  # type: ignore[no-redef]
    from results import TrainingRunResult, merge_metrics_file  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

VERL_PPO_DIR = Path(__file__).resolve().parent.parent.parent / "verl_ppo"
RLHF_BACKENDS = ("native", "verl")
VERL_MODES = ("lora", "full")
# Copied from the SFT checkpoint so the aligned model keeps our chat template and pad token.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
VAL_PROMPTS = 16


@dataclass(frozen=True)
class VerlSettings:
    mode: str
    verl_dir: Path
    nproc: int


def rlhf_backend(stage_config: Mapping[str, Any]) -> str:
    backend = str(stage_config.get("backend", "native"))
    if backend not in RLHF_BACKENDS:
        raise ValueError(f"stages.rlhf.backend must be one of {RLHF_BACKENDS}, got {backend!r}")
    return backend


def verl_settings(stage_config: Mapping[str, Any], nproc: int) -> VerlSettings:
    verl_config = stage_config.get("verl") or {}
    mode = str(verl_config.get("mode", "full"))
    if mode not in VERL_MODES:
        raise ValueError(f"stages.rlhf.verl.mode must be one of {VERL_MODES}, got {mode!r}")
    verl_dir = verl_config.get("dir") or os.environ.get("VERL_DIR")
    if not verl_dir:
        raise ValueError("stages.rlhf.verl.dir (or VERL_DIR) must point at verl_ppo/setup_env.sh's checkout")
    return VerlSettings(mode=mode, verl_dir=Path(verl_dir), nproc=nproc)


def bench_env(raw_config: Mapping[str, Any], nproc: int) -> Dict[str, str]:
    """run_ppo.sh sizes from the rlhf YAML, so both backends train on the same batches."""
    training, data = raw_config.get("training", {}), raw_config.get("data", {})
    return {
        "BATCH": str(int(training.get("per_device_batch_size", 8)) * nproc),
        "STEPS": str(int(training.get("total_steps", 1000))),
        "PROMPT_LEN": str(int(data.get("max_prompt_length", 512))),
        "RESPONSE_LEN": str(int(data.get("max_response_length", 512))),
    }


def run_verl_rlhf(
    *,
    sft_checkpoint: str,
    reward_checkpoint: str,
    prompts_file: str,
    output_dir: Path,
    raw_config: Dict[str, Any],
    config_path: Optional[str],
    settings: VerlSettings,
    promotion: PromotionConfig,
) -> TrainingRunResult:
    python = settings.verl_dir / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"{python} is missing; run verl_ppo/setup_env.sh with VERL_DIR={settings.verl_dir}")
    work_dir = output_dir / "verl"
    final_path = output_dir / "checkpoint-final"
    sizes = bench_env(raw_config, settings.nproc)
    tags = {"artifact_type": "full_model", "algorithm": "ppo", "rlhf_backend": "verl", "verl_mode": settings.mode}

    prepare_prompts = _load_verl_ppo_module("prepare_prompts")
    prompts = prepare_prompts.load_prompts(Path(prompts_file))
    prepare_prompts.write_parquet(prompts, work_dir / "data", val_count=min(VAL_PROMPTS, len(prompts) - 1))

    lifecycle = MLflowTrainingLifecycle.from_training_config(
        raw_config, model_type="rlhf", config_path=config_path, promotion=promotion, extra_tags=tags
    )
    with lifecycle:
        env = {
            **os.environ,
            **sizes,
            "MODE": settings.mode,
            "SFT_CKPT": sft_checkpoint,
            "RM_CKPT": reward_checkpoint,
            "DATA_DIR": str(work_dir / "data"),
            "OUTPUT_DIR": str(work_dir),
            "VERL_DIR": str(settings.verl_dir),
            "NGPUS": str(settings.nproc),
        }
        logger.info("Running verl PPO (%s) on %d GPUs: %s", settings.mode, settings.nproc, sizes)
        subprocess.run(["bash", str(VERL_PPO_DIR / "run_ppo.sh")], check=True, env=env)

        install_checkpoint(work_dir / "hf", final_path, sft_checkpoint)
        verl_report = _load_verl_ppo_module("verl_report")
        steps = verl_report.parse_steps((work_dir / "train.log").read_text(encoding="utf-8"))
        if not steps:
            raise RuntimeError(f"no step metrics in {work_dir / 'train.log'}")
        for index, step in enumerate(steps, start=1):
            lifecycle.log_metrics(step, step=index)

        final_metrics = verl_final_metrics(steps, verl_report.summarize(steps, batch=int(sizes["BATCH"])))
        final_metrics["artifact_checksum_ready"] = 1.0 if final_path.is_dir() else 0.0
        if promotion.metrics_path:
            final_metrics = merge_metrics_file(promotion.metrics_path, str(final_path), final_metrics)
        lifecycle_result = lifecycle.register_checkpoint(str(final_path), final_metrics=final_metrics, extra_tags=tags)
    return TrainingRunResult(checkpoint_path=str(final_path), metrics=final_metrics, lifecycle_result=lifecycle_result)


def verl_final_metrics(steps: List[Dict[str, float]], timings: Mapping[str, float]) -> Dict[str, Any]:
    """The native backend's final metric keys, from verl's per-step console metrics."""
    return {
        "total_steps": len(steps),
        "kl_coef": steps[-1].get("actor/reward_kl_penalty_coeff", 0.0),
        "rlhf_mean_reward": _mean(steps, "critic/score/mean"),
        "rlhf_mean_kl": _mean(steps, "actor/reward_kl_penalty"),
        "task_eval_score": 0.0,
        "safety_score": 1.0,
        "latency_p95_ms": 0.0,
        "cost_per_1k_tokens_usd": 0.0,
        **timings,
    }


def install_checkpoint(merged_dir: Path, final_path: Path, sft_checkpoint: str) -> None:
    """Move verl's merged actor to checkpoint-final and give it the SFT tokenizer."""
    if not merged_dir.is_dir():
        raise FileNotFoundError(f"verl's model merger left no {merged_dir}")
    if final_path.exists():
        shutil.rmtree(final_path)
    sft = Path(sft_checkpoint)
    if (merged_dir / "config.json").exists():
        shutil.move(str(merged_dir), str(final_path))
    else:  # A LoRA run exported only the adapter: merge it into the SFT weights.
        from .export import merge_adapter

        adapter = next((path.parent for path in sorted(merged_dir.rglob("adapter_config.json"))), None)
        if adapter is None:
            raise FileNotFoundError(f"no model weights or LoRA adapter under {merged_dir}")
        merge_adapter(sft_checkpoint, str(adapter), str(final_path), tokenizer_path=sft_checkpoint)
    for name in TOKENIZER_FILES:
        if (sft / name).exists():
            shutil.copy2(sft / name, final_path / name)
    if (sft / "generation_config.json").exists() and not (final_path / "generation_config.json").exists():
        shutil.copy2(sft / "generation_config.json", final_path / "generation_config.json")


def _mean(steps: List[Dict[str, float]], key: str) -> float:
    values = [step[key] for step in steps if key in step]
    return sum(values) / len(values) if values else 0.0


def _load_verl_ppo_module(name: str) -> ModuleType:
    """verl_ppo/ holds standalone scripts, not a package; load one by path."""
    spec = importlib.util.spec_from_file_location(f"verl_ppo_{name}", VERL_PPO_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
