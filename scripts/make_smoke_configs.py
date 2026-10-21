#!/usr/bin/env python3
"""
Write short-run GPU smoke configs derived from the repo's stage YAMLs.

The smoke run uses config/{sft,reward,dpo,rlhf}_config.yaml with a handful of
overrides (model, data paths, a few steps, no MLflow), so the GPU job exercises
the same keys a real run reads:

    python scripts/make_smoke_configs.py --preset 1b --root /workspace/smoke/runs/abc123

writes <root>/configs/{sft,reward,dpo,rlhf,sft_resume}.yaml and
<root>/configs/training_pipeline.yaml (pass it as PIPELINE_CONFIG to
scripts/launch_training.sh). Data is expected under <root>/data in the layout
scripts/prepare_data.py writes.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"

# Steps are chosen so every stage writes one intermediate (sharded under FSDP)
# checkpoint at save_steps and a final full model.
PRESETS: Dict[str, Dict[str, Any]] = {
    # 2x A40 48GB: shakes out the FSDP mechanics cheaply.
    "1b": {
        "model": "meta-llama/Llama-3.2-1B",
        "max_length": 512,
        "steps": 20,
        "save_steps": 10,
        "sft_batch": 4,
        "reward_batch": 4,
        "dpo_batch": 2,
        "ppo_batch": 4,
        "ppo_steps": 4,
        "max_prompt_length": 256,
        "max_response_length": 64,
        "rl_prompts": 64,
    },
    # 4x 80GB: the target model.
    "8b": {
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_length": 512,
        "steps": 20,
        "save_steps": 10,
        "sft_batch": 2,
        "reward_batch": 2,
        "dpo_batch": 1,
        "ppo_batch": 2,
        "ppo_steps": 4,
        "max_prompt_length": 256,
        "max_response_length": 64,
        "rl_prompts": 64,
    },
}
# 8B PPO speed benchmark against verl (verl_ppo/run_ppo.sh reads the rlhf
# settings back from rlhf.yaml): 16 prompts per GPU (64 on 4 GPUs) x 512
# response tokens for 6 steps. verl holds out 16 prompts for validation, so
# 400 prompts still leave 6 full batches of 64.
PRESETS["8b_ppo_bench"] = {
    **PRESETS["8b"],
    "ppo_batch": 16,
    "ppo_steps": 6,
    "max_response_length": 512,
    "rl_prompts": 400,
}
# The first real training run: UltraChat / UltraFeedback (prepare_data.py
# --source hf), one epoch each of SFT and RM at the stage YAMLs' own max
# lengths (2048), then 30 PPO steps of 64 prompts on the verl backend
# (full-parameter). Micro-batches are halved with gradient accumulation 2 so a
# batch of 2048-token rows fits in 80 GB. verl filters the ~7% of prompts over
# 512 tokens and holds out 16, so 2400 prompts still cover 30 x 64.
PRESETS["8b_real"] = {
    **PRESETS["8b"],
    "max_length": None,
    "steps": None,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 10,
    "sft_batch": 2,
    "reward_batch": 1,
    "dpo_batch": 1,
    "grad_accum": 2,
    "ppo_batch": 16,
    "ppo_steps": 30,
    "max_prompt_length": 512,
    "max_response_length": 512,
    "rl_prompts": 2400,
    "data": {"source": "hf", "sft_examples": 20000, "preference_examples": 20000},
    "rlhf_backend": "verl",
    "dataset_version": "ultrachat200k-ultrafeedback-20k-seed0",
}
# Smoke presets repeat prepare_data.py's synthetic examples.
SYNTHETIC_DATA = {"source": "synthetic", "sft_examples": 400, "preference_examples": 400}


def data_args(preset_name: str) -> List[str]:
    """scripts/prepare_data.py arguments for a preset (besides --output-dir)."""
    preset = PRESETS[preset_name]
    data = {**SYNTHETIC_DATA, **preset.get("data", {})}
    return [
        "--source", data["source"],
        "--sft-examples", str(data["sft_examples"]),
        "--preference-examples", str(data["preference_examples"]),
        "--rl-prompts", str(preset["rl_prompts"]),
    ]


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load(name: str) -> Dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _trainer_overrides(preset: Mapping[str, Any], batch: int, output_dir: Path) -> Dict[str, Any]:
    """
    Overrides shared by the HF Trainer stages (SFT, reward, DPO).

    max_length None keeps the stage YAML's limit; steps None trains one epoch.
    """
    training: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": batch,
        "gradient_accumulation_steps": preset.get("grad_accum", 1),
        "logging_steps": preset.get("logging_steps", 1),
        "eval_steps": preset.get("eval_steps", preset["save_steps"]),
        "save_steps": preset["save_steps"],
        "save_total_limit": 2,
        "dataloader_num_workers": 2,
    }
    if preset["steps"] is not None:
        training["max_steps"] = preset["steps"]
    return {
        "model": {"precision": "bf16", "gradient_checkpointing": True},
        "data": {} if preset["max_length"] is None else {"max_length": preset["max_length"]},
        "training": training,
        "distributed": {"fsdp": {"cpu_ram_efficient_loading": True}},
        "tracking": {"mlflow_enabled": False, "tensorboard_enabled": False, "wandb_enabled": False},
        "safety": {"early_stopping": {"enabled": False}},
        "debug": {"enabled": False},
    }


def build_configs(preset_name: str, root: Path, model: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Return {file name: YAML mapping} for one smoke run rooted at root."""
    preset = dict(PRESETS[preset_name])
    if model:
        preset["model"] = model
    data, outputs, configs = root / "data", root / "outputs", root / "configs"

    sft = deep_update(
        _load("sft_config.yaml"),
        deep_update(
            _trainer_overrides(preset, preset["sft_batch"], outputs / "sft"),
            {
                "model": {"name": preset["model"]},
                "data": {"train_file": str(data / "sft" / "train.json"), "val_file": str(data / "sft" / "val.json")},
                "lora": {"enabled": False},
            },
        ),
    )
    sft["model"].pop("cache_dir", None)

    # Second SFT job: resume from the first one's sharded step-save_steps
    # checkpoint with the same GPU count, writing elsewhere.
    sft_resume = deep_update(
        sft,
        {
            "training": {
                "output_dir": str(outputs / "sft_resume"),
                "resume_from_checkpoint": str(outputs / "sft" / f"checkpoint-{preset['save_steps']}"),
            }
        },
    )

    preference_files = {
        "train_file": str(data / "preferences" / "train.json"),
        "val_file": str(data / "preferences" / "val.json"),
    }
    reward = deep_update(
        _load("reward_config.yaml"),
        deep_update(
            _trainer_overrides(preset, preset["reward_batch"], outputs / "reward"),
            {"data": preference_files},
        ),
    )
    dpo = deep_update(
        _load("dpo_config.yaml"),
        deep_update(
            _trainer_overrides(preset, preset["dpo_batch"], outputs / "dpo"),
            {"data": preference_files, "dpo": {"reference_batch_size": preset["dpo_batch"]}},
        ),
    )
    for stage in (reward, dpo):
        # The pipeline injects the SFT checkpoint as model.name.
        stage["model"].pop("cache_dir", None)

    rlhf = deep_update(
        _load("rlhf_config.yaml"),
        {
            "model": {"policy": {"precision": "bf16", "gradient_checkpointing": True}},
            "data": {
                "prompt_file": str(data / "prompts" / "rl_prompts.json"),
                "max_prompt_length": preset["max_prompt_length"],
                "max_response_length": preset["max_response_length"],
            },
            "algorithm": {"ppo": {"ppo_epochs": 2}},
            "training": {
                "output_dir": str(outputs / "aligned"),
                "total_steps": preset["ppo_steps"],
                "per_device_batch_size": preset["ppo_batch"],
                "save_steps": preset["ppo_steps"] // 2,
                "logging_steps": 1,
            },
            "tracking": {"mlflow_enabled": False, "tensorboard_enabled": False, "wandb_enabled": False},
        },
    )
    for role in ("policy", "reference", "reward", "value"):
        rlhf["model"][role].pop("cache_dir", None)

    pipeline = deep_update(
        _load("training_pipeline.yaml"),
        {
            "output_dir": str(outputs),
            "distributed": {"launcher": "torchrun", "nproc_per_node": "auto"},
            "stages": {
                "sft": {"config": str(configs / "sft.yaml"), "output_dir": str(outputs / "sft"), "use_lora": False},
                "reward": {"config": str(configs / "reward.yaml"), "output_dir": str(outputs / "reward")},
                "dpo": {"config": str(configs / "dpo.yaml"), "output_dir": str(outputs / "dpo")},
                "rlhf": {
                    "config": str(configs / "rlhf.yaml"),
                    "output_dir": str(outputs / "aligned"),
                    "prompts_file": str(data / "prompts" / "rl_prompts.json"),
                    # verl's checkout comes from $VERL_DIR (runpod_gpu_smoke.sh exports it).
                    "backend": preset.get("rlhf_backend", "native"),
                    "verl": {"mode": "full", "dir": None},
                },
            },
            "promotion": {
                "metrics_path": str(outputs / "eval" / "training_candidate.json"),
                "gate_config": str(CONFIG_DIR / "eval_gate.training.json"),
            },
            "artifact": {"manifest_path": str(outputs / "model_artifact_manifest.json"), "dataset_version": preset.get("dataset_version", "gpu-smoke")},
            "awq": {"handoff_path": str(outputs / "awq_handoff.json")},
            "serving": {"handoff_path": str(outputs / "serving_handoff.json")},
        },
    )

    return {
        "sft.yaml": sft,
        "sft_resume.yaml": sft_resume,
        "reward.yaml": reward,
        "dpo.yaml": dpo,
        "rlhf.yaml": rlhf,
        "training_pipeline.yaml": pipeline,
    }


def write_configs(preset_name: str, root: Path, model: str | None = None) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, mapping in build_configs(preset_name, root, model).items():
        (config_dir / name).write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return config_dir / "training_pipeline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="1b")
    parser.add_argument("--root", help="Smoke run directory (configs/, data/, outputs/ under it)")
    parser.add_argument("--model", help="Override the preset's base model")
    parser.add_argument("--data-args", action="store_true", help="Print the preset's prepare_data.py arguments and exit")
    args = parser.parse_args()
    if args.data_args:
        print(" ".join(data_args(args.preset)))
        return
    if not args.root:
        parser.error("--root is required")
    print(write_configs(args.preset, Path(args.root), args.model))


if __name__ == "__main__":
    main()
