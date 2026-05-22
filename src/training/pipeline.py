"""
Aggregated training pipeline for SFT -> reward -> RLHF handoff.

The module intentionally keeps heavyweight trainer imports lazy so dry-runs and
host smoke checks can validate pipeline wiring without installing the full CUDA
training stack.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

try:
    from .results import TrainingRunResult
except ImportError:  # pragma: no cover - supports direct script execution
    from results import TrainingRunResult  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

PIPELINE_STAGE_ORDER = (
    "sft",
    "reward",
    "rlhf",
    "training_eval",
    "mlflow_promotion",
    "serving_handoff",
)

STAGE_ALIASES = {
    "all": PIPELINE_STAGE_ORDER,
    "eval": ("training_eval",),
    "promotion": ("mlflow_promotion",),
    "handoff": ("serving_handoff",),
}


@dataclass(frozen=True)
class PipelineArtifacts:
    """Resolved artifact paths used by the pipeline."""

    output_dir: Path
    sft_output_dir: Path
    reward_output_dir: Path
    rlhf_output_dir: Path
    metrics_path: Path
    handoff_path: Path
    expected_awq_path: str
    gate_config_path: Path


class TrainingPipeline:
    """Run and validate the end-to-end training artifact flow."""

    def __init__(
        self,
        config_path: str,
        stages: str = "all",
        output_dir: Optional[str] = None,
        baseline_metrics: Optional[str] = None,
        skip_promotion: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.stages = resolve_stages(stages)
        self.output_override = Path(output_dir) if output_dir else None
        self.baseline_metrics = baseline_metrics
        self.skip_promotion = skip_promotion
        self.dry_run = dry_run
        self.artifacts = self._resolve_artifacts()

    def build_execution_plan(self) -> Dict[str, Any]:
        """Return the deterministic stage plan used by dry-run and tests."""
        return {
            "config_path": str(self.config_path),
            "dry_run": self.dry_run,
            "stages": list(self.stages),
            "stage_order": list(PIPELINE_STAGE_ORDER),
            "configs": {
                "sft": str(self._stage_config_path("sft")),
                "reward": str(self._stage_config_path("reward")),
                "rlhf": str(self._stage_config_path("rlhf")),
            },
            "path_checks": {
                "sft_config_exists": self._stage_config_path("sft").exists(),
                "reward_config_exists": self._stage_config_path("reward").exists(),
                "rlhf_config_exists": self._stage_config_path("rlhf").exists(),
                "gate_config_exists": self.artifacts.gate_config_path.exists(),
                "output_parent_exists": self.artifacts.output_dir.parent.exists(),
            },
            "outputs": {
                "root": str(self.artifacts.output_dir),
                "sft": str(self.artifacts.sft_output_dir),
                "reward": str(self.artifacts.reward_output_dir),
                "rlhf": str(self.artifacts.rlhf_output_dir),
                "metrics": str(self.artifacts.metrics_path),
                "serving_handoff": str(self.artifacts.handoff_path),
            },
            "injections": {
                "reward.model.name": "${sft.checkpoint_path}",
                "rlhf.model.policy.name": "${sft.checkpoint_path}",
                "rlhf.model.reference.name": "${sft.checkpoint_path}",
                "rlhf.model.value.name": "${sft.checkpoint_path}",
                "rlhf.model.reward.name": "${reward.checkpoint_path}",
                "rlhf.promotion_metrics_path": str(self.artifacts.metrics_path),
                "rlhf.gate_config_path": str(self.artifacts.gate_config_path),
                "rlhf.baseline_metrics_path": self.baseline_metrics,
                "rlhf.skip_promotion": self.skip_promotion,
            },
            "serving_handoff": {
                "expected_awq_path": self.artifacts.expected_awq_path,
                "quantization": self._serving_config().get("quantization", "awq"),
                "dtype": self._serving_config().get("dtype", "float16"),
            },
        }

    def run(self) -> Dict[str, Any]:
        """Execute configured stages or return a dry-run plan."""
        plan = self.build_execution_plan()
        if self.dry_run:
            return {"status": "dry_run", "plan": plan}

        results: Dict[str, Any] = {"plan": plan, "stages": {}}
        stage_results: Dict[str, TrainingRunResult] = {}

        if "sft" in self.stages:
            sft_result = self._run_sft()
            stage_results["sft"] = sft_result
            results["stages"]["sft"] = result_to_dict(sft_result)

        if "reward" in self.stages:
            sft_checkpoint = require_checkpoint(stage_results, "sft", self.artifacts.sft_output_dir / "final")
            reward_result = self._run_reward(sft_checkpoint)
            stage_results["reward"] = reward_result
            results["stages"]["reward"] = result_to_dict(reward_result)

        if "rlhf" in self.stages:
            sft_checkpoint = require_checkpoint(stage_results, "sft", self.artifacts.sft_output_dir / "final")
            reward_checkpoint = require_checkpoint(stage_results, "reward", self.artifacts.reward_output_dir / "final")
            self._write_training_metrics_seed(
                sft_metrics=stage_results.get("sft").metrics if "sft" in stage_results else {},
                reward_metrics=stage_results.get("reward").metrics if "reward" in stage_results else {},
                aligned_checkpoint=None,
            )
            rlhf_result = self._run_rlhf(sft_checkpoint, reward_checkpoint)
            stage_results["rlhf"] = rlhf_result
            results["stages"]["rlhf"] = result_to_dict(rlhf_result)

        if "training_eval" in self.stages:
            aligned_checkpoint = require_checkpoint(
                stage_results,
                "rlhf",
                self.artifacts.rlhf_output_dir / "checkpoint-final",
            )
            metrics = self._write_training_metrics_seed(
                sft_metrics=stage_results.get("sft").metrics if "sft" in stage_results else {},
                reward_metrics=stage_results.get("reward").metrics if "reward" in stage_results else {},
                rlhf_metrics=stage_results.get("rlhf").metrics if "rlhf" in stage_results else {},
                aligned_checkpoint=aligned_checkpoint,
            )
            results["stages"]["training_eval"] = {
                "metrics_path": str(self.artifacts.metrics_path),
                "metrics": metrics,
            }

        if "mlflow_promotion" in self.stages:
            results["stages"]["mlflow_promotion"] = {
                "status": "handled_by_rlhf_lifecycle",
                "metrics_path": str(self.artifacts.metrics_path),
                "gate_config_path": str(self.artifacts.gate_config_path),
                "skip_promotion": self.skip_promotion,
            }

        if "serving_handoff" in self.stages:
            aligned_checkpoint = require_checkpoint(
                stage_results,
                "rlhf",
                self.artifacts.rlhf_output_dir / "checkpoint-final",
            )
            handoff = self.write_serving_handoff(aligned_checkpoint)
            results["stages"]["serving_handoff"] = handoff

        results["status"] = "completed"
        return results

    def write_serving_handoff(self, aligned_checkpoint: str) -> Dict[str, Any]:
        """Write the serving handoff manifest for a final aligned checkpoint."""
        checkpoint = Path(aligned_checkpoint)
        validation = validate_checkpoint_artifact(checkpoint)
        serving_config = self._serving_config()
        handoff = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "final_aligned_checkpoint": str(checkpoint),
            "expected_awq_target_path": self.artifacts.expected_awq_path,
            "quantization": serving_config.get("quantization", "awq"),
            "dtype": serving_config.get("dtype", "float16"),
            "artifact_validation": validation,
            "next_steps": [
                "Convert final_aligned_checkpoint to vLLM-supported AWQ weights outside this pipeline.",
                "Place converted weights at expected_awq_target_path or update serving config to the actual AWQ path.",
                "Keep QUANTIZATION=awq and DTYPE=float16 aligned across serving YAML, Docker Compose, and Kubernetes manifests.",
                "Verify the vLLM server directly before routing traffic through the API gateway.",
            ],
            "serving_config_hints": {
                "config_file": serving_config.get("config_file", "config/serving_config.yaml"),
                "docker_compose_service": serving_config.get("docker_compose_service", "vllm-server"),
                "kubernetes_manifest": serving_config.get(
                    "kubernetes_manifest",
                    "k8s/deployments/vllm-deployment.yaml",
                ),
            },
        }
        write_json(self.artifacts.handoff_path, handoff)
        return handoff

    def _run_sft(self) -> TrainingRunResult:
        from .sft_trainer import SFTConfig, SFTTrainer

        config = SFTConfig.from_yaml(str(self._stage_config_path("sft")))
        config.output_dir = str(self.artifacts.sft_output_dir)
        config.raw_config = update_nested(
            config.raw_config,
            {"training": {"output_dir": str(self.artifacts.sft_output_dir)}},
        )
        config.skip_promotion = True
        use_lora = bool(self._stage_config("sft").get("use_lora", False))
        return SFTTrainer(config, use_lora=use_lora).train()

    def _run_reward(self, sft_checkpoint: str) -> TrainingRunResult:
        from .reward_trainer import RewardModelConfig, RewardModelTrainer

        config = RewardModelConfig.from_yaml(str(self._stage_config_path("reward")))
        config.model_name = sft_checkpoint
        config.output_dir = str(self.artifacts.reward_output_dir)
        config.raw_config = update_nested(
            config.raw_config,
            {
                "model": {"name": sft_checkpoint},
                "training": {"output_dir": str(self.artifacts.reward_output_dir)},
            },
        )
        config.skip_promotion = True
        return RewardModelTrainer(config).train()

    def _run_rlhf(self, sft_checkpoint: str, reward_checkpoint: str) -> TrainingRunResult:
        from .rlhf_trainer import RLHFTrainer

        config = build_rlhf_config(str(self._stage_config_path("rlhf")))
        config.policy_model_path = sft_checkpoint
        config.reward_model_path = reward_checkpoint
        config.output_dir = str(self.artifacts.rlhf_output_dir)
        config.raw_config = update_nested(
            config.raw_config,
            {
                "model": {
                    "policy": {"name": sft_checkpoint},
                    "reference": {"name": sft_checkpoint},
                    "value": {"name": sft_checkpoint},
                    "reward": {"name": reward_checkpoint},
                },
                "training": {"output_dir": str(self.artifacts.rlhf_output_dir)},
            },
        )
        config.promotion_metrics_path = str(self.artifacts.metrics_path)
        config.baseline_metrics_path = self.baseline_metrics
        config.gate_config_path = str(self.artifacts.gate_config_path)
        config.skip_promotion = self.skip_promotion

        prompts_file = (
            self._stage_config("rlhf").get("prompts_file")
            or config.raw_config.get("data", {}).get("prompt_file")
        )
        if not prompts_file:
            raise ValueError("RLHF prompts_file is required")
        return RLHFTrainer(config).train(str(prompts_file))

    def _write_training_metrics_seed(
        self,
        sft_metrics: Mapping[str, Any],
        reward_metrics: Mapping[str, Any],
        rlhf_metrics: Optional[Mapping[str, Any]] = None,
        aligned_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics = load_json_if_exists(self.artifacts.metrics_path)
        metrics.update(
            {
                "sft_eval_loss": float_metric(
                    sft_metrics,
                    ("eval_loss", "train_loss", "sft_eval_loss"),
                    default=0.0,
                ),
                "reward_accuracy": float_metric(
                    reward_metrics,
                    ("reward_accuracy", "eval_reward_accuracy"),
                    default=0.0,
                ),
                "reward_margin": float_metric(
                    reward_metrics,
                    ("reward_margin", "eval_reward_margin"),
                    default=0.0,
                ),
            }
        )
        if rlhf_metrics:
            metrics.update(
                {
                    "rlhf_mean_reward": float_metric(rlhf_metrics, ("rlhf_mean_reward",), default=0.0),
                    "rlhf_mean_kl": float_metric(rlhf_metrics, ("rlhf_mean_kl",), default=0.0),
                }
            )
        metrics["artifact_ready"] = (
            1.0 if aligned_checkpoint and Path(aligned_checkpoint).is_dir() else 0.0
        )
        write_json(self.artifacts.metrics_path, metrics)
        return metrics

    def _resolve_artifacts(self) -> PipelineArtifacts:
        output_dir = self.output_override or Path(self.config.get("output_dir", "outputs/training_pipeline"))
        stages = self.config.get("stages", {})
        promotion = self.config.get("promotion", {})
        serving = self._serving_config()

        if self.output_override:
            sft_output = output_dir / "models" / "sft"
            reward_output = output_dir / "models" / "reward"
            rlhf_output = output_dir / "models" / "aligned"
            metrics_path = output_dir / "eval" / "training_candidate.json"
            handoff_path = output_dir / "serving_handoff.json"
        else:
            sft_output = Path(stages.get("sft", {}).get("output_dir", output_dir / "models" / "sft"))
            reward_output = Path(stages.get("reward", {}).get("output_dir", output_dir / "models" / "reward"))
            rlhf_output = Path(stages.get("rlhf", {}).get("output_dir", output_dir / "models" / "aligned"))
            metrics_path = Path(promotion.get("metrics_path", output_dir / "eval" / "training_candidate.json"))
            handoff_path = Path(serving.get("handoff_path", output_dir / "serving_handoff.json"))
        gate_config_path = Path(promotion.get("gate_config", "config/eval_gate.training.json"))

        return PipelineArtifacts(
            output_dir=output_dir,
            sft_output_dir=sft_output,
            reward_output_dir=reward_output,
            rlhf_output_dir=rlhf_output,
            metrics_path=metrics_path,
            handoff_path=handoff_path,
            expected_awq_path=str(serving.get("expected_awq_path", "/workspace/models/awq/checkpoint-final")),
            gate_config_path=gate_config_path,
        )

    def _stage_config(self, stage: str) -> Dict[str, Any]:
        return dict((self.config.get("stages") or {}).get(stage) or {})

    def _stage_config_path(self, stage: str) -> Path:
        stage_config = self._stage_config(stage)
        if "config" not in stage_config:
            raise ValueError(f"Pipeline stage '{stage}' is missing a config path")
        return Path(stage_config["config"])

    def _serving_config(self) -> Dict[str, Any]:
        return dict(self.config.get("serving") or {})

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Pipeline config must be a YAML mapping: {path}")
        return data


def resolve_stages(stages: str | Sequence[str]) -> List[str]:
    """Resolve CLI stage names into canonical fixed-order stage names."""
    if isinstance(stages, str):
        requested = [part.strip() for part in stages.split(",") if part.strip()]
    else:
        requested = list(stages)
    if not requested:
        requested = ["all"]

    expanded: List[str] = []
    for stage in requested:
        expanded.extend(STAGE_ALIASES.get(stage, (stage,)))

    invalid = [stage for stage in expanded if stage not in PIPELINE_STAGE_ORDER]
    if invalid:
        raise ValueError(f"Unknown pipeline stages: {', '.join(invalid)}")

    return [stage for stage in PIPELINE_STAGE_ORDER if stage in set(expanded)]


def build_rlhf_config(config_path: str):
    """Load RLHFConfig from YAML without forcing that logic into the CLI only."""
    from .rlhf_trainer import RLHFConfig

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    ppo_config = config_dict.get("algorithm", {}).get("ppo", {})
    training_config = config_dict.get("training", {})
    generation_config = config_dict.get("generation", {})
    data_config = config_dict.get("data", {})

    return RLHFConfig(
        policy_model_path=config_dict["model"]["policy"]["name"],
        reward_model_path=config_dict["model"]["reward"]["name"],
        output_dir=training_config.get("output_dir", "./outputs/rlhf"),
        num_epochs=ppo_config.get("num_epochs", 4),
        num_steps=training_config.get("total_steps", 10000),
        batch_size=training_config.get("per_device_batch_size", 4),
        learning_rate=training_config.get("policy_learning_rate", 1e-6),
        init_kl_coef=ppo_config.get("init_kl_coef", ppo_config.get("kl_coef", 0.05)),
        target_kl=ppo_config.get("target_kl", 0.1),
        clip_range=ppo_config.get("clip_range", 0.2),
        value_loss_coef=ppo_config.get("value_loss_coef", 0.1),
        entropy_coef=ppo_config.get("entropy_coef", 0.01),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        gamma=ppo_config.get("gamma", 0.99),
        max_prompt_length=data_config.get("max_prompt_length", 512),
        max_response_length=data_config.get("max_response_length", 512),
        temperature=generation_config.get("temperature", 0.7),
        top_p=generation_config.get("top_p", 0.9),
        raw_config=config_dict,
        config_path=config_path,
    )


def validate_checkpoint_artifact(checkpoint_path: Path) -> Dict[str, Any]:
    """Validate the minimum Hugging Face checkpoint files expected for handoff."""
    files = {path.name for path in checkpoint_path.iterdir()} if checkpoint_path.is_dir() else set()
    has_weights = any(
        name.endswith((".bin", ".safetensors"))
        or name.startswith("pytorch_model")
        or name.startswith("model-")
        for name in files
    )
    has_tokenizer = any(
        name in {
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        }
        for name in files
    )
    return {
        "exists": checkpoint_path.is_dir(),
        "config_json": "config.json" in files,
        "tokenizer": has_tokenizer,
        "weights": has_weights,
        "files": sorted(files),
        "ready_for_awq_conversion": checkpoint_path.is_dir() and "config.json" in files and has_tokenizer and has_weights,
    }


def require_checkpoint(
    results: Mapping[str, TrainingRunResult],
    stage: str,
    fallback: Path,
) -> str:
    """Resolve a prior stage checkpoint path from run result or deterministic fallback."""
    if stage in results:
        return results[stage].checkpoint_path
    return str(fallback)


def update_nested(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copy of base with nested mapping updates applied."""
    merged = copy.deepcopy(dict(base or {}))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = update_nested(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def result_to_dict(result: TrainingRunResult) -> Dict[str, Any]:
    return {
        "checkpoint_path": result.checkpoint_path,
        "metrics": result.metrics,
        "lifecycle_result": (
            None
            if result.lifecycle_result is None
            else {
                key: value
                for key, value in vars(result.lifecycle_result).items()
                if key != "gate_results"
            }
        ),
    }


def float_metric(
    metrics: Mapping[str, Any],
    keys: Iterable[str],
    default: float,
) -> float:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
