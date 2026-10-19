"""
MLflow lifecycle integration for training jobs.

This module keeps training code focused on model work while providing a shared
path for experiment tracking, checkpoint registration, and alias-based
candidate/champion promotion.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers is a runtime dependency
    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base class used when transformers is unavailable."""


try:
    from scripts.eval_gate import evaluate_gates, load_config, load_json, render_results
except ImportError:  # pragma: no cover - supports direct module execution
    evaluate_gates = load_config = load_json = render_results = None  # type: ignore[assignment]


try:
    from .distributed import is_main_process, world_size
except ImportError:  # pragma: no cover - supports direct module execution
    from distributed import is_main_process, world_size  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionConfig:
    """Promotion-gate CLI/config inputs."""

    metrics_path: Optional[str] = None
    baseline_metrics_path: Optional[str] = None
    gate_config_path: Optional[str] = None
    skip_promotion: bool = False


@dataclass(frozen=True)
class LifecycleConfig:
    """Resolved MLflow lifecycle settings for one training job."""

    enabled: bool
    tracking_uri: str
    experiment_name: str
    auto_register: bool
    model_name: str
    model_type: str
    run_name: Optional[str] = None
    config_path: Optional[str] = None
    raw_config: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelLifecycleResult:
    """Result of checkpoint registration and optional promotion."""

    model_name: str
    version: str
    run_id: str
    artifact_uri: str
    validation_status: str
    promoted: bool = False
    gate_results: List[Any] = field(default_factory=list)
    message: Optional[str] = None


class MLflowMetricsCallback(TrainerCallback):
    """Hugging Face Trainer callback that forwards numeric logs to MLflow."""

    def __init__(self, lifecycle: "MLflowTrainingLifecycle"):
        self.lifecycle = lifecycle

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if logs:
            self.lifecycle.log_metrics(logs, step=getattr(state, "global_step", None))


class MLflowTrainingLifecycle:
    """Context manager for one MLflow-backed training lifecycle."""

    artifact_path = "model"
    candidate_alias = "candidate"
    champion_alias = "champion"

    def __init__(
        self,
        config: LifecycleConfig,
        promotion: Optional[PromotionConfig] = None,
        mlflow_module: Any = None,
        client: Any = None,
    ):
        self.config = config
        self.promotion = promotion or PromotionConfig()
        self._mlflow = mlflow_module
        self.client = client
        self._run_context = None
        self._run = None
        self.run_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @classmethod
    def from_training_config(
        cls,
        raw_config: Optional[Dict[str, Any]],
        model_type: str,
        config_path: Optional[str] = None,
        promotion: Optional[PromotionConfig] = None,
        extra_tags: Optional[Dict[str, str]] = None,
        mlflow_module: Any = None,
        client: Any = None,
    ) -> "MLflowTrainingLifecycle":
        raw_config = raw_config or {}
        tracking_config = raw_config.get("tracking") or {}
        registry_config = raw_config.get("registry") or {}

        promotion = promotion or PromotionConfig()
        gate_config_path = (
            promotion.gate_config_path
            or registry_config.get("gate_config")
            or registry_config.get("promotion_gate_config")
        )
        promotion = PromotionConfig(
            metrics_path=promotion.metrics_path,
            baseline_metrics_path=promotion.baseline_metrics_path,
            gate_config_path=gate_config_path,
            skip_promotion=promotion.skip_promotion,
        )

        model_name = (
            registry_config.get("model_name")
            or _model_name_from_config(raw_config)
            or f"{model_type}-model"
        )

        tags = {
            "project": "llm-infrastructure",
            "model_type": model_type,
            "lifecycle": "training",
            "world_size": str(world_size()),
        }
        tags.update(_stringify_tags(registry_config.get("tags") or {}))
        if extra_tags:
            tags.update(_stringify_tags(extra_tags))

        config = LifecycleConfig(
            # One run per job: under torchrun only rank 0 talks to MLflow.
            enabled=bool(tracking_config.get("mlflow_enabled", False)) and is_main_process(),
            tracking_uri=str(
                tracking_config.get("mlflow_tracking_uri")
                or os.getenv("MLFLOW_TRACKING_URI")
                or "http://localhost:5000"
            ),
            experiment_name=str(
                tracking_config.get("mlflow_experiment_name")
                or f"{model_type}-training"
            ),
            auto_register=bool(registry_config.get("auto_register", True)),
            model_name=str(model_name),
            model_type=model_type,
            run_name=(
                tracking_config.get("mlflow_run_name")
                or tracking_config.get("wandb_run_name")
            ),
            config_path=config_path,
            raw_config=raw_config,
            tags=tags,
        )
        return cls(config, promotion, mlflow_module=mlflow_module, client=client)

    def __enter__(self) -> "MLflowTrainingLifecycle":
        if not self.enabled:
            return self

        mlflow = self._require_mlflow()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        if self.client is None:
            self.client = self._new_client()

        experiment_id = self._get_or_create_experiment()
        self._run_context = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.config.run_name,
        )
        self._run = self._run_context.__enter__()
        self.run_id = self._run.info.run_id

        mlflow.set_tags(self.config.tags)
        git_commit = _git_commit()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)
        mlflow.log_params(flatten_params(self.config.raw_config))
        if self.config.config_path and Path(self.config.config_path).exists():
            mlflow.log_artifact(self.config.config_path, artifact_path="config")

        logger.info(
            "Started MLflow run %s for %s",
            self.run_id,
            self.config.model_name,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:  # type: ignore[no-untyped-def]
        if not self.enabled or self._run_context is None:
            return False

        if exc_type is not None:
            try:
                self._require_mlflow().set_tag("run_status", "failed")
            except Exception:
                logger.exception("Failed to tag MLflow run failure")

        self._run_context.__exit__(exc_type, exc, traceback)
        self._run_context = None
        self._run = None
        return False

    def log_metrics(
        self,
        metrics: Optional[Dict[str, Any]],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log numeric metrics if lifecycle tracking is enabled."""
        if not self.enabled or not metrics:
            return

        mlflow = self._require_mlflow()
        for key, value in flatten_numeric_metrics(metrics, prefix=prefix).items():
            mlflow.log_metric(key, value, step=step)

    def register_checkpoint(
        self,
        checkpoint_path: str,
        final_metrics: Optional[Dict[str, Any]] = None,
        extra_tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Optional[ModelLifecycleResult]:
        """Log and register a final checkpoint, then evaluate promotion gates."""
        if not self.enabled:
            return None

        if not self.run_id:
            raise RuntimeError("MLflow run is not active; use lifecycle as a context manager")

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        if final_metrics:
            self.log_metrics(final_metrics, prefix="final")

        mlflow = self._require_mlflow()
        if checkpoint.is_dir():
            mlflow.log_artifacts(str(checkpoint), artifact_path=self.artifact_path)
        else:
            mlflow.log_artifact(str(checkpoint), artifact_path=self.artifact_path)

        artifact_uri = f"runs:/{self.run_id}/{self.artifact_path}"
        if not self.config.auto_register:
            logger.info("MLflow auto_register is disabled; checkpoint artifacts logged only")
            return None

        self._ensure_registered_model()
        version_tags = {
            **self.config.tags,
            **_stringify_tags(extra_tags or {}),
            "artifact_format": "huggingface_checkpoint",
            "artifact_path": self.artifact_path,
            "lifecycle_alias": self.candidate_alias,
            "validation_status": "pending",
        }
        model_version = self._create_model_version(artifact_uri, version_tags)
        version = str(model_version.version)

        if description:
            self.client.update_model_version(
                name=self.config.model_name,
                version=version,
                description=description,
            )

        self._set_version_tags(version, version_tags)
        self.client.set_registered_model_alias(
            self.config.model_name,
            self.candidate_alias,
            version,
        )

        result = self._evaluate_and_promote(version, artifact_uri)
        logger.info(
            "Registered MLflow model %s v%s with validation_status=%s promoted=%s",
            self.config.model_name,
            version,
            result.validation_status,
            result.promoted,
        )
        return result

    def _evaluate_and_promote(self, version: str, artifact_uri: str) -> ModelLifecycleResult:
        if self.promotion.skip_promotion:
            self._set_version_tags(version, {"validation_status": "pending"})
            return self._result(version, artifact_uri, "pending", False, message="promotion skipped")

        if not self.promotion.metrics_path:
            self._set_version_tags(version, {"validation_status": "pending"})
            return self._result(version, artifact_uri, "pending", False, message="promotion metrics absent")

        try:
            gate_results = self._run_promotion_gates()
        except Exception as exc:
            message = f"promotion gate failed closed: {exc}"
            logger.warning(message)
            self._log_text(message, "eval/gate_error.txt")
            self._set_version_tags(
                version,
                {
                    "validation_status": "failed",
                    "validation_error": str(exc),
                },
            )
            return self._result(version, artifact_uri, "failed", False, message=message)

        passed = all(result.passed for result in gate_results)
        self._log_gate_results(gate_results)

        if not passed:
            self._set_version_tags(version, {"validation_status": "failed"})
            return self._result(version, artifact_uri, "failed", False, gate_results)

        self._promote_to_champion(version)
        self._set_version_tags(
            version,
            {
                "validation_status": "passed",
                "lifecycle_alias": self.champion_alias,
            },
        )
        return self._result(version, artifact_uri, "passed", True, gate_results)

    def _run_promotion_gates(self) -> List[Any]:
        if not self.promotion.gate_config_path:
            raise ValueError("promotion metrics were provided but no gate config was configured")
        if load_json is None or load_config is None or evaluate_gates is None:
            raise RuntimeError("scripts.eval_gate could not be imported")

        metrics = load_json(self.promotion.metrics_path)
        self._log_eval_metrics(metrics)

        thresholds, min_delta = load_config(self.promotion.gate_config_path)
        if not thresholds and not min_delta:
            raise ValueError("promotion gate config does not contain any rules")

        baseline = None
        if min_delta:
            baseline = self._load_baseline_metrics()
            if baseline is None:
                raise ValueError("baseline metrics are required for promotion delta gates")

        return evaluate_gates(
            metrics=metrics,
            thresholds=thresholds,
            baseline=baseline,
            min_delta=min_delta,
        )

    def _log_eval_metrics(self, metrics: Dict[str, Any]) -> None:
        mlflow = self._require_mlflow()
        metrics_path = self.promotion.metrics_path
        if metrics_path:
            mlflow.log_artifact(metrics_path, artifact_path="eval")
        if self.promotion.baseline_metrics_path:
            mlflow.log_artifact(self.promotion.baseline_metrics_path, artifact_path="eval")
        if hasattr(mlflow, "log_dict"):
            mlflow.log_dict(metrics, "eval/promotion_metrics.json")
        self.log_metrics(metrics, prefix="promotion")

    def _load_baseline_metrics(self) -> Optional[Dict[str, Any]]:
        if load_json is None:
            return None
        if self.promotion.baseline_metrics_path:
            return load_json(self.promotion.baseline_metrics_path)

        try:
            champion = self.client.get_model_version_by_alias(
                self.config.model_name,
                self.champion_alias,
            )
        except Exception:
            return None

        run_id = getattr(champion, "run_id", None)
        if not run_id:
            return None

        try:
            path = self._require_mlflow().artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/eval/promotion_metrics.json"
            )
        except Exception:
            logger.exception("Could not download previous champion promotion metrics")
            return None
        return load_json(path)

    def _log_gate_results(self, gate_results: Iterable[Any]) -> None:
        results = list(gate_results)
        if render_results is not None:
            self._log_text(render_results(results), "eval/gate_results.txt")
        for result in results:
            key = _metric_key(f"gate.{result.name}.passed")
            self._require_mlflow().log_metric(key, 1.0 if result.passed else 0.0)

    def _promote_to_champion(self, version: str) -> None:
        try:
            previous = self.client.get_model_version_by_alias(
                self.config.model_name,
                self.champion_alias,
            )
        except Exception:
            previous = None

        if previous is not None and str(getattr(previous, "version", "")) != str(version):
            self._set_version_tags(
                str(previous.version),
                {"lifecycle_alias": "archived", "replaced_by_version": str(version)},
            )

        self.client.set_registered_model_alias(
            self.config.model_name,
            self.champion_alias,
            version,
        )

    def _result(
        self,
        version: str,
        artifact_uri: str,
        validation_status: str,
        promoted: bool,
        gate_results: Optional[List[Any]] = None,
        message: Optional[str] = None,
    ) -> ModelLifecycleResult:
        return ModelLifecycleResult(
            model_name=self.config.model_name,
            version=version,
            run_id=self.run_id or "",
            artifact_uri=artifact_uri,
            validation_status=validation_status,
            promoted=promoted,
            gate_results=gate_results or [],
            message=message,
        )

    def _require_mlflow(self) -> Any:
        if self._mlflow is not None:
            return self._mlflow

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking is enabled, but the 'mlflow' package is not installed"
            ) from exc

        self._mlflow = mlflow
        return self._mlflow

    def _new_client(self) -> Any:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking is enabled, but MlflowClient could not be imported"
            ) from exc
        return MlflowClient(tracking_uri=self.config.tracking_uri)

    def _get_or_create_experiment(self) -> str:
        mlflow = self._require_mlflow()
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment:
            return experiment.experiment_id
        return mlflow.create_experiment(
            self.config.experiment_name,
            tags={"project": "llm-infrastructure"},
        )

    def _ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.config.model_name)
        except Exception:
            tags = {
                "project": "llm-infrastructure",
                "model_type": self.config.model_type,
            }
            try:
                self.client.create_registered_model(
                    self.config.model_name,
                    tags=tags,
                )
            except TypeError:
                self.client.create_registered_model(self.config.model_name)

    def _create_model_version(self, artifact_uri: str, tags: Dict[str, str]) -> Any:
        try:
            return self.client.create_model_version(
                name=self.config.model_name,
                source=artifact_uri,
                run_id=self.run_id,
                tags=tags,
            )
        except TypeError:
            return self.client.create_model_version(
                name=self.config.model_name,
                source=artifact_uri,
                run_id=self.run_id,
            )

    def _set_version_tags(self, version: str, tags: Dict[str, str]) -> None:
        for key, value in _stringify_tags(tags).items():
            self.client.set_model_version_tag(
                self.config.model_name,
                version,
                key,
                value,
            )

    def _log_text(self, text: str, artifact_file: str) -> None:
        mlflow = self._require_mlflow()
        if hasattr(mlflow, "log_text"):
            mlflow.log_text(text, artifact_file)
        else:
            logger.info("%s\n%s", artifact_file, text)


def flatten_params(data: Dict[str, Any]) -> Dict[str, str]:
    """Flatten config data into MLflow parameter-safe strings."""
    return {
        key: _param_value(value)
        for key, value in _flatten(data).items()
        if _is_param_value(value)
    }


def flatten_numeric_metrics(
    data: Dict[str, Any],
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    """Flatten nested numeric metrics for MLflow logging."""
    metrics: Dict[str, float] = {}
    for key, value in _flatten(data).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        metric_key = _metric_key(f"{prefix}.{key}" if prefix else key)
        metrics[metric_key] = float(value)
    return metrics


def _flatten(data: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(data, dict):
        flattened: Dict[str, Any] = {}
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value, child_prefix))
        return flattened
    if isinstance(data, list):
        return {prefix: data}
    return {prefix: data}


def _is_param_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list))


def _param_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        rendered = json.dumps(value, sort_keys=True)
    else:
        rendered = str(value)
    if len(rendered) > 250:
        return f"{rendered[:247]}..."
    return rendered


def _metric_key(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-/]", "_", raw)


def _stringify_tags(tags: Dict[str, Any]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in tags.items()}


def _model_name_from_config(raw_config: Dict[str, Any]) -> Optional[str]:
    model_config = raw_config.get("model") or {}
    if isinstance(model_config.get("name"), str):
        return Path(model_config["name"]).name or model_config["name"]
    policy_config = model_config.get("policy") or {}
    if isinstance(policy_config.get("name"), str):
        return Path(policy_config["name"]).name or policy_config["name"]
    return None


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
