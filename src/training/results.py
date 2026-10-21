"""Shared training run result contracts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingRunResult:
    """Final artifact and metrics produced by one training stage."""

    checkpoint_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    lifecycle_result: Optional[Any] = None


def merge_metrics_file(metrics_path: str, checkpoint_path: str, final_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a stage's final metrics into the pipeline promotion metrics JSON."""
    metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    metrics.update(final_metrics)
    metrics["artifact_ready"] = 1.0 if os.path.isdir(checkpoint_path) else 0.0
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics
