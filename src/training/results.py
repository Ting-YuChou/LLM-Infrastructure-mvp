"""Shared training run result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingRunResult:
    """Final artifact and metrics produced by one training stage."""

    checkpoint_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    lifecycle_result: Optional[Any] = None
