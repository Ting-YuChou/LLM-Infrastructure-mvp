#!/usr/bin/env python3
"""
Evaluate model promotion gates from JSON metrics.

Example:
  python scripts/eval_gate.py \
    --metrics outputs/evals/candidate.json \
    --baseline outputs/evals/production.json \
    --config config/eval_gate.json
"""

from __future__ import annotations

import argparse
import json
import operator
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class GateResult:
    """One evaluated gate rule."""

    name: str
    passed: bool
    observed: float
    expected: float
    message: str


@dataclass(frozen=True)
class ThresholdRule:
    """Absolute metric threshold, for example win_rate>=0.55."""

    metric: str
    op: str
    value: float

    @property
    def name(self) -> str:
        return f"{self.metric}{self.op}{self.value}"

    def evaluate(self, metrics: Dict[str, Any]) -> GateResult:
        observed = get_metric(metrics, self.metric)
        passed = OPERATORS[self.op](observed, self.value)
        return GateResult(
            name=self.name,
            passed=passed,
            observed=observed,
            expected=self.value,
            message=f"{self.metric}={observed:g} {self.op} {self.value:g}",
        )


@dataclass(frozen=True)
class MinDeltaRule:
    """Regression guard against a baseline metric."""

    metric: str
    min_delta: float

    @property
    def name(self) -> str:
        return f"{self.metric}.delta>={self.min_delta}"

    def evaluate(self, metrics: Dict[str, Any], baseline: Dict[str, Any]) -> GateResult:
        observed = get_metric(metrics, self.metric)
        baseline_value = get_metric(baseline, self.metric)
        delta = observed - baseline_value
        passed = delta >= self.min_delta
        return GateResult(
            name=self.name,
            passed=passed,
            observed=delta,
            expected=self.min_delta,
            message=(
                f"{self.metric} delta={delta:g} "
                f"(candidate={observed:g}, baseline={baseline_value:g})"
            ),
        )


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_metric(metrics: Dict[str, Any], metric_path: str) -> float:
    """Read a nested metric using dot notation."""
    value: Any = metrics
    for part in metric_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Metric '{metric_path}' not found")
        value = value[part]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric '{metric_path}' must be numeric, got {type(value).__name__}")
    return float(value)


def parse_threshold(raw: str) -> ThresholdRule:
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if op in raw:
            metric, value = raw.split(op, 1)
            metric = metric.strip()
            if not metric:
                raise ValueError(f"Invalid threshold '{raw}': missing metric")
            return ThresholdRule(metric=metric, op=op, value=float(value.strip()))
    raise ValueError(f"Invalid threshold '{raw}': expected metric>=value syntax")


def parse_min_delta(raw: str) -> MinDeltaRule:
    if ":" in raw:
        metric, value = raw.split(":", 1)
    elif "=" in raw:
        metric, value = raw.split("=", 1)
    else:
        raise ValueError(f"Invalid min-delta '{raw}': expected metric:delta syntax")
    metric = metric.strip()
    if not metric:
        raise ValueError(f"Invalid min-delta '{raw}': missing metric")
    return MinDeltaRule(metric=metric, min_delta=float(value.strip()))


def load_config(path: Optional[str]) -> tuple[List[ThresholdRule], List[MinDeltaRule]]:
    if not path:
        return [], []

    config = load_json(path)
    thresholds = [
        parse_threshold(raw)
        for raw in config.get("thresholds", [])
    ]
    min_delta = [
        MinDeltaRule(metric=metric, min_delta=float(value))
        for metric, value in (config.get("min_delta") or {}).items()
    ]
    return thresholds, min_delta


def evaluate_gates(
    metrics: Dict[str, Any],
    thresholds: Iterable[ThresholdRule],
    baseline: Optional[Dict[str, Any]] = None,
    min_delta: Iterable[MinDeltaRule] = (),
) -> List[GateResult]:
    results = [rule.evaluate(metrics) for rule in thresholds]
    for rule in min_delta:
        if baseline is None:
            raise ValueError(f"Baseline metrics are required for {rule.name}")
        results.append(rule.evaluate(metrics, baseline))
    return results


def render_results(results: List[GateResult]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"{status} {result.name}: {result.message}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model promotion gates")
    parser.add_argument("--metrics", required=True, help="Candidate metrics JSON path")
    parser.add_argument("--baseline", help="Baseline metrics JSON path")
    parser.add_argument("--config", help="Gate config JSON path")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Absolute threshold, for example win_rate>=0.55",
    )
    parser.add_argument(
        "--min-delta",
        action="append",
        default=[],
        help="Baseline regression guard, for example win_rate:-0.02",
    )
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    baseline = load_json(args.baseline) if args.baseline else None
    config_thresholds, config_min_delta = load_config(args.config)
    thresholds = config_thresholds + [parse_threshold(raw) for raw in args.threshold]
    min_delta = config_min_delta + [parse_min_delta(raw) for raw in args.min_delta]

    if not thresholds and not min_delta:
        raise ValueError("At least one --threshold, --min-delta, or --config rule is required")

    results = evaluate_gates(
        metrics=metrics,
        thresholds=thresholds,
        baseline=baseline,
        min_delta=min_delta,
    )
    print(render_results(results))
    if all(result.passed for result in results):
        print("eval-gate-ok")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
