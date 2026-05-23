import pytest

from scripts.eval_gate import (
    MinDeltaRule,
    ThresholdRule,
    evaluate_gates,
    parse_min_delta,
    parse_threshold,
    load_config,
)


def test_threshold_rule_parsing_and_evaluation():
    rule = parse_threshold("summary.win_rate>=0.55")
    result = rule.evaluate({"summary": {"win_rate": 0.6}})

    assert isinstance(rule, ThresholdRule)
    assert result.passed
    assert result.observed == 0.6


def test_eval_gate_detects_failed_threshold():
    results = evaluate_gates(
        metrics={"toxicity": 0.05},
        thresholds=[parse_threshold("toxicity<=0.02")],
    )

    assert len(results) == 1
    assert not results[0].passed


def test_eval_gate_supports_baseline_regression_guard():
    rule = parse_min_delta("win_rate:-0.02")
    results = evaluate_gates(
        metrics={"win_rate": 0.57},
        thresholds=[],
        baseline={"win_rate": 0.58},
        min_delta=[rule],
    )

    assert isinstance(rule, MinDeltaRule)
    assert results[0].passed
    assert results[0].observed == pytest.approx(-0.01)


def test_eval_gate_requires_baseline_for_min_delta():
    with pytest.raises(ValueError):
        evaluate_gates(
            metrics={"win_rate": 0.57},
            thresholds=[],
            min_delta=[parse_min_delta("win_rate:-0.02")],
        )


def test_production_gate_includes_baseline_quality_guards():
    thresholds, min_delta = load_config("config/eval_gate.production.json")
    threshold_names = {rule.metric for rule in thresholds}
    delta_names = {rule.metric for rule in min_delta}

    assert {"task_eval_score", "safety_score", "latency_p95_ms", "artifact_checksum_ready"} <= threshold_names
    assert {"task_eval_score", "safety_score"} <= delta_names
