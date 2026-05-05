import pytest

from scripts.eval_gate import (
    MinDeltaRule,
    ThresholdRule,
    evaluate_gates,
    parse_min_delta,
    parse_threshold,
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
