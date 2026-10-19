import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training import hf_config  # noqa: E402

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _base_config(**training):
    return {
        "model": {"precision": "bf16"},
        "training": {"learning_rate": 1e-5, "eval_steps": 50, **training},
        "tracking": {},
    }


@pytest.mark.parametrize(
    "value,expected",
    [("bf16", "bf16"), ("bfloat16", "bf16"), ("FP16", "fp16"), ("float32", "fp32")],
)
def test_resolve_precision_normalizes_aliases(value, expected):
    assert hf_config.resolve_precision({"precision": value}) == expected


def test_resolve_precision_defaults_to_bf16_and_rejects_unknown():
    assert hf_config.resolve_precision({}) == "bf16"
    with pytest.raises(ValueError):
        hf_config.resolve_precision({"precision": "int8"})


def test_fp16_full_fine_tune_loads_fp32_master_weights():
    assert hf_config.model_load_dtype("fp16", full_fine_tune=True) is torch.float32
    assert hf_config.model_load_dtype("fp16", full_fine_tune=False) is torch.float16
    assert hf_config.model_load_dtype("bf16", full_fine_tune=True) is torch.bfloat16


def test_precision_flags_follow_model_precision():
    kwargs = hf_config.build_training_arguments_kwargs(
        _base_config(), output_dir="/tmp/out", has_eval=True
    )
    assert kwargs["bf16"] is True
    assert kwargs["fp16"] is False
    assert kwargs["learning_rate"] == 1e-5
    assert kwargs["output_dir"] == "/tmp/out"


def test_legacy_training_precision_keys_are_rejected():
    with pytest.raises(ValueError, match="model.precision"):
        hf_config.build_training_arguments_kwargs(
            _base_config(fp16=True), output_dir="/tmp/out", has_eval=True
        )


def test_legacy_evaluation_strategy_maps_to_eval_strategy():
    kwargs = hf_config.build_training_arguments_kwargs(
        _base_config(evaluation_strategy="epoch"), output_dir="/tmp/out", has_eval=True
    )
    assert kwargs["eval_strategy"] == "epoch"
    assert "evaluation_strategy" not in kwargs


def test_without_eval_dataset_eval_is_disabled():
    kwargs = hf_config.build_training_arguments_kwargs(
        _base_config(), output_dir="/tmp/out", has_eval=False
    )
    assert kwargs["eval_strategy"] == "no"
    assert "eval_steps" not in kwargs


def test_optimizer_and_gradient_checkpointing_are_wired():
    config = _base_config(optimizer="adamw_torch_fused")
    config["model"]["gradient_checkpointing"] = True
    kwargs = hf_config.build_training_arguments_kwargs(config, output_dir="/tmp/out", has_eval=True)
    assert kwargs["optim"] == "adamw_torch_fused"
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_early_stopping_enables_best_model_tracking():
    config = _base_config(save_strategy="steps")
    config["safety"] = {
        "early_stopping": {"enabled": True, "patience": 2, "metric": "eval_reward_accuracy", "mode": "max"}
    }
    kwargs = hf_config.build_training_arguments_kwargs(config, output_dir="/tmp/out", has_eval=True)
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_reward_accuracy"
    assert kwargs["greater_is_better"] is True
    assert hf_config.early_stopping_settings(config, has_eval=False) is None


def test_report_targets_skip_mlflow_and_default_to_none():
    assert hf_config.report_targets({}) == ["none"]
    assert hf_config.report_targets(
        {"tensorboard_enabled": True, "wandb_enabled": True, "mlflow_enabled": True}
    ) == ["tensorboard", "wandb"]


@pytest.mark.parametrize("config_name", ["sft_config.yaml", "reward_config.yaml"])
def test_repo_training_configs_use_bf16_and_build_kwargs(config_name):
    raw_config = yaml.safe_load((CONFIG_DIR / config_name).read_text())
    kwargs = hf_config.build_training_arguments_kwargs(raw_config, output_dir="/tmp/out", has_eval=True)
    assert kwargs["bf16"] is True
    assert kwargs["fp16"] is False
    assert kwargs["eval_strategy"] == "steps"
