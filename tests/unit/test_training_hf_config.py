import os
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


def _distributed_config(strategy="fsdp", gradient_checkpointing=True, **fsdp):
    config = _base_config()
    config["model"]["gradient_checkpointing"] = gradient_checkpointing
    config["distributed"] = {"strategy": strategy, "fsdp": fsdp}
    return config


def test_fsdp_kwargs_auto_wrap_llama_layers_and_move_checkpointing_into_fsdp():
    kwargs = hf_config.build_training_arguments_kwargs(
        _distributed_config(), output_dir="/tmp/out", has_eval=True, world_size=8
    )

    # Without auto_wrap accelerate wraps the whole model as one FSDP unit.
    assert kwargs["fsdp"] == "full_shard auto_wrap"
    assert kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["LlamaDecoderLayer"]
    assert kwargs["fsdp_config"]["use_orig_params"] is True
    # Trainer raises if gradient_checkpointing and FSDP activation_checkpointing are both on.
    assert kwargs["fsdp_config"]["activation_checkpointing"] is True
    assert "gradient_checkpointing" not in kwargs
    assert "gradient_checkpointing_kwargs" not in kwargs
    assert kwargs["ddp_timeout"] == 3600


def test_fsdp_offload_and_explicit_activation_checkpointing():
    kwargs = hf_config.build_training_arguments_kwargs(
        _distributed_config(offload=True, activation_checkpointing=False),
        output_dir="/tmp/out",
        has_eval=False,
        world_size=8,
    )

    assert kwargs["fsdp"] == "full_shard auto_wrap offload"
    assert kwargs["fsdp_config"]["activation_checkpointing"] is False
    assert kwargs["gradient_checkpointing"] is True


def test_ddp_kwargs_keep_gradient_checkpointing_and_disable_unused_parameter_search():
    kwargs = hf_config.build_training_arguments_kwargs(
        _distributed_config(strategy="ddp"), output_dir="/tmp/out", has_eval=True, world_size=8
    )

    assert kwargs["ddp_find_unused_parameters"] is False
    assert "fsdp" not in kwargs
    assert kwargs["gradient_checkpointing"] is True


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_single_process_ignores_distributed_strategy(strategy, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = _distributed_config(strategy=strategy)

    kwargs = hf_config.build_training_arguments_kwargs(config, output_dir="/tmp/out", has_eval=True)

    assert not {"fsdp", "fsdp_config", "ddp_timeout", "ddp_find_unused_parameters"} & set(kwargs)
    assert kwargs["gradient_checkpointing"] is True
    assert hf_config.fsdp_environment(config, world_size=1) == {}


def test_world_size_defaults_to_torchrun_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")

    kwargs = hf_config.build_training_arguments_kwargs(
        _distributed_config(), output_dir="/tmp/out", has_eval=True
    )

    assert kwargs["fsdp"] == "full_shard auto_wrap"


@pytest.mark.parametrize(
    "config",
    [
        _distributed_config(strategy="zero3"),
        _distributed_config(version=2),
        _distributed_config(sharding="no_shard"),
        _distributed_config(cpu_ram_efficient_loading=True, sync_module_states=False),
        {**_distributed_config(), "training": {"deepspeed": "ds.json"}},
    ],
)
def test_invalid_distributed_settings_raise(config):
    with pytest.raises(ValueError):
        hf_config.build_training_arguments_kwargs(config, output_dir="/tmp/out", has_eval=True, world_size=8)


def test_fsdp_environment_shards_intermediate_checkpoints():
    assert hf_config.fsdp_environment(_distributed_config(), world_size=8) == {
        "FSDP_STATE_DICT_TYPE": "SHARDED_STATE_DICT"
    }
    assert hf_config.fsdp_environment(
        _distributed_config(checkpoint_state_dict_type="full"), world_size=8
    ) == {"FSDP_STATE_DICT_TYPE": "FULL_STATE_DICT"}
    assert hf_config.fsdp_environment(_distributed_config(strategy="ddp"), world_size=8) == {}


@pytest.mark.parametrize("config_name", ["sft_config.yaml", "reward_config.yaml"])
def test_repo_training_configs_shard_with_fsdp_on_eight_gpus(config_name):
    raw_config = yaml.safe_load((CONFIG_DIR / config_name).read_text())

    kwargs = hf_config.build_training_arguments_kwargs(
        raw_config, output_dir="/tmp/out", has_eval=True, world_size=8
    )

    assert kwargs["fsdp"] == "full_shard auto_wrap"
    assert kwargs["fsdp_config"]["activation_checkpointing"] is True
    assert "gradient_checkpointing" not in kwargs


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_real_training_arguments_export_fsdp_settings_to_accelerate(restore_environ):
    transformers = pytest.importorskip("transformers")
    raw_config = yaml.safe_load((CONFIG_DIR / "sft_config.yaml").read_text())
    raw_config["tracking"] = {}
    kwargs = hf_config.build_training_arguments_kwargs(
        raw_config, output_dir="/tmp/out", has_eval=True, world_size=8
    )
    os.environ.update(hf_config.fsdp_environment(raw_config, world_size=8))

    args = transformers.TrainingArguments(use_cpu=True, **kwargs)

    # accelerate builds its FSDP plugin from these variables (it needs a GPU to
    # construct, so the plugin itself is only exercised on RunPod).
    assert os.environ["ACCELERATE_USE_FSDP"] == "true"
    assert os.environ["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
    assert os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] == "LlamaDecoderLayer"
    assert os.environ["FSDP_USE_ORIG_PARAMS"] == "true"
    assert os.environ["FSDP_STATE_DICT_TYPE"] == "SHARDED_STATE_DICT"
    assert args.gradient_checkpointing is False
    assert args.fsdp_config["activation_checkpointing"] is True
