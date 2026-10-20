"""
End-to-end smoke tests against the real transformers Trainer with a tiny Llama.

Other training tests stub transformers, which is how TrainingArguments(
evaluation_strategy=...) and the reward eval crash went unnoticed. These tests
run the trainers for a couple of CPU steps instead. They need torch,
transformers and datasets (peft for the LoRA case) and skip otherwise.
"""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

SRC_DIR = Path(__file__).parent.parent.parent / "src"
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
sys.path.insert(0, str(SRC_DIR))

_STUBBABLE_PREFIXES = ("transformers", "datasets", "peft", "training")


# Real modules imported by an earlier test, reused because datasets cannot be
# imported twice in one process (pyarrow extension types register globally).
_REAL_MODULES = {}


def _is_stubbable(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _STUBBABLE_PREFIXES)


def _must_swap(name, module):
    """Stub modules (no import spec) and training modules bound to stubs."""
    return getattr(module, "__spec__", None) is None or name.split(".")[0] == "training"


@pytest.fixture
def real_training():
    """Import the training modules against real transformers/datasets, then restore stubs."""
    before = {name: module for name, module in sys.modules.items() if _is_stubbable(name)}
    saved = {name: module for name, module in before.items() if _must_swap(name, module)}
    for name in saved:
        del sys.modules[name]
    sys.modules.update(_REAL_MODULES)
    try:
        for required in ("torch", "transformers", "datasets", "tokenizers"):
            if importlib.util.find_spec(required) is None:
                pytest.skip(f"{required} is not installed")
        modules = {
            name: importlib.import_module(f"training.{name}")
            for name in ("sft_trainer", "reward_trainer", "dpo_trainer", "rlhf_trainer")
        }
        modules["transformers"] = importlib.import_module("transformers")
        yield modules
    finally:
        for name in [name for name in sys.modules if _is_stubbable(name)]:
            module = sys.modules[name]
            if before.get(name) is not module:
                _REAL_MODULES[name] = module
                del sys.modules[name]
        sys.modules.update(saved)


def _use_cpu(module, training_arguments_cls):
    """Wrap the real TrainingArguments so tests never pick up CUDA/MPS."""
    module.TrainingArguments = lambda **kwargs: training_arguments_cls(use_cpu=True, **kwargs)


@pytest.fixture
def tiny_llama_dir(real_training, tmp_path):
    """A plain base checkpoint: no chat tokens, pad or template, like Llama 3 base."""
    from tiny_llama import build_tiny_llama, build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    model_dir = tmp_path / "tiny-llama"
    build_tiny_llama(len(tokenizer)).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def _write_json(path, rows):
    path.write_text(json.dumps(rows))
    return str(path)


def _raw_config(base_config_name, model_dir, output_dir, train_file, val_file, **overrides):
    """Start from the repo YAML so the real config keys go through TrainingArguments."""
    raw = yaml.safe_load((CONFIG_DIR / base_config_name).read_text())
    raw["model"].update(name=str(model_dir), precision="fp32", use_flash_attention=False)
    raw["model"].pop("cache_dir", None)
    raw["data"].update(train_file=train_file, val_file=val_file, max_length=24)
    raw["training"].update(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    raw["tracking"].update(mlflow_enabled=False, tensorboard_enabled=False, wandb_enabled=False)
    raw.update(overrides)
    return raw


def _sft_rows(count):
    return [
        {"prompt": "explain tests", "completion": "they catch regressions"}
        for _ in range(count)
    ]


@pytest.mark.parametrize("use_lora", [False, True])
def test_sft_trains_and_evaluates_with_real_trainer(real_training, tiny_llama_dir, tmp_path, use_lora):
    if use_lora and importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed")
    sft_trainer = real_training["sft_trainer"]
    _use_cpu(sft_trainer, real_training["transformers"].TrainingArguments)

    raw = _raw_config(
        "sft_config.yaml",
        tiny_llama_dir,
        tmp_path / "sft",
        _write_json(tmp_path / "train.json", _sft_rows(4)),
        _write_json(tmp_path / "val.json", _sft_rows(2)),
    )
    config = sft_trainer.SFTConfig(
        model_name=raw["model"]["name"],
        train_file=raw["data"]["train_file"],
        val_file=raw["data"]["val_file"],
        output_dir=raw["training"]["output_dir"],
        max_length=24,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        raw_config=raw,
    )

    result = sft_trainer.SFTTrainer(config, use_lora=use_lora).train()

    assert "eval_loss" in result.metrics
    saved_tokenizer = real_training["transformers"].AutoTokenizer.from_pretrained(result.checkpoint_path)
    assert "<|start_header_id|>" in saved_tokenizer.chat_template
    assert saved_tokenizer.eos_token == "<|eot_id|>"
    assert saved_tokenizer.pad_token not in {"<|eot_id|>", "</s>"}
    generation_config = json.loads(Path(result.checkpoint_path, "generation_config.json").read_text())
    assert saved_tokenizer.convert_tokens_to_ids("<|eot_id|>") in generation_config["eos_token_id"]
    weights = "adapter_model.safetensors" if use_lora else "model.safetensors"
    assert Path(result.checkpoint_path, weights).exists()


def test_reward_model_trains_and_reports_pairwise_metrics(real_training, tiny_llama_dir, tmp_path):
    reward_trainer = real_training["reward_trainer"]
    _use_cpu(reward_trainer, real_training["transformers"].TrainingArguments)

    pairs = [
        {"prompt": "answer question", "chosen": "good answer", "rejected": "bad answer"}
        for _ in range(4)
    ]
    raw = _raw_config(
        "reward_config.yaml",
        tiny_llama_dir,
        tmp_path / "reward",
        _write_json(tmp_path / "pairs_train.json", pairs),
        _write_json(tmp_path / "pairs_val.json", pairs[:2]),
    )
    config = reward_trainer.RewardModelConfig(
        model_name=raw["model"]["name"],
        train_file=raw["data"]["train_file"],
        val_file=raw["data"]["val_file"],
        output_dir=raw["training"]["output_dir"],
        max_length=24,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        raw_config=raw,
    )

    result = reward_trainer.RewardModelTrainer(config).train()

    assert 0.0 <= result.metrics["reward_accuracy"] <= 1.0
    assert "eval_reward_margin" in result.metrics
    assert Path(result.checkpoint_path, "model.safetensors").exists()


@pytest.fixture
def tiny_sft_dir(real_training, tiny_llama_dir, tmp_path):
    """tiny_llama_dir after SFT: chat tokens already in the vocab, as in a real SFT checkpoint."""
    from training.chat_format import ensure_chat_tokenizer, resize_embeddings_if_needed

    transformers = real_training["transformers"]
    tokenizer = ensure_chat_tokenizer(transformers.AutoTokenizer.from_pretrained(tiny_llama_dir))
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny_llama_dir)
    resize_embeddings_if_needed(model, tokenizer)
    sft_dir = tmp_path / "tiny-sft"
    model.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    return sft_dir


def _dpo_config(dpo_trainer, model_dir, tmp_path, learning_rate):
    pairs = [
        {"prompt": "answer question", "chosen": "good answer", "rejected": "bad answer"}
        for _ in range(4)
    ]
    raw = _raw_config(
        "dpo_config.yaml",
        model_dir,
        tmp_path / "dpo",
        _write_json(tmp_path / "pairs_train.json", pairs),
        _write_json(tmp_path / "pairs_val.json", pairs[:2]),
    )
    return dpo_trainer.DPOConfig(
        model_name=raw["model"]["name"],
        train_file=raw["data"]["train_file"],
        val_file=raw["data"]["val_file"],
        output_dir=raw["training"]["output_dir"],
        max_length=24,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=learning_rate,
        beta=0.1,
        raw_config=raw,
    )


def test_dpo_loss_starts_at_ln2_against_its_own_reference(real_training, tiny_sft_dir, tmp_path):
    import math

    dpo_trainer = real_training["dpo_trainer"]
    _use_cpu(dpo_trainer, real_training["transformers"].TrainingArguments)

    # lr=0 keeps the policy equal to the reference, so every pair's DPO loss is
    # -log(sigmoid(0)) = ln 2 and the implicit rewards are 0.
    result = dpo_trainer.DPOModelTrainer(_dpo_config(dpo_trainer, tiny_sft_dir, tmp_path, 0.0)).train()

    assert result.metrics["eval_loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert result.metrics["eval_rewards_chosen"] == pytest.approx(0.0, abs=1e-4)
    assert result.metrics["dpo_reward_margin"] == pytest.approx(0.0, abs=1e-4)
    checkpoint = Path(result.checkpoint_path)
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "generation_config.json").exists()


def test_dpo_training_moves_policy_towards_chosen(real_training, tiny_sft_dir, tmp_path):
    dpo_trainer = real_training["dpo_trainer"]
    _use_cpu(dpo_trainer, real_training["transformers"].TrainingArguments)
    config = _dpo_config(dpo_trainer, tiny_sft_dir, tmp_path, 5e-2)
    config.raw_config["training"].update(max_steps=8, lr_scheduler_type="constant", warmup_ratio=0.0)

    result = dpo_trainer.DPOModelTrainer(config).train()

    assert result.metrics["dpo_reward_margin"] > 0
    assert result.metrics["dpo_reward_accuracy"] == 1.0


def test_dpo_labels_mask_prompt_and_padding(real_training, tiny_sft_dir):
    reward_trainer = real_training["reward_trainer"]
    tokenizer = real_training["transformers"].AutoTokenizer.from_pretrained(tiny_sft_dir)
    preprocessor = reward_trainer.RewardDataPreprocessor(tokenizer, max_length=24, include_labels=True)

    batch = preprocessor.create_comparison_pairs(
        {"prompt": ["answer question"], "chosen": ["good answer"], "rejected": ["bad"]}
    )

    for side in ("chosen", "rejected"):
        ids, mask, labels = (batch[f"{side}_{key}"][0] for key in ("input_ids", "attention_mask", "labels"))
        response = [label for label in labels if label != -100]
        # Response tokens end with <|eot_id|> and sit right before the padding.
        assert response[-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>")
        end = sum(mask)
        assert labels[end - len(response):end] == ids[end - len(response):end]
        assert all(label == -100 for label in labels[end:])


def test_ppo_value_head_accepts_bf16_policy_hidden_states(real_training, tiny_llama_dir):
    import torch

    rlhf_trainer = real_training["rlhf_trainer"]
    transformers = real_training["transformers"]
    trainer = rlhf_trainer.RLHFTrainer.__new__(rlhf_trainer.RLHFTrainer)
    trainer.config = rlhf_trainer.RLHFConfig(policy_model_path="p", reward_model_path="r", output_dir="o")
    trainer.device = torch.device("cpu")
    trainer.policy_model = transformers.AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir,
        torch_dtype=torch.bfloat16,
    )
    trainer.value_head = rlhf_trainer.ValueHead(trainer.policy_model.config.hidden_size)

    input_ids = torch.tensor([[1, 5, 6, 7, 8]])
    response_ids = input_ids[:, 3:]
    log_probs, entropy, values = trainer.compute_log_probs_and_entropy_per_token(
        input_ids,
        response_ids,
        attention_mask=torch.ones_like(input_ids),
    )

    assert values.dtype is torch.float32
    assert values.shape == response_ids.shape
    assert log_probs.shape == response_ids.shape
    assert torch.isfinite(values).all()
