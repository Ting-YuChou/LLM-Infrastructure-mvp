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
WORDS = (
    "explain tests they catch regressions write code good bad answer question "
    "the a is and to of models train serve llama"
).split()


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
            for name in ("sft_trainer", "reward_trainer", "rlhf_trainer")
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
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers

    transformers = real_training["transformers"]
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for word in WORDS + ["###", "instruction:", "response:", "prompt:"]:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    model_dir = tmp_path / "tiny-llama"
    transformers.LlamaForCausalLM(config).save_pretrained(model_dir)
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
    assert Path(result.checkpoint_path, "tokenizer_config.json").exists()
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


def test_ppo_value_head_accepts_bf16_policy_hidden_states(real_training, tiny_llama_dir):
    import torch

    rlhf_trainer = real_training["rlhf_trainer"]
    transformers = real_training["transformers"]
    trainer = rlhf_trainer.RLHFTrainer.__new__(rlhf_trainer.RLHFTrainer)
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
