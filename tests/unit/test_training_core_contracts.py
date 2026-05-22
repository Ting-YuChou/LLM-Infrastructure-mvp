import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = lambda *args, **kwargs: []
datasets_stub.Dataset = type(
    "Dataset",
    (),
    {"from_list": classmethod(lambda cls, data: data)},
)
sys.modules.setdefault("datasets", datasets_stub)

peft_stub = types.ModuleType("peft")
peft_stub.LoraConfig = lambda *args, **kwargs: SimpleNamespace()
peft_stub.get_peft_model = lambda model, config: model
peft_stub.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
sys.modules.setdefault("peft", peft_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = SimpleNamespace()
transformers_stub.AutoModelForCausalLM = SimpleNamespace()
transformers_stub.AutoModelForSequenceClassification = SimpleNamespace()
transformers_stub.PreTrainedModel = torch.nn.Module
transformers_stub.TrainingArguments = lambda **kwargs: kwargs
transformers_stub.Trainer = object
transformers_stub.TrainerCallback = object
transformers_stub.default_data_collator = lambda features: features
transformers_stub.get_scheduler = lambda *args, **kwargs: None
sys.modules.setdefault("transformers", transformers_stub)

from training import reward_trainer, rlhf_trainer, sft_trainer  # noqa: E402


class WordTokenizer:
    eos_token = "</s>"
    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(
        self,
        texts,
        truncation=False,
        max_length=None,
        padding=False,
        return_tensors=None,
    ):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        all_ids = []
        all_masks = []

        for text in items:
            tokens = str(text).replace("\n", " ").split()
            ids = [index + 1 for index, _ in enumerate(tokens)]
            if truncation and max_length is not None:
                ids = ids[:max_length]

            attention_mask = [1] * len(ids)
            if padding == "max_length":
                pad_to = max_length or len(ids)
                pad_count = max(0, pad_to - len(ids))
                ids = ids + [self.pad_token_id] * pad_count
                attention_mask = attention_mask + [0] * pad_count

            all_ids.append(ids)
            all_masks.append(attention_mask)

        if single:
            return {"input_ids": all_ids[0], "attention_mask": all_masks[0]}
        return {"input_ids": all_ids, "attention_mask": all_masks}


def test_sft_batched_preprocessing_masks_prompt_and_padding_labels():
    tokenizer = WordTokenizer()
    preprocessor = sft_trainer.SFTDataPreprocessor(tokenizer, max_length=24)

    tokenized = preprocessor.tokenize_function(
        {
            "prompt": ["Explain tests"],
            "completion": ["They catch regressions"],
        }
    )

    labels = tokenized["labels"][0]
    attention_mask = tokenized["attention_mask"][0]
    prompt_length = len(
        tokenizer(
            preprocessor.format_prompt("Explain tests"),
            truncation=True,
            max_length=24,
            padding=False,
            return_tensors=None,
        )["input_ids"]
    )

    assert all(label == -100 for label in labels[:prompt_length])
    assert any(
        label != -100
        for label, mask in zip(labels[prompt_length:], attention_mask[prompt_length:])
        if mask == 1
    )
    assert all(label == -100 for label, mask in zip(labels, attention_mask) if mask == 0)


class FakeRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids=None, attention_mask=None):
        return SimpleNamespace(logits=input_ids[:, :1].float())


def test_reward_eval_metrics_report_accuracy_and_margin(tmp_path):
    config = reward_trainer.RewardModelConfig(
        model_name="base",
        train_file="train.json",
        val_file="val.json",
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
    )
    trainer = reward_trainer.RewardModelTrainer(config)
    dataset = [
        {
            "chosen_input_ids": [3, 0],
            "chosen_attention_mask": [1, 0],
            "rejected_input_ids": [1, 0],
            "rejected_attention_mask": [1, 0],
        },
        {
            "chosen_input_ids": [1, 0],
            "chosen_attention_mask": [1, 0],
            "rejected_input_ids": [2, 0],
            "rejected_attention_mask": [1, 0],
        },
    ]

    metrics = trainer.evaluate_reward_model(FakeRewardModel(), dataset)

    assert metrics["reward_accuracy"] == 0.5
    assert metrics["reward_margin"] == 0.5
    assert metrics["reward_eval_examples"] == 2.0


class CapturingSequenceRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_ids = None
        self.last_attention_mask = None

    def forward(self, input_ids=None, attention_mask=None):
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        return SimpleNamespace(logits=attention_mask.sum(dim=1, keepdim=True).float())


def test_rlhf_compute_reward_uses_sequence_classifier_logits_and_attention_mask():
    trainer = rlhf_trainer.RLHFTrainer.__new__(rlhf_trainer.RLHFTrainer)
    reward_model = CapturingSequenceRewardModel()
    trainer.reward_model = reward_model
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)

    prompt_ids = torch.tensor([[5, 6, 0]])
    response_ids = torch.tensor([[7, 0]])
    prompt_mask = torch.tensor([[1, 1, 0]])
    response_mask = torch.tensor([[1, 0]])

    rewards = trainer.compute_reward(
        prompt_ids,
        response_ids,
        prompt_attention_mask=prompt_mask,
        response_attention_mask=response_mask,
    )

    assert rewards.tolist() == [3.0]
    assert reward_model.last_input_ids.tolist() == [[5, 6, 0, 7, 0]]
    assert reward_model.last_attention_mask.tolist() == [[1, 1, 0, 1, 0]]

