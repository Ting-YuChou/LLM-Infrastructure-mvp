import sys
import types
from pathlib import Path
from types import SimpleNamespace

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

from training import reward_trainer, rlhf_trainer, sft_trainer  # noqa: E402


class FakeLifecycle:
    enabled = True

    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lifecycle_enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("lifecycle_exit")
        return False

    def log_metrics(self, metrics, step=None, prefix=None):
        self.events.append(("log_metrics", dict(metrics)))

    def register_checkpoint(self, checkpoint_path, final_metrics=None, extra_tags=None):
        self.events.append(("register", checkpoint_path, dict(final_metrics or {})))
        assert Path(checkpoint_path).exists()


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def __init__(self, events):
        self.events = events

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.events.append("tokenizer_save")


class FakeHFTrainer:
    def __init__(self, *args, events=None, **kwargs):
        self.events = events

    def train(self):
        self.events.append("train")
        return SimpleNamespace(metrics={"train_loss": 0.1})

    def evaluate(self):
        self.events.append("evaluate")
        return {"eval_loss": 0.2}

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.events.append("save_model")


def test_sft_registers_only_after_final_checkpoint_save(monkeypatch, tmp_path):
    events = []
    config = sft_trainer.SFTConfig(
        model_name="base",
        train_file="train.json",
        val_file=None,
        output_dir=str(tmp_path),
        raw_config={"tracking": {"mlflow_enabled": True}, "registry": {"model_name": "sft"}},
    )
    trainer = sft_trainer.SFTTrainer(config)

    monkeypatch.setattr(trainer, "_build_lifecycle", lambda: FakeLifecycle(events))
    monkeypatch.setattr(
        trainer,
        "load_model_and_tokenizer",
        lambda: (object(), FakeTokenizer(events)),
    )
    monkeypatch.setattr(
        sft_trainer.SFTDataPreprocessor,
        "prepare_datasets",
        lambda self, train_file, val_file: ("train", None),
    )
    monkeypatch.setattr(sft_trainer, "TrainingArguments", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        sft_trainer,
        "DataCollatorForLanguageModeling",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        sft_trainer,
        "Trainer",
        lambda *args, **kwargs: FakeHFTrainer(*args, events=events, **kwargs),
    )

    trainer.train()

    assert events.index("save_model") < _event_index(events, "register")
    assert events.index("tokenizer_save") < _event_index(events, "register")


def test_reward_registers_only_after_final_checkpoint_save(monkeypatch, tmp_path):
    events = []
    config = reward_trainer.RewardModelConfig(
        model_name="base",
        train_file="train.json",
        val_file=None,
        output_dir=str(tmp_path),
        raw_config={"tracking": {"mlflow_enabled": True}, "registry": {"model_name": "reward"}},
    )
    trainer = reward_trainer.RewardModelTrainer(config)

    monkeypatch.setattr(trainer, "_build_lifecycle", lambda: FakeLifecycle(events))
    monkeypatch.setattr(
        trainer,
        "load_model_and_tokenizer",
        lambda: (object(), FakeTokenizer(events)),
    )
    monkeypatch.setattr(
        reward_trainer.RewardDataPreprocessor,
        "prepare_datasets",
        lambda self, train_file, val_file: ("train", None),
    )
    monkeypatch.setattr(reward_trainer, "TrainingArguments", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        reward_trainer,
        "RewardTrainer",
        lambda *args, **kwargs: FakeHFTrainer(*args, events=events, **kwargs),
    )

    trainer.train()

    assert events.index("save_model") < _event_index(events, "register")
    assert events.index("tokenizer_save") < _event_index(events, "register")


def test_rlhf_registers_only_after_final_checkpoint_save(monkeypatch, tmp_path):
    events = []
    config = rlhf_trainer.RLHFConfig(
        policy_model_path="policy",
        reward_model_path="reward",
        output_dir=str(tmp_path),
        num_epochs=0,
        num_steps=0,
        raw_config={"tracking": {"mlflow_enabled": True}, "registry": {"model_name": "rlhf"}},
    )
    trainer = rlhf_trainer.RLHFTrainer.__new__(rlhf_trainer.RLHFTrainer)
    trainer.config = config
    trainer.kl_coef = 0.05
    trainer.tokenizer = object()

    def fake_save_checkpoint(step):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        events.append("save_checkpoint")
        return str(path)

    monkeypatch.setattr(trainer, "_build_lifecycle", lambda: FakeLifecycle(events))
    monkeypatch.setattr(trainer, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(rlhf_trainer, "PromptDataset", lambda *args, **kwargs: [1])

    trainer.train("prompts.json")

    assert events.index("save_checkpoint") < _event_index(events, "register")


def _event_index(events, event_name):
    for index, event in enumerate(events):
        if isinstance(event, tuple) and event[0] == event_name:
            return index
        if event == event_name:
            return index
    raise AssertionError(f"event not found: {event_name}")
