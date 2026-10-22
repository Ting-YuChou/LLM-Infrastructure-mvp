"""
The colocated vLLM rollout engine (training/vllm_rollout.py) against a fake vllm package.

vLLM only installs on GPU machines, so these tests check the contract around
it: prompt/response layout, the weight sync before every rollout, and sleep
around generation. test_training_real_smoke.py runs PPO end to end through it
with a real PEFT policy.
"""

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training import vllm_rollout  # noqa: E402


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeEngineModel:
    """Stands in for the vLLM model: records what load_weights received."""

    def __init__(self):
        self.loaded = {}

    def load_weights(self, weights):
        for name, tensor in weights:
            self.loaded[name] = tensor.detach().clone()


class FakeLLM:
    """Records calls; each prompt gets responses[i % len(responses)]."""

    instances = []
    responses = [[7, 8]]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.model = FakeEngineModel()
        FakeLLM.instances.append(self)

    def sleep(self, level=1):
        self.calls.append(("sleep", level))

    def wake_up(self, tags=None):
        self.calls.append(("wake_up", tuple(tags or ())))

    def apply_model(self, func):
        self.calls.append(("apply_model",))
        return [func(self.model)]

    def generate(self, prompts, sampling_params, use_tqdm=True):
        self.calls.append(("generate", [p["prompt_token_ids"] for p in prompts]))
        return [
            types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=tuple(self.responses[i % len(self.responses)]))])
            for i in range(len(prompts))
        ]


def install_fake_vllm(monkeypatch):
    """Put a fake vllm package (FakeLLM) on sys.modules; also used by test_training_real_smoke.py."""
    FakeLLM.instances = []
    FakeLLM.responses = [[7, 8]]
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = FakeLLM, FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    return FakeLLM


@pytest.fixture
def fake_vllm(monkeypatch):
    return install_fake_vllm(monkeypatch)


def _rollout(**overrides):
    kwargs = dict(
        model_path="sft-final",
        max_prompt_length=512,
        max_response_length=64,
        stop_token_ids=[9, 10],
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        seed=3,
        settings={"gpu_memory_utilization": 0.25},
        external_launcher=True,
    )
    kwargs.update(overrides)
    return vllm_rollout.VLLMRollout(**kwargs)


def test_prompts_lose_left_padding_and_responses_are_right_padded():
    prompt_ids = torch.tensor([[0, 0, 5, 6], [3, 4, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert vllm_rollout.unpadded_prompts(prompt_ids, mask) == [[5, 6], [3, 4, 5, 6]]

    ids = vllm_rollout.responses_to_tensor([[7, 9], [7, 8, 8]], pad_token_id=0, device=torch.device("cpu"))
    assert ids.tolist() == [[7, 9, 0], [7, 8, 8]]
    # Every response empty still yields one column for the mask/concat code.
    assert vllm_rollout.responses_to_tensor([[], []], 0, torch.device("cpu")).shape == (2, 1)


def test_engine_name_is_validated():
    assert vllm_rollout.rollout_engine(None) == "hf"
    assert vllm_rollout.rollout_engine({"engine": "VLLM"}) == "vllm"
    with pytest.raises(ValueError, match="generation.engine"):
        vllm_rollout.rollout_engine({"engine": "sglang"})


def test_rollout_syncs_weights_between_waking_weights_and_kv_cache(fake_vllm, monkeypatch):
    weight = torch.ones(2, 2)
    monkeypatch.setattr(vllm_rollout, "merged_lora_weights", lambda policy: iter([("model.w.weight", weight)]))
    rollout = _rollout()
    llm = fake_vllm.instances[0]
    assert llm.kwargs["model"] == "sft-final"
    assert "enable_lora" not in llm.kwargs  # the LoRA path was ~9x slower
    assert not llm.kwargs["enable_prefix_caching"]
    assert llm.kwargs["max_model_len"] == 576
    assert llm.kwargs["gpu_memory_utilization"] == 0.25
    assert llm.kwargs["distributed_executor_backend"] == "external_launcher"
    assert llm.kwargs["seed"] == 3
    params = rollout.sampling_params
    assert (params.temperature, params.top_p, params.top_k, params.max_tokens) == (0.7, 0.9, 50, 64)
    assert params.stop_token_ids == [9, 10]

    fake_vllm.responses = [[7, 9], [8, 8, 8]]
    responses = rollout.generate(object(), [[1, 2], [3]])

    assert responses == [[7, 9], [8, 8, 8]]
    assert llm.calls == [
        ("sleep", 1),
        ("wake_up", ("weights",)),
        ("apply_model",),
        ("wake_up", ("kv_cache",)),
        ("generate", [[1, 2], [3]]),
        ("sleep", 1),
    ]
    assert torch.equal(llm.model.loaded["model.w.weight"], weight)


def test_single_process_engine_runs_in_process_without_sleep(fake_vllm, monkeypatch):
    monkeypatch.setattr(vllm_rollout, "merged_lora_weights", lambda policy: iter([]))
    rollout = _rollout(external_launcher=False, settings={"sleep_between_rollouts": False})
    llm = fake_vllm.instances[0]
    assert "distributed_executor_backend" not in llm.kwargs
    assert vllm_rollout.os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert not llm.kwargs["enable_sleep_mode"]
    rollout.generate(object(), [[1]])
    assert [call[0] for call in llm.calls] == ["apply_model", "generate"]


def test_torchrun_always_gets_the_external_launcher(fake_vllm, tmp_path, monkeypatch):
    config = types.SimpleNamespace(
        policy_model_path="m", output_dir=str(tmp_path), max_prompt_length=16, max_response_length=8,
        temperature=1.0, top_p=1.0, top_k=-1, seed=0,
    )
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    vllm_rollout.build_vllm_rollout(config, {}, [9], rank=0, distributed=False, dtype=torch.bfloat16)
    assert "distributed_executor_backend" not in fake_vllm.instances[-1].kwargs

    # torchrun --nproc_per_node 1: one rank, but still torchrun's environment.
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")
    vllm_rollout.build_vllm_rollout(config, {}, [9], rank=0, distributed=False, dtype=torch.bfloat16)
    assert fake_vllm.instances[-1].kwargs["distributed_executor_backend"] == "external_launcher"


def test_modules_to_save_are_rejected(fake_vllm):
    config = types.SimpleNamespace(policy_model_path="m")
    with pytest.raises(ValueError, match="modules_to_save"):
        vllm_rollout.build_vllm_rollout(
            config, {"lora": {"modules_to_save": ["lm_head"]}}, [9], 0, False, torch.bfloat16
        )
