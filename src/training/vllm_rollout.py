"""
PPO rollouts with a vLLM engine colocated with the trainer (generation.engine: vllm).

policy.generate() spends most of a PPO step decoding token by token. Each rank
instead keeps a vLLM engine on its own GPU, loaded with the policy's base (the
SFT checkpoint). Before every rollout the trainer writes base + LoRA delta into
the engine's copies of the LoRA-targeted layers, so sampling always uses the
current policy at the base model's speed. (Serving the adapter through vLLM's
LoRA path instead was ~9x slower for a 1B Llama on an A100: 18 s vs 1.9 s for
16 x 512 tokens.) Nothing else in the base changes, which is why this needs a
LoRA policy. Between rollouts the engine sleeps (weights to CPU, KV cache
freed), so the PPO update gets the GPU memory back.

Only sampling moves to vLLM. Old log-probs, values, the reference KL and the
reward still come from the trainer's own forward passes on the sampled tokens.

The engine always runs in this process (torchrun: the external launcher;
otherwise V1 multiprocessing off), so the weight sync is a GPU copy through
LLM.apply_model rather than a serialized transfer. vLLM is only installed on
GPU machines (requirements.txt pins it; the CPU test environment does not), so
it is imported when the engine is built.
"""

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

ROLLOUT_ENGINES = ("hf", "vllm")
DEFAULT_GPU_MEMORY_UTILIZATION = 0.3


def rollout_engine(generation_config: Optional[Mapping[str, Any]]) -> str:
    engine = str((generation_config or {}).get("engine", "hf")).strip().lower()
    if engine not in ROLLOUT_ENGINES:
        raise ValueError(f"Unsupported generation.engine {engine!r}; expected one of {', '.join(ROLLOUT_ENGINES)}")
    return engine


def unpadded_prompts(prompt_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[int]]:
    """Token ids of each left-padded prompt without its padding."""
    return [row[mask.bool()].tolist() for row, mask in zip(prompt_ids.cpu(), attention_mask.cpu())]


def responses_to_tensor(
    responses: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Right-pad sampled responses into one [batch, longest] tensor, the layout
    policy.generate() returns after the prompt. A response that stopped keeps
    its stop token, so response_mask_from_stops covers it.
    """
    length = max(1, max((len(ids) for ids in responses), default=0))
    padded = [list(ids) + [pad_token_id] * (length - len(ids)) for ids in responses]
    return torch.tensor(padded, dtype=torch.long, device=device)


@torch.no_grad()
def merged_lora_weights(policy_model: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    (Hugging Face name, base + LoRA delta) for every LoRA-wrapped layer of a PEFT model.

    The trainer's weights stay untouched: each merged tensor is a new one,
    built and handed over one layer at a time.
    """
    from peft.tuners.lora import LoraLayer

    base_model = policy_model.get_base_model()
    for name, module in base_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        weight = module.get_base_layer().weight
        delta = sum(module.get_delta_weight(adapter) for adapter in module.active_adapters)
        yield f"{name}.weight", weight + delta.to(weight.dtype)


class VLLMRollout:
    """One vLLM engine per rank that samples from the base model with the current LoRA merged in."""

    def __init__(
        self,
        model_path: str,
        max_prompt_length: int,
        max_response_length: int,
        stop_token_ids: List[int],
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        settings: Optional[Mapping[str, Any]] = None,
        external_launcher: bool = False,
        dtype: str = "bfloat16",
    ):
        settings = dict(settings or {})
        if not external_launcher:
            # In-process engine, so apply_model can copy GPU tensors directly.
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams

        self.sleep = bool(settings.get("sleep_between_rollouts", True))
        engine_kwargs: Dict[str, Any] = dict(
            model=model_path,
            tokenizer=model_path,
            dtype=dtype,
            seed=seed,
            max_model_len=max_prompt_length + max_response_length,
            gpu_memory_utilization=float(settings.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)),
            enable_sleep_mode=self.sleep,
            # Cached prefixes would carry KV computed with the previous weights.
            enable_prefix_caching=False,
        )
        if external_launcher:
            # Under torchrun, each rank runs its engine in-process on its own
            # GPU with torchrun's rendezvous. The default executor spawns an
            # engine process that sets up its own NCCL group, which hangs
            # there (seen with nproc_per_node=1 on Colab).
            engine_kwargs["distributed_executor_backend"] = "external_launcher"
        logger.info("Starting the vLLM rollout engine: %s", engine_kwargs)
        self.llm = LLM(**engine_kwargs)
        self.sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_response_length,
            stop_token_ids=list(stop_token_ids),
            detokenize=False,
        )
        if self.sleep:
            self.llm.sleep(level=1)

    def _load_policy_weights(self, policy_model: Any) -> None:
        def load(model):
            model.load_weights(merged_lora_weights(policy_model))

        self.llm.apply_model(load)

    def generate(self, policy_model: Any, prompts: List[List[int]]) -> List[List[int]]:
        """Sample one response per prompt (token ids, stop token included) from the current policy."""
        start = time.perf_counter()
        if self.sleep:
            # vLLM wakes up through its own allocator, which cannot use the
            # blocks torch's caching allocator still holds from the update.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Weights first, synced, then the KV cache: the merged tensors are
            # built while the cache is still free.
            self.llm.wake_up(tags=["weights"])
        self._load_policy_weights(policy_model)
        if self.sleep:
            self.llm.wake_up(tags=["kv_cache"])
        synced = time.perf_counter()
        try:
            outputs = self.llm.generate(
                [{"prompt_token_ids": ids} for ids in prompts],
                self.sampling_params,
                use_tqdm=False,
            )
            generated = time.perf_counter()
        finally:
            if self.sleep:
                self.llm.sleep(level=1)
        # The fixed cost around sampling decides whether vLLM beats
        # policy.generate() on short rollouts.
        logger.info(
            "vLLM rollout: wake+sync=%.2fs generate=%.2fs sleep=%.2fs",
            synced - start,
            generated - synced,
            time.perf_counter() - generated,
        )
        return [list(output.outputs[0].token_ids) for output in outputs]


def launched_by_torchrun() -> bool:
    return "TORCHELASTIC_RUN_ID" in os.environ


def build_vllm_rollout(
    config: Any,
    raw_config: Mapping[str, Any],
    stop_token_ids: List[int],
    rank: int,
    distributed: bool,
    dtype: torch.dtype,
) -> VLLMRollout:
    """The trainer's VLLMRollout, from the rlhf YAML (generation.vllm)."""
    if (raw_config.get("lora") or {}).get("modules_to_save"):
        raise ValueError(
            "generation.engine vllm syncs only the LoRA-wrapped layers into the engine; "
            "lora.modules_to_save would train weights it never sees"
        )
    generation = raw_config.get("generation") or {}
    return VLLMRollout(
        model_path=config.policy_model_path,
        max_prompt_length=config.max_prompt_length,
        max_response_length=config.max_response_length,
        stop_token_ids=stop_token_ids,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        # Ranks sample different responses, as with policy.generate().
        seed=config.seed + rank,
        settings=generation.get("vllm"),
        external_launcher=distributed or launched_by_torchrun(),
        dtype={torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(dtype, "float32"),
    )
