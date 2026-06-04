import asyncio
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.engine import NanoLlamaEngine  # noqa: E402
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, prompt, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        base = 10 if "alpha" in prompt else 20
        return {"input_ids": torch.tensor([[base, base + 1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class FakePagedModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def new_paged_cache(self, num_blocks, block_size):
        return PagedKVCache(
            num_layers=1,
            num_blocks=num_blocks,
            block_size=block_size,
            num_key_value_heads=1,
            head_dim=1,
            dtype=torch.float32,
            device="cpu",
        )

    def forward_paged(
        self,
        input_ids,
        positions,
        cache,
        slot_mapping,
        block_tables,
        context_lens,
        query_lens,
        is_prefill,
    ):
        del positions, cache, slot_mapping, block_tables, context_lens
        self.calls.append(
            {
                "tokens": int(input_ids.numel()),
                "query_lens": list(query_lens),
                "is_prefill": is_prefill,
            }
        )
        logits = torch.zeros(input_ids.numel(), 128)
        logits[:, 2] = 10.0
        return logits


async def _run_two_requests():
    engine = NanoLlamaEngine(
        NanoVLLMConfig(
            max_new_tokens=1,
            kvcache_block_size=4,
            num_kvcache_blocks=8,
            max_num_seqs=4,
            max_num_batched_tokens=16,
            enable_prefix_cache=False,
            attention_backend="torch",
            eager_load=False,
        )
    )
    engine.tokenizer = FakeTokenizer()
    engine.model = FakePagedModel()
    engine._initialize_batching_state()
    engine.initialized = True
    try:
        results = await asyncio.gather(
            engine.generate_async("alpha prompt", SamplingOptions(max_tokens=1)),
            engine.generate_async("beta prompt", SamplingOptions(max_tokens=1)),
        )
    finally:
        await engine.shutdown()
    return engine.model.calls, results


def test_engine_batches_concurrent_prefill_requests():
    calls, results = asyncio.run(_run_two_requests())

    assert [result.token_ids for result in results] == [[2], [2]]
    assert calls[0]["is_prefill"] is True
    assert calls[0]["query_lens"] == [2, 2]
    assert calls[0]["tokens"] == 4
