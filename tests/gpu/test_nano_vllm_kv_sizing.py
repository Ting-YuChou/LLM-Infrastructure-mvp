"""num_kvcache_blocks: 0 sizes the paged KV cache from free GPU memory."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.engine import KV_RESERVE_BYTES, NanoLlamaEngine  # noqa: E402
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402
from test_nano_vllm_forward_gpu import CONFIG, _model  # noqa: E402


def _engine(utilization):
    engine = NanoLlamaEngine(
        NanoVLLMConfig(
            num_kvcache_blocks=0,
            gpu_memory_utilization=utilization,
            max_model_len=128,
            max_num_batched_tokens=128,
            attention_backend="triton",
        )
    )
    engine.model = _model("cuda", torch.bfloat16, use_triton=True, attention_backend="triton")
    return engine


def test_kv_cache_takes_the_budget_left_after_weights_and_activations():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    # About 1 GiB of KV budget on top of whatever is already in use.
    utilization = min(1.0, (total - free + KV_RESERVE_BYTES + 1024**3) / total)
    engine = _engine(utilization)
    num_blocks = engine._size_kv_cache()

    per_block = PagedKVCache.bytes_per_block(
        CONFIG.num_hidden_layers, 16, CONFIG.num_key_value_heads, CONFIG.head_dim, torch.bfloat16
    )
    assert 0.8 * 1024**3 < num_blocks * per_block <= 1024**3

    engine._initialize_batching_state()  # allocates the sized cache
    assert engine.block_manager.num_blocks == num_blocks
    _, total = torch.cuda.mem_get_info()
    assert torch.cuda.memory_allocated() <= utilization * total


def test_kv_sizing_refuses_a_budget_below_one_sequence():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    engine = _engine(max(0.01, (total - free) / total - 0.01))
    with pytest.raises(RuntimeError, match="KV blocks fit"):
        engine._size_kv_cache()
