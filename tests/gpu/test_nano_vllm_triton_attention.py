"""Triton paged decode attention and fused RoPE + KV write vs the references."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels.paged_attention import (  # noqa: E402
    is_triton_available,
    paged_decode_attention,
    paged_decode_attention_reference,
    paged_prefill_attention,
    paged_prefill_attention_reference,
)
from serving.nano_vllm.kernels.rope_cache import (  # noqa: E402
    rope_and_cache_write,
    rope_and_cache_write_reference,
)
from serving.nano_vllm.kernels.rotary import compute_inv_freq, rope_cos_sin  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="requires CUDA and Triton",
)

LLAMA_8B = (32, 8, 128)  # q heads, kv heads, head_dim
TINY = (4, 2, 16)  # the tiny HF model in the CPU tests
CONTEXTS = [1, 15, 16, 17, 63, 64, 65, 1000, 4097]


def _cache(num_blocks, page_size, kv_heads, head_dim, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (num_blocks + 1, page_size, kv_heads, head_dim)  # + null block
    keys = torch.randn(shape, generator=generator, device="cuda").to(dtype)
    values = torch.randn(shape, generator=generator, device="cuda").to(dtype)
    return keys, values


def _shuffled_tables(context_lens, page_size, num_blocks, seed=0):
    order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(seed)).tolist()
    tables = []
    for context_len in context_lens:
        need = max(1, (context_len + page_size - 1) // page_size)
        tables.append([order.pop() for _ in range(need)])
    width = max(len(table) for table in tables)
    null = num_blocks
    padded = [table + [null] * (width - len(table)) for table in tables]
    return torch.tensor(padded, dtype=torch.int32, device="cuda")


def _decode_case(batch, page_size, heads, dtype, seed=0):
    q_heads, kv_heads, head_dim = heads
    generator = torch.Generator().manual_seed(seed)
    context_lens = [CONTEXTS[i] for i in torch.randint(len(CONTEXTS), (batch,), generator=generator)]
    num_blocks = sum((c + page_size - 1) // page_size for c in context_lens) + 4
    keys, values = _cache(num_blocks, page_size, kv_heads, head_dim, dtype, seed)
    tables = _shuffled_tables(context_lens, page_size, num_blocks, seed)
    query = torch.randn(batch, q_heads, head_dim, device="cuda").to(dtype)
    lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    return query, keys, values, tables, lens


@pytest.mark.parametrize("heads", [LLAMA_8B, TINY], ids=["llama8b", "tiny"])
@pytest.mark.parametrize("num_splits", [1, 4, 32, None], ids=["s1", "s4", "s32", "auto"])
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
@pytest.mark.parametrize("page_size", [4, 16])
def test_decode_kernel_matches_reference(page_size, batch, num_splits, heads):
    query, keys, values, tables, lens = _decode_case(batch, page_size, heads, torch.bfloat16)
    scale = 1.0 / math.sqrt(heads[2])

    out = paged_decode_attention(
        query, keys, values, tables, lens, scale, use_triton=True, num_splits=num_splits
    )
    expected = paged_decode_attention_reference(query, keys, values, tables, lens, scale)

    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_decode_kernel_fp32_is_exact_to_tolerance():
    query, keys, values, tables, lens = _decode_case(8, 16, LLAMA_8B, torch.float32)
    scale = 1.0 / math.sqrt(128)

    out = paged_decode_attention(query, keys, values, tables, lens, scale, use_triton=True)
    expected = paged_decode_attention_reference(query, keys, values, tables, lens, scale)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_splits", [1, 8])
def test_decode_kernel_returns_zero_for_an_empty_context(num_splits):
    query, keys, values, tables, lens = _decode_case(3, 16, LLAMA_8B, torch.bfloat16)
    lens[1] = 0
    keys[tables.long().flatten()] = float("nan")  # must never be read for row 1
    keys[tables[0].long()] = 0.0
    keys[tables[2].long()] = 0.0

    out = paged_decode_attention(
        query, keys, values, tables, lens, 0.1, use_triton=True, num_splits=num_splits
    )

    assert torch.isfinite(out).all()
    assert (out[1] == 0).all()


# (query_len, context_len): whole prompts, chunks after a cached prefix,
# single tokens, a tile-sized and a long prompt.
PREFILL_SHAPES = [(37, 37), (5, 70), (1, 1), (1, 33), (64, 64), (65, 129), (300, 1000), (17, 17)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("heads", [LLAMA_8B, TINY], ids=["llama8b", "tiny"])
@pytest.mark.parametrize("page_size", [4, 16])
def test_prefill_kernel_matches_reference(page_size, heads, dtype):
    q_heads, kv_heads, head_dim = heads
    query_lens = [q for q, _ in PREFILL_SHAPES]
    context_lens = [c for _, c in PREFILL_SHAPES]
    num_blocks = sum((c + page_size - 1) // page_size for c in context_lens) + 4
    keys, values = _cache(num_blocks, page_size, kv_heads, head_dim, dtype)
    tables = _shuffled_tables(context_lens, page_size, num_blocks)
    query = torch.randn(sum(query_lens), q_heads, head_dim, device="cuda").to(dtype)
    cu_seqlens = torch.tensor([0] + query_lens, device="cuda").cumsum(0).to(torch.int32)
    lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(head_dim)

    out = paged_prefill_attention(
        query, keys, values, tables, cu_seqlens, lens, query_lens, context_lens, scale,
        use_triton=True,
    )
    expected = paged_prefill_attention_reference(
        query, keys, values, tables, query_lens, context_lens, scale
    )

    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out.float(), expected.float(), **tol)


@pytest.mark.parametrize("heads", [LLAMA_8B, TINY], ids=["llama8b", "tiny"])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_rope_cache_write_matches_reference(num_tokens, heads):
    q_heads, kv_heads, head_dim = heads
    dtype = torch.bfloat16
    scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    inv_freq = compute_inv_freq(head_dim, 500000.0, scaling, device="cuda")
    cos_table, sin_table = rope_cos_sin(inv_freq, 8192, dtype)
    positions = torch.randint(0, 8192, (num_tokens,), device="cuda")
    cos, sin = cos_table[positions], sin_table[positions]
    qkv = torch.randn(num_tokens, (q_heads + 2 * kv_heads) * head_dim, device="cuda").to(dtype)
    num_blocks, page_size = 32, 16
    # Distinct slots; the last token goes to the null block like a padded row.
    slots = torch.randperm(num_blocks * page_size, device="cuda")[:num_tokens]
    slots[-1] = num_blocks * page_size

    caches = []
    for fn in (rope_and_cache_write, rope_and_cache_write_reference):
        keys, values = _cache(num_blocks, page_size, kv_heads, head_dim, dtype, seed=3)
        kwargs = {"use_triton": True} if fn is rope_and_cache_write else {}
        query = fn(qkv, cos, sin, keys, values, slots, q_heads, kv_heads, **kwargs)
        caches.append((query, keys, values))

    (q_fast, k_fast, v_fast), (q_ref, k_ref, v_ref) = caches
    # fp32 math in the kernel vs bf16 steps in the reference: within 1-2 ulp.
    torch.testing.assert_close(q_fast.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_fast.float(), k_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v_fast, v_ref, rtol=0, atol=0)
