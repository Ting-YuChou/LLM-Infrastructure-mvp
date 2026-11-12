"""The Triton block attention kernel and the drafter's fused norm + RoPE +
KV write vs their references, and the paths that use them: a verify step of
the target and a draft of the DFlash drafter."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "unit"))

from serving.nano_vllm.dflash import DFlashDrafter  # noqa: E402
from serving.nano_vllm.kernels.block_attention import (  # noqa: E402
    BLOCK_CONFIGS,
    paged_block_attention,
    paged_block_attention_reference,
)
from serving.nano_vllm.kernels.fp8 import fp8_dequantize_reference, fp8_quantize_reference  # noqa: E402
from serving.nano_vllm.kernels.paged_attention import (  # noqa: E402
    is_triton_available,
    paged_decode_attention,
    paged_prefill_attention,
)
from serving.nano_vllm.kernels.rope_cache import (  # noqa: E402
    norm_rope_and_cache_write,
    norm_rope_and_cache_write_reference,
)
from serving.nano_vllm.kernels.rotary import compute_inv_freq, rope_cos_sin  # noqa: E402
from serving.nano_vllm.metadata import build_batch  # noqa: E402
from test_nano_vllm_dflash import TINY_CONFIG, _drafter_state, _save_checkpoint  # noqa: E402
from test_nano_vllm_forward_gpu import BLOCK_SIZE, CONFIG, PROMPTS, _model, _rows  # noqa: E402
from test_nano_vllm_triton_attention import LLAMA_8B, TINY, _cache, _shuffled_tables  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="requires CUDA and Triton",
)

# Context lengths before the block: none, inside and across pages and tiles.
PREFIXES = [0, 1, 15, 16, 17, 63, 64, 65, 120, 1000, 4090]


def _block_case(batch, q_max, page_size, heads, dtype, varlen, seed=0):
    """Block queries [batch * q_max], a cache, shuffled tables, context_lens
    (the block's tokens included) and q_lens (None when not ``varlen``; else
    0..q_max, with NaN in the padding query rows)."""
    q_heads, kv_heads, head_dim = heads
    generator = torch.Generator().manual_seed(seed)
    prefixes = [PREFIXES[i] for i in torch.randint(len(PREFIXES), (batch,), generator=generator)]
    if varlen:
        q_lens = torch.randint(q_max + 1, (batch,), generator=generator).tolist()
        q_lens[0] = q_max
    else:
        q_lens = [q_max] * batch
    context_lens = [prefix + q_len for prefix, q_len in zip(prefixes, q_lens)]
    num_blocks = sum((c + page_size - 1) // page_size for c in context_lens) + batch + 4
    keys, values = _cache(num_blocks, page_size, kv_heads, head_dim, dtype, seed)
    tables = _shuffled_tables(context_lens, page_size, num_blocks, seed)
    query = torch.randn(batch * q_max, q_heads, head_dim, device="cuda").to(dtype)
    padding = torch.tensor([t >= q_len for q_len in q_lens for t in range(q_max)], device="cuda")
    query[padding] = float("nan")
    lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device="cuda") if varlen else None
    return query, keys, values, tables, lens, q_lens_t, padding


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("heads", [LLAMA_8B, TINY], ids=["llama8b", "tiny"])
@pytest.mark.parametrize("num_splits", [1, 4, 32, None], ids=["s1", "s4", "s32", "auto"])
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
@pytest.mark.parametrize("q_max", [10, 3])
def test_block_kernel_matches_reference(q_max, batch, num_splits, heads, causal):
    query, keys, values, tables, lens, q_lens, padding = _block_case(
        batch, q_max, 16, heads, torch.bfloat16, varlen=True
    )
    scale = 1.0 / math.sqrt(heads[2])

    out = paged_block_attention(
        query, keys, values, tables, lens, q_lens, q_max, scale, causal,
        use_triton=True, num_splits=num_splits,
    )
    expected = paged_block_attention_reference(
        query, keys, values, tables, lens, q_lens, q_max, scale, causal
    )

    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert (out[padding] == 0).all()


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("q_max", [10, 5, 1])
def test_block_kernel_fp32_is_exact_to_tolerance(q_max, page_size, causal):
    query, keys, values, tables, lens, q_lens, _ = _block_case(
        8, q_max, page_size, LLAMA_8B, torch.float32, varlen=True
    )
    scale = 1.0 / math.sqrt(128)

    out = paged_block_attention(
        query, keys, values, tables, lens, q_lens, q_max, scale, causal, use_triton=True
    )
    expected = paged_block_attention_reference(
        query, keys, values, tables, lens, q_lens, q_max, scale, causal
    )

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


# Every tier, and the settings tune.py sweeps: rows per program (row tiles
# re-reading the context), tile sizes, fewer splits.
LAUNCH_CONFIGS = [c for _, c in BLOCK_CONFIGS] + [
    {"block_n": n, "num_warps": w, "num_stages": st, "block_r": r}
    for r in (16, 32, 64) for n, w, st in ((64, 8, 1), (64, 4, 2), (32, 4, 2))
]


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("config", LAUNCH_CONFIGS, ids=str)
def test_every_launch_config_matches_reference(config, causal):
    query, keys, values, tables, lens, q_lens, _ = _block_case(
        5, 10, 16, LLAMA_8B, torch.float32, varlen=True, seed=2
    )
    scale = 1.0 / math.sqrt(128)

    out = paged_block_attention(
        query, keys, values, tables, lens, q_lens, 10, scale, causal, use_triton=True,
        num_splits=4, config=config,
    )
    expected = paged_block_attention_reference(
        query, keys, values, tables, lens, q_lens, 10, scale, causal
    )

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("q_max", [10, 3, 1])
@pytest.mark.parametrize("splits_shift", [0, 1, 5])
def test_row_tiles_and_fewer_splits_give_the_same_attention(splits_shift, q_max, causal):
    query, keys, values, tables, lens, q_lens, padding = _block_case(
        3, q_max, 16, LLAMA_8B, torch.float32, varlen=True, seed=5
    )
    scale = 1.0 / math.sqrt(128)
    launch = {"block_n": 64, "num_warps": 4, "num_stages": 2, "splits_shift": splits_shift}
    args = (query, keys, values, tables, lens, q_lens, q_max, scale, causal)

    whole = paged_block_attention(*args, use_triton=True, config=launch)
    for block_r in (16, 32):
        tiled = paged_block_attention(*args, use_triton=True, config={**launch, "block_r": block_r})
        torch.testing.assert_close(tiled, whole, rtol=1e-5, atol=1e-5)
        assert (tiled[padding] == 0).all()
    torch.testing.assert_close(
        whole, paged_block_attention_reference(*args), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
def test_full_blocks_without_q_lens_equal_full_blocks_with(causal):
    # What the engine runs: every sequence has q_max queries, no q_lens.
    query, keys, values, tables, lens, _, _ = _block_case(
        6, 10, 16, LLAMA_8B, torch.bfloat16, varlen=False
    )
    scale = 1.0 / math.sqrt(128)
    full = torch.full((6,), 10, dtype=torch.int32, device="cuda")

    out = paged_block_attention(
        query, keys, values, tables, lens, None, 10, scale, causal, use_triton=True
    )
    with_lens = paged_block_attention(
        query, keys, values, tables, lens, full, 10, scale, causal, use_triton=True
    )
    expected = paged_block_attention_reference(
        query, keys, values, tables, lens, None, 10, scale, causal
    )

    assert torch.equal(out, with_lens)
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_causal_block_kernel_matches_the_prefill_and_decode_kernels():
    # The verify rows' attention before this kernel (prefill), and row 0 of
    # a block of one (a decode step), in fp32.
    query, keys, values, tables, lens, _, _ = _block_case(
        5, 10, 16, LLAMA_8B, torch.float32, varlen=False, seed=3
    )
    scale = 1.0 / math.sqrt(128)
    cu_seqlens = torch.arange(0, 51, 10, dtype=torch.int32, device="cuda")

    out = paged_block_attention(
        query, keys, values, tables, lens, None, 10, scale, True, use_triton=True
    )
    prefill = paged_prefill_attention(
        query, keys, values, tables, cu_seqlens, lens, [10] * 5, lens.tolist(), scale,
        use_triton=True,
    )
    torch.testing.assert_close(out, prefill, rtol=1e-4, atol=1e-4)

    first = query[::10].contiguous()
    one = paged_block_attention(
        first, keys, values, tables, lens - 9, None, 1, scale, True, use_triton=True
    )
    decode = paged_decode_attention(first, keys, values, tables, lens - 9, scale, use_triton=True)
    torch.testing.assert_close(one, decode, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(one, out[::10], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("num_splits", [1, 8])
def test_nothing_outside_a_sequences_context_or_queries_leaks(num_splits, causal):
    query, keys, values, tables, lens, q_lens, padding = _block_case(
        6, 10, 16, LLAMA_8B, torch.bfloat16, varlen=True, seed=1
    )
    q_lens[2] = 0  # a sequence with no query
    padding[20:30] = True
    scale = 1.0 / math.sqrt(128)
    args = (tables, lens, q_lens, 10, scale, causal)
    expected = paged_block_attention(
        query, keys, values, *args, use_triton=True, num_splits=num_splits
    )

    # Slots past each context, unused blocks and the null block.
    used = torch.zeros(keys.shape[:2], dtype=torch.bool, device="cuda")
    for table, context_len in zip(tables.tolist(), lens.tolist()):
        for position in range(context_len):
            used[table[position // 16], position % 16] = True
    keys, values = keys.clone(), values.clone()
    keys[~used] = float("nan")
    values[~used] = float("nan")
    out = paged_block_attention(query, keys, values, *args, use_triton=True, num_splits=num_splits)

    assert torch.isfinite(out).all()
    assert torch.equal(out, expected)
    assert (out[padding] == 0).all() and (out[20:30] == 0).all()


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
@pytest.mark.parametrize("num_splits", [1, 4, None], ids=["s1", "s4", "auto"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_block_kernel_on_an_fp8_cache_matches_reference(dtype, num_splits, causal):
    query, keys, values, tables, lens, q_lens, _ = _block_case(
        7, 10, 16, LLAMA_8B, dtype, varlen=True, seed=4
    )
    keys, key_scales = fp8_quantize_reference(keys.float())
    values, value_scales = fp8_quantize_reference(values.float())
    scale = 1.0 / math.sqrt(128)
    scales = dict(key_scales=key_scales, value_scales=value_scales)

    out = paged_block_attention(
        query, keys, values, tables, lens, q_lens, 10, scale, causal, use_triton=True,
        num_splits=num_splits, **scales,
    )
    expected = paged_block_attention_reference(
        query, keys, values, tables, lens, q_lens, 10, scale, causal, **scales
    )
    # The same attention over the dequantized cache, in full precision.
    dense = paged_block_attention_reference(
        query, fp8_dequantize_reference(keys, key_scales),
        fp8_dequantize_reference(values, value_scales), tables, lens, q_lens, 10, scale, causal,
    )

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out.float(), expected.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(expected.float(), dense.float(), rtol=tol, atol=tol)


def test_block_kernel_writes_into_a_slice_of_the_output():
    query, keys, values, tables, lens, _, _ = _block_case(
        3, 10, 16, LLAMA_8B, torch.bfloat16, varlen=False
    )
    scale = 1.0 / math.sqrt(128)
    expected = paged_block_attention(
        query, keys, values, tables, lens, None, 10, scale, True, use_triton=True
    )

    out = torch.full((40, 32, 128), 7.0, dtype=torch.bfloat16, device="cuda")
    returned = paged_block_attention(
        query, keys, values, tables, lens, None, 10, scale, True, use_triton=True, out=out[:30]
    )

    assert torch.equal(out[:30], expected) and torch.equal(returned, expected)
    assert (out[30:] == 7.0).all()


# --- the drafter's q_norm/k_norm + RoPE + KV write --------------------------


def _norm_rope_inputs(heads, tokens, dtype, with_query, seed=0):
    q_heads, kv_heads, head_dim = heads
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator, device="cuda") * scale).to(dtype)

    cos_table, sin_table = rope_cos_sin(compute_inv_freq(head_dim, 10000.0, device="cuda"), 512, dtype)
    positions = torch.randint(0, 512, (tokens,), generator=generator, device="cuda")
    num_blocks, page = 2 * tokens // 16 + 4, 16
    slots = torch.randperm(num_blocks * page, generator=generator, device="cuda")[:tokens]
    slots[-1] = num_blocks * page  # the null block's first slot
    return dict(
        query=randn(tokens, q_heads * head_dim, scale=3.0) if with_query else None,
        kv=randn(tokens, 2 * kv_heads * head_dim, scale=3.0),
        q_norm=1 + randn(head_dim, scale=0.1),
        k_norm=1 + randn(head_dim, scale=0.1),
        eps=1e-6,
        cos=cos_table[positions],
        sin=sin_table[positions],
        slot_mapping=slots,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
    ), (num_blocks + 1, page, kv_heads, head_dim)


def _norm_rope(fn, inputs, cache_shape, dtype, fp8=False, **kwargs):
    """(rotated query or None, written K rows, written V rows[, scales])."""
    cache_dtype = torch.float8_e4m3fn if fp8 else dtype
    keys = torch.zeros(cache_shape, dtype=cache_dtype, device="cuda")
    values = torch.zeros_like(keys)
    scales = {}
    if fp8:
        scales = dict(
            key_scales=torch.zeros(cache_shape[:3], device="cuda"),
            value_scales=torch.zeros(cache_shape[:3], device="cuda"),
        )
    rotated = fn(key_cache=keys, value_cache=values, **inputs, **scales, **kwargs)
    slots = inputs["slot_mapping"]
    if fp8:
        keys = fp8_dequantize_reference(keys, scales["key_scales"])
        values = fp8_dequantize_reference(values, scales["value_scales"])
    flat = cache_shape[2] * cache_shape[3]
    return rotated, keys.reshape(-1, flat)[slots], values.reshape(-1, flat)[slots]


@pytest.mark.parametrize("with_query", [True, False], ids=["block", "context"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("heads", [LLAMA_8B, TINY], ids=["llama8b", "tiny"])
@pytest.mark.parametrize("tokens", [1, 10, 300])
def test_norm_rope_cache_write_matches_reference(tokens, heads, dtype, with_query):
    inputs, cache_shape = _norm_rope_inputs(heads, tokens, dtype, with_query)

    got = _norm_rope(norm_rope_and_cache_write, inputs, cache_shape, dtype, use_triton=True)
    expected = _norm_rope(norm_rope_and_cache_write_reference, inputs, cache_shape, dtype)

    assert (got[0] is None) == (not with_query)
    # bf16: the reference rotates op by op in bf16, the kernel in fp32.
    tol = 1e-5 if dtype == torch.float32 else 5e-2
    for mine, want in zip(got, expected):
        if want is not None:
            torch.testing.assert_close(mine.float(), want.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("with_query", [True, False], ids=["block", "context"])
def test_norm_rope_cache_write_into_an_fp8_cache_matches_reference(with_query):
    inputs, cache_shape = _norm_rope_inputs(LLAMA_8B, 40, torch.float32, with_query, seed=1)

    got = _norm_rope(norm_rope_and_cache_write, inputs, cache_shape, torch.float32, fp8=True,
                     use_triton=True)
    expected = _norm_rope(norm_rope_and_cache_write_reference, inputs, cache_shape, torch.float32,
                          fp8=True)

    if with_query:
        torch.testing.assert_close(got[0], expected[0], rtol=1e-5, atol=1e-5)
    # A rounding difference moves an e4m3 code by one step, 1/8 of the value
    # at most; nearly every value decodes to the same number.
    for mine, want in zip(got[1:], expected[1:]):
        error = (mine - want).abs()
        assert (error <= 0.126 * want.abs() + 1e-6).all()
        assert (error <= 1e-5 * want.abs() + 1e-7).float().mean() > 0.99


# --- a verify step of the target ---------------------------------------------


def _verify_step(model, with_prefill_row):
    """Prefill two prompts, then verify a block of 5 tokens after each (with
    a third prompt prefilling next to them): the logits of every block token
    and of the prefill row's last."""
    cache = model.new_paged_cache(num_blocks=16, block_size=BLOCK_SIZE)
    first, second, third = PROMPTS[0], PROMPTS[1], list(range(200, 240))
    tables = [[1, 3, 5], [7, 9], [11, 13, 15]]
    ids, positions, metadata = build_batch(
        **_rows(cache, [(first, 0, tables[0]), (second, 0, tables[1])]),
        is_prefill=True, device=model.device,
    )
    model.forward_batch(ids, positions, cache, metadata)
    rows = [
        ([50, 51, 52, 53, 54], len(first), tables[0]),
        ([60, 61, 62, 63, 64], len(second), tables[1]),
    ]
    if with_prefill_row:
        rows.append((third, 0, tables[2]))
    ids, positions, metadata = build_batch(
        **_rows(cache, rows), is_prefill=True, device=model.device, block_rows=2,
    )
    assert (metadata.num_block_rows, metadata.block_len) == (2, 5)
    logits = model.forward_batch(ids, positions, cache, metadata)
    assert logits.shape[0] == 10 + with_prefill_row
    return logits.float().cpu()


@pytest.mark.parametrize("with_prefill_row", [False, True], ids=["verify", "verify+prefill"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_verify_step_triton_matches_the_reference(kv_cache_dtype, with_prefill_row):
    kwargs = dict(kv_cache_dtype=kv_cache_dtype)
    reference = _verify_step(_model("cuda", torch.float32, use_triton=False, **kwargs), with_prefill_row)
    triton_fp32 = _verify_step(
        _model("cuda", torch.float32, use_triton=False, attention_backend="triton", **kwargs),
        with_prefill_row,
    )
    triton_bf16 = _verify_step(
        _model("cuda", torch.bfloat16, use_triton=True, attention_backend="triton", **kwargs),
        with_prefill_row,
    )

    if kv_cache_dtype == "fp8":
        # As test_fp8_kv_cache_triton_matches_the_reference: the kernel's
        # division can round a code differently, and a bf16 model quantizes
        # bf16-rounded keys, so it is compared by direction.
        torch.testing.assert_close(triton_fp32, reference, rtol=2e-3, atol=2e-3)
        cosine = torch.nn.functional.cosine_similarity(triton_bf16, reference, dim=-1)
        assert cosine.min() > 0.998, cosine
    else:
        torch.testing.assert_close(triton_fp32, reference, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(triton_bf16, reference, rtol=0.05, atol=0.05)


def test_verify_rows_give_the_logits_of_decoding_the_same_tokens_one_by_one():
    model = _model("cuda", torch.float32, use_triton=False, attention_backend="triton")
    block = _verify_step(model, with_prefill_row=False)

    cache = model.new_paged_cache(num_blocks=16, block_size=BLOCK_SIZE)
    first, second = PROMPTS[0], PROMPTS[1]
    tables = [[1, 3, 5], [7, 9]]
    ids, positions, metadata = build_batch(
        **_rows(cache, [(first, 0, tables[0]), (second, 0, tables[1])]),
        is_prefill=True, device=model.device,
    )
    model.forward_batch(ids, positions, cache, metadata)
    steps = []
    for offset in range(5):
        ids, positions, metadata = build_batch(
            **_rows(cache, [
                ([50 + offset], len(first) + offset, tables[0]),
                ([60 + offset], len(second) + offset, tables[1]),
            ]),
            is_prefill=False, device=model.device,
        )
        steps.append(model.forward_batch(ids, positions, cache, metadata).float().cpu())
    one_by_one = torch.stack(steps, dim=1).reshape(10, -1)  # [seq, offset] -> block rows

    torch.testing.assert_close(block, one_by_one, rtol=1e-4, atol=1e-4)


# --- a draft of the DFlash drafter --------------------------------------------

DRAFTER_CONFIG = {
    **TINY_CONFIG,
    "hidden_size": CONFIG.hidden_size,
    "vocab_size": CONFIG.vocab_size,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_target_layers": CONFIG.num_hidden_layers,
    "dflash_config": {"mask_token_id": 3, "target_layer_ids": [0]},
}


def drafter_checkpoint(path: Path, seed: int = 0) -> Path:
    """A random drafter for the tiny GPU target (test_nano_vllm_forward_gpu)."""
    return _save_checkpoint(path, DRAFTER_CONFIG, _drafter_state(DRAFTER_CONFIG, seed=seed))


def _draft(drafter, context_lens, anchors):
    """write_context of random taps, then one draft: logits [B, K, vocab]."""
    device, page = drafter.device, 16
    q = drafter.config.block_size
    generator = torch.Generator().manual_seed(7)
    cache = drafter.new_paged_cache(64, page)
    order = torch.randperm(64, generator=generator).tolist()
    tables = [[order.pop() for _ in range((n + q + page - 1) // page)] for n in context_lens]
    width = 16
    block_tables = torch.tensor(
        [table + [cache.null_block] * (width - len(table)) for table in tables],
        dtype=torch.int32, device=device,
    )

    def slots(seq, positions):
        return [tables[seq][p // page] * page + p % page for p in positions]

    taps = torch.randn(sum(context_lens), drafter.config.hidden_size, generator=generator)
    context_positions = [p for n in context_lens for p in range(n)]
    context_slots = [s for seq, n in enumerate(context_lens) for s in slots(seq, range(n))]
    drafter.write_context(
        taps.to(device=device, dtype=drafter.dtype), torch.tensor(context_positions, device=device),
        cache, torch.tensor(context_slots, device=device),
    )
    positions = [n + t for n in context_lens for t in range(q)]
    block_slots = [s for seq, n in enumerate(context_lens) for s in slots(seq, range(n, n + q))]
    logits = drafter.draft(
        torch.tensor(anchors, device=device), torch.tensor(positions, device=device), cache,
        block_tables, torch.tensor(context_lens, dtype=torch.int32, device=device),
        torch.tensor(block_slots, device=device),
    )
    return logits.view(len(anchors), q - 1, -1).float().cpu()


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_drafter_triton_matches_the_reference(tmp_path, kv_cache_dtype):
    path = drafter_checkpoint(tmp_path / "drafter")
    context_lens, anchors = [1, 6, 37, 70], [7, 41, 12, 300]

    def run(dtype, use_triton, attention_backend):
        target = _model("cuda", dtype, use_triton=use_triton, attention_backend=attention_backend,
                        kv_cache_dtype=kv_cache_dtype)
        return _draft(DFlashDrafter.from_pretrained(path, target), context_lens, anchors)

    reference = run(torch.float32, False, "torch")
    triton_fp32 = run(torch.float32, False, "triton")
    triton_bf16 = run(torch.bfloat16, True, "triton")

    # Logits of several units through norm weights of 0.5-1.5: bf16 is
    # compared by direction (and on an fp8 cache codes can sit a step apart).
    tol = 2e-3 if kv_cache_dtype == "fp8" else 1e-4
    torch.testing.assert_close(triton_fp32, reference, rtol=tol, atol=tol)
    cosine = torch.nn.functional.cosine_similarity(triton_bf16, reference, dim=-1)
    assert cosine.min() > 0.998, cosine
