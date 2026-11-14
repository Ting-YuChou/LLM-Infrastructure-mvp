import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels import (  # noqa: E402
    paged_block_attention,
    paged_block_attention_reference,
    paged_prefill_attention_reference,
)
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402

BLOCK_SIZE = 4
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
GROUP = Q_HEADS // KV_HEADS
SCALE = 1.0 / math.sqrt(HEAD_DIM)
Q_MAX = 5
# (q_len, context_len), the block's own tokens counted in the context: a
# block that is the whole context, full and short blocks after a cached
# prefix, an empty row, contexts ending inside, at and across cache blocks.
SHAPES = [(5, 5), (5, 13), (3, 10), (0, 6), (1, 8), (4, 16)]


def _cache(kv_cache_dtype="auto", num_blocks=20):
    return PagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
        kv_cache_dtype=kv_cache_dtype,
    )


def _setup(kv_cache_dtype="auto", shapes=SHAPES):
    """A cache holding each sequence's K/V in shuffled blocks, the padded
    block tables, the K/V per sequence as written, and block queries."""
    torch.manual_seed(0)
    cache = _cache(kv_cache_dtype)
    order = torch.randperm(cache.num_blocks, generator=torch.Generator().manual_seed(1)).tolist()
    tables, written = [], []
    for _, context_len in shapes:
        table = [order.pop() for _ in range((context_len + BLOCK_SIZE - 1) // BLOCK_SIZE)]
        slots = torch.tensor(
            [table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in range(context_len)]
        )
        key = torch.randn(context_len, KV_HEADS, HEAD_DIM)
        value = torch.randn(context_len, KV_HEADS, HEAD_DIM)
        cache.update(0, slots, key, value)
        tables.append(table)
        written.append((key, value))
    width = max(len(table) for table in tables) + 1  # padded with the null block
    block_tables = torch.tensor(
        [table + [cache.null_block] * (width - len(table)) for table in tables], dtype=torch.int32
    )
    query = torch.randn(len(shapes) * Q_MAX, Q_HEADS, HEAD_DIM)
    return cache, block_tables, written, query


def _lens(shapes=SHAPES):
    return (
        torch.tensor([context for _, context in shapes], dtype=torch.int32),
        torch.tensor([q for q, _ in shapes], dtype=torch.int32),
    )


def _block(cache, block_tables, query, causal, shapes=SHAPES, fn=paged_block_attention_reference):
    context_lens, q_lens = _lens(shapes)
    return fn(
        query, cache.keys[0], cache.values[0], block_tables, context_lens, q_lens, Q_MAX, SCALE,
        causal=causal, key_scales=cache.key_scales[0], value_scales=cache.value_scales[0],
    )


def _dense(query, key, value, visible):
    """Softmax attention of [q, Hq, D] queries over [T, Hkv, D] keys/values
    with a [q, T] visibility mask, one query head at a time."""
    out = torch.zeros_like(query)
    for head in range(Q_HEADS):
        k, v = key[:, head // GROUP], value[:, head // GROUP]
        scores = (query[:, head] @ k.T) * SCALE
        scores = scores.masked_fill(~visible, float("-inf"))
        out[:, head] = torch.softmax(scores, dim=-1) @ v
    return out


def _padding_rows(shapes=SHAPES):
    return torch.tensor([t >= q_len for q_len, _ in shapes for t in range(Q_MAX)])


def test_causal_block_attention_matches_the_prefill_reference():
    cache, block_tables, _, query = _setup()
    out = _block(cache, block_tables, query, causal=True)

    real = ~_padding_rows()
    expected = paged_prefill_attention_reference(
        query[real],  # packed: each sequence's q_len real rows
        cache.keys[0],
        cache.values[0],
        block_tables,
        [q for q, _ in SHAPES],
        [context for _, context in SHAPES],
        SCALE,
    )
    torch.testing.assert_close(out[real], expected, rtol=1e-5, atol=1e-6)


def test_bidirectional_block_attention_matches_dense_softmax_over_context_and_block():
    cache, block_tables, written, query = _setup()
    out = _block(cache, block_tables, query, causal=False)

    for seq, ((q_len, context_len), (key, value)) in enumerate(zip(SHAPES, written)):
        rows = slice(seq * Q_MAX, seq * Q_MAX + q_len)
        # Every block query sees the cached prefix and the whole block.
        visible = torch.ones(q_len, context_len, dtype=torch.bool)
        expected = _dense(query[rows], key, value, visible)
        torch.testing.assert_close(out[rows], expected, rtol=1e-5, atol=1e-6)


def test_causal_block_attention_matches_dense_softmax_up_to_each_query():
    cache, block_tables, written, query = _setup()
    out = _block(cache, block_tables, query, causal=True)

    for seq, ((q_len, context_len), (key, value)) in enumerate(zip(SHAPES, written)):
        rows = slice(seq * Q_MAX, seq * Q_MAX + q_len)
        query_pos = context_len - q_len + torch.arange(q_len)
        visible = torch.arange(context_len).view(1, -1) <= query_pos.view(-1, 1)
        expected = _dense(query[rows], key, value, visible)
        torch.testing.assert_close(out[rows], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("causal", [True, False])
def test_rows_past_q_len_and_empty_sequences_return_zero(causal):
    cache, block_tables, _, query = _setup()
    out = _block(cache, block_tables, query, causal=causal)

    padding = _padding_rows()
    assert padding.sum() == sum(Q_MAX - q for q, _ in SHAPES)
    assert torch.equal(out[padding], torch.zeros_like(out[padding]))
    assert out[~padding].abs().sum(dim=(1, 2)).min() > 0
    empty = [seq for seq, (q_len, _) in enumerate(SHAPES) if q_len == 0]
    assert empty and all(not out[seq * Q_MAX : (seq + 1) * Q_MAX].any() for seq in empty)


@pytest.mark.parametrize("causal", [True, False])
def test_padding_queries_and_slots_past_the_context_do_not_leak(causal):
    cache, block_tables, _, query = _setup()
    expected = _block(cache, block_tables, query, causal=causal)

    # Whatever sits in padding query rows, in slots past a context's end, in
    # unused blocks and in the null block must not change any real row.
    used = torch.zeros(cache.num_blocks + 1, BLOCK_SIZE, dtype=torch.bool)
    for (_, context_len), table in zip(SHAPES, block_tables.tolist()):
        for position in range(context_len):
            used[table[position // BLOCK_SIZE], position % BLOCK_SIZE] = True
    cache.keys[0][~used] = float("nan")
    cache.values[0][~used] = float("nan")
    padding = _padding_rows()
    query = query.clone()
    query[padding] = float("nan")

    out = _block(cache, block_tables, query, causal=causal)
    assert torch.isfinite(out).all()
    assert torch.equal(out, expected)


@pytest.mark.parametrize("causal", [True, False])
def test_block_attention_does_not_depend_on_the_batch_or_q_max(causal):
    # A sequence alone, with q_max equal to its q_len, gives the rows it has
    # in the padded batch.
    cache, block_tables, _, query = _setup()
    out = _block(cache, block_tables, query, causal=causal)

    for seq, (q_len, context_len) in enumerate(SHAPES):
        if q_len == 0:
            continue
        rows = slice(seq * Q_MAX, seq * Q_MAX + q_len)
        alone = paged_block_attention_reference(
            query[rows], cache.keys[0], cache.values[0], block_tables[seq : seq + 1],
            torch.tensor([context_len], dtype=torch.int32), torch.tensor([q_len], dtype=torch.int32),
            q_len, SCALE, causal=causal,
        )
        torch.testing.assert_close(out[rows], alone, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("causal", [True, False])
def test_block_attention_on_an_fp8_cache_reads_the_dequantized_kv(causal):
    cache, block_tables, written, query = _setup("fp8")
    assert cache.keys[0].dtype == torch.float8_e4m3fn
    out = _block(cache, block_tables, query, causal=causal)

    # Exactly the attention over what the cache decodes to ...
    for seq, (q_len, context_len) in enumerate(SHAPES):
        key, value = cache.read_sequence(0, block_tables[seq], context_len)
        key, value = key[0].transpose(0, 1), value[0].transpose(0, 1)
        if causal:
            query_pos = context_len - q_len + torch.arange(q_len)
            visible = torch.arange(context_len).view(1, -1) <= query_pos.view(-1, 1)
        else:
            visible = torch.ones(q_len, context_len, dtype=torch.bool)
        rows = slice(seq * Q_MAX, seq * Q_MAX + q_len)
        torch.testing.assert_close(out[rows], _dense(query[rows], key, value, visible), rtol=1e-5, atol=1e-6)

    # ... the same as the prefill reference on that cache, and close to the
    # full-precision result.
    if causal:
        real = ~_padding_rows()
        expected = paged_prefill_attention_reference(
            query[real], cache.keys[0], cache.values[0], block_tables,
            [q for q, _ in SHAPES], [context for _, context in SHAPES], SCALE,
            cache.key_scales[0], cache.value_scales[0],
        )
        torch.testing.assert_close(out[real], expected, rtol=1e-5, atol=1e-6)
    full_cache, full_tables, _, full_query = _setup("auto")
    assert torch.equal(full_query, query) and torch.equal(full_tables, block_tables)
    full = _block(full_cache, full_tables, full_query, causal=causal)
    assert not torch.equal(out, full)
    torch.testing.assert_close(out, full, rtol=0.0, atol=0.1)


def test_entry_point_calls_the_reference_and_fills_out():
    cache, block_tables, _, query = _setup()
    expected = _block(cache, block_tables, query, causal=False)

    assert torch.equal(_block(cache, block_tables, query, causal=False, fn=paged_block_attention), expected)
    context_lens, q_lens = _lens()
    out = torch.empty_like(query)
    returned = paged_block_attention(
        query, cache.keys[0], cache.values[0], block_tables, context_lens, q_lens, Q_MAX, SCALE,
        causal=False, use_triton=True, out=out,
    )
    assert returned is out
    assert torch.equal(out, expected)


def test_block_attention_rejects_a_query_that_is_not_batch_times_q_max_rows():
    cache, block_tables, _, query = _setup()
    context_lens, q_lens = _lens()
    with pytest.raises(ValueError, match="batch \\* q_max"):
        paged_block_attention_reference(
            query[:-1], cache.keys[0], cache.values[0], block_tables, context_lens, q_lens,
            Q_MAX, SCALE, causal=True,
        )
