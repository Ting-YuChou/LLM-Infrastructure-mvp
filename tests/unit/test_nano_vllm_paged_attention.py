import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels import (  # noqa: E402
    causal_attention,
    paged_decode_attention_reference,
    paged_prefill_attention_reference,
)
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402
from serving.nano_vllm.metadata import build_batch  # noqa: E402

BLOCK_SIZE = 4
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _filled_cache(num_blocks=16):
    torch.manual_seed(0)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
    )
    cache.keys[0].normal_()
    cache.values[0].normal_()
    return cache


def _old_path(query, cache, table, context_len, query_positions):
    """The pre-Phase-1 path: read_sequence + repeat_kv + causal_attention."""
    key, value = cache.read_sequence(0, table, context_len)
    repeats = Q_HEADS // KV_HEADS
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    out = causal_attention(query.transpose(0, 1).unsqueeze(0), key, value, query_positions)
    return out.squeeze(0).transpose(0, 1)


def _tables(context_lens, num_blocks):
    """Non-contiguous, shuffled block ids per sequence."""
    order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(1)).tolist()
    tables = []
    for context_len in context_lens:
        need = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        tables.append([order.pop() for _ in range(need)])
    return tables


def _padded(tables, null_block):
    width = max(len(table) for table in tables)
    return torch.tensor(
        [table + [null_block] * (width - len(table)) for table in tables],
        dtype=torch.int32,
    )


def test_decode_reference_matches_read_sequence_path():
    cache = _filled_cache()
    context_lens = [1, 3, 4, 5, 9, 16]  # inside, at and across block boundaries
    tables = _tables(context_lens, cache.num_blocks)
    query = torch.randn(len(context_lens), Q_HEADS, HEAD_DIM)

    out = paged_decode_attention_reference(
        query,
        cache.keys[0],
        cache.values[0],
        _padded(tables, cache.null_block),
        torch.tensor(context_lens, dtype=torch.int32),
        SCALE,
    )

    for row, (table, context_len) in enumerate(zip(tables, context_lens)):
        expected = _old_path(
            query[row : row + 1], cache, table, context_len, torch.tensor([context_len - 1])
        )
        torch.testing.assert_close(out[row], expected[0], rtol=1e-5, atol=1e-5)


def test_decode_reference_ignores_non_finite_values_past_the_context():
    cache = _filled_cache()
    table = [3, 7]
    # Slot 2 of block 7 is past context_len 6, and the null block pads the
    # table: neither may leak into the output even if it holds NaN.
    cache.keys[0][7, 2:] = float("nan")
    cache.values[0][7, 2:] = float("nan")
    cache.keys[0][cache.null_block] = float("nan")
    cache.values[0][cache.null_block] = float("nan")
    query = torch.randn(2, Q_HEADS, HEAD_DIM)
    tables = torch.tensor([table + [cache.null_block], [cache.null_block] * 3], dtype=torch.int32)

    out = paged_decode_attention_reference(
        query,
        cache.keys[0],
        cache.values[0],
        tables,
        torch.tensor([6, 0], dtype=torch.int32),
        SCALE,
    )

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1], torch.zeros(Q_HEADS, HEAD_DIM))  # empty context
    expected = _old_path(query[:1], cache, table, 6, torch.tensor([5]))
    torch.testing.assert_close(out[0], expected[0], rtol=1e-5, atol=1e-5)


def test_prefill_reference_matches_causal_attention_with_cached_prefix():
    cache = _filled_cache()
    # (query_len, context_len): plain prompt, chunk after a cached prefix,
    # single token after a prefix.
    shapes = [(5, 5), (3, 10), (1, 7)]
    tables = _tables([context for _, context in shapes], cache.num_blocks)
    query_lens = [q for q, _ in shapes]
    context_lens = [c for _, c in shapes]
    query = torch.randn(sum(query_lens), Q_HEADS, HEAD_DIM)

    out = paged_prefill_attention_reference(
        query,
        cache.keys[0],
        cache.values[0],
        _padded(tables, cache.null_block),
        query_lens,
        context_lens,
        SCALE,
    )

    start = 0
    for table, (query_len, context_len) in zip(tables, shapes):
        positions = torch.arange(context_len - query_len, context_len)
        expected = _old_path(query[start : start + query_len], cache, table, context_len, positions)
        torch.testing.assert_close(out[start : start + query_len], expected, rtol=1e-5, atol=1e-5)
        start += query_len


def test_build_batch_pads_tables_and_points_logits_at_last_tokens():
    input_ids, positions, metadata = build_batch(
        input_ids=[5, 6, 7, 8],
        positions=[0, 1, 2, 9],
        slots=[0, 1, 2, 13],
        block_tables=[[0], [2, 3, 1]],
        context_lens=[3, 10],
        query_lens=[3, 1],
        is_prefill=True,
        null_block=8,
        num_slots=36,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [5, 6, 7, 8]
    assert positions.tolist() == [0, 1, 2, 9]
    assert metadata.block_tables.dtype == torch.int32
    assert metadata.block_tables.tolist() == [[0, 8, 8], [2, 3, 1]]
    assert metadata.cu_seqlens_q.tolist() == [0, 3, 4]
    assert metadata.context_lens.tolist() == [3, 10]
    assert metadata.logits_indices.tolist() == [2, 3]
    assert metadata.decode_only is False


@pytest.mark.parametrize("num_decode_rows", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("prefill_lens", [[3], [2, 7], [1, 4, 6]])
def test_build_batch_aligns_every_field_to_16_bytes(num_decode_rows, prefill_lens):
    # Triton compiles a kernel variant per pointer alignment, so no field may
    # start at a shape-dependent offset, including a mixed batch's prefill rows.
    query_lens = [1] * num_decode_rows + prefill_lens
    tables = [[i % 8] * (1 + i % 3) for i in range(len(query_lens))]
    num_tokens = sum(query_lens)
    _, _, metadata = build_batch(
        input_ids=list(range(num_tokens)),
        positions=[0] * num_tokens,
        slots=list(range(num_tokens)),
        block_tables=tables,
        context_lens=[n + 2 for n in query_lens],
        query_lens=query_lens,
        is_prefill=True,
        null_block=8,
        num_slots=36,
        device=torch.device("cpu"),
    )
    rows = metadata.num_decode_rows  # a leading 1-token prefill also counts
    mixed = 0 < rows < len(query_lens)
    fields = [metadata.slot_mapping, metadata.block_tables, metadata.context_lens,
              metadata.cu_seqlens_q, metadata.logits_indices]
    if mixed:
        fields += [metadata.prefill_cu_seqlens_q, metadata.prefill_context_lens,
                   metadata.prefill_block_tables]
        assert metadata.prefill_context_lens.tolist() == metadata.context_lens[rows:].tolist()
        assert metadata.prefill_block_tables.tolist() == metadata.block_tables[rows:].tolist()
    else:
        assert metadata.prefill_block_tables is None
    assert all(t.data_ptr() % 16 == 0 for t in fields)


def test_build_batch_rejects_slots_outside_the_cache():
    with pytest.raises(ValueError, match="outside the KV cache"):
        build_batch(
            input_ids=[1],
            positions=[0],
            slots=[36],
            block_tables=[[0]],
            context_lens=[1],
            query_lens=[1],
            is_prefill=False,
            null_block=8,
            num_slots=36,
            device=torch.device("cpu"),
        )
