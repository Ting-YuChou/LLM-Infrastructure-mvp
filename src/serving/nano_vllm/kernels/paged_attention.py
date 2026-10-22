"""Attention over the paged KV cache.

Cache layout per layer is [num_blocks, block_size, kv_heads, head_dim]; query
head ``h`` reads kv head ``h // (q_heads // kv_heads)``, as in transformers'
repeat_kv. The references below are the PyTorch path and the ground truth for
the kernels: they compute in fp32 and never read sizes back to the host.
"""

from __future__ import annotations

import torch


def _gather_blocks(cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """[..., width] block ids -> [..., width * block_size, kv_heads, head_dim]."""
    blocks = cache[block_table.long()]
    return blocks.flatten(-4, -3)


def paged_decode_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One query token per sequence: [batch, q_heads, head_dim] -> same shape.

    Reads the full block-table width and masks positions >= context_lens, so
    shapes never depend on the lengths. A row with context_len 0 returns 0.
    """
    batch, q_heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    group = q_heads // kv_heads
    keys = _gather_blocks(key_cache, block_tables).float()  # [B, T, Hkv, D]
    values = _gather_blocks(value_cache, block_tables).float()
    positions = torch.arange(keys.shape[1], device=query.device)
    valid = positions.view(1, -1) < context_lens.view(-1, 1).to(positions.dtype)

    q = query.float().view(batch, kv_heads, group, head_dim)
    scores = torch.einsum("bhgd,bthd->bhgt", q, keys) * scale
    scores = scores.masked_fill(~valid.view(batch, 1, 1, -1), float("-inf"))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    # Masked slots may hold stale values from freed blocks; zero them so a
    # non-finite one cannot leak in through 0 * value.
    values = values.masked_fill(~valid.view(batch, -1, 1, 1), 0.0)
    out = torch.einsum("bhgt,bthd->bhgd", probs, values)
    return out.reshape(batch, q_heads, head_dim).to(query.dtype)


def paged_prefill_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_lens: list[int],
    context_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Causal attention for packed queries: [num_tokens, q_heads, head_dim].

    Sequence ``i`` has ``query_lens[i]`` new tokens at the end of its
    ``context_lens[i]`` cached ones (their K/V already written), so a prefix
    cache hit or an earlier prefill chunk is just a longer context. The
    lengths are host lists; this path loops over sequences.
    """
    _, q_heads, head_dim = query.shape
    block_size, kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = q_heads // kv_heads
    outputs = []
    start = 0
    for seq_index, (query_len, context_len) in enumerate(zip(query_lens, context_lens)):
        num_blocks = (context_len + block_size - 1) // block_size
        table = block_tables[seq_index, :num_blocks]
        keys = _gather_blocks(key_cache, table)[:context_len].float()  # [T, Hkv, D]
        values = _gather_blocks(value_cache, table)[:context_len].float()

        q = query[start : start + query_len].float()
        q = q.view(query_len, kv_heads, group, head_dim)
        scores = torch.einsum("qhgd,thd->hgqt", q, keys) * scale
        # Query i sits at position context_len - query_len + i.
        query_pos = torch.arange(query_len, device=query.device) + context_len - query_len
        key_pos = torch.arange(context_len, device=query.device)
        causal = key_pos.view(1, -1) <= query_pos.view(-1, 1)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hgqt,thd->qhgd", probs, values)
        outputs.append(out.reshape(query_len, q_heads, head_dim))
        start += query_len
    return torch.cat(outputs, dim=0).to(query.dtype)


def paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return paged_decode_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )


def paged_prefill_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_lens: list[int],
    context_lens: list[int],
    scale: float,
) -> torch.Tensor:
    return paged_prefill_attention_reference(
        query, key_cache, value_cache, block_tables, query_lens, context_lens, scale
    )
