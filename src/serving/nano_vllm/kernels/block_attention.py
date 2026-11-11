"""Attention of a fixed-size block of queries per sequence over the paged KV cache.

Speculative decoding runs a small block of new tokens per sequence each
cycle: the target verifies Q = K + 1 tokens causally, the DFlash drafter
denoises [anchor, MASK x K] with every block token attending to the whole
context and to the whole block. Both read a sequence's context once for all
of its block queries, so one routine serves, with the mask as a flag.

Shapes are static: every sequence owns ``q_max`` query rows (row
``seq * q_max + t``), of which the first ``q_lens[seq]`` are real. As in the
prefill path the block's own K/V are already in the cache: ``context_lens``
counts them, and query ``t`` sits at position ``context_len - q_len + t``.

Only the PyTorch reference exists so far; it is the ground truth for the
Triton kernel that follows.
"""

from __future__ import annotations

from typing import Optional

import torch

from .paged_attention import _gather_kv


def paged_block_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    q_lens: torch.Tensor,
    q_max: int,
    scale: float,
    causal: bool,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[batch * q_max, q_heads, head_dim] block queries -> same shape.

    ``causal=True`` lets query ``t`` see the keys up to its own position;
    ``causal=False`` lets every query of a sequence see all ``context_len``
    keys. Rows with ``t >= q_lens[seq]`` return 0. Like the decode reference
    this reads the full block-table width and masks, so no size depends on
    the lengths and nothing is read back to the host.
    """
    rows, q_heads, head_dim = query.shape
    batch = block_tables.shape[0]
    if rows != batch * q_max:
        raise ValueError(f"query has {rows} rows, expected batch * q_max = {batch * q_max}")
    kv_heads = key_cache.shape[2]
    group = q_heads // kv_heads
    keys = _gather_kv(key_cache, key_scales, block_tables)  # [B, T, Hkv, D]
    values = _gather_kv(value_cache, value_scales, block_tables)

    key_pos = torch.arange(keys.shape[1], device=query.device).view(1, 1, -1)
    context = context_lens.long().view(batch, 1, 1)
    q_len = q_lens.long().view(batch, 1, 1)
    offsets = torch.arange(q_max, device=query.device).view(1, q_max, 1)
    # Keys below ``limit`` are visible: through the query's own position when
    # causal (never past the context, as offsets < q_len), else the context.
    limit = context - q_len + offsets + 1 if causal else context
    valid = (key_pos < limit) & (offsets < q_len)  # [B, q_max, T]

    q = query.float().view(batch, q_max, kv_heads, group, head_dim)
    scores = torch.einsum("bqhgd,bthd->bhgqt", q, keys) * scale
    scores = scores.masked_fill(~valid.view(batch, 1, 1, q_max, -1), float("-inf"))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    # Slots past the context may hold stale values from freed blocks; zero
    # them so a non-finite one cannot leak in through 0 * value.
    values = values.masked_fill(~(key_pos < context).view(batch, -1, 1, 1), 0.0)
    out = torch.einsum("bhgqt,bthd->bqhgd", probs, values)
    return out.reshape(rows, q_heads, head_dim).to(query.dtype)


def paged_block_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    q_lens: torch.Tensor,
    q_max: int,
    scale: float,
    causal: bool,
    use_triton: bool = False,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Block attention (see the module docstring), into ``out`` when given.

    ``key_scales``/``value_scales``: the cache holds e4m3 codes (kernels/fp8.py).
    There is no Triton kernel yet: ``use_triton`` is accepted and ignored.
    """
    del use_triton
    result = paged_block_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, q_lens, q_max, scale, causal,
        key_scales, value_scales,
    )
    return result if out is None else out.copy_(result)
