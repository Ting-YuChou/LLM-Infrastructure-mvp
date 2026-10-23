"""Attention over the paged KV cache.

Cache layout per layer is [num_blocks, block_size, kv_heads, head_dim]; query
head ``h`` reads kv head ``h // (q_heads // kv_heads)``, as in transformers'
repeat_kv. The references below are the PyTorch path and the ground truth for
the kernels: they compute in fp32 and never read sizes back to the host.

The Triton decode kernel is flash-decoding: each program owns one (sequence,
kv head, split) and reads its slice of the context once for all the query
heads of that kv head, keeping a running softmax; a second kernel merges the
splits by their log-sum-exp. Lengths and block ids come from device tensors,
and the split count depends only on the batch size, so a captured CUDA graph
stays valid as contexts grow.

With ``key_scales``/``value_scales`` the cache holds e4m3 codes with one scale
per (slot, kv head) (kernels/fp8.py). The kernels decode the codes into bf16
for tl.dot and fold the scales into the scores (K) and the probabilities (V).
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch

from .fp8 import FP8_HALF_SCALE, fp8_dequantize_reference

try:  # pragma: no cover - exercised only in CUDA/Triton environments
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

MAX_DECODE_SPLITS = 32
# Launch settings, from benchmarks/phase1/tune.py on an A100 (LOG.md, run 3).
# Decode: block_n is the context tokens per inner-loop tile, independent of
# the cache block size. Small batches split every context many ways (up to
# 32 splits at B=1) and want wide programs with no pipelining; from B=64 a
# sequence has a single split and a long, three-stage tile wins. Each tier
# is within 3% of the best config of every swept (B, ctx). The choice
# depends only on the batch size, so it is fixed per CUDA graph.
DECODE_CONFIGS = (  # (smallest batch, config)
    (1, {"block_n": 64, "num_warps": 8, "num_stages": 1}),
    (16, {"block_n": 32, "num_warps": 4, "num_stages": 2}),
    (64, {"block_n": 128, "num_warps": 4, "num_stages": 3}),
)
# Prefill: query rows x context tokens per program iteration. Within 1.3%
# of the best config on every swept shape (1.3-1.4x over 64 x 64).
PREFILL_CONFIG = {"block_m": 128, "block_n": 32, "num_warps": 4, "num_stages": 2}

if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton
    from .fp8 import fp8_decode_scaled, fp8_prescale_query

    _FP8_HALF = tl.constexpr(FP8_HALF_SCALE)

    @triton.jit
    def _paged_decode_split_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        out_ptr,
        part_out_ptr,
        part_lse_ptr,
        k_scale_ptr,
        v_scale_ptr,
        scale_log2e,
        stride_qb,
        stride_qh,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_sb,
        stride_st,
        stride_bt,
        stride_ob,
        stride_oh,
        stride_pb,
        stride_ph,
        stride_ps,
        stride_lb,
        stride_lh,
        GROUP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        KV_FP8: tl.constexpr,
    ):
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)

        # Contiguous context range of this split, a multiple of BLOCK_N and at
        # least two tiles long; splits past the end do no work.
        context_len = tl.load(context_lens_ptr + seq)
        split_len = (context_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_len = tl.maximum(split_len, 2 * BLOCK_N)
        split_len = (split_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        start = split * split_len
        end = tl.minimum(start + split_len, context_len)

        offs_h = tl.arange(0, BLOCK_H)
        head_mask = offs_h < GROUP
        heads = kv_head * GROUP + offs_h
        offs_d = tl.arange(0, HEAD_DIM)
        # The GROUP query heads of this kv head, padded to BLOCK_H rows so
        # tl.dot can use tensor cores.
        q = tl.load(
            q_ptr + seq * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        if KV_FP8:  # decoded keys are their values times 2**-120
            q = fp8_prescale_query(q)

        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
        table = block_tables_ptr + seq.to(tl.int64) * stride_bt
        for tile_start in range(start, end, BLOCK_N):
            offs_n = tile_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < end
            blocks = tl.load(table + offs_n // PAGE_SIZE, mask=n_mask, other=0)
            within = (offs_n % PAGE_SIZE).to(tl.int64)
            k_rows = blocks.to(tl.int64) * stride_kb + within * stride_kt + kv_head * stride_kh
            v_rows = blocks.to(tl.int64) * stride_vb + within * stride_vt + kv_head * stride_vh
            k = tl.load(
                k_cache_ptr + k_rows[:, None] + offs_d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            if KV_FP8:
                s_rows = blocks.to(tl.int64) * stride_sb + within * stride_st + kv_head
                k_scale = tl.load(k_scale_ptr + s_rows, mask=n_mask, other=0.0)
                v_scale = tl.load(v_scale_ptr + s_rows, mask=n_mask, other=0.0)
                scores = tl.dot(q, tl.trans(fp8_decode_scaled(k, q)), input_precision=INPUT_PRECISION)
                scores = scores * (k_scale * (scale_log2e * _FP8_HALF))[None, :]
            else:
                scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
                scores = scores * scale_log2e
            scores = tl.where(n_mask[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(
                v_cache_ptr + v_rows[:, None] + offs_d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            if KV_FP8:
                pv = (p * (v_scale * _FP8_HALF)[None, :]).to(q.dtype)
                acc = acc * alpha[:, None] + tl.dot(
                    pv, fp8_decode_scaled(v, q), input_precision=INPUT_PRECISION
                )
            else:
                acc = acc * alpha[:, None] + tl.dot(
                    p.to(v.dtype), v, input_precision=INPUT_PRECISION
                )
            m_i = m_new

        if KV_FP8:
            acc = acc * _FP8_HALF
        # An empty split (or context) has l_i == 0: write 0, not 0/0.
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        result = acc * inv_l[:, None]
        if NUM_SPLITS == 1:
            tl.store(
                out_ptr + seq * stride_ob + heads[:, None] * stride_oh + offs_d[None, :],
                result.to(out_ptr.dtype.element_ty),
                mask=head_mask[:, None],
            )
        else:
            tl.store(
                part_out_ptr
                + seq * stride_pb
                + heads[:, None] * stride_ph
                + split * stride_ps
                + offs_d[None, :],
                result,
                mask=head_mask[:, None],
            )
            # Base-2 log-sum-exp of this split's scores; -inf when empty.
            tl.store(
                part_lse_ptr + seq * stride_lb + heads * stride_lh + split,
                m_i + tl.log2(l_i),
                mask=head_mask,
            )

    @triton.jit
    def _paged_decode_reduce_kernel(
        part_out_ptr,
        part_lse_ptr,
        out_ptr,
        stride_pb,
        stride_ph,
        stride_ps,
        stride_lb,
        stride_lh,
        stride_ob,
        stride_oh,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        seq = tl.program_id(0)
        head = tl.program_id(1)
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)
        lse = tl.load(part_lse_ptr + seq * stride_lb + head * stride_lh + offs_s)
        lse_max = tl.max(lse, axis=0)
        lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
        weights = tl.exp2(lse - lse_max)
        parts = tl.load(
            part_out_ptr
            + seq * stride_pb
            + head * stride_ph
            + offs_s[:, None] * stride_ps
            + offs_d[None, :]
        )
        total = tl.sum(weights, axis=0)
        inv_total = tl.where(total > 0, 1.0 / total, 0.0)
        out = tl.sum(parts * weights[:, None], axis=0) * inv_total
        tl.store(
            out_ptr + seq * stride_ob + head * stride_oh + offs_d,
            out.to(out_ptr.dtype.element_ty),
        )

    @triton.jit
    def _prefill_tile(
        acc,
        m_i,
        l_i,
        q,
        q_pos,
        table,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        tile_start,
        kv_end,
        kv_head,
        scale_log2e,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_sb,
        stride_st,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MASKED: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        KV_FP8: tl.constexpr,
    ):
        """One BLOCK_N context tile of the online softmax. MASKED tiles (the
        diagonal and the context end) apply the causal and bounds masks."""
        offs_n = tile_start + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        if MASKED:
            n_mask = offs_n < kv_end
            blocks = tl.load(table + offs_n // PAGE_SIZE, mask=n_mask, other=0)
        else:
            blocks = tl.load(table + offs_n // PAGE_SIZE)
        within = (offs_n % PAGE_SIZE).to(tl.int64)
        k_rows = blocks.to(tl.int64) * stride_kb + within * stride_kt + kv_head * stride_kh
        v_rows = blocks.to(tl.int64) * stride_vb + within * stride_vt + kv_head * stride_vh
        if MASKED:
            k = tl.load(k_cache_ptr + k_rows[:, None] + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        else:
            k = tl.load(k_cache_ptr + k_rows[:, None] + offs_d[None, :])
        if KV_FP8:
            s_rows = blocks.to(tl.int64) * stride_sb + within * stride_st + kv_head
            if MASKED:
                k_scale = tl.load(k_scale_ptr + s_rows, mask=n_mask, other=0.0)
                v_scale = tl.load(v_scale_ptr + s_rows, mask=n_mask, other=0.0)
            else:
                k_scale = tl.load(k_scale_ptr + s_rows)
                v_scale = tl.load(v_scale_ptr + s_rows)
            scores = tl.dot(q, tl.trans(fp8_decode_scaled(k, q)), input_precision=INPUT_PRECISION)
            scores = scores * (k_scale * (scale_log2e * _FP8_HALF))[None, :]
        else:
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale_log2e
        if MASKED:
            visible = n_mask[None, :] & (offs_n[None, :] <= q_pos[:, None])
            scores = tl.where(visible, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if MASKED:
            v = tl.load(v_cache_ptr + v_rows[:, None] + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        else:
            v = tl.load(v_cache_ptr + v_rows[:, None] + offs_d[None, :])
        if KV_FP8:
            pv = (p * (v_scale * _FP8_HALF)[None, :]).to(q.dtype)
            acc = acc * alpha[:, None] + tl.dot(pv, fp8_decode_scaled(v, q), input_precision=INPUT_PRECISION)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=INPUT_PRECISION)
        return acc, m_new, l_i

    @triton.jit
    def _paged_prefill_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        cu_seqlens_ptr,
        context_lens_ptr,
        out_ptr,
        k_scale_ptr,
        v_scale_ptr,
        scale_log2e,
        stride_qt,
        stride_qh,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_sb,
        stride_st,
        stride_bt,
        stride_ot,
        stride_oh,
        GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        KV_FP8: tl.constexpr,
    ):
        seq = tl.program_id(0)
        tile = tl.program_id(1)
        head = tl.program_id(2)
        kv_head = head // GROUP

        q_start = tl.load(cu_seqlens_ptr + seq)
        q_len = tl.load(cu_seqlens_ptr + seq + 1) - q_start
        context_len = tl.load(context_lens_ptr + seq)
        # The new tokens sit at the end of the context, after any cached
        # prefix (an earlier chunk or a prefix-cache hit).
        prefix = context_len - q_len
        row0 = tile * BLOCK_M
        offs_m = row0 + tl.arange(0, BLOCK_M)
        m_mask = offs_m < q_len
        q_pos = prefix + offs_m
        offs_d = tl.arange(0, HEAD_DIM)
        q_rows = (q_start + offs_m).to(tl.int64)
        q = tl.load(
            q_ptr + q_rows[:, None] * stride_qt + head * stride_qh + offs_d[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        if KV_FP8:  # decoded keys are their values times 2**-120
            q = fp8_prescale_query(q)

        # Causal: the last row of this tile sees keys up to prefix + row0 +
        # BLOCK_M - 1. Programs past the end of a short sequence do no work.
        kv_end = tl.minimum(context_len, prefix + row0 + BLOCK_M)
        kv_end = tl.where(row0 < q_len, kv_end, 0)
        # Keys below prefix + row0 + 1 are visible to every row of the tile:
        # those whole tiles skip the masks.
        full_end = tl.minimum((prefix + row0 + 1) // BLOCK_N * BLOCK_N, kv_end)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        table = block_tables_ptr + seq.to(tl.int64) * stride_bt
        for tile_start in range(0, full_end, BLOCK_N):
            acc, m_i, l_i = _prefill_tile(
                acc, m_i, l_i, q, q_pos, table, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
                tile_start, kv_end, kv_head, scale_log2e, stride_kb, stride_kt, stride_kh,
                stride_vb, stride_vt, stride_vh, stride_sb, stride_st, HEAD_DIM, PAGE_SIZE,
                BLOCK_N, False, INPUT_PRECISION, KV_FP8,
            )
        for tile_start in range(full_end, kv_end, BLOCK_N):
            acc, m_i, l_i = _prefill_tile(
                acc, m_i, l_i, q, q_pos, table, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
                tile_start, kv_end, kv_head, scale_log2e, stride_kb, stride_kt, stride_kh,
                stride_vb, stride_vt, stride_vh, stride_sb, stride_st, HEAD_DIM, PAGE_SIZE,
                BLOCK_N, True, INPUT_PRECISION, KV_FP8,
            )

        if KV_FP8:
            acc = acc * _FP8_HALF
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        tl.store(
            out_ptr + q_rows[:, None] * stride_ot + head * stride_oh + offs_d[None, :],
            (acc * inv_l[:, None]).to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def decode_num_splits(batch_size: int, kv_heads: int, device: torch.device) -> int:
    """Splits per sequence: enough programs to fill the GPU about four times.

    Depends only on the batch shape (never on context lengths), so one CUDA
    graph per batch size stays valid as sequences grow.
    """
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    wanted = math.ceil(4 * _num_sms(device_index) / (batch_size * kv_heads))
    return min(MAX_DECODE_SPLITS, triton.next_power_of_2(max(1, wanted)))


def decode_config(batch_size: int) -> dict:
    """Launch settings of the Triton decode kernel for a batch size."""
    return next(c for smallest, c in reversed(DECODE_CONFIGS) if batch_size >= smallest)


def _gather_blocks(cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """[..., width] block ids -> [..., width * block_size, *cache.shape[2:]]."""
    blocks = cache[block_table.long()]
    return blocks.flatten(block_table.ndim - 1, block_table.ndim)


def _gather_kv(cache: torch.Tensor, scales: Optional[torch.Tensor], block_table: torch.Tensor) -> torch.Tensor:
    """The block table's K or V in fp32, dequantized when ``scales`` is given."""
    if scales is None:
        return _gather_blocks(cache, block_table).float()
    # Through uint8: not every device implements index ops for float8.
    codes = _gather_blocks(cache.view(torch.uint8), block_table).view(torch.float8_e4m3fn)
    return fp8_dequantize_reference(codes, _gather_blocks(scales, block_table))


def paged_decode_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One query token per sequence: [batch, q_heads, head_dim] -> same shape.

    Reads the full block-table width and masks positions >= context_lens, so
    shapes never depend on the lengths. A row with context_len 0 returns 0.
    """
    batch, q_heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    group = q_heads // kv_heads
    keys = _gather_kv(key_cache, key_scales, block_tables)  # [B, T, Hkv, D]
    values = _gather_kv(value_cache, value_scales, block_tables)
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
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
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
        keys = _gather_kv(key_cache, key_scales, table)[:context_len]  # [T, Hkv, D]
        values = _gather_kv(value_cache, value_scales, table)[:context_len]

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


def is_triton_available() -> bool:
    return TRITON_AVAILABLE


def paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    use_triton: bool = False,
    num_splits: Optional[int] = None,
    config: Optional[dict] = None,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[batch, q_heads, head_dim] decode queries -> attention output, same shape
    (written into ``out`` when given, e.g. a row slice of a mixed batch's output).
    ``key_scales``/``value_scales``: the cache holds e4m3 codes (kernels/fp8.py)."""
    if use_triton and TRITON_AVAILABLE and query.is_cuda:  # pragma: no cover
        return _paged_decode_attention_triton(
            query, key_cache, value_cache, block_tables, context_lens, scale, num_splits,
            config or decode_config(query.shape[0]), out, key_scales, value_scales,
        )
    result = paged_decode_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, scale, key_scales, value_scales
    )
    return result if out is None else out.copy_(result)


def _fp8_operands(key_cache, value_cache, key_scales, value_scales):
    """The kernels' cache and scale arguments: uint8 codes (Triton takes
    float8e4nv only from sm_89 on) and the scale strides, or dummies."""
    if key_scales is None:
        return key_cache, value_cache, key_cache, value_cache, (0, 0)
    if not (key_scales.is_contiguous() and value_scales.is_contiguous()):
        raise ValueError("the KV scales must be contiguous")
    return (
        key_cache.view(torch.uint8), value_cache.view(torch.uint8), key_scales, value_scales,
        key_scales.stride()[:2],
    )


def _paged_decode_attention_triton(  # pragma: no cover - depends on CUDA/Triton
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_splits: Optional[int],
    config: dict,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, q_heads, head_dim = query.shape
    _, page_size, kv_heads, cache_head_dim = key_cache.shape
    if cache_head_dim != head_dim or q_heads % kv_heads != 0:
        raise ValueError("query and KV cache head layouts do not match")
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError("the Triton decode kernel needs a power-of-two head_dim >= 16")
    if query.stride(-1) != 1 or key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("head_dim must be the contiguous dimension")
    group = q_heads // kv_heads
    if num_splits is None:
        num_splits = decode_num_splits(batch, kv_heads, query.device)
    if num_splits & (num_splits - 1):
        raise ValueError("num_splits must be a power of two")

    if out is None:
        out = torch.empty_like(query)
    if batch == 0:
        return out
    if num_splits == 1:
        part_out = part_lse = out  # unused
        part_strides = (0, 0, 0)
        lse_strides = (0, 0)
    else:
        part_out = torch.empty(
            batch, q_heads, num_splits, head_dim, dtype=torch.float32, device=query.device
        )
        part_lse = torch.empty(
            batch, q_heads, num_splits, dtype=torch.float32, device=query.device
        )
        part_strides = part_out.stride()[:3]
        lse_strides = part_lse.stride()[:2]

    key_cache, value_cache, k_scales, v_scales, scale_strides = _fp8_operands(
        key_cache, value_cache, key_scales, value_scales
    )
    _paged_decode_split_kernel[(batch, kv_heads, num_splits)](
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        out,
        part_out,
        part_lse,
        k_scales,
        v_scales,
        scale * 1.4426950408889634,  # log2(e): the kernel uses exp2
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        *scale_strides,
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        *part_strides,
        *lse_strides,
        GROUP=group,
        BLOCK_H=max(16, triton.next_power_of_2(group)),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=config["block_n"],
        NUM_SPLITS=num_splits,
        INPUT_PRECISION="ieee" if query.dtype == torch.float32 else "tf32",
        KV_FP8=key_scales is not None,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    if num_splits > 1:
        _paged_decode_reduce_kernel[(batch, q_heads)](
            part_out,
            part_lse,
            out,
            *part_strides,
            *lse_strides,
            out.stride(0),
            out.stride(1),
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
    return out


def paged_prefill_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens_host: list[int],
    context_lens_host: list[int],
    scale: float,
    use_triton: bool = False,
    config: Optional[dict] = None,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Causal attention for packed queries [num_tokens, q_heads, head_dim].

    The kernel reads lengths from the device tensors; the reference (and the
    kernel's grid size) use the host copies. Written into ``out`` when given.
    ``key_scales``/``value_scales``: the cache holds e4m3 codes (kernels/fp8.py).
    """
    if use_triton and TRITON_AVAILABLE and query.is_cuda:  # pragma: no cover
        return _paged_prefill_attention_triton(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seqlens_q,
            context_lens,
            max(query_lens_host, default=0),
            scale,
            config or PREFILL_CONFIG,
            out,
            key_scales,
            value_scales,
        )
    result = paged_prefill_attention_reference(
        query, key_cache, value_cache, block_tables, query_lens_host, context_lens_host, scale,
        key_scales, value_scales,
    )
    return result if out is None else out.copy_(result)


def _paged_prefill_attention_triton(  # pragma: no cover - depends on CUDA/Triton
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    max_query_len: int,
    scale: float,
    config: dict,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens, q_heads, head_dim = query.shape
    _, page_size, kv_heads, cache_head_dim = key_cache.shape
    if cache_head_dim != head_dim or q_heads % kv_heads != 0:
        raise ValueError("query and KV cache head layouts do not match")
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError("the Triton prefill kernel needs a power-of-two head_dim >= 16")
    if query.stride(-1) != 1 or key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("head_dim must be the contiguous dimension")
    if out is None:
        out = torch.empty_like(query)
    num_seqs = cu_seqlens_q.numel() - 1
    if num_tokens == 0 or num_seqs == 0:
        return out
    grid = (num_seqs, triton.cdiv(max_query_len, config["block_m"]), q_heads)
    key_cache, value_cache, k_scales, v_scales, scale_strides = _fp8_operands(
        key_cache, value_cache, key_scales, value_scales
    )
    _paged_prefill_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_tables,
        cu_seqlens_q,
        context_lens,
        out,
        k_scales,
        v_scales,
        scale * 1.4426950408889634,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        *scale_strides,
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        GROUP=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        INPUT_PRECISION="ieee" if query.dtype == torch.float32 else "tf32",
        KV_FP8=key_scales is not None,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out
