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

The Triton kernel is the decode kernel (paged_attention.py) with more rows:
a program owns one (sequence, kv head, split, row tile) and reads its slice
of the context once for ``block_r`` of the ``q_max * group`` queries of that
kv head (all of them when one row tile covers them), where the prefill
kernel launches a program per query head and reads the whole context in
each. A second kernel merges the splits.
"""

from __future__ import annotations

from typing import Optional

import torch

from .fp8 import FP8_HALF_SCALE
from .paged_attention import TRITON_AVAILABLE, _fp8_operands, _gather_kv, decode_num_splits

if TRITON_AVAILABLE:  # pragma: no cover - exercised only in CUDA/Triton environments
    import triton
    import triton.language as tl

# Launch settings per batch size, from benchmarks/phase1/tune.py on an A100
# for a block of 10 (phase3 LOG.md, M3). block_n is the context tokens per
# inner-loop tile. block_r (optional) caps the query rows of a program, the
# rest going to further row tiles that read the same context again (from the
# L2 cache, mostly); splits_shift (optional) halves the decode kernel's
# split count that many times. Below 64 sequences the kernel is bound by a
# program's latency, not by memory bandwidth: decode-sized programs of 16
# rows win, with half the splits for up to three sequences and a quarter
# from four (one split from 16). From 64 a program takes the whole block
# and reads the context once. Each tier is within 3% of the best swept
# config of every swept context, except 256 and 1024 tokens for one
# sequence (38% and 8% off; the tier is the one that holds 4096). The fp8
# cache was not swept and uses the same tiers.
BLOCK_CONFIGS = (  # (smallest batch, config)
    (1, {"block_n": 64, "num_warps": 4, "num_stages": 2, "block_r": 16, "splits_shift": 1}),
    (4, {"block_n": 64, "num_warps": 4, "num_stages": 2, "block_r": 16, "splits_shift": 2}),
    (64, {"block_n": 64, "num_warps": 4, "num_stages": 2}),
)

if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton
    from .fp8 import fp8_decode_scaled, fp8_prescale_query

    _FP8_HALF = tl.constexpr(FP8_HALF_SCALE)
    # The running maximum starts here, not at -inf: a causal row can see no
    # key of a whole split, and exp2(-inf - -inf) would be NaN. Any real
    # score is far above it, and it still gives an empty row lse = -inf.
    _NO_SCORE = tl.constexpr(-1.0e30)

    @triton.jit
    def _paged_block_split_kernel(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        q_lens_ptr,
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
        Q_MAX: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        ROW_TILES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        CAUSAL: tl.constexpr,
        VARLEN: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        KV_FP8: tl.constexpr,
    ):
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2) // ROW_TILES
        row_tile = tl.program_id(2) % ROW_TILES

        # This split's context range, as in the decode kernel: a multiple of
        # BLOCK_N and at least two tiles long; splits past the end do no work.
        context_len = tl.load(context_lens_ptr + seq)
        if VARLEN:
            q_len = tl.load(q_lens_ptr + seq)
        else:
            q_len = Q_MAX
        split_len = (context_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_len = tl.maximum(split_len, 2 * BLOCK_N)
        split_len = (split_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        start = split * split_len
        end = tl.minimum(start + split_len, context_len)

        # Row r is block token r // GROUP under query head r % GROUP of this
        # kv head; this program has BLOCK_R of them (a power of two >= 16, so
        # tl.dot can use tensor cores), the last row tile padded.
        offs_r = row_tile * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_t = offs_r // GROUP
        heads = kv_head * GROUP + offs_r % GROUP
        row_mask = offs_r < Q_MAX * GROUP
        live = offs_t < q_len
        rows = (seq * Q_MAX + offs_t).to(tl.int64)
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(
            q_ptr + rows[:, None] * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
            mask=(row_mask & live)[:, None],
            other=0.0,
        )
        if KV_FP8:  # decoded keys are their values times 2**-120
            q = fp8_prescale_query(q)
        # Keys below a row's limit are visible to it: through its own position
        # when causal, else the whole context; none for a row with no query.
        # The limit never passes context_len, and a tile reaches past this
        # split's end only where the end is context_len.
        if CAUSAL:
            limit = context_len - q_len + offs_t + 1
        else:
            limit = context_len + offs_t * 0
        limit = tl.where(live, limit, 0)

        m_i = tl.full([BLOCK_R], _NO_SCORE, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_R], dtype=tl.float32)
        acc = tl.zeros([BLOCK_R, HEAD_DIM], dtype=tl.float32)
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
            scores = tl.where(offs_n[None, :] < limit[:, None], scores, float("-inf"))
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
        # A row that saw nothing (no query, or a split after its position)
        # has l_i == 0: write 0, not 0/0.
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        result = acc * inv_l[:, None]
        if NUM_SPLITS == 1:
            tl.store(
                out_ptr + rows[:, None] * stride_ob + heads[:, None] * stride_oh + offs_d[None, :],
                result.to(out_ptr.dtype.element_ty),
                mask=row_mask[:, None],
            )
        else:
            tl.store(
                part_out_ptr
                + rows[:, None] * stride_pb
                + heads[:, None] * stride_ph
                + split * stride_ps
                + offs_d[None, :],
                result,
                mask=row_mask[:, None],
            )
            # Base-2 log-sum-exp of the scores this row saw; -inf for none.
            tl.store(
                part_lse_ptr + rows * stride_lb + heads * stride_lh + split,
                m_i + tl.log2(l_i),
                mask=row_mask,
            )


    @triton.jit
    def _paged_block_reduce_kernel(
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
        GROUP: tl.constexpr,
        GROUP_P2: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Merge the splits of one (row, kv head) for its GROUP query heads.
        All splits are loaded at once: a loop over them, with a program per
        (sequence, query head), waits on one load after another and is
        slower from four splits on (LOG.md, M3)."""
        row = tl.program_id(0).to(tl.int64)
        kv_head = tl.program_id(1)
        offs_g = tl.arange(0, GROUP_P2)
        g_mask = offs_g < GROUP
        heads = kv_head * GROUP + offs_g
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)
        lse = tl.load(
            part_lse_ptr + row * stride_lb + heads[:, None] * stride_lh + offs_s[None, :],
            mask=g_mask[:, None],
            other=float("-inf"),
        )
        lse_max = tl.max(lse, axis=1)
        lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
        weights = tl.exp2(lse - lse_max[:, None])
        parts = tl.load(
            part_out_ptr
            + row * stride_pb
            + heads[:, None, None] * stride_ph
            + offs_s[None, :, None] * stride_ps
            + offs_d[None, None, :],
            mask=g_mask[:, None, None],
            other=0.0,
        )
        total = tl.sum(weights, axis=1)
        inv_total = tl.where(total > 0, 1.0 / total, 0.0)
        out = tl.sum(parts * weights[:, :, None], axis=1) * inv_total[:, None]
        tl.store(
            out_ptr + row * stride_ob + heads[:, None] * stride_oh + offs_d[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=g_mask[:, None],
        )


def block_config(batch_size: int) -> dict:
    """Launch settings of the Triton block kernel for a batch size."""
    return next(c for smallest, c in reversed(BLOCK_CONFIGS) if batch_size >= smallest)


def paged_block_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    q_lens: Optional[torch.Tensor],
    q_max: int,
    scale: float,
    causal: bool,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[batch * q_max, q_heads, head_dim] block queries -> same shape.

    ``causal=True`` lets query ``t`` see the keys up to its own position;
    ``causal=False`` lets every query of a sequence see all ``context_len``
    keys. Rows with ``t >= q_lens[seq]`` return 0; ``q_lens=None`` means
    every sequence has ``q_max`` queries. Like the decode reference this
    reads the full block-table width and masks, so no size depends on the
    lengths and nothing is read back to the host.
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
    if q_lens is None:
        q_lens = torch.full_like(context_lens, q_max)
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
    q_lens: Optional[torch.Tensor],
    q_max: int,
    scale: float,
    causal: bool,
    use_triton: bool = False,
    num_splits: Optional[int] = None,
    config: Optional[dict] = None,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Block attention (see the module docstring), into ``out`` when given.

    ``q_lens=None``: every sequence has ``q_max`` queries (what the engine
    runs). ``key_scales``/``value_scales``: the cache holds e4m3 codes
    (kernels/fp8.py).
    """
    if use_triton and TRITON_AVAILABLE and query.is_cuda:  # pragma: no cover
        return _paged_block_attention_triton(
            query, key_cache, value_cache, block_tables, context_lens, q_lens, q_max, scale,
            causal, num_splits, config or block_config(block_tables.shape[0]),
            out, key_scales, value_scales,
        )
    result = paged_block_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, q_lens, q_max, scale, causal,
        key_scales, value_scales,
    )
    return result if out is None else out.copy_(result)


def _paged_block_attention_triton(  # pragma: no cover - depends on CUDA/Triton
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    q_lens: Optional[torch.Tensor],
    q_max: int,
    scale: float,
    causal: bool,
    num_splits: Optional[int],
    config: dict,
    out: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
    value_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows, q_heads, head_dim = query.shape
    batch = block_tables.shape[0]
    _, page_size, kv_heads, cache_head_dim = key_cache.shape
    if rows != batch * q_max:
        raise ValueError(f"query has {rows} rows, expected batch * q_max = {batch * q_max}")
    if cache_head_dim != head_dim or q_heads % kv_heads != 0:
        raise ValueError("query and KV cache head layouts do not match")
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError("the Triton block kernel needs a power-of-two head_dim >= 16")
    if query.stride(-1) != 1 or key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("head_dim must be the contiguous dimension")
    group = q_heads // kv_heads
    if num_splits is None:
        num_splits = decode_num_splits(batch, kv_heads, query.device)
        num_splits = max(1, num_splits >> config.get("splits_shift", 0))
    if num_splits & (num_splits - 1):
        raise ValueError("num_splits must be a power of two")

    if out is None:
        out = torch.empty_like(query)
    if rows == 0:
        return out
    if num_splits == 1:
        part_out = part_lse = out  # unused
        part_strides = (0, 0, 0)
        lse_strides = (0, 0)
    else:
        part_out = torch.empty(
            rows, q_heads, num_splits, head_dim, dtype=torch.float32, device=query.device
        )
        part_lse = torch.empty(rows, q_heads, num_splits, dtype=torch.float32, device=query.device)
        part_strides = part_out.stride()[:3]
        lse_strides = part_lse.stride()[:2]

    key_cache, value_cache, k_scales, v_scales, scale_strides = _fp8_operands(
        key_cache, value_cache, key_scales, value_scales
    )
    # Rows per program: all of a kv head's q_max * group queries, or
    # config["block_r"] of them with the rest in further row tiles.
    block_r = max(16, triton.next_power_of_2(q_max * group))
    block_r = min(block_r, config.get("block_r", block_r))
    row_tiles = triton.cdiv(q_max * group, block_r)
    _paged_block_split_kernel[(batch, kv_heads, num_splits * row_tiles)](
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        context_lens if q_lens is None else q_lens,
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
        Q_MAX=q_max,
        GROUP=group,
        BLOCK_R=block_r,
        ROW_TILES=row_tiles,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=config["block_n"],
        NUM_SPLITS=num_splits,
        CAUSAL=causal,
        VARLEN=q_lens is not None,
        INPUT_PRECISION="ieee" if query.dtype == torch.float32 else "tf32",
        KV_FP8=key_scales is not None,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    if num_splits > 1:
        _paged_block_reduce_kernel[(rows, kv_heads)](
            part_out,
            part_lse,
            out,
            *part_strides,
            *lse_strides,
            out.stride(0),
            out.stride(1),
            GROUP=group,
            GROUP_P2=triton.next_power_of_2(group),
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
    return out
