"""RoPE on q/k plus the KV-cache write, straight from the fused QKV output.

In PyTorch this is a split, two rotations of about five kernels each and two
index_put writes per layer; the Triton kernel does it in one launch, one
program per token. ``cos``/``sin`` are already gathered for each token's
position ([num_tokens, head_dim], halves duplicated as in transformers).
"""

from __future__ import annotations

import torch

from .rotary import apply_rope_cos_sin

try:  # pragma: no cover - exercised only in CUDA/Triton environments
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    @triton.jit
    def _rope_cache_write_kernel(
        qkv_ptr,
        cos_ptr,
        sin_ptr,
        slot_ptr,
        q_out_ptr,
        k_cache_ptr,
        v_cache_ptr,
        stride_qkv,
        stride_cos,
        stride_qo_t,
        stride_qo_h,
        stride_k_slot,
        stride_k_head,
        stride_v_slot,
        stride_v_head,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEADS_P2: tl.constexpr,
        KV_HEADS_P2: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        HALF: tl.constexpr = HEAD_DIM // 2
        offs = tl.arange(0, HALF)
        # rotate_half pairs dimension i with i + HALF; math in fp32.
        cos = tl.load(cos_ptr + token * stride_cos + offs).to(tl.float32)[None, :]
        sin = tl.load(sin_ptr + token * stride_cos + offs).to(tl.float32)[None, :]
        row = qkv_ptr + token * stride_qkv

        heads = tl.arange(0, HEADS_P2)
        q_mask = (heads < NUM_HEADS)[:, None]
        q_src = row + heads[:, None] * HEAD_DIM + offs[None, :]
        x1 = tl.load(q_src, mask=q_mask, other=0.0).to(tl.float32)
        x2 = tl.load(q_src + HALF, mask=q_mask, other=0.0).to(tl.float32)
        q_dst = q_out_ptr + token * stride_qo_t + heads[:, None] * stride_qo_h + offs[None, :]
        out_type = q_out_ptr.dtype.element_ty
        tl.store(q_dst, (x1 * cos - x2 * sin).to(out_type), mask=q_mask)
        tl.store(q_dst + HALF, (x2 * cos + x1 * sin).to(out_type), mask=q_mask)

        slot = tl.load(slot_ptr + token).to(tl.int64)
        kv_heads = tl.arange(0, KV_HEADS_P2)
        kv_mask = (kv_heads < NUM_KV_HEADS)[:, None]
        k_src = row + NUM_HEADS * HEAD_DIM + kv_heads[:, None] * HEAD_DIM + offs[None, :]
        x1 = tl.load(k_src, mask=kv_mask, other=0.0).to(tl.float32)
        x2 = tl.load(k_src + HALF, mask=kv_mask, other=0.0).to(tl.float32)
        k_dst = k_cache_ptr + slot * stride_k_slot + kv_heads[:, None] * stride_k_head + offs[None, :]
        cache_type = k_cache_ptr.dtype.element_ty
        tl.store(k_dst, (x1 * cos - x2 * sin).to(cache_type), mask=kv_mask)
        tl.store(k_dst + HALF, (x2 * cos + x1 * sin).to(cache_type), mask=kv_mask)

        offs_d = tl.arange(0, HEAD_DIM)
        v_src = (
            row + (NUM_HEADS + NUM_KV_HEADS) * HEAD_DIM + kv_heads[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        v = tl.load(v_src, mask=kv_mask, other=0.0)
        v_dst = v_cache_ptr + slot * stride_v_slot + kv_heads[:, None] * stride_v_head + offs_d[None, :]
        tl.store(v_dst, v.to(v_cache_ptr.dtype.element_ty), mask=kv_mask)


def rope_and_cache_write_reference(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    num_tokens = qkv.shape[0]
    head_dim = key_cache.shape[-1]
    query, key, value = qkv.split(
        [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    query = apply_rope_cos_sin(query.view(num_tokens, num_heads, head_dim), cos, sin)
    key = apply_rope_cos_sin(key.view(num_tokens, num_kv_heads, head_dim), cos, sin)
    key_cache.view(-1, num_kv_heads, head_dim)[slot_mapping] = key
    value_cache.view(-1, num_kv_heads, head_dim)[slot_mapping] = value.view(
        num_tokens, num_kv_heads, head_dim
    )
    return query


def rope_and_cache_write(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    use_triton: bool = False,
) -> torch.Tensor:
    """Rotate q and k, write k and v to their slots; return q [tokens, heads, dim].

    ``qkv`` is [num_tokens, (num_heads + 2 * num_kv_heads) * head_dim] and the
    caches are contiguous [num_blocks, block_size, num_kv_heads, head_dim].
    """
    if use_triton and TRITON_AVAILABLE and qkv.is_cuda:  # pragma: no cover
        return _rope_and_cache_write_triton(
            qkv, cos, sin, key_cache, value_cache, slot_mapping, num_heads, num_kv_heads
        )
    return rope_and_cache_write_reference(
        qkv, cos, sin, key_cache, value_cache, slot_mapping, num_heads, num_kv_heads
    )


def _rope_and_cache_write_triton(  # pragma: no cover - depends on CUDA/Triton
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    num_tokens = qkv.shape[0]
    head_dim = key_cache.shape[-1]
    if qkv.stride(-1) != 1 or cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("qkv and cos/sin must be contiguous in the last dimension")
    if not (key_cache.is_contiguous() and value_cache.is_contiguous()):
        raise ValueError("the KV cache must be contiguous")
    query = torch.empty(
        num_tokens, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device
    )
    if num_tokens == 0:
        return query
    flat_k = key_cache.view(-1, num_kv_heads, head_dim)
    flat_v = value_cache.view(-1, num_kv_heads, head_dim)
    _rope_cache_write_kernel[(num_tokens,)](
        qkv,
        cos,
        sin,
        slot_mapping,
        query,
        flat_k,
        flat_v,
        qkv.stride(0),
        cos.stride(0),
        query.stride(0),
        query.stride(1),
        flat_k.stride(0),
        flat_k.stride(1),
        flat_v.stride(0),
        flat_v.stride(1),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEADS_P2=triton.next_power_of_2(num_heads),
        KV_HEADS_P2=triton.next_power_of_2(num_kv_heads),
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return query
