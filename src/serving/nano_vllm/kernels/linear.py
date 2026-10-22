"""Linear layers for decode: a Triton GEMV for a few rows, F.linear otherwise.

At batch sizes up to GEMV_MAX_ROWS a projection is bound by reading its
weight once from HBM. cuBLAS reaches only 63-73% of peak bandwidth on the
o_proj and qkv shapes at M=1 (Phase 0), mostly from having too few thread
blocks in flight. This kernel tiles the output features, splits K when the
weight is short and wide so every SM stays busy, and accumulates in fp32
with the rows padded to 16 for tensor cores.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch
import torch.nn.functional as F

try:  # pragma: no cover - exercised only in CUDA/Triton environments
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

LINEAR_BACKENDS = ("torch", "triton")
GEMV_MAX_ROWS = 16
GEMV_BLOCK_N = 64
GEMV_BLOCK_K = 128
MAX_SPLIT_K = 8


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    @triton.jit
    def _gemv_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        part_ptr,
        M,
        N,
        K,
        k_per_split,
        stride_xm,
        stride_wn,
        stride_om,
        stride_ps,
        stride_pm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        split = tl.program_id(1)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        x_rows = x_ptr + offs_m[:, None] * stride_xm
        w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_start = split * k_per_split
        k_end = tl.minimum(k_start + k_per_split, K)
        for k in range(k_start, k_end, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            if EVEN_K:
                x = tl.load(x_rows + offs_k[None, :], mask=m_mask[:, None], other=0.0)
                w = tl.load(w_rows + offs_k[None, :], mask=n_mask[:, None], other=0.0)
            else:
                k_mask = offs_k < k_end
                x = tl.load(
                    x_rows + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0
                )
                w = tl.load(
                    w_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
                )
            acc += tl.dot(x, tl.trans(w))

        out_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            tl.store(
                out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                acc.to(out_ptr.dtype.element_ty),
                mask=out_mask,
            )
        else:
            tl.store(
                part_ptr + split * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :],
                acc,
                mask=out_mask,
            )

    @triton.jit
    def _gemv_reduce_kernel(
        part_ptr, out_ptr, N, stride_ps, stride_pm, stride_om,
        SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        total = tl.zeros([BLOCK], dtype=tl.float32)
        for split in tl.static_range(SPLIT_K):
            total += tl.load(part_ptr + split * stride_ps + row * stride_pm + offs, mask=mask)
        tl.store(out_ptr + row * stride_om + offs, total.to(out_ptr.dtype.element_ty), mask=mask)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def gemv_split_k(n: int, k: int, device: torch.device) -> int:
    """Split K until there are about two thread blocks per SM."""
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    blocks = math.ceil(n / GEMV_BLOCK_N)
    split = 1
    while (
        split < MAX_SPLIT_K
        and blocks * split < 2 * _num_sms(device_index)
        and k // (2 * split) >= 2 * GEMV_BLOCK_K  # keep >= 2 K tiles per split
    ):
        split *= 2
    return split


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    backend: str = "torch",
    split_k: Optional[int] = None,
) -> torch.Tensor:
    """y = x @ weight.T (+ bias). backend="triton" uses the GEMV for <= 16 rows."""
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    if (
        backend == "triton"
        and TRITON_AVAILABLE
        and x.is_cuda
        and bias is None
        and 0 < rows <= GEMV_MAX_ROWS
        and x.dtype in (torch.bfloat16, torch.float16)
    ):  # pragma: no cover - depends on CUDA/Triton
        return _gemv_triton(x, weight, split_k)
    return F.linear(x, weight, bias)


def _gemv_triton(  # pragma: no cover - depends on CUDA/Triton
    x: torch.Tensor, weight: torch.Tensor, split_k: Optional[int]
) -> torch.Tensor:
    k = x.shape[-1]
    n = weight.shape[0]
    if weight.shape[1] != k or weight.stride(-1) != 1:
        raise ValueError("weight must be [out_features, in_features], contiguous rows")
    flat = x.reshape(-1, k)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    m = flat.shape[0]
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if split_k is None:
        split_k = gemv_split_k(n, k, x.device)
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), GEMV_BLOCK_K) * GEMV_BLOCK_K
    split_k = triton.cdiv(k, k_per_split)
    if split_k > 1:
        part = torch.empty(split_k, m, n, dtype=torch.float32, device=x.device)
        part_strides = (part.stride(0), part.stride(1))
    else:
        part, part_strides = out, (0, 0)
    grid = (triton.cdiv(n, GEMV_BLOCK_N), split_k)
    _gemv_kernel[grid](
        flat,
        weight,
        out,
        part,
        m,
        n,
        k,
        k_per_split,
        flat.stride(0),
        weight.stride(0),
        out.stride(0),
        *part_strides,
        BLOCK_M=GEMV_MAX_ROWS,
        BLOCK_N=GEMV_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        SPLIT_K=split_k,
        EVEN_K=k % k_per_split == 0 and k_per_split % GEMV_BLOCK_K == 0,
        num_warps=4,
        num_stages=4,
    )
    if split_k > 1:
        block = 1024
        _gemv_reduce_kernel[(m, triton.cdiv(n, block))](
            part, out, n, *part_strides, out.stride(0), SPLIT_K=split_k, BLOCK=block
        )
    return out.view(*x.shape[:-1], n)
