"""Linear layers for decode: a GEMV for a few rows, F.linear otherwise.

At batch sizes up to GEMV_MAX_ROWS a projection is bound by reading its
weight once from HBM. cuBLAS reaches only 63-73% of peak bandwidth on the
o_proj and qkv shapes at M=1 (Phase 0), mostly from having too few thread
blocks in flight. Two versions:

- triton: tiles the output features, splits K when the weight is short and
  wide so every SM stays busy, accumulates in fp32 with the rows padded to
  16 for tensor cores (up to 16 rows);
- cuda: csrc/gemv_bf16.cu, CUDA cores with 128-bit loads and
  CUDA_GEMV_CONFIG["rows_per_warp"] weight rows per warp, built with
  torch.utils.cpp_extension on first use; used up to CUDA_GEMV_MAX_ROWS.

Quantized weights (W4A16Weight, w4a16.py; W8A16Weight, w8a16.py) take their own path.
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

from .cuda_ext import load_cuda_extension  # noqa: E402
from .w4a16 import W4A16Weight, w4a16_linear  # noqa: E402
from .w8a16 import W8A16Weight, w8a16_linear  # noqa: E402

LINEAR_BACKENDS = ("torch", "triton", "cuda")
GEMV_MAX_ROWS = 16
# The CUDA kernel takes up to 8 rows, but its fp32 FMAs on CUDA cores only
# keep up with cuBLAS's tensor cores for 1-2 rows (LOG.md, run 3).
CUDA_GEMV_MAX_ROWS = 2
GEMV_BLOCK_N = 32
GEMV_BLOCK_K = 128
MAX_SPLIT_K = 8
# Launch settings, from benchmarks/phase1/tune.py on an A100 (LOG.md, run 3):
# within 2% of the best config of each Llama 3.1 8B decode shape.
GEMV_CONFIG = {"block_n": GEMV_BLOCK_N, "block_k": GEMV_BLOCK_K, "num_warps": 4, "num_stages": 3}
CUDA_GEMV_CONFIG = {"rows_per_warp": 2}


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


def gemv_split_k(
    n: int, k: int, device: torch.device, block_n: int = GEMV_BLOCK_N, block_k: int = GEMV_BLOCK_K
) -> int:
    """Split K until there are about two thread blocks per SM."""
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    blocks = math.ceil(n / block_n)
    split = 1
    while (
        split < MAX_SPLIT_K
        and blocks * split < 2 * _num_sms(device_index)
        and k // (2 * split) >= 2 * block_k  # keep >= 2 K tiles per split
    ):
        split *= 2
    return split


def linear(
    x: torch.Tensor,
    weight: torch.Tensor | W4A16Weight | W8A16Weight,
    bias: Optional[torch.Tensor] = None,
    backend: str = "torch",
    split_k: Optional[int] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """y = x @ weight.T (+ bias).

    backend="triton" uses the Triton GEMV for <= 16 rows, backend="cuda" the
    CUDA one for <= 2 rows of bf16; everything else goes to F.linear.
    ``config`` overrides the chosen kernel's launch settings (GEMV_CONFIG or
    CUDA_GEMV_CONFIG). A W4A16Weight always goes to w4a16_linear and a
    W8A16Weight to w8a16_linear, whatever the backend.
    """
    if isinstance(weight, W4A16Weight):
        return w4a16_linear(x, weight, bias, config)
    if isinstance(weight, W8A16Weight):
        return w8a16_linear(x, weight, bias, config)
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    if (
        backend == "triton"
        and TRITON_AVAILABLE
        and x.is_cuda
        and bias is None
        and 0 < rows <= GEMV_MAX_ROWS
        and x.dtype in (torch.bfloat16, torch.float16)
    ):  # pragma: no cover - depends on CUDA/Triton
        return _gemv_triton(x, weight, split_k, config or GEMV_CONFIG)
    if (
        backend == "cuda"
        and x.is_cuda
        and bias is None
        and 0 < rows <= CUDA_GEMV_MAX_ROWS
        and x.dtype == torch.bfloat16
        and weight.is_contiguous()
        and x.shape[-1] % 8 == 0
    ):  # pragma: no cover - depends on CUDA
        extension = load_cuda_extension()
        if extension is not None:
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            rows_per_warp = (config or CUDA_GEMV_CONFIG)["rows_per_warp"]
            out = extension.gemv_bf16(flat, weight, rows_per_warp)
            return out.view(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight, bias)


def _gemv_triton(  # pragma: no cover - depends on CUDA/Triton
    x: torch.Tensor, weight: torch.Tensor, split_k: Optional[int], config: dict = GEMV_CONFIG
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
    block_n, block_k = config["block_n"], config["block_k"]
    if split_k is None:
        split_k = gemv_split_k(n, k, x.device, block_n, block_k)
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    split_k = triton.cdiv(k, k_per_split)
    if split_k > 1:
        part = torch.empty(split_k, m, n, dtype=torch.float32, device=x.device)
        part_strides = (part.stride(0), part.stride(1))
    else:
        part, part_strides = out, (0, 0)
    grid = (triton.cdiv(n, block_n), split_k)
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
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        EVEN_K=k % k_per_split == 0 and k_per_split % block_k == 0,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    if split_k > 1:
        block = 1024
        _gemv_reduce_kernel[(m, triton.cdiv(n, block))](
            part, out, n, *part_strides, out.stride(0), SPLIT_K=split_k, BLOCK=block
        )
    return out.view(*x.shape[:-1], n)
