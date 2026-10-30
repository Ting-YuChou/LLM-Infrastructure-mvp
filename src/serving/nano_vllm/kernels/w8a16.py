"""W8A16 lm_head: int8 weights with one fp32 scale per output row.

The decode step reads lm_head's [vocab, hidden] weight once per step: 1.05
GB in bf16 for Llama 3's 128,256 x 4,096, about 0.7 ms on an A100 at 95% of
peak bandwidth, the largest single read after the decoder projections under
4-bit weights. Stored as int8 it is half that.

Quantization is symmetric per row: s = absmax / 127, q = round(w / s) in
-127..127, so w ~= q * s. The kernel multiplies x by q in bf16 (|q| <= 127
is exact in bf16) with fp32 accumulation, then scales each column of the
output by s once: x @ (q * s).T = (x @ q.T) * s exactly in real numbers, so
the only rounding the int8 weight adds over bf16 is the quantization itself.

One Triton kernel serves every row count: BLOCK_M rows of x per program
(16 for decode-sized batches, 64 past that), BLOCK_N output columns, K in
BLOCK_K steps. Off CUDA (or without Triton) it is F.linear on the
dequantized weight.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Launch settings by row count (up to "rows" rows); benchmarks/phase1 kbench
# lm_head picks among them.
W8A16_CONFIGS = (
    {"rows": 16, "block_m": 16, "block_n": 32, "block_k": 256, "num_warps": 4, "num_stages": 3},
    {"rows": 1 << 30, "block_m": 64, "block_n": 128, "block_k": 64, "num_warps": 4, "num_stages": 3},
)


@dataclass(frozen=True)
class W8A16Weight:
    """An int8 [out_features, in_features] weight with fp32 per-row scales."""

    qweight: torch.Tensor  # int8 [N, K]
    scales: torch.Tensor  # fp32 [N]
    dtype: torch.dtype  # the activations' (and outputs') dtype

    @property
    def shape(self) -> torch.Size:
        return self.qweight.shape

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    def nbytes(self) -> int:
        return self.qweight.numel() + self.scales.numel() * 4


def quantize_w8a16(weight: torch.Tensor) -> W8A16Weight:
    """Symmetric per-row int8 of a [N, K] weight; outputs keep its dtype."""
    if weight.ndim != 2:
        raise ValueError("quantize_w8a16 expects a 2-D [out_features, in_features] weight")
    w = weight.float()
    scales = w.abs().amax(dim=1) / 127
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # an all-zero row: any scale
    q = torch.round(w / scales[:, None]).clamp_(-127, 127).to(torch.int8)
    return W8A16Weight(qweight=q.contiguous(), scales=scales.contiguous(), dtype=weight.dtype)


def dequantize_w8a16(weight: W8A16Weight, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """q * s in fp32, rounded once to ``dtype`` (default: the weight's)."""
    return (weight.qweight.float() * weight.scales[:, None]).to(dtype or weight.dtype)


def w8a16_linear_reference(x: torch.Tensor, weight: W8A16Weight) -> torch.Tensor:
    """(x @ q.T) * s in fp32, rounded once to x's dtype."""
    return ((x.float() @ weight.qweight.float().T) * weight.scales).to(x.dtype)


def w8a16_linear(
    x: torch.Tensor, weight: W8A16Weight, bias: Optional[torch.Tensor] = None, config: Optional[dict] = None
) -> torch.Tensor:
    """y = x @ (q * s).T (+ bias). ``config`` overrides the W8A16_CONFIGS entry."""
    k = x.shape[-1]
    if weight.shape[1] != k:
        raise ValueError(f"x has {k} features, the weight expects {weight.shape[1]}")
    rows = x.numel() // k if k else 0
    if (
        TRITON_AVAILABLE
        and x.is_cuda
        and rows > 0
        and x.dtype in (torch.bfloat16, torch.float16)
    ):  # pragma: no cover - depends on CUDA/Triton
        out = _w8a16_triton(x.reshape(rows, k), weight, config)
        out = out.view(*x.shape[:-1], weight.shape[0])
        return out if bias is None else out + bias
    return F.linear(x, dequantize_w8a16(weight, x.dtype), bias)


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    @triton.jit
    def _w8a16_kernel(
        x_ptr, w_ptr, s_ptr, out_ptr, M, N, K,
        stride_xm, stride_wn, stride_om,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        x_rows = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
        w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            if EVEN_K:
                x = tl.load(x_rows + offs_k[None, :], mask=m_mask[:, None], other=0.0)
                w = tl.load(w_rows + offs_k[None, :], mask=n_mask[:, None], other=0)
            else:
                k_mask = offs_k < K
                x = tl.load(x_rows + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                w = tl.load(w_rows + offs_k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0)
            acc += tl.dot(x, tl.trans(w.to(x.dtype)))  # |q| <= 127: exact in bf16 and fp16
        scales = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        tl.store(
            out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :],
            (acc * scales[None, :]).to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def w8a16_config(rows: int) -> dict:
    return next(c for c in W8A16_CONFIGS if rows <= c["rows"])


def _w8a16_triton(  # pragma: no cover - depends on CUDA/Triton
    x: torch.Tensor, weight: W8A16Weight, config: Optional[dict]
) -> torch.Tensor:
    if x.stride(-1) != 1:
        x = x.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    config = {**w8a16_config(m), **(config or {})}
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, config["block_n"]), triton.cdiv(m, config["block_m"]))
    _w8a16_kernel[grid](
        x, weight.qweight, weight.scales, out, m, n, k,
        x.stride(0), weight.qweight.stride(0), out.stride(0),
        BLOCK_M=config["block_m"], BLOCK_N=config["block_n"], BLOCK_K=config["block_k"],
        EVEN_K=k % config["block_k"] == 0,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
    return out
