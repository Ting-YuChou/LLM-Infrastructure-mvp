"""W4A16 linear layers: 4-bit integer weights, bf16/fp16 activations.

Weights are quantized when the model loads, so no checkpoint format is
involved: round-to-nearest, symmetric, along K in groups of
W4A16_GROUP_SIZE. Each group of a weight row gets one scale
s = absmax / 7.5 and each weight is stored as q = clamp(round(w / s) + 8,
0, 15), so w ~= (q - 8) * s. This is GPTQ's symmetric scheme without the
error feedback (plain RTN), with scales stored in the activation dtype.

Layout:
- qweight: int32 [N, K // 8]. Each int32 holds 8 consecutive K values of
  one row; nibble p (bits 4p..4p+3) holds k = 8j + W4A16_PACK_ORDER[p].
  The order puts k = 2i and 2i + 1 under the mask 0x000F000F << 4i, which
  is what the CUDA kernel dequantizes two at a time.
- scales: [N, K // group_size] in the activation dtype.

Decode-sized batches use csrc/w4a16.cu, which has three kernels over the
same qweight:
- gemv: CUDA cores, 8 rows of x per launch (looped up to
  W4A16_GEMV_MAX_ROWS); fp32 FMAs limit it to small batches;
- mma: tensor cores (mma.m16n8k16), up to W4A16_MMA_MAX_ROWS rows in one
  pass over the weight, each block reading x from L2; needs group_size 128;
- wide: tensor cores for 17-64 rows, warps of a block share one copy of x
  in shared memory, scales folded into the dequantized weights.
The tensor-core kernels split K across blocks when the weight is short; the
splits meet in a persistent fp32 workspace (w4a16_workspace).
Bigger batches dequantize the whole weight and call F.linear: they are
compute-bound, so the extra pass over the weight is a small share of their
time.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .cuda_ext import load_cuda_extension

logger = logging.getLogger(__name__)

W4A16_GROUP_SIZE = 128
W4A16_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
# The GEMV takes 8 rows of x per launch; up to here the op loops over 8-row
# chunks (each chunk re-reads the 4-bit weight).
W4A16_GEMV_MAX_ROWS = 32
W4A16_MMA_MAX_ROWS = 64
# Launch settings, from benchmarks/phase1/tune.py on an A100 (LOG.md, run 5).
# kernel: "gemv", "mma", "wide", or "auto" (gemv up to auto_gemv_rows, mma
# up to auto_mma_rows, then wide); splits: K splits of the tensor-core
# kernels, 0 lets the op choose from the grid size; gemv_prefetch: K steps
# of weight loads the GEMV keeps in flight while it computes (0, 1, 2, 4).
W4A16_CONFIG = {
    "kernel": "auto",
    "auto_gemv_rows": 2,
    "auto_mma_rows": 16,
    "rows_per_warp": 2,
    "splits": 0,
    "gemv_prefetch": 0,
}
_WORKSPACE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_RETIRED: list[tuple[torch.Tensor, torch.Tensor]] = []


@dataclass(frozen=True)
class W4A16Weight:
    """A quantized [out_features, in_features] weight."""

    qweight: torch.Tensor
    scales: torch.Tensor
    group_size: int = W4A16_GROUP_SIZE

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.qweight.shape[0], self.qweight.shape[1] * 8))

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.scales.dtype

    def nbytes(self) -> int:
        return self.qweight.numel() * 4 + self.scales.numel() * self.scales.element_size()


def quantize_w4a16(weight: torch.Tensor, group_size: int = W4A16_GROUP_SIZE) -> W4A16Weight:
    """RTN-quantize a [N, K] weight; scales keep the weight's dtype."""
    if weight.ndim != 2:
        raise ValueError("quantize_w4a16 expects a 2-D [out_features, in_features] weight")
    if sys.byteorder != "little":  # pragma: no cover
        raise RuntimeError("the int4 packing assumes a little-endian host")
    n, k = weight.shape
    if group_size % 32 or k % group_size:
        raise ValueError(
            f"in_features ({k}) must be a multiple of group_size ({group_size}), "
            "itself a multiple of 32"
        )
    groups = weight.float().reshape(n, k // group_size, group_size)
    scales = groups.abs().amax(dim=-1) / 7.5
    # An all-zero group would divide by zero; any scale reproduces it.
    scales = torch.where(scales == 0, torch.ones_like(scales), scales).to(weight.dtype)
    q = torch.round(groups / scales.float().unsqueeze(-1)) + 8
    q = q.clamp_(0, 15).to(torch.uint8).reshape(n, k // 8, 8)
    nibbles = q[..., list(W4A16_PACK_ORDER)]
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)  # [N, K/8, 4] bytes
    qweight = packed.contiguous().view(torch.int32).reshape(n, k // 8)
    return W4A16Weight(qweight=qweight, scales=scales.contiguous(), group_size=group_size)


def unpack_w4a16(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [N, K // 8] -> uint8 [N, K] values in 0..15, in K order.

    k = 2i sits at bits 4i and k = 2i + 1 at bits 16 + 4i of its word. The
    shifts are built on the device (no host-to-device copy), so this also
    runs inside a CUDA-graph capture.
    """
    n, words = qweight.shape
    k = torch.arange(8, device=qweight.device, dtype=torch.int32)
    shifts = 4 * (k // 2) + 16 * (k % 2)
    nibbles = (qweight.unsqueeze(-1) >> shifts) & 0xF  # arithmetic shift; the mask drops the sign
    return nibbles.to(torch.uint8).reshape(n, words * 8)


def dequantize_w4a16(weight: W4A16Weight, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """(q - 8) * s in fp32, rounded once to ``dtype`` (default: the scales' dtype)."""
    dtype = dtype or weight.dtype
    if weight.qweight.is_cuda and dtype == weight.dtype:
        extension = load_cuda_extension("w4a16")
        if extension is not None:  # pragma: no cover - depends on CUDA
            return extension.dequantize_w4a16(weight.qweight, weight.scales, weight.group_size)
    n, k = weight.shape
    q = unpack_w4a16(weight.qweight).float() - 8
    w = q.reshape(n, k // weight.group_size, weight.group_size) * weight.scales.float().unsqueeze(-1)
    return w.reshape(n, k).to(dtype)


def w4a16_linear_reference(
    x: torch.Tensor, weight: W4A16Weight, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """x @ dequant(weight).T with the dequantized weight and the product in fp32."""
    out = F.linear(x.float(), dequantize_w4a16(weight, torch.float32)).to(x.dtype)
    return out if bias is None else out + bias


def w4a16_linear(
    x: torch.Tensor,
    weight: W4A16Weight,
    bias: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """y = x @ dequant(weight).T (+ bias). ``config`` overrides W4A16_CONFIG keys."""
    k = x.shape[-1]
    if weight.shape[1] != k:
        raise ValueError(f"x has {k} features, the weight expects {weight.shape[1]}")
    rows = x.numel() // k if k else 0
    config = {**W4A16_CONFIG, **(config or {})}
    kernel = _pick_kernel(rows, weight, config)
    if (
        kernel is not None
        and x.is_cuda
        and x.dtype == weight.dtype
        and x.dtype in (torch.bfloat16, torch.float16)
    ):  # pragma: no cover - depends on CUDA
        extension = load_cuda_extension("w4a16")
        if extension is not None:
            flat = x.reshape(rows, k).contiguous()
            if kernel in ("mma", "wide"):
                workspace, counters = w4a16_workspace(x.device, weight.shape[0])
                op = extension.gemm_w4a16_mma if kernel == "mma" else extension.gemm_w4a16_wide
                out = op(
                    flat, weight.qweight, weight.scales, workspace, counters,
                    weight.group_size, config["splits"],
                )
            else:
                out = extension.gemv_w4a16(
                    flat, weight.qweight, weight.scales, weight.group_size, config["rows_per_warp"],
                    config["gemv_prefetch"],
                )
            out = out.view(*x.shape[:-1], weight.shape[0])
            return out if bias is None else out + bias
    return F.linear(x, dequantize_w4a16(weight, x.dtype), bias)


def w4a16_workspace(device: torch.device, out_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed split-K buffers for up to W4A16_MMA_MAX_ROWS x ``out_features``.

    The kernels leave them zeroed, so one pair per device serves every call.
    CUDA graphs keep raw pointers, so a buffer that grows is retired, not
    freed, and growing during a capture is refused: reserve the largest
    out_features first (the model does at load).
    """
    current = _WORKSPACE.get(device)
    if current is not None and current[0].numel() >= W4A16_MMA_MAX_ROWS * out_features:
        return current
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("w4a16_workspace must be reserved before capturing a CUDA graph")
    if current is not None:
        _RETIRED.append(current)
    workspace = torch.zeros(W4A16_MMA_MAX_ROWS * out_features, dtype=torch.float32, device=device)
    counters = torch.zeros(out_features // 16 + 1, dtype=torch.int32, device=device)
    _WORKSPACE[device] = (workspace, counters)
    return workspace, counters


def _pick_kernel(rows: int, weight: W4A16Weight, config: dict) -> Optional[str]:
    """"gemv", "mma", "wide", or None for dequantize + F.linear."""
    kernel = config["kernel"]
    if kernel == "auto":
        if rows <= config["auto_gemv_rows"]:
            kernel = "gemv"
        else:
            kernel = "mma" if rows <= config["auto_mma_rows"] else "wide"
    if kernel in ("mma", "wide") and (weight.group_size != 128 or rows > W4A16_MMA_MAX_ROWS):
        kernel = "gemv"
    if kernel == "gemv" and rows > W4A16_GEMV_MAX_ROWS:
        return None
    return kernel if rows > 0 else None
