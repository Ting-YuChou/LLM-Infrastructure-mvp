"""W4A16 linear layers: 4-bit integer weights, bf16/fp16 activations.

A weight is 4-bit codes q in groups of W4A16_GROUP_SIZE along K, each group
of a row with a scale s and a zero point z: w = (q - z) * s. Two sources:
- AWQ checkpoints (AutoAWQ's GEMM format, awq_unpack): asymmetric, the
  same codes, zero points and scales vLLM serves with awq_marlin;
- quantize_w4a16, at load from a full-precision checkpoint:
  round-to-nearest, symmetric, s = absmax / 7.5 and q = clamp(round(w / s)
  + 8, 0, 15), so z = 8 (GPTQ's symmetric scheme without the error
  feedback). Scales are stored in the activation dtype.

Layout:
- qweight: int32 [N, K // 8]. Each int32 holds 8 consecutive K values of
  one row; nibble p (bits 4p..4p+3) holds k = 8j + W4A16_PACK_ORDER[p].
  The order puts k = 2i and 2i + 1 under the mask 0x000F000F << 4i, which
  is what the CUDA kernel dequantizes two at a time.
- scales: [N, K // group_size] in the activation dtype.

The "tiled" layout (layout="tiled") reorders the same codes for the
Marlin-style kernel, gemm_w4a16_tiled:
- qweight: int32 [K // 16, N // 64, 32, 4]. Each [32, 4] block is the
  mma.m16n8k16 B operand of a 16 x 64 (k x n) tile: lane 4g + c holds
  column g of the eight n8 tiles at k = 2c, 2c + 1, 2c + 8, 2c + 9, so one
  16-byte load gives a lane all its fragments. Word i holds n8 tiles 2i and
  2i + 1; the mask 0x000F000F << 4s picks pair s = (tile 2i, k 2c + {0, 1}),
  (tile 2i, k 2c + 8 + {0, 1}), (tile 2i + 1, k 2c + {0, 1}),
  (tile 2i + 1, k 2c + 8 + {0, 1}) for s = 0..3, i.e. the b0 and b1
  registers of the two tiles, low half first.
- scales: [K // 128, N // 64, 8, 8], [.., g, j] being column 64t + 8j + g,
  so one 16-byte load gives lane (g, c) the scales of its eight columns.
- zeros: int32 [K // 128, N // 64, 8, 2], [.., g, w] holding the zero
  points of columns 64t + 8j + g, j = 4w + b, in byte b. The kernel turns a
  code into magic + q by OR-ing it into the mantissa of magic (128.0 in
  bf16, 1024.0 in fp16, both with an ulp of 1); one byte permute of z and
  magic's high byte gives magic + z in both halves, so one subtraction
  leaves q - z exactly. One byte per zero point, half of a bf16 scale.
  Symmetric weights store z = 8.
It needs group_size 128 and N a multiple of 128 (W4A16_TILED_N_MULTIPLE).
Row-layout weights are symmetric only (zeros None).

The model quantizes into the tiled layout (W4A16_CONFIG["layout"]); a
weight that does not fit it stays in rows. Decode-sized batches of
row-layout weights use csrc/w4a16.cu, which has three kernels over the same
qweight:
- gemv: CUDA cores, 8 rows of x per launch (looped up to
  W4A16_GEMV_MAX_ROWS); fp32 FMAs limit it to small batches;
- mma: tensor cores (mma.m16n8k16), up to W4A16_MMA_MAX_ROWS rows in one
  pass over the weight, each block reading x from L2; needs group_size 128;
- wide: tensor cores for 17-64 rows, warps of a block share one copy of x
  in shared memory, scales folded into the dequantized weights.
The tensor-core kernels split K across blocks when the weight is short; the
splits meet in a persistent fp32 workspace (w4a16_workspace).
Tiled weights use gemm_w4a16_tiled up to W4A16_CONFIG["tiled_max_rows"]
rows, one launch taking past 64 rows several chunks of rows; see the kernel
for its schedule. Past that (prefill chunks) gemm_w4a16_large takes them,
up to W4A16_LARGE_MAX_ROWS rows per launch: the same schedule over chunks
of 128 or 256 rows whose warps split rows and columns instead of K, so each
dequantized fragment feeds four m16 tiles. Row-layout weights past their
kernels' limits, and anything without the extension, dequantize the whole
weight and call F.linear.
"""

from __future__ import annotations

import functools
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
# AutoAWQ (version "gemm") packs 8 output columns per int32: nibble p of word
# j holds column 8j + AWQ_PACK_ORDER[p]; AWQ_UNPACK_ORDER inverts it.
AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
# The GEMV takes 8 rows of x per launch; up to here the op loops over 8-row
# chunks (each chunk re-reads the 4-bit weight).
W4A16_GEMV_MAX_ROWS = 32
W4A16_MMA_MAX_ROWS = 64
# Launch settings, from benchmarks/phase1/tune.py on an A100 (LOG.md, run 5).
# kernel: "gemv", "mma", "wide", or "auto" (gemv up to auto_gemv_rows, mma
# up to auto_mma_rows, then wide); splits: K splits of the tensor-core
# kernels, 0 lets the op choose from the grid size.
W4A16_CONFIG = {
    "kernel": "auto",
    "auto_gemv_rows": 2,
    "auto_mma_rows": 16,
    "rows_per_warp": 2,
    "splits": 0,
    # The layout the model quantizes into ("rows" or "tiled"); weights the
    # tiled kernel cannot take stay in rows. Tiled since run marlin-t2:
    # decode 5.35 / 7.31 / 16.82 ms at b=1/16/64 vs 5.70 / 8.29 / 23.03.
    "layout": "tiled",
    # gemm_w4a16_tiled: "auto" or one of W4A16_TILED_TILES; blocks per SM of
    # the persistent grid, or an explicit grid size (tests force schedules);
    # up to tiled_max_rows rows it beats dequantize + F.linear (run marlin-t1c:
    # at 128 rows 531 vs 913 us for a layer's four projections; at 256 rows
    # in decode, run concur-c1, dequantizing took 13.4 ms of a 50 ms step;
    # kbench awq-a1: at 384 / 512 rows 911 / 1,225 vs 1,167 / 1,386 us).
    "tiled_tile": "auto",
    "tiled_blocks_per_sm": 1,
    "tiled_grid": 0,
    "tiled_max_rows": 512,
    # gemm_w4a16_large past tiled_max_rows instead of dequantize + F.linear.
    # Off: in run awq-a1 it ran at ~182 TFLOPs at 1,024-2,048 rows against
    # cuBLAS's ~257, so dequantizing first was 5-20% faster there (vLLM's
    # Marlin too was slower than dequantize + cuBLAS). "auto" or one of
    # W4A16_LARGE_TILES.
    "large": False,
    "large_tile": "auto",
}
W4A16_LAYOUTS = ("rows", "tiled")
# (TILE_K, TILE_N) the tiled kernel is built for, by m16 tiles (16-row
# groups of x): wide K tiles while x is small, wide N tiles once the fp32
# accumulators and x tiles grow (shared memory).
W4A16_TILED_TILES = {
    "128x128": (128, 128),
    "256x128": (256, 128),
    "64x256": (64, 256),
    "64x128": (64, 128),
}
_TILED_TILES_BY_MT = {
    1: ("128x128", "256x128", "64x256"),
    2: ("128x128", "256x128", "64x256"),
    3: ("64x256", "64x128", "128x128"),
    4: ("64x256", "64x128", "128x128"),
}
W4A16_TILED_N_MULTIPLE = 128
# gemm_w4a16_large's (rows per chunk, TILE_N), csrc/w4a16.cu's LargeShape,
# and rows per launch (kLargeMaxRows: chunks x slices counters fit
# w4a16_workspace's N / 16).
W4A16_LARGE_TILES = {"128x256": (128, 256), "256x128": (256, 128)}
W4A16_LARGE_MAX_ROWS = 2048
# Rows of one chunk at most (16 x the largest m16 tile count built), and of
# one launch (csrc/w4a16.cu's kTiledMaxChunkRows, kTiledMaxRows).
W4A16_TILED_MAX_CHUNK_ROWS = 64
W4A16_TILED_MAX_ROWS = 512
# The trace build's row per block (csrc/w4a16.cu, kTrace*): stamps by index.
TILED_TRACE_EVENTS = 24
TILED_TRACE_FINISHES = 4  # slice ends recorded per block, 4 stamps each
TILED_TRACE_GT_ENTRY, TILED_TRACE_CLK_ENTRY, TILED_TRACE_CLK_FIRST, TILED_TRACE_TILES = 0, 1, 2, 3
TILED_TRACE_FINISH0 = 4  # start, warps summed, counter bumped or y written, done
TILED_TRACE_FLAGS = 20  # bit f: last to arrive at finish f; bit 8 + f: sole owner of that slice;
# bit 16 + f: last, and saw so before storing its slot (no slot, no atomic)
TILED_TRACE_CLK_EXIT, TILED_TRACE_GT_EXIT, TILED_TRACE_TICKET = 21, 22, 23
_WORKSPACE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_RETIRED: list[tuple[torch.Tensor, torch.Tensor]] = []


@dataclass(frozen=True)
class W4A16Weight:
    """A quantized [out_features, in_features] weight."""

    qweight: torch.Tensor
    scales: torch.Tensor
    group_size: int = W4A16_GROUP_SIZE
    layout: str = "rows"
    zeros: Optional[torch.Tensor] = None  # tiled only: packed zero points (module docstring)

    @property
    def shape(self) -> torch.Size:
        if self.layout == "tiled":
            return torch.Size((self.qweight.shape[1] * 64, self.qweight.shape[0] * 16))
        return torch.Size((self.qweight.shape[0], self.qweight.shape[1] * 8))

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.scales.dtype

    def nbytes(self) -> int:
        total = self.qweight.numel() * 4 + self.scales.numel() * self.scales.element_size()
        if self.zeros is not None:
            total += self.zeros.numel() * self.zeros.element_size()
        return total


def w4a16_tiled_supported(n: int, k: int, group_size: int = W4A16_GROUP_SIZE) -> bool:
    """Whether a [n, k] weight fits the tiled layout and its kernel."""
    return group_size == 128 and k % 128 == 0 and n % W4A16_TILED_N_MULTIPLE == 0


def quantize_w4a16(
    weight: torch.Tensor, group_size: int = W4A16_GROUP_SIZE, layout: str = "rows"
) -> W4A16Weight:
    """RTN-quantize a [N, K] weight; scales keep the weight's dtype."""
    if weight.ndim != 2:
        raise ValueError("quantize_w4a16 expects a 2-D [out_features, in_features] weight")
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
    q = q.clamp_(0, 15).to(torch.uint8).reshape(n, k)
    return pack_w4a16(q, scales, group_size=group_size, layout=layout)


def pack_w4a16(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    group_size: int = W4A16_GROUP_SIZE,
    layout: str = "rows",
) -> W4A16Weight:
    """uint8 codes [N, K] in 0..15, scales [N, K / group_size] and zero points
    (uint8, same shape; None for symmetric, z = 8) -> a W4A16Weight in
    ``layout``. Scales keep their dtype."""
    if sys.byteorder != "little":  # pragma: no cover
        raise RuntimeError("the int4 packing assumes a little-endian host")
    if layout not in W4A16_LAYOUTS:
        raise ValueError(f"layout must be one of: {', '.join(W4A16_LAYOUTS)}")
    n, k = codes.shape
    if scales.shape != (n, k // group_size) or (zeros is not None and zeros.shape != scales.shape):
        raise ValueError(f"scales and zeros must be [{n}, {k // group_size}] for a {n} x {k} weight")
    if layout == "rows":
        if zeros is not None:
            raise ValueError("zero points need the tiled layout (row-layout kernels are symmetric)")
        nibbles = codes.reshape(n, k // 8, 8)[..., list(W4A16_PACK_ORDER)]
        packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)  # [N, K/8, 4] bytes
        qweight = packed.contiguous().view(torch.int32).reshape(n, k // 8)
        return W4A16Weight(qweight=qweight, scales=scales.contiguous(), group_size=group_size)
    if not w4a16_tiled_supported(n, k, group_size):
        raise ValueError(
            f"the tiled layout needs group_size 128 and out_features a multiple of "
            f"{W4A16_TILED_N_MULTIPLE} (got {n} x {k}, group_size {group_size})"
        )
    if zeros is None:
        zeros = torch.full(scales.shape, 8, dtype=torch.uint8, device=scales.device)
    return W4A16Weight(
        qweight=_pack_tiled(codes),
        scales=_tile_scales(scales),
        group_size=group_size,
        layout="tiled",
        zeros=_tile_zeros(zeros),
    )


def awq_unpack(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One AutoAWQ GEMM-format projection -> (codes, zeros, scales) in pack_w4a16's
    [N, K] / [N, K / group_size] orientation.

    AutoAWQ stores the transpose of nn.Linear: qweight int32 [K, N / 8] and
    qzeros int32 [K / group_size, N / 8], 8 columns per word (AWQ_PACK_ORDER),
    and scales [K / group_size, N]; w = (q - z) * s.
    """
    def columns(packed: torch.Tensor) -> torch.Tensor:  # int32 [R, N / 8] -> uint8 [R, N]
        shifts = 4 * torch.arange(8, device=packed.device, dtype=torch.int32)
        nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF  # arithmetic shift; the mask drops the sign
        return nibbles[..., list(AWQ_UNPACK_ORDER)].to(torch.uint8).reshape(packed.shape[0], -1)

    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("AWQ qweight and qzeros must be int32")
    k, words = qweight.shape
    if qzeros.shape[1] != words or scales.shape != (qzeros.shape[0], words * 8) or k % qzeros.shape[0]:
        raise ValueError(
            f"AWQ tensors do not match: qweight {tuple(qweight.shape)}, qzeros "
            f"{tuple(qzeros.shape)}, scales {tuple(scales.shape)}"
        )
    return columns(qweight).t().contiguous(), columns(qzeros).t().contiguous(), scales.t().contiguous()


def _pack_tiled(q: torch.Tensor) -> torch.Tensor:
    """uint8 codes [N, K] -> int32 [K/16, N/64, 32, 4].

    Code (n, k), n = 64t + 8(2i + jj) + g and k = 16r + 8b + 2c + h, sits in
    word [r, t, 4g + c, i] at nibble 4h + 2jj + b (module docstring). The
    views split n and k into those factors: (t, i, jj, g, r, b, c, h) for
    [N, K], (r, t, g, c, i, h, jj, b) for the words.
    """
    n, k = q.shape
    t = q.view(n // 64, 4, 2, 8, k // 16, 2, 4, 2)
    t = t.permute(4, 0, 3, 6, 1, 7, 2, 5).reshape(k // 16, n // 64, 32, 4, 8)
    packed = t[..., 0::2] | (t[..., 1::2] << 4)  # bytes: nibbles (0, 1), (2, 3), (4, 5), (6, 7)
    return packed.contiguous().view(torch.int32).reshape(k // 16, n // 64, 32, 4)


def _unpack_tiled(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [K/16, N/64, 32, 4] -> uint8 codes [N, K]."""
    rows16, tiles = qweight.shape[:2]
    shifts = 4 * torch.arange(8, device=qweight.device, dtype=torch.int32)
    nibbles = ((qweight.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    t = nibbles.view(rows16, tiles, 8, 4, 4, 2, 2, 2)
    return t.permute(1, 4, 6, 2, 0, 7, 3, 5).reshape(tiles * 64, rows16 * 16)


def _tile_scales(scales: torch.Tensor) -> torch.Tensor:
    """[N, K/128] -> [K/128, N/64, 8 (g), 8 (j)], column 64t + 8j + g."""
    n, groups = scales.shape
    return scales.view(n // 64, 8, 8, groups).permute(3, 0, 2, 1).contiguous()


def _tile_zeros(zeros: torch.Tensor) -> torch.Tensor:
    """uint8 [N, K/128] -> int32 [K/128, N/64, 8 (g), 2 (w)]: column
    64t + 8(4w + b) + g in byte b of word [.., g, w]."""
    n, groups = zeros.shape
    z = zeros.view(n // 64, 2, 4, 8, groups).permute(4, 0, 3, 1, 2)  # [G, T, g, w, b]
    return z.contiguous().view(torch.int32).reshape(groups, n // 64, 8, 2)  # little-endian bytes


def _untile_zeros(tiled: torch.Tensor) -> torch.Tensor:
    """int32 [K/128, N/64, 8, 2] -> uint8 [N, K/128] (inverse of _tile_zeros)."""
    groups, tiles = tiled.shape[:2]
    z = tiled.contiguous().view(torch.uint8).view(groups, tiles, 8, 2, 4)  # [G, T, g, w, b]
    return z.permute(1, 3, 4, 2, 0).reshape(tiles * 64, groups).clone()


def _untile_scales(tiled: torch.Tensor) -> torch.Tensor:
    """[K/128, N/64, 8 (g), 8 (j)] -> [N, K/128] (inverse of _tile_scales)."""
    groups, tiles = tiled.shape[:2]
    return tiled.permute(1, 3, 2, 0).reshape(tiles * 64, groups)


def w4a16_row_scales(weight: W4A16Weight) -> torch.Tensor:
    """The scales as [N, K / group_size] whatever the layout."""
    if weight.layout != "tiled":
        return weight.scales
    return _untile_scales(weight.scales)


def w4a16_row_zeros(weight: W4A16Weight) -> torch.Tensor:
    """The zero points z as fp32 [N, K / group_size] whatever the layout."""
    if weight.zeros is None:
        n, k = weight.shape
        return torch.full((n, k // weight.group_size), 8.0, device=weight.device)
    return _untile_zeros(weight.zeros).float()


def unpack_w4a16(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [N, K // 8] (rows) or [K // 16, N // 64, 32, 4] (tiled) -> uint8 [N, K]
    values in 0..15, in K order.

    Rows: k = 2i sits at bits 4i and k = 2i + 1 at bits 16 + 4i of its word.
    The shifts are built on the device (no host-to-device copy), so this also
    runs inside a CUDA-graph capture.
    """
    if qweight.ndim == 4:
        return _unpack_tiled(qweight)
    n, words = qweight.shape
    k = torch.arange(8, device=qweight.device, dtype=torch.int32)
    shifts = 4 * (k // 2) + 16 * (k % 2)
    nibbles = (qweight.unsqueeze(-1) >> shifts) & 0xF  # arithmetic shift; the mask drops the sign
    return nibbles.to(torch.uint8).reshape(n, words * 8)


def dequantize_w4a16(weight: W4A16Weight, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """(q - z) * s in fp32, rounded once to ``dtype`` (default: the scales' dtype)."""
    dtype = dtype or weight.dtype
    if weight.qweight.is_cuda and dtype == weight.dtype:
        extension = load_cuda_extension("w4a16")
        if extension is not None:  # pragma: no cover - depends on CUDA
            if weight.layout == "tiled":
                return extension.dequantize_w4a16_tiled(weight.qweight, weight.scales, weight.zeros)
            return extension.dequantize_w4a16(weight.qweight, weight.scales, weight.group_size)
    n, k = weight.shape
    groups = k // weight.group_size
    q = unpack_w4a16(weight.qweight).float().reshape(n, groups, weight.group_size)
    w = (q - w4a16_row_zeros(weight).unsqueeze(-1)) * w4a16_row_scales(weight).float().unsqueeze(-1)
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
    if weight.layout == "tiled":
        return _tiled_linear(x, weight, bias, rows, config)
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
                    flat, weight.qweight, weight.scales, weight.group_size, config["rows_per_warp"]
                )
            out = out.view(*x.shape[:-1], weight.shape[0])
            return out if bias is None else out + bias
    return F.linear(x, dequantize_w4a16(weight, x.dtype), bias)


def _tiled_linear(
    x: torch.Tensor, weight: W4A16Weight, bias: Optional[torch.Tensor], rows: int, config: dict
) -> torch.Tensor:
    """gemm_w4a16_tiled up to tiled_max_rows rows (one launch), gemm_w4a16_large past that."""
    n, k = weight.shape
    tiled = 0 < rows <= min(config["tiled_max_rows"], W4A16_TILED_MAX_ROWS)
    if (
        x.is_cuda
        and x.dtype == weight.dtype
        and x.dtype in (torch.bfloat16, torch.float16)
        and (tiled or (rows > 0 and config["large"]))
    ):  # pragma: no cover - depends on CUDA
        extension = load_cuda_extension("w4a16")
        if extension is not None:
            flat = x.reshape(rows, k).contiguous()
            out = torch.empty(rows, n, dtype=x.dtype, device=x.device)
            _, counters = w4a16_workspace(x.device, n)
            args = (weight.qweight, weight.scales, weight.zeros, counters)
            if tiled:
                tile_k, tile_n, grid = tiled_launch_config(rows, n, k, config, _sm_count(x.device))
                extension.gemm_w4a16_tiled(flat, *args, out, tile_k, tile_n, grid)
            else:
                for start in range(0, rows, W4A16_LARGE_MAX_ROWS):
                    stop = min(start + W4A16_LARGE_MAX_ROWS, rows)
                    tile_m, tile_n, grid = large_launch_config(stop - start, n, config, _sm_count(x.device))
                    extension.gemm_w4a16_large(flat[start:stop], *args, out[start:stop], tile_m, tile_n, grid)
            out = out.view(*x.shape[:-1], n)
            return out if bias is None else out + bias
    return F.linear(x, dequantize_w4a16(weight, x.dtype), bias)


def large_launch_config(rows: int, n: int, config: dict, sm_count: int) -> tuple[int, int, int]:
    """(rows per chunk, TILE_N, grid) of one gemm_w4a16_large launch. "auto"
    takes the tile whose chunks of rows keep the most blocks busy (each
    chunk gets grid // chunks of the persistent grid), 128x256 on a tie."""
    grid = config["tiled_grid"] or sm_count * config["tiled_blocks_per_sm"]
    name = config["large_tile"]
    if name == "auto":
        fits = [t for t, (_, tile_n) in W4A16_LARGE_TILES.items() if n % tile_n == 0]
        if not fits:
            raise ValueError(f"no gemm_w4a16_large tile divides N = {n}")

        def busy(tile: str) -> int:
            chunks = -(-rows // W4A16_LARGE_TILES[tile][0])
            return chunks * max(1, grid // chunks) if chunks <= grid else grid

        name = max(fits, key=lambda t: (busy(t), t == "128x256"))
    tile_m, tile_n = W4A16_LARGE_TILES[name]
    if n % tile_n:
        raise ValueError(f"tile {name} needs N divisible by {tile_n}, got {n}")
    return tile_m, tile_n, grid


def w4a16_tiled_trace(
    x: torch.Tensor, weight: W4A16Weight, config: Optional[dict] = None
) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - depends on CUDA
    """One gemm_w4a16_tiled launch (bf16) through the trace build.

    Returns (y, trace): trace is int64 [blocks, TILED_TRACE_EVENTS], one row
    per block in the layout of TILED_TRACE_*, chunk by chunk past 64 rows
    (tiled_schedule's "blocks"). y is bit-identical to w4a16_linear's.
    """
    config = {**W4A16_CONFIG, **(config or {})}
    rows, k = x.shape
    n = weight.shape[0]
    if weight.layout != "tiled" or x.dtype != torch.bfloat16 or not 0 < rows <= W4A16_TILED_MAX_ROWS:
        raise ValueError(f"the trace build takes a tiled weight and 1 to {W4A16_TILED_MAX_ROWS} bf16 rows")
    extension = load_cuda_extension("w4a16")
    if extension is None:
        raise RuntimeError("building kernels/csrc/w4a16.cu failed (see the log)")
    tile_k, tile_n, grid = tiled_launch_config(rows, n, k, config, _sm_count(x.device))
    blocks = tiled_schedule(n, k, tile_k, tile_n, grid, tiled_chunk_rows(rows)[0])["blocks"]
    _, counters = w4a16_workspace(x.device, n)
    out = torch.empty(rows, n, dtype=x.dtype, device=x.device)
    trace = torch.empty(blocks + 1, TILED_TRACE_EVENTS, dtype=torch.int64, device=x.device)
    extension.gemm_w4a16_tiled(
        x.contiguous(), weight.qweight, weight.scales, weight.zeros, counters, out, tile_k, tile_n, grid, trace
    )
    return out, trace[:blocks]


def tiled_chunk_rows(rows: int) -> tuple[int, int]:
    """(chunks, rows per chunk) of a gemm_w4a16_tiled launch over ``rows``: up
    to 64 rows one chunk, past that the fewest 64-row chunks, evened out to
    whole m16 tiles (96 rows: two of 48; csrc/w4a16.cu's tiled_mt)."""
    chunks = -(-rows // W4A16_TILED_MAX_CHUNK_ROWS)
    chunk_rows = 16 * -(-rows // (16 * chunks))
    return -(-rows // chunk_rows), chunk_rows


def tiled_launch_config(rows: int, n: int, k: int, config: dict, sm_count: int) -> tuple[int, int, int]:
    """(TILE_K, TILE_N, grid) of one gemm_w4a16_tiled launch over ``rows`` rows;
    the tile is built for the rows of one chunk, the grid is shared by the chunks."""
    chunks, chunk_rows = tiled_chunk_rows(rows)
    mt = chunk_rows // 16
    name = config["tiled_tile"]
    if name == "auto":
        if mt <= 2:
            name = "256x128" if k % 256 == 0 else "128x128"
        else:  # wide slices pay off when there are many of them, or from three chunks of rows
            # (tune.py, runs marlin-t1c, down-d2e, chunk-c2, chunk-c5)
            name = "64x256" if n % 256 == 0 and (n >= 16384 or chunks >= 3) else "128x128"
    if name not in _TILED_TILES_BY_MT[mt]:
        raise ValueError(f"tile {name} is not built for {rows} rows (m16 tiles: {mt})")
    tile_k, tile_n = W4A16_TILED_TILES[name]
    if n % tile_n or k % tile_k:
        raise ValueError(f"tile {name} needs a weight divisible by {tile_n} x {tile_k}, got {n} x {k}")
    grid = config["tiled_grid"] or sm_count * config["tiled_blocks_per_sm"]
    return tile_k, tile_n, grid


def tiled_schedule(n: int, k: int, tile_k: int, tile_n: int, grid: int, chunks: int = 1) -> dict:
    """The work split of gemm_w4a16_tiled, computed as the kernel does.

    Tiles are numbered slice * (K / TILE_K) + k tile, a slice being TILE_N
    output columns. Block b takes tiles [b * iters, (b + 1) * iters): a
    stripe that may start or end inside a slice, so every block gets the
    same work whatever the shape. Blocks sharing a slice each store their
    fp32 partial in slot (block + slice) and the last to arrive sums the
    slots in block order. Each chunk of rows (tiled_chunk_rows) gets
    grid // chunks blocks with this same split. Returns iters, each block's
    tile range, each slice's first and last block, and the number of slots,
    all per chunk, and the blocks of the launch.
    """
    k_tiles, slices = k // tile_k, n // tile_n
    total = k_tiles * slices
    iters = -(-total // max(1, grid // chunks))
    blocks = -(-total // iters)
    return {
        "iters": iters,
        "ranges": [(b * iters, min((b + 1) * iters, total)) for b in range(blocks)],
        "owners": [(s * k_tiles // iters, (s * k_tiles + k_tiles - 1) // iters) for s in range(slices)],
        "slots": blocks + slices,
        "blocks": chunks * blocks,
    }


@functools.lru_cache(maxsize=None)
def _sm_count(device: torch.device) -> int:  # pragma: no cover - depends on CUDA
    return torch.cuda.get_device_properties(device).multi_processor_count


def w4a16_workspace(device: torch.device, out_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed split-K buffers for up to W4A16_MMA_MAX_ROWS x ``out_features``.

    The kernels leave them zeroed, so one pair per device serves every call.
    gemm_w4a16_tiled uses only the counters (one per output slice).
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
