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
It needs group_size 128 and N a multiple of 128 (W4A16_TILED_N_MULTIPLE).

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
Tiled weights use gemm_w4a16_tiled (one launch per 64 rows, up to
W4A16_CONFIG["tiled_max_rows"]); see the kernel for its schedule.
Bigger batches dequantize the whole weight and call F.linear: they are
compute-bound, so the extra pass over the weight is a small share of their
time.
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
    # at 128 rows 531 vs 913 us for a layer's four projections, at 256 even).
    "tiled_tile": "auto",
    "tiled_blocks_per_sm": 1,
    "tiled_grid": 0,
    "tiled_max_rows": 128,
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
W4A16_TILED_MAX_LAUNCH_ROWS = 64
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
        return self.qweight.numel() * 4 + self.scales.numel() * self.scales.element_size()


def w4a16_tiled_supported(n: int, k: int, group_size: int = W4A16_GROUP_SIZE) -> bool:
    """Whether a [n, k] weight fits the tiled layout and its kernel."""
    return group_size == 128 and k % 128 == 0 and n % W4A16_TILED_N_MULTIPLE == 0


def quantize_w4a16(
    weight: torch.Tensor, group_size: int = W4A16_GROUP_SIZE, layout: str = "rows"
) -> W4A16Weight:
    """RTN-quantize a [N, K] weight; scales keep the weight's dtype."""
    if weight.ndim != 2:
        raise ValueError("quantize_w4a16 expects a 2-D [out_features, in_features] weight")
    if sys.byteorder != "little":  # pragma: no cover
        raise RuntimeError("the int4 packing assumes a little-endian host")
    if layout not in W4A16_LAYOUTS:
        raise ValueError(f"layout must be one of: {', '.join(W4A16_LAYOUTS)}")
    n, k = weight.shape
    if group_size % 32 or k % group_size:
        raise ValueError(
            f"in_features ({k}) must be a multiple of group_size ({group_size}), "
            "itself a multiple of 32"
        )
    if layout == "tiled" and not w4a16_tiled_supported(n, k, group_size):
        raise ValueError(
            f"the tiled layout needs group_size 128 and out_features a multiple of "
            f"{W4A16_TILED_N_MULTIPLE} (got {n} x {k}, group_size {group_size})"
        )
    groups = weight.float().reshape(n, k // group_size, group_size)
    scales = groups.abs().amax(dim=-1) / 7.5
    # An all-zero group would divide by zero; any scale reproduces it.
    scales = torch.where(scales == 0, torch.ones_like(scales), scales).to(weight.dtype)
    q = torch.round(groups / scales.float().unsqueeze(-1)) + 8
    q = q.clamp_(0, 15).to(torch.uint8).reshape(n, k)
    if layout == "tiled":
        return W4A16Weight(
            qweight=_pack_tiled(q), scales=_tile_scales(scales), group_size=group_size, layout="tiled"
        )
    nibbles = q.reshape(n, k // 8, 8)[..., list(W4A16_PACK_ORDER)]
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)  # [N, K/8, 4] bytes
    qweight = packed.contiguous().view(torch.int32).reshape(n, k // 8)
    return W4A16Weight(qweight=qweight, scales=scales.contiguous(), group_size=group_size)


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


def w4a16_row_scales(weight: W4A16Weight) -> torch.Tensor:
    """The scales as [N, K / group_size] whatever the layout."""
    if weight.layout != "tiled":
        return weight.scales
    groups, tiles = weight.scales.shape[:2]
    return weight.scales.permute(1, 3, 2, 0).reshape(tiles * 64, groups)


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
    """(q - 8) * s in fp32, rounded once to ``dtype`` (default: the scales' dtype)."""
    dtype = dtype or weight.dtype
    if weight.qweight.is_cuda and dtype == weight.dtype:
        extension = load_cuda_extension("w4a16")
        if extension is not None:  # pragma: no cover - depends on CUDA
            if weight.layout == "tiled":
                return extension.dequantize_w4a16_tiled(weight.qweight, weight.scales)
            return extension.dequantize_w4a16(weight.qweight, weight.scales, weight.group_size)
    n, k = weight.shape
    q = unpack_w4a16(weight.qweight).float() - 8
    scales = w4a16_row_scales(weight).float().unsqueeze(-1)
    w = q.reshape(n, k // weight.group_size, weight.group_size) * scales
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
    """gemm_w4a16_tiled, one launch per 64 rows, up to tiled_max_rows rows."""
    n, k = weight.shape
    if (
        x.is_cuda
        and x.dtype == weight.dtype
        and x.dtype in (torch.bfloat16, torch.float16)
        and 0 < rows <= config["tiled_max_rows"]
    ):  # pragma: no cover - depends on CUDA
        extension = load_cuda_extension("w4a16")
        if extension is not None:
            flat = x.reshape(rows, k).contiguous()
            out = torch.empty(rows, n, dtype=x.dtype, device=x.device)
            _, counters = w4a16_workspace(x.device, n)
            for r0 in range(0, rows, W4A16_TILED_MAX_LAUNCH_ROWS):
                r1 = min(rows, r0 + W4A16_TILED_MAX_LAUNCH_ROWS)
                tile_k, tile_n, grid = tiled_launch_config(r1 - r0, n, k, config, _sm_count(x.device))
                extension.gemm_w4a16_tiled(
                    flat[r0:r1], weight.qweight, weight.scales, counters, out[r0:r1], tile_k, tile_n, grid
                )
            out = out.view(*x.shape[:-1], n)
            return out if bias is None else out + bias
    return F.linear(x, dequantize_w4a16(weight, x.dtype), bias)


def w4a16_tiled_trace(
    x: torch.Tensor, weight: W4A16Weight, config: Optional[dict] = None
) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - depends on CUDA
    """One gemm_w4a16_tiled launch (bf16, up to 64 rows) through the trace build.

    Returns (y, trace): trace is int64 [blocks, TILED_TRACE_EVENTS], one row
    per block in the layout of TILED_TRACE_*. y is bit-identical to w4a16_linear's.
    """
    config = {**W4A16_CONFIG, **(config or {})}
    rows, k = x.shape
    n = weight.shape[0]
    if weight.layout != "tiled" or x.dtype != torch.bfloat16 or not 0 < rows <= W4A16_TILED_MAX_LAUNCH_ROWS:
        raise ValueError("the trace build takes a tiled weight and 1 to 64 bf16 rows")
    extension = load_cuda_extension("w4a16")
    if extension is None:
        raise RuntimeError("building kernels/csrc/w4a16.cu failed (see the log)")
    tile_k, tile_n, grid = tiled_launch_config(rows, n, k, config, _sm_count(x.device))
    blocks = len(tiled_schedule(n, k, tile_k, tile_n, grid)["ranges"])
    _, counters = w4a16_workspace(x.device, n)
    out = torch.empty(rows, n, dtype=x.dtype, device=x.device)
    trace = torch.empty(blocks + 1, TILED_TRACE_EVENTS, dtype=torch.int64, device=x.device)
    extension.gemm_w4a16_tiled(
        x.contiguous(), weight.qweight, weight.scales, counters, out, tile_k, tile_n, grid, trace
    )
    return out, trace[:blocks]


def tiled_launch_config(rows: int, n: int, k: int, config: dict, sm_count: int) -> tuple[int, int, int]:
    """(TILE_K, TILE_N, grid) of one gemm_w4a16_tiled launch over ``rows`` <= 64 rows."""
    mt = (rows + 15) // 16
    name = config["tiled_tile"]
    if name == "auto":
        if mt <= 2:
            name = "256x128" if k % 256 == 0 else "128x128"
        else:  # wide slices only pay off when there are many of them (tune.py, run marlin-t1c)
            name = "64x256" if n >= 16384 and n % 256 == 0 else "64x128"
    if name not in _TILED_TILES_BY_MT[mt]:
        raise ValueError(f"tile {name} is not built for {rows} rows (m16 tiles: {mt})")
    tile_k, tile_n = W4A16_TILED_TILES[name]
    if n % tile_n or k % tile_k:
        raise ValueError(f"tile {name} needs a weight divisible by {tile_n} x {tile_k}, got {n} x {k}")
    grid = config["tiled_grid"] or sm_count * config["tiled_blocks_per_sm"]
    return tile_k, tile_n, grid


def tiled_schedule(n: int, k: int, tile_k: int, tile_n: int, grid: int) -> dict:
    """The work split of gemm_w4a16_tiled, computed as the kernel does.

    Tiles are numbered slice * (K / TILE_K) + k tile, a slice being TILE_N
    output columns. Block b takes tiles [b * iters, (b + 1) * iters): a
    stripe that may start or end inside a slice, so every block gets the
    same work whatever the shape. Blocks sharing a slice each store their
    fp32 partial in slot (block + slice) and the last to arrive sums the
    slots in block order. Returns iters, each block's tile range, each
    slice's first and last block, and the number of slots.
    """
    k_tiles, slices = k // tile_k, n // tile_n
    total = k_tiles * slices
    iters = -(-total // grid)
    blocks = -(-total // iters)
    return {
        "iters": iters,
        "ranges": [(b * iters, min((b + 1) * iters, total)) for b in range(blocks)],
        "owners": [(s * k_tiles // iters, (s * k_tiles + k_tiles - 1) // iters) for s in range(slices)],
        "slots": blocks + slices,
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
