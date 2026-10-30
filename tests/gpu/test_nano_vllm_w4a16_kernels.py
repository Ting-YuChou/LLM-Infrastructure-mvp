"""The W4A16 CUDA kernels (csrc/w4a16.cu) vs the PyTorch dequantization."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels import w4a16 as w4a16_mod  # noqa: E402
from serving.nano_vllm.kernels.cuda_ext import load_cuda_extension  # noqa: E402
from serving.nano_vllm.kernels.w4a16 import (  # noqa: E402
    TILED_TRACE_CLK_ENTRY,
    TILED_TRACE_CLK_EXIT,
    TILED_TRACE_CLK_FIRST,
    TILED_TRACE_FINISH0,
    TILED_TRACE_FINISHES,
    TILED_TRACE_FLAGS,
    TILED_TRACE_GT_ENTRY,
    TILED_TRACE_GT_EXIT,
    TILED_TRACE_TICKET,
    TILED_TRACE_TILES,
    W4A16_CONFIG,
    W4A16_TILED_TILES,
    dequantize_w4a16,
    pack_w4a16,
    quantize_w4a16,
    tiled_chunk_rows,
    tiled_launch_config,
    tiled_schedule,
    unpack_w4a16,
    w4a16_linear,
    w4a16_row_scales,
    w4a16_row_zeros,
    w4a16_tiled_trace,
)

# Llama 3.1 8B projections (out_features, in_features) plus a row count that
# leaves the last warp's rows partly out of range.
SHAPES = {
    "qkv": (6144, 4096),
    "o_proj": (4096, 4096),
    "down": (4096, 14336),
    "ragged": (300, 1024),
}


# The tiled kernel needs N % 128 == 0; n384 only fits TILE_N 128.
TILED_SHAPES = {
    "qkv": (6144, 4096),
    "o_proj": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
    "n384": (384, 1024),
}


def _weight(n, k, dtype, group_size=128, layout="rows"):
    w = (torch.randn(n, k, device="cuda") / k**0.5).to(dtype)
    return quantize_w4a16(w, group_size=group_size, layout=layout)


def _awq_weight(n, k, dtype):
    """Asymmetric codes and zero points, as an AWQ checkpoint has (tiled)."""
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device="cuda")
    zeros = torch.randint(0, 16, (n, k // 128), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(n, k // 128, device="cuda") * 2 / (15 * k**0.5)).to(dtype)
    return pack_w4a16(codes, scales, zeros, layout="tiled")


def _reference(x, weight):
    return x.float() @ dequantize_w4a16_torch(weight).T


def dequantize_w4a16_torch(weight):
    n, k = weight.shape
    g = weight.group_size
    q = unpack_w4a16(weight.qweight).float().view(n, k // g, g) - w4a16_row_zeros(weight).unsqueeze(-1)
    return (q * w4a16_row_scales(weight).float().unsqueeze(-1)).view(n, k)


def test_w4a16_extension_builds():
    assert load_cuda_extension("w4a16") is not None, "see the log for the nvcc error"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantize_on_gpu_matches_cpu(dtype):
    w = torch.randn(64, 512).to(dtype)
    on_cpu = quantize_w4a16(w)
    on_gpu = quantize_w4a16(w.cuda())
    assert torch.equal(on_gpu.qweight.cpu(), on_cpu.qweight)
    assert torch.equal(on_gpu.scales.cpu(), on_cpu.scales)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [128, 64])
def test_dequantize_kernel_matches_torch(dtype, group_size):
    weight = _weight(300, 1024, dtype, group_size)
    expected = dequantize_w4a16_torch(weight).to(dtype)
    assert torch.equal(dequantize_w4a16(weight), expected)


def _check(x, weight, **config):
    out = w4a16_linear(x, weight, config=config)
    assert out.shape == (x.shape[0], weight.shape[0]) and out.dtype == x.dtype
    tol = 1.6e-2 if x.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out.float(), _reference(x, weight), rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "kernel,rows",
    [("gemv", r) for r in (1, 2, 3, 8, 9, 32)]
    + [("mma", r) for r in (1, 5, 16, 17, 40, 64)]
    + [("wide", r) for r in (1, 17, 33, 48, 64)],
)
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES)
def test_kernels_match_dequantized_matmul(shape, kernel, rows):
    weight = _weight(*shape, torch.bfloat16)
    x = torch.randn(rows, shape[1], device="cuda").to(torch.bfloat16)
    _check(x, weight, kernel=kernel)


@pytest.mark.parametrize("rows_per_warp", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemv_launch_variants_and_dtypes(rows_per_warp, dtype):
    weight = _weight(300, 1024, dtype, group_size=64)
    x = torch.randn(5, 1024, device="cuda").to(dtype)
    _check(x, weight, kernel="gemv", rows_per_warp=rows_per_warp)


@pytest.mark.parametrize("splits", [0, 1, 2, 3, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kernel", ["mma", "wide"])
def test_tensor_core_k_splits_and_dtypes(kernel, splits, dtype):
    weight = _weight(300, 2048, dtype)
    x = torch.randn(20, 2048, device="cuda").to(dtype)
    _check(x, weight, kernel=kernel, splits=splits)


def test_split_k_leaves_the_workspace_zeroed():
    """The last split re-zeroes what it summed, so calls need no memset."""
    weight = _weight(4096, 4096, torch.bfloat16)
    for kernel, rows in (("mma", 8), ("wide", 40)):
        x = torch.randn(rows, 4096, device="cuda").to(torch.bfloat16)
        for _ in range(3):
            _check(x, weight, kernel=kernel, splits=4)
        workspace, counters = w4a16_mod.w4a16_workspace(x.device, 4096)
        torch.cuda.synchronize()
        assert not workspace.any() and not counters.any()


@pytest.mark.parametrize("kernel", ["mma", "wide"])
def test_tensor_cores_with_other_group_sizes_fall_back_to_the_gemv(kernel):
    weight = _weight(256, 1024, torch.bfloat16, group_size=64)
    x = torch.randn(3, 1024, device="cuda").to(torch.bfloat16)
    _check(x, weight, kernel=kernel)


def test_large_batches_dequantize_and_call_f_linear():
    rows = w4a16_mod.W4A16_MMA_MAX_ROWS + 1
    weight = _weight(256, 512, torch.bfloat16)
    x = torch.randn(rows, 512, device="cuda").to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, dequantize_w4a16(weight))
    assert torch.equal(w4a16_linear(x, weight), expected)


@pytest.mark.parametrize("kernel,rows", [("gemv", 4), ("mma", 12), ("wide", 40)])
def test_kernels_are_captured_by_a_cuda_graph(kernel, rows):
    weight = _weight(4096, 4096, torch.bfloat16)
    x = torch.randn(rows, 4096, device="cuda").to(torch.bfloat16)
    config = {"kernel": kernel}
    w4a16_linear(x, weight, config=config)  # warm-up outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w4a16_linear(x, weight, config=config)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(out.float(), _reference(x, weight), rtol=1.6e-2, atol=1.6e-2)


# --- gemm_w4a16_tiled (tiled layout) ----------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tiled_dequantize_kernel_matches_the_row_kernel_bit_for_bit(dtype):
    w = (torch.randn(384, 1024, device="cuda") / 32).to(dtype)
    rows, tiled = quantize_w4a16(w), quantize_w4a16(w, layout="tiled")
    assert torch.equal(dequantize_w4a16(tiled), dequantize_w4a16(rows))
    assert torch.equal(dequantize_w4a16(tiled), dequantize_w4a16_torch(rows).to(dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tiled_dequantize_kernel_with_zero_points_matches_torch(dtype):
    """(q - z) is exact, times the scale in fp32, rounded once: bit for bit."""
    weight = _awq_weight(384, 1024, dtype)
    assert torch.equal(dequantize_w4a16(weight), dequantize_w4a16_torch(weight).to(dtype))


@pytest.mark.parametrize("rows", [1, 7, 16, 33, 64, 100, 256, 512])
@pytest.mark.parametrize("shape", TILED_SHAPES.values(), ids=TILED_SHAPES)
def test_tiled_kernel_with_zero_points_matches_dequantized_matmul(shape, rows):
    weight = _awq_weight(*shape, torch.bfloat16)
    x = torch.randn(rows, shape[1], device="cuda").to(torch.bfloat16)
    _check(x, weight, tiled_max_rows=512)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,tile", [(5, "128x128"), (5, "256x128"), (5, "64x256"), (40, "64x128"), (40, "64x256"), (100, "128x128")]
)
def test_tiled_kernel_with_zero_points_in_every_tile_and_dtype(rows, tile, dtype):
    """Up to MT 2 the zero points are decoded once per tile, past that once per row."""
    weight = _awq_weight(1024, 2048, dtype)
    x = torch.randn(rows, 2048, device="cuda").to(dtype)
    _check(x, weight, tiled_tile=tile, tiled_max_rows=512)
    _check(x, weight, tiled_tile=tile, tiled_grid=5, tiled_max_rows=512)


@pytest.mark.parametrize("rows", [1, 2, 7, 16, 17, 33, 48, 64, 65, 96, 100, 130, 192, 256, 512])
@pytest.mark.parametrize("shape", TILED_SHAPES.values(), ids=TILED_SHAPES)
def test_tiled_kernel_matches_dequantized_matmul(shape, rows):
    weight = _weight(*shape, torch.bfloat16, layout="tiled")
    x = torch.randn(rows, shape[1], device="cuda").to(torch.bfloat16)
    _check(x, weight, tiled_max_rows=512)  # past 64 rows: one launch over chunks of rows


@pytest.mark.parametrize("grid", ["one_block_per_slice", 1, 3, 7, 0])
@pytest.mark.parametrize(
    "rows,tile",
    [(5, "128x128"), (5, "256x128"), (5, "64x256"), (20, "128x128"), (20, "256x128"), (40, "64x256"),
     (40, "64x128"), (40, "128x128"), (64, "64x256"), (64, "128x128"), (96, "64x256"), (100, "128x128"),
     (128, "64x128"), (200, "64x256"), (256, "128x128")],
)
def test_tiled_kernel_schedules_and_tiles(rows, tile, grid):
    """grid = slices gives each block one slice (no cross-block sum, the
    layout alone); 1, 3, 7 and the default make stripes that cross slices
    and slices shared by several blocks. Past 64 rows each chunk of rows
    gets grid // chunks blocks (at least one)."""
    n, k = 1024, 2048
    if grid == "one_block_per_slice":
        grid = n // W4A16_TILED_TILES[tile][1] * tiled_chunk_rows(rows)[0]
    weight = _weight(n, k, torch.bfloat16, layout="tiled")
    x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
    _check(x, weight, tiled_tile=tile, tiled_grid=grid, tiled_max_rows=512)


@pytest.mark.parametrize("rows", [3, 40])
def test_tiled_kernel_in_fp16(rows):
    weight = _weight(1024, 2048, torch.float16, layout="tiled")
    x = torch.randn(rows, 2048, device="cuda").to(torch.float16)
    _check(x, weight)
    _check(x, weight, tiled_grid=5)


def test_tiled_kernel_is_deterministic_and_leaves_the_buffers_zeroed():
    """The slots are summed in block order, so repeated calls agree bit for
    bit; the counters return to zero and the row kernels' workspace is never
    touched."""
    weight = _weight(4096, 14336, torch.bfloat16, layout="tiled")
    for rows in (1, 16, 64, 128, 256):
        x = torch.randn(rows, 14336, device="cuda").to(torch.bfloat16)
        outs = [w4a16_linear(x, weight, config={"tiled_max_rows": 512}) for _ in range(3)]
        assert all(torch.equal(outs[0], out) for out in outs[1:])
    workspace, counters = w4a16_mod.w4a16_workspace(x.device, 4096)
    torch.cuda.synchronize()
    assert not workspace.any() and not counters.any()


def test_tiled_large_batches_dequantize_and_call_f_linear():
    rows = w4a16_mod.W4A16_CONFIG["tiled_max_rows"] + 1
    weight = _weight(256, 512, torch.bfloat16, layout="tiled")
    x = torch.randn(rows, 512, device="cuda").to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, dequantize_w4a16(weight))
    assert torch.equal(w4a16_linear(x, weight), expected)


@pytest.mark.parametrize("rows", [4, 40, 100, 256])
def test_tiled_kernel_is_captured_by_a_cuda_graph(rows):
    weight = _weight(4096, 4096, torch.bfloat16, layout="tiled")
    x = torch.randn(rows, 4096, device="cuda").to(torch.bfloat16)
    config = {"tiled_max_rows": 512}
    w4a16_linear(x, weight, config=config)  # warm-up outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w4a16_linear(x, weight, config=config)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(out.float(), _reference(x, weight), rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.parametrize(
    "shape,rows,tile,grid",
    [((4096, 14336), 1, "256x128", 0), ((4096, 14336), 8, "128x128", 0), ((4096, 14336), 40, "64x128", 0),
     ((28672, 4096), 64, "64x256", 0), ((1024, 2048), 5, "256x128", 3), ((1024, 2048), 5, "128x128", 8),
     ((4096, 14336), 128, "128x128", 0), ((1024, 2048), 100, "64x256", 7)],
)
def test_tiled_trace_build(shape, rows, tile, grid):
    """The trace build gives the same y bit for bit, and its stamps follow the
    schedule: tiles per block, ordered stamps, one last arrival per shared
    slice, a sole-owner flag for unshared ones, exit tickets 0..blocks-1.
    Past 64 rows the rows of the trace go chunk by chunk, each chunk on the
    same schedule."""
    n, k = shape
    weight = _weight(n, k, torch.bfloat16, layout="tiled")
    x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
    config = {"tiled_tile": tile, "tiled_grid": grid, "tiled_max_rows": 512}
    out, trace = w4a16_tiled_trace(x, weight, config)
    assert torch.equal(out, w4a16_linear(x, weight, config=config))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tile_k, tile_n, grid = tiled_launch_config(rows, n, k, {**W4A16_CONFIG, **config}, sms)
    chunks = tiled_chunk_rows(rows)[0]
    schedule = tiled_schedule(n, k, tile_k, tile_n, grid, chunks)
    k_tiles = k // tile_k
    t = trace.cpu().tolist()
    assert len(t) == schedule["blocks"] == chunks * len(schedule["ranges"])
    assert sorted(row[TILED_TRACE_TICKET] for row in t) == list(range(len(t)))
    for c in range(chunks):
        _check_trace_chunk(t[c * len(schedule["ranges"]) : (c + 1) * len(schedule["ranges"])], schedule, k_tiles)


def _check_trace_chunk(t, schedule, k_tiles):
    finish_of = {}  # (block, slice) -> stamps index f
    for b, ((begin, end), row) in enumerate(zip(schedule["ranges"], t)):
        assert row[TILED_TRACE_TILES] == end - begin
        assert row[TILED_TRACE_GT_ENTRY] <= row[TILED_TRACE_GT_EXIT]
        stamps = [row[TILED_TRACE_CLK_ENTRY], row[TILED_TRACE_CLK_FIRST]]
        first_slice, last_slice = begin // k_tiles, (end - 1) // k_tiles
        for f in range(min(TILED_TRACE_FINISHES, last_slice - first_slice + 1)):
            stamps += row[TILED_TRACE_FINISH0 + 4 * f : TILED_TRACE_FINISH0 + 4 * f + 4]
            finish_of[(b, first_slice + f)] = f
        stamps.append(row[TILED_TRACE_CLK_EXIT])
        assert all(stamps) and stamps == sorted(stamps), (b, stamps)
    for s, (lo, hi) in enumerate(schedule["owners"]):
        if any((b, s) not in finish_of for b in range(lo, hi + 1)):
            continue  # past the recorded slice ends of some block
        flags = [(t[b][TILED_TRACE_FLAGS] >> finish_of[(b, s)]) & 0x101 for b in range(lo, hi + 1)]
        if lo == hi:
            assert flags == [0x100], (s, flags)
        else:
            assert sorted(flags) == [0] * (hi - lo) + [1], (s, flags)
