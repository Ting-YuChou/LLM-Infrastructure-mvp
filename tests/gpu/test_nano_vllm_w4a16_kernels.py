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
    dequantize_w4a16,
    quantize_w4a16,
    unpack_w4a16,
    w4a16_linear,
)

# Llama 3.1 8B projections (out_features, in_features) plus a row count that
# leaves the last warp's rows partly out of range.
SHAPES = {
    "qkv": (6144, 4096),
    "o_proj": (4096, 4096),
    "down": (4096, 14336),
    "ragged": (300, 1024),
}


def _weight(n, k, dtype, group_size=128):
    w = (torch.randn(n, k, device="cuda") / k**0.5).to(dtype)
    return quantize_w4a16(w, group_size=group_size)


def _reference(x, weight):
    return x.float() @ dequantize_w4a16_torch(weight).T


def dequantize_w4a16_torch(weight):
    n, k = weight.shape
    q = unpack_w4a16(weight.qweight).float() - 8
    g = weight.group_size
    return (q.view(n, k // g, g) * weight.scales.float().unsqueeze(-1)).view(n, k)


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
    "kernel,rows", [("gemv", r) for r in (1, 2, 3, 8, 9, 32)] + [("mma", r) for r in (1, 5, 16, 17, 40, 64)]
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
def test_mma_k_splits_and_dtypes(splits, dtype):
    weight = _weight(300, 2048, dtype)
    x = torch.randn(20, 2048, device="cuda").to(dtype)
    _check(x, weight, kernel="mma", splits=splits)


def test_mma_with_other_group_sizes_falls_back_to_the_gemv():
    weight = _weight(256, 1024, torch.bfloat16, group_size=64)
    x = torch.randn(3, 1024, device="cuda").to(torch.bfloat16)
    _check(x, weight, kernel="mma")


def test_large_batches_dequantize_and_call_f_linear():
    rows = w4a16_mod.W4A16_MMA_MAX_ROWS + 1
    weight = _weight(256, 512, torch.bfloat16)
    x = torch.randn(rows, 512, device="cuda").to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, dequantize_w4a16(weight))
    assert torch.equal(w4a16_linear(x, weight), expected)


@pytest.mark.parametrize("kernel,rows", [("gemv", 4), ("mma", 24)])
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
