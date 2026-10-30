"""The int8 lm_head kernel (kernels/w8a16.py) on the GPU."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels.w8a16 import (  # noqa: E402
    TRITON_AVAILABLE,
    W8A16_DENSE_ROWS,
    quantize_w8a16,
    w8a16_linear,
    w8a16_linear_reference,
)

VOCAB, HIDDEN = 128256, 4096


def _check(rows, n, k, dtype=torch.bfloat16):
    weight = quantize_w8a16((torch.randn(n, k, device="cuda") / k**0.5).to(dtype))
    x = torch.randn(rows, k, device="cuda").to(dtype)
    out = w8a16_linear(x, weight)
    assert out.shape == (rows, n) and out.dtype == dtype
    tol = 1.6e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out.float(), w8a16_linear_reference(x.float(), weight), rtol=tol, atol=tol)


def test_triton_is_available():
    assert TRITON_AVAILABLE


@pytest.mark.parametrize("rows", [1, 3, 16, 17, 64, 65, 100, 256])
def test_lm_head_shape(rows):
    _check(rows, VOCAB, HIDDEN)


@pytest.mark.parametrize("rows,n,k", [(1, 1000, 4000), (40, 1000, 4000), (2048, 512, 4096)])
def test_ragged_shapes_and_many_rows(rows, n, k):
    _check(rows, n, k)


@pytest.mark.parametrize("rows", [W8A16_DENSE_ROWS, W8A16_DENSE_ROWS + 1, 256])
def test_dense_copy_past_the_int8_rows(rows):
    dense = (torch.randn(VOCAB, HIDDEN, device="cuda") / HIDDEN**0.5).bfloat16()
    weight = quantize_w8a16(dense, keep_dense=True)
    x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
    out = w8a16_linear(x, weight)
    if rows > W8A16_DENSE_ROWS:
        assert torch.equal(out, torch.nn.functional.linear(x, dense))
    else:
        torch.testing.assert_close(out.float(), w8a16_linear_reference(x.float(), weight), rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.parametrize("rows", [2, 64])
def test_fp16(rows):
    _check(rows, 4096, 4096, torch.float16)


@pytest.mark.parametrize("rows", [4, 64])
def test_captured_by_a_cuda_graph(rows):
    weight = quantize_w8a16((torch.randn(VOCAB, HIDDEN, device="cuda") / 64).bfloat16())
    x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
    w8a16_linear(x, weight)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w8a16_linear(x, weight)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(out.float(), w8a16_linear_reference(x.float(), weight), rtol=1.6e-2, atol=1.6e-2)
