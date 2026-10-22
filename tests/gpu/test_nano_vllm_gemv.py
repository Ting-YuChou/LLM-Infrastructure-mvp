"""The Triton and CUDA GEMVs (linear_backend="triton" / "cuda") vs an fp32 matmul."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch.nn.functional as F  # noqa: E402

from serving.nano_vllm.kernels.linear import TRITON_AVAILABLE, linear  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="requires CUDA and Triton"
)

# Llama 3.1 8B decode shapes (out_features, in_features) plus ragged ones.
SHAPES = {
    "qkv": (6144, 4096),
    "o_proj": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
    "ragged": (100, 1000),
}


def _check(rows, out_features, in_features, split_k=None, backend="triton"):
    x = torch.randn(rows, in_features, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(out_features, in_features, device="cuda") / in_features**0.5).to(
        torch.bfloat16
    )
    out = linear(x, weight, backend=backend, split_k=split_k)
    expected = x.float() @ weight.float().T
    assert out.shape == (rows, out_features) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.parametrize("rows", [1, 2, 5, 16])
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES)
def test_gemv_matches_fp32_matmul(shape, rows):
    _check(rows, *shape)


@pytest.mark.parametrize("split_k", [1, 2, 8])
def test_gemv_split_k_variants_agree(split_k):
    _check(3, 4096, 14336, split_k=split_k)


def test_lm_head_shape():
    _check(4, 128256, 4096)


def test_more_than_16_rows_falls_back_to_f_linear():
    x = torch.randn(17, 4096, device="cuda").to(torch.bfloat16)
    weight = torch.randn(4096, 4096, device="cuda").to(torch.bfloat16)

    assert torch.equal(linear(x, weight, backend="triton"), F.linear(x, weight))


# CUDA: in_features must be a multiple of 8, so "ragged" uses 1000 -> 1000.
CUDA_SHAPES = {**{k: v for k, v in SHAPES.items() if k != "ragged"}, "ragged": (100, 1000)}


def test_cuda_extension_builds():
    from serving.nano_vllm.kernels.linear import load_cuda_extension

    assert load_cuda_extension() is not None, "see the log for the nvcc error"


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("shape", CUDA_SHAPES.values(), ids=CUDA_SHAPES)
def test_cuda_gemv_matches_fp32_matmul(shape, rows):
    _check(rows, *shape, backend="cuda")
