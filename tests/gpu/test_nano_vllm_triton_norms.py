"""Triton RMSNorm, fused add + RMSNorm and silu_and_mul vs the references."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels import (  # noqa: E402
    fused_add_rms_norm,
    fused_add_rms_norm_reference,
    rms_norm,
    rms_norm_reference,
    rmsnorm_triton_available,
    silu_and_mul,
    silu_and_mul_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not rmsnorm_triton_available(),
    reason="requires CUDA and Triton",
)

# One bf16 ulp at |x| ~ 4 is 2**-5; differences come from reduction order.
BF16 = dict(rtol=1.6e-2, atol=3.2e-2)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_rms_norm_matches_hf_rounding_in_bf16(rows):
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 3
    weight = (1 + 0.1 * torch.randn(4096, device="cuda")).to(torch.bfloat16)

    torch.testing.assert_close(
        rms_norm(x, weight, eps=1e-5, use_triton=True),
        rms_norm_reference(x, weight, eps=1e-5),
        **BF16,
    )


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_fused_add_rms_norm_matches_reference(rows):
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 4
    weight = (1 + 0.1 * torch.randn(4096, device="cuda")).to(torch.bfloat16)
    x_before, residual_before = x.clone(), residual.clone()

    normed, new_residual = fused_add_rms_norm(x, residual, weight, eps=1e-5, use_triton=True)
    expected_normed, expected_residual = fused_add_rms_norm_reference(x, residual, weight, eps=1e-5)

    torch.testing.assert_close(new_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_normed, **BF16)
    assert torch.equal(x, x_before) and torch.equal(residual, residual_before)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_silu_and_mul_matches_reference(rows):
    gate_up = torch.randn(rows, 2 * 14336, device="cuda", dtype=torch.bfloat16) * 3

    torch.testing.assert_close(
        silu_and_mul(gate_up, use_triton=True), silu_and_mul_reference(gate_up), **BF16
    )
