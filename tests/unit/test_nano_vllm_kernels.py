import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels import (  # noqa: E402
    apply_rope,
    causal_attention,
    rms_norm,
    rms_norm_reference,
    rmsnorm_triton_available,
    silu_mul,
    silu_mul_reference,
)


def test_rms_norm_pytorch_fallback_matches_reference():
    x = torch.randn(3, 5, 8)
    weight = torch.randn(8)

    actual = rms_norm(x, weight, eps=1e-6, use_triton=False)
    expected = rms_norm_reference(x, weight, eps=1e-6)

    torch.testing.assert_close(actual, expected)


def test_silu_mul_pytorch_fallback_matches_reference():
    gate = torch.randn(2, 4, 8)
    up = torch.randn(2, 4, 8)

    torch.testing.assert_close(silu_mul(gate, up), silu_mul_reference(gate, up))


def test_rope_preserves_shape_and_changes_positions():
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)

    rotated_query, rotated_key = apply_rope(query, key, torch.arange(3))

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    assert not torch.allclose(rotated_query[:, :, 1], query[:, :, 1])


def test_causal_attention_masks_future_keys():
    query = torch.ones(1, 1, 2, 2)
    key = torch.ones(1, 1, 3, 2)
    value = torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [100.0, 0.0]]]])

    output = causal_attention(query, key, value, torch.tensor([0, 1]))

    assert output.shape == (1, 1, 2, 2)
    torch.testing.assert_close(output[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert output[0, 0, 1, 0] < 100.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not rmsnorm_triton_available(),
    reason="requires CUDA and Triton",
)
def test_triton_rms_norm_matches_pytorch_reference():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)

    actual = rms_norm(x, weight, use_triton=True)
    expected = rms_norm_reference(x, weight)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
