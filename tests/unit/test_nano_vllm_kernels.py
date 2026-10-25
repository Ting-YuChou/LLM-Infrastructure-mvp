import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

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


def _hf_llama_config(**overrides):
    from transformers import LlamaConfig

    values = dict(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=500000.0,
        max_position_embeddings=256,
    )
    values.update(overrides)
    return LlamaConfig(**values)


@pytest.mark.parametrize(
    "rope_scaling",
    [
        None,
        {"rope_type": "linear", "factor": 4.0},
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8,
        },
    ],
)
def test_rope_matches_transformers_rotate_half_layout(rope_scaling):
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    from serving.nano_vllm.kernels.rotary import compute_inv_freq

    config = _hf_llama_config(rope_scaling=rope_scaling)
    head_dim = config.hidden_size // config.num_attention_heads
    torch.manual_seed(0)
    query = torch.randn(1, 4, 6, head_dim)
    key = torch.randn(1, 2, 6, head_dim)
    positions = torch.tensor([0, 1, 2, 7, 30, 100])

    rotary = LlamaRotaryEmbedding(config)
    cos, sin = rotary(query, positions.view(1, -1))
    expected_query, expected_key = apply_rotary_pos_emb(query, key, cos, sin)

    # 500000.0 as set in _hf_llama_config; transformers 5 has no config.rope_theta.
    inv_freq = compute_inv_freq(head_dim, 500000.0, rope_scaling)
    torch.testing.assert_close(inv_freq, rotary.inv_freq, rtol=0, atol=0)
    actual_query, actual_key = apply_rope(
        query, key, positions, theta=500000.0, inv_freq=inv_freq
    )
    torch.testing.assert_close(actual_query, expected_query, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_key, expected_key, rtol=1e-5, atol=1e-6)


def test_llama3_rope_scaling_changes_frequencies():
    from serving.nano_vllm.kernels.rotary import compute_inv_freq

    scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8,
    }
    plain = compute_inv_freq(16, 500000.0)
    scaled = compute_inv_freq(16, 500000.0, scaling)

    # Llama 3 scaling only ever lowers frequencies (stretches wavelengths).
    assert torch.all(scaled <= plain)
    assert not torch.allclose(scaled, plain)


def test_unsupported_rope_scaling_raises_clear_error():
    from serving.nano_vllm.kernels.rotary import compute_inv_freq

    with pytest.raises(ValueError, match="yarn"):
        compute_inv_freq(16, 10000.0, {"rope_type": "yarn", "factor": 4.0})
    with pytest.raises(ValueError, match="low_freq_factor"):
        compute_inv_freq(16, 10000.0, {"rope_type": "llama3", "factor": 8.0})


def test_rms_norm_reference_handles_large_fp16_inputs():
    # Squaring ~1000 in fp16 overflows (max ~65504); variance must be fp32.
    x = torch.full((2, 64), 1000.0, dtype=torch.float16)
    x[:, ::2] = -1000.0
    weight = torch.ones(64, dtype=torch.float16)

    out = rms_norm_reference(x, weight)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.abs().float(), torch.ones(2, 64), atol=1e-3, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not rmsnorm_triton_available(),
    reason="requires CUDA and Triton",
)
def test_triton_rms_norm_handles_large_fp16_inputs():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float16) * 1000.0
    weight = torch.randn(128, device="cuda", dtype=torch.float16)

    actual = rms_norm(x, weight, use_triton=True)
    expected = rms_norm_reference(x, weight)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_fused_add_rms_norm_reference_adds_then_normalizes():
    from serving.nano_vllm.kernels import fused_add_rms_norm

    torch.manual_seed(0)
    x, residual = torch.randn(3, 16), torch.randn(3, 16)
    weight = torch.randn(16)

    normed, new_residual = fused_add_rms_norm(x, residual, weight, eps=1e-5)

    torch.testing.assert_close(new_residual, residual + x)
    torch.testing.assert_close(normed, rms_norm_reference(residual + x, weight, eps=1e-5))


def test_silu_and_mul_reference_splits_gate_and_up():
    from serving.nano_vllm.kernels import silu_and_mul

    torch.manual_seed(0)
    gate_up = torch.randn(3, 2 * 8)

    out = silu_and_mul(gate_up)

    torch.testing.assert_close(out, silu_mul_reference(gate_up[:, :8], gate_up[:, 8:]))
