import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.kernels import (  # noqa: E402
    W4A16Weight,
    dequantize_w4a16,
    linear,
    quantize_w4a16,
    w4a16_linear_reference,
)
from serving.nano_vllm.kernels.w4a16 import W4A16_PACK_ORDER, unpack_w4a16  # noqa: E402
from serving.nano_vllm.model import NanoLlamaConfig, NanoLlamaForCausalLM  # noqa: E402

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _expected_codes(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """RTN codes computed element by element from the definition."""
    n, k = weight.shape
    groups = weight.reshape(n, k // group_size, group_size)
    scales = (groups.abs().amax(-1) / 7.5).to(weight.dtype).float()
    return (torch.round(groups / scales.unsqueeze(-1)) + 8).clamp(0, 15).reshape(n, k).to(torch.uint8)


def test_quantize_packs_rtn_codes_in_the_kernel_order():
    torch.manual_seed(0)
    weight = torch.randn(6, 256)
    quantized = quantize_w4a16(weight, group_size=64)

    assert quantized.qweight.dtype == torch.int32
    assert quantized.qweight.shape == (6, 32)
    assert quantized.scales.shape == (6, 4)
    assert quantized.shape == (6, 256)
    codes = _expected_codes(weight, 64)
    torch.testing.assert_close(unpack_w4a16(quantized.qweight), codes, rtol=0, atol=0)
    # Nibble p of word j holds k = 8j + PACK_ORDER[p]: k = 2i and 2i + 1 sit
    # under the mask 0x000F000F << 4i.
    raw = quantized.qweight.to(torch.int64) & 0xFFFFFFFF
    for p, k in enumerate(W4A16_PACK_ORDER):
        nibble = (raw >> (4 * p)) & 0xF
        torch.testing.assert_close(nibble.to(torch.uint8), codes[:, k::8], rtol=0, atol=0)


def test_dequantize_is_within_half_a_step_of_the_weight():
    torch.manual_seed(1)
    weight = torch.randn(32, 512) * 0.05
    quantized = quantize_w4a16(weight)
    restored = dequantize_w4a16(quantized)

    step = quantized.scales.repeat_interleave(128, dim=1)
    error = (restored - weight).abs()
    # Half a step from rounding; the group's largest value may clamp at
    # +7 instead of +7.5 steps, so allow a whole step there.
    assert (error <= step * 0.5 + 1e-7).float().mean() > 0.99
    assert (error <= step + 1e-7).all()


def test_all_zero_group_stays_zero():
    weight = torch.randn(2, 256)
    weight[0, :128] = 0
    restored = dequantize_w4a16(quantize_w4a16(weight))
    assert torch.equal(restored[0, :128], torch.zeros(128))


def test_bf16_scales_keep_the_weight_dtype():
    quantized = quantize_w4a16(torch.randn(4, 128, dtype=torch.bfloat16))
    assert quantized.scales.dtype == torch.bfloat16
    assert quantized.dtype == torch.bfloat16


@pytest.mark.parametrize("k, group", [(100, 128), (256, 48), (192, 128)])
def test_quantize_rejects_shapes_the_kernel_cannot_read(k, group):
    with pytest.raises(ValueError, match="multiple"):
        quantize_w4a16(torch.randn(4, k), group_size=group)


def test_linear_dispatches_quantized_weights_on_cpu():
    torch.manual_seed(2)
    x = torch.randn(3, 5, 256)
    quantized = quantize_w4a16(torch.randn(64, 256))
    bias = torch.randn(64)

    out = linear(x, quantized, bias, backend="triton")

    expected = x @ dequantize_w4a16(quantized).T + bias
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w4a16_linear_reference(x, quantized, bias), expected, rtol=1e-5, atol=1e-5)


def _config() -> NanoLlamaConfig:
    return NanoLlamaConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def _weights(config: NanoLlamaConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    hidden, inter = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * config.head_dim
    shapes = {
        "q_proj": (hidden, hidden), "k_proj": (kv, hidden), "v_proj": (kv, hidden),
        "o_proj": (hidden, hidden), "gate_proj": (inter, hidden), "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }
    weights = {
        "model.embed_tokens.weight": torch.randn(config.vocab_size, hidden) * 0.5,
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(config.vocab_size, hidden) * 0.1,
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden)
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden)
        for name, shape in shapes.items():
            module = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            weights[f"{prefix}.{module}.{name}.weight"] = torch.randn(*shape) * 0.1
    return weights


def _model(weights, weight_quant="none") -> NanoLlamaForCausalLM:
    return NanoLlamaForCausalLM(
        config=_config(),
        weights=weights,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_triton=False,
        weight_quant=weight_quant,
    )


def _paged_logits(model: NanoLlamaForCausalLM) -> tuple[torch.Tensor, torch.Tensor]:
    cache = model.new_paged_cache(num_blocks=4, block_size=4)
    prefill = model.forward_paged(
        input_ids=torch.tensor([3, 9, 27, 4, 11]),
        positions=torch.arange(5),
        cache=cache,
        slot_mapping=torch.arange(5),
        block_tables=torch.tensor([[0, 1]]),
        context_lens=torch.tensor([5]),
        query_lens=[5],
        is_prefill=True,
    )
    decode = model.forward_paged(
        input_ids=torch.tensor([8]),
        positions=torch.tensor([5]),
        cache=cache,
        slot_mapping=torch.tensor([5]),
        block_tables=torch.tensor([[0, 1]]),
        context_lens=torch.tensor([6]),
        query_lens=[1],
        is_prefill=False,
    )
    return prefill, decode


def test_w4a16_model_matches_a_model_holding_the_dequantized_weights():
    weights = _weights(_config())
    quantized = _model(weights, weight_quant="w4a16")
    # RTN groups run along each row, so quantizing q/k/v (gate/up) one by one
    # gives the same codes as quantizing the fused projection.
    dequantized = dict(weights)
    for name, tensor in weights.items():
        if name.endswith(tuple(f"{p}.weight" for p in PROJECTIONS)):
            dequantized[name] = dequantize_w4a16(quantize_w4a16(tensor))
    reference = _model(dequantized)

    layer = quantized.layers[0]
    assert all(isinstance(w, W4A16Weight) for w in (layer.qkv, layer.o_proj, layer.gate_up, layer.down))
    assert isinstance(quantized.lm_head_weight, torch.Tensor)
    for actual, expected in zip(_paged_logits(quantized), _paged_logits(reference)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    ids = torch.tensor([[3, 9, 27, 4]])
    torch.testing.assert_close(
        quantized.forward(ids, quantized.new_cache(1, 8)),
        reference.forward(ids, reference.new_cache(1, 8)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_unknown_weight_quant_is_rejected():
    with pytest.raises(ValueError, match="weight_quant"):
        _model(_weights(_config()), weight_quant="int8")
    with pytest.raises(ValueError, match="weight_quant"):
        NanoVLLMConfig(weight_quant="awq")


def test_weight_quant_comes_from_yaml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "nano.yaml"
    path.write_text("engine:\n  weight_quant: w4a16\n")
    assert NanoVLLMConfig.from_yaml(str(path)).weight_quant == "w4a16"
    monkeypatch.setenv("WEIGHT_QUANT", "none")
    assert NanoVLLMConfig.from_yaml(str(path)).weight_quant == "none"
