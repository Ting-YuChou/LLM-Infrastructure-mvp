import itertools
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
from serving.nano_vllm.kernels import w4a16 as w4a16_mod  # noqa: E402
from serving.nano_vllm.kernels.w4a16 import (  # noqa: E402
    W4A16_PACK_ORDER,
    W4A16_TILED_TILES,
    tiled_launch_config,
    tiled_schedule,
    unpack_w4a16,
    w4a16_row_scales,
)
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


def test_tiled_layout_puts_each_code_in_its_mma_fragment():
    """Decode the tiled words with the PTX fragment layout of mma.m16n8k16.

    Lane 4g + c holds B[k][n] for n = g and k = 2c, 2c + 1 (register b0),
    2c + 8, 2c + 9 (b1) of each n8 tile; word i serves tiles 2i and 2i + 1
    and pair s = 0..3 (mask 0x000F000F << 4s) is b0, b1 of tile 2i, then
    b0, b1 of tile 2i + 1, low half first.
    """
    torch.manual_seed(4)
    n, k = 128, 256
    weight = torch.randn(n, k)
    codes = unpack_w4a16(quantize_w4a16(weight).qweight).tolist()
    tiled = quantize_w4a16(weight, layout="tiled")
    assert tiled.layout == "tiled" and tiled.shape == (n, k)
    assert tiled.qweight.shape == (k // 16, n // 64, 32, 4) and tiled.qweight.dtype == torch.int32
    words = (tiled.qweight.to(torch.int64) & 0xFFFFFFFF).tolist()
    for r, t, lane, i, s, h in itertools.product(range(k // 16), range(n // 64), range(32), range(4), range(4), range(2)):
        g, c = divmod(lane, 4)
        col = 64 * t + 8 * (2 * i + s // 2) + g
        row = 16 * r + 2 * c + 8 * (s % 2) + h
        assert (words[r][t][lane][i] >> (4 * s + 16 * h)) & 0xF == codes[col][row]


def test_tiled_scales_give_each_lane_its_columns():
    torch.manual_seed(5)
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rows, tiled = quantize_w4a16(weight), quantize_w4a16(weight, layout="tiled")
    assert tiled.scales.shape == (2, 2, 8, 8) and tiled.scales.dtype == torch.bfloat16
    for group, t, g, j in itertools.product(range(2), range(2), range(8), range(8)):
        assert tiled.scales[group, t, g, j] == rows.scales[64 * t + 8 * j + g, group]
    assert torch.equal(w4a16_row_scales(tiled), rows.scales)


def test_tiled_weights_dequantize_and_multiply_like_rows():
    torch.manual_seed(6)
    weight = torch.randn(256, 384)
    rows, tiled = quantize_w4a16(weight), quantize_w4a16(weight, layout="tiled")
    assert torch.equal(unpack_w4a16(tiled.qweight), unpack_w4a16(rows.qweight))
    assert torch.equal(dequantize_w4a16(tiled), dequantize_w4a16(rows))
    assert tiled.nbytes() == rows.nbytes()
    x = torch.randn(2, 3, 384)
    torch.testing.assert_close(linear(x, tiled, backend="triton"), linear(x, rows, backend="triton"), rtol=0, atol=0)


@pytest.mark.parametrize("n, k, group", [(320, 256, 128), (256, 256, 64)])
def test_tiled_layout_rejects_shapes_its_kernel_cannot_take(n, k, group):
    assert not w4a16_mod.w4a16_tiled_supported(n, k, group)
    with pytest.raises(ValueError, match="tiled layout"):
        quantize_w4a16(torch.randn(n, k), group_size=group, layout="tiled")
    with pytest.raises(ValueError, match="layout"):
        quantize_w4a16(torch.randn(256, 256), layout="marlin")


def test_tiled_launch_config_picks_a_tile_built_for_the_rows():
    config = dict(w4a16_mod.W4A16_CONFIG)
    assert tiled_launch_config(1, 4096, 4096, config, 108) == (256, 128, 108)
    assert tiled_launch_config(32, 4096, 14336, config, 108) == (256, 128, 108)
    assert tiled_launch_config(1, 4096, 384, config, 108) == (128, 128, 108)
    assert tiled_launch_config(33, 4096, 4096, config, 108) == (64, 128, 108)
    assert tiled_launch_config(64, 28672, 4096, config, 108) == (64, 256, 108)
    assert tiled_launch_config(64, 384, 4096, config, 108) == (64, 128, 108)
    assert tiled_launch_config(8, 4096, 4096, {**config, "tiled_blocks_per_sm": 2}, 108)[2] == 216
    assert tiled_launch_config(8, 4096, 4096, {**config, "tiled_grid": 7}, 108)[2] == 7
    assert tiled_launch_config(64, 4096, 4096, {**config, "tiled_tile": "128x128"}, 108) == (128, 128, 108)
    with pytest.raises(ValueError, match="not built"):
        tiled_launch_config(64, 4096, 4096, {**config, "tiled_tile": "256x128"}, 108)
    with pytest.raises(ValueError, match="divisible"):
        tiled_launch_config(8, 384, 4096, {**config, "tiled_tile": "64x256"}, 108)
    with pytest.raises(ValueError, match="divisible"):
        tiled_launch_config(8, 4096, 384, {**config, "tiled_tile": "256x128"}, 108)


@pytest.mark.parametrize("n, k", [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336), (384, 256)])
def test_tiled_schedule_gives_every_tile_to_one_block(n, k):
    """The kernel's stripes cover each (slice, k tile) once, each slice's
    owners are the blocks whose stripes meet it, and the partial-sum slots
    (block + slice) never collide."""
    for tile_k, tile_n in W4A16_TILED_TILES.values():
        if n % tile_n:
            continue
        k_tiles, slices = k // tile_k, n // tile_n
        for grid in (1, 3, 7, slices, 108, 216):
            plan = tiled_schedule(n, k, tile_k, tile_n, grid)
            ranges = plan["ranges"]
            assert len(ranges) <= grid and all(begin < end for begin, end in ranges)
            assert ranges[0][0] == 0 and ranges[-1][1] == k_tiles * slices
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
            owner = [b for b, (begin, end) in enumerate(ranges) for _ in range(begin, end)]
            slots = set()
            for s, (first, last) in enumerate(plan["owners"]):
                assert sorted(set(owner[s * k_tiles:(s + 1) * k_tiles])) == list(range(first, last + 1))
                if grid == slices:
                    assert first == last  # one block per slice: no cross-block reduction
                for b in range(first, last + 1):
                    assert b + s not in slots
                    slots.add(b + s)
            assert max(slots) < plan["slots"]


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


def test_w4a16_model_quantizes_into_the_configured_layout(monkeypatch):
    weights = _weights(_config())
    assert w4a16_mod.W4A16_CONFIG["layout"] == "tiled"
    tiled = _model(weights, weight_quant="w4a16")
    monkeypatch.setitem(w4a16_mod.W4A16_CONFIG, "layout", "rows")
    rows = _model(weights, weight_quant="w4a16")

    layer = tiled.layers[0]
    assert all(w.layout == "tiled" for w in (layer.qkv, layer.o_proj, layer.gate_up, layer.down))
    assert all(w.layout == "rows" for w in (rows.layers[0].qkv, rows.layers[0].down))
    assert layer.qkv.shape == rows.layers[0].qkv.shape
    for actual, expected in zip(_paged_logits(tiled), _paged_logits(rows)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
