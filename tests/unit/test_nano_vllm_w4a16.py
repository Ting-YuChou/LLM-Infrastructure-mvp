import contextlib
import dataclasses
import itertools
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm import model as model_mod  # noqa: E402
from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.kernels import (  # noqa: E402
    W4A16Weight,
    dequantize_w4a16,
    linear,
    quantize_w4a16,
    quantize_w8a16,
    w4a16_linear_reference,
)
from serving.nano_vllm.kernels import w4a16 as w4a16_mod  # noqa: E402
from serving.nano_vllm.kernels.w4a16 import (  # noqa: E402
    W4A16_PACK_ORDER,
    W4A16_TILED_TILES,
    awq_unpack,
    pack_w4a16,
    tiled_chunk_rows,
    tiled_launch_config,
    tiled_schedule,
    unpack_w4a16,
    w4a16_row_scales,
    w4a16_row_zeros,
)
from tiny_llama import AWQ_QUANT_CONFIG, awq_checkpoint, awq_pack_columns  # noqa: E402

from serving.nano_vllm.model import (  # noqa: E402
    NanoLlamaConfig,
    NanoLlamaForCausalLM,
    resolve_weight_quant,
)

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
    # Tiled weights also store their zero points (8), a byte per scale.
    assert tiled.nbytes() == rows.nbytes() + rows.scales.numel()
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
    assert tiled_launch_config(33, 4096, 4096, config, 108) == (128, 128, 108)
    assert tiled_launch_config(64, 28672, 4096, config, 108) == (64, 256, 108)
    assert tiled_launch_config(64, 384, 4096, config, 108) == (128, 128, 108)
    assert tiled_launch_config(8, 4096, 4096, {**config, "tiled_blocks_per_sm": 2}, 108)[2] == 216
    assert tiled_launch_config(8, 4096, 4096, {**config, "tiled_grid": 7}, 108)[2] == 7
    assert tiled_launch_config(64, 4096, 4096, {**config, "tiled_tile": "64x128"}, 108) == (64, 128, 108)
    with pytest.raises(ValueError, match="not built"):
        tiled_launch_config(64, 4096, 4096, {**config, "tiled_tile": "256x128"}, 108)
    with pytest.raises(ValueError, match="divisible"):
        tiled_launch_config(8, 384, 4096, {**config, "tiled_tile": "64x256"}, 108)
    with pytest.raises(ValueError, match="divisible"):
        tiled_launch_config(8, 4096, 384, {**config, "tiled_tile": "256x128"}, 108)
    # Past 64 rows the tile is the one built for a chunk's rows.
    assert tiled_launch_config(96, 4096, 14336, config, 108) == (128, 128, 108)  # two chunks of 48
    assert tiled_launch_config(128, 4096, 14336, config, 108) == (128, 128, 108)
    assert tiled_launch_config(192, 4096, 14336, config, 108) == (64, 256, 108)  # from three chunks
    assert tiled_launch_config(256, 28672, 4096, config, 108) == (64, 256, 108)
    with pytest.raises(ValueError, match="not built"):
        tiled_launch_config(128, 4096, 4096, {**config, "tiled_tile": "256x128"}, 108)


@pytest.mark.parametrize(
    "rows, chunks, chunk_rows",
    [(1, 1, 16), (17, 1, 32), (64, 1, 64), (65, 2, 48), (96, 2, 48), (100, 2, 64), (128, 2, 64),
     (130, 3, 48), (160, 3, 64), (192, 3, 64), (256, 4, 64), (512, 8, 64)],
)
def test_tiled_chunk_rows(rows, chunks, chunk_rows):
    assert tiled_chunk_rows(rows) == (chunks, chunk_rows)
    assert (chunks - 1) * chunk_rows < rows <= chunks * chunk_rows and chunk_rows % 16 == 0


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


def test_tiled_schedule_splits_the_grid_between_chunks():
    """Each chunk of rows stripes the whole weight over grid // chunks blocks."""
    n, k, tile_k, tile_n = 4096, 14336, 128, 128
    one = tiled_schedule(n, k, tile_k, tile_n, 54)
    two = tiled_schedule(n, k, tile_k, tile_n, 108, chunks=2)
    assert {key: two[key] for key in one if key != "blocks"} == {key: v for key, v in one.items() if key != "blocks"}
    assert one["blocks"] == len(one["ranges"]) and two["blocks"] == 2 * one["blocks"]
    assert tiled_schedule(n, k, tile_k, tile_n, 3, chunks=4)["iters"] == (k // tile_k) * (n // tile_n)


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


def _model(weights, weight_quant="none", config=None) -> NanoLlamaForCausalLM:
    return NanoLlamaForCausalLM(
        config=config or _config(),
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
        NanoVLLMConfig(weight_quant="gptq")
    assert NanoVLLMConfig(weight_quant="awq").weight_quant == "awq"


def test_weight_quant_comes_from_yaml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "nano.yaml"
    path.write_text("engine:\n  weight_quant: w4a16\n")
    assert NanoVLLMConfig.from_yaml(str(path)).weight_quant == "w4a16"
    monkeypatch.setenv("WEIGHT_QUANT", "none")
    assert NanoVLLMConfig.from_yaml(str(path)).weight_quant == "none"


AWQ_CONFIG = AWQ_QUANT_CONFIG
_awq_pack_columns = awq_pack_columns
_awq_checkpoint = awq_checkpoint


def test_awq_unpack_inverts_autoawq_packing():
    torch.manual_seed(5)
    n, k = 384, 256
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    zeros = torch.randint(0, 16, (n, k // 128), dtype=torch.uint8)
    scales = torch.rand(n, k // 128).half()
    got_codes, got_zeros, got_scales = awq_unpack(
        _awq_pack_columns(codes.t()), _awq_pack_columns(zeros.t()), scales.t().contiguous()
    )
    torch.testing.assert_close(got_codes, codes, rtol=0, atol=0)
    torch.testing.assert_close(got_zeros, zeros, rtol=0, atol=0)
    torch.testing.assert_close(got_scales, scales, rtol=0, atol=0)
    # Column 8j + 1 is nibble 4 (bits 16..19) of word j.
    one = torch.zeros(1, 8, dtype=torch.uint8)
    one[0, 1] = 0xA
    assert awq_unpack(torch.tensor([[0xA << 16]], dtype=torch.int32), torch.zeros(1, 1, dtype=torch.int32),
                      torch.ones(1, 8))[0].t().tolist() == one.tolist()


def test_awq_unpack_of_packed_words_concatenated_along_n_equals_concatenated_parts():
    torch.manual_seed(6)
    parts = [torch.randint(0, 16, (n, 128), dtype=torch.uint8) for n in (128, 64, 64)]
    zeros = [torch.randint(0, 16, (n, 1), dtype=torch.uint8) for n in (128, 64, 64)]
    scales = [torch.rand(n, 1) for n in (128, 64, 64)]
    fused = awq_unpack(
        torch.cat([_awq_pack_columns(c.t()) for c in parts], dim=1),
        torch.cat([_awq_pack_columns(z.t()) for z in zeros], dim=1),
        torch.cat([s.t() for s in scales], dim=1),
    )
    for got, expected in zip(fused, (torch.cat(parts), torch.cat(zeros), torch.cat(scales))):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_tiled_zero_points_dequantize_to_q_minus_z_times_s(dtype):
    torch.manual_seed(7)
    n, k = 256, 384
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    zeros = torch.randint(0, 16, (n, k // 128), dtype=torch.uint8)
    scales = torch.rand(n, k // 128).to(dtype)
    weight = pack_w4a16(codes, scales, zeros, layout="tiled")

    assert weight.zeros.shape == (k // 128, n // 64, 8, 2) and weight.zeros.dtype == torch.int32
    # Column 64t + 8(4w + b) + g is byte b of word [group, t, g, w].
    raw = weight.zeros.to(torch.int64) & 0xFFFFFFFF
    for group, t, g, j in itertools.product(range(k // 128), range(n // 64), range(8), range(8)):
        word = int(raw[group, t, g, j // 4])
        assert (word >> (8 * (j % 4))) & 0xFF == int(zeros[64 * t + 8 * j + g, group])
    torch.testing.assert_close(w4a16_row_zeros(weight), zeros.float(), rtol=0, atol=0)
    q = codes.float().reshape(n, k // 128, 128)
    expected = ((q - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)).reshape(n, k)
    torch.testing.assert_close(dequantize_w4a16(weight, torch.float32), expected, rtol=0, atol=0)
    assert weight.nbytes() == codes.numel() // 2 + scales.numel() * (scales.element_size() + 1)


def test_rtn_tiled_weights_store_zero_point_8_and_match_rows():
    torch.manual_seed(8)
    weight = torch.randn(256, 256).bfloat16()
    tiled = quantize_w4a16(weight, layout="tiled")
    rows = quantize_w4a16(weight)
    assert rows.zeros is None
    assert torch.all(tiled.zeros == torch.tensor(0x08080808, dtype=torch.int32))
    torch.testing.assert_close(w4a16_row_zeros(tiled), w4a16_row_zeros(rows), rtol=0, atol=0)
    torch.testing.assert_close(dequantize_w4a16(tiled), dequantize_w4a16(rows), rtol=0, atol=0)


def test_zero_points_need_the_tiled_layout():
    codes = torch.zeros(128, 128, dtype=torch.uint8)
    with pytest.raises(ValueError, match="tiled"):
        pack_w4a16(codes, torch.ones(128, 1), torch.zeros(128, 1, dtype=torch.uint8), layout="rows")


def test_awq_checkpoint_matches_a_model_holding_the_dequantized_weights():
    weights = _weights(_config())
    awq, dequantized = _awq_checkpoint(weights)
    config = dataclasses.replace(_config(), quantization_config=AWQ_CONFIG)
    model = _model(awq, weight_quant="none", config=config)  # an AWQ checkpoint loads as awq
    reference = _model(dequantized)

    assert model.weight_quant == "awq"
    layer = model.layers[0]
    assert all(w.layout == "tiled" and w.zeros is not None for w in (layer.qkv, layer.o_proj, layer.gate_up, layer.down))
    assert isinstance(model.lm_head_weight, torch.Tensor)
    for actual, expected in zip(_paged_logits(model), _paged_logits(reference)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_weight_quant_resolution_against_the_checkpoint():
    plain = _config()
    awq = dataclasses.replace(plain, quantization_config=AWQ_CONFIG)
    assert resolve_weight_quant("none", plain) == "none"
    assert resolve_weight_quant("w4a16", plain) == "w4a16"
    assert resolve_weight_quant("none", awq) == "awq"
    assert resolve_weight_quant("awq", awq) == "awq"
    with pytest.raises(ValueError, match="needs an AWQ checkpoint"):
        resolve_weight_quant("awq", plain)
    with pytest.raises(ValueError, match="already AWQ"):
        resolve_weight_quant("w4a16", awq)
    for change in ({"bits": 8}, {"group_size": 64}, {"zero_point": False}, {"version": "gemv"},
                   {"quant_method": "gptq"}):
        with pytest.raises(ValueError, match="GEMM format"):
            resolve_weight_quant("none", dataclasses.replace(plain, quantization_config={**AWQ_CONFIG, **change}))


def test_awq_model_needs_every_projection_in_awq_form():
    awq, _ = _awq_checkpoint(_weights(_config()))
    config = dataclasses.replace(_config(), quantization_config=AWQ_CONFIG)
    del awq["model.layers.1.mlp.down_proj.qzeros"]
    with pytest.raises(KeyError, match="down_proj.qzeros"):
        _model(awq, weight_quant="awq", config=config)


def test_cuda_load_builds_each_weight_on_a_side_stream_and_copies_it_back(monkeypatch):
    """Built on the current stream, the conversions' freed temporaries took
    the weights after them and their segments could not be released; the copy
    has to be made on the current stream, after the build finished."""
    events = []

    class Stream:
        def __init__(self, device=None, name="side"):
            self.device, self.name = device, name

        def synchronize(self):
            events.append(f"sync {self.name}")

    current = [Stream(name="main")]

    @contextlib.contextmanager
    def stream(s):
        previous, current[0] = current[0], s
        try:
            yield
        finally:
            current[0] = previous

    clone_weight = model_mod._clone_weight

    def clone(weight):
        events.append(f"clone on {current[0].name}")
        return clone_weight(weight)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current[0])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(model_mod, "_clone_weight", clone)

    built = {
        "plain": torch.arange(4.0),
        "w4": quantize_w4a16(torch.randn(128, 128), layout="tiled"),
        "w8": quantize_w8a16(torch.randn(8, 16)),
    }

    def load_weight(name, tensor):
        events.append(f"build {name} on {current[0].name}")
        return built[name]

    model = types.SimpleNamespace(device=torch.device("cuda"), _load_weight=load_weight)
    loaded = NanoLlamaForCausalLM._load_weights(model, dict.fromkeys(built))

    assert events == [
        event
        for name in built
        for event in (f"build {name} on side", "sync side", "clone on main", "sync main")
    ] + ["empty_cache"]
    assert loaded["w8"].dense is None and loaded["w8"].dtype == built["w8"].dtype
    for name, weight in built.items():
        pairs = [(loaded[name], weight)] if name == "plain" else [
            (getattr(loaded[name], field), getattr(weight, field))
            for field in ("qweight", "scales", "zeros") if hasattr(weight, field)
        ]
        for copy, original in pairs:
            assert torch.equal(copy, original) and copy.data_ptr() != original.data_ptr()


def test_config_reads_quantization_config(tmp_path):
    import json

    data = {
        "model_type": "llama", "vocab_size": 32, "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 1, "num_attention_heads": 4, "quantization_config": AWQ_CONFIG,
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    assert NanoLlamaConfig.from_pretrained(tmp_path).quantization_config == AWQ_CONFIG
    del data["quantization_config"]
    (tmp_path / "config.json").write_text(json.dumps(data))
    assert NanoLlamaConfig.from_pretrained(tmp_path).quantization_config is None


def test_large_launch_config_keeps_the_most_blocks_busy():
    from serving.nano_vllm.kernels.w4a16 import W4A16_CONFIG, large_launch_config

    # 1,024 rows: 8 chunks of 128 keep 8 x 13 = 104 of 108 SMs busy, 4 of 256 all 108.
    assert large_launch_config(1024, 4096, W4A16_CONFIG, 108) == (256, 128, 108)
    # 512 rows: 4 x 27 either way; the tie goes to 128x256.
    assert large_launch_config(512, 4096, W4A16_CONFIG, 108) == (128, 256, 108)
    # Only 256x128 divides N = 384.
    assert large_launch_config(300, 384, W4A16_CONFIG, 108)[:2] == (256, 128)
    assert large_launch_config(300, 4096, {**W4A16_CONFIG, "large_tile": "256x128", "tiled_grid": 7}, 108) == (
        256, 128, 7)
    with pytest.raises(ValueError, match="divisible"):
        large_launch_config(300, 384, {**W4A16_CONFIG, "large_tile": "128x256"}, 108)
