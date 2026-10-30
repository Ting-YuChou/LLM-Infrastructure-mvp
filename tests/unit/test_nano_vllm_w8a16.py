"""int8 lm_head (kernels/w8a16.py) on the CPU path, and its model wiring."""

import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from test_nano_vllm_w4a16 import _config, _model, _paged_logits, _weights  # noqa: E402

from serving.nano_vllm import model as model_mod  # noqa: E402
from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.kernels import W8A16Weight, linear  # noqa: E402
from serving.nano_vllm.kernels.w8a16 import (  # noqa: E402
    dequantize_w8a16,
    quantize_w8a16,
    w8a16_config,
    w8a16_linear_reference,
)
from serving.nano_vllm.model import NanoLlamaForCausalLM, resolve_lm_head_quant  # noqa: E402


def test_quantize_is_symmetric_per_row_within_half_a_step():
    torch.manual_seed(0)
    weight = torch.randn(64, 256)
    weight[3] = 0  # an all-zero row keeps a usable scale
    q = quantize_w8a16(weight)
    assert q.qweight.dtype == torch.int8 and q.scales.dtype == torch.float32
    assert q.qweight.abs().max() <= 127 and q.nbytes() == 64 * 256 + 64 * 4
    torch.testing.assert_close(q.scales[[0, 1]], weight[[0, 1]].abs().amax(1) / 127)
    assert q.scales[3] == 1 and not q.qweight[3].any()
    err = (dequantize_w8a16(q, torch.float32) - weight).abs()
    assert torch.all(err <= q.scales[:, None] / 2 + 1e-6)


def test_linear_dispatches_int8_weights():
    torch.manual_seed(1)
    q = quantize_w8a16(torch.randn(96, 128))
    x = torch.randn(2, 3, 128)
    out = linear(x, q, backend="triton")
    assert out.shape == (2, 3, 96)
    torch.testing.assert_close(out, w8a16_linear_reference(x, q), rtol=1e-5, atol=1e-5)


def test_configs_cover_every_row_count():
    assert w8a16_config(1)["block_m"] == 16 and w8a16_config(16)["block_m"] == 16
    assert w8a16_config(17)["block_m"] == 64 and w8a16_config(64)["block_m"] == 64
    assert w8a16_config(65)["block_m"] == 128 and w8a16_config(4096)["block_m"] == 128


def test_lm_head_quant_resolution(monkeypatch):
    assert resolve_lm_head_quant("none", "awq") == "none"
    assert resolve_lm_head_quant("int8", "none") == "int8"
    monkeypatch.setattr(model_mod, "AUTO_LM_HEAD_QUANT", {"awq": "int8"})
    assert resolve_lm_head_quant("auto", "awq") == "int8"
    assert resolve_lm_head_quant("auto", "none") == "none"
    with pytest.raises(ValueError, match="lm_head_quant"):
        resolve_lm_head_quant("int4", "awq")


def _int8_head_model(weights, config=None):
    return NanoLlamaForCausalLM(
        config=config or _config(), weights=weights, device=torch.device("cpu"), dtype=torch.float32,
        use_triton=False, lm_head_quant="int8",
    )


def test_int8_lm_head_model_matches_a_model_holding_the_dequantized_head():
    weights = _weights(_config())
    model = _int8_head_model(weights)
    assert isinstance(model.lm_head_weight, W8A16Weight)
    assert isinstance(model.weights["lm_head.weight"], W8A16Weight)  # no bf16 copy kept
    reference = _model({**weights, "lm_head.weight": dequantize_w8a16(quantize_w8a16(weights["lm_head.weight"]))})
    for actual, expected in zip(_paged_logits(model), _paged_logits(reference)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    ids = torch.tensor([[3, 9, 27, 4]])
    torch.testing.assert_close(
        model.forward(ids, model.new_cache(1, 8)), reference.forward(ids, reference.new_cache(1, 8)),
        rtol=1e-5, atol=1e-5,
    )


def test_tied_embeddings_quantize_a_copy_and_keep_the_lookup_exact():
    weights = _weights(_config())
    del weights["lm_head.weight"]
    config = dataclasses.replace(_config(), tie_word_embeddings=True)
    model = _int8_head_model(weights, config)
    assert isinstance(model.lm_head_weight, W8A16Weight)
    assert torch.equal(model.weights["model.embed_tokens.weight"], weights["model.embed_tokens.weight"])


def test_lm_head_quant_comes_from_yaml_and_env(tmp_path, monkeypatch):
    assert NanoVLLMConfig().lm_head_quant == "auto"
    with pytest.raises(ValueError, match="lm_head_quant"):
        NanoVLLMConfig(lm_head_quant="fp8")
    path = tmp_path / "nano.yaml"
    path.write_text("engine:\n  lm_head_quant: int8\n")
    assert NanoVLLMConfig.from_yaml(str(path)).lm_head_quant == "int8"
    monkeypatch.setenv("LM_HEAD_QUANT", "none")
    assert NanoVLLMConfig.from_yaml(str(path)).lm_head_quant == "none"
