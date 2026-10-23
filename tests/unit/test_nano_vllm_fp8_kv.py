"""The FP8 KV cache on the CPU: quantization error, cache writes and reads, the
attention references and a whole forward against the full-precision cache."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.kernels import (  # noqa: E402
    paged_decode_attention_reference,
    paged_prefill_attention_reference,
    rope_and_cache_write_reference,
)
from serving.nano_vllm.kernels.fp8 import (  # noqa: E402
    FP8_MAX,
    fp8_dequantize_reference,
    fp8_quantize_reference,
)
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402
from test_nano_vllm_model import (  # noqa: E402
    _decode_batch,
    _load_nano,
    _prefill_two_sequences,
    _save_tiny_hf_llama,
)

BLOCK_SIZE = 4
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _cache(kv_cache_dtype, num_blocks=16):
    return PagedKVCache(
        num_layers=1, num_blocks=num_blocks, block_size=BLOCK_SIZE, num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM, dtype=torch.float32, device="cpu", kv_cache_dtype=kv_cache_dtype,
    )


def _within_fp8_error(got, want, scale):
    # e4m3 rounds to 3 mantissa bits (half a step: 1/16 of the value) and
    # its subnormals to multiples of scale * 2**-9.
    bound = want.abs() / 16 + scale.unsqueeze(-1) * 2.0**-10
    assert ((got - want).abs() <= bound * 1.0001).all()


def test_quantize_uses_the_full_range_and_stays_within_half_a_step():
    torch.manual_seed(0)
    x = torch.randn(64, KV_HEADS, HEAD_DIM) * torch.logspace(-3, 3, 64).view(-1, 1, 1)
    codes, scale = fp8_quantize_reference(x)
    assert codes.dtype == torch.float8_e4m3fn and scale.shape == x.shape[:-1]
    assert torch.equal(codes.float().abs().amax(dim=-1), torch.full_like(scale, FP8_MAX))
    _within_fp8_error(fp8_dequantize_reference(codes, scale), x, scale)
    zeros, zero_scale = fp8_quantize_reference(torch.zeros(3, HEAD_DIM))
    assert torch.equal(zero_scale, torch.ones(3)) and not zeros.float().any()


def test_bytes_per_block_counts_codes_and_scales():
    full = PagedKVCache.bytes_per_block(32, 16, 8, 128, torch.bfloat16)
    fp8 = PagedKVCache.bytes_per_block(32, 16, 8, 128, torch.bfloat16, "fp8")
    assert full == 2 * 32 * 16 * 8 * 128 * 2
    assert fp8 == 2 * 32 * 16 * 8 * (128 + 4)
    assert full / fp8 == pytest.approx(1.94, abs=0.01)


def test_fp8_cache_update_and_read_sequence_round_trip():
    cache = _cache("fp8")
    assert cache.keys[0].dtype == torch.float8_e4m3fn
    assert cache.key_scales[0].shape == (17, BLOCK_SIZE, KV_HEADS)
    torch.manual_seed(0)
    key, value = torch.randn(6, KV_HEADS, HEAD_DIM), torch.randn(6, KV_HEADS, HEAD_DIM) * 50
    table = [3, 9]
    cache.update(0, torch.tensor([12, 13, 14, 15, 36, 37]), key, value)
    got_k, got_v = cache.read_sequence(0, table, 6)
    assert got_k.dtype == torch.float32
    _within_fp8_error(got_k[0].transpose(0, 1), key, key.abs().amax(-1) / FP8_MAX)
    _within_fp8_error(got_v[0].transpose(0, 1), value, value.abs().amax(-1) / FP8_MAX)
    assert _cache("auto").key_scales == [None]


def test_rope_write_quantizes_the_rotated_key_and_the_value():
    torch.manual_seed(0)
    tokens = 5
    qkv = torch.randn(tokens, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM)
    angles = torch.rand(tokens, HEAD_DIM // 2) * 3
    cos, sin = torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)
    slots = torch.tensor([0, 5, 6, 7, 30])
    full, fp8 = _cache("auto"), _cache("fp8")
    q_full = rope_and_cache_write_reference(
        qkv, cos, sin, full.keys[0], full.values[0], slots, Q_HEADS, KV_HEADS
    )
    q_fp8 = rope_and_cache_write_reference(
        qkv, cos, sin, fp8.keys[0], fp8.values[0], slots, Q_HEADS, KV_HEADS,
        fp8.key_scales[0], fp8.value_scales[0],
    )
    assert torch.equal(q_full, q_fp8)
    for codes, scales, want in ((fp8.keys, fp8.key_scales, full.keys), (fp8.values, fp8.value_scales, full.values)):
        expected_codes, expected_scales = fp8_quantize_reference(want[0].view(-1, KV_HEADS, HEAD_DIM)[slots])
        assert torch.equal(codes[0].view(torch.uint8).view(-1, KV_HEADS, HEAD_DIM)[slots],
                           expected_codes.view(torch.uint8))
        assert torch.equal(scales[0].view(-1, KV_HEADS)[slots], expected_scales)


def _filled(kv_cache_dtype):
    """The same random K/V written to a full-precision or an fp8 cache."""
    torch.manual_seed(0)
    cache = _cache(kv_cache_dtype)
    slots = torch.arange(cache.total_slots)
    cache.update(0, slots, torch.randn(len(slots), KV_HEADS, HEAD_DIM),
                 torch.randn(len(slots), KV_HEADS, HEAD_DIM))
    return cache


def _args(cache):
    return cache.keys[0], cache.values[0]


def _scales(cache):
    return dict(key_scales=cache.key_scales[0], value_scales=cache.value_scales[0])


def test_decode_and_prefill_references_read_fp8_like_the_dequantized_cache():
    full, fp8 = _filled("auto"), _filled("fp8")
    # The full-precision cache holding exactly the dequantized fp8 values:
    # attention over it must equal attention over the codes.
    dequantized = _cache("auto")
    dequantized.keys[0].copy_(fp8_dequantize_reference(fp8.keys[0], fp8.key_scales[0]))
    dequantized.values[0].copy_(fp8_dequantize_reference(fp8.values[0], fp8.value_scales[0]))
    tables = torch.tensor([[3, 1, 4, 16], [9, 2, 16, 16]], dtype=torch.int32)
    context_lens = torch.tensor([13, 6], dtype=torch.int32)
    query = torch.randn(2, Q_HEADS, HEAD_DIM)
    decode = paged_decode_attention_reference(query, *_args(fp8), tables, context_lens, SCALE, **_scales(fp8))
    torch.testing.assert_close(
        decode, paged_decode_attention_reference(query, *_args(dequantized), tables, context_lens, SCALE)
    )
    approx = paged_decode_attention_reference(query, *_args(full), tables, context_lens, SCALE)
    assert (decode - approx).abs().max() < 0.05

    prefill_query = torch.randn(5 + 2, Q_HEADS, HEAD_DIM)
    prefill = paged_prefill_attention_reference(
        prefill_query, *_args(fp8), tables, [5, 2], [13, 6], SCALE, **_scales(fp8)
    )
    torch.testing.assert_close(
        prefill,
        paged_prefill_attention_reference(prefill_query, *_args(dequantized), tables, [5, 2], [13, 6], SCALE),
    )


def test_forward_with_an_fp8_cache_stays_close_to_the_full_precision_one(tmp_path):
    _, path = _save_tiny_hf_llama(tmp_path)
    full_model, fp8_model = _load_nano(path), _load_nano(path, kv_cache_dtype="fp8")
    logits = []
    for model in (full_model, fp8_model):
        cache = model.new_paged_cache(num_blocks=8, block_size=4)
        assert cache.quantized == (model is fp8_model)
        prefill = _prefill_two_sequences(model, cache)
        input_ids, positions, metadata = _decode_batch(model, cache)
        decode = model.forward_batch(input_ids, positions, cache, metadata)
        logits.append(torch.cat([prefill, decode]))
    full, fp8 = logits
    # The tiny random model's logits are nearly flat, so the argmax is not a
    # fair check here (quality.py measures top-1 agreement on the 8B).
    assert not torch.equal(full, fp8)
    assert (full - fp8).abs().max() < 0.05 * full.abs().max()
    assert torch.nn.functional.cosine_similarity(full, fp8, dim=-1).min() > 0.999
