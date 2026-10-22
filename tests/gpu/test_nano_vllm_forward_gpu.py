"""GPU checks for the nano_vllm forward pass.

Run on Colab by `benchmarks/phase1/run_phase1.sh` (tests step); on a machine
without CUDA every test here is skipped.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.metadata import build_batch  # noqa: E402
from serving.nano_vllm.model import NanoLlamaConfig, NanoLlamaForCausalLM  # noqa: E402

CONFIG = NanoLlamaConfig(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,  # GQA group of 4, as in Llama 3.1 8B
    max_position_embeddings=256,
    rope_theta=500000.0,
    rope_scaling={
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    },
)
BLOCK_SIZE = 16
PROMPTS = [list(range(3, 40)), list(range(100, 117)), [7]]  # 37, 17 and 1 tokens


def _random_weights(config):
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.05

    hidden, inter = config.hidden_size, config.intermediate_size
    kv_dim = config.num_key_value_heads * config.head_dim
    weights = {
        "model.embed_tokens.weight": rand(config.vocab_size, hidden),
        "model.norm.weight": 1 + rand(hidden),
        "lm_head.weight": rand(config.vocab_size, hidden),
    }
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": 1 + rand(hidden),
                f"{prefix}.post_attention_layernorm.weight": 1 + rand(hidden),
                f"{prefix}.self_attn.q_proj.weight": rand(hidden, hidden),
                f"{prefix}.self_attn.k_proj.weight": rand(kv_dim, hidden),
                f"{prefix}.self_attn.v_proj.weight": rand(kv_dim, hidden),
                f"{prefix}.self_attn.o_proj.weight": rand(hidden, hidden),
                f"{prefix}.mlp.gate_proj.weight": rand(inter, hidden),
                f"{prefix}.mlp.up_proj.weight": rand(inter, hidden),
                f"{prefix}.mlp.down_proj.weight": rand(hidden, inter),
            }
        )
    return weights


def _model(device, dtype, use_triton, attention_backend="torch", linear_backend="torch"):
    return NanoLlamaForCausalLM(
        config=CONFIG,
        weights=_random_weights(CONFIG),
        device=torch.device(device),
        dtype=dtype,
        use_triton=use_triton,
        attention_backend=attention_backend,
        max_model_len=128,
        linear_backend=linear_backend,
    )


def _tables():
    """Non-contiguous block ids, as after some allocation churn."""
    tables, next_block = [], 1
    for prompt in PROMPTS:
        need = (len(prompt) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        tables.append([next_block + 2 * i for i in range(need)])
        next_block += 2 * need
    return tables


def _batch(model, cache, decode: bool):
    tables = _tables()
    input_ids, positions, slots, context_lens, query_lens = [], [], [], [], []
    for prompt, table in zip(PROMPTS, tables):
        if decode:
            tokens, start = [prompt[-1] + 1], len(prompt)
        else:
            tokens, start = prompt, 0
        pos = list(range(start, start + len(tokens)))
        input_ids += tokens
        positions += pos
        slots += [table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in pos]
        context_lens.append(start + len(tokens))
        query_lens.append(len(tokens))
    return build_batch(
        input_ids=input_ids,
        positions=positions,
        slots=slots,
        block_tables=tables,
        context_lens=context_lens,
        query_lens=query_lens,
        is_prefill=not decode,
        null_block=cache.null_block,
        num_slots=cache.total_slots,
        device=model.device,
    )


def _prefill_then_decode(model):
    cache = model.new_paged_cache(num_blocks=16, block_size=BLOCK_SIZE)
    ids, positions, metadata = _batch(model, cache, decode=False)
    prefill = model.forward_batch(ids, positions, cache, metadata)
    ids, positions, metadata = _batch(model, cache, decode=True)
    decode = model.forward_batch(ids, positions, cache, metadata)
    return prefill.float().cpu(), decode.float().cpu()


@pytest.mark.parametrize("attention_backend", ["torch", "triton"])
def test_decode_step_never_syncs_with_the_host(attention_backend):
    model = _model("cuda", torch.bfloat16, use_triton=True, attention_backend=attention_backend)
    cache = model.new_paged_cache(num_blocks=16, block_size=BLOCK_SIZE)
    ids, positions, metadata = _batch(model, cache, decode=False)
    model.forward_batch(ids, positions, cache, metadata)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        ids, positions, metadata = _batch(model, cache, decode=True)
        logits = model.forward_batch(ids, positions, cache, metadata)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert logits.shape == (len(PROMPTS), CONFIG.vocab_size)
    assert torch.isfinite(logits).all()


def test_cuda_fp32_matches_cpu():
    cpu = _prefill_then_decode(_model("cpu", torch.float32, use_triton=False))
    gpu = _prefill_then_decode(_model("cuda", torch.float32, use_triton=False))

    for expected, actual in zip(cpu, gpu):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_cuda_bf16_with_triton_tracks_fp32():
    reference = _prefill_then_decode(_model("cuda", torch.float32, use_triton=False))
    fast = _prefill_then_decode(_model("cuda", torch.bfloat16, use_triton=True))

    for expected, actual in zip(reference, fast):
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_triton_attention_backend_matches_torch_backend():
    reference = _prefill_then_decode(_model("cuda", torch.float32, use_triton=False))
    triton_fp32 = _prefill_then_decode(
        _model("cuda", torch.float32, use_triton=False, attention_backend="triton")
    )
    triton_bf16 = _prefill_then_decode(
        _model("cuda", torch.bfloat16, use_triton=True, attention_backend="triton")
    )

    for expected, actual in zip(reference, triton_fp32):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for expected, actual in zip(reference, triton_bf16):
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_all_triton_paths_track_fp32():
    reference = _prefill_then_decode(_model("cuda", torch.float32, use_triton=False))
    fast = _prefill_then_decode(
        _model(
            "cuda", torch.bfloat16, use_triton=True, attention_backend="triton",
            linear_backend="triton",
        )
    )

    for expected, actual in zip(reference, fast):
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
