import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.metadata import build_batch  # noqa: E402
from serving.nano_vllm.model import (  # noqa: E402
    NanoLlamaConfig,
    NanoLlamaForCausalLM,
    fuse_projections,
    torch_dtype_from_string,
)


def _toy_weights(config: NanoLlamaConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    head_dim = config.head_dim
    kv_dim = config.num_key_value_heads * head_dim
    weights = {
        "model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size)
        * 0.02,
        "model.norm.weight": torch.ones(config.hidden_size),
        "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size) * 0.02,
    }
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": torch.ones(config.hidden_size),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(
                    config.hidden_size
                ),
                f"{prefix}.self_attn.q_proj.weight": torch.randn(
                    config.hidden_size, config.hidden_size
                )
                * 0.02,
                f"{prefix}.self_attn.k_proj.weight": torch.randn(
                    kv_dim, config.hidden_size
                )
                * 0.02,
                f"{prefix}.self_attn.v_proj.weight": torch.randn(
                    kv_dim, config.hidden_size
                )
                * 0.02,
                f"{prefix}.self_attn.o_proj.weight": torch.randn(
                    config.hidden_size, config.hidden_size
                )
                * 0.02,
                f"{prefix}.mlp.gate_proj.weight": torch.randn(
                    config.intermediate_size,
                    config.hidden_size,
                )
                * 0.02,
                f"{prefix}.mlp.up_proj.weight": torch.randn(
                    config.intermediate_size,
                    config.hidden_size,
                )
                * 0.02,
                f"{prefix}.mlp.down_proj.weight": torch.randn(
                    config.hidden_size,
                    config.intermediate_size,
                )
                * 0.02,
            }
        )
    return weights


def test_toy_nano_llama_forward_prefill_and_decode():
    config = NanoLlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )
    model = NanoLlamaForCausalLM(
        config=config,
        weights=_toy_weights(config),
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_triton=False,
    )
    cache = model.new_cache(batch_size=1, max_seq_len=5)

    prefill_logits = model.forward(torch.tensor([[1, 2, 3]]), cache=cache, start_pos=0)
    decode_logits = model.forward(torch.tensor([[4]]), cache=cache, start_pos=3)

    assert prefill_logits.shape == (1, 3, config.vocab_size)
    assert decode_logits.shape == (1, 1, config.vocab_size)
    assert int(cache.lengths[0].item()) == 4


def test_toy_nano_llama_forward_paged_prefill_and_decode():
    config = NanoLlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )
    model = NanoLlamaForCausalLM(
        config=config,
        weights=_toy_weights(config),
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_triton=False,
    )
    cache = model.new_paged_cache(num_blocks=2, block_size=4)
    prefill_logits = model.forward_paged(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        cache=cache,
        slot_mapping=torch.tensor([0, 1, 2]),
        block_tables=torch.tensor([[0]]),
        context_lens=torch.tensor([3]),
        query_lens=[3],
        is_prefill=True,
    )
    decode_logits = model.forward_paged(
        input_ids=torch.tensor([4]),
        positions=torch.tensor([3]),
        cache=cache,
        slot_mapping=torch.tensor([3]),
        block_tables=torch.tensor([[0]]),
        context_lens=torch.tensor([4]),
        query_lens=[1],
        is_prefill=False,
    )

    assert prefill_logits.shape == (3, config.vocab_size)
    assert decode_logits.shape == (1, config.vocab_size)


LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8,
}


def _save_tiny_hf_llama(path, **overrides):
    """Save a random 2-layer GQA transformers Llama; return (hf_model, path)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    values = dict(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    torch.manual_seed(1234)
    hf_model = LlamaForCausalLM(LlamaConfig(**values, attn_implementation="eager"))
    hf_model.eval()
    hf_model.save_pretrained(str(path))
    return hf_model, path


def _hf_logits(hf_model, token_ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return hf_model(torch.tensor([token_ids])).logits[0]


def _load_nano(path, **kwargs) -> NanoLlamaForCausalLM:
    return NanoLlamaForCausalLM.from_pretrained(
        path, dtype="float32", device="cpu", use_triton=False, **kwargs
    )


HF_PARITY_CASES = {
    "plain": {},
    "llama3_rope_scaling": {"rope_scaling": LLAMA3_SCALING},
    "tied_embeddings": {"tie_word_embeddings": True},
}


@pytest.mark.parametrize("overrides", HF_PARITY_CASES.values(), ids=HF_PARITY_CASES)
def test_static_forward_matches_transformers_llama(tmp_path, overrides):
    hf_model, path = _save_tiny_hf_llama(tmp_path, **overrides)
    model = _load_nano(path)
    prompt = [5, 17, 3, 88, 42, 9, 61, 2, 33, 70, 11]
    next_token = 25

    cache = model.new_cache(batch_size=1, max_seq_len=len(prompt) + 1)
    prefill = model.forward(torch.tensor([prompt]), cache=cache, start_pos=0)
    decode = model.forward(
        torch.tensor([[next_token]]), cache=cache, start_pos=len(prompt)
    )

    expected = _hf_logits(hf_model, prompt + [next_token])
    torch.testing.assert_close(prefill[0], expected[:-1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode[0, 0], expected[-1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("overrides", HF_PARITY_CASES.values(), ids=HF_PARITY_CASES)
def test_paged_forward_matches_transformers_llama(tmp_path, overrides):
    hf_model, path = _save_tiny_hf_llama(tmp_path, **overrides)
    model = _load_nano(path)
    block_size = 4
    cache = model.new_paged_cache(num_blocks=8, block_size=block_size)
    seq_a = [5, 17, 3, 88, 42, 9, 61, 2, 33, 70, 11]  # blocks 0, 1, 2
    seq_b = [7, 7, 50, 1, 95]  # blocks 3, 4
    tables = {"a": [0, 1, 2], "b": [3, 4]}

    def slots(table, positions):
        return [table[p // block_size] * block_size + p % block_size for p in positions]

    def run(segments):
        """segments: list of (name, tokens, start_position)."""
        input_ids, positions, slot_mapping, context_lens, query_lens = [], [], [], [], []
        for name, tokens, start in segments:
            pos = list(range(start, start + len(tokens)))
            input_ids += tokens
            positions += pos
            slot_mapping += slots(tables[name], pos)
            context_lens.append(start + len(tokens))
            query_lens.append(len(tokens))
        max_blocks = max(len(tables[name]) for name, _, _ in segments)
        block_tables = [
            tables[name] + [-1] * (max_blocks - len(tables[name]))
            for name, _, _ in segments
        ]
        return model.forward_paged(
            input_ids=torch.tensor(input_ids),
            positions=torch.tensor(positions),
            cache=cache,
            slot_mapping=torch.tensor(slot_mapping),
            block_tables=torch.tensor(block_tables),
            context_lens=torch.tensor(context_lens),
            query_lens=query_lens,
            is_prefill=True,
        )

    # Chunked prefill of A (5 tokens) batched with all of B, then A's rest.
    first = run([("a", seq_a[:5], 0), ("b", seq_b, 0)])
    second = run([("a", seq_a[5:], 5)])
    # One decode step for both sequences in the same batch.
    decode = run([("a", [25], len(seq_a)), ("b", [60], len(seq_b))])

    expected_a = _hf_logits(hf_model, seq_a + [25])
    expected_b = _hf_logits(hf_model, seq_b + [60])
    tol = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(first[:5], expected_a[:5], **tol)
    torch.testing.assert_close(first[5:], expected_b[:-1], **tol)
    torch.testing.assert_close(second, expected_a[5:-1], **tol)
    torch.testing.assert_close(decode[0], expected_a[-1], **tol)
    torch.testing.assert_close(decode[1], expected_b[-1], **tol)


def _prefill_two_sequences(model, cache):
    """Prefill A (6 tokens, blocks 5 and 2) and B (3 tokens, block 7)."""
    input_ids, positions, metadata = build_batch(
        input_ids=[5, 17, 3, 88, 42, 9] + [7, 50, 1],
        positions=list(range(6)) + list(range(3)),
        slots=[20, 21, 22, 23, 8, 9] + [28, 29, 30],
        block_tables=[[5, 2], [7]],
        context_lens=[6, 3],
        query_lens=[6, 3],
        is_prefill=True,
        null_block=cache.null_block,
        num_slots=cache.total_slots,
        device=model.device,
    )
    return model.forward_batch(input_ids, positions, cache, metadata)


def _decode_batch(model, cache, extra_padding_rows=0):
    """One decode step for A and B, optionally padded like a CUDA-graph bucket."""
    null = cache.null_block
    null_slot = null * cache.block_size
    pad = extra_padding_rows
    return build_batch(
        input_ids=[25, 60] + [0] * pad,
        positions=[6, 3] + [0] * pad,
        slots=[10, 31] + [null_slot] * pad,
        block_tables=[[5, 2], [7]] + [[null]] * pad,
        context_lens=[7, 4] + [1] * pad,
        query_lens=[1, 1] + [1] * pad,
        is_prefill=False,
        null_block=null,
        num_slots=cache.total_slots,
        device=model.device,
    )


def test_forward_batch_returns_logits_for_last_token_of_each_sequence(tmp_path):
    hf_model, path = _save_tiny_hf_llama(tmp_path)
    model = _load_nano(path)
    cache = model.new_paged_cache(num_blocks=8, block_size=4)

    prefill = _prefill_two_sequences(model, cache)

    assert prefill.shape == (2, model.config.vocab_size)
    tol = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(prefill[0], _hf_logits(hf_model, [5, 17, 3, 88, 42, 9])[-1], **tol)
    torch.testing.assert_close(prefill[1], _hf_logits(hf_model, [7, 50, 1])[-1], **tol)


def test_padding_rows_into_the_null_block_leave_real_rows_unchanged(tmp_path):
    _, path = _save_tiny_hf_llama(tmp_path)
    model = _load_nano(path)
    plain_cache = model.new_paged_cache(num_blocks=8, block_size=4)
    padded_cache = model.new_paged_cache(num_blocks=8, block_size=4)
    _prefill_two_sequences(model, plain_cache)
    _prefill_two_sequences(model, padded_cache)

    ids, positions, metadata = _decode_batch(model, plain_cache)
    plain = model.forward_batch(ids, positions, plain_cache, metadata)
    ids, positions, metadata = _decode_batch(model, padded_cache, extra_padding_rows=2)
    padded = model.forward_batch(ids, positions, padded_cache, metadata)

    assert padded.shape == (4, model.config.vocab_size)
    assert torch.isfinite(padded).all()
    tol = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(padded[:2], plain, **tol)
    # The padding rows wrote their KV only into the null block.
    real_blocks = slice(0, padded_cache.num_blocks)
    for layer_idx in range(model.config.num_hidden_layers):
        torch.testing.assert_close(
            padded_cache.keys[layer_idx][real_blocks],
            plain_cache.keys[layer_idx][real_blocks],
            **tol,
        )


def test_triton_backend_falls_back_to_the_references_on_cpu(tmp_path):
    _, path = _save_tiny_hf_llama(tmp_path)
    outputs = []
    for backend in ("torch", "triton"):
        model = _load_nano(path, attention_backend=backend)
        cache = model.new_paged_cache(num_blocks=8, block_size=4)
        prefill = _prefill_two_sequences(model, cache)
        ids, positions, metadata = _decode_batch(model, cache)
        outputs.append((prefill, model.forward_batch(ids, positions, cache, metadata)))

    for expected, actual in zip(*outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_projections_are_fused_at_load(tmp_path):
    _, path = _save_tiny_hf_llama(tmp_path)
    model = _load_nano(path)
    config = model.config
    kv_dim = config.num_key_value_heads * config.head_dim

    assert "model.layers.0.self_attn.q_proj.weight" not in model.weights
    assert "model.layers.0.mlp.up_proj.weight" not in model.weights
    assert model.weights["model.layers.0.self_attn.qkv_proj.weight"].shape == (
        config.hidden_size + 2 * kv_dim,
        config.hidden_size,
    )
    assert model.weights["model.layers.0.mlp.gate_up_proj.weight"].shape == (
        2 * config.intermediate_size,
        config.hidden_size,
    )


def test_fuse_projections_rejects_a_partial_bias():
    weights = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(2, 4),
        "model.layers.0.self_attn.q_proj.bias": torch.zeros(4),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(8, 4),
    }

    with pytest.raises(ValueError, match="bias"):
        fuse_projections(weights, num_layers=1)


def test_config_reads_dtype_tying_and_rope_scaling(tmp_path):
    _save_tiny_hf_llama(
        tmp_path,
        torch_dtype="bfloat16",
        tie_word_embeddings=True,
        rope_scaling=LLAMA3_SCALING,
    )
    config = NanoLlamaConfig.from_pretrained(tmp_path)

    assert config.tie_word_embeddings is True
    assert config.rope_scaling["rope_type"] == "llama3"
    assert torch_dtype_from_string("auto", config.torch_dtype) == torch.float32
    assert torch_dtype_from_string("auto", "bfloat16") == torch.bfloat16
    assert torch_dtype_from_string("auto", torch.bfloat16) == torch.bfloat16
    with pytest.raises(ValueError):
        torch_dtype_from_string("auto", None)


def test_auto_dtype_follows_checkpoint_torch_dtype(tmp_path):
    import json

    _save_tiny_hf_llama(tmp_path)
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    data.pop("dtype", None)
    data["torch_dtype"] = "bfloat16"  # how Llama 3 checkpoints declare it
    config_path.write_text(json.dumps(data))

    assert NanoLlamaConfig.from_pretrained(tmp_path).torch_dtype == "bfloat16"
    assert torch_dtype_from_string("auto", "bfloat16") == torch.bfloat16


def test_untied_checkpoint_without_lm_head_is_rejected():
    config = NanoLlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    weights = _toy_weights(config)
    del weights["lm_head.weight"]

    with pytest.raises(KeyError, match="tie_word_embeddings"):
        NanoLlamaForCausalLM(
            config=config,
            weights=weights,
            device=torch.device("cpu"),
            dtype=torch.float32,
            use_triton=False,
        )


def test_unsupported_rope_scaling_config_is_rejected(tmp_path):
    import json

    _save_tiny_hf_llama(tmp_path)
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    data["rope_scaling"] = {"rope_type": "dynamic", "factor": 2.0}
    config_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="dynamic"):
        NanoLlamaConfig.from_pretrained(tmp_path)
