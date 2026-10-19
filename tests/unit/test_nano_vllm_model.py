import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.model import NanoLlamaConfig, NanoLlamaForCausalLM  # noqa: E402


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
