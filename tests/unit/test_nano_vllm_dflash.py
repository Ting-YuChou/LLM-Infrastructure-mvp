import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.dflash import (  # noqa: E402
    DFlashConfig,
    DFlashDrafter,
    default_target_layer_ids,
)
from serving.nano_vllm.kernels import W4A16Weight, dequantize_w4a16, quantize_w4a16  # noqa: E402
from serving.nano_vllm.model import NanoLlamaConfig, NanoLlamaForCausalLM  # noqa: E402

# config.json of z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat, as published.
PUBLIC_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
    "block_size": 10,
    "bos_token_id": 151643,
    "dflash_config": {"mask_token_id": 128002, "target_layer_ids": [1, 8, 15, 22, 29]},
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "layer_types": ["full_attention"] * 5,
    "max_position_embeddings": 40960,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 32,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "4.57.3",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 128256,
}

VOCAB, HIDDEN, TARGET_LAYERS = 64, 32, 4
MASK_ID = 3
CACHE_BLOCK = 4
# A tiny drafter in the public checkpoint's shape: GQA, an explicit head_dim
# (heads * head_dim != hidden), two tapped target layers.
TINY_CONFIG = {
    **PUBLIC_CONFIG,
    "block_size": 5,
    "dflash_config": {"mask_token_id": MASK_ID, "target_layer_ids": [0, 2]},
    "head_dim": 16,
    "hidden_size": HIDDEN,
    "intermediate_size": 48,
    "layer_types": ["full_attention"] * 2,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_target_layers": TARGET_LAYERS,
    "vocab_size": VOCAB,
}


def _drafter_state(config: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights under the checkpoint's tensor names."""
    generator = torch.Generator().manual_seed(seed)

    def matrix(out_features, in_features):
        return torch.randn(out_features, in_features, generator=generator) / in_features**0.5

    def norm_weight(size):  # not ones, so a swapped or skipped norm shows
        return 0.5 + torch.rand(size, generator=generator)

    hidden, inter = config["hidden_size"], config["intermediate_size"]
    head_dim = config["head_dim"]
    attn_dim = config["num_attention_heads"] * head_dim
    kv_dim = config["num_key_value_heads"] * head_dim
    taps = len(config["dflash_config"]["target_layer_ids"])
    state = {
        "fc.weight": matrix(hidden, taps * hidden),
        "hidden_norm.weight": norm_weight(hidden),
        "norm.weight": norm_weight(hidden),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"layers.{layer}"
        state.update({
            f"{prefix}.input_layernorm.weight": norm_weight(hidden),
            f"{prefix}.post_attention_layernorm.weight": norm_weight(hidden),
            f"{prefix}.self_attn.q_proj.weight": matrix(attn_dim, hidden),
            f"{prefix}.self_attn.k_proj.weight": matrix(kv_dim, hidden),
            f"{prefix}.self_attn.v_proj.weight": matrix(kv_dim, hidden),
            f"{prefix}.self_attn.o_proj.weight": matrix(hidden, attn_dim),
            f"{prefix}.self_attn.q_norm.weight": norm_weight(head_dim),
            f"{prefix}.self_attn.k_norm.weight": norm_weight(head_dim),
            f"{prefix}.mlp.gate_proj.weight": matrix(inter, hidden),
            f"{prefix}.mlp.up_proj.weight": matrix(inter, hidden),
            f"{prefix}.mlp.down_proj.weight": matrix(hidden, inter),
        })
    return state


def _save_checkpoint(path: Path, config: dict, state: dict[str, torch.Tensor]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors_torch.save_file(state, str(path / "model.safetensors"))
    return path


def _target(hidden=HIDDEN, vocab=VOCAB, kv_cache_dtype="auto") -> NanoLlamaForCausalLM:
    """A random target: the drafter reads its embedding and lm_head only."""
    config = NanoLlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=TARGET_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    generator = torch.Generator().manual_seed(100)

    def matrix(out_features, in_features):
        return torch.randn(out_features, in_features, generator=generator) / in_features**0.5

    weights = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, generator=generator),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": matrix(vocab, hidden),
    }
    for layer in range(TARGET_LAYERS):
        prefix = f"model.layers.{layer}"
        weights.update({
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
            f"{prefix}.self_attn.q_proj.weight": matrix(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": matrix(hidden // 2, hidden),
            f"{prefix}.self_attn.v_proj.weight": matrix(hidden // 2, hidden),
            f"{prefix}.self_attn.o_proj.weight": matrix(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": matrix(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight": matrix(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": matrix(hidden, 2 * hidden),
        })
    return NanoLlamaForCausalLM(
        config, weights, device=torch.device("cpu"), dtype=torch.float32, use_triton=False,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.fixture
def tiny(tmp_path):
    """(drafter loaded from a checkpoint directory, its state dict, target)."""
    state = _drafter_state(TINY_CONFIG)
    path = _save_checkpoint(tmp_path / "drafter", TINY_CONFIG, state)
    target = _target()
    return DFlashDrafter.from_pretrained(path, target), state, target


# --- the reference forward, dense, from the public dflash.py ----------------


def _rms(x, weight, eps):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def dense_draft(state, config, embed, lm_head, taps, anchor_id, num_masks):
    """DFlashDraftModel.forward for one sequence with no cache, as
    spec_generate calls it: ``taps`` [context, taps * hidden] is the
    concatenated target hidden states of the context, the block is
    [anchor, MASK x num_masks] at the positions right after the context.
    Returns the logits of the mask positions.
    """
    eps = config["rms_norm_eps"]
    heads, kv_heads, head_dim = (
        config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"],
    )
    context_len, q_len = taps.shape[0], num_masks + 1
    block_ids = torch.full((q_len,), config["dflash_config"]["mask_token_id"], dtype=torch.long)
    block_ids[0] = anchor_id
    hidden = embed[block_ids]  # noise_embedding
    target_hidden = _rms(taps @ state["fc.weight"].T, state["hidden_norm.weight"], eps)

    # Qwen3RotaryEmbedding over position_ids = arange(context + block).
    inv_freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.arange(context_len + q_len).float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()  # [context + block, head_dim]

    for layer in range(config["num_hidden_layers"]):
        w = {name.split(".", 2)[2]: t for name, t in state.items() if name.startswith(f"layers.{layer}.")}
        residual = hidden
        x = _rms(hidden, w["input_layernorm.weight"], eps)
        q = (x @ w["self_attn.q_proj.weight"].T).view(q_len, heads, head_dim)
        q = _rms(q, w["self_attn.q_norm.weight"], eps).transpose(0, 1)  # [heads, q, d]
        k = torch.cat([
            target_hidden @ w["self_attn.k_proj.weight"].T, x @ w["self_attn.k_proj.weight"].T,
        ]).view(context_len + q_len, kv_heads, head_dim)
        v = torch.cat([
            target_hidden @ w["self_attn.v_proj.weight"].T, x @ w["self_attn.v_proj.weight"].T,
        ]).view(context_len + q_len, kv_heads, head_dim)
        k = _rms(k, w["self_attn.k_norm.weight"], eps).transpose(0, 1)  # [kv_heads, t, d]
        v = v.transpose(0, 1)
        q = q * cos[-q_len:] + _rotate_half(q) * sin[-q_len:]
        k = k * cos + _rotate_half(k) * sin
        k = k.repeat_interleave(heads // kv_heads, dim=0)
        v = v.repeat_interleave(heads // kv_heads, dim=0)
        # attention_mask is None and is_causal False: no mask at all.
        weights = torch.softmax(q @ k.transpose(1, 2) * head_dim**-0.5, dim=-1)
        attn = (weights @ v).transpose(0, 1).reshape(q_len, heads * head_dim)
        hidden = residual + attn @ w["self_attn.o_proj.weight"].T
        residual = hidden
        x = _rms(hidden, w["post_attention_layernorm.weight"], eps)
        mlp = torch.nn.functional.silu(x @ w["mlp.gate_proj.weight"].T) * (x @ w["mlp.up_proj.weight"].T)
        hidden = residual + mlp @ w["mlp.down_proj.weight"].T
    return _rms(hidden, state["norm.weight"], eps)[-num_masks:] @ lm_head.T


# --- paged-cache plumbing ---------------------------------------------------


class Pool:
    """Shuffled, non-contiguous block tables over a drafter cache."""

    def __init__(self, drafter, num_seqs, max_len, num_blocks=40):
        self.cache = drafter.new_paged_cache(num_blocks, CACHE_BLOCK)
        order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(5)).tolist()
        per_seq = (max_len + CACHE_BLOCK - 1) // CACHE_BLOCK
        self.tables = [[order.pop() for _ in range(per_seq)] for _ in range(num_seqs)]
        # One null-block column of padding, as the engine's tables may have.
        self.block_tables = torch.tensor(
            [table + [self.cache.null_block] for table in self.tables], dtype=torch.int32
        )

    def slots(self, seq, positions):
        table = self.tables[seq]
        return [table[p // CACHE_BLOCK] * CACHE_BLOCK + p % CACHE_BLOCK for p in positions]

    def write(self, drafter, taps, spans):
        """write_context for spans [(seq, start, end)], packed in that order."""
        rows = torch.cat([taps[seq][start:end] for seq, start, end in spans])
        positions = [p for _, start, end in spans for p in range(start, end)]
        slots = [s for seq, start, end in spans for s in self.slots(seq, range(start, end))]
        drafter.write_context(rows, torch.tensor(positions), self.cache, torch.tensor(slots))

    def draft(self, drafter, anchors, context_lens, num_masks):
        q = num_masks + 1
        positions = [c + t for c in context_lens for t in range(q)]
        slots = [s for seq, c in enumerate(context_lens) for s in self.slots(seq, range(c, c + q))]
        logits = drafter.draft(
            torch.tensor(anchors), torch.tensor(positions), self.cache, self.block_tables,
            torch.tensor(context_lens, dtype=torch.int32), torch.tensor(slots),
        )
        return logits.view(len(anchors), num_masks, -1)


def _taps(lens, hidden=HIDDEN, num_taps=2, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(n, num_taps * hidden, generator=generator) for n in lens]


def _dense(state, target, taps, anchors, context_lens, num_masks, config=TINY_CONFIG):
    return torch.stack([
        dense_draft(
            state, config, target.weights["model.embed_tokens.weight"], target.lm_head_weight,
            taps[seq][:context_len], anchor, num_masks,
        )
        for seq, (anchor, context_len) in enumerate(zip(anchors, context_lens))
    ])


def _assert_same_drafts(got, expected):
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(got.argmax(-1), expected.argmax(-1))


ANCHORS = [7, 41, 12]
CONTEXT_LENS = [1, 6, 11]  # one token; ending inside and across cache blocks
K = TINY_CONFIG["block_size"] - 1


# --- tests ------------------------------------------------------------------


def test_config_parses_the_public_checkpoint_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(PUBLIC_CONFIG), encoding="utf-8")
    config = DFlashConfig.from_pretrained(tmp_path)

    assert config == DFlashConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        block_size=10,
        mask_token_id=128002,
        target_layer_ids=(1, 8, 15, 22, 29),
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=40960,
        num_target_layers=32,
        torch_dtype="bfloat16",
    )
    assert config.num_draft_tokens == 9


def test_config_defaults_follow_the_reference(tmp_path):
    # Without target_layer_ids the reference spaces them over the target's
    # layers; without head_dim it is hidden / heads.
    data = {**PUBLIC_CONFIG, "dflash_config": {"mask_token_id": 128002}}
    del data["head_dim"]
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    config = DFlashConfig.from_pretrained(tmp_path)

    assert config.target_layer_ids == (1, 8, 15, 22, 29)
    assert config.head_dim == 128
    assert default_target_layer_ids(32, 5) == [1, 8, 15, 22, 29]
    assert default_target_layer_ids(28, 1) == [14]
    assert default_target_layer_ids(36, 8) == [1, 6, 10, 15, 19, 24, 28, 33]


@pytest.mark.parametrize(
    "change, match",
    [
        ({"dflash_config": None}, "not a DFlash drafter"),
        ({"block_size": None}, "not a DFlash drafter"),
        ({"dflash_config": {"target_layer_ids": [1]}}, "mask_token_id"),
        ({"attention_bias": True}, "attention_bias"),
        ({"rope_scaling": {"rope_type": "llama3"}}, "rope_scaling"),
        ({"dflash_config": {"mask_token_id": 1}, "num_target_layers": None}, "target_layer_ids"),
    ],
)
def test_config_rejects_what_the_drafter_does_not_model(tmp_path, change, match):
    data = {**PUBLIC_CONFIG, **change}
    data = {key: value for key, value in data.items() if value is not None or key == "rope_scaling"}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        DFlashConfig.from_pretrained(tmp_path)


def test_paged_drafter_matches_the_dense_reference_forward(tiny):
    drafter, state, target = tiny
    taps = _taps(CONTEXT_LENS)
    pool = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])

    logits = pool.draft(drafter, ANCHORS, CONTEXT_LENS, K)

    assert logits.shape == (len(ANCHORS), K, VOCAB)
    _assert_same_drafts(logits, _dense(state, target, taps, ANCHORS, CONTEXT_LENS, K))


def test_drafter_shares_the_target_embedding_and_lm_head(tiny):
    drafter, _, target = tiny
    assert drafter.embed_tokens is target.weights["model.embed_tokens.weight"]
    assert drafter.target.lm_head_weight is target.lm_head_weight
    assert not any("embed" in name or "lm_head" in name for name in drafter.weights)
    cache = drafter.new_paged_cache(8, CACHE_BLOCK)
    assert len(cache.keys) == TINY_CONFIG["num_hidden_layers"]
    assert cache.keys[0].shape == (9, CACHE_BLOCK, 2, 16)


def test_context_written_in_chunks_equals_context_written_at_once(tiny):
    drafter, _, _ = tiny
    taps = _taps(CONTEXT_LENS)
    once = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    once.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])

    # A prefill chunk, single decode tokens and verify-sized pieces, with the
    # sequences interleaved and out of order.
    chunked = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    chunked.write(drafter, taps, [(2, 0, 5), (1, 0, 2)])
    chunked.write(drafter, taps, [(1, 2, 3)])
    chunked.write(drafter, taps, [(0, 0, 1), (2, 5, 9), (1, 3, 6)])
    chunked.write(drafter, taps, [(2, 9, 11)])

    for layer in range(TINY_CONFIG["num_hidden_layers"]):
        torch.testing.assert_close(chunked.cache.keys[layer], once.cache.keys[layer], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked.cache.values[layer], once.cache.values[layer], rtol=1e-5, atol=1e-6)
    _assert_same_drafts(
        chunked.draft(drafter, ANCHORS, CONTEXT_LENS, K), once.draft(drafter, ANCHORS, CONTEXT_LENS, K)
    )


def test_draft_writes_only_the_lookahead_slots(tiny):
    drafter, _, _ = tiny
    taps = _taps(CONTEXT_LENS)
    pool = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])
    before = [(k.clone(), v.clone()) for k, v in zip(pool.cache.keys, pool.cache.values)]

    pool.draft(drafter, ANCHORS, CONTEXT_LENS, K)

    lookahead = torch.zeros(pool.cache.total_slots, dtype=torch.bool)
    for seq, context_len in enumerate(CONTEXT_LENS):
        lookahead[pool.slots(seq, range(context_len, context_len + K + 1))] = True
    for (key, value), new_key, new_value in zip(before, pool.cache.keys, pool.cache.values):
        for old, new in ((key, new_key), (value, new_value)):
            old, new = old.flatten(0, 1), new.flatten(0, 1)
            assert torch.equal(new[~lookahead], old[~lookahead])
            assert (new[lookahead] != old[lookahead]).flatten(1).any(dim=1).all()


def test_second_draft_after_more_context_reuses_the_cache(tiny):
    drafter, state, target = tiny
    # Each cycle accepts a different number of tokens per sequence: their
    # features overwrite the first lookahead slots, the next block starts
    # inside what the last block wrote and leaves stale block K/V after it.
    accepted = [[1, K + 1, 3], [2, 1, K + 1]]
    final_lens = [c + sum(cycle[seq] for cycle in accepted) for seq, c in enumerate(CONTEXT_LENS)]
    taps = _taps(final_lens)
    pool = Pool(drafter, len(CONTEXT_LENS), max(final_lens) + K + 1)
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])

    context_lens = list(CONTEXT_LENS)
    anchors = list(ANCHORS)
    for cycle in [[0, 0, 0]] + accepted:
        if any(cycle):
            pool.write(
                drafter, taps,
                [(seq, c, c + n) for seq, (c, n) in enumerate(zip(context_lens, cycle)) if n],
            )
        context_lens = [c + n for c, n in zip(context_lens, cycle)]
        anchors = [(a * 5 + 1) % VOCAB for a in anchors]
        logits = pool.draft(drafter, anchors, context_lens, K)
        _assert_same_drafts(logits, _dense(state, target, taps, anchors, context_lens, K))
    assert context_lens == final_lens

    # The same as a cache that never held an earlier block.
    fresh = Pool(drafter, len(CONTEXT_LENS), max(final_lens) + K + 1)
    fresh.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(final_lens)])
    _assert_same_drafts(logits, fresh.draft(drafter, anchors, final_lens, K))


@pytest.mark.parametrize("num_masks", [1, 2])
def test_shorter_blocks_match_the_dense_reference(tiny, num_masks):
    drafter, state, target = tiny
    taps = _taps(CONTEXT_LENS)
    pool = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])

    logits = pool.draft(drafter, ANCHORS, CONTEXT_LENS, num_masks)
    _assert_same_drafts(logits, _dense(state, target, taps, ANCHORS, CONTEXT_LENS, num_masks))


def test_draft_of_one_sequence_does_not_depend_on_the_batch(tiny):
    drafter, _, _ = tiny
    taps = _taps(CONTEXT_LENS)
    pool = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])
    batched = pool.draft(drafter, ANCHORS, CONTEXT_LENS, K)

    for seq, (anchor, context_len) in enumerate(zip(ANCHORS, CONTEXT_LENS)):
        positions = list(range(context_len, context_len + K + 1))
        alone = drafter.draft(
            torch.tensor([anchor]), torch.tensor(positions), pool.cache,
            pool.block_tables[seq : seq + 1], torch.tensor([context_len], dtype=torch.int32),
            torch.tensor(pool.slots(seq, positions)),
        )
        _assert_same_drafts(alone.view(K, -1), batched[seq])


def test_drafter_on_an_fp8_cache_stays_close_to_full_precision(tmp_path):
    state = _drafter_state(TINY_CONFIG)
    path = _save_checkpoint(tmp_path / "drafter", TINY_CONFIG, state)
    target = _target(kv_cache_dtype="fp8")
    drafter = DFlashDrafter.from_pretrained(path, target)
    taps = _taps(CONTEXT_LENS)
    pool = Pool(drafter, len(CONTEXT_LENS), max(CONTEXT_LENS) + K + 1)
    assert pool.cache.quantized and pool.cache.keys[0].dtype == torch.float8_e4m3fn
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(CONTEXT_LENS)])

    logits = pool.draft(drafter, ANCHORS, CONTEXT_LENS, K)
    expected = _dense(state, target, taps, ANCHORS, CONTEXT_LENS, K)

    assert not torch.equal(logits, expected)
    # e4m3 keeps three mantissa bits per K/V element.
    error = (logits - expected).abs()
    assert error.max() < 0.15 * expected.abs().max()
    assert error.mean() < 0.03 * expected.abs().max()


def test_w4a16_drafter_equals_a_drafter_with_the_dequantized_weights(tmp_path):
    # in_features must be a multiple of the 128-wide quantization groups.
    config = {
        **TINY_CONFIG, "hidden_size": 128, "intermediate_size": 256, "head_dim": 32,
        "vocab_size": 48,
    }
    state = _drafter_state(config)
    path = _save_checkpoint(tmp_path / "drafter", config, state)
    target = _target(hidden=128, vocab=48)
    drafter = DFlashDrafter.from_pretrained(path, target, weight_quant="w4a16")

    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    quantized = [name for name, w in drafter.weights.items() if isinstance(w, W4A16Weight)]
    assert len(quantized) == 5 * config["num_hidden_layers"]  # q, kv, o, gate_up, down
    assert not isinstance(drafter.fc, W4A16Weight)
    rounded = {
        name: dequantize_w4a16(quantize_w4a16(t)).to(t.dtype)
        if name.rpartition(".")[0].endswith(projections) else t
        for name, t in state.items()
    }
    assert not torch.equal(rounded["layers.0.self_attn.q_proj.weight"], state["layers.0.self_attn.q_proj.weight"])

    lens = [3, 9]
    taps = _taps(lens, hidden=128)
    pool = Pool(drafter, len(lens), max(lens) + K + 1)
    pool.write(drafter, taps, [(seq, 0, n) for seq, n in enumerate(lens)])
    logits = pool.draft(drafter, [5, 17], lens, K)

    expected = _dense(rounded, target, taps, [5, 17], lens, K, config=config)
    torch.testing.assert_close(logits, expected, rtol=1e-3, atol=1e-3)


def test_drafter_rejects_a_target_or_checkpoint_it_does_not_fit(tiny, tmp_path):
    drafter, state, target = tiny
    config = drafter.config

    with pytest.raises(ValueError, match="hidden_size and vocab_size must match"):
        DFlashDrafter(config, state, _target(vocab=VOCAB + 1))
    with pytest.raises(ValueError, match="weight_quant"):
        DFlashDrafter(config, state, target, weight_quant="awq")
    missing = {name: t for name, t in state.items() if name != "layers.1.self_attn.k_norm.weight"}
    with pytest.raises(KeyError, match="k_norm"):
        DFlashDrafter(config, missing, target)
    with pytest.raises(ValueError, match="fc.weight"):
        DFlashDrafter(config, {**state, "fc.weight": state["fc.weight"][:, :HIDDEN]}, target)
    with pytest.raises(ValueError, match="no bias"):
        DFlashDrafter(config, {**state, "layers.0.self_attn.q_proj.bias": torch.zeros(64)}, target)

    deep = {**TINY_CONFIG, "dflash_config": {"mask_token_id": MASK_ID, "target_layer_ids": [0, 4]}}
    path = _save_checkpoint(tmp_path / "deep", deep, state)
    with pytest.raises(ValueError, match="outside the target's 4 layers"):
        DFlashDrafter.from_pretrained(path, target)
    other = _save_checkpoint(tmp_path / "other", {**TINY_CONFIG, "num_target_layers": 32}, state)
    with pytest.raises(ValueError, match="32-layer target"):
        DFlashDrafter.from_pretrained(other, target)
    with pytest.raises(FileNotFoundError):
        DFlashDrafter.from_pretrained(tmp_path / "absent", target)


def test_draft_rejects_blocks_of_the_wrong_size(tiny):
    drafter, _, _ = tiny
    pool = Pool(drafter, 1, 16)
    for rows in (1, TINY_CONFIG["block_size"] + 1):
        with pytest.raises(ValueError, match="anchor plus 1 to 4 mask tokens"):
            drafter.draft(
                torch.tensor([1]), torch.arange(rows), pool.cache, pool.block_tables[:1],
                torch.tensor([0], dtype=torch.int32), torch.tensor(pool.slots(0, range(rows))),
            )
    with pytest.raises(ValueError, match="batch \\* \\(K \\+ 1\\)"):
        drafter.draft(
            torch.tensor([1, 2]), torch.arange(5), pool.cache, pool.block_tables[:1],
            torch.tensor([0, 0], dtype=torch.int32), torch.arange(5),
        )
