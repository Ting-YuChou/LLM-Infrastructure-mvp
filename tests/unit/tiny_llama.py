"""Tiny tokenizer/Llama builders for tests that need the real transformers stack."""

import importlib.util
import sys

WORDS = (
    "explain tests they catch regressions write code good bad answer question "
    "the a is and to of models train serve llama user assistant system hi"
).split()

LLAMA3_PAD_TOKEN = "<|reserved_special_token_0|>"


def real_transformers_available() -> bool:
    """True when real transformers/tokenizers can be imported (not a test stub)."""
    for name in ("transformers", "tokenizers"):
        module = sys.modules.get(name)
        if module is not None:
            if getattr(module, "__spec__", None) is None:
                return False
        elif importlib.util.find_spec(name) is None:
            return False
    return True


def build_tiny_tokenizer(extra_special_tokens=()):
    """Word-level fast tokenizer with BOS/EOS; chat tokens only if passed in."""
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    if extra_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(extra_special_tokens)})
    return tokenizer


def build_tiny_llama(vocab_size, seed=0, causal_lm=True):
    """1-layer GQA Llama small enough for CPU tests."""
    import torch
    import transformers

    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
    )
    model_cls = transformers.LlamaForCausalLM if causal_lm else transformers.LlamaForSequenceClassification
    if not causal_lm:
        config.num_labels = 1
    return model_cls(config)


# AutoAWQ's GEMM format, written out from its definition rather than from the
# loader: qweight [K, N / 8] with nibble p of word j holding column
# 8j + AWQ_ORDER[p], qzeros likewise, scales [K / 128, N]; w = (q - z) * s.
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_QUANT_CONFIG = {"bits": 4, "group_size": 128, "zero_point": True, "version": "gemm", "quant_method": "awq"}
AWQ_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def awq_pack_columns(values):
    """uint8 [R, N] -> int32 [R, N / 8] the AutoAWQ way."""
    import torch

    r, n = values.shape
    words = torch.zeros(r, n // 8, dtype=torch.int64)
    for p, column in enumerate(AWQ_ORDER):
        words |= values[:, column::8].to(torch.int64).cpu() << (4 * p)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def awq_quantize(weight, group_size=128):
    """Min-max asymmetric codes, zero points and fp16 scales of a [N, K] weight."""
    import torch

    n, k = weight.shape
    groups = weight.float().reshape(n, k // group_size, group_size)
    lo, hi = groups.amin(-1), groups.amax(-1)
    scales = ((hi - lo) / 15).clamp_min(1e-6).half()
    zeros = torch.round(-lo / scales.float()).clamp(0, 15)
    codes = (torch.round(groups / scales.float().unsqueeze(-1)) + zeros.unsqueeze(-1)).clamp(0, 15)
    return codes.reshape(n, k).to(torch.uint8), zeros.to(torch.uint8), scales


def awq_checkpoint(weights, group_size=128):
    """The decoder projections of a Llama state dict as an AWQ checkpoint, and
    the same state dict holding their dequantized weights (fp32 of the fp16
    scales), for a reference model."""
    awq, dequantized = {}, {}
    for name, tensor in weights.items():
        base = name.rpartition(".")[0]
        if not base.endswith(AWQ_PROJECTIONS):
            awq[name] = dequantized[name] = tensor
            continue
        codes, zeros, scales = awq_quantize(tensor, group_size)
        awq[f"{base}.qweight"] = awq_pack_columns(codes.t())
        awq[f"{base}.qzeros"] = awq_pack_columns(zeros.t())
        awq[f"{base}.scales"] = scales.t().contiguous()
        q = codes.float().reshape(*zeros.shape, group_size)
        w = (q - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)
        dequantized[name] = w.reshape(tensor.shape).to(tensor.dtype)
    return awq, dequantized
