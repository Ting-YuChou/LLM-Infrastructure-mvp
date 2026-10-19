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
