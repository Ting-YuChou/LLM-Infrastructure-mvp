"""
Merge a LoRA adapter into its base model and save a full Hugging Face checkpoint.

The reward model, PPO and the AWQ/vLLM handoff all expect a full model
directory (config.json + weights + tokenizer), while LoRA runs only write an
adapter. This runs on CPU so it never competes with training for GPU memory.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .chat_format import apply_generation_config, ensure_chat_tokenizer, resize_embeddings_if_needed
except ImportError:  # pragma: no cover - supports direct script execution
    from chat_format import (  # type: ignore[no-redef]
        apply_generation_config,
        ensure_chat_tokenizer,
        resize_embeddings_if_needed,
    )

logger = logging.getLogger(__name__)


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_path: Optional[str] = None,
) -> str:
    """Load base + adapter on CPU, merge the LoRA weights and save the result."""
    from peft import PeftModel

    tokenizer = ensure_chat_tokenizer(AutoTokenizer.from_pretrained(tokenizer_path or adapter_path))
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    # The adapter was trained on the chat-token vocabulary; match it before loading.
    resize_embeddings_if_needed(model, tokenizer)
    apply_generation_config(model, tokenizer)
    merged = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged adapter %s into %s -> %s", adapter_path, base_model_path, output_dir)
    return output_dir
