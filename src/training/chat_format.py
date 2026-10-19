"""
One chat format for SFT, reward modeling, PPO and serving.

Every stage renders prompts with the Llama 3 chat template stored on the
tokenizer, so the format the policy is trained on is the format the reward
model scores and the format vLLM serves (the template is saved with the
checkpoint tokenizer, as chat_template.jinja on transformers 4.57).

Llama 3 base checkpoints ship the header/eot special tokens but their
embeddings are untrained: full fine-tuning trains them, LoRA runs need
embed_tokens/lm_head in modules_to_save.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
END_OF_TURN = "<|eot_id|>"
END_OF_TEXT = "<|end_of_text|>"
CHAT_SPECIAL_TOKENS = (START_HEADER, END_HEADER, END_OF_TURN)

# Dedicated pad tokens, in preference order. Padding with EOS hides the real
# EOS from the loss masks (and from PPO), so the model never learns to stop.
# Llama 3.1 names id 128004 <|finetune_right_pad_id|>; Llama 3.0 has it as a
# reserved special token. <|pad|> is only added for tokenizers that have neither.
PAD_TOKEN_CANDIDATES = (
    "<|finetune_right_pad_id|>",
    "<|reserved_special_token_0|>",
)
FALLBACK_PAD_TOKEN = "<|pad|>"

# Meta's Llama 3 Instruct template.
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    " + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


def ensure_chat_tokenizer(tokenizer: Any) -> Any:
    """
    Give the tokenizer the chat template, a dedicated pad token and <|eot_id|> as EOS.

    Idempotent: a tokenizer saved from an SFT checkpoint passes through unchanged.
    Tokens are only added when missing (not the case for Llama 3 tokenizers);
    callers must then resize the model embeddings, see resize_embeddings_if_needed.
    """
    vocab = tokenizer.get_vocab()
    missing = [token for token in CHAT_SPECIAL_TOKENS if token not in vocab]
    if missing:
        logger.warning("Adding missing chat special tokens to tokenizer: %s", missing)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_additional_special_tokens=False,
        )

    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    stop_tokens = {tokenizer.eos_token, END_OF_TURN, END_OF_TEXT}
    if tokenizer.pad_token is None or tokenizer.pad_token in stop_tokens:
        vocab = tokenizer.get_vocab()
        pad_token = next((token for token in PAD_TOKEN_CANDIDATES if token in vocab), None)
        if pad_token is None:
            tokenizer.add_special_tokens({"pad_token": FALLBACK_PAD_TOKEN})
        else:
            tokenizer.pad_token = pad_token

    # Generation ends at the end of the assistant turn.
    tokenizer.eos_token = END_OF_TURN
    return tokenizer


def stop_token_ids(tokenizer: Any) -> List[int]:
    """Token ids that end a generation: <|eot_id|> plus <|end_of_text|> when present."""
    vocab = tokenizer.get_vocab()
    ids = [vocab[END_OF_TURN]]
    if END_OF_TEXT in vocab and vocab[END_OF_TEXT] not in ids:
        ids.append(vocab[END_OF_TEXT])
    return ids


def resize_embeddings_if_needed(model: Any, tokenizer: Any) -> None:
    """Grow the embedding matrix when ensure_chat_tokenizer added tokens."""
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        logger.warning("Resizing token embeddings to %d", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))


def apply_generation_config(model: Any, tokenizer: Any) -> None:
    """Make generate()/vLLM stop on the chat stop tokens and pad with the pad token."""
    generation_config = model.generation_config
    generation_config.eos_token_id = stop_token_ids(tokenizer)
    generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def user_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    """Build the message list for a single-turn prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def format_prompt(tokenizer: Any, prompt: str, system: Optional[str] = None) -> str:
    """Render a prompt up to and including the assistant header."""
    return tokenizer.apply_chat_template(
        user_messages(prompt, system),
        tokenize=False,
        add_generation_prompt=True,
    )


def format_completion(completion: str) -> str:
    """Render the assistant reply the way the template closes a turn."""
    return f"{str(completion).strip()}{END_OF_TURN}"


def encode(tokenizer: Any, text: str) -> List[int]:
    """Tokenize already-templated text; the template supplies BOS itself."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_prompt_and_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> tuple[List[int], List[int]]:
    """
    Token ids for the templated prompt and completion.

    The two parts are tokenized separately so the prompt/response boundary used
    for loss masking and reward scoring is exact.
    """
    return encode(tokenizer, format_prompt(tokenizer, prompt)), encode(
        tokenizer, format_completion(completion)
    )


def pad_to_length(
    sequences: Sequence[Sequence[int]],
    length: int,
    pad_value: int,
    left: bool = False,
) -> List[List[int]]:
    """Pad every sequence to `length` on the right (or left)."""
    padded = []
    for sequence in sequences:
        padding = [pad_value] * (length - len(sequence))
        padded.append(padding + list(sequence) if left else list(sequence) + padding)
    return padded
