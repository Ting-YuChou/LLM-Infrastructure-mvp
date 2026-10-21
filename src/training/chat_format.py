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
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

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


def validate_messages(messages: Sequence[Mapping[str, Any]], last_role: str) -> List[Dict[str, str]]:
    """
    Check a conversation and return it as plain {"role", "content"} dicts.

    An optional system message comes first, then user and assistant turns
    alternate starting with user, and the last message has role `last_role`
    ("user" for a prompt, "assistant" for a training conversation).
    """
    turns = [{"role": str(message["role"]), "content": str(message["content"])} for message in messages]
    body = turns[1:] if turns and turns[0]["role"] == "system" else turns
    expected = [("user", "assistant")[index % 2] for index in range(len(body))]
    if not body or [turn["role"] for turn in body] != expected or body[-1]["role"] != last_role:
        roles = [turn["role"] for turn in turns]
        raise ValueError(
            f"Conversation roles must be [system,] user, assistant, ... ending with {last_role}; got {roles}"
        )
    return turns


def prompt_messages(prompt: Union[str, Sequence[Mapping[str, Any]]], system: Optional[str] = None) -> List[Dict[str, str]]:
    """Messages for a prompt given as text or as a conversation that ends with a user turn."""
    if isinstance(prompt, str):
        return user_messages(prompt, system)
    if system:
        raise ValueError("Pass the system message inside the conversation")
    return validate_messages(prompt, last_role="user")


def format_prompt(
    tokenizer: Any,
    prompt: Union[str, Sequence[Mapping[str, Any]]],
    system: Optional[str] = None,
) -> str:
    """Render a prompt (text or conversation) up to and including the assistant header."""
    return tokenizer.apply_chat_template(
        prompt_messages(prompt, system),
        tokenize=False,
        add_generation_prompt=True,
    )


def format_completion(completion: str) -> str:
    """Render the assistant reply the way the template closes a turn."""
    return f"{str(completion).strip()}{END_OF_TURN}"


def encode(tokenizer: Any, text: str) -> List[int]:
    """Tokenize already-templated text; the template supplies BOS itself."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


class EncodedConversation(NamedTuple):
    input_ids: List[int]
    labels: List[int]
    # len(input_ids) after each assistant turn: the places a conversation can be cut.
    turn_ends: List[int]


def encode_conversation(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> EncodedConversation:
    """
    Token ids and SFT labels for a conversation that ends with an assistant turn.

    Every assistant reply (its content and closing <|eot_id|>) is labelled; the
    system and user turns and the assistant headers are -100. Each piece ends
    or starts at a template special token, so tokenizing the pieces separately
    gives the same ids as tokenizing the whole rendered conversation, and a
    reply's ids are the format_completion ids that RM and PPO use.
    """
    turns = validate_messages(messages, last_role="assistant")
    input_ids: List[int] = []
    labels: List[int] = []
    turn_ends: List[int] = []
    rendered = ""
    for index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(turns[:index], tokenize=False, add_generation_prompt=True)
        if not prompt.startswith(rendered):
            raise ValueError("The chat template does not render a conversation as a prefix of its continuation")
        prompt_ids = encode(tokenizer, prompt[len(rendered):])
        reply = format_completion(turn["content"])
        reply_ids = encode(tokenizer, reply)
        input_ids += prompt_ids + reply_ids
        labels += [-100] * len(prompt_ids) + reply_ids
        turn_ends.append(len(input_ids))
        rendered = prompt + reply
    return EncodedConversation(input_ids, labels, turn_ends)


def encode_prompt_and_completion(
    tokenizer: Any,
    prompt: Union[str, Sequence[Mapping[str, Any]]],
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


class PaddingCollator:
    """
    Pad each batch to its longest row instead of padding every row to max_length.

    All list-valued features in a batch share one length, so the chosen and
    rejected halves of a preference batch can be concatenated. *input_ids pad
    with the pad token, *labels with -100 and the rest (attention masks) with 0,
    on the right; scalar features such as DPO reference log-probs are stacked.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def pad_value(self, key: str) -> int:
        if key.endswith("input_ids"):
            return self.pad_token_id
        if key.endswith("labels"):
            return -100
        return 0

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        rows = [
            {key: value.tolist() if hasattr(value, "tolist") else value for key, value in feature.items()}
            for feature in features
        ]
        sequence_keys = [key for key, value in rows[0].items() if isinstance(value, (list, tuple))]
        length = max((len(row[key]) for row in rows for key in sequence_keys), default=0)
        if self.pad_to_multiple_of:
            length = -(-length // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch: Dict[str, Any] = {}
        for key in rows[0]:
            if key in sequence_keys:
                padded = pad_to_length([row[key] for row in rows], length, self.pad_value(key))
                batch[key] = torch.tensor(padded, dtype=torch.long)
            else:
                batch[key] = torch.tensor([row[key] for row in rows])
        return batch
