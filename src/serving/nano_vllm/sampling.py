"""Sampling helpers for token-by-token decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SamplingOptions:
    # None means "engine default" (NanoVLLMConfig.max_new_tokens, capped to fit
    # max_model_len). An explicit value is never silently clamped.
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    stop: Optional[list[str]] = None
    # Extra token ids that end generation (vLLM extension); always honored,
    # unlike the model's EOS ids which ignore_eos can disable.
    stop_token_ids: Optional[list[int]] = None
    # Keep generating past the model's EOS ids (vLLM extension; benchmarks use
    # it to get exactly max_tokens per request).
    ignore_eos: bool = False


def filter_logits(
    logits: torch.Tensor,
    top_p: float = 1.0,
    top_k: int = -1,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to a 1D logits tensor."""
    if logits.ndim != 1:
        raise ValueError("filter_logits expects a 1D tensor")

    filtered = logits.clone()
    if top_k is not None and top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = float("-inf")

    return filtered


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = -1,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample one token id from the final-position logits."""
    if logits.ndim == 2:
        logits = logits[-1]
    if logits.ndim != 1:
        raise ValueError("sample_next_token expects a 1D or [seq, vocab] tensor")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    filtered = filter_logits(logits / temperature, top_p=top_p, top_k=top_k)
    probs = torch.softmax(filtered.float(), dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def sample_tokens(
    logits: torch.Tensor,
    options: list[SamplingOptions],
) -> list[int]:
    """Sample one token per row of [num_seqs, vocab] logits, with one host sync.

    Same rules as :func:`sample_next_token` per row: temperature <= 0 is
    greedy; otherwise top-k, then top-p over what is left, then multinomial,
    falling back to argmax when the filtered distribution is degenerate.
    """
    if logits.ndim != 2 or logits.shape[0] != len(options):
        raise ValueError("sample_tokens expects [num_seqs, vocab] logits, one row per option")
    greedy = logits.argmax(dim=-1)
    if all(option.temperature <= 0 for option in options):
        return greedy.tolist()

    device = logits.device
    vocab = logits.shape[-1]
    temperature = torch.tensor(
        [option.temperature if option.temperature > 0 else 1.0 for option in options],
        device=device,
    )
    top_k = torch.tensor(
        [
            option.top_k if option.top_k is not None and 0 < option.top_k < vocab else vocab
            for option in options
        ],
        device=device,
    )
    top_p = torch.tensor(
        [
            option.top_p if option.top_p is not None and 0.0 < option.top_p < 1.0 else 1.0
            for option in options
        ],
        device=device,
    )
    is_greedy = torch.tensor([option.temperature <= 0 for option in options], device=device)

    scaled = logits.float() / temperature.view(-1, 1)
    sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
    ranks = torch.arange(vocab, device=device).view(1, -1)
    sorted_logits = sorted_logits.masked_fill(ranks >= top_k.view(-1, 1), float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)
    # Drop a token once the mass before it already exceeds top_p (the first
    # token always stays).
    before = torch.cumsum(probs, dim=-1) - probs
    probs = probs.masked_fill(before > top_p.view(-1, 1), 0.0)
    total = probs.sum(dim=-1, keepdim=True)
    degenerate = ~torch.isfinite(probs).all(dim=-1, keepdim=True) | (total <= 0)
    # A degenerate row samples rank 0 (the argmax) instead of erroring out.
    fallback = (ranks == 0).to(probs.dtype)
    probs = torch.where(degenerate, fallback, probs)
    choice = torch.multinomial(probs, num_samples=1)
    sampled = sorted_indices.gather(-1, choice).squeeze(-1)
    return torch.where(is_greedy, greedy, sampled).tolist()
