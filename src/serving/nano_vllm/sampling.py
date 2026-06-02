"""Sampling helpers for token-by-token decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SamplingOptions:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    stop: Optional[list[str]] = None


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
