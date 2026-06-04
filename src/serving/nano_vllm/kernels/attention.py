"""Readable attention helpers for the nano decoder."""

from __future__ import annotations

import math

import torch


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    use_triton: bool = False,
) -> torch.Tensor:
    """Compute causal attention for [batch, heads, tokens, head_dim] tensors."""
    del use_triton  # v0.1 keeps attention readable before adding paged kernels.
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be 4D tensors")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1]:
        raise ValueError("query and key batch/head dimensions must match")

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    key_positions = torch.arange(key.shape[-2], device=query.device)
    mask = key_positions.view(1, -1) <= query_positions.to(query.device).view(-1, 1)
    scores = scores.masked_fill(~mask.view(1, 1, mask.shape[0], mask.shape[1]), -1e30)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    return torch.matmul(probs, value)


def decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: int,
    use_triton: bool = False,
) -> torch.Tensor:
    """Single-token decode attention wrapper."""
    positions = torch.tensor([position], dtype=torch.long, device=query.device)
    return causal_attention(query, key, value, positions, use_triton=use_triton)
