"""Rotary positional embedding helper."""

from __future__ import annotations

import torch


def _apply_rope_one(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000.0,
    use_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors shaped [batch, heads, tokens, head_dim]."""
    del use_triton  # v0.1 keeps RoPE as a readable PyTorch helper.
    if query.shape[-1] % 2 != 0 or key.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    device = query.device
    dtype = query.dtype
    head_dim = query.shape[-1]
    positions = positions.to(device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype=dtype).view(1, 1, positions.numel(), -1)
    sin = freqs.sin().to(dtype=dtype).view(1, 1, positions.numel(), -1)
    return _apply_rope_one(query, cos, sin), _apply_rope_one(key, cos, sin)
