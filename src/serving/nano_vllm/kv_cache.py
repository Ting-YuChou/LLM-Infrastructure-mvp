"""Static KV cache used by the educational nano decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class StaticKVCache:
    """Fixed-size per-layer key/value cache.

    Layout is [batch, kv_heads, max_seq_len, head_dim]. v0.1 intentionally uses
    a simple contiguous cache before introducing paged/block allocation.
    """

    num_layers: int
    batch_size: int
    num_key_value_heads: int
    max_seq_len: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device | str

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.max_seq_len,
            self.head_dim,
        )
        self.keys = [
            torch.empty(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.values = [
            torch.empty(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.lengths = torch.zeros(
            self.num_layers, dtype=torch.long, device=self.device
        )

    def update(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write [batch, kv_heads, tokens, head_dim] key/value slices."""
        if positions.ndim != 1:
            raise ValueError("positions must be a 1D tensor")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        expected_prefix = (self.batch_size, self.num_key_value_heads, positions.numel())
        if tuple(key.shape[:3]) != expected_prefix:
            raise ValueError(
                "key/value shape must be "
                f"{expected_prefix + (self.head_dim,)}, got {tuple(key.shape)}"
            )
        if int(positions.max().item()) >= self.max_seq_len:
            raise ValueError("KV cache max_seq_len exceeded")

        positions = positions.to(device=self.device, dtype=torch.long)
        self.keys[layer_idx][:, :, positions, :] = key.to(self.device)
        self.values[layer_idx][:, :, positions, :] = value.to(self.device)
        self.lengths[layer_idx] = torch.maximum(
            self.lengths[layer_idx],
            positions.max() + 1,
        )

    def read(
        self,
        layer_idx: int,
        end_position: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = (
            int(self.lengths[layer_idx].item())
            if end_position is None
            else end_position
        )
        end = min(end, self.max_seq_len)
        return (
            self.keys[layer_idx][:, :, :end, :],
            self.values[layer_idx][:, :, :end, :],
        )

    def reset(self) -> None:
        self.lengths.zero_()
