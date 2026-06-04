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


@dataclass
class PagedKVCache:
    """Global block-addressed KV cache.

    Layout is [num_blocks, block_size, kv_heads, head_dim] per layer. Scheduler
    state maps each sequence's logical token positions to physical cache slots.
    """

    num_layers: int
    num_blocks: int
    block_size: int
    num_key_value_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device | str

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        shape = (
            self.num_blocks,
            self.block_size,
            self.num_key_value_heads,
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

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    def update(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write [tokens, kv_heads, head_dim] key/value slices into physical slots."""
        if slot_mapping.ndim != 1:
            raise ValueError("slot_mapping must be 1D")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        expected = (slot_mapping.numel(), self.num_key_value_heads, self.head_dim)
        if tuple(key.shape) != expected:
            raise ValueError(
                f"key/value shape must be {expected}, got {tuple(key.shape)}"
            )

        slot_mapping = slot_mapping.to(device=self.device, dtype=torch.long)
        if slot_mapping.numel() == 0:
            return
        if int(slot_mapping.max().item()) >= self.total_slots:
            raise ValueError("slot_mapping references a slot outside the KV cache")

        flat_keys = self.keys[layer_idx].view(
            self.total_slots,
            self.num_key_value_heads,
            self.head_dim,
        )
        flat_values = self.values[layer_idx].view(
            self.total_slots,
            self.num_key_value_heads,
            self.head_dim,
        )
        flat_keys[slot_mapping] = key.to(device=self.device)
        flat_values[slot_mapping] = value.to(device=self.device)

    def read_sequence(
        self,
        layer_idx: int,
        block_table: list[int] | torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one sequence's cache as [1, kv_heads, seq_len, head_dim]."""
        if seq_len <= 0:
            empty = torch.empty(
                (1, self.num_key_value_heads, 0, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty

        if isinstance(block_table, torch.Tensor):
            blocks = [int(item) for item in block_table.tolist() if int(item) >= 0]
        else:
            blocks = list(block_table)
        required_blocks = (seq_len + self.block_size - 1) // self.block_size
        if len(blocks) < required_blocks:
            raise ValueError("block_table does not cover requested sequence length")

        slots = []
        for position in range(seq_len):
            block_id = blocks[position // self.block_size]
            offset = position % self.block_size
            slots.append(block_id * self.block_size + offset)
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=self.device)
        flat_keys = self.keys[layer_idx].view(
            self.total_slots,
            self.num_key_value_heads,
            self.head_dim,
        )
        flat_values = self.values[layer_idx].view(
            self.total_slots,
            self.num_key_value_heads,
            self.head_dim,
        )
        key = flat_keys.index_select(0, slot_tensor)
        value = flat_values.index_select(0, slot_tensor)
        return (
            key.transpose(0, 1).unsqueeze(0).contiguous(),
            value.transpose(0, 1).unsqueeze(0).contiguous(),
        )
