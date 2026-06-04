"""Paged KV block allocation and prefix-cache bookkeeping."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

from .sequence import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    content_hash: int = -1
    token_ids: list[int] = field(default_factory=list)

    def reset_for_use(self) -> None:
        self.ref_count = 1
        self.content_hash = -1
        self.token_ids = []

    def update_hash(self, content_hash: int, token_ids: list[int]) -> None:
        self.content_hash = content_hash
        self.token_ids = list(token_ids)


class BlockManager:
    """Manage fixed-size KV blocks for paged attention."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_cache: bool = True,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks = [Block(block_id) for block_id in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = {}

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        digest = hashlib.blake2b(digest_size=8)
        digest.update(prefix.to_bytes(8, "little", signed=True))
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, "little", signed=True))
        return int.from_bytes(digest.digest(), "little", signed=False)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def can_allocate(self, seq: Sequence) -> int:
        cached_blocks = self._count_prefix_cache_hits(seq)
        new_blocks = seq.num_blocks - cached_blocks
        if self.num_free_blocks < new_blocks:
            return -1
        return cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0) -> None:
        if seq.block_table:
            raise ValueError("sequence already has allocated blocks")
        previous_hash = -1
        for block_index in range(num_cached_blocks):
            token_ids = seq.block(block_index)
            previous_hash = self.compute_hash(token_ids, previous_hash)
            block_id = self.hash_to_block_id[previous_hash]
            block = self.blocks[block_id]
            if block_id not in self.used_block_ids:
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                block.ref_count = 1
            else:
                block.ref_count += 1
            seq.block_table.append(block_id)

        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_block(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        needs_new_block_for_last_token = len(seq) % self.block_size == 1
        return self.num_free_blocks >= int(needs_new_block_for_last_token)

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence) -> None:
        if not self.enable_prefix_cache or not seq.block_table:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return

        previous_hash = (
            -1 if start == 0 else self.blocks[seq.block_table[start - 1]].content_hash
        )
        for block_index in range(start, end):
            block = self.blocks[seq.block_table[block_index]]
            token_ids = seq.block(block_index)
            content_hash = self.compute_hash(token_ids, previous_hash)
            block.update_hash(content_hash, token_ids)
            self.hash_to_block_id[content_hash] = block.block_id
            previous_hash = content_hash

    def slot_for_position(self, seq: Sequence, position: int) -> int:
        block_index = position // self.block_size
        block_offset = position % self.block_size
        return seq.block_table[block_index] * self.block_size + block_offset

    def _count_prefix_cache_hits(self, seq: Sequence) -> int:
        if not self.enable_prefix_cache:
            return 0
        previous_hash = -1
        cached_blocks = 0
        for block_index in range(max(0, seq.num_blocks - 1)):
            token_ids = seq.block(block_index)
            previous_hash = self.compute_hash(token_ids, previous_hash)
            block_id = self.hash_to_block_id.get(previous_hash, -1)
            if block_id == -1:
                break
            if self.blocks[block_id].token_ids != token_ids:
                break
            cached_blocks += 1
        return cached_blocks

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("No free KV blocks available")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        if (
            block.content_hash != -1
            and self.hash_to_block_id.get(block.content_hash) == block_id
        ):
            del self.hash_to_block_id[block.content_hash]
        block.reset_for_use()
        self.used_block_ids.add(block_id)
        return block_id

    def _release_block(self, block_id: int) -> None:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
