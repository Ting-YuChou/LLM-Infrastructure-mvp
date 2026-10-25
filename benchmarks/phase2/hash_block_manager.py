"""The hash-chained prefix cache BlockManager as of Phase 1 (commit 8e049e7),
kept verbatim as the baseline for prefix_bench.py."""


from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

from src.serving.nano_vllm.sequence import Sequence


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
        # Signed, so a hash fed back in as the next block's prefix fits in 8
        # bytes (an unsigned value >= 2**63 overflowed to_bytes above).
        return int.from_bytes(digest.digest(), "little", signed=True)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def can_allocate(self, seq: Sequence) -> int:
        """Return the number of prefix-cache hit blocks, or -1 if it won't fit."""
        hit_block_ids = self._prefix_cache_hit_block_ids(seq)
        if self._free_blocks_needed(seq, hit_block_ids) > self.num_free_blocks:
            return -1
        return len(hit_block_ids)

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0) -> None:
        """Allocate all blocks for ``seq`` or, on failure, change nothing."""
        if seq.block_table:
            raise ValueError("sequence already has allocated blocks")
        hit_block_ids = self._prefix_cache_hit_block_ids(seq)
        if num_cached_blocks > len(hit_block_ids):
            raise RuntimeError("prefix cache changed since can_allocate()")
        hit_block_ids = hit_block_ids[:num_cached_blocks]
        if self._free_blocks_needed(seq, hit_block_ids) > self.num_free_blocks:
            raise RuntimeError("No free KV blocks available")

        block_table: list[int] = []
        try:
            for block_id in hit_block_ids:
                block = self.blocks[block_id]
                if block_id not in self.used_block_ids:
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                    block.ref_count = 1
                else:
                    block.ref_count += 1
                block_table.append(block_id)
            for _ in range(num_cached_blocks, seq.num_blocks):
                block_table.append(self._allocate_block())
        except BaseException:
            self._release_blocks(block_table)
            raise
        seq.block_table = block_table
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        self._release_blocks(seq.block_table)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        return self.num_free_blocks >= self._blocks_to_append(seq)

    def may_append(self, seq: Sequence) -> None:
        for _ in range(self._blocks_to_append(seq)):
            seq.block_table.append(self._allocate_block())

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

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

    def _blocks_to_append(self, seq: Sequence) -> int:
        # Blocks needed so every token, including the one about to be decoded,
        # has a KV slot (block_size == 1 needs a block on every step).
        return max(0, seq.num_blocks - len(seq.block_table))

    def _free_blocks_needed(self, seq: Sequence, hit_block_ids: list[int]) -> int:
        # Cache hits with ref_count 0 still sit in the free list, so reviving
        # them consumes free blocks just like fresh allocations do.
        revived = sum(
            1 for block_id in set(hit_block_ids) if block_id not in self.used_block_ids
        )
        return seq.num_blocks - len(hit_block_ids) + revived

    def _prefix_cache_hit_block_ids(self, seq: Sequence) -> list[int]:
        if not self.enable_prefix_cache:
            return []
        previous_hash = -1
        hit_block_ids: list[int] = []
        # The last block is never reused so at least one token gets computed.
        for block_index in range(max(0, seq.num_blocks - 1)):
            token_ids = seq.block(block_index)
            previous_hash = self.compute_hash(token_ids, previous_hash)
            block_id = self.hash_to_block_id.get(previous_hash, -1)
            if block_id == -1:
                break
            block = self.blocks[block_id]
            if block.content_hash != previous_hash or block.token_ids != token_ids:
                break
            hit_block_ids.append(block_id)
        return hit_block_ids

    def _release_blocks(self, block_ids: list[int]) -> None:
        for block_id in reversed(block_ids):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_block(block_id)

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
