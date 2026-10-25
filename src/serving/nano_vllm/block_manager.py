"""Paged KV block allocation with a radix-tree prefix cache.

Every physical block is in one of three states:
  free    in ``free_block_ids``, holding nothing worth keeping
  cached  owned by the radix tree and locked by no sequence (evictable)
  used    in some sequence's block table: either a tree block on a path the
          sequence has locked, or a private block only that sequence owns

The radix tree is block-granular (SGLang's RadixAttention, at block size):
each node's edge is a run of whole blocks, its key the token ids they hold.
A sequence's block table is a prefix of tree blocks (``num_shared_blocks``,
the path to ``seq.radix_node`` locked) followed by private blocks. When a step
completes full blocks, :meth:`commit` hands them to the tree; when the
sequence finishes, aborts or is preempted, the path is unlocked and the
private blocks are freed, while the tree keeps the shared ones until eviction
reclaims the least recently used unlocked leaves.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .sequence import Sequence


class RadixNode:
    __slots__ = ("key", "blocks", "children", "parent", "lock_ref", "last_access")

    def __init__(self, key: tuple[int, ...], blocks: list[int], parent: Optional["RadixNode"]):
        self.key = key  # len(blocks) * block_size token ids
        self.blocks = blocks
        self.children: dict[tuple[int, ...], RadixNode] = {}  # by the first block's tokens
        self.parent = parent
        self.lock_ref = 0  # sequences whose locked path includes this node
        self.last_access = 0


@dataclass
class PrefixCacheStats:
    """Counters since the start; the engine exports them as metrics."""

    query_tokens: int = 0  # prompt tokens of admitted requests
    hit_tokens: int = 0  # of those, served from the cache
    recompute_query_tokens: int = 0  # the same for preempted sequences coming back
    recompute_hit_tokens: int = 0
    evicted_blocks: int = 0


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
        self._num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.root = RadixNode((), [], None)
        self.stats = PrefixCacheStats()
        self._evictable = 0  # blocks in unlocked tree nodes
        self._clock = itertools.count(1)
        self._leaf_heap: list[tuple[int, int, RadixNode]] = []  # (last_access, tiebreak, node)
        self._tiebreak = itertools.count()

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        """Blocks an allocation can get: free ones plus evictable cached ones."""
        return len(self.free_block_ids) + self._evictable

    @property
    def num_cached_blocks(self) -> int:
        return self._evictable

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - self.num_free_blocks

    # --- allocation ---------------------------------------------------------

    def can_allocate(self, seq: Sequence) -> int:
        """Return the number of prefix-cache hit blocks, or -1 if it won't fit."""
        hit_blocks, node, matched = self._match(seq)
        if self._blocks_needed(seq, hit_blocks, node, matched) > self.num_free_blocks:
            return -1
        return len(hit_blocks)

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0) -> None:
        """Allocate all blocks for ``seq`` or, on failure, change nothing."""
        if seq.block_table:
            raise ValueError("sequence already has allocated blocks")
        hit_blocks, node, matched = self._match(seq, max_blocks=num_cached_blocks)
        if len(hit_blocks) < num_cached_blocks:
            raise RuntimeError("prefix cache changed since can_allocate()")
        if self._blocks_needed(seq, hit_blocks, node, matched) > self.num_free_blocks:
            raise RuntimeError("No free KV blocks available")

        if node is not self.root and matched < len(node.blocks):
            node = self._split(node, matched)
        # Lock the hits before allocating: eviction must not reclaim them.
        self._lock(node)
        self._touch(node)
        fresh = seq.num_blocks - len(hit_blocks)
        self._reserve(fresh)
        seq.block_table = list(hit_blocks) + [self.free_block_ids.popleft() for _ in range(fresh)]
        seq.radix_node = node
        seq.num_shared_blocks = len(hit_blocks)
        seq.num_cached_tokens = len(hit_blocks) * self.block_size

        hit_tokens = len(hit_blocks) * self.block_size
        if seq.num_completion_tokens:  # recomputing after a preemption
            self.stats.recompute_query_tokens += seq.num_tokens
            self.stats.recompute_hit_tokens += hit_tokens
        else:
            self.stats.query_tokens += seq.num_tokens
            self.stats.hit_tokens += hit_tokens

    def deallocate(self, seq: Sequence) -> None:
        """Unlock the sequence's tree path and free its private blocks."""
        seq.epoch += 1
        for block_id in seq.block_table[seq.num_shared_blocks :]:
            self.free_block_ids.append(block_id)
        if seq.radix_node is not None:
            self._unlock(seq.radix_node)
        seq.block_table.clear()
        seq.radix_node = None
        seq.num_shared_blocks = 0
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        return self.num_free_blocks >= self._blocks_to_append(seq)

    def may_append(self, seq: Sequence) -> None:
        count = self._blocks_to_append(seq)
        self._reserve(count)
        for _ in range(count):
            seq.block_table.append(self.free_block_ids.popleft())

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def slot_for_position(self, seq: Sequence, position: int) -> int:
        block_index = position // self.block_size
        block_offset = position % self.block_size
        return seq.block_table[block_index] * self.block_size + block_offset

    # --- prefix cache -------------------------------------------------------

    def commit(self, seq: Sequence, num_computed_tokens: Optional[int] = None) -> None:
        """Hand the blocks a finished step completed to the tree.

        Blocks fully covered by ``num_computed_tokens`` (default: before
        ``num_cached_tokens`` advances, ``num_cached_tokens +
        num_scheduled_tokens``) become tree blocks, and the sequence's lock
        moves down to the node that ends with them. A block whose content the
        tree already holds is swapped for the tree's copy and freed (the step
        that wrote it has finished).
        """
        if not self.enable_prefix_cache or seq.radix_node is None:
            return
        bs = self.block_size
        if num_computed_tokens is None:
            num_computed_tokens = seq.num_cached_tokens + seq.num_scheduled_tokens
        end = num_computed_tokens // bs
        index = seq.num_shared_blocks
        if index >= end:
            return
        tokens = seq.token_ids
        table = seq.block_table
        node = seq.radix_node
        while index < end:
            child = node.children.get(tuple(tokens[index * bs : (index + 1) * bs]))
            if child is None:
                key = tuple(tokens[index * bs : end * bs])
                blocks = table[index:end]
                if node is not self.root and not node.children and node.lock_ref == 1:
                    # Only this sequence uses the leaf: extend its edge, so
                    # decoding grows one node instead of a chain of 1-block nodes.
                    node.key += key
                    node.blocks += blocks
                else:
                    new = RadixNode(key, blocks, node)
                    node.children[key[:bs]] = new
                    self._evictable += len(blocks)  # unlocked until _move_lock
                    self._move_lock(node, new)
                    node = new
                index = end
                break
            matched = self._match_edge(child, tokens, index, end)
            for offset in range(matched):
                tree_block = child.blocks[offset]
                if table[index + offset] != tree_block:  # same content, other copy
                    self.free_block_ids.append(table[index + offset])
                    table[index + offset] = tree_block
            if matched < len(child.blocks):
                child = self._split(child, matched)
            self._move_lock(node, child)
            node = child
            index += matched
        self._touch(node)
        seq.radix_node = node
        seq.num_shared_blocks = end

    def reset_prefix_cache(self) -> int:
        """Evict every unlocked tree block; returns how many were freed."""
        freed = self._evictable
        self._evict(freed)
        return freed

    def stats_dict(self) -> dict:
        s = self.stats
        return {
            "kv_blocks_total": self._num_blocks,
            "kv_blocks_free": len(self.free_block_ids),
            "kv_blocks_cached": self._evictable,
            "kv_blocks_used": self.num_used_blocks,
            "prefix_query_tokens": s.query_tokens,
            "prefix_hit_tokens": s.hit_tokens,
            "prefix_hit_rate": s.hit_tokens / s.query_tokens if s.query_tokens else 0.0,
            "recompute_query_tokens": s.recompute_query_tokens,
            "recompute_hit_tokens": s.recompute_hit_tokens,
            "evicted_blocks": s.evicted_blocks,
        }

    # Inspection, for tests (CppBlockManager has the same two).
    def tree_dump(self) -> list[tuple]:
        """(node, parent, key, blocks, lock_ref) for every node but the root."""
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                parent = 0 if node is self.root else id(node)
                out.append((id(child), parent, child.key, list(child.blocks), child.lock_ref))
                stack.append(child)
        return out

    def path_blocks(self, node) -> list[int]:
        blocks: list[int] = []
        while node is not None and node is not self.root:
            blocks = node.blocks + blocks
            node = node.parent
        return blocks

    # --- internals ------------------------------------------------------------

    def _match(
        self, seq: Sequence, max_blocks: Optional[int] = None
    ) -> tuple[list[int], RadixNode, int]:
        """Longest cached prefix of whole blocks: (blocks, deepest node, blocks
        matched in that node). The last block is never reused, so at least one
        token gets computed (its logits are needed)."""
        if not self.enable_prefix_cache:
            return [], self.root, 0
        bs = self.block_size
        limit = max(0, seq.num_blocks - 1)
        if max_blocks is not None:
            limit = min(limit, max_blocks)
        tokens = seq.token_ids
        node, matched, index = self.root, 0, 0
        blocks: list[int] = []
        while index < limit:
            child = node.children.get(tuple(tokens[index * bs : (index + 1) * bs]))
            if child is None:
                break
            matched = self._match_edge(child, tokens, index, limit)
            blocks.extend(child.blocks[:matched])
            node, index = child, index + matched
            if matched < len(child.blocks):
                break
        if node is self.root:
            matched = 0
        return blocks, node, matched

    def _match_edge(self, node: RadixNode, tokens: list[int], index: int, limit: int) -> int:
        """Whole blocks of ``node``'s edge equal to tokens from block ``index``
        on (at most ``limit - index``); the first block is known to match."""
        bs = self.block_size
        matched = 1
        while matched < len(node.blocks) and index + matched < limit:
            start = (index + matched) * bs
            if tuple(tokens[start : start + bs]) != node.key[matched * bs : (matched + 1) * bs]:
                break
            matched += 1
        return matched

    def _blocks_needed(
        self, seq: Sequence, hit_blocks: list[int], node: RadixNode, matched: int
    ) -> int:
        # Hits in unlocked nodes stop being evictable once locked, so they
        # cost free capacity just like fresh blocks.
        revived = 0
        walk = node
        count = matched
        while walk is not self.root:
            if walk.lock_ref == 0:
                revived += count
            walk = walk.parent
            count = len(walk.blocks) if walk is not None else 0
        return seq.num_blocks - len(hit_blocks) + revived

    def _blocks_to_append(self, seq: Sequence) -> int:
        # Blocks needed so every token, including the one about to be decoded
        # (possibly still in flight), has a KV slot (block_size == 1 needs a
        # block on every step).
        needed = -(-seq.num_logical_tokens // self.block_size)
        return max(0, needed - len(seq.block_table))

    def _split(self, node: RadixNode, matched: int) -> RadixNode:
        """Cut ``node``'s edge after ``matched`` blocks; return the new upper part."""
        bs = self.block_size
        upper = RadixNode(node.key[: matched * bs], node.blocks[:matched], node.parent)
        upper.lock_ref = node.lock_ref
        upper.last_access = node.last_access
        node.parent.children[node.key[:bs]] = upper
        node.key = node.key[matched * bs :]
        node.blocks = node.blocks[matched:]
        node.parent = upper
        upper.children[node.key[:bs]] = node
        return upper

    def _lock(self, node: RadixNode) -> None:
        while node is not self.root:
            if node.lock_ref == 0:
                self._evictable -= len(node.blocks)
            node.lock_ref += 1
            node = node.parent

    def _unlock(self, node: RadixNode) -> None:
        while node is not self.root:
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self._evictable += len(node.blocks)
                if not node.children:
                    self._push_leaf(node)
            node = node.parent

    def _move_lock(self, upper: RadixNode, lower: RadixNode) -> None:
        """Move a sequence's lock from ``upper`` down to its descendant ``lower``."""
        self._lock(lower)
        self._unlock(upper)

    def _touch(self, node: RadixNode) -> None:
        now = next(self._clock)
        while node is not self.root:
            node.last_access = now
            node = node.parent

    def _push_leaf(self, node: RadixNode) -> None:
        heapq.heappush(self._leaf_heap, (node.last_access, next(self._tiebreak), node))

    def _reserve(self, count: int) -> None:
        """Make sure the free list holds ``count`` blocks, evicting if needed."""
        missing = count - len(self.free_block_ids)
        if missing > 0:
            if missing > self._evictable:
                raise RuntimeError("No free KV blocks available")
            self._evict(missing)

    def _evict(self, count: int) -> None:
        """Free ``count`` blocks from the tails of the least recently used unlocked leaves."""
        bs = self.block_size
        while count > 0:
            if not self._leaf_heap:
                raise RuntimeError("prefix cache accounting is inconsistent: no evictable leaf")
            last_access, _, node = heapq.heappop(self._leaf_heap)
            if (
                node.parent is None
                or node.children
                or node.lock_ref
                or not node.blocks
                or last_access != node.last_access
            ):
                continue  # stale entry
            take = min(count, len(node.blocks))
            first_block_key = node.key[:bs]
            self.free_block_ids.extend(node.blocks[-take:])
            del node.blocks[-take:]
            node.key = node.key[: len(node.blocks) * bs]
            self._evictable -= take
            self.stats.evicted_blocks += take
            count -= take
            if node.blocks:
                self._push_leaf(node)
            else:
                parent = node.parent
                del parent.children[first_block_key]
                node.parent = None
                if parent is not self.root and not parent.children and parent.lock_ref == 0:
                    self._push_leaf(parent)
