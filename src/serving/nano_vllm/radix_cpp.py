"""The C++ radix tree (csrc/radix_tree.cpp): build it with nanobind and wrap it.

:class:`CppBlockManager` has BlockManager's interface (prefix_cache_backend:
cpp picks it) and keeps the Sequence fields the same way; ``radix_node`` is a
node id here. The extension is compiled once per source/nanobind/Python
version with the system C++ compiler (no CMake) into NANO_EXT_DIR (default
~/.cache/nano_vllm) and reused from there.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .block_manager import PrefixCacheStats
from .sequence import Sequence

logger = logging.getLogger(__name__)

SOURCE = Path(__file__).parent / "csrc" / "radix_tree.cpp"
_lock = threading.Lock()
_module = None


def load_radix_extension():
    """Import the compiled extension, building it on first use."""
    global _module
    with _lock:
        if _module is not None:
            return _module
        import nanobind

        key = SOURCE.read_bytes() + nanobind.__version__.encode() + sys.version.encode()
        digest = hashlib.sha256(key).hexdigest()[:16]
        root = Path(os.environ.get("NANO_EXT_DIR") or Path.home() / ".cache" / "nano_vllm")
        target = root / f"nano_radix_{digest}" / f"nano_radix{sysconfig.get_config_var('EXT_SUFFIX')}"
        if not target.exists():
            _build(nanobind, target)
        spec = importlib.util.spec_from_file_location("nano_radix", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module = module
        return module


def _build(nanobind, target: Path) -> None:
    nb = Path(nanobind.include_dir()).parent
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        out = Path(tmp) / target.name
        cmd = [
            os.environ.get("CXX", "c++"),
            "-std=c++17", "-O2", "-DNDEBUG", "-shared", "-fPIC",
            "-fvisibility=hidden", "-fno-strict-aliasing",
            f"-I{nb / 'include'}",
            f"-I{nb / 'ext' / 'robin_map' / 'include'}",
            f"-I{sysconfig.get_paths()['include']}",
            str(nb / "src" / "nb_combined.cpp"),
            str(SOURCE),
            "-o", str(out),
        ]
        if sys.platform == "darwin":
            cmd[1:1] = ["-undefined", "dynamic_lookup"]
        logger.info("building %s", target)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        os.replace(out, target)  # atomic: concurrent builders leave one file


class CppBlockManager:
    """BlockManager with the radix tree and block allocator in C++."""

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self._impl = load_radix_extension().RadixBlockManager(
            num_blocks, block_size, enable_prefix_cache
        )
        self._num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self._stats = PrefixCacheStats()
        self._synced: dict[int, int] = {}  # seq_id -> tokens already in C++

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def stats(self) -> PrefixCacheStats:
        self._stats.evicted_blocks = self._impl.evicted_blocks()  # counted in C++
        return self._stats

    @property
    def num_free_blocks(self) -> int:
        return self._impl.num_free_list() + self._impl.num_evictable()

    @property
    def num_cached_blocks(self) -> int:
        return self._impl.num_evictable()

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - self.num_free_blocks

    def _sync(self, seq: Sequence) -> None:
        pushed = self._synced.get(seq.seq_id)
        if pushed is None:
            self._impl.add_sequence(seq.seq_id, seq.token_ids)
        elif pushed < seq.num_tokens:
            self._impl.extend_sequence(seq.seq_id, seq.token_ids[pushed:])
        else:
            return
        self._synced[seq.seq_id] = seq.num_tokens

    def can_allocate(self, seq: Sequence) -> int:
        self._sync(seq)
        return self._impl.can_allocate(seq.seq_id)

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0) -> None:
        if seq.block_table:
            raise ValueError("sequence already has allocated blocks")
        self._sync(seq)
        table, node, shared = self._impl.allocate(seq.seq_id, num_cached_blocks)
        seq.block_table = table
        seq.radix_node = node
        seq.num_shared_blocks = shared
        seq.num_cached_tokens = shared * self.block_size
        hit_tokens = shared * self.block_size
        if seq.num_completion_tokens:
            self._stats.recompute_query_tokens += seq.num_tokens
            self._stats.recompute_hit_tokens += hit_tokens
        else:
            self._stats.query_tokens += seq.num_tokens
            self._stats.hit_tokens += hit_tokens

    def deallocate(self, seq: Sequence) -> None:
        seq.epoch += 1
        node = seq.radix_node if seq.radix_node is not None else -1
        self._impl.deallocate(node, seq.block_table[seq.num_shared_blocks :])
        if self._synced.pop(seq.seq_id, None) is not None:
            self._impl.remove_sequence(seq.seq_id)
        seq.block_table.clear()
        seq.radix_node = None
        seq.num_shared_blocks = 0
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        return self.num_free_blocks >= self._blocks_to_append(seq)

    def may_append(self, seq: Sequence) -> None:
        count = self._blocks_to_append(seq)
        if count:
            seq.block_table.extend(self._impl.take_blocks(count))

    def blocks_for_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def slot_for_position(self, seq: Sequence, position: int) -> int:
        return seq.block_table[position // self.block_size] * self.block_size + position % self.block_size

    def commit(self, seq: Sequence, num_computed_tokens: Optional[int] = None) -> None:
        if not self.enable_prefix_cache or seq.radix_node is None:
            return
        if num_computed_tokens is None:
            num_computed_tokens = seq.num_cached_tokens + seq.num_scheduled_tokens
        end = num_computed_tokens // self.block_size
        start = seq.num_shared_blocks
        if start >= end:
            return
        self._sync(seq)
        node, shared, swaps = self._impl.commit(
            seq.seq_id, seq.radix_node, start, end, seq.block_table[start:end]
        )
        for index, block in swaps:
            seq.block_table[index] = block
        seq.radix_node = node
        seq.num_shared_blocks = shared

    def reset_prefix_cache(self) -> int:
        return self._impl.reset_prefix_cache()

    def stats_dict(self) -> dict:
        s = self._stats
        return {
            "kv_blocks_total": self._num_blocks,
            "kv_blocks_free": self._impl.num_free_list(),
            "kv_blocks_cached": self._impl.num_evictable(),
            "kv_blocks_used": self.num_used_blocks,
            "prefix_query_tokens": s.query_tokens,
            "prefix_hit_tokens": s.hit_tokens,
            "prefix_hit_rate": s.hit_tokens / s.query_tokens if s.query_tokens else 0.0,
            "recompute_query_tokens": s.recompute_query_tokens,
            "recompute_hit_tokens": s.recompute_hit_tokens,
            "evicted_blocks": self._impl.evicted_blocks(),
        }

    # Inspection, for tests: the same shapes as BlockManager's.
    @property
    def free_block_ids(self) -> list[int]:
        return list(self._impl.free_list())

    def tree_dump(self) -> list[tuple]:
        return [(n, p, tuple(k), list(b), lock) for n, p, k, b, lock in self._impl.tree_dump()]

    def path_blocks(self, node) -> list[int]:
        return [] if node is None else list(self._impl.path_blocks(node))

    def _blocks_to_append(self, seq: Sequence) -> int:
        needed = -(-seq.num_logical_tokens // self.block_size)
        return max(0, needed - len(seq.block_table))
