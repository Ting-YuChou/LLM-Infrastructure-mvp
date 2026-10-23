"""The radix-tree prefix cache in BlockManager: sharing, splits, eviction, locks.

The property test drives random admissions, steps and finishes and checks,
after every operation, that block ownership is a partition, that lock counts
and the evictable count match the live sequences, and that every cache hit
hands a sequence blocks whose KV was computed for exactly its prefix.
"""

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.sequence import Sequence  # noqa: E402


def _seq(tokens, block_size=4):
    Sequence.block_size = block_size
    return Sequence(list(tokens), SamplingOptions(max_tokens=64))


def _admit(manager, seq):
    hits = manager.can_allocate(seq)
    assert hits >= 0
    manager.allocate(seq, hits)
    return hits


def _compute(manager, seq, num_tokens=None):
    """Mark the uncached tokens (or num_tokens of them) as computed and commit."""
    seq.num_scheduled_tokens = (
        seq.num_tokens - seq.num_cached_tokens if num_tokens is None else num_tokens
    )
    manager.commit(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0


def _depth(node):
    return 1 + max((_depth(child) for child in node.children.values()), default=0)


def test_concurrent_identical_prompts_share_blocks_after_commit():
    manager = BlockManager(num_blocks=16, block_size=4)
    a, b = _seq(range(1, 11)), _seq(range(1, 11))  # 2 full blocks + 2 tokens
    assert _admit(manager, a) == 0 and _admit(manager, b) == 0  # nothing cached yet
    assert manager.num_free_blocks == 10

    _compute(manager, a)
    _compute(manager, b)  # b's two full blocks duplicate a's: swapped and freed

    assert b.block_table[:2] == a.block_table[:2]
    assert b.num_shared_blocks == 2 and a.num_shared_blocks == 2
    assert manager.num_free_blocks == 12
    manager.deallocate(a)
    manager.deallocate(b)
    assert manager.num_free_blocks == 16 and manager.num_cached_blocks == 2


def test_partial_match_splits_an_edge_and_keeps_both_branches():
    manager = BlockManager(num_blocks=16, block_size=4)
    first = _seq(range(1, 14))  # 3 full blocks + 1
    _admit(manager, first)
    _compute(manager, first)
    manager.deallocate(first)
    assert _depth(manager.root) == 2  # root -> one 3-block node

    branch = _seq([1, 2, 3, 4, 9, 9, 9, 9, 7])
    assert _admit(manager, branch) == 1
    _compute(manager, branch)
    manager.deallocate(branch)

    assert _depth(manager.root) == 3  # [1..4] -> {[5..12], [9 9 9 9]}
    again = _seq(range(1, 15))
    assert manager.can_allocate(again) == 3


def test_decode_blocks_extend_the_sequence_leaf_in_place():
    manager = BlockManager(num_blocks=16, block_size=4)
    seq = _seq([1, 2, 3])
    _admit(manager, seq)
    _compute(manager, seq)
    for token in range(10, 22):  # decode 12 tokens, one block every 4
        seq.append_token(token)
        manager.may_append(seq)
        _compute(manager, seq, num_tokens=1)

    assert seq.num_shared_blocks == 3
    assert _depth(manager.root) == 2  # one node with 3 blocks, not a chain


def test_eviction_takes_least_recently_used_unlocked_leaf():
    manager = BlockManager(num_blocks=6, block_size=4)
    prompts = {
        "a": [1, 1, 1, 1, 0],
        "b": [2, 2, 2, 2, 0],
        "c": [3, 3, 3, 3, 0],
    }
    for name in ("a", "b", "c"):
        seq = _seq(prompts[name])
        _admit(manager, seq)
        _compute(manager, seq)
        manager.deallocate(seq)
    reuse = _seq(prompts["a"])  # touch a: b is now the oldest
    assert _admit(manager, reuse) == 1
    manager.deallocate(reuse)
    assert manager.num_cached_blocks == 3

    big = _seq(range(100, 117))  # 5 blocks: 3 free + 2 evicted
    _admit(manager, big)

    assert manager.stats.evicted_blocks == 2
    manager.deallocate(big)
    assert manager.can_allocate(_seq(prompts["a"])) == 1  # a survived
    assert manager.can_allocate(_seq(prompts["b"])) == 0
    assert manager.can_allocate(_seq(prompts["c"])) == 0


def test_locked_prefix_is_never_evicted():
    manager = BlockManager(num_blocks=4, block_size=4)
    holder = _seq([5, 5, 5, 5, 5, 5, 5, 5, 0])
    _admit(manager, holder)
    _compute(manager, holder)  # 2 tree blocks, locked by holder
    assert manager.num_free_blocks == 1

    assert manager.can_allocate(_seq(range(100, 106))) == -1  # needs 2
    manager.deallocate(holder)
    assert manager.can_allocate(_seq(range(100, 106))) == 0


def test_preempted_sequence_recomputes_from_its_own_cached_blocks():
    manager = BlockManager(num_blocks=16, block_size=4)
    seq = _seq(range(1, 10))
    _admit(manager, seq)
    _compute(manager, seq)
    for token in range(50, 58):
        seq.append_token(token)
        manager.may_append(seq)
        _compute(manager, seq, num_tokens=1)
    manager.deallocate(seq)  # what preemption does

    assert _admit(manager, seq) == 4  # 17 tokens: every full block but the last
    assert manager.stats.recompute_hit_tokens == 16


def test_reset_prefix_cache_frees_unlocked_blocks():
    manager = BlockManager(num_blocks=8, block_size=4)
    seq = _seq(range(1, 10))
    _admit(manager, seq)
    _compute(manager, seq)
    manager.deallocate(seq)
    assert manager.reset_prefix_cache() == 2
    assert len(manager.free_block_ids) == 8 and not manager.root.children


# --- property test --------------------------------------------------------------


def _check_invariants(manager, live, content):
    bs = manager.block_size
    owners = {}

    def walk(node, prefix_blocks, prefix_tokens):
        for first, child in node.children.items():
            assert child.parent is node and child.key[:bs] == first
            assert len(child.key) == len(child.blocks) * bs and child.blocks
            blocks = prefix_blocks + child.blocks
            tokens = prefix_tokens + child.key
            for i, block in enumerate(child.blocks):
                assert block not in owners, "block owned twice"
                owners[block] = "tree"
                index = len(prefix_blocks) + i
                assert content.get(block) == tokens[: (index + 1) * bs], "tree block holds other KV"
            locks = sum(
                1 for seq in live if _on_path(seq.radix_node, child, manager.root)
            )
            assert child.lock_ref == locks
            walk(child, blocks, tokens)

    walk(manager.root, [], ())
    for seq in live:
        for block in seq.block_table[seq.num_shared_blocks :]:
            assert block not in owners, "private block shared"
            owners[block] = seq.seq_id
        path = _path_blocks(seq.radix_node, manager.root)
        assert seq.block_table[: seq.num_shared_blocks] == path
    for block in manager.free_block_ids:
        assert block not in owners, "free block in use"
        owners[block] = "free"
    assert sorted(owners) == list(range(manager.num_blocks))
    evictable = _evictable(manager.root)
    assert manager.num_cached_blocks == evictable
    assert manager.num_free_blocks == len(manager.free_block_ids) + evictable


def _on_path(node, target, root):
    while node is not None and node is not root:
        if node is target:
            return True
        node = node.parent
    return False


def _path_blocks(node, root):
    blocks = []
    while node is not None and node is not root:
        blocks = node.blocks + blocks
        node = node.parent
    return blocks


def _evictable(node):
    total = 0
    for child in node.children.values():
        if child.lock_ref == 0:
            total += len(child.blocks)
        total += _evictable(child)
    return total


@pytest.mark.parametrize("seed", range(12))
def test_random_workload_keeps_ownership_locks_and_kv_content_consistent(seed):
    rng = random.Random(seed)
    bs = rng.choice([1, 2, 4])
    manager = BlockManager(num_blocks=rng.randint(6, 24), block_size=bs)
    # A few shared stems so prompts overlap at various depths.
    stems = [[rng.randint(0, 3) for _ in range(rng.randint(0, 12))] for _ in range(4)]
    content: dict[int, tuple] = {}  # block -> the token prefix whose KV it holds
    live: list[Sequence] = []

    def write_kv(seq, start, end):
        # The forward pass writes KV for positions [start, end); a block holds
        # a complete prefix once its last position is written.
        for index in range(start // bs, end // bs):
            content[seq.block_table[index]] = tuple(seq.token_ids[: (index + 1) * bs])
        if end % bs:
            content.pop(seq.block_table[end // bs], None)

    for _ in range(300):
        op = rng.random()
        if op < 0.35:
            stem = rng.choice(stems)
            seq = _seq(stem + [rng.randint(0, 3) for _ in range(rng.randint(1, 6))], bs)
            hits = manager.can_allocate(seq)
            if hits < 0:
                continue
            manager.allocate(seq, hits)
            for index in range(hits):
                expected = tuple(seq.token_ids[: (index + 1) * bs])
                assert content[seq.block_table[index]] == expected, "hit with wrong KV"
            for block in seq.block_table[hits:]:
                content.pop(block, None)  # fresh block: whatever it held is gone
            start = seq.num_cached_tokens
            seq.num_scheduled_tokens = seq.num_tokens - start
            write_kv(seq, start, seq.num_tokens)
            manager.commit(seq)
            seq.num_cached_tokens = seq.num_tokens
            seq.num_scheduled_tokens = 0
            live.append(seq)
        elif op < 0.75 and live:
            seq = rng.choice(live)
            seq.append_token(rng.randint(0, 3))
            if not manager.can_append(seq):
                live.remove(seq)
                manager.deallocate(seq)  # preempted
            else:
                before = len(seq.block_table)
                manager.may_append(seq)
                for block in seq.block_table[before:]:
                    content.pop(block, None)
                seq.num_scheduled_tokens = 1
                write_kv(seq, seq.num_tokens - 1, seq.num_tokens)
                manager.commit(seq)
                seq.num_cached_tokens += 1
                seq.num_scheduled_tokens = 0
        elif live:
            seq = live.pop(rng.randrange(len(live)))
            manager.deallocate(seq)
        _check_invariants(manager, live, content)

    for seq in live:
        manager.deallocate(seq)
    _check_invariants(manager, [], content)
    assert manager.num_free_blocks == manager.num_blocks
