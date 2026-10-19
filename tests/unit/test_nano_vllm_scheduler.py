import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.scheduler import Scheduler  # noqa: E402
from serving.nano_vllm.sequence import Sequence, SequenceStatus  # noqa: E402


def _seq(tokens, max_tokens=2, block_size=4):
    Sequence.block_size = block_size
    return Sequence(tokens, SamplingOptions(max_tokens=max_tokens), eos_token_id=99)


def test_block_manager_allocates_releases_and_reuses_prefix_cache():
    manager = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
    seq = _seq([1, 2, 3, 4, 5])

    cached = manager.can_allocate(seq)
    manager.allocate(seq, cached)
    seq.num_scheduled_tokens = 5
    manager.hash_blocks(seq)

    assert cached == 0
    assert len(seq.block_table) == 2
    first_block = seq.block_table[0]
    manager.deallocate(seq)

    clone = _seq([1, 2, 3, 4, 9])
    cached_clone = manager.can_allocate(clone)
    manager.allocate(clone, cached_clone)

    assert cached_clone == 1
    assert clone.block_table[0] == first_block
    assert manager.blocks[first_block].ref_count == 1


def test_scheduler_chunked_prefill_and_decode_batching():
    manager = BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=4, max_num_batched_tokens=3)
    seq = _seq([1, 2, 3, 4, 5])
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [seq]
    assert seq.num_scheduled_tokens == 3
    scheduler.postprocess(scheduled, [7], is_prefill)
    assert seq.status == SequenceStatus.WAITING

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert seq.num_scheduled_tokens == 2
    scheduler.postprocess(scheduled, [8], is_prefill)
    assert seq.status == SequenceStatus.RUNNING
    assert seq.completion_token_ids == [8]

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is False
    assert scheduled == [seq]
    assert seq.num_scheduled_tokens == 1


def test_scheduler_rejects_sequence_that_can_never_fit_in_kv_cache():
    # Previously this sequence was admitted and then preempted forever
    # (prompt + max_tokens = 6 slots, but the cache holds 1 block * 2 slots).
    manager = BlockManager(num_blocks=1, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=1, max_num_batched_tokens=4)
    seq = _seq([1, 2], max_tokens=4, block_size=2)

    with pytest.raises(ValueError, match="KV cache"):
        scheduler.add(seq)
    assert scheduler.is_finished()


def test_scheduler_preempts_youngest_when_decode_needs_block_without_capacity():
    manager = BlockManager(num_blocks=2, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=2, max_num_batched_tokens=4)
    older = _seq([1, 2], max_tokens=2, block_size=2)
    younger = _seq([3, 4], max_tokens=2, block_size=2)
    scheduler.add(older)
    scheduler.add(younger)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True and scheduled == [older, younger]
    scheduler.postprocess(scheduled, [5, 6], is_prefill)
    assert manager.num_free_blocks == 0

    # Both need a new block for decode; only the older one can get it.
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == [older]
    assert younger.status == SequenceStatus.WAITING
    assert younger.block_table == []
    assert list(scheduler.waiting) == [younger]


def _blocks_snapshot(manager):
    return (
        list(manager.free_block_ids),
        set(manager.used_block_ids),
        dict(manager.hash_to_block_id),
        [(block.ref_count, block.content_hash) for block in manager.blocks],
    )


def _fill_prefix_cache(manager):
    """Leave blocks 0/1 hashed for tokens 1..8 but free (ref_count 0)."""
    cached = _seq([1, 2, 3, 4, 5, 6, 7, 8, 9])
    manager.allocate(cached, manager.can_allocate(cached))
    cached.num_scheduled_tokens = cached.num_tokens
    manager.hash_blocks(cached)
    manager.deallocate(cached)
    other = _seq([50])
    manager.allocate(other, manager.can_allocate(other))
    return other


def test_can_allocate_counts_free_listed_prefix_hits():
    manager = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
    _fill_prefix_cache(manager)
    assert manager.num_free_blocks == 3
    # 2 prefix hits that still sit in the free list + 2 fresh blocks = 4 > 3.
    seq = _seq([1, 2, 3, 4, 5, 6, 7, 8, 40, 41, 42, 43, 44])

    assert manager.can_allocate(seq) == -1

    fits = _seq([1, 2, 3, 4, 5, 6, 7, 8, 40])
    assert manager.can_allocate(fits) == 2
    manager.allocate(fits, 2)
    assert manager.num_free_blocks == 0


def test_allocate_failure_leaves_block_manager_unchanged():
    manager = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
    _fill_prefix_cache(manager)
    before = _blocks_snapshot(manager)
    seq = _seq([1, 2, 3, 4, 5, 6, 7, 8, 40, 41, 42, 43, 44])

    with pytest.raises(RuntimeError):
        manager.allocate(seq, 2)

    assert seq.block_table == []
    assert _blocks_snapshot(manager) == before


def test_scheduler_enforces_max_num_seqs_across_steps():
    manager = BlockManager(num_blocks=16, block_size=4, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=2, max_num_batched_tokens=64)
    seqs = [_seq([index + 1], max_tokens=4) for index in range(3)]
    for seq in seqs:
        scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True and scheduled == seqs[:2]
    scheduler.postprocess(scheduled, [7, 7], is_prefill)

    # The third request must wait: two sequences are already running.
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is False
    assert scheduled == seqs[:2]
    assert len(scheduler.running) == 2
    assert seqs[2].block_table == []

    scheduler.abort(seqs[0])
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True and scheduled == [seqs[2]]


def test_decode_scheduling_is_round_robin_under_token_budget():
    manager = BlockManager(num_blocks=16, block_size=4, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=3, max_num_batched_tokens=2)
    seqs = [_seq([index + 1], max_tokens=8) for index in range(3)]
    for seq in seqs:
        scheduler.add(seq)
    while scheduler.waiting:
        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill is True
        scheduler.postprocess(scheduled, [7] * len(scheduled), is_prefill)

    counts = {seq.seq_id: 0 for seq in seqs}
    for _ in range(3):
        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill is False and len(scheduled) == 2
        for seq in scheduled:
            counts[seq.seq_id] += 1
        scheduler.postprocess(scheduled, [7] * len(scheduled), is_prefill)

    assert sorted(counts.values()) == [2, 2, 2]


def test_decode_allocates_a_block_per_token_when_block_size_is_one():
    manager = BlockManager(num_blocks=4, block_size=1, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=1, max_num_batched_tokens=4)
    seq = _seq([1, 2], max_tokens=2, block_size=1)
    scheduler.add(seq)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [3], is_prefill)

    scheduled, is_prefill = scheduler.schedule()

    assert scheduled == [seq] and is_prefill is False
    assert len(seq.block_table) == 3
    assert manager.slot_for_position(seq, 2) == seq.block_table[2]


def test_postprocess_reports_only_appended_tokens():
    manager = BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=4, max_num_batched_tokens=3)
    seq = _seq([1, 2, 3, 4])  # 3-token chunk leaves exactly one prompt token
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert scheduler.postprocess(scheduled, [7], is_prefill) == []
    assert seq.completion_token_ids == []

    scheduled, is_prefill = scheduler.schedule()
    assert scheduler.postprocess(scheduled, [8], is_prefill) == [(seq, 8)]
    assert seq.completion_token_ids == [8]


def test_scheduler_abort_frees_blocks_and_dequeues():
    manager = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=2, max_num_batched_tokens=8)
    seq = _seq([1, 2, 3])
    scheduler.add(seq)
    scheduler.schedule()

    scheduler.abort(seq)

    assert seq.status == SequenceStatus.FINISHED
    assert seq.finish_reason == "abort"
    assert manager.num_free_blocks == 4
    assert scheduler.is_finished()


def test_sequence_finish_reasons_cover_eos_lists_stop_ids_and_length():
    Sequence.block_size = 4
    seq = Sequence(
        [1],
        SamplingOptions(max_tokens=3, stop_token_ids=[42]),
        eos_token_id=[128001, 128009],
    )
    seq.append_token(128009)
    assert seq.should_finish(128009) and seq.finish_reason == "stop"

    seq = Sequence([1], SamplingOptions(max_tokens=3, stop_token_ids=[42]))
    seq.append_token(42)
    assert seq.should_finish(42) and seq.finish_reason == "stop"

    seq = Sequence([1], SamplingOptions(max_tokens=1), eos_token_id=[5])
    seq.append_token(4)
    assert seq.should_finish(4) and seq.finish_reason == "length"


def test_prefix_hash_chain_never_overflows():
    # Each hash is fed back as the next block's 8-byte signed prefix.
    previous = -1
    for block_index in range(256):
        previous = BlockManager.compute_hash([block_index, 1, 2, 3], previous)
        assert -(2**63) <= previous < 2**63
