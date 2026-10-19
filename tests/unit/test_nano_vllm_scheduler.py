import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.scheduler import Scheduler  # noqa: E402
from serving.nano_vllm.sequence import Sequence, SequenceStatus  # noqa: E402


def _seq(tokens, max_tokens=2):
    Sequence.block_size = 4
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


def test_scheduler_preempts_when_decode_needs_block_without_capacity():
    manager = BlockManager(num_blocks=1, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(manager, max_num_seqs=1, max_num_batched_tokens=4)
    seq = _seq([1, 2], max_tokens=4)
    Sequence.block_size = 2
    scheduler.add(seq)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [2], is_prefill)

    assert seq.status == SequenceStatus.RUNNING
    assert len(seq.block_table) == 1

    scheduled, is_prefill = scheduler.schedule()
    assert scheduled == []
    assert is_prefill is False
    assert seq.status == SequenceStatus.WAITING
    assert seq in scheduler.waiting
