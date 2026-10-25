"""Piecewise graphs' data flow on the CPU, where the pieces "replay" eagerly.

The same steps run through ModelRunner with piecewise graphs and fully eager
on separate caches; the logits must agree step by step. The steps get
smaller, so stale buffers from a bigger step (padded tokens' slots, logits
rows) would show up as a corrupted cache in a later step.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.model_runner import ModelRunner  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.sequence import Sequence  # noqa: E402
from test_nano_vllm_model import _load_nano, _save_tiny_hf_llama  # noqa: E402

BLOCK_SIZE = 16
PROMPT_LENS = [17, 30, 5, 64, 33]


def _run(model, pieces: bool) -> tuple[list[torch.Tensor], list[int]]:
    """Four steps (prefill, mixed, chunked prefill, decode); logits per step."""
    Sequence.block_size = BLOCK_SIZE
    cache = model.new_paged_cache(num_blocks=48, block_size=BLOCK_SIZE)
    manager = BlockManager(num_blocks=48, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    runner = ModelRunner(model, cache, manager)
    buckets: list[int] = []
    if pieces:
        runner.capture_piecewise_graphs(max_tokens=96, max_rows=8)
        replay = runner._replay_pieces

        def counted(input_ids, positions, metadata, bucket):
            buckets.append(bucket)
            return replay(input_ids, positions, metadata, bucket)

        runner._replay_pieces = counted
    greedy = SamplingOptions(max_tokens=64, temperature=0.0)
    seqs = [
        Sequence(prompt_token_ids=[(7 * i + j) % 90 + 1 for j in range(length)], sampling=greedy)
        for i, length in enumerate(PROMPT_LENS)
    ]
    for seq in seqs:
        manager.allocate(seq)
    logits_per_step = []

    def step(decodes, prefills):
        for seq, tokens in prefills:
            seq.num_scheduled_tokens = tokens
        batch = decodes + [seq for seq, _ in prefills]
        logits = runner.forward(batch, is_prefill=bool(prefills), use_graphs=pieces).clone()
        logits_per_step.append(logits)
        for seq in decodes:
            seq.num_cached_tokens += 1
        for seq, tokens in prefills:
            seq.num_cached_tokens += tokens
            seq.num_scheduled_tokens = 0
        for seq in decodes + [seq for seq, _ in prefills if seq.num_cached_tokens == len(seq)]:
            seq.is_prefill = False
            seq.append_token(int(logits[batch.index(seq)].argmax()))
            manager.may_append(seq)

    a, b, c, d, e = seqs
    step([], [(a, 17), (b, 30), (c, 5)])  # 52 tokens: size 64
    step([a, b, c], [(d, 40), (e, 33)])  # 76: mixed, d's first chunk, size 96
    step([a, b, c, e], [(d, 24)])  # 28: d's last chunk, size 32 (stale slots past 28)
    step([a, b, c, d, e], [])  # 5 decode rows: size 8
    step([a, b, c, d, e], [])
    return logits_per_step, buckets


def test_piecewise_steps_match_the_eager_forward(tmp_path):
    _, path = _save_tiny_hf_llama(tmp_path)
    model = _load_nano(path)
    eager, _ = _run(model, pieces=False)
    piecewise, buckets = _run(model, pieces=True)
    assert buckets == [64, 96, 32, 8, 8]
    for got, want in zip(piecewise, eager):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_steps_past_the_largest_size_run_eagerly(tmp_path):
    _, path = _save_tiny_hf_llama(tmp_path)
    model = _load_nano(path)
    Sequence.block_size = BLOCK_SIZE
    cache = model.new_paged_cache(num_blocks=48, block_size=BLOCK_SIZE)
    manager = BlockManager(num_blocks=48, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    runner = ModelRunner(model, cache, manager)
    runner.capture_piecewise_graphs(max_tokens=40, max_rows=8)
    assert sorted(runner._pieces.graphs) == [8, 16, 24, 32, 40]
    seq = Sequence(prompt_token_ids=list(range(1, 65)), sampling=SamplingOptions(max_tokens=4))
    manager.allocate(seq)
    seq.num_scheduled_tokens = 64
    runner._replay_pieces = lambda *args: pytest.fail("64 tokens exceed the largest size")
    assert runner.forward([seq], is_prefill=True).shape[0] == 1
