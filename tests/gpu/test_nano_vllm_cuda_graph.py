"""CUDA graphs in ModelRunner (decode graphs and piecewise graphs): replay must
match the eager forward."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.model_runner import ModelRunner  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.sequence import Sequence  # noqa: E402
from test_nano_vllm_forward_gpu import _model  # noqa: E402

BLOCK_SIZE = 16
PROMPT_LENS = [17, 30, 5, 64, 33, 1, 47, 16]


def _runner(attention_backend, max_graph_batch, weight_quant="none", kv_cache_dtype="auto"):
    # "w4a16_tiled": W4A16 weights in the tiled layout (gemm_w4a16_tiled);
    # "awq_int8head": an AWQ checkpoint with lm_head in int8 (kernels/w8a16.py).
    layout = "tiled" if weight_quant == "w4a16_tiled" else None
    lm_head_quant = "int8" if weight_quant == "awq_int8head" else "none"
    weight_quant = {"w4a16_tiled": "w4a16", "awq_int8head": "awq"}.get(weight_quant, weight_quant)
    model = _model(
        "cuda", torch.bfloat16, use_triton=True, attention_backend=attention_backend,
        weight_quant=weight_quant, kv_cache_dtype=kv_cache_dtype, w4a16_layout=layout,
        lm_head_quant=lm_head_quant,
    )
    Sequence.block_size = BLOCK_SIZE
    cache = model.new_paged_cache(num_blocks=96, block_size=BLOCK_SIZE)
    manager = BlockManager(num_blocks=96, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    runner = ModelRunner(model, cache, manager)
    runner.capture_decode_graphs(max_batch=max_graph_batch, max_model_len=128)
    return runner, manager


@pytest.mark.parametrize(
    "attention_backend,batch,max_graph_batch,weight_quant,kv_cache_dtype",
    [
        ("triton", 1, 8, "none", "auto"),
        ("triton", 3, 8, "none", "auto"),
        ("triton", 8, 8, "none", "auto"),
        ("triton", 3, 2, "none", "auto"),
        ("torch", 3, 8, "none", "auto"),
        ("triton", 3, 8, "w4a16", "auto"),
        ("triton", 3, 8, "w4a16_tiled", "auto"),
        ("triton", 3, 8, "awq", "auto"),
        ("triton", 3, 8, "awq_int8head", "auto"),
        ("triton", 3, 8, "none", "fp8"),
    ],
)
def test_graph_replay_matches_eager(attention_backend, batch, max_graph_batch, weight_quant, kv_cache_dtype):
    runner, manager = _runner(attention_backend, max_graph_batch, weight_quant, kv_cache_dtype)
    greedy = SamplingOptions(max_tokens=64, temperature=0.0)
    seqs = [
        Sequence(prompt_token_ids=[(7 * i + j) % 500 + 1 for j in range(length)], sampling=greedy)
        for i, length in enumerate(PROMPT_LENS[:batch])
    ]
    for seq in seqs:
        manager.allocate(seq)
        seq.num_scheduled_tokens = len(seq)
    logits = runner.forward(seqs, is_prefill=True).float().cpu()
    for seq in seqs:  # what the scheduler does once the prefill launches
        seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = len(seq), 0, False

    # 20 steps cross block boundaries, so block tables grow between replays.
    for _ in range(20):
        for seq, token in zip(seqs, logits.argmax(-1).tolist()):
            seq.append_token(token)
            manager.may_append(seq)
        replayed = runner.forward(seqs, is_prefill=False).float().cpu()
        eager = runner.forward(seqs, is_prefill=False, use_graphs=False).float().cpu()
        torch.testing.assert_close(replayed, eager, rtol=2e-2, atol=2e-2)
        logits = eager
        for seq in seqs:
            seq.num_cached_tokens += 1


@pytest.mark.parametrize(
    "attention_backend,weight_quant,kv_cache_dtype",
    [
        ("triton", "none", "auto"),
        ("torch", "none", "auto"),
        ("triton", "w4a16", "auto"),
        ("triton", "w4a16_tiled", "auto"),
        ("triton", "awq", "auto"),
        ("triton", "awq_int8head", "auto"),
        ("triton", "none", "fp8"),
    ],
)
def test_piecewise_replay_matches_eager(attention_backend, weight_quant, kv_cache_dtype):
    """Prefill, mixed, chunked-prefill and large decode steps in piecewise graphs.

    The steps get smaller, so each one replays over buffers a bigger step
    left stale (padded slots and logits rows must be reset).
    """
    runner, manager = _runner(
        attention_backend, max_graph_batch=2, weight_quant=weight_quant, kv_cache_dtype=kv_cache_dtype
    )
    runner.capture_piecewise_graphs(max_tokens=96, max_rows=8)
    buckets = []
    replay = runner._replay_pieces

    def counted(input_ids, positions, metadata, bucket):
        buckets.append(bucket)
        return replay(input_ids, positions, metadata, bucket)

    runner._replay_pieces = counted
    greedy = SamplingOptions(max_tokens=64, temperature=0.0)
    seqs = [
        Sequence(prompt_token_ids=[(7 * i + j) % 500 + 1 for j in range(length)], sampling=greedy)
        for i, length in enumerate(PROMPT_LENS[:5])  # 17, 30, 5, 64, 33
    ]
    for seq in seqs:
        manager.allocate(seq)

    def step(decodes, prefills):
        """One step: decode rows first, then (seq, tokens) prefill chunks."""
        for seq, tokens in prefills:
            seq.num_scheduled_tokens = tokens
        batch = decodes + [seq for seq, _ in prefills]
        piecewise = runner.forward(batch, is_prefill=bool(prefills)).float().cpu()
        eager = runner.forward(batch, is_prefill=bool(prefills), use_graphs=False).float().cpu()
        torch.testing.assert_close(piecewise, eager, rtol=2e-2, atol=2e-2)
        for seq in decodes:
            seq.num_cached_tokens += 1
        for seq, tokens in prefills:
            seq.num_cached_tokens += tokens
            seq.num_scheduled_tokens = 0
        finished = decodes + [seq for seq, _ in prefills if seq.num_cached_tokens == len(seq)]
        for seq in finished:
            seq.is_prefill = False
            seq.append_token(int(eager[batch.index(seq)].argmax()))
            manager.may_append(seq)

    a, b, c, d, e = seqs
    step([], [(a, 17), (b, 30), (c, 5)])  # 52 tokens: a prefill step, size 64
    step([a, b, c], [(d, 40), (e, 33)])  # 76: mixed, d's first chunk, size 96
    step([a, b, c, e], [(d, 24)])  # 28: d's last chunk after 40 cached tokens, size 32
    step([a, b, c, d, e], [])  # 5 decode rows, past the decode graphs (2): size 8
    assert buckets == [64, 96, 32, 8]
