"""Decode CUDA graphs in ModelRunner: replay must match the eager forward."""

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


def _runner(attention_backend, max_graph_batch, weight_quant="none"):
    model = _model(
        "cuda", torch.bfloat16, use_triton=True, attention_backend=attention_backend,
        weight_quant=weight_quant,
    )
    Sequence.block_size = BLOCK_SIZE
    cache = model.new_paged_cache(num_blocks=96, block_size=BLOCK_SIZE)
    manager = BlockManager(num_blocks=96, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    runner = ModelRunner(model, cache, manager)
    runner.capture_decode_graphs(max_batch=max_graph_batch, max_model_len=128)
    return runner, manager


@pytest.mark.parametrize(
    "attention_backend,batch,max_graph_batch,weight_quant",
    [
        ("triton", 1, 8, "none"),
        ("triton", 3, 8, "none"),
        ("triton", 8, 8, "none"),
        ("triton", 3, 2, "none"),
        ("torch", 3, 8, "none"),
        ("triton", 3, 8, "w4a16"),
    ],
)
def test_graph_replay_matches_eager(attention_backend, batch, max_graph_batch, weight_quant):
    runner, manager = _runner(attention_backend, max_graph_batch, weight_quant)
    greedy = SamplingOptions(max_tokens=64, temperature=0.0)
    seqs = [
        Sequence(prompt_token_ids=[(7 * i + j) % 500 + 1 for j in range(length)], sampling=greedy)
        for i, length in enumerate(PROMPT_LENS[:batch])
    ]
    for seq in seqs:
        manager.allocate(seq)
        seq.num_scheduled_tokens = len(seq)
    logits = runner.forward(seqs, is_prefill=True).float().cpu()

    # 20 steps cross block boundaries, so block tables grow between replays.
    for _ in range(20):
        for seq, token in zip(seqs, logits.argmax(-1).tolist()):
            seq.append_token(token)
            manager.may_append(seq)
        replayed = runner.forward(seqs, is_prefill=False).float().cpu()
        eager = runner.forward(seqs, is_prefill=False, use_graphs=False).float().cpu()
        torch.testing.assert_close(replayed, eager, rtol=2e-2, atol=2e-2)
        logits = eager
