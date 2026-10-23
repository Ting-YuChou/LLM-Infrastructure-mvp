"""Async scheduling on the GPU: in-flight tokens feed the next step on the device.

The same requests run synchronously and with async scheduling (decode CUDA
graphs, where the gather of in-flight tokens is captured in the graph, and
eagerly); every configuration must produce the same greedy tokens.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.core import EngineCore, EngineRequest  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from test_nano_vllm_forward_gpu import _model  # noqa: E402

PROMPTS = [[(11 * i + j) % 500 + 1 for j in range(24)] for i in range(6)]


def _generate(async_scheduling, graphs, scheduling="mixed", sync_debug=False):
    config = NanoVLLMConfig(
        max_model_len=128,
        kvcache_block_size=16,
        num_kvcache_blocks=64,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        attention_backend="triton",
        scheduling=scheduling,
        async_scheduling=async_scheduling,
        enforce_eager=not graphs,
        cuda_graph_max_bs=8,
    )
    core = EngineCore(config, _model("cuda", torch.bfloat16, use_triton=True, attention_backend="triton"))
    core.capture_graphs()
    sampling = SamplingOptions(max_tokens=40, temperature=0.0, ignore_eos=True)
    for req_id, prompt in enumerate(PROMPTS):
        core.add_request(EngineRequest(req_id, prompt, sampling))
    tokens = defaultdict(list)
    steps = 0
    try:
        while core.has_work():
            if sync_debug and steps == 3:  # past the prefill: steady decode steps
                torch.cuda.set_sync_debug_mode("error")
            outputs = core.step()
            steps += 1
            for output in outputs:
                assert output.error is None, output.error
                tokens[output.req_id].append(output.token_id)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return [tokens[i] for i in range(len(PROMPTS))]


@pytest.mark.parametrize("scheduling", ["mixed", "prefill_first"])
def test_async_scheduling_matches_sync(scheduling):
    expected = _generate(False, graphs=True, scheduling=scheduling)
    assert all(len(t) == 40 for t in expected)
    assert _generate(True, graphs=True, scheduling=scheduling) == expected
    assert _generate(True, graphs=False, scheduling=scheduling) == expected


def test_async_steps_make_no_host_sync_besides_collecting():
    # Under "error" a synchronizing op (a D2H copy into pageable memory,
    # .item(), .tolist() of a GPU tensor) raises; waiting on the previous
    # step's event and reading its pinned tokens is not one of them.
    _generate(True, graphs=True, sync_debug=True)
