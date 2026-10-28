"""Async scheduling on the GPU: in-flight tokens feed the next step on the device.

The same requests run synchronously and with async scheduling (decode CUDA
graphs, where the gather of in-flight tokens is captured in the graph, and
eagerly); every configuration must produce the same greedy tokens. A late
request joins while the others decode, so some steps are mixed (piecewise
graphs, or eager).
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
from serving.nano_vllm.kernels import w4a16  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from test_nano_vllm_forward_gpu import _model  # noqa: E402

PROMPTS = [[(11 * i + j) % 500 + 1 for j in range(24)] for i in range(6)]
LATE_PROMPT = [(13 * j) % 500 + 1 for j in range(40)]
LATE_STEP = 5  # inside the sync-debug window, which starts at step 3
GREEDY = SamplingOptions(max_tokens=40, temperature=0.0, ignore_eos=True)
SAMPLED = SamplingOptions(max_tokens=40, temperature=0.7, top_p=0.9, ignore_eos=True)


def _generate(
    async_scheduling,
    graphs,
    scheduling="mixed",
    sync_debug=False,
    weight_quant="none",
    sampling=GREEDY,
    late_request=False,
):
    config = NanoVLLMConfig(
        max_model_len=128,
        kvcache_block_size=16,
        num_kvcache_blocks=64,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        attention_backend="triton",
        weight_quant=weight_quant,
        scheduling=scheduling,
        async_scheduling=async_scheduling,
        enforce_eager=not graphs,
        cuda_graph_max_bs=8,
    )
    model = _model(
        "cuda", torch.bfloat16, use_triton=True, attention_backend="triton", weight_quant=weight_quant
    )
    core = EngineCore(config, model)
    core.capture_graphs()
    for req_id, prompt in enumerate(PROMPTS):
        core.add_request(EngineRequest(req_id, prompt, sampling))
    tokens = defaultdict(list)
    steps = 0
    try:
        while core.has_work():
            if sync_debug and steps == 3:  # past the first prefill
                torch.cuda.set_sync_debug_mode("error")
            if late_request and steps == LATE_STEP:
                core.add_request(EngineRequest(len(PROMPTS), LATE_PROMPT, sampling))
            outputs = core.step()
            steps += 1
            for output in outputs:
                assert output.error is None, output.error
                tokens[output.req_id].append(output.token_id)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return [tokens[i] for i in range(len(PROMPTS) + late_request)]


@pytest.mark.parametrize(
    "scheduling,weight_quant,layout",
    [("mixed", "none", "rows"), ("prefill_first", "none", "rows"), ("mixed", "w4a16", "rows"),
     ("mixed", "w4a16", "tiled")],
)
def test_async_scheduling_matches_sync(scheduling, weight_quant, layout, monkeypatch):
    # Split-K W4A16 (row layout) merges its splits with atomics, whose order
    # varies from run to run; one split keeps those kernels bitwise
    # reproducible. The tiled kernel sums its partials in a fixed order.
    monkeypatch.setitem(w4a16.W4A16_CONFIG, "splits", 1)
    monkeypatch.setitem(w4a16.W4A16_CONFIG, "layout", layout)
    settings = dict(scheduling=scheduling, weight_quant=weight_quant, late_request=True)
    expected = _generate(False, graphs=True, **settings)
    assert all(len(t) == 40 for t in expected)
    assert _generate(True, graphs=True, **settings) == expected
    assert _generate(True, graphs=False, **settings) == expected


def test_sampled_tokens_match_between_sync_and_async():
    # A draw comes from (seed, sequence id, position) through the CUDA sampler,
    # so a request samples the same tokens however its steps are scheduled.
    settings = dict(sampling=SAMPLED, late_request=True)
    expected = _generate(False, graphs=True, **settings)
    assert all(len(t) == 40 for t in expected)
    assert _generate(True, graphs=True, **settings) == expected
    assert _generate(True, graphs=False, **settings) == expected


@pytest.mark.parametrize("sampling", [GREEDY, SAMPLED], ids=["greedy", "sampled"])
@pytest.mark.parametrize("graphs", [True, False], ids=["graphs", "eager"])
@pytest.mark.parametrize("late_request", [False, True], ids=["decode", "mixed"])
def test_async_steps_make_no_host_sync_besides_collecting(sampling, graphs, late_request):
    # Under "error" a synchronizing op (a D2H copy into pageable memory,
    # .item(), .tolist() of a GPU tensor) raises; waiting on the previous
    # step's event and reading its pinned tokens is not one of them.
    tokens = _generate(True, graphs=graphs, sync_debug=True, sampling=sampling, late_request=late_request)
    assert all(len(t) == 40 for t in tokens)
