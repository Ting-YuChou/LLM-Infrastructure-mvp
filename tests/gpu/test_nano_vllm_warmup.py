"""Serving compiles no Triton kernel once the engine has warmed up.

A kernel variant first needed mid-serving compiles on the spot, stalling every
running request for seconds (Phase 2 LOG: the first M2 W1 sweep). After
capture_graphs, which compiles the eager step's variants and captures the
decode graphs, a mixed workload with every decode row count, eager and graph
decode steps and several block-table widths must find every kernel in
Triton's in-memory cache.
"""

import random
import sys
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


class _TritonCompiles:
    """Kernels Triton had to compile (or load from disk): its JIT cache-miss hook."""

    def __enter__(self):
        import triton

        self.kernels = []

        def hook(*args, **kwargs):
            self.kernels.append(kwargs.get("repr", args))
            return False  # compile as usual

        knobs = getattr(triton, "knobs", None)
        if knobs is not None and hasattr(knobs.runtime, "jit_cache_hook"):  # Triton >= 3.4
            self._owner, self._name = knobs.runtime, "jit_cache_hook"
        else:
            from triton.runtime.jit import JITFunction

            self._owner, self._name = JITFunction, "cache_hook"
        self._old = getattr(self._owner, self._name)
        setattr(self._owner, self._name, hook)
        return self

    def __exit__(self, *exc):
        setattr(self._owner, self._name, self._old)


@pytest.mark.parametrize("linear_backend", ["torch", "triton"])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_no_kernel_compiles_after_warm_up(linear_backend, async_scheduling):
    config = NanoVLLMConfig(
        max_model_len=128,
        kvcache_block_size=16,
        num_kvcache_blocks=256,
        max_num_seqs=16,
        max_num_batched_tokens=48,
        attention_backend="triton",
        scheduling="mixed",
        async_scheduling=async_scheduling,
        cuda_graph_max_bs=8,  # 9-16 decode rows run eagerly
    )
    model = _model("cuda", torch.bfloat16, use_triton=True, attention_backend="triton",
                   linear_backend=linear_backend)
    core = EngineCore(config, model)
    core.capture_graphs()

    rng = random.Random(0)
    pending = [
        EngineRequest(i, [rng.randrange(1, 500) for _ in range(rng.randrange(2, 100))],
                      SamplingOptions(max_tokens=rng.randrange(2, 28), temperature=0.0, ignore_eos=True))
        for i in range(48)
    ]
    decode_rows = set()
    run = core.model_runner.forward

    def forward(seqs, is_prefill, *args, **kwargs):
        decode_rows.add(sum(not seq.is_prefill for seq in seqs) if is_prefill else len(seqs))
        return run(seqs, is_prefill, *args, **kwargs)

    core.model_runner.forward = forward
    with _TritonCompiles() as compiles:
        steps = 0
        while pending or core.has_work():
            if pending and steps % 3 == 0:  # arrivals over time: mixed steps of every size
                for _ in range(rng.randrange(1, 4)):
                    if pending:
                        core.add_request(pending.pop(0))
            for output in core.step():
                assert output.error is None, output.error
            steps += 1
    assert compiles.kernels == [], compiles.kernels
    assert len(decode_rows) > 8, sorted(decode_rows)  # the workload covered many row counts
