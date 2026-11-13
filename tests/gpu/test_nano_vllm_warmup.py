"""Serving compiles no Triton kernel once the engine has warmed up.

A kernel variant first needed mid-serving compiles on the spot, stalling every
running request for seconds (Phase 2 LOG: the first M2 W1 sweep). After
capture_graphs, which compiles the eager step's variants and captures the
decode graphs, a mixed workload with every decode row count, eager and graph
decode steps and several block-table widths must find every kernel in
Triton's in-memory cache. With a drafter the decode rows are verify rows of
any of the policy's lengths, or plain decode rows whose tokens go through
the drafter's context writes: the same for every number of them.
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
from serving.nano_vllm.spec import SpecPolicy  # noqa: E402
from test_nano_vllm_block_attention_gpu import drafter_checkpoint  # noqa: E402
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


@pytest.mark.parametrize(
    "linear_backend,async_scheduling,kv_cache_dtype",
    [
        ("torch", False, "auto"),
        ("torch", True, "auto"),
        ("triton", False, "auto"),
        ("triton", True, "auto"),
        ("triton", True, "fp8"),
    ],
)
def test_no_kernel_compiles_after_warm_up(linear_backend, async_scheduling, kv_cache_dtype):
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
                   linear_backend=linear_backend, kv_cache_dtype=kv_cache_dtype)
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


@pytest.mark.parametrize(
    "linear_backend,kv_cache_dtype", [("torch", "auto"), ("triton", "auto"), ("triton", "fp8")]
)
def test_no_kernel_compiles_after_warm_up_with_a_drafter(tmp_path, linear_backend, kv_cache_dtype):
    config = NanoVLLMConfig(
        max_model_len=128,
        kvcache_block_size=16,
        num_kvcache_blocks=256,
        max_num_seqs=16,
        max_num_batched_tokens=96,  # 16 verify rows of 5 tokens fit
        attention_backend="triton",
        scheduling="mixed",
        speculative_model=str(drafter_checkpoint(tmp_path / "drafter")),
    )
    model = _model("cuda", torch.bfloat16, use_triton=True, attention_backend="triton",
                   linear_backend=linear_backend, kv_cache_dtype=kv_cache_dtype)
    core = EngineCore(config, model)
    assert core.proposer is not None and core.proposer.num_draft_tokens == 4
    core.capture_graphs()
    lens = core.spec_policy.verify_lens
    assert lens == (1, 2, 3, 4, 5)
    # Every graph was timed, the decode graphs as length 1.
    assert set(core.spec_policy.times) == {
        (size, length) for size in (1, 2, 4, 8, 16) for length in lens
    }

    class EveryLength(SpecPolicy):
        """Each length in turn, with runs of plain steps (overlapped)."""

        steps = 0

        def choose(self, batch):
            self.steps += 1
            return (1, 1, 2, 5, 1, 3, 4, 4, 1, 1, 1, 5)[self.steps % 12]

    core.spec_policy = EveryLength(lens, core.spec_policy.times)

    rng = random.Random(0)
    pending = [
        EngineRequest(i, [rng.randrange(1, 500) for _ in range(rng.randrange(2, 90))],
                      SamplingOptions(max_tokens=rng.randrange(2, 28), temperature=0.0, ignore_eos=True))
        for i in range(48)
    ]
    verify_rows = set()
    launch_spec = core.model_runner._launch_spec

    def spy(seqs, slot, block=None):
        verify_rows.add((sum(not seq.is_prefill for seq in seqs), block))
        return launch_spec(seqs, slot, block)

    core.model_runner._launch_spec = spy
    with _TritonCompiles() as compiles:
        steps = 0
        while pending or core.has_work():
            if pending and steps % 3 == 0:
                for _ in range(rng.randrange(1, 4)):
                    if pending:
                        core.add_request(pending.pop(0))
            for output in core.step():
                assert output.error is None, output.error
            steps += 1
    assert compiles.kernels == [], compiles.kernels
    assert len({rows for rows, _ in verify_rows}) > 8, sorted(verify_rows)
    assert {block for _, block in verify_rows} == {2, 3, 4, 5}
    assert all(count > 0 for count in core.spec_rows.values()), core.spec_rows
