"""The speculative cycle as one graph (ModelRunner.capture_spec_graphs).

On the CPU a "graph" is the cycle run eagerly from the static buffers, so
these check everything but the capture itself: the staging of a step's rows,
the padding up to a graph's size, the acceptance computed on the device, and
that a step the graphs do not cover still takes the eager path.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.core import EngineCore, EngineRequest  # noqa: E402
from serving.nano_vllm.sampling import Sampler, SamplingOptions  # noqa: E402
from test_nano_vllm_dflash import TINY_CONFIG  # noqa: E402
from test_nano_vllm_engine_batching import GREEDY  # noqa: E402
from test_nano_vllm_model import _load_nano, _save_tiny_hf_llama  # noqa: E402
from test_nano_vllm_spec_exact import PROMPTS, _drafter_path, _hf_greedy, _hf_taps  # noqa: E402

MAX_TOKENS = 24


@pytest.fixture
def target(tmp_path):
    """(transformers model, its path): four layers, the drafter's sizes."""
    return _save_tiny_hf_llama(
        tmp_path / "target",
        vocab_size=TINY_CONFIG["vocab_size"],
        hidden_size=TINY_CONFIG["hidden_size"],
        intermediate_size=64,
        num_hidden_layers=TINY_CONFIG["num_target_layers"],
    )


class _Oracle:
    """Proposes the tokens of ``expected`` (one sequence's prompt and greedy
    continuation) at the block's positions, wrong at every ``wrong_every``-th
    position: it reads nothing but the batch's tensors."""

    tap_layers = ()
    device_only = True

    def __init__(self, expected: list[int], num_draft_tokens: int, wrong_every: int):
        self.expected = torch.tensor(expected)
        self.num_draft_tokens = num_draft_tokens
        self.wrong_every = wrong_every

    def observe(self, taps, positions, slot_mapping):
        raise AssertionError("no taps were asked for")

    def propose(self, batch):
        block = batch.num_draft_tokens + 1
        positions = batch.positions.view(-1, block)[:, 1:]
        drafts = self.expected[positions.clamp(max=self.expected.numel() - 1)]
        return torch.where(positions % self.wrong_every == 0, drafts + 1, drafts)


def _core(path, proposer=None, graphs=None, **overrides) -> EngineCore:
    config = NanoVLLMConfig(**{
        **dict(
            model_name=str(path), max_model_len=64, max_new_tokens=MAX_TOKENS, kvcache_block_size=4,
            num_kvcache_blocks=48, max_num_seqs=4, max_num_batched_tokens=64,  # every prompt at once
            attention_backend="torch", eager_load=False, enable_prefix_cache=False,
        ),
        **overrides,
    })
    core = EngineCore(config, _load_nano(path), proposer=proposer)
    if graphs:
        core.model_runner.capture_spec_graphs(max_batch=graphs, max_model_len=config.max_model_len)
    return core


def _run(core: EngineCore, prompts: list[list[int]], arrive_every: int = 0, sampling=None):
    """Token ids per request, and how many cycles took the graph / the eager
    path. ``sampling``: one SamplingOptions or one per request (default greedy)."""
    sampling = sampling or SamplingOptions(max_tokens=MAX_TOKENS, **GREEDY)
    if not isinstance(sampling, list):
        sampling = [sampling] * len(prompts)
    runner = core.model_runner
    paths = {"graph": 0, "eager": 0}
    replay, forward = runner._replay_spec, runner._forward_prepared

    def spy_replay(seqs, bucket, slot, block):
        paths["graph"] += 1
        assert bucket >= len(seqs)
        return replay(seqs, bucket, slot, block)

    def spy_forward(input_ids, positions, metadata, *args, **kwargs):
        paths["eager"] += bool(metadata.num_block_rows)
        return forward(input_ids, positions, metadata, *args, **kwargs)

    runner._replay_spec, runner._forward_prepared = spy_replay, spy_forward
    outputs: dict[int, list[int]] = {index: [] for index in range(len(prompts))}
    pending = list(enumerate(prompts))
    steps = 0
    while pending or core.has_work():
        if pending and (not arrive_every or steps % arrive_every == 0):
            for req_id, prompt in pending[: 1 if arrive_every else len(pending)]:
                core.add_request(EngineRequest(req_id, list(prompt), sampling[req_id]))
            pending = pending[1:] if arrive_every else []
        for output in core.step():
            assert output.error is None, output.error
            if output.token_id is not None:
                outputs[output.req_id].append(output.token_id)
        steps += 1
    return [outputs[index] for index in range(len(prompts))], paths


@pytest.mark.parametrize("requests", [1, 3])
def test_graph_cycles_accept_what_the_eager_cycles_accept(target, requests):
    hf_model, path = target
    prompt = PROMPTS["a"]
    expected = _hf_greedy(hf_model, prompt, MAX_TOKENS)
    prompts = [prompt] * requests  # one continuation, so one oracle serves every row

    def run(graphs):
        proposer = _Oracle(prompt + expected, num_draft_tokens=3, wrong_every=5)
        core = _core(path, proposer, graphs)
        outputs, paths = _run(core, prompts)
        return outputs, paths, core.stats()

    eager, eager_paths, eager_stats = run(None)
    graph, graph_paths, graph_stats = run(4)  # sizes 1, 2 and 4: three rows are padded to four
    assert eager == graph == [expected] * requests
    assert eager_paths["graph"] == 0 and graph_paths["eager"] == 0
    assert graph_paths["graph"] == eager_paths["eager"] > 0
    for name in ("spec_num_drafts", "spec_num_accepted_tokens"):
        assert graph_stats[name] == eager_stats[name]
    # The oracle is right most of the time: a cycle yields more than a token.
    assert graph_stats["spec_num_accepted_tokens"] > graph_stats["spec_num_drafts"]


def test_a_proposer_that_reads_the_host_gets_no_graph(target):
    _, path = target
    proposer = _Oracle(PROMPTS["a"], num_draft_tokens=3, wrong_every=2)
    proposer.device_only = False
    core = _core(path, proposer, graphs=4)
    assert core.model_runner._spec is None


def test_dflash_cycles_in_graphs_next_to_eager_steps_with_prefill_rows(tmp_path, target):
    hf_model, path = target
    core = _core(path, graphs=2, speculative_model=_drafter_path(tmp_path, 0), max_num_batched_tokens=8)
    prompts = list(PROMPTS.values())
    # One arrival every third step: verify rows next to prompt chunks (eager),
    # one or two verify rows alone (graphs), three alone (past the graphs: eager).
    outputs, paths = _run(core, prompts, arrive_every=3)
    assert outputs == [_hf_greedy(hf_model, prompt, MAX_TOKENS) for prompt in prompts]
    assert paths["graph"] > 0 and paths["eager"] > 0
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks


def test_graph_cycles_write_the_drafter_context_of_the_computed_tokens(tmp_path, target):
    hf_model, path = target
    core = _core(path, graphs=4, speculative_model=_drafter_path(tmp_path, 0))
    for req_id, prompt in enumerate(PROMPTS.values()):
        core.add_request(EngineRequest(req_id, list(prompt), SamplingOptions(max_tokens=40, **GREEDY)))
    while min((seq.num_completion_tokens for seq in core.requests.values()), default=0) < 9:
        assert all(output.error is None for output in core.step())
    assert len(core.requests) == len(PROMPTS)

    proposer, config = core.proposer, core.config
    drafter, used = proposer.drafter, proposer.cache
    for seq in core.requests.values():
        computed = seq.num_cached_tokens
        slots = torch.tensor([core.block_manager.slot_for_position(seq, p) for p in range(computed)])
        fresh = drafter.new_paged_cache(config.num_kvcache_blocks, config.kvcache_block_size)
        drafter.write_context(
            _hf_taps(hf_model, seq.token_ids[:computed]), torch.arange(computed), fresh, slots
        )
        for layer in range(drafter.config.num_hidden_layers):
            for name in ("keys", "values"):
                got = getattr(used, name)[layer].reshape(used.total_slots, -1)[slots]
                want = getattr(fresh, name)[layer].reshape(fresh.total_slots, -1)[slots]
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_sampler_stage_sets_padded_rows_to_greedy_and_draw_needs_no_flags():
    sampler = Sampler(max_rows=4, device="cpu", seed=0)
    hot = SamplingOptions(temperature=0.8, top_p=0.9, top_k=5)
    assert sampler.stage([hot] * 4, [1, 2, 3, 4], [9, 9, 9, 9], vocab=16) == (True, True, True)
    cold = SamplingOptions(temperature=0.0)
    assert sampler.stage([cold, cold], [1, 2], [3, 4], vocab=16, padded_rows=4) == (False, False, False)
    assert sampler.params[Sampler.TEMPERATURE].view(torch.float32).tolist() == [0.0] * 4

    logits = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    out = torch.zeros(4, dtype=torch.long)
    sampler.draw(logits, out)  # every kernel's path, as a CUDA graph runs it
    assert out.tolist() == logits.argmax(dim=-1).tolist()
