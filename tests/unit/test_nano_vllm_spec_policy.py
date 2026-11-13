"""The verify length of each speculative step (spec.SpecPolicy).

A step runs 1 to K + 1 tokens per decode row: the anchor and as many drafts
as the policy asks for, none being a plain decode step. Whatever it asks
for, step after step, the tokens are the target's own, and every token the
target computes is context for the drafter.
"""

import asyncio
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.core import EngineRequest  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.spec import SpecPolicy  # noqa: E402
from test_nano_vllm_dflash import TINY_CONFIG  # noqa: E402
from test_nano_vllm_engine_batching import (  # noqa: E402
    GREEDY,
    ListTokenizer,
    _assert_all_blocks_free,
    _engine,
)
from test_nano_vllm_model import _save_tiny_hf_llama  # noqa: E402
from test_nano_vllm_spec_engine import PROMPTS as CHAIN_PROMPTS  # noqa: E402
from test_nano_vllm_spec_engine import ChainModel, ScriptedProposer  # noqa: E402
from test_nano_vllm_spec_exact import PROMPTS, _drafter_path, _hf_greedy, _hf_taps  # noqa: E402
from test_nano_vllm_spec_graph import MAX_TOKENS, _core, _Oracle, _run  # noqa: E402
from test_nano_vllm_spec_sampling import _TableDrafter  # noqa: E402

SAMPLED = SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.9, top_p=0.95, top_k=40)
# Plain steps in a row (overlapped when the engine schedules asynchronously),
# cycles of every length after a plain step and after another cycle.
SCRIPT = [1, 1, 1, 3, 5, 1, 2, 4, 4, 1, 1, 5, 2, 1, 3]


@pytest.fixture
def target(tmp_path):
    return _save_tiny_hf_llama(
        tmp_path / "target",
        vocab_size=TINY_CONFIG["vocab_size"],
        hidden_size=TINY_CONFIG["hidden_size"],
        intermediate_size=64,
        num_hidden_layers=TINY_CONFIG["num_target_layers"],
    )


class _Scripted(SpecPolicy):
    """Verify lengths from a list, step after step."""

    def __init__(self, verify_lens, script):
        super().__init__(verify_lens)
        self.script = [length for length in script if length in self.verify_lens]
        self.steps = 0
        self.updates: list[tuple[int, list[int]]] = []

    def choose(self, batch, mixed=False):
        self.steps += 1
        return self.script[(self.steps - 1) % len(self.script)]

    def update(self, drafted, accepted):
        self.updates.append((drafted, list(accepted)))
        super().update(drafted, accepted)


def _scripted(core, script=SCRIPT):
    core.spec_policy = _Scripted(core.spec_policy.verify_lens, script)
    return core


def _graph_core(path, proposer=None, graphs=None, pending_rows=None, **overrides):
    """``_core`` with the decode graphs too (on the CPU: the step run from
    the static buffers), so a plain step leaves its context rows pending."""
    core = _core(path, proposer, graphs, **overrides)
    if graphs:
        core.model_runner.capture_decode_graphs(
            max_batch=graphs, max_model_len=core.config.max_model_len, pending_rows=pending_rows
        )
    return core


def _count_flushes(runner):
    """Rows each flush_context call wrote."""
    flushed = []
    flush = runner.flush_context

    def spy():
        if runner._pending is not None and runner._pending.rows:
            flushed.append(runner._pending.rows)
        return flush()

    runner.flush_context = spy
    return flushed


# Steps of 1 to 64 sequences whose time is a pass over the weights (5 ms)
# plus 0.1 ms per row past the sixteenth, and 1 ms more for a cycle's drafter.
def _times(lens=(1, 2, 3, 4, 6, 10), sizes=(1, 2, 4, 8, 16, 32, 64)):
    return {
        (size, length): 5.0 + 0.1 * max(0, size * length - 16) + (length > 1)
        for size in sizes for length in lens
    }


# --- the policy --------------------------------------------------------------


def test_expected_tokens_are_the_bonus_token_and_the_drafts_that_survive():
    policy = SpecPolicy((1, 2, 3, 4))
    policy.rates = [0.5, 0.5, 1.0]
    assert [policy.expected_tokens(length) for length in (1, 2, 3, 4)] == [1.0, 1.5, 1.75, 2.0]


def test_without_step_times_every_cycle_is_the_longest():
    policy = SpecPolicy((1, 2, 3, 10))
    assert {policy.choose(batch) for batch in (0, 1, 16, 64)} == {10}


def test_few_sequences_run_long_cycles_and_many_run_plain_steps():
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=0)
    chosen = [policy.choose(batch) for batch in (1, 2, 4, 8, 16, 32, 64)]
    assert chosen[0] == 10 and chosen[-1] == 1
    assert chosen == sorted(chosen, reverse=True)  # never longer with more sequences
    assert len(set(chosen)) > 2  # lengths in between, too
    # A batch between two graph sizes costs what the next size does.
    assert policy.choose(5) == policy.choose(8)
    assert policy.cost(100, 1) == pytest.approx(policy.cost(64, 1) * 100 / 64)  # past the graphs: eager


def test_acceptance_moves_the_choice():
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), rate=0.2, probe_every=0)
    assert policy.choose(16) > 1
    for _ in range(60):
        policy.update(policy.choose(16) - 1, [0] * 16)  # nothing accepted
    assert policy.rates[0] < 0.05
    assert policy.choose(16) == 1 and policy.choose(1) == 1
    # Drafts all accepted, seen through the probes: back to long cycles.
    policy.probe_every = 4
    for _ in range(400):
        length = policy.choose(1)
        policy.update(length - 1, [length - 1])
    assert policy.choose(1) == 10 and min(policy.rates) > 0.9


def test_a_drafts_rate_counts_the_cycles_that_reached_it():
    policy = SpecPolicy((1, 3, 5), rate=1.0)  # rate 1: the last step's shares
    policy.update(4, [0, 0, 1, 4])
    # Draft 1: accepted in 2 of 4; draft 2: in 1 of the 2 that reached it;
    # drafts 3 and 4: in the one cycle that got there.
    assert policy.rates == [0.5, 0.5, 1.0, 1.0]
    policy.update(2, [2, 1, 1, 0])
    assert policy.rates[:2] == [0.75, pytest.approx(1 / 3)]
    assert policy.rates[2:] == [1.0, 1.0]  # a cycle of 3 says nothing about drafts 3 and 4
    policy.update(4, [0, 0])
    assert policy.rates == [0.0, pytest.approx(1 / 3), 1.0, 1.0]  # nothing reached draft 2
    before = list(policy.rates)
    policy.update(0, [])
    policy.update(4, [])
    assert policy.rates == before


def test_one_step_in_probe_every_runs_the_next_longer_length():
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=8)
    chosen = [policy.choose(64) for _ in range(32)]
    assert chosen.count(1) == 28 and chosen.count(2) == 4
    assert [index for index, length in enumerate(chosen) if length == 2] == [7, 15, 23, 31]
    assert {policy.choose(1) for _ in range(32)} == {10}  # nothing longer to probe


def test_plain_steps_are_tried_when_they_look_close_to_the_best_length():
    """A kind of step that never runs is never timed. Where the estimate
    puts plain steps near the best cycle, every other probe is a few plain
    steps in a row; where they are far off, none is."""
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=8)
    assert policy.choose(1) == 10
    assert 1 not in {policy.choose(1) for _ in range(64)}  # one sequence: a cycle is 2.6x a plain step
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=8)
    best = policy.choose(16)
    assert 1 < best < 10
    chosen = [best] + [policy.choose(16) for _ in range(40)]
    longer = min(length for length in policy.verify_lens if length > best)
    # Choices 8 and 24: one step at the next longer length. Choice 16: four
    # plain steps (the three after the first are not choices), and again
    # at choice 32.
    assert chosen[7] == longer and chosen[26] == longer
    assert chosen[15:19] == [1, 1, 1, 1] and chosen[34:38] == [1, 1, 1, 1]
    probes = {7, 26, *range(15, 19), *range(34, 38)}
    assert [length for index, length in enumerate(chosen) if index not in probes] == [best] * 31
    # Once plain steps have been timed and found slow, they are left alone.
    for _ in range(50):
        policy.record(16, 1, 3 * policy.times[(16, 1)])
    assert 1 not in {policy.choose(16) for _ in range(64)}


def test_time_measured_while_serving_favors_longer_cycles():
    """What a step takes over its graph's time on an empty context (the
    sequences' KV) is paid once per step, whatever the verify length."""
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=0)
    assert policy.choose(64) == 1
    for _ in range(50):
        policy.record(64, 1, policy.times[(64, 1)] + 40.0)  # long contexts
    assert policy.extra[(64, False)] == pytest.approx(40.0, rel=0.02)
    assert policy.cost(64, 1) == pytest.approx(policy.times[(64, 1)] + 40.0, rel=0.02)
    # Not measured yet for cycles: assumed somewhat more than a plain step's.
    assert policy.cost(64, 2) == pytest.approx(policy.times[(64, 2)] + 40.0 * SpecPolicy.CYCLE_EXTRA, rel=0.02)
    assert policy.choose(64) > 1
    for _ in range(50):
        policy.record(64, 3, policy.times[(64, 3)] + 45.0)
    assert policy.cost(64, 6) == pytest.approx(policy.times[(64, 6)] + 45.0, rel=0.02)
    # Other sizes keep their own measurements; a stalled step is capped.
    assert (16, False) not in policy.extra and policy.cost(16, 1) == policy.times[(16, 1)]
    policy.record(16, 1, 10_000.0)
    assert policy.extra[(16, False)] == pytest.approx(3 * policy.times[(16, 1)])
    policy.record(3, 1, 1.0)  # not a profiled graph: ignored
    assert (3, False) not in policy.extra


def test_steps_with_prompt_rows_are_timed_and_chosen_apart():
    """Alone, a cycle of 2 beats a plain step for 64 sequences; next to
    prompt rows it takes 8 ms more than a plain step does there, more than
    its extra tokens are worth at the rate of the best step alone."""
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=0)
    policy.rates = [0.8] * 9
    times = policy.times
    for _ in range(40):
        policy.record(64, 2, times[(64, 2)] - 5.0)  # overlapped cycles: cheaper than the table
        policy.record(64, 1, times[(64, 1)])
    assert policy.choose(64) == 2
    # Nothing measured next to prompt rows yet: as alone.
    assert policy.choose(64, mixed=True) == 2
    assert policy.mixed_cost(64, 2) == policy.cost(64, 2)
    for prompt_ms in (2.0, 6.0, 4.0):  # the prompt's own share varies; both kinds carry it
        policy.record(64, 2, times[(64, 2)] + prompt_ms + 8.0, mixed=True)
        policy.record(64, 1, times[(64, 1)] + prompt_ms, mixed=True)
    assert policy.mixed[(64, True)] == pytest.approx(12.0) and policy.mixed[(64, False)] == pytest.approx(4.0)
    assert policy.mixed_cost(64, 3) == pytest.approx(times[(64, 3)] + 12.0)
    assert policy.choose(64, mixed=True) == 1
    assert policy.choose(64) == 2  # steps without prompt rows are not touched
    assert (64, True) in policy.extra and policy.extra[(64, True)] == pytest.approx(-5.0)
    # The prompt's share alone does not move the choice: 30 ms more on both.
    even = SpecPolicy((1, 2, 3, 4, 6, 10), _times(), probe_every=0)
    even.rates = [0.8] * 9
    alone = even.choose(4)
    assert alone > 1
    for _ in range(3):
        even.record(4, alone, times[(4, alone)] + 30.0, mixed=True)
        even.record(4, 1, times[(4, 1)] + 30.0, mixed=True)
    assert even.choose(4, mixed=True) == alone
    # A stalled step is capped alone, a long prompt is not.
    even.record(4, 1, 10_000.0, mixed=True)
    assert even.mixed[(4, False)] > 2000.0
    # Other sizes keep their own times.
    assert (16, True) not in policy.mixed and policy.mixed_cost(16, 2) == policy.cost(16, 2)
    policy.set_times(times)
    assert policy.mixed == {}


def test_one_mixed_step_in_sixteen_runs_the_other_kind():
    policy = SpecPolicy((1, 2, 3, 4, 6, 10), _times())
    policy.rates = [0.8] * 9
    assert policy.choose(1, mixed=True) == 10
    chosen = [policy.choose(1, mixed=True) for _ in range(SpecPolicy.MIXED_PROBE_EVERY * 3)]
    assert chosen.count(1) == 3 and set(chosen) == {1, 10}
    # Where plain steps are best, a cycle is tried: the best of them.
    plain = SpecPolicy((1, 2, 3, 4, 6, 10), _times())
    plain.rates = [0.3] * 9
    assert plain.choose(64, mixed=True) == 1
    chosen = [plain.choose(64, mixed=True) for _ in range(SpecPolicy.MIXED_PROBE_EVERY * 2)]
    assert chosen.count(1) == len(chosen) - 2 and set(chosen) == {1, 2}
    # Mixed choices leave the other steps' probes alone.
    assert plain._choices == 0


def test_config_verify_lens(monkeypatch, tmp_path):
    assert NanoVLLMConfig().speculative_verify_lens == (1, 2, 3, 4, 6, 10)
    assert NanoVLLMConfig(speculative_verify_lens=[1, 5]).speculative_verify_lens == (1, 5)
    with pytest.raises(ValueError):
        NanoVLLMConfig(speculative_verify_lens=(0, 2))
    with pytest.raises(ValueError):
        NanoVLLMConfig(speculative_policy="sometimes")
    monkeypatch.setenv("SPECULATIVE_VERIFY_LENS", "1, 3,10")
    monkeypatch.setenv("SPECULATIVE_POLICY", "fixed")
    config = NanoVLLMConfig.from_yaml(None)
    assert config.speculative_verify_lens == (1, 3, 10) and config.speculative_policy == "fixed"
    monkeypatch.delenv("SPECULATIVE_VERIFY_LENS")
    monkeypatch.delenv("SPECULATIVE_POLICY")
    path = tmp_path / "config.yaml"
    path.write_text("engine:\n  speculative_verify_lens: [1, 2, 10]\n", encoding="utf-8")
    assert NanoVLLMConfig.from_yaml(str(path)).speculative_verify_lens == (1, 2, 10)


def test_the_engines_verify_lens_are_the_configured_ones_within_the_block(tmp_path, target):
    _, path = target
    drafter = _drafter_path(tmp_path, 0)  # a block of 5
    core = _core(path, speculative_model=drafter)
    assert core.spec_policy.verify_lens == (1, 2, 3, 4, 5)
    assert core.model_runner.verify_lens == (2, 3, 4, 5)
    core = _core(path, speculative_model=drafter, speculative_verify_lens=(1, 3, 6, 10))
    assert core.spec_policy.verify_lens == (1, 3, 5)  # the block is always one
    core = _core(path, speculative_model=drafter, speculative_policy="fixed")
    assert core.spec_policy.verify_lens == (5,) and core.model_runner.verify_lens == (5,)
    core = _core(path, speculative_model=drafter, num_speculative_tokens=2)
    assert core.spec_policy.verify_lens == (1, 2, 3)
    assert _core(path).spec_policy is None


# --- the engine under a policy -------------------------------------------------


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
def test_tokens_do_not_depend_on_the_verify_lengths(tmp_path, target, sampled, graphs, async_scheduling):
    hf_model, path = target
    prompts = list(PROMPTS.values())
    sampling = SAMPLED if sampled else None
    expected, _ = _run(_core(path), prompts, sampling=sampling)
    if not sampled:
        assert expected == [_hf_greedy(hf_model, prompt, MAX_TOKENS) for prompt in prompts]
    core = _scripted(_graph_core(
        path, graphs=graphs, speculative_model=_drafter_path(tmp_path, 0),
        async_scheduling=async_scheduling, max_num_batched_tokens=8,
    ))
    runner = core.model_runner
    flushed = _count_flushes(runner)
    propose = core.proposer.propose

    def drafts_from_a_complete_context(batch):
        assert runner._pending is None or runner._pending.rows == 0
        return propose(batch)

    core.proposer.propose = drafts_from_a_complete_context
    draft_graphs = []
    if graphs:
        spec = runner._spec
        for key, replay in list(spec.draft_graphs.items()):
            spec.draft_graphs[key] = lambda key=key, replay=replay: (draft_graphs.append(key), replay())[1]
    # An arrival every third step: decode rows of any length next to prompt chunks.
    outputs, paths = _run(core, prompts, arrive_every=3, sampling=sampling)
    assert outputs == expected
    assert bool(flushed) == bool(graphs)  # the plain steps' context rows, in batches
    if graphs:
        # A cycle next to prompt rows drafts through its draft graph.
        assert len(draft_graphs) == paths["eager"] > 0
        assert {block for _, block in draft_graphs} >= {2, 3, 5}
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks
    assert paths["eager"] > 0 and (paths["graph"] > 0) == bool(graphs)

    stats = core.stats()
    rows = {length: stats[f"spec_rows_len_{length}"] for length in (1, 2, 3, 4, 5)}
    assert all(count > 0 for count in rows.values()), rows
    cycles = sum(count for length, count in rows.items() if length > 1)
    assert stats["spec_num_drafts"] == cycles
    assert stats["spec_num_draft_tokens"] == sum((length - 1) * count for length, count in rows.items())
    # Every token after a request's first came from a decode row: one per
    # plain row, a cycle's accepted drafts and its own (the last row of a
    # request may bring more than max_tokens left room for).
    emitted = rows[1] + cycles + stats["spec_num_accepted_tokens"]
    tokens = sum(len(output) - 1 for output in outputs)
    assert tokens <= emitted <= tokens + len(prompts) * 4
    # The policy heard of every cycle, with the drafts its step ran.
    updates = core.spec_policy.updates
    assert sum(len(accepted) for _, accepted in updates) == cycles
    assert {drafted for drafted, _ in updates} == {1, 2, 3, 4}
    assert all(0 <= count <= drafted for drafted, accepted in updates for count in accepted)


@pytest.mark.parametrize("graphs", [0, 64, 5], ids=["cycle graphs only", "decode graphs too", "5 pending rows"])
def test_steps_outside_the_graphs_leave_their_context_pending(tmp_path, target, graphs):
    """Prompt chunks and plain rows next to them run outside the decode
    graphs; with a pending context their rows wait for the next flush too
    (a step with more tokens than it holds writes them itself, after the
    pending ones), and the drafter's context ends up what its reference
    computes."""
    hf_model, path = target
    settings = dict(graphs=4, speculative_model=_drafter_path(tmp_path, 0), max_num_batched_tokens=8)
    core = _graph_core(path, pending_rows=graphs, **settings) if graphs else _core(path, **settings)
    _scripted(core, script=[1, 1, 1, 1, 1, 1, 3, 1, 1, 5])
    runner = core.model_runner
    flushed = _count_flushes(runner)
    observed = []
    observe = core.proposer.observe
    core.proposer.observe = lambda taps, positions, slots: (observed.append(taps.shape[0]), observe(taps, positions, slots))[1]
    pending = list(enumerate(PROMPTS.values()))
    steps = 0
    while pending or min((seq.num_completion_tokens for seq in core.requests.values()), default=0) < 20:
        if pending and steps % 4 == 0:
            req_id, prompt = pending.pop(0)
            core.add_request(EngineRequest(req_id, list(prompt), SamplingOptions(max_tokens=40, **GREEDY)))
        assert all(output.error is None for output in core.step())
        steps += 1
    core._collect_inflight()
    assert len(core.requests) == len(PROMPTS)
    if graphs == 64:
        # Every context write outside a cycle went through a flush: padded
        # to 16 or 64 rows, never a step's own handful.
        assert flushed and max(flushed) <= 64
        assert len([rows for rows in observed if rows in (16, 64)]) == len(flushed)
    elif graphs:
        assert flushed and max(flushed) <= 5
        assert max(observed) > 5  # a prompt chunk with a decode row: written at once
    else:
        assert runner._pending is None and not flushed
    runner.flush_context()
    drafter, used = core.proposer.drafter, core.proposer.cache
    for seq in core.requests.values():
        computed = seq.num_cached_tokens
        slots = torch.tensor([core.block_manager.slot_for_position(seq, p) for p in range(computed)])
        fresh = drafter.new_paged_cache(core.config.num_kvcache_blocks, core.config.kvcache_block_size)
        drafter.write_context(
            _hf_taps(hf_model, seq.token_ids[:computed]), torch.arange(computed), fresh, slots
        )
        for layer in range(drafter.config.num_hidden_layers):
            for name in ("keys", "values"):
                got = getattr(used, name)[layer].reshape(used.total_slots, -1)[slots]
                want = getattr(fresh, name)[layer].reshape(fresh.total_slots, -1)[slots]
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("script", [SCRIPT, [1]], ids=["every length", "plain steps only"])
@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
def test_plain_steps_write_the_drafter_context_too(tmp_path, target, graphs, script):
    hf_model, path = target
    core = _scripted(
        _graph_core(path, graphs=graphs, speculative_model=_drafter_path(tmp_path, 0)), script
    )
    runner = core.model_runner
    flushed = _count_flushes(runner)
    for req_id, prompt in enumerate(PROMPTS.values()):
        core.add_request(EngineRequest(req_id, list(prompt), SamplingOptions(max_tokens=40, **GREEDY)))
    while min((seq.num_completion_tokens for seq in core.requests.values()), default=0) < 30:
        assert all(output.error is None for output in core.step())
    assert len(core.requests) == len(PROMPTS)
    assert core.spec_rows[1] > 0 and (core.spec_rows[5] > 0) == (script is SCRIPT)
    core._collect_inflight()
    if graphs:
        pending = runner._pending
        assert pending.capacity == 64 and max(flushed) <= 64
        if script == [1]:
            # No cycle asked for the context: written when the rows ran out
            # (three sequences, 21 steps), the rest still pending.
            assert flushed == [63] and 0 < pending.rows <= 63
            assert len(pending.host_slots) == pending.rows
    else:
        assert runner._pending is None
    runner.flush_context()

    drafter, used = core.proposer.drafter, core.proposer.cache
    config = core.config
    for seq in core.requests.values():
        computed = seq.num_cached_tokens
        assert computed >= seq.num_tokens - 1
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


def test_a_pending_slot_is_written_before_it_is_computed_again(tmp_path, target):
    """A decode graph's rows wait for the next flush. A second write to one
    of their slots (the block was freed and reused, or the step is run
    again) must come after the first: the pending rows are flushed first."""
    hf_model, path = target
    core = _scripted(
        _graph_core(path, graphs=4, speculative_model=_drafter_path(tmp_path, 0)), script=[1]
    )
    runner = core.model_runner
    core.add_request(EngineRequest(0, list(PROMPTS["a"]), SamplingOptions(max_tokens=40, **GREEDY)))
    core.step()
    while core.requests[0].num_completion_tokens < 3:
        core.step()
    core._collect_inflight()
    seq = core.requests[0]
    pending = runner._pending
    before = pending.rows
    assert before > 0
    flushed = _count_flushes(runner)
    core.block_manager.may_append(seq)
    slot = core.block_manager.slot_for_position(seq, seq.num_cached_tokens)

    def plain_step():
        return runner.collect(runner.launch([seq], is_prefill=False, draft_tokens=0))

    plain_step()  # a new position: joins the pending rows
    assert flushed == [] and pending.rows == before + 1 and slot in pending.host_slots
    plain_step()  # the same position again
    assert flushed == [before + 1] and pending.rows == 1 and pending.host_slots == {slot}
    runner.flush_context()
    assert pending.rows == 0 and not pending.host_slots
    # The slot holds the context of the token computed there.
    drafter, used = core.proposer.drafter, core.proposer.cache
    computed = seq.num_cached_tokens + 1
    fresh = drafter.new_paged_cache(core.config.num_kvcache_blocks, core.config.kvcache_block_size)
    slots = torch.tensor([core.block_manager.slot_for_position(seq, p) for p in range(computed)])
    drafter.write_context(_hf_taps(hf_model, seq.token_ids[:computed]), torch.arange(computed), fresh, slots)
    for layer in range(drafter.config.num_hidden_layers):
        got = used.keys[layer].reshape(used.total_slots, -1)[slots]
        want = fresh.keys[layer].reshape(fresh.total_slots, -1)[slots]
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def _plain_core(tmp_path, path, prompts, pending_rows=None):
    """An engine whose every step is a plain one through the decode graphs,
    stepped until each request has a few tokens and nothing is in flight."""
    core = _scripted(
        _graph_core(
            path, graphs=4, speculative_model=_drafter_path(tmp_path, 0), pending_rows=pending_rows
        ),
        script=[1],
    )
    for req_id, prompt in enumerate(prompts):
        core.add_request(EngineRequest(req_id, list(prompt), SamplingOptions(max_tokens=40, **GREEDY)))
    core.step()
    while min(seq.num_completion_tokens for seq in core.requests.values()) < 3:
        core.step()
    core._collect_inflight()
    return core


def _context_keys(core, slots):
    """The drafter's keys of every layer at ``slots``."""
    cache = core.proposer.cache
    index = torch.tensor(slots)
    return torch.stack([keys.reshape(cache.total_slots, -1)[index] for keys in cache.keys])


def test_a_flush_leaves_no_row_to_be_written_again(tmp_path, target):
    """A flush is padded to a fixed number of rows. The rows it wrote must
    not be written again by the next one: their slots may hold another
    sequence's context by then."""
    _, path = target
    core = _plain_core(tmp_path, path, [PROMPTS["a"], PROMPTS["b"]])
    runner, pending = core.model_runner, core.model_runner._pending
    seqs = list(core.requests.values())
    assert pending.rows >= 4
    # The last pending row: the second sequence's last computed token.
    slot = core.block_manager.slot_for_position(seqs[1], seqs[1].num_cached_tokens - 1)
    assert slot in pending.host_slots
    runner.flush_context()
    # The slot gets another context (as if its block had gone to another sequence).
    width = pending.taps.shape[1]
    other = torch.randn(1, width, generator=torch.Generator().manual_seed(5))
    core.proposer.observe(other, torch.tensor([7]), torch.tensor([slot]))
    kept = _context_keys(core, [slot]).clone()
    # One more plain row and its flush: row 0 only.
    core.block_manager.may_append(seqs[0])
    runner.collect(runner.launch([seqs[0]], is_prefill=False, draft_tokens=0))
    assert pending.rows == 1
    runner.flush_context()
    assert torch.equal(_context_keys(core, [slot]), kept)


@pytest.mark.parametrize("chunk", [1, 6], ids=["a row that waits", "a chunk written at once"])
def test_an_eager_steps_context_is_written_after_the_pending_rows(tmp_path, target, chunk):
    """A step outside the decode graphs computes a slot whose row is still
    pending. Whether its own rows wait too or (more tokens than the pending
    context holds) are written at once, the older row must not land after."""
    hf_model, path = target
    core = _plain_core(tmp_path, path, [PROMPTS["a"]], pending_rows=5)
    runner, pending = core.model_runner, core.model_runner._pending
    assert pending.capacity == 5
    seq = core.requests[0]
    position = seq.num_cached_tokens
    core.block_manager.may_append(seq)
    slot = core.block_manager.slot_for_position(seq, position)
    runner.collect(runner.launch([seq], is_prefill=False, draft_tokens=0))  # pending: the token at position
    assert slot in pending.host_slots
    # The position is computed again with another token, eagerly: alone, or
    # as the end of a chunk of six tokens.
    seq.token_ids[position] = (seq.token_ids[position] + 1) % TINY_CONFIG["vocab_size"]
    if chunk > 1:
        seq.is_prefill, seq.num_cached_tokens, seq.num_scheduled_tokens = True, position - chunk + 1, chunk
    runner.forward([seq], chunk > 1, use_graphs=False)
    assert pending.rows == (1 if chunk == 1 else 0)
    runner.flush_context()
    drafter = core.proposer.drafter
    fresh = drafter.new_paged_cache(core.config.num_kvcache_blocks, core.config.kvcache_block_size)
    tokens = seq.token_ids[: position + 1]
    drafter.write_context(
        _hf_taps(hf_model, tokens)[-1:], torch.tensor([position]), fresh, torch.tensor([slot])
    )
    want = torch.stack([keys.reshape(fresh.total_slots, -1)[slot : slot + 1] for keys in fresh.keys])
    torch.testing.assert_close(_context_keys(core, [slot]), want, rtol=1e-4, atol=1e-4)


def test_a_right_drafter_is_accepted_at_every_length(target):
    hf_model, path = target
    prompt = PROMPTS["a"]
    expected = _hf_greedy(hf_model, prompt, MAX_TOKENS)
    proposer = _Oracle(prompt + expected, num_draft_tokens=4, wrong_every=10_000)
    core = _scripted(_core(path, proposer, graphs=4), script=[3, 2, 5, 4])
    outputs, _ = _run(core, [prompt])
    assert outputs == [expected]
    # Every draft accepted, so a cycle of length Q yields Q tokens (the
    # oracle knows nothing past the last token, where the last cycle reaches).
    updates = core.spec_policy.updates
    assert len(updates) > 4 and {drafted for drafted, _ in updates} == {1, 2, 3, 4}
    assert all(accepted == [drafted] for drafted, accepted in updates[:-1])
    assert core.spec_policy.rates[0] > SpecPolicy.PRIOR_RATE


@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
def test_drafts_of_any_length_are_drawn_with_the_noise_of_their_positions(target, graphs):
    """A cycle of K drafts draws draft j with the noise of position
    num_tokens + j, at the request's settings, whatever K is this step."""
    from serving.nano_vllm.sampling import Sampler

    _, path = target
    prompts = list(PROMPTS.values())
    proposer = _TableDrafter(TINY_CONFIG["vocab_size"], 4)
    # Every cycle collected before the next: the host knows each row's
    # position at its launch (test_nano_vllm_spec_chain.py covers the rest).
    core = _scripted(_core(
        path, proposer, graphs, seed=11, speculative_draft_temperature=0.6, max_num_batched_tokens=8,
        async_scheduling=False,
    ))
    runner = core.model_runner
    cycles = []  # (sequence id, tokens before the cycle) per verify row, and the drafts
    launch, propose = runner._launch_spec, proposer.propose

    def spy_launch(seqs, slot, block=None, chain=None):
        cycles.append([[(seq.seq_id, seq.num_tokens) for seq in seqs if not seq.is_prefill], None])
        return launch(seqs, slot, block, chain)

    def spy_propose(batch):
        drafts = propose(batch)
        cycles[-1][1] = drafts[: len(cycles[-1][0])].clone()  # a graph's padded rows follow
        return drafts

    runner._launch_spec, proposer.propose = spy_launch, spy_propose
    settings = [SAMPLED, SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.0), SAMPLED]
    # An arrival every third step: cycles next to prompt chunks draft too
    # (through the draft graphs, when there are graphs).
    _, paths = _run(core, prompts, arrive_every=3, sampling=settings)
    assert paths["eager"] > 0

    reference = Sampler(max_rows=4, device="cpu", seed=11, temperature_scale=0.6)
    out = torch.zeros(4, dtype=torch.long)
    lengths = set()
    for rows, drafts in cycles:
        for (seq_id, num_tokens), drafted in zip(rows, drafts):
            count = drafted.numel()
            lengths.add(count)
            positions = list(range(num_tokens, num_tokens + count))
            reference.sample(
                proposer.table[positions], [settings[seq_id]] * count, [seq_id] * count, positions, out
            )
            assert drafted.tolist() == out[:count].tolist()
    assert lengths == {1, 2, 3, 4}


def test_the_policys_choice_follows_the_number_of_running_sequences(tmp_path, target):
    """With step times as on a GPU (a cycle pays below some number of
    sequences), the engine drafts while few requests run and stops when
    many do: the same tokens either way."""
    hf_model, path = target
    prompts = [list(PROMPTS["a"]), list(PROMPTS["b"]), list(PROMPTS["c"]), [9, 8, 7, 6]]
    core = _core(path, speculative_model=_drafter_path(tmp_path, 0), max_num_batched_tokens=64)
    lens = core.spec_policy.verify_lens
    # One or two sequences: a cycle costs what a plain step does. More: per row.
    core.spec_policy.set_times({
        (size, length): 5.0 if size <= 2 else 5.0 * size * length
        for size in (1, 2, 4) for length in lens
    })
    core.spec_policy.probe_every = 0
    chosen = []
    choose = core.spec_policy.choose

    def spy(batch, mixed=False):
        chosen.append((batch, choose(batch, mixed)))
        return chosen[-1][1]

    core.spec_policy.choose = spy
    sampling = [SamplingOptions(max_tokens=n, **GREEDY) for n in (MAX_TOKENS, MAX_TOKENS, 6, 6)]
    outputs, _ = _run(core, prompts, sampling=sampling)
    assert outputs == [
        _hf_greedy(hf_model, prompt, option.max_tokens) for prompt, option in zip(prompts, sampling)
    ]
    assert {length for batch, length in chosen if batch > 2} == {1}
    assert {length for batch, length in chosen if 0 < batch <= 2} == {5}
    assert any(batch > 2 for batch, _ in chosen) and any(batch == 2 for batch, _ in chosen)


def test_the_engine_times_steps_with_prompt_rows_apart(tmp_path, target):
    """Steps with prompt rows next to decode rows go to the policy as such,
    chosen as such when prompts wait, and every step is counted by kind."""
    _, path = target
    prompts = list(PROMPTS.values())
    expected, _ = _run(_core(path), prompts)
    core = _graph_core(
        path, graphs=4, speculative_model=_drafter_path(tmp_path, 0), max_num_batched_tokens=8,
    )
    policy = core.spec_policy
    policy.set_times({(size, length): 5.0 for size in (1, 2, 4) for length in policy.verify_lens})
    policy.probe_every = 0
    asked, recorded = [], []
    choose, record = policy.choose, policy.record

    def spy_choose(batch, mixed=False):
        asked.append((mixed, bool(core.scheduler.waiting)))
        return choose(batch, mixed)

    def spy_record(bucket, length, milliseconds, rate=0.1, mixed=False):
        recorded.append((bucket, length, mixed))
        return record(bucket, length, milliseconds, rate, mixed)

    policy.choose, policy.record = spy_choose, spy_record
    outputs, paths = _run(core, prompts, arrive_every=3)
    assert outputs == expected
    assert all(mixed == waiting for mixed, waiting in asked) and {mixed for mixed, _ in asked} == {False, True}
    assert {mixed for _, _, mixed in recorded} == {False, True}
    assert policy.mixed and policy.extra
    stats = core.stats()
    steps = {name[len("spec_steps_"):]: count for name, count in stats.items() if name.startswith("spec_steps_")}
    assert steps["cycle"] == paths["graph"] and steps["cycle_mixed"] == paths["eager"] > 0
    assert sum(mixed for _, _, mixed in recorded) == steps["cycle_mixed"] + steps.get("plain_mixed", 0)
    assert steps["prompt"] > 0
    assert all(stats[f"spec_step_ms_{kind}"] >= 0 for kind in steps)

    # Plain steps next to prompt rows are timed as theirs.
    core = _scripted(_graph_core(
        path, graphs=4, speculative_model=_drafter_path(tmp_path, 0), max_num_batched_tokens=8,
    ), [1, 3])
    core.spec_policy.set_times({(size, length): 5.0 for size in (1, 2, 4) for length in policy.verify_lens})
    outputs, _ = _run(core, prompts, arrive_every=3)
    assert outputs == expected
    stats = core.stats()
    assert stats["spec_steps_plain_mixed"] > 0 and stats["spec_steps_plain"] > 0
    assert {cycle for _, cycle in core.spec_policy.mixed} == {False, True}


@pytest.mark.parametrize("scheduling", ["mixed", "prefill_first"])
@pytest.mark.parametrize("accept", [[0], [99], [0, 2, 99, 1, 3]], ids=["none", "all", "mixed"])
def test_scripted_target_and_proposer_under_changing_lengths(scheduling, accept):
    """The host side alone (test_nano_vllm_spec_engine's models): the verify
    length changes every step, EOS and max_tokens fall inside cycles."""
    sampling = SamplingOptions(max_tokens=23, **GREEDY)

    def run(proposer):
        async def scenario():
            engine = _engine(
                ListTokenizer(CHAIN_PROMPTS), ChainModel(), async_scheduling=True,
                max_num_batched_tokens=24, scheduling=scheduling, num_kvcache_blocks=64,
            )
            if proposer is not None:
                engine.proposer = proposer
                engine._initialize_batching_state()
                core = engine.core
                assert core.spec_policy.verify_lens == (1, 2, 3, 4, 5)
                _scripted(core)
            try:
                results = await asyncio.gather(
                    *(engine.generate_async(name, sampling) for name in CHAIN_PROMPTS)
                )
            finally:
                await engine.shutdown()
            return engine, results

        return asyncio.run(scenario())

    _, plain = run(None)
    engine, spec = run(ScriptedProposer(4, accept))
    assert [r.token_ids for r in spec] == [r.token_ids for r in plain]
    assert all(len(r.token_ids) == 23 and r.finish_reason == "length" for r in spec)
    _assert_all_blocks_free(engine)
    stats = engine.stats()
    assert stats["spec_rows_len_1"] > 0 and stats["spec_num_drafts"] > 0
