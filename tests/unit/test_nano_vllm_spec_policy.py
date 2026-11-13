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

    def choose(self, batch):
        self.steps += 1
        return self.script[(self.steps - 1) % len(self.script)]

    def update(self, drafted, accepted):
        self.updates.append((drafted, list(accepted)))
        super().update(drafted, accepted)


def _scripted(core, script=SCRIPT):
    core.spec_policy = _Scripted(core.spec_policy.verify_lens, script)
    return core


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
    core = _scripted(_core(
        path, graphs=graphs, speculative_model=_drafter_path(tmp_path, 0),
        async_scheduling=async_scheduling, max_num_batched_tokens=8,
    ))
    # An arrival every third step: decode rows of any length next to prompt chunks.
    outputs, paths = _run(core, prompts, arrive_every=3, sampling=sampling)
    assert outputs == expected
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


@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
def test_plain_steps_write_the_drafter_context_too(tmp_path, target, graphs):
    hf_model, path = target
    core = _scripted(_core(path, graphs=graphs, speculative_model=_drafter_path(tmp_path, 0)))
    for req_id, prompt in enumerate(PROMPTS.values()):
        core.add_request(EngineRequest(req_id, list(prompt), SamplingOptions(max_tokens=40, **GREEDY)))
    while min((seq.num_completion_tokens for seq in core.requests.values()), default=0) < 12:
        assert all(output.error is None for output in core.step())
    assert len(core.requests) == len(PROMPTS)
    assert core.spec_rows[1] > 0 and core.spec_rows[5] > 0
    core._collect_inflight()

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
    core = _scripted(_core(path, proposer, graphs, seed=11, speculative_draft_temperature=0.6))
    runner = core.model_runner
    cycles = []  # (sequence id, tokens before the cycle) per verify row, and the drafts
    launch, propose = runner._launch_spec, proposer.propose

    def spy_launch(seqs, slot, block=None):
        cycles.append([[(seq.seq_id, seq.num_tokens) for seq in seqs if not seq.is_prefill], None])
        return launch(seqs, slot, block)

    def spy_propose(batch):
        drafts = propose(batch)
        cycles[-1][1] = drafts[: len(cycles[-1][0])].clone()  # a graph's padded rows follow
        return drafts

    runner._launch_spec, proposer.propose = spy_launch, spy_propose
    settings = [SAMPLED, SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.0), SAMPLED]
    _run(core, prompts, sampling=settings)

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

    def spy(batch):
        chosen.append((batch, choose(batch)))
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
