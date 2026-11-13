"""Speculative cycles launched behind a cycle still in flight.

With async scheduling the engine launches a cycle before the one before it
is collected: the host does not know yet how many drafts that one accepts,
so the new cycle's graph takes each row's anchor and position from what the
cycle in flight leaves on the device. On the CPU a "graph" is the cycle run
eagerly from the static buffers, so these check what the host and the graph
agree on (where a chained row starts, how far its KV slots reach, what
happens to a row whose sequence ended meanwhile), not the GPU's ordering.
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
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from test_nano_vllm_dflash import TINY_CONFIG  # noqa: E402
from test_nano_vllm_engine_batching import GREEDY  # noqa: E402
from test_nano_vllm_model import _load_nano, _save_tiny_hf_llama  # noqa: E402
from test_nano_vllm_spec_exact import PROMPTS, _drafter_path  # noqa: E402
from test_nano_vllm_spec_graph import MAX_TOKENS, _core, _Oracle, _run  # noqa: E402
from test_nano_vllm_spec_policy import _graph_core, _scripted  # noqa: E402

SAMPLED = SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.9, top_p=0.95, top_k=40)
DRAFTED = 3
BLOCK_SIZE = 4  # _core's kvcache_block_size


@pytest.fixture
def target(tmp_path):
    return _save_tiny_hf_llama(
        tmp_path / "target",
        vocab_size=TINY_CONFIG["vocab_size"],
        hidden_size=TINY_CONFIG["hidden_size"],
        intermediate_size=64,
        num_hidden_layers=TINY_CONFIG["num_target_layers"],
    )


def _expected(path, prompts, sampling=None, **overrides):
    """The tokens without a drafter."""
    outputs, _ = _run(_core(path, **overrides), prompts, sampling=sampling)
    return outputs


def _oracle(path, prompt, wrong_every=5, sampling=None):
    """Right about ``prompt``'s continuation but at every ``wrong_every``-th
    position (what it proposes for other prompts is noise)."""
    (expected,) = _expected(path, [prompt], sampling)
    return _Oracle(prompt + expected, num_draft_tokens=DRAFTED, wrong_every=wrong_every)


def _drive(core, requests):
    """Token ids and finish reason per request id."""
    outputs = {request.req_id: [] for request in requests}
    reasons = {}
    for request in requests:
        core.add_request(request)
    while core.has_work():
        for output in core.step():
            assert output.error is None, output.error
            if output.token_id is not None:
                outputs[output.req_id].append(output.token_id)
            if output.finish_reason is not None:
                reasons[output.req_id] = output.finish_reason
    return outputs, reasons


def _tight_core(path, proposer=None, graphs=None, **overrides) -> EngineCore:
    """``_core`` with a target loaded for max_model_len itself: its RoPE
    table ends there unless the engine extends it."""
    config = NanoVLLMConfig(**{
        **dict(
            model_name=str(path), max_model_len=64, max_new_tokens=MAX_TOKENS, kvcache_block_size=BLOCK_SIZE,
            num_kvcache_blocks=48, max_num_seqs=4, max_num_batched_tokens=64,
            attention_backend="torch", eager_load=False, enable_prefix_cache=False,
        ),
        **overrides,
    })
    core = EngineCore(config, _load_nano(path, max_model_len=config.max_model_len), proposer=proposer)
    if graphs:
        core.model_runner.capture_spec_graphs(max_batch=graphs, max_model_len=config.max_model_len)
    return core


def _cycle_rows(stats):
    return sum(count for name, count in stats.items() if name.startswith("spec_rows_len_") and name != "spec_rows_len_1")


# --- the tokens ---------------------------------------------------------------


@pytest.mark.parametrize("arrive_every", [0, 3], ids=["at once", "arrivals"])
@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize("drafter", ["dflash", "oracle"])
def test_chained_cycles_yield_the_tokens_of_cycles_collected_one_by_one(
    tmp_path, target, drafter, sampled, arrive_every
):
    _, path = target
    prompts = list(PROMPTS.values())
    sampling = SAMPLED if sampled else None
    expected = _expected(path, prompts, sampling)

    def run(async_scheduling):
        if drafter == "oracle":
            core = _core(
                path, _oracle(path, prompts[0], sampling=sampling), graphs=4,
                async_scheduling=async_scheduling,
            )
        else:
            core = _core(
                path, graphs=4, speculative_model=_drafter_path(tmp_path, 0),
                num_speculative_tokens=DRAFTED, async_scheduling=async_scheduling,
            )
        outputs, paths = _run(core, prompts, arrive_every=arrive_every, sampling=sampling)
        assert outputs == expected
        assert core.block_manager.num_free_blocks == core.block_manager.num_blocks
        return core.stats(), paths

    stats, paths = run(True)
    alone, alone_paths = run(False)
    assert paths["chained"] > 0 and alone_paths["chained"] == 0
    assert alone["spec_num_discarded"] == 0
    # A row is a cycle of its sequence, or it landed after the sequence's end.
    assert stats["spec_num_drafts"] + stats["spec_num_discarded"] == _cycle_rows(stats)
    # Each sequence went through the same cycles either way.
    for name in ("spec_num_drafts", "spec_num_draft_tokens", "spec_num_accepted_tokens"):
        assert stats[name] == alone[name], name
    if arrive_every:
        assert paths["eager"] > 0  # a cycle next to prompt rows waits for the one in flight


def test_plain_steps_and_cycles_of_any_length_between_chained_cycles(tmp_path, target):
    _, path = target
    prompts = list(PROMPTS.values())
    expected = _expected(path, prompts, SAMPLED)
    # Runs of cycles of changing lengths with plain steps in between.
    script = [4, 4, 2, 3, 1, 1, 4, 3, 3, 1, 2, 2, 4]
    core = _scripted(
        _graph_core(
            path, graphs=4, speculative_model=_drafter_path(tmp_path, 1),
            num_speculative_tokens=DRAFTED,
        ),
        script,
    )
    outputs, paths = _run(core, prompts, sampling=SAMPLED)
    assert outputs == expected
    assert paths["chained"] > 3
    stats = core.stats()
    assert all(stats[f"spec_rows_len_{length}"] > 0 for length in (1, 2, 3, 4))
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks


# --- what a chained row runs on -------------------------------------------------


def test_a_chained_row_starts_where_the_cycle_in_flight_ends(target):
    """The anchor a chained cycle drafts from is the token the host appends
    at that position once the cycle before it lands, its context every token
    before, and its block's KV slots the sequence's own."""
    _, path = target
    prompts = list(PROMPTS.values())
    expected = _expected(path, prompts, SAMPLED)
    proposer = _oracle(path, prompts[0], wrong_every=4, sampling=SAMPLED)
    core = _core(path, proposer, graphs=4)
    runner = core.model_runner
    replay, propose = runner._replay_spec, proposer.propose
    launched = []  # per cycle graph: [(sequence, its tokens then, chained?)], and its DraftBatch

    def spy_replay(seqs, bucket, slot, block, chain=None):
        chained = chain or [-1] * len(seqs)
        launched.append([[(seq, seq.num_tokens, row >= 0) for seq, row in zip(seqs, chained)], None])
        return replay(seqs, bucket, slot, block, chain)

    def spy_propose(batch):
        rows = len(launched[-1][0])
        block = batch.num_draft_tokens + 1
        tables = [list(seq.block_table) for seq, _, _ in launched[-1][0]]
        launched[-1][1] = (
            batch.anchor_ids[:rows].tolist(),
            batch.positions.view(-1, block)[:rows].tolist(),
            batch.slot_mapping.view(-1, block)[:rows].tolist(),
            batch.context_lens[:rows].tolist(),
            tables,
        )
        return propose(batch)

    runner._replay_spec, proposer.propose = spy_replay, spy_propose
    outputs, _ = _run(core, prompts, sampling=SAMPLED)
    assert outputs == expected
    full = {index: prompt + output for index, (prompt, output) in enumerate(zip(prompts, outputs))}

    chained_rows = ahead = 0
    for rows, (anchors, positions, slots, context_lens, tables) in launched:
        for (seq, known_tokens, chained), anchor, block, block_slots, context, table in zip(
            rows, anchors, positions, slots, context_lens, tables
        ):
            first = block[0]
            assert block == list(range(first, first + len(block)))
            assert context == first
            # Every block token has a slot in the sequence's own blocks.
            assert block_slots == [
                table[position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE for position in block
            ]
            tokens = full[seq.seq_id]
            if first < len(tokens):
                assert anchor == tokens[first]
            if chained:
                chained_rows += 1
                # The host knew of fewer tokens than the row starts after.
                assert first >= known_tokens
                ahead += first > known_tokens
            else:
                assert first == known_tokens - 1
    assert chained_rows > 5 and ahead > 0  # some cycle in flight accepted a draft


def test_drafts_of_a_chained_cycle_are_drawn_with_the_noise_of_their_positions(target):
    from serving.nano_vllm.sampling import Sampler
    from test_nano_vllm_spec_sampling import _TableDrafter

    _, path = target
    prompts = list(PROMPTS.values())
    proposer = _TableDrafter(TINY_CONFIG["vocab_size"], DRAFTED)
    core = _core(path, proposer, graphs=4, seed=11, speculative_draft_temperature=0.6)
    runner = core.model_runner
    replay, propose = runner._replay_spec, proposer.propose
    cycles = []  # per cycle graph: sequence ids, chained?, the rows' first draft positions, the drafts

    def spy_replay(seqs, bucket, slot, block, chain=None):
        cycles.append([[seq.seq_id for seq in seqs], chain is not None, None, None])
        return replay(seqs, bucket, slot, block, chain)

    def spy_propose(batch):
        drafts = propose(batch)
        rows = len(cycles[-1][0])
        cycles[-1][2] = (batch.positions.view(-1, DRAFTED + 1)[:rows, 0] + 1).tolist()
        cycles[-1][3] = drafts[:rows].clone()
        return drafts

    runner._replay_spec, proposer.propose = spy_replay, spy_propose
    settings = [SAMPLED, SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.0), SAMPLED]
    _run(core, prompts, sampling=settings)

    reference = Sampler(max_rows=DRAFTED, device="cpu", seed=11, temperature_scale=0.6)
    out = torch.zeros(DRAFTED, dtype=torch.long)
    checked = 0
    for seq_ids, chained, firsts, drafts in cycles:
        for seq_id, first, drafted in zip(seq_ids, firsts, drafts):
            positions = list(range(first, first + DRAFTED))
            reference.sample(
                proposer.table[positions], [settings[seq_id]] * DRAFTED, [seq_id] * DRAFTED, positions, out
            )
            assert drafted.tolist() == out.tolist()
            checked += chained
    assert checked > 10


# --- sequences that end while a cycle is in flight --------------------------------


def test_a_row_behind_the_cycle_that_ends_its_sequence_is_discarded(target):
    _, path = target
    prompt = PROMPTS["a"]
    (plain,) = _expected(path, [prompt])
    # A token the target emits for the first time some cycles in.
    stop_at = next(i for i in range(8, len(plain)) if plain[i] not in plain[:i])
    eos = (plain[stop_at],)
    sampling = SamplingOptions(max_tokens=MAX_TOKENS, **GREEDY)

    def run(async_scheduling):
        core = _scripted(
            _core(path, _oracle(path, prompt, wrong_every=7), graphs=4, async_scheduling=async_scheduling),
            [DRAFTED + 1],
        )
        outputs, reasons = _drive(core, [EngineRequest(0, list(prompt), sampling, eos)])
        assert outputs[0] == plain[: stop_at + 1] and reasons == {0: "stop"}
        assert core.block_manager.num_free_blocks == core.block_manager.num_blocks
        return core

    core = run(True)
    stats = core.stats()
    assert stats["spec_num_discarded"] == 1
    assert stats["spec_num_drafts"] + 1 == _cycle_rows(stats)
    # The policy hears of the sequence's cycles only.
    assert sum(len(accepted) for _, accepted in core.spec_policy.updates) == stats["spec_num_drafts"]
    assert run(False).stats()["spec_num_discarded"] == 0


@pytest.mark.parametrize("wrong_every", [5, 1000], ids=["some drafts wrong", "every draft right"])
def test_no_cycle_is_launched_behind_one_that_may_be_the_last(target, wrong_every):
    """One request that ends at max_tokens: when the cycle in flight may
    reach it, the engine waits for it instead of spending a cycle on a
    sequence that is likely over."""
    _, path = target
    prompt = PROMPTS["a"]
    (plain,) = _expected(path, [prompt])
    core = _core(path, _oracle(path, prompt, wrong_every=wrong_every), graphs=4)
    outputs, paths = _run(core, [prompt])
    assert outputs == [plain]
    stats = core.stats()
    assert paths["chained"] > 2
    assert stats["spec_num_discarded"] == 0
    assert stats["spec_num_drafts"] == _cycle_rows(stats)


def test_chained_cycles_under_preemption(tmp_path, target):
    _, path = target
    prompts = list(PROMPTS.values())
    expected = _expected(path, prompts, SAMPLED)
    # 24 + 11, 24 + 10 and 24 + 3 tokens, and each row's lookahead, in 20
    # blocks of 4: verify rows must preempt.
    core = _core(
        path, graphs=4, speculative_model=_drafter_path(tmp_path, 0), num_speculative_tokens=DRAFTED,
        num_kvcache_blocks=20,
    )
    outputs, paths = _run(core, prompts, sampling=SAMPLED)
    assert outputs == expected
    assert paths["chained"] > 0
    assert core.block_manager.stats.recompute_query_tokens > 0
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks


def test_blocks_committed_under_chained_cycles_hold_the_appended_tokens(target):
    """The prefix cache gets a block when the tokens in it are final; a
    later request that extends an earlier output reuses those blocks."""
    _, path = target
    prompt = PROMPTS["a"]
    core = _core(path, _oracle(path, prompt, wrong_every=4), graphs=4, enable_prefix_cache=True)
    (first,), paths = _run(core, [prompt])
    assert paths["chained"] > 0
    longer = prompt + first[:14]
    (expected,) = _expected(path, [longer])
    before = core.block_manager.stats.hit_tokens
    (again,), _ = _run(core, [longer])
    assert again == expected
    # The first request's generated tokens were found in the cache.
    assert core.block_manager.stats.hit_tokens - before > len(prompt)


def test_chained_cycles_up_to_max_model_len(target):
    """A request that fills max_model_len, next to one that goes on: the
    row launched behind its last cycle starts two cycles' drafts past the
    last token the host knew of, in its block table and the RoPE table."""
    _, path = target
    long, short = PROMPTS["a"] + [21, 22], PROMPTS["c"]
    room = 64 - len(long)
    sampling = SamplingOptions(max_tokens=room, **GREEDY)
    expected = _expected(path, [long, short], sampling, max_new_tokens=room)
    assert len(expected[0]) == room
    # Right about every token of the long request: four tokens a cycle, the
    # last of them from 62 known tokens.
    proposer = _Oracle(long + expected[0], num_draft_tokens=DRAFTED, wrong_every=1000)
    core = _tight_core(path, proposer, graphs=4, max_new_tokens=room)
    reached = []
    stage = core.model_runner._stage_spec

    def spy_stage(seqs, bucket, slot, block, verify, chain=None):
        reached.extend(len(seq.block_table) for seq in seqs)
        return stage(seqs, bucket, slot, block, verify, chain)

    core.model_runner._stage_spec = spy_stage
    outputs, paths = _run(core, [long, short], sampling=sampling)
    assert outputs == expected
    assert paths["chained"] > 5
    assert core.stats()["spec_num_discarded"] >= 1
    # 62 tokens, the one in flight and two cycles' drafts.
    assert max(reached) == -(-(62 + 1 + 2 * DRAFTED) // BLOCK_SIZE)
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks


def test_a_batch_no_cycle_graph_covers_is_not_chained(target):
    """Three sequences with graphs for two: their cycles run eagerly, each
    collected before the next; the last sequences' cycles chain."""
    _, path = target
    prompts = list(PROMPTS.values())
    sampling = [
        SamplingOptions(max_tokens=MAX_TOKENS, **GREEDY),
        SamplingOptions(max_tokens=MAX_TOKENS, **GREEDY),
        SamplingOptions(max_tokens=8, **GREEDY),
    ]
    expected = _expected(path, prompts, sampling)
    core = _core(path, _oracle(path, prompts[0]), graphs=2)
    outputs, paths = _run(core, prompts, sampling=sampling)
    assert outputs == expected
    assert paths["eager"] > 0 and paths["chained"] > 0
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks

    # Eight tokens a step: two rows of four (a graph), then three of two.
    core = _scripted(_core(path, _oracle(path, prompts[0]), graphs=2, max_num_batched_tokens=8), [4, 2])
    outputs, paths = _run(core, prompts, sampling=sampling)
    assert outputs == expected
    assert paths["eager"] > 0 and paths["graph"] > 0


# --- the pieces ------------------------------------------------------------------


def test_cycles_chain_unless_told_not_to(target, monkeypatch, tmp_path):
    _, path = target
    prompts = list(PROMPTS.values())
    expected = _expected(path, prompts)
    core = _core(path, _oracle(path, prompts[0]), graphs=4, speculative_chain=False)
    outputs, paths = _run(core, prompts)
    assert outputs == expected
    assert paths["chained"] == 0 and paths["graph"] > 0
    assert core.stats()["spec_num_discarded"] == 0

    assert NanoVLLMConfig().speculative_chain is True
    monkeypatch.setenv("SPECULATIVE_CHAIN", "0")
    assert NanoVLLMConfig.from_yaml(None).speculative_chain is False
    monkeypatch.delenv("SPECULATIVE_CHAIN")
    config = tmp_path / "config.yaml"
    config.write_text("engine:\n  speculative_chain: false\n", encoding="utf-8")
    assert NanoVLLMConfig.from_yaml(str(config)).speculative_chain is False


def test_the_lookahead_is_two_cycles_drafts_only_where_cycles_chain(tmp_path, target):
    _, path = target
    oracle = _Oracle([0], num_draft_tokens=DRAFTED, wrong_every=2)
    assert _core(path, oracle).scheduler.max_lookahead == 2 * DRAFTED
    assert _core(path, oracle, async_scheduling=False).scheduler.max_lookahead == DRAFTED
    assert _core(path, oracle, speculative_chain=False).scheduler.max_lookahead == DRAFTED
    assert _core(path).scheduler.max_lookahead == 0
    core = _tight_core(path, speculative_model=_drafter_path(tmp_path, 0))
    full = core.proposer.num_draft_tokens
    assert core.scheduler.max_lookahead == 2 * full
    # Both RoPE tables reach the furthest position a chained block can.
    reach = core.config.max_model_len + 2 * full
    assert core.model.rope_cos.shape[0] >= reach
    assert core.proposer.drafter.rope_cos.shape[0] >= reach


def test_a_cycle_can_only_follow_rows_of_a_cycle_graph(target):
    _, path = target
    prompt = PROMPTS["c"]
    core = _core(path, _Oracle(prompt, num_draft_tokens=DRAFTED, wrong_every=2), graphs=4, async_scheduling=False)
    core.add_request(EngineRequest(0, list(prompt), SamplingOptions(max_tokens=MAX_TOKENS, **GREEDY)))
    core.step()  # the prompt
    core.step()  # a cycle graph of one row
    (seq,) = core.scheduler.running
    runner = core.model_runner
    with pytest.raises(ValueError, match="cycle graph launched just before"):
        runner.launch([seq], False, [1], DRAFTED)  # it had no second row
    runner.launch([seq], False, None, 0)  # a plain step leaves nothing to follow
    with pytest.raises(ValueError, match="cycle graph launched just before"):
        runner.launch([seq], False, [0], DRAFTED)
