"""The speculative step's host side, with a scripted target and proposer.

The target's token after each row is a function of that row's input token
and position, so a wrong draft fed to the verify pass gives wrong tokens on
the rows after it; if the engine kept any of them, the output would differ
from the same engine without a proposer.
"""

import asyncio
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.spec import accept_prefix  # noqa: E402
from test_nano_vllm_engine_batching import (  # noqa: E402
    GREEDY,
    ByteTokenizer,
    FakePagedModel,
    ListTokenizer,
    _assert_all_blocks_free,
    _collect,
    _engine,
    _wait_for,
)

VOCAB = 512
EOS = 300


def chain(token: int, position: int) -> int:
    """The target's token after ``token`` at ``position`` (never EOS)."""
    return (token * 7 + position * 13 + 5) % 250


class ChainModel(FakePagedModel):
    """Logits for every row in logits_indices: ``next_token(input, position)``."""

    def __init__(self, next_token=chain):
        super().__init__()
        self.next_token = next_token

    def forward_batch(self, input_ids, positions, cache, metadata):
        del cache
        self.calls.append({"query_lens": list(metadata.query_lens)})
        rows = metadata.logits_indices
        logits = torch.zeros(rows.numel(), VOCAB)
        for out, row in enumerate(rows.tolist()):
            logits[out, self.next_token(int(input_ids[row]), int(positions[row]))] = 10.0
        return logits


class ScriptedProposer:
    """Drafts the target's own continuation, wrong from a scripted depth on.

    ``accept`` gives, cycle after cycle of a sequence, how many drafts are
    right (cycled); the one after them is wrong, the rest follow the wrong
    one as the target would, so nothing after a miss can match by accident
    of the script.
    """

    tap_layers = ()

    def __init__(self, num_draft_tokens: int, accept, next_token=chain):
        self.num_draft_tokens = num_draft_tokens
        self.accept = list(accept)
        self.next_token = next_token
        self.cycles: dict[int, int] = {}
        self.batches: list[int] = []

    def observe(self, taps, positions, slot_mapping):
        raise AssertionError("no taps were asked for")

    def propose(self, batch):
        k = self.num_draft_tokens
        self.batches.append(len(batch.seqs))
        assert batch.positions.numel() == len(batch.seqs) * (k + 1)
        drafts = []
        for index, seq in enumerate(batch.seqs):
            assert int(batch.anchor_ids[index]) == seq.last_token
            assert int(batch.context_lens[index]) == seq.num_tokens - 1
            cycle = self.cycles.get(seq.seq_id, 0)
            self.cycles[seq.seq_id] = cycle + 1
            right = min(self.accept[cycle % len(self.accept)], k)
            token, row = seq.last_token, []
            for depth in range(k):
                token = self.next_token(token, seq.num_tokens - 1 + depth)
                if depth == right:
                    token = (token + 1) % 250
                row.append(token)
            drafts.append(row)
        return torch.tensor(drafts, dtype=torch.long)


def _run(prompts, sampling, proposer=None, model=None, tokenizer=None, **overrides):
    async def scenario():
        engine = _engine(
            tokenizer or ListTokenizer(prompts), model or ChainModel(),
            async_scheduling=False, **overrides,
        )
        if proposer is not None:
            engine.proposer = proposer
            engine._initialize_batching_state()
        try:
            results = await asyncio.gather(
                *(engine.generate_async(name, sampling) for name in prompts)
            )
        finally:
            await engine.shutdown()
        return engine, results

    return asyncio.run(scenario())


PROMPTS = {
    "a": [5, 17, 3, 88, 42, 9, 61, 8, 33, 70, 11],
    "b": [5, 17, 3, 88, 42, 9, 61, 8, 40, 41],  # shares two full blocks with a
    "c": [7, 7, 50],
}


def test_accept_prefix_counts_leading_matches():
    drafts = torch.tensor([[1, 2, 3], [1, 9, 3], [9, 2, 3], [1, 2, 3]])
    target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 0, 3]])
    assert accept_prefix(drafts, target).tolist() == [3, 1, 0, 2]


@pytest.mark.parametrize("scheduling", ["mixed", "prefill_first"])
@pytest.mark.parametrize("draft_tokens", [1, 4, 9])
@pytest.mark.parametrize("accept", [[0], [99], [0, 2, 99, 1, 3]], ids=["none", "all", "mixed"])
def test_speculative_output_is_the_targets_own(scheduling, draft_tokens, accept):
    sampling = SamplingOptions(max_tokens=23, **GREEDY)
    settings = dict(max_num_batched_tokens=24, scheduling=scheduling, num_kvcache_blocks=64)
    _, plain = _run(PROMPTS, sampling, **settings)
    proposer = ScriptedProposer(draft_tokens, accept)
    engine, spec = _run(PROMPTS, sampling, proposer=proposer, **settings)

    assert [r.token_ids for r in spec] == [r.token_ids for r in plain]
    assert all(len(r.token_ids) == 23 and r.finish_reason == "length" for r in spec)
    _assert_all_blocks_free(engine)
    stats = engine.stats()
    assert stats["spec_num_drafts"] == sum(proposer.cycles.values())
    assert stats["spec_num_draft_tokens"] == stats["spec_num_drafts"] * draft_tokens
    # Every token but each request's first came out of a cycle (the last
    # cycle of a request may be cut short by max_tokens).
    emitted = stats["spec_num_accepted_tokens"] + stats["spec_num_drafts"]
    assert 3 * 22 <= emitted <= 3 * (22 + draft_tokens)
    # 22 tokens after the prefill's one, at accepted + 1 per cycle.
    per_cycle = [min(a, draft_tokens) + 1 for a in accept]
    if len(set(per_cycle)) == 1:
        assert set(proposer.cycles.values()) == {-(-22 // per_cycle[0])}


def test_speculative_rows_share_a_step_and_the_token_budget():
    proposer = ScriptedProposer(4, [99])
    sampling = SamplingOptions(max_tokens=11, **GREEDY)
    # 12 tokens a step: two verify rows of 5, not three.
    engine, results = _run(
        PROMPTS, sampling, proposer=proposer, max_num_batched_tokens=12, scheduling="prefill_first"
    )
    assert max(proposer.batches) == 2
    spec_steps = [call["query_lens"] for call in engine.model.calls if 5 in call["query_lens"]]
    assert spec_steps and all(sum(lens) <= 12 for lens in spec_steps)
    assert all(len(r.token_ids) == 11 for r in results)
    _assert_all_blocks_free(engine)


@pytest.mark.parametrize("accept", [[99], [1], [0]])
def test_eos_inside_an_accepted_block_ends_the_request_there(accept):
    prompt = [10, 11, 12]

    def with_eos(token, position):  # the token at position 3 + 6 is EOS
        return EOS if position == len(prompt) + 5 else chain(token, position)

    results = []
    for proposer in (None, ScriptedProposer(9, accept, next_token=with_eos)):
        engine, (result,) = _run(
            {"p": prompt}, SamplingOptions(max_tokens=20, **GREEDY), proposer=proposer,
            model=ChainModel(with_eos), tokenizer=_EosTokenizer({"p": prompt}),
        )
        _assert_all_blocks_free(engine)
        results.append(result)
    plain, spec = results
    assert spec.token_ids == plain.token_ids
    assert spec.token_ids[-1] == EOS and len(spec.token_ids) == 7
    assert spec.finish_reason == plain.finish_reason == "stop"


class _EosTokenizer(ListTokenizer):
    eos_token_id = EOS


@pytest.mark.parametrize("max_tokens", [1, 2, 6, 7, 11])
def test_max_tokens_cuts_a_block_short(max_tokens):
    sampling = SamplingOptions(max_tokens=max_tokens, **GREEDY)
    _, plain = _run(PROMPTS, sampling)
    engine, spec = _run(PROMPTS, sampling, proposer=ScriptedProposer(4, [99]))
    assert [r.token_ids for r in spec] == [r.token_ids for r in plain]
    assert all(len(r.token_ids) == max_tokens and r.finish_reason == "length" for r in spec)
    _assert_all_blocks_free(engine)


def test_stream_delivers_every_token_and_finishes_on_the_last():
    async def scenario():
        engine = _engine(ListTokenizer(PROMPTS), ChainModel(), async_scheduling=False)
        engine.proposer = ScriptedProposer(4, [99])
        engine._initialize_batching_state()
        try:
            seq = engine.prepare_request("a", SamplingOptions(max_tokens=12, **GREEDY))
            return await _collect(engine, seq), seq
        finally:
            await engine.shutdown()

    outputs, seq = asyncio.run(scenario())
    assert [o.finished for o in outputs] == [False] * (len(outputs) - 1) + [True]
    assert seq.num_completion_tokens == 12


def test_stop_string_and_abort_free_every_block():
    text = "hello world, this is a long reply"
    prompt = "ab"

    def scripted(token, position):
        index = position - (len(prompt) - 1)
        return text.encode()[index] if index < len(text) else EOS

    async def scenario(proposer):
        engine = _engine(ByteTokenizer(), ChainModel(scripted), async_scheduling=False)
        if proposer is not None:
            engine.proposer = proposer
            engine._initialize_batching_state()
        try:
            stopped = await engine.generate_async(
                prompt, SamplingOptions(max_tokens=30, stop=["world"], **GREEDY)
            )
            # A consumer that goes away after the first output.
            seq = engine.prepare_request(prompt, SamplingOptions(max_tokens=30, **GREEDY))
            stream = engine.stream_request(seq)
            await stream.__anext__()
            await stream.aclose()
            await _wait_for(lambda: engine.scheduler.is_finished())
        finally:
            await engine.shutdown()
        return engine, stopped

    _, plain = asyncio.run(scenario(None))
    engine, spec = asyncio.run(scenario(ScriptedProposer(9, [99], next_token=scripted)))
    assert spec.text == plain.text == "hello "
    assert spec.finish_reason == "stop"
    _assert_all_blocks_free(engine)


@pytest.mark.parametrize("scheduling", ["mixed", "prefill_first"])
def test_preemption_recomputes_and_keeps_the_output(scheduling):
    # 12 blocks of 4 slots for 4 requests growing to 24 tokens plus the
    # lookahead: verify rows must preempt.
    prompts = {name: [10 * i + 1 + j for j in range(6)] for i, name in enumerate("abcd")}
    sampling = SamplingOptions(max_tokens=18, **GREEDY)
    settings = dict(num_kvcache_blocks=12, max_num_batched_tokens=20, scheduling=scheduling)
    _, plain = _run(prompts, sampling, num_kvcache_blocks=32, max_num_batched_tokens=20)
    proposer = ScriptedProposer(4, [2, 99, 0])
    engine, spec = _run(prompts, sampling, proposer=proposer, **settings)
    assert [r.token_ids for r in spec] == [r.token_ids for r in plain]
    assert engine.stats()["recompute_query_tokens"] > 0
    _assert_all_blocks_free(engine)


def test_a_request_needs_room_for_the_lookahead():
    from serving.nano_vllm.engine import InvalidRequestError

    async def scenario():
        engine = _engine(ListTokenizer(PROMPTS), ChainModel(), num_kvcache_blocks=4)
        engine.proposer = ScriptedProposer(4, [99])
        engine._initialize_batching_state()
        try:
            assert engine.kv_capacity == 12
            with pytest.raises(InvalidRequestError):
                engine.prepare_request("c", SamplingOptions(max_tokens=10, **GREEDY))
            return await engine.generate_async("c", SamplingOptions(max_tokens=9, **GREEDY))
        finally:
            await engine.shutdown()

    assert len(asyncio.run(scenario()).token_ids) == 9


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_prefix_cache_holds_the_accepted_tokens(backend):
    if backend == "cpp":
        pytest.importorskip("nanobind")
    sampling = SamplingOptions(max_tokens=14, **GREEDY)
    settings = dict(enable_prefix_cache=True, prefix_cache_backend=backend, num_kvcache_blocks=48)
    plain_engine, plain = _run(PROMPTS, sampling, **settings)

    async def scenario():
        engine = _engine(ListTokenizer(PROMPTS), ChainModel(), async_scheduling=False, **settings)
        engine.proposer = ScriptedProposer(4, [1, 99, 0, 3])
        engine._initialize_batching_state()
        try:
            first = [await engine.generate_async(name, sampling) for name in PROMPTS]
            tree = engine.block_manager.tree_dump()
            hits = engine.stats()["prefix_hit_tokens"]
            # Each prompt again, extended by its own reply: all cached blocks.
            again = [await engine.generate_async(name, sampling) for name in PROMPTS]
        finally:
            await engine.shutdown()
        return engine, first, again, tree, hits

    engine, first, again, tree, hits = asyncio.run(scenario())
    expected = [r.token_ids for r in plain]
    assert [r.token_ids for r in first] == expected
    assert [r.token_ids for r in again] == expected
    # Every cached block's key is a run of some request's prompt + reply.
    streams = [PROMPTS[name] + reply for name, reply in zip(PROMPTS, expected)]
    for _, _, key, blocks, _ in tree:
        assert len(key) == 4 * len(blocks)
        assert any(
            list(key) == stream[start : start + len(key)]
            for stream in streams for start in range(0, len(stream), 4)
        )
    # No fewer than one token at a time leaves behind (a cycle cut short by
    # max_tokens has computed the last token too).
    def cached(dump):
        return sum(len(blocks) for _, _, _, blocks, _ in dump)

    assert cached(tree) >= cached(plain_engine.block_manager.tree_dump()) > 0
    assert engine.stats()["prefix_hit_tokens"] > hits
    _assert_all_blocks_free(engine)
