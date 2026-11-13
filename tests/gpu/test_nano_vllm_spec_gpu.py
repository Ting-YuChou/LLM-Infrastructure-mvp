"""The speculative cycle as one CUDA graph (ModelRunner.capture_spec_graphs):
a replay must do what the eager cycle does, without waiting for the GPU, at
every verify length; a plain decode graph keeps the drafter's context; and a
cycle launched behind one still in flight is the cycle the host would have
launched once that one's tokens were back."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.core import EngineCore, EngineRequest  # noqa: E402
from serving.nano_vllm.dflash import DFlashDrafter  # noqa: E402
from serving.nano_vllm.kernels import w4a16  # noqa: E402
from serving.nano_vllm.model_runner import ModelRunner  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.sequence import Sequence  # noqa: E402
from serving.nano_vllm.spec import DFlashProposer, accept_prefix  # noqa: E402
from test_nano_vllm_block_attention_gpu import drafter_checkpoint  # noqa: E402
from test_nano_vllm_forward_gpu import _model  # noqa: E402

BLOCK_SIZE = 16
PROMPT_LENS = [17, 30, 5, 64, 33]
MAX_MODEL_LEN = 128


class _Hinted:
    """The DFlash proposer, but where ``hints[row, position]`` holds a token
    (>= 0) that token is the draft for the position: a drafter with random
    weights is hardly ever right, and what a cycle does with accepted drafts
    needs some. It reads the batch's tensors only, so it runs in the graphs."""

    device_only = True

    def __init__(self, inner, rows, positions):
        self.inner = inner
        self.num_draft_tokens, self.tap_layers = inner.num_draft_tokens, inner.tap_layers
        self.cache, self.drafter = inner.cache, inner.drafter
        device = inner.drafter.device
        self.hints = torch.full((rows, positions), -1, dtype=torch.long, device=device)
        self.rows = torch.arange(rows, device=device).view(rows, 1)

    def observe(self, taps, positions, slot_mapping):
        self.inner.observe(taps, positions, slot_mapping)

    def propose(self, batch):
        drafts = self.inner.propose(batch)
        size, drafted = drafts.shape
        positions = batch.positions.view(size, drafted + 1)[:, 1:]
        hints = self.hints[self.rows[:size], positions]
        return torch.where(hints >= 0, hints, drafts)


def _runner(
    tmp_path, max_graph_batch, weight_quant="none", kv_cache_dtype="auto", drafter_quant="none",
    verify_lens=None, hinted=False,
):
    lm_head_quant = "int8" if weight_quant == "awq_int8head" else "none"
    weight_quant = "awq" if weight_quant == "awq_int8head" else weight_quant
    model = _model(
        "cuda", torch.bfloat16, use_triton=True, attention_backend="triton", linear_backend="triton",
        weight_quant=weight_quant, kv_cache_dtype=kv_cache_dtype, lm_head_quant=lm_head_quant,
    )
    drafter = DFlashDrafter.from_pretrained(
        drafter_checkpoint(tmp_path / "drafter"), model, weight_quant=drafter_quant
    )
    Sequence.block_size = BLOCK_SIZE
    cache = model.new_paged_cache(num_blocks=96, block_size=BLOCK_SIZE)
    manager = BlockManager(num_blocks=96, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    proposer = DFlashProposer(drafter, drafter.new_paged_cache(96, BLOCK_SIZE))
    # As EngineCore: a chained block starts up to a cycle's drafts further on.
    model.extend_rope_table(MAX_MODEL_LEN + 2 * proposer.num_draft_tokens)
    drafter.extend_rope_table(MAX_MODEL_LEN + 2 * proposer.num_draft_tokens)
    if hinted:
        proposer = _Hinted(proposer, max_graph_batch, MAX_MODEL_LEN + 2 * proposer.num_draft_tokens)
    runner = ModelRunner(model, cache, manager, max_rows=8, proposer=proposer, verify_lens=verify_lens)
    runner.capture_spec_graphs(
        max_batch=max_graph_batch, max_model_len=MAX_MODEL_LEN, keep_outputs=True
    )
    return runner, manager


def _prefilled(runner, manager, count):
    greedy = SamplingOptions(max_tokens=64, temperature=0.0)
    seqs = [
        Sequence(prompt_token_ids=[(7 * i + j) % 500 + 1 for j in range(length)], sampling=greedy)
        for i, length in enumerate(PROMPT_LENS[:count])
    ]
    for seq in seqs:
        manager.allocate(seq)
        seq.num_scheduled_tokens = len(seq)
    tokens = runner.run(seqs, is_prefill=True)
    for seq, token in zip(seqs, tokens):  # what the scheduler does with a finished prompt
        seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = len(seq), 0, False
        seq.append_token(token)
    return seqs


def _graph_outputs(runner, batch, block=None):
    """What the last replay for ``batch`` sequences of ``block`` tokens left:
    verify logits [batch, block, vocab], drafts [batch, K] and the sampled
    tokens [batch, block]."""
    block = block or runner.block
    logits, drafts = runner._spec.outputs[(runner._spec.bucket(batch, block), block)]
    return (
        logits[: batch * block].float().cpu().view(batch, block, -1),
        drafts[:batch].cpu(),
        runner.sampled[: batch * block].cpu().view(batch, block),
    )


def _cycle(runner, seqs, block):
    return runner.collect(runner.launch(seqs, is_prefill=False, draft_tokens=block - 1))


def _eager_cycle(runner, seqs, block):
    """The cycle without the graphs: tokens per sequence, verify logits, drafts."""
    seen = {}
    propose, sample = runner.proposer.propose, runner.sampler.sample

    def spy_propose(batch):
        seen["drafts"] = propose(batch)
        return seen["drafts"]

    def spy_sample(logits, *args, **kwargs):
        seen["logits"] = logits
        return sample(logits, *args, **kwargs)

    graphs, runner._spec = runner._spec, None
    runner.proposer.propose, runner.sampler.sample = spy_propose, spy_sample
    try:
        tokens = _cycle(runner, seqs, block)
    finally:
        runner._spec = graphs
        runner.proposer.propose, runner.sampler.sample = propose, sample
    return tokens, seen["logits"].float().cpu(), seen["drafts"].cpu()


@pytest.mark.parametrize(
    "batch,max_graph_batch,weight_quant,kv_cache_dtype,drafter_quant,blocks",
    [
        (1, 4, "none", "auto", "none", (5,)),
        (3, 4, "none", "auto", "none", (5,)),  # three rows padded to four
        (4, 4, "none", "auto", "none", (5,)),
        (5, 5, "none", "auto", "none", (5,)),  # the largest size is not a listed one
        (3, 4, "awq_int8head", "auto", "none", (5,)),
        (3, 4, "w4a16", "auto", "w4a16", (5,)),
        (3, 4, "none", "fp8", "none", (5,)),
        (3, 4, "none", "auto", "w4a16", (5,)),
        # Shorter verify rows, and a length that changes from cycle to cycle.
        (3, 4, "none", "auto", "none", (2,)),
        (4, 4, "none", "auto", "none", (3,)),
        (3, 4, "none", "auto", "none", (4, 2, 5, 3)),
        (3, 4, "awq_int8head", "fp8", "w4a16", (3, 5, 2, 4)),
    ],
)
def test_graph_cycle_matches_the_eager_cycle(
    tmp_path, batch, max_graph_batch, weight_quant, kv_cache_dtype, drafter_quant, blocks
):
    runner, manager = _runner(
        tmp_path, max_graph_batch, weight_quant, kv_cache_dtype, drafter_quant, verify_lens=blocks
    )
    seqs = _prefilled(runner, manager, batch)
    compared = 0
    # Twelve cycles cross block boundaries; from the seventh only the first
    # sequence runs, in a smaller graph, over buffers the others left stale.
    for cycle in range(12):
        block = blocks[cycle % len(blocks)]
        active = seqs if cycle < 6 else seqs[:1]
        for seq in active:
            manager.may_append(seq, lookahead=block - 1)
        # A cycle only writes KV past the cached tokens, so the same state
        # can run it twice.
        replayed = _cycle(runner, active, block)
        graph_logits, graph_drafts, choices = _graph_outputs(runner, len(active), block)
        assert graph_drafts.shape == (len(active), block - 1)
        # The draft graph alone (what a step with prompt rows replays): the
        # cycle graph's drafts, from the same state.
        bucket = runner._spec.bucket(len(active), block)
        with torch.inference_mode():  # as ModelRunner.launch runs it
            runner._stage_spec(active, bucket, 0, block, verify=False)
            runner._spec.draft_graphs[(bucket, block)]()
        assert torch.equal(runner._spec.drafts[: len(active), : block - 1].cpu(), graph_drafts)
        eager, eager_logits, eager_drafts = _eager_cycle(runner, active, block)
        eager_logits = eager_logits.view(len(active), block, -1)

        # What the replay returned is what its own logits and drafts say:
        # every row's token has the row's largest logit (greedy), and a
        # sequence gets its tokens up to the first draft the target refused.
        chosen = graph_logits.gather(-1, choices.unsqueeze(-1)).squeeze(-1)
        assert torch.equal(chosen, graph_logits.max(dim=-1).values)
        accepted = accept_prefix(graph_drafts, choices).tolist()
        assert replayed == [choices[row, : accepted[row] + 1].tolist() for row in range(len(active))]
        # The anchor's logits do not depend on the drafts; the rest of a
        # block is comparable where both paths drafted the same tokens (a
        # near-tie in the drafter's logits can go either way).
        torch.testing.assert_close(graph_logits[:, 0], eager_logits[:, 0], rtol=2e-2, atol=2e-2)
        same = (graph_drafts == eager_drafts).all(dim=1)
        torch.testing.assert_close(graph_logits[same], eager_logits[same], rtol=2e-2, atol=2e-2)
        compared += int(same.sum())

        for seq, tokens in zip(active, eager):
            for token in tokens:
                seq.append_token(token)
            seq.num_cached_tokens = seq.num_tokens - 1
    assert compared >= 0.5 * (6 * batch + 6)


def test_a_decode_graph_writes_the_drafter_context(tmp_path):
    """A plain step between cycles: its tokens' context must reach the
    drafter's cache (through the pending rows and their flush) as the
    eager step leaves it."""
    runner, manager = _runner(tmp_path, max_graph_batch=4, verify_lens=(2, 5))
    runner.capture_decode_graphs(max_batch=4, max_model_len=MAX_MODEL_LEN)
    seqs = _prefilled(runner, manager, 3)
    assert runner._pending.rows == sum(PROMPT_LENS[:3])  # the prompts' rows wait too
    runner.flush_context()
    dcache = runner.proposer.cache
    for step in range(6):
        for seq in seqs:
            manager.may_append(seq)
        slots = torch.tensor(
            [manager.slot_for_position(seq, seq.num_cached_tokens) for seq in seqs], device="cuda"
        )

        def written():
            return [
                getattr(dcache, name)[layer].reshape(dcache.total_slots, -1)[slots].float().clone()
                for layer in range(len(dcache.keys)) for name in ("keys", "values")
            ]

        handle = runner.launch(seqs, is_prefill=False, draft_tokens=0)
        tokens = runner.collect(handle)
        assert handle.graph_size == 4 and handle.spec_seqs == 0
        graph_logits = runner._graphs.outputs[4][:3].float().clone()
        # The graph left the rows pending; the flush writes them.
        assert runner._pending.rows == 3 and runner._pending.host_slots == set(slots.tolist())
        runner.flush_context()
        assert runner._pending.rows == 0
        from_graph = written()
        for layer in range(len(dcache.keys)):
            dcache.keys[layer].reshape(dcache.total_slots, -1)[slots] = 0
            dcache.values[layer].reshape(dcache.total_slots, -1)[slots] = 0
        eager_logits = runner.forward(seqs, False, use_graphs=False).float()
        runner.flush_context()  # the eager step's rows wait like the graph's
        torch.testing.assert_close(graph_logits, eager_logits, rtol=2e-2, atol=2e-2)
        for got, want in zip(from_graph, written()):
            assert want.abs().max() > 0
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        for seq, token in zip(seqs, tokens):
            seq.append_token(token)
            seq.num_cached_tokens = seq.num_tokens - 1
    # And a cycle right after the plain steps runs from that context.
    for seq in seqs:
        manager.may_append(seq, lookahead=4)
    assert all(1 <= len(row) <= 5 for row in _cycle(runner, seqs, 5))


def test_every_graph_gets_a_step_time(tmp_path):
    runner, manager = _runner(tmp_path, max_graph_batch=4, verify_lens=(2, 3, 5))
    runner.capture_decode_graphs(max_batch=4, max_model_len=MAX_MODEL_LEN)
    seqs = _prefilled(runner, manager, 3)
    for seq in seqs:
        manager.may_append(seq, lookahead=4)
    before = _cycle(runner, seqs, 5)
    times = runner.profile_graphs(repeats=2)
    assert set(times) == {(size, length) for size in (1, 2, 4) for length in (1, 2, 3, 5)}
    assert all(0.0 < ms < 1000.0 for ms in times.values())
    # The timing ran every graph on the null block: the sequences' state stands.
    assert _cycle(runner, seqs, 5) == before


def test_a_graph_cycle_launches_without_waiting_for_the_gpu(tmp_path):
    runner, manager = _runner(tmp_path, max_graph_batch=4)
    seqs = _prefilled(runner, manager, 3)
    for seq in seqs:
        manager.may_append(seq, lookahead=runner.proposer.num_draft_tokens)
    runner.run(seqs, is_prefill=False)  # first use of everything
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        handle = runner.launch(seqs, is_prefill=False)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    tokens = runner.collect(handle)
    assert handle.spec_seqs == 3 and len(tokens) == 3
    assert all(1 <= len(row) <= runner.block for row in tokens)


def test_sampled_rows_in_a_graph_cycle_follow_their_settings(tmp_path):
    """A graph runs every sampling kernel whatever the rows ask for: greedy
    rows next to sampled ones still take the argmax, and a sampled row's
    draws repeat (they come from the seed, its sequence id and positions)."""
    runner, manager = _runner(tmp_path, max_graph_batch=4)
    seqs = _prefilled(runner, manager, 3)
    seqs[1].sampling = SamplingOptions(max_tokens=64, temperature=1.5, top_p=0.9, top_k=40)
    for seq in seqs:
        manager.may_append(seq, lookahead=runner.proposer.num_draft_tokens)
    first = runner.run(seqs, is_prefill=False)
    logits, _, choices = _graph_outputs(runner, 3)
    is_max = logits.gather(-1, choices.unsqueeze(-1)).squeeze(-1) == logits.max(dim=-1).values
    assert is_max[0].all() and is_max[2].all()
    assert not is_max[1].all()  # five draws at temperature 1.5
    assert first[1] == choices[1, : len(first[1])].tolist()
    # The same state again: the same draws.
    assert runner.run(seqs, is_prefill=False) == first


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_a_draft_row_and_the_verify_row_of_its_position_draw_alike(tmp_path, graph):
    """Coupled sampling: from the same logits, the draft for a position is
    the token the target's verify row draws there."""
    runner, manager = _runner(tmp_path, max_graph_batch=4)
    seqs = _prefilled(runner, manager, 3)
    seqs[0].sampling = SamplingOptions(max_tokens=64, temperature=1.0, top_p=1.0)
    seqs[2].sampling = SamplingOptions(max_tokens=64, temperature=0.8, top_p=0.9, top_k=50)
    block, drafted = runner.block, runner.proposer.num_draft_tokens
    vocab = runner._spec.vocab
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = (3 * torch.randn(3, block, vocab, device="cuda", generator=generator)).to(torch.bfloat16)

    verify = torch.zeros(3 * block, dtype=torch.long, device="cuda")
    runner.sampler.sample(logits.view(3 * block, vocab), *runner._spec_sampling(seqs, 3, block), verify)
    draft_logits = logits[:, :drafted].reshape(3 * drafted, vocab).contiguous()
    if graph:  # as _replay_spec stages it, with a padded fourth sequence
        runner.draft_sampler.stage(*runner._draft_sampling(seqs, drafted), vocab, padded_rows=4 * drafted)
        drafts = runner._graph_draft_draw(draft_logits)
    else:
        drafts = runner._draft_draw(seqs, 0, drafted)(draft_logits)

    verify, drafts = verify.view(3, block).cpu(), drafts.view(3, drafted).cpu()
    assert torch.equal(drafts, verify[:, :drafted])
    # The sampled rows drew: not every token is its row's largest logit.
    values = logits.float().cpu()
    is_max = values.gather(-1, verify.unsqueeze(-1)).squeeze(-1) == values.max(dim=-1).values
    assert is_max[1].all() and not is_max[0].all()


# --- a cycle launched behind a cycle in flight ---------------------------------


def _append(seqs, cycles):
    """What the scheduler does with a collected cycle's tokens."""
    for seq, tokens in zip(seqs, cycles):
        for token in tokens:
            seq.append_token(token)
        seq.num_cached_tokens = seq.num_tokens - 1


@pytest.mark.parametrize(
    "weight_quant,kv_cache_dtype,drafter_quant,blocks",
    [
        ("none", "auto", "none", (5,)),
        ("none", "auto", "none", (3, 5, 2, 4)),
        ("none", "fp8", "none", (4, 2, 5)),
        ("awq_int8head", "auto", "w4a16", (5, 3)),
    ],
)
def test_a_cycle_behind_one_in_flight_is_the_cycle_launched_after_it_landed(
    tmp_path, monkeypatch, weight_quant, kv_cache_dtype, drafter_quant, blocks
):
    """Cycle B is launched while cycle A runs: its anchors, positions, KV
    slots and sampling positions come from what A leaves on the device. Run
    again once the host knows A's tokens (a cycle only writes KV past the
    cached tokens, so the same state can run it twice), it must draft and
    verify the same tokens."""
    # One split: the W4A16 kernels then sum in a fixed order (as in test_nano_vllm_async).
    monkeypatch.setitem(w4a16.W4A16_CONFIG, "splits", 1)
    monkeypatch.setitem(w4a16.W4A16_CONFIG, "layout", "tiled")
    runner, manager = _runner(
        tmp_path, 4, weight_quant, kv_cache_dtype, drafter_quant, verify_lens=blocks, hinted=True
    )
    seqs = _prefilled(runner, manager, 3)
    # A sampled row: its draws depend on the positions the graph computes.
    seqs[1].sampling = SamplingOptions(max_tokens=64, temperature=1.2, top_p=0.95, top_k=50)
    rows = list(range(len(seqs)))

    # The target's own next tokens, one cycle at a time, then back to where
    # the sequences were: cycles only write KV past the cached tokens. They
    # become the drafts, but for every fourth position.
    lengths = [seq.num_tokens for seq in seqs]
    for cycle in range(40):
        for seq in seqs:
            manager.may_append(seq, lookahead=blocks[0] - 1)
        _append(seqs, _cycle(runner, seqs, blocks[0]))
    hints = runner.proposer.hints
    for row, seq in enumerate(seqs):
        known = torch.tensor(seq.token_ids, device="cuda")
        wrong = torch.arange(seq.num_tokens, device="cuda") % 4 == 0
        hints[row, : seq.num_tokens] = torch.where(wrong, (known + 1) % 500, known)
        del seq.token_ids[lengths[row] :]
        seq.num_cached_tokens = seq.num_tokens - 1

    moved = further = 0
    for cycle in range(8):
        first, second = blocks[cycle % len(blocks)], blocks[(cycle + 1) % len(blocks)]
        for seq in seqs:
            manager.may_append(seq, lookahead=first - 1)
        inflight = runner.launch(seqs, is_prefill=False, draft_tokens=first - 1)
        # The host only knows the cycle in flight moves each sequence 1 to
        # ``first`` tokens on.
        for seq in seqs:
            manager.may_append(seq, lookahead=first + second - 1)
        torch.cuda.set_sync_debug_mode("error")
        try:
            chained = runner.launch(seqs, is_prefill=False, gather_rows=rows, draft_tokens=second - 1)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        landed = runner.collect(inflight)
        behind = runner.collect(chained)
        assert chained.chains and chained.spec_seqs == 3
        logits, drafts, choices = _graph_outputs(runner, 3, second)
        state = runner._spec.state[:, :3].clone()
        moved += sum(len(tokens) > 1 for tokens in landed)
        further += sum(len(tokens) > 1 for tokens in behind)

        _append(seqs, landed)
        again = _cycle(runner, seqs, second)
        again_logits, again_drafts, again_choices = _graph_outputs(runner, 3, second)
        assert behind == again
        assert torch.equal(drafts, again_drafts) and torch.equal(choices, again_choices)
        torch.testing.assert_close(logits, again_logits, rtol=2e-2, atol=2e-2)
        # Where each sequence goes next: its last token and that token's position.
        assert torch.equal(state, runner._spec.state[:, :3])
        _append(seqs, again)
        assert state[0].tolist() == [seq.last_token for seq in seqs]
        assert state[1].tolist() == [seq.num_tokens - 1 for seq in seqs]
    # Cycles in flight accepted drafts (the host was behind), and so did
    # the ones launched behind them.
    assert moved >= 6 and further >= 6, (moved, further)


def _serve(tmp_path, async_scheduling, prompts, sampling, late=None, weight_quant="none"):
    """Token ids per request from an engine core with the drafter (every
    cycle the whole block), how many cycle graphs it launched behind a cycle
    in flight, and its stats. ``sampling``: one SamplingOptions or one per
    request (the late one included)."""
    config = NanoVLLMConfig(
        max_model_len=MAX_MODEL_LEN,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=96,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        attention_backend="triton",
        weight_quant=weight_quant,
        scheduling="mixed",
        async_scheduling=async_scheduling,
        cuda_graph_max_bs=8,
        speculative_model=str(drafter_checkpoint(tmp_path / "drafter")),
        speculative_policy="fixed",
    )
    model = _model(
        "cuda", torch.bfloat16, use_triton=True, attention_backend="triton", weight_quant=weight_quant
    )
    core = EngineCore(config, model)
    core.capture_graphs()
    replay = core.model_runner._replay_spec
    chained = []

    def spy(seqs, bucket, slot, block, chain=None):
        chained.append(chain is not None)
        return replay(seqs, bucket, slot, block, chain)

    core.model_runner._replay_spec = spy
    count = len(prompts) + (late is not None)
    if not isinstance(sampling, list):
        sampling = [sampling] * count
    for req_id, prompt in enumerate(prompts):
        core.add_request(EngineRequest(req_id, prompt, sampling[req_id]))
    tokens = {index: [] for index in range(count)}
    steps = 0
    try:
        while core.has_work():
            if steps == 3:  # past the first prefill: no host sync but collecting
                torch.cuda.set_sync_debug_mode("error")
            if late is not None and steps == 6:
                core.add_request(EngineRequest(len(prompts), late, sampling[-1]))
            for output in core.step():
                assert output.error is None, output.error
                tokens[output.req_id].append(output.token_id)
            steps += 1
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert core.block_manager.num_free_blocks == core.block_manager.num_blocks
    return [tokens[index] for index in sorted(tokens)], sum(chained), core.stats()


@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
def test_one_requests_tokens_with_chained_cycles(tmp_path, sampled):
    """One request at a time: the same cycles on the same rows whether each
    is collected before the next or not, so the same tokens bit for bit."""
    sampling = SamplingOptions(
        max_tokens=48, temperature=0.8 if sampled else 0.0, top_p=0.9, ignore_eos=True
    )
    prompt = [(11 * j) % 500 + 1 for j in range(24)]
    expected, none, alone = _serve(tmp_path, False, [prompt], sampling)
    tokens, chained, stats = _serve(tmp_path, True, [prompt], sampling)
    assert len(expected[0]) == 48 and none == 0
    assert tokens == expected
    assert chained > 5
    assert stats["spec_num_discarded"] == 0
    for name in ("spec_num_drafts", "spec_num_accepted_tokens"):
        assert stats[name] == alone[name]


def test_many_requests_tokens_with_chained_cycles(tmp_path, monkeypatch):
    """Requests that end at a stop token, with a late arrival: sequences end
    while a row of theirs is in flight (discarded), prompt rows make a step
    wait for the cycle in flight, and the rest chain."""
    monkeypatch.setitem(w4a16.W4A16_CONFIG, "splits", 1)
    monkeypatch.setitem(w4a16.W4A16_CONFIG, "layout", "tiled")
    free = SamplingOptions(max_tokens=40, temperature=0.0, ignore_eos=True)
    prompts = [[(11 * i + j) % 500 + 1 for j in range(12 + 5 * i)] for i in range(5)]
    late = [(13 * j) % 500 + 1 for j in range(40)]
    unstopped, _, _ = _serve(tmp_path, False, prompts, free, late)
    assert all(len(row) == 40 for row in unstopped)
    # Each request stops at a token it first emits some steps in, at a
    # different step each: nothing the host can see coming.
    sampling, stops = [], []
    for index, row in enumerate(unstopped):
        at = next((i for i in range(8 + 4 * index, 40) if row[i] not in row[:i]), None)
        stops.append(None if at is None else row[at])
        sampling.append(free if at is None else SamplingOptions(
            max_tokens=40, temperature=0.0, ignore_eos=True, stop_token_ids=[row[at]]
        ))
    assert sum(stop is not None for stop in stops) >= 4
    expected, none, _ = _serve(tmp_path, False, prompts, sampling, late)
    assert none == 0
    for row, stop in zip(expected, stops):
        assert len(row) == 40 or row[-1] == stop
    assert sum(len(row) < 40 for row in expected) >= 4
    tokens, chained, stats = _serve(tmp_path, True, prompts, sampling, late)
    assert chained > 5
    assert stats["spec_num_discarded"] > 0
    # A discarded row changes how many rows its step pads to, and a kernel's
    # sums with it; a greedy stream may then split at a near-tie, as between
    # any two batch sizes. Nearly every request must come out the same.
    same = sum(got == want for got, want in zip(tokens, expected))
    assert same >= len(expected) - 1, (tokens, expected)
