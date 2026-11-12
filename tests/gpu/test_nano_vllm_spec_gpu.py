"""The speculative cycle as one CUDA graph (ModelRunner.capture_spec_graphs):
a replay must do what the eager cycle does, without waiting for the GPU."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.dflash import DFlashDrafter  # noqa: E402
from serving.nano_vllm.model_runner import ModelRunner  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.sequence import Sequence  # noqa: E402
from serving.nano_vllm.spec import DFlashProposer, accept_prefix  # noqa: E402
from test_nano_vllm_block_attention_gpu import drafter_checkpoint  # noqa: E402
from test_nano_vllm_forward_gpu import _model  # noqa: E402

BLOCK_SIZE = 16
PROMPT_LENS = [17, 30, 5, 64, 33]
MAX_MODEL_LEN = 128


def _runner(tmp_path, max_graph_batch, weight_quant="none", kv_cache_dtype="auto", drafter_quant="none"):
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
    model.extend_rope_table(MAX_MODEL_LEN + proposer.num_draft_tokens)
    runner = ModelRunner(model, cache, manager, max_rows=8, proposer=proposer)
    runner.capture_spec_graphs(max_batch=max_graph_batch, max_model_len=MAX_MODEL_LEN)
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


def _graph_outputs(runner, batch):
    """What the last replay for ``batch`` sequences left: verify logits
    [batch, block, vocab], drafts [batch, K] and the sampled tokens [batch, block]."""
    block = runner.block
    logits, drafts = runner._spec.outputs[runner._spec.bucket(batch)]
    return (
        logits[: batch * block].float().cpu().view(batch, block, -1),
        drafts[:batch].cpu(),
        runner.sampled[: batch * block].cpu().view(batch, block),
    )


def _eager_cycle(runner, seqs):
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
        tokens = runner.run(seqs, is_prefill=False)
    finally:
        runner._spec = graphs
        runner.proposer.propose, runner.sampler.sample = propose, sample
    return tokens, seen["logits"].float().cpu(), seen["drafts"].cpu()


@pytest.mark.parametrize(
    "batch,max_graph_batch,weight_quant,kv_cache_dtype,drafter_quant",
    [
        (1, 4, "none", "auto", "none"),
        (3, 4, "none", "auto", "none"),  # three rows padded to four
        (4, 4, "none", "auto", "none"),
        (5, 5, "none", "auto", "none"),  # the largest size is not a listed one
        (3, 4, "awq_int8head", "auto", "none"),
        (3, 4, "w4a16", "auto", "w4a16"),
        (3, 4, "none", "fp8", "none"),
        (3, 4, "none", "auto", "w4a16"),
    ],
)
def test_graph_cycle_matches_the_eager_cycle(
    tmp_path, batch, max_graph_batch, weight_quant, kv_cache_dtype, drafter_quant
):
    runner, manager = _runner(tmp_path, max_graph_batch, weight_quant, kv_cache_dtype, drafter_quant)
    block, drafted = runner.block, runner.proposer.num_draft_tokens
    seqs = _prefilled(runner, manager, batch)
    compared = 0
    # Twelve cycles cross block boundaries; from the seventh only the first
    # sequence runs, in a smaller graph, over buffers the others left stale.
    for cycle in range(12):
        active = seqs if cycle < 6 else seqs[:1]
        for seq in active:
            manager.may_append(seq, lookahead=drafted)
        # A cycle only writes KV past the cached tokens, so the same state
        # can run it twice.
        replayed = runner.run(active, is_prefill=False)
        graph_logits, graph_drafts, choices = _graph_outputs(runner, len(active))
        eager, eager_logits, eager_drafts = _eager_cycle(runner, active)
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
    runner.sampler.sample(logits.view(3 * block, vocab), *runner._spec_sampling(seqs, 3), verify)
    draft_logits = logits[:, :drafted].reshape(3 * drafted, vocab).contiguous()
    if graph:  # as _replay_spec stages it, with a padded fourth sequence
        runner.draft_sampler.stage(*runner._draft_sampling(seqs), vocab, padded_rows=4 * drafted)
        drafts = runner._graph_draft_draw(draft_logits)
    else:
        drafts = runner._draft_draw(seqs, slot=0)(draft_logits)

    verify, drafts = verify.view(3, block).cpu(), drafts.view(3, drafted).cpu()
    assert torch.equal(drafts, verify[:, :drafted])
    # The sampled rows drew: not every token is its row's largest logit.
    values = logits.float().cpu()
    is_max = values.gather(-1, verify.unsqueeze(-1)).squeeze(-1) == values.max(dim=-1).values
    assert is_max[1].all() and not is_max[0].all()
