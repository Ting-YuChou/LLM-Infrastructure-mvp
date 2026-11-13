"""Speculative decoding of sampled requests.

A token is drawn with noise that depends only on (seed, sequence id,
position), so with one seed a request's tokens are the same with and without
a drafter, whatever the drafter proposes and however it draws its drafts.
Coupled sampling draws the drafts with the noise the target will use at the
same positions: a drafter whose logits are the target's then has every draft
accepted, at any temperature.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from test_nano_vllm_dflash import TINY_CONFIG  # noqa: E402
from test_nano_vllm_model import _save_tiny_hf_llama  # noqa: E402
from test_nano_vllm_spec_exact import PROMPTS, _drafter_path  # noqa: E402
from test_nano_vllm_spec_graph import MAX_TOKENS, _core, _Oracle, _run  # noqa: E402

SAMPLED = SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.9, top_p=0.95, top_k=40)
DRAFTED = 3


@pytest.fixture
def target(tmp_path):
    return _save_tiny_hf_llama(
        tmp_path / "target",
        vocab_size=TINY_CONFIG["vocab_size"],
        hidden_size=TINY_CONFIG["hidden_size"],
        intermediate_size=64,
        num_hidden_layers=TINY_CONFIG["num_target_layers"],
    )


def _plain(path, prompts, sampling=SAMPLED, **overrides):
    outputs, _ = _run(_core(path, **overrides), prompts, sampling=sampling)
    return outputs


class _TargetAsDrafter:
    """Drafts from the target's own logits (transformers, one token at a
    time): what a perfect drafter would propose."""

    tap_layers = ()

    def __init__(self, hf_model, num_draft_tokens: int):
        self.hf_model = hf_model
        self.num_draft_tokens = num_draft_tokens

    def observe(self, taps, positions, slot_mapping):
        raise AssertionError("no taps were asked for")

    def propose(self, batch):
        drafted, vocab = batch.num_draft_tokens, self.hf_model.config.vocab_size
        tokens = [list(seq.token_ids) for seq in batch.seqs]
        logits = torch.zeros(len(tokens) * drafted, vocab)
        for j in range(drafted):
            for row, ids in enumerate(tokens):
                with torch.no_grad():
                    logits[row * drafted + j] = self.hf_model(torch.tensor([ids])).logits[0, -1]
            # Rows of later drafts are not known yet; a row's draw is its own.
            drawn = logits.argmax(dim=-1) if batch.draw is None else batch.draw(logits)
            for row, ids in enumerate(tokens):
                ids.append(int(drawn[row * drafted + j]))
        return torch.tensor([ids[-drafted:] for ids in tokens])


def test_sampled_output_is_not_the_greedy_one(target):
    _, path = target
    prompts = list(PROMPTS.values())
    greedy = _plain(path, prompts, sampling=SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.0))
    assert _plain(path, prompts) != greedy
    assert _plain(path, prompts) == _plain(path, prompts)  # one seed, one stream
    assert _plain(path, prompts, seed=1) != _plain(path, prompts)


@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
@pytest.mark.parametrize("draft_temperature", [1.0, 0.5, 0.0], ids=["coupled", "half", "argmax drafts"])
def test_sampled_tokens_with_a_drafter_are_the_tokens_without(tmp_path, target, draft_temperature, graphs):
    _, path = target
    prompts = list(PROMPTS.values())
    expected = _plain(path, prompts)
    core = _core(
        path, graphs=graphs, speculative_model=_drafter_path(tmp_path, 0),
        speculative_draft_temperature=draft_temperature, max_num_batched_tokens=8,
    )
    # An arrival every third step: verify rows next to prompt chunks too.
    outputs, paths = _run(core, prompts, arrive_every=3, sampling=SAMPLED)
    assert outputs == expected
    assert (core.model_runner.draft_sampler is not None) == (draft_temperature > 0)
    assert paths["eager"] > 0 and (paths["graph"] > 0) == bool(graphs)


@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
def test_greedy_and_sampled_requests_in_one_batch(tmp_path, target, graphs):
    _, path = target
    prompts = list(PROMPTS.values())
    mixed = [SAMPLED, SamplingOptions(max_tokens=MAX_TOKENS, temperature=0.0), SAMPLED]
    expected, _ = _run(_core(path), prompts, sampling=mixed)
    core = _core(path, graphs=graphs, speculative_model=_drafter_path(tmp_path, 1))
    outputs, _ = _run(core, prompts, sampling=mixed)
    assert outputs == expected
    assert expected[1] == _plain(path, prompts, sampling=mixed[1])[1]


@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
def test_accepted_drafts_of_a_sampled_request_keep_its_stream(target, graphs):
    _, path = target
    prompt = PROMPTS["a"]
    (expected,) = _plain(path, [prompt])
    # Right about the sampled continuation, except at every fifth position.
    proposer = _Oracle(prompt + expected, num_draft_tokens=DRAFTED, wrong_every=5)
    core = _core(path, proposer, graphs)
    outputs, _ = _run(core, [prompt], sampling=SAMPLED)
    assert outputs == [expected]
    stats = core.stats()
    assert stats["spec_num_accepted_tokens"] > stats["spec_num_drafts"]


def test_a_drafter_with_the_targets_logits_has_every_coupled_draft_accepted(target):
    hf_model, path = target
    prompts = list(PROMPTS.values())
    expected = _plain(path, prompts)

    def run(draft_temperature):
        core = _core(
            path, _TargetAsDrafter(hf_model, DRAFTED),
            speculative_draft_temperature=draft_temperature, max_new_tokens=MAX_TOKENS,
        )
        outputs, _ = _run(core, prompts, sampling=SAMPLED)
        assert outputs == expected
        stats = core.stats()
        return stats["spec_num_accepted_tokens"], stats["spec_num_drafts"]

    accepted, cycles = run(1.0)
    # MAX_TOKENS is a multiple of a cycle's DRAFTED + 1 tokens: no cycle is cut short.
    assert MAX_TOKENS % (DRAFTED + 1) == 0
    assert (accepted, cycles) == (DRAFTED * cycles, len(prompts) * MAX_TOKENS // (DRAFTED + 1))
    argmax_accepted, argmax_cycles = run(0.0)
    assert argmax_accepted / argmax_cycles < 0.7 * DRAFTED
    # Between the two: more than the argmax, less than everything.
    half_accepted, half_cycles = run(0.5)
    assert argmax_accepted / argmax_cycles < half_accepted / half_cycles < DRAFTED


class _TableDrafter:
    """Draft logits that depend on the position alone (a fixed table), drawn
    by ``DraftBatch.draw``: it reads nothing but the batch's tensors."""

    tap_layers = ()
    device_only = True

    def __init__(self, vocab: int, num_draft_tokens: int):
        self.table = torch.randn(128, vocab, generator=torch.Generator().manual_seed(3))
        self.num_draft_tokens = num_draft_tokens

    def observe(self, taps, positions, slot_mapping):
        raise AssertionError("no taps were asked for")

    def propose(self, batch):
        positions = batch.positions.view(-1, batch.num_draft_tokens + 1)[:, 1:]
        return batch.draw(self.table[positions.reshape(-1)]).view(positions.shape)


@pytest.mark.parametrize("graphs", [None, 4], ids=["eager", "graphs"])
@pytest.mark.parametrize("draft_temperature", [1.0, 0.6])
def test_drafts_are_drawn_with_the_noise_of_the_positions_they_propose(target, graphs, draft_temperature):
    from serving.nano_vllm.sampling import Sampler

    _, path = target
    prompts = list(PROMPTS.values())
    proposer = _TableDrafter(TINY_CONFIG["vocab_size"], DRAFTED)
    core = _core(path, proposer, graphs, seed=11, speculative_draft_temperature=draft_temperature)
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

    # The same draws from a sampler of our own: seed, sequence id, position,
    # at the fraction of each request's temperature.
    reference = Sampler(max_rows=DRAFTED, device="cpu", seed=11, temperature_scale=draft_temperature)
    out = torch.zeros(DRAFTED, dtype=torch.long)
    checked = 0
    for rows, drafts in cycles:
        for (seq_id, num_tokens), drafted in zip(rows, drafts):
            count = drafted.numel()  # the step's drafts per sequence
            positions = list(range(num_tokens, num_tokens + count))
            reference.sample(
                proposer.table[positions], [settings[seq_id]] * count, [seq_id] * count, positions, out
            )
            assert drafted.tolist() == out[:count].tolist()
            checked += 1
    assert checked > 20
