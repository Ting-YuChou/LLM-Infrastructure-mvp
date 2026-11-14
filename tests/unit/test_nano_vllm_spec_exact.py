"""Speculative decoding with a real (tiny) target and a DFlash drafter.

The drafter has random weights, so it is nearly always wrong; what these
check is that the output is the target's own whatever it proposes, and that
the drafter is fed what its reference forward is fed: the target's hidden
states of every token whose KV is computed.
"""

import asyncio
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.core import EngineCore, EngineRequest  # noqa: E402
from serving.nano_vllm.metadata import build_batch  # noqa: E402
from serving.nano_vllm.model_runner import ModelRunner  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.sequence import Sequence  # noqa: E402
from test_nano_vllm_dflash import TINY_CONFIG, _drafter_state, _save_checkpoint  # noqa: E402
from test_nano_vllm_engine_batching import (  # noqa: E402
    GREEDY,
    ListTokenizer,
    _assert_all_blocks_free,
    _engine,
)
from test_nano_vllm_model import _load_nano, _save_tiny_hf_llama  # noqa: E402

TAP_LAYERS = tuple(TINY_CONFIG["dflash_config"]["target_layer_ids"])  # (0, 2)
PROMPTS = {
    "a": [5, 17, 3, 58, 42, 9, 61, 8, 33, 50, 11],
    "b": [5, 17, 3, 58, 42, 9, 61, 8, 40, 41],  # shares two full blocks with a
    "c": [7, 7, 50],
}


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


def _drafter_path(tmp_path, seed: int) -> str:
    path = tmp_path / f"drafter{seed}"
    return str(_save_checkpoint(path, TINY_CONFIG, _drafter_state(TINY_CONFIG, seed=seed)))


def _hf_taps(hf_model, token_ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        hidden = hf_model(torch.tensor([token_ids]), output_hidden_states=True).hidden_states
    return torch.cat([hidden[layer + 1][0] for layer in TAP_LAYERS], dim=-1)


def _hf_greedy(hf_model, prompt_ids: list[int], max_tokens: int) -> list[int]:
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            ids.append(int(hf_model(torch.tensor([ids])).logits[0, -1].argmax()))
            if ids[-1] == hf_model.config.eos_token_id:
                break
    return ids[len(prompt_ids) :]


def test_taps_are_the_hidden_states_after_the_tapped_layers(target):
    hf_model, path = target
    model = _load_nano(path)
    token_ids = PROMPTS["a"]
    cache = model.new_paged_cache(num_blocks=4, block_size=4)
    input_ids, positions, metadata = build_batch(
        input_ids=token_ids, positions=list(range(len(token_ids))), slots=list(range(len(token_ids))),
        block_tables=[[0, 1, 2]], context_lens=[len(token_ids)], query_lens=[len(token_ids)],
        is_prefill=True, null_block=cache.null_block, num_slots=cache.total_slots, device=model.device,
    )
    logits, taps = model.forward_batch_taps(input_ids, positions, cache, metadata, TAP_LAYERS)
    plain = model.forward_batch(input_ids, positions, cache, metadata)
    assert torch.equal(logits, plain)
    torch.testing.assert_close(taps, _hf_taps(hf_model, token_ids), rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match="before the last"):
        model.forward_batch_taps(input_ids, positions, cache, metadata, (3,))


@pytest.mark.parametrize("scheduling", ["mixed", "prefill_first"])
@pytest.mark.parametrize("draft_tokens", [0, 2])
def test_output_matches_transformers_greedy_whatever_the_drafter(
    tmp_path, target, scheduling, draft_tokens
):
    hf_model, path = target
    sampling = SamplingOptions(max_tokens=12, **GREEDY)

    def run(drafter_seed):
        async def scenario():
            engine = _engine(
                ListTokenizer(PROMPTS),
                _load_nano(path),
                model_name=str(path),
                max_num_batched_tokens=12,  # chunked prefill next to verify rows
                enable_prefix_cache=True,
                scheduling=scheduling,
                speculative_model=None if drafter_seed is None else _drafter_path(tmp_path, drafter_seed),
                num_speculative_tokens=draft_tokens,
            )
            try:
                first = await asyncio.gather(*(engine.generate_async(name, sampling) for name in PROMPTS))
                # Again, from blocks the prefix cache kept (both KV pools).
                second = await asyncio.gather(*(engine.generate_async(name, sampling) for name in PROMPTS))
            finally:
                await engine.shutdown()
            _assert_all_blocks_free(engine)
            assert (engine.core.proposer is None) == (drafter_seed is None)
            return [r.token_ids for r in first], [r.token_ids for r in second]

        return asyncio.run(scenario())

    expected = [_hf_greedy(hf_model, ids, 12) for ids in PROMPTS.values()]
    assert run(None) == (expected, expected)
    assert run(0) == (expected, expected)
    assert run(1) == (expected, expected)


def test_drafter_context_is_the_hidden_states_of_the_computed_tokens(tmp_path, target):
    hf_model, path = target
    model = _load_nano(path)
    config = NanoVLLMConfig(
        model_name=str(path), max_model_len=64, max_new_tokens=40, kvcache_block_size=4,
        num_kvcache_blocks=32, max_num_seqs=4, max_num_batched_tokens=8,
        attention_backend="torch", eager_load=False, speculative_model=_drafter_path(tmp_path, 0),
    )
    core = EngineCore(config, model)
    for req_id, prompt in enumerate(PROMPTS.values()):
        core.add_request(
            EngineRequest(req_id, list(prompt), SamplingOptions(max_tokens=40, **GREEDY))
        )
    while min((seq.num_completion_tokens for seq in core.requests.values()), default=0) < 9:
        assert all(output.error is None for output in core.step())
    assert len(core.requests) == len(PROMPTS)

    proposer = core.proposer
    drafter, used = proposer.drafter, proposer.cache
    for seq in core.requests.values():
        computed = seq.num_cached_tokens
        assert computed == seq.num_tokens - 1
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


class _Recorder:
    """A proposer that keeps what observe() is given."""

    num_draft_tokens = 3
    tap_layers = TAP_LAYERS

    def __init__(self):
        self.seen = []

    def observe(self, taps, positions, slot_mapping):
        self.seen.append((taps.clone(), positions.clone(), slot_mapping.clone()))

    def propose(self, batch):
        return torch.full((len(batch.seqs), self.num_draft_tokens), 7, dtype=torch.long)


@pytest.mark.parametrize("pieces", [False, True])
def test_a_speculative_step_observes_every_token_with_and_without_piecewise_graphs(target, pieces):
    hf_model, path = target
    model = _load_nano(path)
    Sequence.block_size = 4
    cache = model.new_paged_cache(num_blocks=16, block_size=4)
    manager = BlockManager(num_blocks=16, block_size=4, enable_prefix_cache=False)
    recorder = _Recorder()
    runner = ModelRunner(model, cache, manager, max_rows=4, proposer=recorder)
    if pieces:
        # One logits row per sequence in the graphs; the step below has five.
        runner.capture_piecewise_graphs(max_tokens=32, max_rows=4)
    greedy = SamplingOptions(max_tokens=8, **GREEDY)
    decoding = Sequence(prompt_token_ids=list(PROMPTS["a"]), sampling=greedy)
    prefilling = Sequence(prompt_token_ids=list(PROMPTS["c"]), sampling=greedy)
    for seq in (decoding, prefilling):
        manager.allocate(seq)

    # The first sequence's prompt, then a step with its verify row and the
    # second one's prompt.
    decoding.num_scheduled_tokens = decoding.num_tokens
    (first,) = runner.run([decoding], is_prefill=True)
    assert first == _hf_greedy(hf_model, PROMPTS["a"], 1)[0]
    decoding.num_cached_tokens = decoding.num_tokens
    decoding.append_token(first)
    decoding.is_prefill = False
    manager.may_append(decoding, lookahead=3)
    prefilling.num_scheduled_tokens = prefilling.num_tokens
    sampled = []
    sample = runner.sampler.sample
    runner.sampler.sample = lambda logits, options, seq_ids, positions, out, slot=0: (
        sampled.append((list(seq_ids), list(positions))),
        sample(logits, options, seq_ids, positions, out, slot=slot),
    )
    tokens = runner.run([decoding, prefilling], is_prefill=True)
    # Each verify row draws as a plain decode step at its position would.
    assert sampled == [(
        [decoding.seq_id] * 4 + [prefilling.seq_id],
        list(range(decoding.num_tokens, decoding.num_tokens + 4)) + [prefilling.num_tokens],
    )]

    block = decoding.token_ids + [7, 7, 7]
    with torch.no_grad():
        choices = hf_model(torch.tensor([block])).logits[0].argmax(dim=-1).tolist()
    start = decoding.num_tokens - 1
    accepted = 0
    while accepted < 3 and choices[start + accepted] == 7:
        accepted += 1
    assert tokens == [
        choices[start : start + accepted + 1], _hf_greedy(hf_model, PROMPTS["c"], 1)[0]
    ]
    taps, positions, _ = recorder.seen[-1]
    assert positions.tolist() == list(range(start, start + 4)) + [0, 1, 2]
    torch.testing.assert_close(taps[:4], _hf_taps(hf_model, block)[start:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(taps[4:], _hf_taps(hf_model, PROMPTS["c"]), rtol=1e-4, atol=1e-4)
