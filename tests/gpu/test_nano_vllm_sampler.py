"""The sort-free CUDA sampler (kernels/csrc/sampler.cu) against sampling.keep_mask.

After a call, each row's state holds the lowest 16-bit key the kernels kept
(the cutoff). The set it implies, every token at or above it, must equal the
float64 reference's for top-k and lie between the reference's sets at top_p
* (1 - 1e-4) and top_p * (1 + 1e-4) for top-p: the kernels sum fp32 masses.
Draws are checked by membership and, against the exact truncated softmax, by
a chi-square test; greedy rows must be torch.argmax bit for bit.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels.sampler import (  # noqa: E402
    STATE_CUT_KEY,
    sampler_extension,
    sampler_state,
)
from serving.nano_vllm.sampling import Sampler, SamplingOptions, keep_mask  # noqa: E402

VOCAB = 128256
SLACK = 1e-4


def _keys(logits):
    """The kernels' order keys of 16-bit logits (NaN as -inf, -0 as +0)."""
    bits = logits.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    inf, neg_inf = (0x7F80, 0xFF80) if logits.dtype == torch.bfloat16 else (0x7C00, 0xFC00)
    bits = torch.where((bits & 0x7FFF) > inf, neg_inf, bits)
    bits = torch.where(bits == 0x8000, 0, bits)
    return torch.where((bits & 0x8000) != 0, ~bits & 0xFFFF, bits | 0x8000)


def _logits(dist, rows, vocab=VOCAB, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    randn = lambda scale: scale * torch.randn(rows, vocab, device="cuda", generator=gen)  # noqa: E731
    if dist == "flat":
        logits = randn(1.0)
    elif dist == "ties":  # few distinct values: cutoffs land on ties
        logits = (randn(2.0) * 2).round() / 2
    else:
        logits = randn(2.5)
        spikes = torch.randint(0, vocab, (rows, 8), device="cuda", generator=gen)
        logits.scatter_add_(1, spikes, 10 + 4 * torch.rand(rows, 8, device="cuda", generator=gen))
    if dist == "masked":  # a tenth of the vocab at -inf, as logit masks leave it
        logits = logits.masked_fill(torch.rand(rows, vocab, device="cuda", generator=gen) < 0.1, float("-inf"))
    return logits.to(dtype)


def _run(logits, options, seed=0, positions=None, seq_ids=None):
    assert sampler_extension() is not None, "building kernels/csrc/sampler.cu failed"
    rows = logits.shape[0]
    sampler = Sampler(rows, logits.device, seed=seed)
    out = torch.full((rows,), -1, dtype=torch.long, device=logits.device)
    sampler.sample(
        logits,
        options,
        list(range(rows)) if seq_ids is None else seq_ids,
        [0] * rows if positions is None else positions,
        out,
    )
    torch.cuda.synchronize()
    cut = sampler_state(logits.device)[:rows, STATE_CUT_KEY].clone()
    return out, cut


def _reference(logits, options, slack=0.0):
    rows, vocab = logits.shape
    temperature = torch.tensor([o.temperature for o in options], dtype=torch.float64, device="cuda")
    top_p = torch.tensor(
        [min(1.0, o.top_p * (1 + slack)) if 0 < o.top_p < 1 else 1.0 for o in options],
        dtype=torch.float64,
        device="cuda",
    )
    top_k = torch.tensor([o.top_k if 0 < o.top_k < vocab else 0 for o in options], device="cuda")
    values = logits.double().nan_to_num(nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    return keep_mask(values, temperature.clamp(min=1e-6), top_p, top_k)


def _check_rows(logits, options, seed=0):
    out, cut = _run(logits, options, seed=seed)
    keys = _keys(logits)
    kept = torch.where(cut.view(-1, 1) >= 0, keys >= cut.view(-1, 1), torch.ones_like(keys, dtype=torch.bool))
    loose = _reference(logits, options, SLACK)
    tight = _reference(logits, options, -SLACK)
    values = logits.float().nan_to_num(nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    argmax_rows = ~torch.isfinite(values.max(dim=-1).values)  # no finite max: the argmax
    for row, option in enumerate(options):
        if option.temperature <= 0 or argmax_rows[row]:
            assert out[row].item() == values[row].argmax().item()
            continue
        assert not (tight[row] & ~kept[row]).any(), f"row {row}: kept too little for {option}"
        assert not (kept[row] & ~loose[row]).any(), f"row {row}: kept too much for {option}"
        assert kept[row, out[row]], f"row {row}: drew a token outside the kept set"
    return out, cut


SETTINGS = {
    "p0.9": SamplingOptions(temperature=0.7, top_p=0.9),
    "cold_p0.9": SamplingOptions(temperature=0.1, top_p=0.9),
    "hot_p0.5": SamplingOptions(temperature=2.0, top_p=0.5),
    "p0.1": SamplingOptions(temperature=1.0, top_p=0.1),
    "k1": SamplingOptions(temperature=0.7, top_p=1.0, top_k=1),
    "k50": SamplingOptions(temperature=0.7, top_p=1.0, top_k=50),
    "k50_p0.9": SamplingOptions(temperature=0.7, top_p=0.9, top_k=50),
    "k1000_p0.5": SamplingOptions(temperature=1.0, top_p=0.5, top_k=1000),
    "plain": SamplingOptions(temperature=1.0, top_p=1.0),
    "greedy": SamplingOptions(temperature=0.0),
}


@pytest.mark.parametrize("dist", ["peaked", "flat", "ties", "masked"])
@pytest.mark.parametrize("setting", SETTINGS.values(), ids=SETTINGS)
def test_kept_set_matches_reference(setting, dist):
    _check_rows(_logits(dist, 64), [setting] * 64)


@pytest.mark.parametrize("rows", [1, 7, 20, 256])  # 1, 2 and 4 loads per thread
@pytest.mark.parametrize("vocab", [VOCAB, 1000])
def test_rows_and_vocab_sizes(rows, vocab):
    options = [list(SETTINGS.values())[r % len(SETTINGS)] for r in range(rows)]
    _check_rows(_logits("peaked", rows, vocab), options)


def test_fp16_logits():
    options = [list(SETTINGS.values())[r % len(SETTINGS)] for r in range(30)]
    _check_rows(_logits("peaked", 30, dtype=torch.float16), options)


def test_top_k_and_top_p_cutoffs_in_both_branches():
    # A top-p cutoff in a higher histogram bin (the key's high bits) than the
    # top-k one needs the second refine pass (K4); one inside the top-k bin is
    # resolved by the first. The first rows have one dominant token (40.0) far
    # above their 1000th, the others a flat top 50 inside one bin.
    logits = torch.cat([_logits("peaked", 32, seed=1), _logits("flat", 32, seed=2)])
    logits[:32, 0] = 40.0
    options = [SamplingOptions(temperature=1.0, top_p=0.5, top_k=1000)] * 32 + [
        SamplingOptions(temperature=0.7, top_p=0.99, top_k=50)
    ] * 32
    _, cut = _check_rows(logits, options)
    keys = _keys(logits)
    kth = keys.sort(dim=-1, descending=True).values[:, [999] * 32 + [49] * 32].diagonal()
    low_bits = sampler_extension().KEY_LOW_BITS
    above = (cut >> low_bits) > (kth >> low_bits)
    assert above[:32].all() and not above[32:].any()


def test_greedy_rows_take_the_first_maximum():
    logits = _logits("ties", 16)
    logits[:, 100] = logits[:, 7] = 50.0  # tied maxima: index 7 wins, as with torch.argmax
    out, _ = _run(logits, [SamplingOptions(temperature=0.0)] * 16)
    assert (out == 7).all()


def test_degenerate_rows_fall_back_to_argmax():
    logits = _logits("peaked", 4)
    logits[0] = float("-inf")
    logits[1, 5] = float("nan")  # counts as -inf
    logits[2, 9] = float("inf")
    out, _ = _check_rows(logits, [SamplingOptions(temperature=0.7, top_p=0.9)] * 4)
    assert out[0].item() == 0 and out[1].item() != 5 and out[2].item() == 9


def test_draws_are_deterministic_per_seed_sequence_and_position():
    logits = _logits("flat", 64)
    options = [SamplingOptions(temperature=1.0, top_p=1.0)] * 64
    first, _ = _run(logits, options, positions=list(range(64)))
    again, _ = _run(logits, options, positions=list(range(64)))
    moved, _ = _run(logits, options, positions=list(range(1, 65)))
    reseeded, _ = _run(logits, options, seed=1, positions=list(range(64)))
    assert torch.equal(first, again)
    assert (first != moved).sum() > 50 and (first != reseeded).sum() > 50
    # A row's draw does not depend on the rows around it.
    alone, _ = _run(logits[10:11].clone(), options[:1], positions=[10], seq_ids=[10])
    assert alone[0] == first[10]


def _chi_square_pvalue(counts, probs):
    expected = probs * counts.sum()
    big = expected >= 5
    observed = torch.cat([counts[big], counts[~big].sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = expected > 0
    stat = ((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum()
    df = keep.sum() - 1
    return torch.special.gammaincc(df / 2, stat / 2).item()


@pytest.mark.parametrize(
    "vocab,option",
    [
        (64, SamplingOptions(temperature=1.0, top_p=1.0)),
        (64, SamplingOptions(temperature=0.8, top_p=0.9)),
        (64, SamplingOptions(temperature=1.2, top_p=1.0, top_k=10)),
        (VOCAB, SamplingOptions(temperature=0.7, top_p=0.9)),
    ],
    ids=["plain", "top_p", "top_k", "full_vocab_top_p"],
)
def test_draws_follow_the_truncated_softmax(vocab, option):
    row = _logits("peaked" if vocab == VOCAB else "flat", 1, vocab, seed=3)
    if vocab != VOCAB:
        row = (row.float() * 1.5).to(torch.bfloat16)
    rows, calls = 256, 800
    logits = row.expand(rows, vocab).contiguous()
    counts = torch.zeros(vocab, dtype=torch.float64, device="cuda")
    sampler = Sampler(rows, "cuda", seed=0)
    out = torch.empty(rows, dtype=torch.long, device="cuda")
    for call in range(calls):
        sampler.sample(logits, [option] * rows, list(range(rows)), [call] * rows, out)
        counts += torch.bincount(out, minlength=vocab).double()
        torch.cuda.synchronize()  # before the next call rewrites the pinned settings
    keep = _reference(row, [option])[0]
    scaled = row[0].double() / option.temperature
    probs = torch.softmax(scaled.masked_fill(~keep, float("-inf")), dim=0)
    assert counts[~keep].sum() == 0
    assert _chi_square_pvalue(counts[keep], probs[keep]) > 1e-3


def test_sampler_replays_in_a_cuda_graph():
    logits = _logits("flat", 16)
    options = [SETTINGS["k50_p0.9"]] * 8 + [SETTINGS["p0.9"]] * 8
    eager, _ = _run(logits, options)
    sampler = Sampler(16, "cuda", seed=0)
    out = torch.empty(16, dtype=torch.long, device="cuda")
    run = lambda: sampler.sample(logits, options, list(range(16)), [0] * 16, out)  # noqa: E731
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_fp32_logits_take_the_pytorch_path():
    logits = _logits("peaked", 8).float()
    out, _ = _run(logits, [SETTINGS["k1"]] * 8)
    assert torch.equal(out, logits.argmax(dim=-1))
