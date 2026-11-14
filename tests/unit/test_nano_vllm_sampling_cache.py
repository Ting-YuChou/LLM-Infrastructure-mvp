import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kv_cache import StaticKVCache  # noqa: E402
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402
from serving.nano_vllm.sampling import filter_logits, sample_next_token  # noqa: E402


def test_sample_next_token_uses_argmax_when_temperature_zero():
    logits = torch.tensor([0.1, 4.0, 1.0])

    assert sample_next_token(logits, temperature=0.0) == 1


def test_filter_logits_keeps_top_k_only():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])

    filtered = filter_logits(logits, top_k=2)

    assert torch.isneginf(filtered[0])
    assert torch.isneginf(filtered[1])
    assert torch.isfinite(filtered[2])
    assert torch.isfinite(filtered[3])


def test_static_kv_cache_update_and_read():
    cache = StaticKVCache(
        num_layers=2,
        batch_size=1,
        num_key_value_heads=2,
        max_seq_len=4,
        head_dim=3,
        dtype=torch.float32,
        device="cpu",
    )
    key = torch.ones(1, 2, 2, 3)
    value = torch.full((1, 2, 2, 3), 2.0)

    cache.update(0, torch.tensor([1, 2]), key, value)
    read_key, read_value = cache.read(0)

    assert read_key.shape == (1, 2, 3, 3)
    assert read_value.shape == (1, 2, 3, 3)
    torch.testing.assert_close(read_key[:, :, 1:3], key)
    torch.testing.assert_close(read_value[:, :, 1:3], value)


def test_paged_kv_cache_update_and_read_sequence():
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=2,
        block_size=2,
        num_key_value_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    key = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]]])
    value = torch.tensor([[[0.0, 1.0]], [[0.0, 2.0]], [[0.0, 3.0]]])

    cache.update(0, torch.tensor([0, 1, 2]), key, value)
    read_key, read_value = cache.read_sequence(0, [0, 1], seq_len=3)

    assert read_key.shape == (1, 1, 3, 2)
    assert read_value.shape == (1, 1, 3, 2)
    torch.testing.assert_close(read_key[0, 0], key[:, 0])
    torch.testing.assert_close(read_value[0, 0], value[:, 0])


def test_sample_tokens_is_argmax_for_greedy_rows():
    from serving.nano_vllm.sampling import SamplingOptions, sample_tokens

    logits = torch.tensor([[0.1, 4.0, 1.0], [3.0, 0.0, 2.0]])
    options = [SamplingOptions(temperature=0.0)] * 2

    assert sample_tokens(logits, options) == [1, 0]


def test_sample_tokens_mixes_greedy_and_filtered_rows():
    from serving.nano_vllm.sampling import SamplingOptions, sample_tokens

    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [0.0, 1.0, 5.0, 2.0],  # greedy
            [3.0, 1.0, 2.9, 0.0],  # top_k=1 always keeps the argmax
            [0.0, 9.0, 0.0, 0.0],  # top_p keeps only the dominant token
            [float("-inf")] * 4,  # degenerate row falls back instead of raising
        ]
    )
    options = [
        SamplingOptions(temperature=0.0),
        SamplingOptions(temperature=1.0, top_k=1, top_p=1.0),
        SamplingOptions(temperature=0.7, top_p=0.5),
        SamplingOptions(temperature=1.0, top_p=1.0),
    ]

    for _ in range(20):
        tokens = sample_tokens(logits, options)
        assert tokens[:3] == [2, 0, 1]
        assert 0 <= tokens[3] < 4


def test_sample_tokens_samples_within_top_k():
    from serving.nano_vllm.sampling import SamplingOptions, sample_tokens

    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 1.1, 0.9, -5.0, -6.0]] * 64)
    options = [SamplingOptions(temperature=1.0, top_k=3, top_p=1.0)] * 64

    tokens = sample_tokens(logits, options)

    assert set(tokens) <= {0, 1, 2}
    assert len(set(tokens)) > 1


def _keep(logits, temperature=1.0, top_p=1.0, top_k=0):
    from serving.nano_vllm.sampling import keep_mask

    rows = logits.shape[0]
    full = lambda value, dtype: torch.full((rows,), value, dtype=dtype)  # noqa: E731
    return keep_mask(logits, full(temperature, logits.dtype), full(top_p, logits.dtype), full(top_k, torch.int32))


def test_keep_mask_keeps_every_token_tied_with_the_top_k_cutoff():
    logits = torch.tensor([[3.0, 2.0, 2.0, 1.0]])

    assert _keep(logits, top_k=2).tolist() == [[True, True, True, False]]


def test_keep_mask_keeps_every_token_tied_with_the_top_p_cutoff():
    # Masses 1, e^-1, e^-1, e^-2: the mass above either 1.0 is 1 < 0.6 * 1.87,
    # so both stay (a sort would have kept only the first of them).
    logits = torch.tensor([[2.0, 1.0, 1.0, 0.0]])

    assert _keep(logits, top_p=0.6).tolist() == [[True, True, True, False]]


def test_keep_mask_applies_top_p_within_what_top_k_kept():
    # The mass above 1.0 is 1.37 of all five tokens' 1.57 (0.87 < 0.9: kept),
    # but 1.37 of the top 3's 1.50 (0.91: dropped).
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])

    assert _keep(logits, top_p=0.9).tolist() == [[True, True, True, False, False]]
    assert _keep(logits, top_p=0.9, top_k=3).tolist() == [[True, True, False, False, False]]


def test_keep_mask_matches_the_per_token_rule():
    gen = torch.Generator().manual_seed(0)
    for trial in range(200):
        logits = (torch.randn(1, 40, generator=gen, dtype=torch.float64) * 2).round()  # many ties
        temperature = [0.3, 0.7, 1.0, 2.0][trial % 4]
        top_p = [0.1, 0.5, 0.9, 1.0][trial // 4 % 4]
        top_k = [0, 1, 5, 20][trial // 16 % 4]
        v = logits[0] / temperature
        in_k = torch.tensor([(v > x).sum() < top_k if top_k else True for x in v])
        mass = torch.exp(v - v.max()) * in_k
        expected = [
            bool(in_k[i]) and (top_p >= 1 or mass[v > v[i]].sum() < top_p * mass.sum())
            for i in range(v.numel())
        ]
        assert _keep(logits, temperature, top_p, top_k)[0].tolist() == expected


def test_sample_rows_draws_from_the_truncated_softmax():
    from serving.nano_vllm.sampling import sample_rows

    torch.manual_seed(0)
    row = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -3.0])
    rows = 40000
    tokens = sample_rows(
        row.expand(rows, -1),
        torch.full((rows,), 0.8),
        torch.full((rows,), 0.9),
        torch.zeros(rows, dtype=torch.int32),
    )
    keep = _keep(row.view(1, -1).double(), 0.8, 0.9)[0]
    probs = torch.softmax((row.double() / 0.8).masked_fill(~keep, float("-inf")), dim=0)
    freq = torch.bincount(tokens, minlength=6).double() / rows

    assert freq[~keep].sum() == 0
    assert torch.allclose(freq, probs, atol=0.01)


def test_sampler_stages_each_rows_settings_and_samples_on_the_cpu():
    from serving.nano_vllm.sampling import Sampler, SamplingOptions

    sampler = Sampler(max_rows=4, device="cpu", seed=7)
    logits = torch.tensor([[0.0, 5.0, 1.0], [4.0, 0.0, 1.0], [0.0, 0.0, 9.0]])
    options = [
        SamplingOptions(temperature=0.0),
        SamplingOptions(temperature=0.7, top_k=1),
        SamplingOptions(temperature=1.3, top_p=0.5, top_k=3),  # k >= vocab: off
    ]
    out = torch.full((4,), -1, dtype=torch.long)
    seq_ids = [1, 2, (5 << 32) + 3]

    sampler.sample(logits, options, seq_ids, [10, 11, 12], out, slot=1)

    assert out.tolist() == [1, 0, 2, -1]
    params = sampler.params[:, :3]
    assert params[Sampler.TEMPERATURE].view(torch.float32).tolist() == pytest.approx([0.0, 0.7, 1.3])
    assert params[Sampler.TOP_P].view(torch.float32).tolist() == pytest.approx([0.9, 0.9, 0.5])
    assert params[Sampler.TOP_K].tolist() == [0, 1, 0]
    assert params[Sampler.POSITION].tolist() == [10, 11, 12]
    assert params[Sampler.SEQ_LO].tolist() == [1, 2, 3]
    assert params[Sampler.SEQ_HI].tolist() == [0, 0, 5]
    assert torch.equal(sampler.host[1], sampler.params)  # staged in the slot it was given


def test_philox4x32_matches_the_published_vectors():
    from serving.nano_vllm.sampling import philox4x32

    def run(counter, key):
        words = philox4x32(tuple(torch.tensor([c], dtype=torch.int64) for c in counter), key)
        return [int(w) for w in words]

    # Random123's known answers for philox4x32-10.
    assert run((0, 0, 0, 0), (0, 0)) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    ones = 0xFFFFFFFF
    assert run((ones,) * 4, (ones, ones)) == [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]
    assert run((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)) == [
        0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1
    ]


def test_log_noise_is_exponential_and_depends_on_seed_sequence_and_position_only():
    from serving.nano_vllm.sampling import philox_log_exponential

    def noise(seed, seq_ids, positions, vocab=1001):
        ids = torch.tensor(seq_ids)
        return philox_log_exponential(seed, ids & 0xFFFFFFFF, ids >> 32, torch.tensor(positions), vocab)

    big = (7 << 32) + 3
    base = noise(5, [1, 2, big], [10, 10, 11])
    assert base.shape == (3, 1001) and base.dtype == torch.float32
    # A row is the same wherever it stands in the batch, and at any vocabulary size.
    assert torch.equal(noise(5, [big, 1], [11, 10]), base[[2, 0]])
    assert torch.equal(noise(5, [2], [10], vocab=64)[0], base[1, :64])
    for other in (noise(6, [1], [10]), noise(5, [1], [11]), noise(5, [1 + (1 << 32)], [10]), noise(5, [2], [10])):
        assert (other[0] != base[0]).float().mean() > 0.99
    # E ~ Exp(1): mean 1, variance 1, P(E > 1) = 1/e.
    e = noise(0, list(range(64)), [0] * 64, vocab=20000).double().exp()
    assert e.mean().item() == pytest.approx(1.0, abs=0.01)
    assert e.var().item() == pytest.approx(1.0, abs=0.02)
    assert (e > 1).double().mean().item() == pytest.approx(0.3679, abs=0.005)


def test_cpu_sampler_draws_from_the_truncated_softmax_with_keyed_noise():
    from serving.nano_vllm.sampling import Sampler, SamplingOptions

    row = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -3.0, -3.0, -3.0])
    rows = 20000
    option = SamplingOptions(temperature=0.8, top_p=0.9)

    def draw(seed, seq_ids, positions):
        sampler = Sampler(max_rows=len(seq_ids), device="cpu", seed=seed)
        out = torch.zeros(len(seq_ids), dtype=torch.long)
        sampler.sample(row.expand(len(seq_ids), -1), [option] * len(seq_ids), seq_ids, positions, out)
        return out

    tokens = draw(3, [9] * rows, list(range(rows)))  # one sequence, a draw per position
    keep = _keep(row.view(1, -1).double(), 0.8, 0.9)[0]
    probs = torch.softmax((row.double() / 0.8).masked_fill(~keep, float("-inf")), dim=0)
    freq = torch.bincount(tokens, minlength=8).double() / rows
    assert freq[~keep].sum() == 0
    assert torch.allclose(freq, probs, atol=0.015)
    # The same (seed, sequence, position) draws the same token in any batch.
    assert torch.equal(draw(3, [9] * rows, list(range(rows))), tokens)
    assert draw(3, [4, 9, 4], [0, 77, 1])[1] == tokens[77]
    assert (draw(4, [9] * rows, list(range(rows))) != tokens).float().mean() > 0.3
    assert (draw(3, [8] * rows, list(range(rows))) != tokens).float().mean() > 0.3


def test_sampler_temperature_scale_lowers_sampled_rows_and_leaves_greedy_rows():
    from serving.nano_vllm.sampling import Sampler, SamplingOptions

    sampler = Sampler(max_rows=3, device="cpu", seed=0, temperature_scale=0.5)
    options = [SamplingOptions(temperature=0.8), SamplingOptions(temperature=0.0), SamplingOptions(temperature=1.2)]
    assert sampler.stage(options, [1, 2, 3], [0, 0, 0], vocab=8) == (True, True, False)
    staged = sampler.params[Sampler.TEMPERATURE].view(torch.float32).tolist()
    assert staged == pytest.approx([0.4, 0.0, 0.6])
