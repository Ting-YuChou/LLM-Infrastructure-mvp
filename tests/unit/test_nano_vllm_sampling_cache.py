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
