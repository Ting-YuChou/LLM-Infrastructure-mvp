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
