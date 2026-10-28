"""Sampling helpers for token-by-token decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingOptions:
    # None means "engine default" (NanoVLLMConfig.max_new_tokens, capped to fit
    # max_model_len). An explicit value is never silently clamped.
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    stop: Optional[list[str]] = None
    # Extra token ids that end generation (vLLM extension); always honored,
    # unlike the model's EOS ids which ignore_eos can disable.
    stop_token_ids: Optional[list[int]] = None
    # Keep generating past the model's EOS ids (vLLM extension; benchmarks use
    # it to get exactly max_tokens per request).
    ignore_eos: bool = False


def keep_mask(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    """The tokens each row may sample after top-k then top-p, [rows, vocab] bool.

    ``temperature`` [rows] > 0, ``top_p`` [rows] (>= 1: off), ``top_k``
    [rows] (<= 0 or >= vocab: off); computed in ``logits.dtype``, NaN-free.
    Top-k keeps a token when fewer than k tokens are larger; top-p then keeps
    one when the probability mass of the larger tokens it left is below
    top_p times the mass of all it left. Tokens equal to a cutoff stay with
    it, so the set depends on the values only, never on a sort order: the
    CUDA kernels (kernels/csrc/sampler.cu) keep the same set.
    """
    vocab = logits.shape[-1]
    scaled = logits / temperature.to(logits.dtype).view(-1, 1)
    values = scaled.sort(dim=-1, descending=True).values
    k = torch.where((top_k > 0) & (top_k < vocab), top_k, vocab).long().view(-1, 1)
    kth = values.gather(1, k - 1)
    top = values[:, :1]
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    mass = torch.exp(values - top).masked_fill(values < kth, 0.0)
    total = mass.cumsum(dim=-1)
    before = F.pad(total[:, :-1], (1, 0))  # mass of the tokens sorted before each one
    p = top_p.to(logits.dtype).view(-1, 1)
    inside = (before < p * total[:, -1:]) | (p >= 1)
    count = inside.sum(dim=-1, keepdim=True).clamp(min=1)
    cutoff = values.gather(1, count - 1)
    return (scaled >= kth) & (scaled >= cutoff)


def filter_logits(
    logits: torch.Tensor,
    top_p: float = 1.0,
    top_k: int = -1,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to a 1D logits tensor."""
    if logits.ndim != 1:
        raise ValueError("filter_logits expects a 1D tensor")
    keep = keep_mask(
        logits.float().view(1, -1),
        torch.ones(1, device=logits.device),
        torch.tensor([top_p if top_p is not None and 0.0 < top_p < 1.0 else 1.0], device=logits.device),
        torch.tensor([top_k if top_k is not None else -1], device=logits.device),
    )[0]
    return logits.masked_fill(~keep, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = -1,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample one token id from the final-position logits."""
    if logits.ndim == 2:
        logits = logits[-1]
    if logits.ndim != 1:
        raise ValueError("sample_next_token expects a 1D or [seq, vocab] tensor")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    filtered = filter_logits(logits / temperature, top_p=top_p, top_k=top_k)
    probs = torch.softmax(filtered.float(), dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def encode_options(options: list[SamplingOptions], vocab: int) -> tuple[list, list, list]:
    """Per-row temperature (<= 0: greedy), top_p (1.0: off) and top_k (0: off)."""
    temperature = [option.temperature for option in options]
    top_p = [
        option.top_p if option.top_p is not None and 0.0 < option.top_p < 1.0 else 1.0
        for option in options
    ]
    top_k = [
        option.top_k if option.top_k is not None and 0 < option.top_k < vocab else 0
        for option in options
    ]
    return temperature, top_p, top_k


def sample_rows(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    """One token per row of [rows, vocab] logits, on their device, no host sync.

    The argmax where temperature <= 0 (or the row has no finite maximum);
    otherwise a draw among the tokens keep_mask leaves by the exponential
    race: the argmax of v - log(E) with v = logits / temperature and E ~
    Exp(1), which picks token i with probability softmax(v)[i]. NaN logits
    count as -inf.
    """
    inf = float("inf")
    logits = torch.nan_to_num(logits.float(), nan=-inf, posinf=inf, neginf=-inf)
    greedy_tokens = logits.argmax(dim=-1)
    greedy = (temperature <= 0) | ~torch.isfinite(logits.max(dim=-1).values)
    temperature = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
    keep = keep_mask(logits, temperature, top_p, top_k)
    noise = torch.empty_like(logits).exponential_().clamp_min_(torch.finfo(torch.float32).tiny).log_()
    scores = (logits / temperature.view(-1, 1)).masked_fill(~keep, -inf) - noise
    return torch.where(greedy, greedy_tokens, scores.argmax(dim=-1))


def sample_tokens(
    logits: torch.Tensor,
    options: list[SamplingOptions],
) -> list[int]:
    """Sample one token per row of [num_seqs, vocab] logits, with one host sync."""
    return sample_tokens_device(logits, options).tolist()


def sample_tokens_device(
    logits: torch.Tensor,
    options: list[SamplingOptions],
) -> torch.Tensor:
    """Sample one token per row of [num_seqs, vocab] logits, on the device.

    :func:`sample_rows` with the settings uploaded through pinned memory, so
    the result can stay on the GPU while the host schedules the next step.
    """
    if logits.ndim != 2 or logits.shape[0] != len(options):
        raise ValueError("sample_tokens expects [num_seqs, vocab] logits, one row per option")
    if all(option.temperature <= 0 for option in options):
        return logits.argmax(dim=-1)
    device = logits.device
    pin = device.type == "cuda"

    def upload(values, dtype):
        return torch.tensor(values, dtype=dtype, pin_memory=pin).to(device, non_blocking=True)

    temperature, top_p, top_k = encode_options(options, logits.shape[-1])
    return sample_rows(
        logits,
        upload(temperature, torch.float32),
        upload(top_p, torch.float32),
        upload(top_k, torch.int32),
    )


class Sampler:
    """Samples one token per row into a device buffer, with no host sync.

    The rows' settings go up in one copy from pinned staging, two slots used
    in turn like the decode graphs' inputs: the caller passes the slot and
    reuses it only once the step that used it was collected. On CUDA the
    sort-free kernels in kernels/csrc/sampler.cu draw the tokens; elsewhere,
    or when they cannot be built, :func:`sample_rows` does.

    A draw's randomness comes from (seed, sequence id, position of the token
    in its sequence), so a request's tokens do not depend on the batch it
    runs in (up to the logits, which the batch size can change slightly).
    """

    # Rows of ``params``: temperature and top_p as float32 bits, top_k (0:
    # off), the token's position, the sequence id's low and high words.
    TEMPERATURE, TOP_P, TOP_K, POSITION, SEQ_LO, SEQ_HI = range(6)

    def __init__(self, max_rows: int, device: torch.device, seed: int = 0):
        self.device = torch.device(device)
        self.seed = seed
        cuda = self.device.type == "cuda"
        self.host = torch.zeros(2, 6, max_rows, dtype=torch.int32, pin_memory=cuda)
        self.params = torch.zeros(6, max_rows, dtype=torch.int32, device=self.device)
        self._cuda = None
        if cuda:
            from .kernels.sampler import sample_cuda, sampler_extension, sampler_workspace

            if sampler_extension() is not None:
                sampler_workspace(self.device, max_rows)  # before any CUDA graph capture
                self._cuda = sample_cuda

    def sample(
        self,
        logits: torch.Tensor,
        options: list[SamplingOptions],
        seq_ids: list[int],
        positions: list[int],
        out: torch.Tensor,
        slot: int = 0,
    ) -> None:
        """Write one token per row of [rows, vocab] logits into ``out[:rows]``."""
        rows, vocab = logits.shape
        if rows != len(options):
            raise ValueError("sample expects one row of logits per option")
        temperature, top_p, top_k = encode_options(options, vocab)
        host = self.host[slot].numpy()
        floats = host[:2].view(np.float32)
        floats[self.TEMPERATURE, :rows] = temperature
        floats[self.TOP_P, :rows] = top_p
        host[self.TOP_K, :rows] = top_k
        host[self.POSITION, :rows] = positions
        ids = np.asarray(seq_ids, dtype=np.int64)
        host[self.SEQ_LO, :rows] = (ids & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
        host[self.SEQ_HI, :rows] = (ids >> 32).astype(np.int32)
        self.params.copy_(self.host[slot], non_blocking=True)

        sampled = [t > 0 for t in temperature]
        filtered = [s and (p < 1.0 or k > 0) for s, p, k in zip(sampled, top_p, top_k)]
        both = [s and p < 1.0 and k > 0 for s, p, k in zip(sampled, top_p, top_k)]
        if self._cuda is not None and self._cuda(
            logits, self.params, out, self.seed,
            any_sampled=any(sampled), any_filter=any(filtered), any_both=any(both),
        ):
            return
        if not any(sampled):
            out[:rows].copy_(logits.argmax(dim=-1))
            return
        params = self.params[:, :rows]
        out[:rows].copy_(
            sample_rows(
                logits,
                params[self.TEMPERATURE].view(torch.float32),
                params[self.TOP_P].view(torch.float32),
                params[self.TOP_K],
            )
        )
