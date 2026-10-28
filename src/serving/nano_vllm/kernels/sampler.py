"""Sort-free sampling kernels (csrc/sampler.cu): one token per row of 16-bit logits.

K1 finds each row's max and argmax (greedy rows end there); for the others,
256-bin histograms over the high and then the low byte of the logits' 16-bit
keys locate the top-k / top-p cutoff exactly (K2, K3, and K4 for rows with
both filters), and the tokens at or above it race with v - log(E), E ~ Exp(1)
from Philox. The chunks of a row meet in a per-row workspace that each
kernel's last block re-zeroes, so one workspace per device serves every call.
sampling.keep_mask is the same cutoff rule in PyTorch.
"""

from __future__ import annotations

import torch

from .cuda_ext import load_cuda_extension

# Words of a row's state at the start of its workspace row (csrc/sampler.cu, kState).
(
    STATE_ARGMAX,
    STATE_MAX,
    STATE_FLAGS,
    STATE_BIN,
    STATE_MASS_ABOVE,
    STATE_THRESHOLD,
    STATE_COUNT_ABOVE,
    STATE_CUT_KEY,
) = range(8)
MAX_VOCAB = 1 << 18  # the u64 fixed-point mass of a row must fit (csrc/sampler.cu, kMaxVocab)

_WORKSPACE: dict[torch.device, torch.Tensor] = {}
_RETIRED: list[torch.Tensor] = []


def sampler_extension():
    """The built csrc/sampler.cu, or None when it cannot be built here."""
    return load_cuda_extension("sampler")


def sampler_workspace(device: torch.device, rows: int) -> torch.Tensor:
    """Zeroed int64 [rows, WORDS_PER_ROW] that the kernels leave zeroed.

    Like w4a16_workspace: a buffer that grows is retired, not freed (CUDA
    graphs keep raw pointers), and growing during a capture is refused, so
    reserve the most rows first (sampling.Sampler does).
    """
    device = torch.device(device)
    current = _WORKSPACE.get(device)
    if current is not None and current.shape[0] >= rows:
        return current
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("sampler_workspace must be reserved before capturing a CUDA graph")
    if current is not None:
        _RETIRED.append(current)
    words = sampler_extension().WORDS_PER_ROW
    workspace = torch.zeros(rows, words, dtype=torch.int64, device=device)
    _WORKSPACE[device] = workspace
    return workspace


def sampler_state(device: torch.device) -> torch.Tensor:
    """Each row's state after the last call, [rows, 8] int64 (STATE_*)."""
    return _WORKSPACE[torch.device(device)][:, :8]


def sampler_supported(logits: torch.Tensor) -> bool:
    return (
        logits.is_cuda
        and logits.dtype in (torch.bfloat16, torch.float16)
        and logits.dim() == 2
        and logits.is_contiguous()
        and logits.shape[-1] % 8 == 0
        and logits.shape[-1] <= MAX_VOCAB
        and logits.data_ptr() % 16 == 0
    )


def sample_cuda(
    logits: torch.Tensor,
    params: torch.Tensor,
    out: torch.Tensor,
    seed: int,
    *,
    any_sampled: bool,
    any_filter: bool,
    any_both: bool,
) -> bool:
    """Write one token per row into ``out[:rows]``; False (nothing written)
    when the kernels cannot take these logits or cannot be built.

    ``params`` is int32 [6, >= rows], the rows of sampling.Sampler; the flags
    say whether any row samples (temperature > 0), has top-k or top-p, and
    has both, which decide the kernels launched.
    """
    if not sampler_supported(logits):
        return False
    extension = sampler_extension()
    if extension is None:
        return False
    workspace = sampler_workspace(logits.device, logits.shape[0])
    extension.sample(logits, params, out, workspace, seed, any_sampled, any_filter, any_both)
    return True
