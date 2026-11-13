"""Speculative decoding: what a cycle needs besides the two models.

A cycle drafts K tokens after the last accepted token (the anchor) of each
sequence, runs the target once over [anchor, draft 1..K] and takes the
target's own token at each of those Q = K + 1 positions. Draft ``j`` is
accepted while it equals the target's token for the same position; the
target's token after the last accepted draft comes for free (the bonus), so
a cycle yields 1 to Q tokens, and they are exactly the tokens the target
would have produced one at a time.

A :class:`Proposer` supplies the drafts. It may read the target through
feature taps (:meth:`Proposer.observe`), as the DFlash drafter does.

The target's token at a position is drawn with noise that depends only on
(seed, sequence id, position), so it is the token a plain decode step would
draw there, at any temperature. Coupled sampling draws the drafts with the
same noise from the drafter's logits: where the two distributions are close
the draws agree, which a draft drawn independently would only do by chance.

A step need not run the whole block: its verify length Q (the anchor and
Q - 1 drafts) is :class:`SpecPolicy`'s choice, from 1 (a plain decode step)
to the proposer's K + 1. The tokens are the same for any choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from .dflash import DFlashDrafter
from .kv_cache import PagedKVCache
from .sequence import Sequence


def accept_prefix(drafts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Accepted drafts per sequence, [B] in 0..K.

    ``drafts`` [B, K]: the proposed tokens for the K positions after the
    anchor. ``target`` [B, Q >= K]: the target's token for the same positions
    (row ``j`` is its choice after [anchor, draft 1..j]). A draft is accepted
    while it and every draft before it match; the cycle's tokens are then
    ``target[b, : accepted[b] + 1]``.
    """
    matches = drafts == target[:, : drafts.shape[1]]
    return matches.long().cumprod(dim=1).sum(dim=1)


@dataclass
class DraftBatch:
    """The sequences to draft for, as the verify pass will address them.

    Rows are per block token: row ``b * Q + t`` is token ``t`` of sequence
    ``b``'s block [anchor, K drafts]. K is this step's
    (:attr:`num_draft_tokens`), at most the proposer's.
    """

    seqs: list[Sequence]
    anchor_ids: torch.Tensor  # [B] the last accepted token of each sequence
    positions: torch.Tensor  # [B * Q] the block tokens' positions
    slot_mapping: torch.Tensor  # [B * Q] their KV slots (the lookahead slots)
    block_tables: torch.Tensor  # [B, width]
    context_lens: torch.Tensor  # [B] tokens before the anchor
    # Draft logits [B * K, vocab] (row ``b * K + j``: the token ``j + 1``
    # positions after sequence ``b``'s anchor) -> a token per row [B * K]:
    # the engine's sampler with each sequence's settings and the noise of
    # the position the row proposes, the one the target's own draw there
    # will use (coupled sampling). None: the argmax.
    draw: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    @property
    def num_draft_tokens(self) -> int:
        """K: the drafts this step asks for per sequence."""
        return self.positions.numel() // len(self.seqs) - 1


class Proposer(Protocol):
    """A source of draft tokens for the engine's speculative step."""

    # K: the most drafts a cycle can ask for (``DraftBatch.num_draft_tokens``
    # says how many a step does).
    num_draft_tokens: int
    # Target layers whose output residual :meth:`observe` gets; () for none.
    tap_layers: tuple[int, ...]
    # True when propose and observe read only the tensors they are given
    # (never ``DraftBatch.seqs`` or the host): the model runner then captures
    # the whole cycle in a CUDA graph. Optional; absent means False.
    device_only: bool

    def observe(self, taps: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Every token the target just computed (prompt chunks and verify
        rows alike): its taps [tokens, len(tap_layers) * hidden], position
        and KV slot. Verify rows past the accepted ones are stale; a later
        step computes those positions again."""

    def propose(self, batch: DraftBatch) -> torch.Tensor:
        """Draft tokens [B, batch.num_draft_tokens] (long, on the device)."""


class DFlashProposer:
    """A DFlash drafter with its KV pool: a draft is the argmax of its
    logits, or what ``DraftBatch.draw`` makes of them.

    The pool shares block ids with the target's cache, so the target's slot
    mapping and block tables address it and a prefix-cache hit covers both.
    A draft writes the block's own K/V into the lookahead slots; the verify
    pass's :meth:`observe` then overwrites them with context K/V.
    """

    device_only = True

    def __init__(self, drafter: DFlashDrafter, cache: PagedKVCache, num_draft_tokens: int = 0):
        full = drafter.config.num_draft_tokens
        if not 0 <= num_draft_tokens <= full:
            raise ValueError(f"num_draft_tokens must be 0 (the full block) to {full}")
        self.drafter = drafter
        self.cache = cache
        self.num_draft_tokens = num_draft_tokens or full
        self.tap_layers = tuple(drafter.config.target_layer_ids)

    def observe(self, taps: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        self.drafter.write_context(taps, positions, self.cache, slot_mapping)

    def propose(self, batch: DraftBatch) -> torch.Tensor:
        logits = self.drafter.draft(
            batch.anchor_ids, batch.positions, self.cache, batch.block_tables,
            batch.context_lens, batch.slot_mapping,
        )
        tokens = logits.argmax(dim=-1) if batch.draw is None else batch.draw(logits)
        return tokens.view(len(batch.seqs), batch.num_draft_tokens)


class SpecPolicy:
    """Chooses each step's verify length Q: the tokens a decode row runs,
    its anchor and Q - 1 drafts. 1 is a plain decode step.

    A cycle of length Q yields ``1 + s_1 + ... + s_{Q-1}`` tokens per
    sequence on average, ``s_j`` being the share of cycles whose first ``j``
    drafts are all accepted, and takes ``cost(batch, Q)`` milliseconds. The
    policy picks the Q with the most tokens per millisecond.

    ``s_j`` is the product of the acceptance rates of drafts 1..j, each the
    share of the cycles that reached its depth (every draft before it
    accepted) in which it was accepted too: running averages over the
    cycles the engine completes (:meth:`update`). A draft's rate does not
    depend on how many drafts follow it (measured with the DFlash drafter:
    a block of 4 or 6 yields what the first 3 or 5 drafts of a block of 10
    do), so one set of rates serves every Q. Depths past the Q in use get no
    cycles; one step in ``probe_every`` runs the next longer length to keep
    their rates current.

    ``cost`` starts from ``times``: a step's milliseconds per (sequences, Q),
    measured at startup by replaying each CUDA graph on an empty context
    (ModelRunner.profile_graphs). What a real step takes on top (reading the
    sequences' KV, the host's share) is measured as the engine runs
    (:meth:`record`), per graph size, apart for plain steps and cycles: it
    does not grow with Q (the block kernel reads a sequence's KV once for
    all its verify rows), which is what makes long contexts favor longer
    cycles. Without ``times`` (no graphs: the CPU, enforce_eager) every step
    runs the longest length.
    """

    # A draft's acceptance rate before any cycle has run it.
    PRIOR_RATE = 0.75
    # A cycle's measured extra time over a plain step's at the same size,
    # used for the kind that has not run yet: it also reads the drafter's
    # KV, and the block kernel is up to 1.3x the decode kernel.
    CYCLE_EXTRA = 1.4

    def __init__(
        self,
        verify_lens: tuple[int, ...],
        times: Optional[dict[tuple[int, int], float]] = None,
        rate: float = 1 / 64,
        probe_every: int = 64,
    ):
        if not verify_lens or min(verify_lens) < 1:
            raise ValueError("verify lengths must be positive")
        self.verify_lens = tuple(sorted(set(verify_lens)))
        self.rates = [self.PRIOR_RATE] * (self.verify_lens[-1] - 1)
        self.rate = rate
        self.probe_every = probe_every
        self.times: dict[tuple[int, int], float] = {}
        self.extra: dict[tuple[int, bool], float] = {}
        self._sizes: dict[int, list[int]] = {}
        self._choices = 0
        if times:
            self.set_times(times)

    def set_times(self, times: dict[tuple[int, int], float]) -> None:
        """``times[(sequences, Q)]``: milliseconds of one step of a graph."""
        self.times = dict(times)
        self.extra = {}
        self._sizes = {
            length: sorted(size for size, q in self.times if q == length)
            for length in self.verify_lens
        }

    def expected_tokens(self, length: int) -> float:
        """Tokens per sequence of a cycle of ``length``."""
        total = survive = 1.0
        for rate in self.rates[: length - 1]:
            survive *= rate
            total += survive
        return total

    def bucket(self, batch: int, length: int) -> Optional[int]:
        """The profiled size a ``batch`` of ``length`` is costed at (the
        largest, when the batch is past them all)."""
        sizes = self._sizes.get(length)
        if not sizes:
            return None
        return next((size for size in sizes if size >= batch), sizes[-1])

    def cost(self, batch: int, length: int) -> float:
        """Milliseconds of a step of ``batch`` sequences at ``length``."""
        bucket = self.bucket(batch, length)
        base = self.times[(bucket, length)]
        cycle = length > 1
        extra = self.extra.get((bucket, cycle))
        if extra is None:
            other = self.extra.get((self.bucket(batch, 1 if cycle else self.verify_lens[-1]), not cycle))
            extra = 0.0 if other is None else other * (self.CYCLE_EXTRA if cycle else 1 / self.CYCLE_EXTRA)
        # Past the largest graph the step runs eagerly, in proportion.
        return max(base + extra, 0.1 * base) * max(1.0, batch / bucket)

    def choose(self, batch: int) -> int:
        """The verify length for a step with ``batch`` decode rows."""
        lens = [length for length in self.verify_lens if self._sizes.get(length)]
        if not lens or batch <= 0:
            return self.verify_lens[-1]
        best = max(lens, key=lambda length: self.expected_tokens(length) / self.cost(batch, length))
        self._choices += 1
        if self.probe_every and self._choices % self.probe_every == 0:
            best = next((length for length in lens if length > best), best)
        return best

    def update(self, drafted: int, accepted: list[int]) -> None:
        """A step's cycles: ``drafted`` drafts each, of which ``accepted[i]``
        (leading ones) were accepted for sequence ``i``."""
        if not accepted or drafted <= 0:
            return
        reached = [0] * (drafted + 1)  # reached[d]: cycles that accepted at least d drafts
        for count in accepted:
            reached[min(count, drafted)] += 1
        for depth in range(drafted - 1, -1, -1):
            reached[depth] += reached[depth + 1]
        for depth in range(drafted):
            tried = reached[depth]
            if not tried:
                break
            weight = 1.0 - math.pow(1.0 - self.rate, tried)
            self.rates[depth] += weight * (reached[depth + 1] / tried - self.rates[depth])

    def record(self, bucket: int, length: int, milliseconds: float, rate: float = 0.1) -> None:
        """A step of graph size ``bucket`` at ``length`` (no prompt rows next
        to it) took ``milliseconds``, launch to collected tokens."""
        base = self.times.get((bucket, length))
        if base is None:
            return
        key = (bucket, length > 1)
        previous = self.extra.get(key)
        # One stalled step (the host was busy elsewhere) is not the step's
        # cost: a sample counts for at most three times the current estimate.
        extra = min(milliseconds - base, 3.0 * (base + max(previous or 0.0, 0.0)))
        self.extra[key] = extra if previous is None else previous + rate * (extra - previous)
