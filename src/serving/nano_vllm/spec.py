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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    ``b``'s block [anchor, K drafts].
    """

    seqs: list[Sequence]
    anchor_ids: torch.Tensor  # [B] the last accepted token of each sequence
    positions: torch.Tensor  # [B * Q] the block tokens' positions
    slot_mapping: torch.Tensor  # [B * Q] their KV slots (the lookahead slots)
    block_tables: torch.Tensor  # [B, width]
    context_lens: torch.Tensor  # [B] tokens before the anchor


class Proposer(Protocol):
    """A source of draft tokens for the engine's speculative step."""

    # K: drafts per cycle.
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
        """Draft tokens [B, K] (long, on the device) for ``batch``."""


class DFlashProposer:
    """A DFlash drafter with its KV pool, drafting greedily.

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
        return logits.argmax(dim=-1).view(len(batch.seqs), self.num_draft_tokens)
