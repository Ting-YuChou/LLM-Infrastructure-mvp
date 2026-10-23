"""Device-side description of one scheduler batch for the paged forward pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BatchMetadata:
    """Everything a paged forward pass needs besides input_ids and positions.

    Tokens of all sequences are packed back to back (varlen, no padding). The
    tensors live on the model's device, so kernels read lengths and block ids
    from there instead of from Python scalars; that keeps the decode step free
    of host syncs and capturable in a CUDA graph. The host lists are kept for
    the PyTorch prefill path, which loops over sequences.
    """

    slot_mapping: torch.Tensor  # [num_tokens] int64, physical KV slot per token
    block_tables: torch.Tensor  # [num_seqs, width] int32, padded with the null block
    context_lens: torch.Tensor  # [num_seqs] int32, KV length including this step
    cu_seqlens_q: torch.Tensor  # [num_seqs + 1] int32, query offsets in the batch
    query_lens: list[int]
    context_lens_host: list[int]
    is_prefill: bool
    # Rows that get logits, normally the last token of each sequence. None
    # means every row (the per-position checks against transformers use it).
    logits_indices: Optional[torch.Tensor] = None
    # A mixed batch puts its decode rows (one query token each) first; the
    # prefill rows after them get query offsets of their own, starting at 0.
    prefill_cu_seqlens_q: Optional[torch.Tensor] = None
    # Every sequence has exactly one query token, so the batched decode
    # attention applies (also true for a prefill chunk of one token).
    decode_only: bool = field(init=False)
    # Leading rows with one query token: the decode kernel takes them.
    num_decode_rows: int = field(init=False)

    def __post_init__(self) -> None:
        self.decode_only = all(query_len == 1 for query_len in self.query_lens)
        self.num_decode_rows = next(
            (i for i, query_len in enumerate(self.query_lens) if query_len != 1),
            len(self.query_lens),
        )

    @property
    def num_seqs(self) -> int:
        return len(self.query_lens)


def build_batch(
    input_ids: list[int],
    positions: list[int],
    slots: list[int],
    block_tables: list[list[int]],
    context_lens: list[int],
    query_lens: list[int],
    is_prefill: bool,
    null_block: int,
    num_slots: int,
    device: torch.device,
    all_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata]:
    """Pack host-side batch lists into device tensors.

    All int64 fields go in one buffer and all int32 fields in another, staged
    in pinned memory on CUDA, so a batch costs two asynchronous H2D copies.
    """
    num_tokens = len(input_ids)
    num_seqs = len(query_lens)
    if not (len(positions) == len(slots) == num_tokens):
        raise ValueError("input_ids, positions and slots must have the same length")
    if not (len(block_tables) == len(context_lens) == num_seqs):
        raise ValueError("block_tables, context_lens and query_lens must match")
    if sum(query_lens) != num_tokens:
        raise ValueError("query_lens must sum to the flat token count")
    if slots and (min(slots) < 0 or max(slots) >= num_slots):
        raise ValueError("slot_mapping references a slot outside the KV cache")

    cu_seqlens = [0]
    for query_len in query_lens:
        cu_seqlens.append(cu_seqlens[-1] + query_len)
    logits_indices = [] if all_logits else [end - 1 for end in cu_seqlens[1:]]
    num_decode_rows = next((i for i, n in enumerate(query_lens) if n != 1), num_seqs)
    # Prefill rows' own offsets when decode rows come first (a mixed batch).
    prefill_cu = (
        [c - cu_seqlens[num_decode_rows] for c in cu_seqlens[num_decode_rows:]]
        if 0 < num_decode_rows < num_seqs
        else []
    )
    width = max([len(table) for table in block_tables] + [1])
    flat_tables: list[int] = []
    for table in block_tables:
        flat_tables.extend(table)
        flat_tables.extend([null_block] * (width - len(table)))

    pin = device.type == "cuda"
    longs = torch.tensor(
        input_ids + positions + slots + logits_indices,
        dtype=torch.long,
        pin_memory=pin,
    ).to(device, non_blocking=True)
    ints = torch.tensor(
        cu_seqlens + context_lens + flat_tables + prefill_cu,
        dtype=torch.int32,
        pin_memory=pin,
    ).to(device, non_blocking=True)

    input_ids_t, positions_t, slots_t, logits_t = longs.split(
        [num_tokens, num_tokens, num_tokens, len(logits_indices)]
    )
    cu_seqlens_t, context_lens_t, tables_t, prefill_cu_t = ints.split(
        [num_seqs + 1, num_seqs, num_seqs * width, len(prefill_cu)]
    )
    metadata = BatchMetadata(
        slot_mapping=slots_t,
        block_tables=tables_t.view(num_seqs, width),
        context_lens=context_lens_t,
        cu_seqlens_q=cu_seqlens_t,
        query_lens=list(query_lens),
        context_lens_host=list(context_lens),
        is_prefill=is_prefill,
        logits_indices=None if all_logits else logits_t,
        prefill_cu_seqlens_q=prefill_cu_t if prefill_cu else None,
    )
    return input_ids_t, positions_t, metadata
