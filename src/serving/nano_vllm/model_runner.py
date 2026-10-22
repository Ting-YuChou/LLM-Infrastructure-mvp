"""Turn a scheduled batch into one forward pass and sampled tokens."""

from __future__ import annotations

import torch

from .block_manager import BlockManager
from .kv_cache import PagedKVCache
from .metadata import BatchMetadata, build_batch
from .sampling import sample_tokens
from .sequence import Sequence


class ModelRunner:
    """Owns the step from scheduler output to sampled token ids.

    The model only needs ``device`` and ``forward_batch(input_ids, positions,
    cache, metadata)`` returning one row of logits per sequence.
    """

    def __init__(self, model, cache: PagedKVCache, block_manager: BlockManager):
        self.model = model
        self.cache = cache
        self.block_manager = block_manager

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions, metadata = self.prepare(seqs, is_prefill)
        logits = self.model.forward_batch(input_ids, positions, self.cache, metadata)
        return sample_tokens(logits, [seq.sampling for seq in seqs])

    def prepare(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata]:
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        for seq in seqs:
            if is_prefill:
                # Uncached prompt tokens (or this chunk of them).
                start = seq.num_cached_tokens
                end = start + seq.num_scheduled_tokens
                input_ids.extend(seq[start:end])
            else:
                # The last sampled token, whose KV is not cached yet.
                start, end = len(seq) - 1, len(seq)
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            slots.extend(
                self.block_manager.slot_for_position(seq, pos) for pos in range(start, end)
            )
            context_lens.append(end)
            query_lens.append(end - start)
        return build_batch(
            input_ids=input_ids,
            positions=positions,
            slots=slots,
            block_tables=[seq.block_table for seq in seqs],
            context_lens=context_lens,
            query_lens=query_lens,
            is_prefill=is_prefill,
            null_block=self.cache.null_block,
            num_slots=self.cache.total_slots,
            device=self.model.device,
        )
