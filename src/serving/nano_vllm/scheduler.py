"""Continuous batching scheduler for nano-vLLM."""

from __future__ import annotations

from contextlib import suppress
from collections import deque
from typing import Optional

from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus


class Scheduler:
    """Prefill-first scheduler with decode batching and simple preemption."""

    def __init__(
        self,
        block_manager: BlockManager,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> None:
        self.block_manager = block_manager
        self.max_num_seqs = max(1, max_num_seqs)
        self.max_num_batched_tokens = max(1, max_num_batched_tokens)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        # A sequence that can never fit would wait (or be preempted) forever.
        max_tokens = seq.num_prompt_tokens + seq.sampling.max_tokens
        capacity = self.block_manager.num_blocks * self.block_manager.block_size
        if max_tokens > capacity:
            raise ValueError(
                f"Request needs up to {max_tokens} KV slots (prompt + max_tokens) "
                f"but the KV cache holds only {capacity}"
            )
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.waiting.append(seq)

    def abort(self, seq: Sequence, finish_reason: str = "abort") -> None:
        """Drop ``seq`` from the queues and free its KV blocks."""
        seq.status = SequenceStatus.FINISHED
        if seq.finish_reason is None:
            seq.finish_reason = finish_reason
        self.block_manager.deallocate(seq)
        with suppress(ValueError):
            self.running.remove(seq)
        with suppress(ValueError):
            self.waiting.remove(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        prefill_batch = self._schedule_prefill()
        if prefill_batch:
            return prefill_batch, True
        return self._schedule_decode(), False

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> list[tuple[Sequence, int]]:
        """Apply a finished model step; return the tokens actually appended.

        A prefill chunk that leaves prompt tokens unprocessed samples a logit
        that is meaningless, so it appends (and returns) nothing.
        """
        emitted: list[tuple[Sequence, int]] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.is_finished:
                # Aborted while its batch was running: release it now.
                self.abort(seq)
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)
            emitted.append((seq, token_id))
            if seq.should_finish(token_id):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                with suppress(ValueError):
                    self.running.remove(seq)
        return emitted

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def unschedulable_head(self) -> Optional[Sequence]:
        """Return the head waiting sequence if nothing can ever free room for it.

        Only meaningful right after schedule() returned nothing: with no running
        sequences and no partially prefilled ones holding blocks, the free pool
        cannot grow, so the head would otherwise wait forever.
        """
        if self.running or not self.waiting:
            return None
        if any(seq.block_table for seq in self.waiting):
            return None
        return self.waiting[0]

    def _num_active(self) -> int:
        # Running sequences plus admitted ones still mid chunked-prefill.
        return len(self.running) + sum(1 for seq in self.waiting if seq.block_table)

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break

            if not seq.block_table:
                if self._num_active() >= self.max_num_seqs:
                    break
                cached_blocks = self.block_manager.can_allocate(seq)
                if cached_blocks == -1:
                    break
                num_tokens = (
                    seq.num_tokens - cached_blocks * self.block_manager.block_size
                )
            else:
                cached_blocks = 0
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if remaining < num_tokens and scheduled:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled.append(seq)
        return scheduled

    def _schedule_decode(self) -> list[Sequence]:
        """Round-robin decode: scheduled sequences rotate to the back.

        When the batch is capped (max_num_seqs or max_num_batched_tokens), the
        sequences skipped this step sit at the front and go first next step.
        """
        scheduled: list[Sequence] = []
        budget = min(self.max_num_seqs, self.max_num_batched_tokens)
        while self.running and len(scheduled) < budget:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # Preempt the youngest request (vLLM's recompute policy); it may
                # be ``seq`` itself. Already scheduled ones are off-limits.
                victim = max([seq, *self.running], key=lambda item: item.seq_id)
                if victim is seq:
                    self.preempt(seq)
                    seq = None
                    break
                self.running.remove(victim)
                self.preempt(victim)
            if seq is None:
                continue
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled.append(seq)

        self.running.extend(scheduled)
        return scheduled
