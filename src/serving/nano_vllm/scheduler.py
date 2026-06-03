"""Continuous batching scheduler for nano-vLLM."""

from __future__ import annotations

from contextlib import suppress
from collections import deque

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
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        prefill_batch = self._schedule_prefill()
        if prefill_batch:
            return prefill_batch, True
        return self._schedule_decode(), False

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> None:
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)
            if seq.should_finish(token_id):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                with suppress(ValueError):
                    self.running.remove(seq)

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break

            if not seq.block_table:
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
        scheduled: list[Sequence] = []
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                    self.running.appendleft(seq)
                    continue
                self.preempt(seq)
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled.append(seq)

        if scheduled:
            self.running.extendleft(reversed(scheduled))
        return scheduled
