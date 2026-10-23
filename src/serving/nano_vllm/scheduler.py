"""Continuous batching scheduler for nano-vLLM."""

from __future__ import annotations

from contextlib import suppress
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus


SCHEDULING_POLICIES = ("prefill_first", "mixed")


@dataclass
class Row:
    """One sequence's part of a launched step."""

    seq: Sequence
    samples: bool  # the step samples a token for it (decode, or the last prompt chunk)
    num_computed_tokens: int  # KV computed once the step finishes
    epoch: int  # the block table it wrote into (see Sequence.epoch)


class Scheduler:
    """Continuous batching with chunked prefill and recompute preemption.

    ``prefill_first``: a step is either all prefill chunks or all decodes, so
    every prefill stalls the running decodes. ``mixed`` (chunked prefill as in
    vLLM v1 and Sarathi): every step decodes all running sequences first and
    fills the rest of the max_num_batched_tokens budget with prefill chunks.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        policy: str = "prefill_first",
    ) -> None:
        if policy not in SCHEDULING_POLICIES:
            raise ValueError(f"scheduling policy must be one of {SCHEDULING_POLICIES}")
        self.block_manager = block_manager
        self.max_num_seqs = max(1, max_num_seqs)
        self.max_num_batched_tokens = max(1, max_num_batched_tokens)
        self.policy = policy
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
        """The next batch and whether it has prefill rows (decode rows come first)."""
        if self.policy == "mixed":
            decodes = self._schedule_decode()
            prefills = self._schedule_prefill(
                token_budget=self.max_num_batched_tokens - len(decodes),
                seq_budget=self.max_num_seqs - len(decodes),
            )
            return decodes + prefills, bool(prefills)
        prefill_batch = self._schedule_prefill()
        if prefill_batch:
            return prefill_batch, True
        return self._schedule_decode(), False

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool = False
    ) -> list[tuple[Sequence, int]]:
        """Apply a finished model step; return the tokens actually appended."""
        del is_prefill  # each row knows from its own counters
        return self.complete(self.launched(seqs), token_ids)

    def launched(self, seqs: list[Sequence]) -> list[Row]:
        """Advance the scheduled sequences as their step launches.

        The KV of the scheduled tokens counts as computed from now on (later
        steps on the same stream see it), and a sampled token is in flight
        until :meth:`complete` brings its value.
        """
        rows = []
        for seq in seqs:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            # A prefill chunk that leaves prompt tokens unprocessed samples
            # a meaningless logit.
            samples = end >= seq.num_tokens
            seq.num_cached_tokens = end
            seq.num_scheduled_tokens = 0
            if samples:
                seq.num_inflight += 1
            rows.append(Row(seq, samples, end, seq.epoch))
        return rows

    def complete(self, rows: list[Row], token_ids: list[int]) -> list[tuple[Sequence, int]]:
        """Apply a finished step's sampled tokens; return the ones appended."""
        emitted: list[tuple[Sequence, int]] = []
        for row, token_id in zip(rows, token_ids):
            seq = row.seq
            if seq.is_finished:
                # Finished or aborted while its step was running: release it.
                self.abort(seq)
                continue
            if row.epoch == seq.epoch:  # still the blocks this step wrote
                self.block_manager.commit(seq, row.num_computed_tokens)
            if not row.samples:
                continue
            # A preempted sequence keeps the token: the step that sampled it
            # read valid KV, and the recompute prefills it with the rest.
            seq.num_inflight -= 1
            seq.append_token(token_id)
            emitted.append((seq, token_id))
            if seq.should_finish(token_id):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                with suppress(ValueError):
                    self.running.remove(seq)
                with suppress(ValueError):
                    self.waiting.remove(seq)
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

    def _schedule_prefill(
        self, token_budget: Optional[int] = None, seq_budget: Optional[int] = None
    ) -> list[Sequence]:
        token_budget = self.max_num_batched_tokens if token_budget is None else token_budget
        seq_budget = self.max_num_seqs if seq_budget is None else seq_budget
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
        for seq in list(self.waiting):
            if len(scheduled) >= seq_budget:
                break
            if seq.num_inflight:
                # Preempted with a token still in flight: it can prefill once
                # that token's value is known.
                continue
            remaining = token_budget - num_batched_tokens
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
                self.waiting.remove(seq)
                self.running.append(seq)
            scheduled.append(seq)
        return scheduled

    def _schedule_decode(self) -> list[Sequence]:
        """Round-robin decode: scheduled sequences rotate to the back.

        When the batch is capped (max_num_seqs or max_num_batched_tokens), the
        sequences skipped this step sit at the front and go first next step.
        """
        scheduled: list[Sequence] = []
        done: list[Sequence] = []
        budget = min(self.max_num_seqs, self.max_num_batched_tokens)
        current: Optional[Sequence] = None
        try:
            while self.running and len(scheduled) < budget:
                seq = current = self.running.popleft()
                if seq.num_completion_tokens + seq.num_inflight >= seq.sampling.max_tokens:
                    # Its last token is in flight; it finishes when it lands.
                    done.append(seq)
                    current = None
                    continue
                while not self.block_manager.can_append(seq):
                    # Free a partly prefilled request's blocks first (it has
                    # produced nothing yet and stays queued), else preempt the
                    # youngest running one (vLLM's recompute policy); it may be
                    # ``seq`` itself. Already scheduled ones are off-limits.
                    partial = [item for item in self.waiting if item.block_table]
                    if partial:
                        self.block_manager.deallocate(max(partial, key=lambda item: item.seq_id))
                        continue
                    victim = max([seq, *self.running], key=lambda item: item.seq_id)
                    if victim is seq:
                        self.preempt(seq)
                        seq = current = None
                        break
                    self.running.remove(victim)
                    self.preempt(victim)
                if seq is None:
                    continue
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
                current = None
        finally:
            # Never drop popped sequences, even if a step above raised, so the
            # engine can still find (and fail) them.
            if current is not None:
                self.running.appendleft(current)
            self.running.extendleft(reversed(done))
            self.running.extend(scheduled)
        return scheduled
