"""Turn a scheduled batch into one forward pass and sampled tokens."""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .block_manager import BlockManager
from .kv_cache import PagedKVCache
from .metadata import BatchMetadata, build_batch
from .nvtx import nvtx_range
from .sampling import Sampler
from .sequence import Sequence
from .spec import DraftBatch, Proposer, accept_prefix

logger = logging.getLogger(__name__)

# Decode batch sizes that get a CUDA graph; a batch runs in the smallest one
# that fits, with the extra rows padded.
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)


@contextlib.contextmanager
def _no_gc_while_capturing():
    """Collect garbage, then keep the collector off while graphs are captured.

    A collection during a capture can destroy unreachable CUDA graphs (an
    earlier engine's, in the GPU tests) and free their memory pool with
    cudaFree, which invalidates the capture in progress. vLLM freezes the
    collector while capturing for the same reason.
    """
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@dataclass
class _DecodeGraphs:
    """Static buffers shared by every captured batch size.

    Rows are padded up to ``max_batch``; each step fills a pinned host
    staging slot and copies it over in two transfers. There are two slots,
    used in turn: with async scheduling the host fills step N+1's slot while
    step N's copies may still be queued behind step N-1; slot N is reused at
    N+2, after step N has been collected.
    """

    max_batch: int
    width: int  # block-table width: enough blocks for max_model_len
    # pinned [2, 5, max_batch]: input_ids, positions, slots, gather rows, and
    # each row's place among the pending context rows (_PendingContext)
    host_longs: torch.Tensor
    host_ints: torch.Tensor  # pinned [2, max_batch * (1 + width)]: context_lens, tables
    longs: torch.Tensor
    ints: torch.Tensor
    cu_seqlens: torch.Tensor
    logits_indices: torch.Tensor
    graphs: dict[int, Callable[[], None]]  # a graph's replay; on the CPU the step run eagerly
    outputs: dict[int, torch.Tensor]

    def bucket(self, batch: int) -> Optional[int]:
        return next((size for size in sorted(self.graphs) if size >= batch), None)


@dataclass
class _PendingContext:
    """Tokens a decode graph computed whose context the proposer has not
    been given yet: their taps, positions and KV slots, on the device.

    A decode graph with a proposer leaves its rows here instead of calling
    ``Proposer.observe``, and :meth:`ModelRunner.flush_context` hands them
    over many steps at a time. For the DFlash drafter a context write reads
    0.25 GB of projections whatever the number of rows, about 0.8 ms on an
    A100, a sixth of a plain AWQ step; over a few hundred rows at once it
    is a few microseconds per row. Nothing reads the proposer's context
    between cycles, so the rows only have to be written before the next
    one, and before another write to one of their slots.
    """

    capacity: int
    taps: torch.Tensor  # [capacity + 1, tap width]; the last row takes the graphs' padded rows
    positions: torch.Tensor  # [capacity + 1] long
    slots: torch.Tensor  # [capacity + 1] long: the null slot where no row is pending
    sizes: tuple[int, ...]  # the row counts a flush is padded to (one kernel shape each)
    rows: int = 0
    host_slots: set[int] = field(default_factory=set)  # the pending rows' slots


# Token counts that get piecewise graphs; a step runs in the smallest that
# fits, its extra tokens padded (vLLM's sizes are similar: 8-wide up to 256,
# then 16-wide).
PIECEWISE_TOKEN_SIZES = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)


@dataclass
class _PiecewiseGraphs:
    """CUDA graphs of everything but the attention, per padded token count.

    Graph ``i`` of a size runs the model from attention ``i - 1``'s output
    to attention ``i``'s query (``forward_head`` / ``forward_between`` /
    ``forward_tail``); the attention, whose kernels depend on how the tokens
    split into sequences, runs eagerly in between, reading ``query`` and
    writing ``attn``. Every graph reads and writes these static buffers,
    shared by all sizes, so a size's graphs hold no tensors of their own.
    Padded tokens write their keys and values into the null block.
    """

    max_rows: int  # logits rows: at most one per sequence
    tokens: torch.Tensor  # [3, max_tokens] long: input ids, positions, slots
    logits_indices: torch.Tensor  # [max_rows] long
    query: torch.Tensor  # [max_tokens, heads, head_dim]: a piece's output
    attn: torch.Tensor  # same shape: the attention's output
    residual: torch.Tensor  # [max_tokens, hidden]
    rope: tuple[torch.Tensor, torch.Tensor]  # cos, sin: [max_tokens, head_dim]
    logits: torch.Tensor  # [max_rows, vocab]
    graphs: dict[int, list[Callable[[], None]]]  # per size: each piece's replay

    def bucket(self, tokens: int) -> Optional[int]:
        return next((size for size in sorted(self.graphs) if size >= tokens), None)


@dataclass
class _SpecGraphs:
    """A whole speculative cycle per (padded number of sequences, verify
    length): draft, verify, the proposer's context writes, sampling and
    acceptance, as one CUDA graph, so a cycle is one launch from the host.

    The static buffers are :class:`_DecodeGraphs`' with a block of tokens
    per row (the first ``size * block`` entries, for a graph of ``block``
    tokens per row), staged in two slots the same way. Padded rows run a
    block on the null block. On the CPU (tests) a "graph" is the cycle run
    eagerly.
    """

    max_batch: int
    width: int  # block-table width: max_model_len plus the lookahead
    vocab: int
    # pinned [2, 3, max_batch * the longest block]: input ids (only each
    # row's first, the anchor, is read), positions, slots
    host_longs: torch.Tensor
    host_ints: torch.Tensor  # pinned [2, max_batch * (1 + width)]: context_lens, tables
    longs: torch.Tensor
    ints: torch.Tensor
    cu_seqlens: dict[int, torch.Tensor]  # per block: [max_batch + 1] int32, each row's first token
    accepted: torch.Tensor  # [max_batch] long: the drafts each row accepted
    graphs: dict[tuple[int, int], Callable[[], None]]  # (size, block)
    # Per graph, when asked for (tests compare them): what its last cycle
    # left, the verify logits [size * block, vocab] and the drafts [size,
    # block - 1]. Kept alive, they stay out of the pool the graphs share.
    outputs: Optional[dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]]

    def bucket(self, batch: int, block: int) -> Optional[int]:
        sizes = sorted(size for size, length in self.graphs if length == block)
        return next((size for size in sizes if size >= batch), None)


@dataclass
class StepHandle:
    """A launched step: its sampled tokens arrive in ``host_out[slot][:rows]``.

    A speculative step has ``spec_seqs`` sequences of ``block`` rows each
    first, then one row per prefill sequence; the accepted-draft count of
    each speculative sequence follows the tokens in ``host_out``.
    """

    rows: int
    slot: int
    event: Optional[torch.cuda.Event]
    spec_seqs: int = 0
    block: int = 1
    # The size of the CUDA graph the whole step replayed (a decode graph or
    # a cycle's); 0 when it ran any other way.
    graph_size: int = 0


class ModelRunner:
    """Owns the step from scheduler output to sampled token ids.

    The model only needs ``device`` and ``forward_batch(input_ids, positions,
    cache, metadata)`` returning one row of logits per sequence. Decode steps
    replay a CUDA graph once :meth:`capture_decode_graphs` has run, other
    steps piecewise graphs once :meth:`capture_piecewise_graphs` has (the
    model then also needs the ``forward_head`` / ``attention`` /
    ``forward_between`` / ``forward_tail`` pieces of ``forward_batch``).

    A step is :meth:`launch` (inputs up, forward, sampling into the device
    buffer ``sampled``, an async copy down) then :meth:`collect` (wait, read
    the tokens). Between the two the host may schedule and launch the next
    step: a row whose input token is still in flight names the previous
    step's row in ``gather_rows`` and reads it from ``sampled`` on the device.

    With a ``proposer`` a step that has decode rows is a speculative cycle
    (:meth:`_launch_spec`, spec.py) unless it is launched with no draft
    tokens: the decode rows become verify rows of ``draft_tokens + 1``
    tokens (one of ``verify_lens``), and :meth:`collect` returns a list of
    tokens for each. A cycle with no prefill rows replays one CUDA graph
    once :meth:`capture_spec_graphs` has run. Every token the target
    computes, in a plain step too, goes to the proposer as context (from
    the decode graphs in batches: :meth:`flush_context`).
    """

    def __init__(
        self,
        model,
        cache: PagedKVCache,
        block_manager: BlockManager,
        max_rows: int = 256,
        seed: int = 0,
        proposer: Optional[Proposer] = None,
        draft_temperature: float = 1.0,
        verify_lens: Optional[tuple[int, ...]] = None,
    ):
        """``verify_lens``: the lengths a cycle's verify rows can have (the
        anchor and 1 to K drafts); default only the proposer's whole block."""
        self.model = model
        self.cache = cache
        self.block_manager = block_manager
        self.proposer = proposer
        # Rows a step samples per sequence at most: a verify row's whole block.
        self.block = proposer.num_draft_tokens + 1 if proposer is not None else 1
        lens = sorted({length for length in verify_lens or () if length > 1} | {self.block})
        if proposer is not None and lens[-1] > self.block:
            raise ValueError(f"verify lengths {lens} exceed the proposer's block of {self.block}")
        self.verify_lens = tuple(lens) if proposer is not None else ()
        self.max_seqs = max_rows
        max_rows *= self.block
        self._graphs: Optional[_DecodeGraphs] = None
        self._pending: Optional[_PendingContext] = None
        self._pieces: Optional[_PiecewiseGraphs] = None
        self._spec: Optional[_SpecGraphs] = None
        device = model.device
        cuda = device.type == "cuda"
        self.max_rows = max_rows
        self.sampled = torch.zeros(max_rows, dtype=torch.long, device=device)
        # The sampled tokens, then a speculative step's accepted counts.
        self.host_out = torch.zeros(2, max_rows + self.max_seqs, dtype=torch.long, pin_memory=cuda)
        self.sampler = Sampler(max_rows, device, seed=seed)
        # Coupled sampling (spec.py): the drafts are drawn by a sampler with
        # the same seed, so a draft row and the verify row for its position
        # use the same noise, at ``draft_temperature`` times the request's
        # temperature (0: the proposer takes its argmax).
        self.draft_sampler: Optional[Sampler] = None
        if proposer is not None and draft_temperature > 0:
            draft_rows = self.max_seqs * proposer.num_draft_tokens
            self.draft_sampler = Sampler(
                draft_rows, device, seed=seed, temperature_scale=draft_temperature
            )
            self.drafted = torch.zeros(draft_rows, dtype=torch.long, device=device)
        self._events = [torch.cuda.Event() for _ in range(2)] if cuda else [None, None]
        self._slot = 0

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """One synchronous step: launch, then collect."""
        return self.collect(self.launch(seqs, is_prefill))

    def launch(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        gather_rows: Optional[list[int]] = None,
        draft_tokens: Optional[int] = None,
    ) -> StepHandle:
        """Enqueue the forward pass and sampling; nothing here waits for the GPU.

        ``draft_tokens``: with a proposer, the drafts per decode row (default
        its K); 0 decodes one token per row as without a proposer.
        """
        rows = len(seqs)
        if rows > self.max_seqs:
            raise ValueError(f"{rows} rows exceed the runner's {self.max_seqs}")
        slot = self._slot
        self._slot ^= 1
        if self.proposer is not None and not seqs[0].is_prefill:
            block = self.block if draft_tokens is None else draft_tokens + 1
            if block > 1:
                if block not in self.verify_lens:
                    raise ValueError(f"a cycle's verify length is one of {self.verify_lens}, not {block}")
                if gather_rows is not None and any(row >= 0 for row in gather_rows):
                    raise ValueError("a speculative step cannot take in-flight input tokens")
                return self._launch_spec(seqs, slot, block)
        graph_size = 0
        if not is_prefill and self._graphs is not None:
            graph_size = self._graphs.bucket(rows) or 0
        logits = self.forward(seqs, is_prefill, gather_rows=gather_rows, slot=slot)
        with nvtx_range("sample"):
            # A row samples the token at position num_tokens + num_inflight
            # (the scheduler counts this step's token as in flight after launch).
            self.sampler.sample(
                logits,
                [seq.sampling for seq in seqs],
                [seq.seq_id for seq in seqs],
                [seq.num_tokens + seq.num_inflight for seq in seqs],
                self.sampled,
                slot=slot,
            )
            self.host_out[slot, :rows].copy_(self.sampled[:rows], non_blocking=True)
            event = self._events[slot]
            if event is not None:
                event.record()
        return StepHandle(rows, slot, event, graph_size=graph_size)

    def collect(self, handle: StepHandle) -> list[int | list[int]]:
        """Wait for a launched step and return its sampled token ids: one per
        row, or for a speculative sequence the list of its cycle's tokens."""
        with nvtx_range("collect"):
            if handle.event is not None:
                handle.event.synchronize()
            spec, block = handle.spec_seqs, handle.block
            flat = self.host_out[handle.slot, : handle.rows + spec].tolist()
            if not spec:
                return flat
            accepted = flat[handle.rows :]
            tokens: list[int | list[int]] = [
                flat[row * block : row * block + accepted[row] + 1] for row in range(spec)
            ]
            return tokens + flat[spec * block : handle.rows]

    # --- the speculative cycle ---------------------------------------------

    def _launch_spec(self, seqs: list[Sequence], slot: int, block: Optional[int] = None) -> StepHandle:
        """Draft, verify and accept for the decode rows of ``seqs`` (they come
        first), in the same forward pass as any prefill rows.

        A decode row holds its last accepted token (the anchor, whose KV is
        not computed yet) and ``block - 1`` drafts at the positions after it,
        with logits for every one of them; the target's token at each
        position is sampled as a plain decode step would sample it there.
        """
        proposer = self.proposer
        block = block or self.block
        self.flush_context()  # the drafts read every computed token's context
        num_spec = next((i for i, seq in enumerate(seqs) if seq.is_prefill), len(seqs))
        if self._spec is not None and num_spec == len(seqs):
            bucket = self._spec.bucket(num_spec, block)
            if bucket is not None:
                with nvtx_range("spec_replay"):
                    return self._replay_spec(seqs, bucket, slot, block)
        with nvtx_range("prepare"):
            input_ids, positions, metadata = self.prepare(seqs, True, spec_rows=num_spec, block=block)
        spec_tokens = num_spec * block
        with nvtx_range("draft"):
            block_ids = input_ids[:spec_tokens].view(num_spec, block)
            drafts = proposer.propose(
                DraftBatch(
                    seqs=seqs[:num_spec],
                    anchor_ids=block_ids[:, 0],
                    positions=positions[:spec_tokens],
                    slot_mapping=metadata.slot_mapping[:spec_tokens],
                    block_tables=metadata.block_tables[:num_spec],
                    context_lens=metadata.context_lens[:num_spec] - block,
                    draw=self._draft_draw(seqs[:num_spec], slot, block - 1),
                )
            )
            block_ids[:, 1:] = drafts
        logits = self._forward_prepared(input_ids, positions, metadata, observe=True)
        rows = logits.shape[0]
        with nvtx_range("sample"):
            self.sampler.sample(
                logits, *self._spec_sampling(seqs, num_spec, block), self.sampled, slot=slot
            )
            target = self.sampled[:spec_tokens].view(num_spec, block)
            host = self.host_out[slot]
            host[:rows].copy_(self.sampled[:rows], non_blocking=True)
            host[rows : rows + num_spec].copy_(accept_prefix(drafts, target), non_blocking=True)
            event = self._events[slot]
            if event is not None:
                event.record()
        return StepHandle(rows, slot, event, spec_seqs=num_spec, block=block)

    def _draft_sampling(
        self, seqs: list[Sequence], drafted: int
    ) -> tuple[list, list[int], list[int]]:
        """Per draft row (``drafted`` per sequence): the sequence's sampling
        options and id, and the position the draft proposes a token for.
        Draft ``j`` (from 0) stands at ``num_tokens + j``, where verify row
        ``j`` samples."""
        options, seq_ids, positions = [], [], []
        for seq in seqs:
            options += [seq.sampling] * drafted
            seq_ids += [seq.seq_id] * drafted
            positions += range(seq.num_tokens, seq.num_tokens + drafted)
        return options, seq_ids, positions

    def _draft_draw(self, seqs: list[Sequence], slot: int, drafted: int):
        """``DraftBatch.draw`` for an eager step's verify rows ``seqs``."""
        if self.draft_sampler is None:
            return None
        rows = self._draft_sampling(seqs, drafted)

        def draw(logits: torch.Tensor) -> torch.Tensor:
            self.draft_sampler.sample(logits, *rows, self.drafted, slot=slot)
            return self.drafted[: logits.shape[0]]

        return draw

    def _graph_draft_draw(self, logits: torch.Tensor) -> torch.Tensor:
        """``DraftBatch.draw`` inside a graph: the settings are staged before
        the replay."""
        self.draft_sampler.draw(logits, self.drafted)
        return self.drafted[: logits.shape[0]]

    def _spec_sampling(
        self, seqs: list[Sequence], num_spec: int, block: int
    ) -> tuple[list, list[int], list[int]]:
        """Per logits row of a speculative step: sampling options, sequence
        id and the position of the token the row samples (token ``t`` of a
        verify row of ``block`` samples the one at ``num_tokens + t``)."""
        options, seq_ids, positions = [], [], []
        for index, seq in enumerate(seqs):
            count = block if index < num_spec else 1
            options += [seq.sampling] * count
            seq_ids += [seq.seq_id] * count
            positions += range(seq.num_tokens, seq.num_tokens + count)
        return options, seq_ids, positions

    def forward(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        use_graphs: bool = True,
        gather_rows: Optional[list[int]] = None,
        slot: int = 0,
    ) -> torch.Tensor:
        """Logits for the last token of each sequence, [len(seqs), vocab]."""
        if use_graphs and not is_prefill and self._graphs is not None:
            bucket = self._graphs.bucket(len(seqs))
            if bucket is not None:
                with nvtx_range("replay"):
                    return self._replay(seqs, bucket, gather_rows, slot)
        # This step's context writes follow the pending ones: a slot that
        # was freed and given to another sequence keeps the later write.
        self.flush_context()
        with nvtx_range("prepare"):
            input_ids, positions, metadata = self.prepare(seqs, is_prefill)
            if gather_rows is not None and any(row >= 0 for row in gather_rows):
                input_ids = self._gather_inputs(input_ids, seqs, gather_rows)
        # Every token the target computes is context for the proposer.
        observe = self.proposer is not None
        return self._forward_prepared(input_ids, positions, metadata, use_graphs, observe)

    def _forward_prepared(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: BatchMetadata,
        use_graphs: bool = True,
        observe: bool = False,
    ) -> torch.Tensor:
        """The forward pass of a packed batch, through the piecewise graphs
        when they cover it; ``observe`` hands every token's taps to the
        proposer."""
        tap_layers = self.proposer.tap_layers if observe else ()
        bucket = None
        if use_graphs and self._pieces is not None and metadata.logits_indices is not None:
            bucket = self._pieces.bucket(input_ids.numel())
        if bucket is not None and tap_layers:
            with nvtx_range("pieces"):
                per_layer = dict.fromkeys(tap_layers)
                logits = self._replay_pieces(input_ids, positions, metadata, bucket, per_layer)
                taps = torch.cat([per_layer[layer] for layer in tap_layers], dim=-1)
        elif bucket is not None:
            with nvtx_range("pieces"):
                logits, taps = self._replay_pieces(input_ids, positions, metadata, bucket), None
        elif tap_layers:
            with nvtx_range("forward"):
                logits, taps = self.model.forward_batch_taps(
                    input_ids, positions, self.cache, metadata, tap_layers
                )
        else:
            with nvtx_range("forward"):
                logits, taps = self.model.forward_batch(input_ids, positions, self.cache, metadata), None
        if taps is not None:
            with nvtx_range("observe"):
                self.proposer.observe(taps, positions, metadata.slot_mapping)
        return logits

    def prepare(
        self, seqs: list[Sequence], is_prefill: bool, spec_rows: int = 0, block: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata]:
        """Pack the rows; ``is_prefill`` says the batch has prefill rows (a mixed
        batch lists its decode rows first), each row follows ``seq.is_prefill``.

        Every row starts at ``num_cached_tokens``. A decode row's input token
        is the one at that position, unless it is still in flight (then a
        placeholder that :meth:`_gather_inputs` replaces). The first
        ``spec_rows`` rows are a speculative step's, ``block`` tokens each:
        the same input token followed by placeholders for the drafts, with
        logits for all of them.
        """
        block = block or self.block
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        for index, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            if index < spec_rows:
                if start != seq.num_tokens - 1:
                    raise ValueError("a speculative row needs all KV but its last token's")
                end = start + block
                input_ids.append(seq.token_ids[start])
                input_ids.extend([0] * (block - 1))
            elif seq.is_prefill:
                # Uncached prompt tokens (or this chunk of them).
                end = start + seq.num_scheduled_tokens
                input_ids.extend(seq[start:end])
            else:
                # The last sampled token, whose KV is not cached yet.
                end = start + 1
                input_ids.append(seq.token_ids[start] if start < seq.num_tokens else 0)
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
            block_rows=spec_rows,
        )

    def _gather_inputs(
        self, input_ids: torch.Tensor, seqs: list[Sequence], gather_rows: list[int]
    ) -> torch.Tensor:
        """Replace in-flight input tokens by the previous step's sampled ones.

        Gathered rows are decode rows, which come first, so row i is token i.
        """
        per_token = list(gather_rows) + [-1] * (input_ids.numel() - len(gather_rows))
        pin = input_ids.device.type == "cuda"
        rows = torch.tensor(per_token, dtype=torch.long, pin_memory=pin).to(
            input_ids.device, non_blocking=True
        )
        return torch.where(rows >= 0, self.sampled[rows.clamp(min=0)], input_ids)

    # --- eager kernel variants ------------------------------------------

    def warm_up_eager(self, max_seqs: int) -> None:
        """Compile every kernel variant an eager step can use, before serving.

        The decode kernel's split count and launch settings are compile-time
        constants chosen from the number of decode rows, so an eager step
        with a row count no CUDA graph covers (a mixed step with 15 decode
        rows, say) would compile one mid-serving, stalling every running
        request for seconds. A mixed batch per decode row count, all on the
        null block, compiles them, and two decode-only batches cover a full
        batch and the GEMV's one-row specialization; build_batch keeps every
        other specialization (pointer alignment, block-table stride) the same
        for all batches. With a proposer the batches' tokens also go through
        its context writes, as every step's do.
        """
        if self.model.device.type != "cuda":
            return
        null = self.cache.null_block
        start = time.perf_counter()
        batches = [[1] * rows + [2] for rows in range(max_seqs)] + [[1], [1] * max_seqs]
        for query_lens in batches:
            prefill = query_lens[-1] > 1
            tokens = sum(query_lens)
            input_ids, positions, metadata = build_batch(
                input_ids=[0] * tokens,
                positions=[p for n in query_lens for p in range(n)],
                slots=[null * self.cache.block_size] * tokens,
                block_tables=[[null]] * len(query_lens),
                context_lens=list(query_lens),
                query_lens=query_lens,
                is_prefill=prefill,
                null_block=null,
                num_slots=self.cache.total_slots,
                device=self.model.device,
            )
            self._forward_prepared(
                input_ids, positions, metadata, use_graphs=False, observe=self.proposer is not None
            )
        torch.cuda.synchronize(self.model.device)
        logger.info(
            "nano-vLLM compiled the eager step's kernels for 0-%d decode rows in %.1f s",
            max_seqs,
            time.perf_counter() - start,
        )
        if self.proposer is not None:
            self._warm_up_spec(max_seqs)

    def _warm_up_spec(self, max_seqs: int) -> None:
        """A speculative cycle per verify length and number of verify rows,
        on the null block: the block kernel's split count and launch settings
        are chosen from the rows, like the decode kernel's, for the target
        and the proposer, and its row count from the length."""
        start = time.perf_counter()
        for block in self.verify_lens:
            for rows in range(1, max_seqs + 1):
                self.null_spec_cycle(rows, block)
        torch.cuda.synchronize(self.model.device)
        logger.info(
            "nano-vLLM compiled the speculative step's kernels for 1-%d verify rows of %s tokens in %.1f s",
            max_seqs,
            list(self.verify_lens),
            time.perf_counter() - start,
        )

    def null_spec_cycle(self, rows: int, block: Optional[int] = None) -> None:
        """Draft and verify ``rows`` blocks (of the longest verify length
        unless ``block`` says) on the null block, eagerly: to compile the
        kernels of that shape, or to measure its memory."""
        null = self.cache.null_block
        block = block or self.block
        tokens = rows * block
        input_ids, positions, metadata = build_batch(
            input_ids=[0] * tokens,
            positions=list(range(block)) * rows,
            slots=[null * self.cache.block_size] * tokens,
            block_tables=[[null]] * rows,
            context_lens=[block] * rows,
            query_lens=[block] * rows,
            is_prefill=True,
            null_block=null,
            num_slots=self.cache.total_slots,
            device=self.model.device,
            block_rows=rows,
        )
        self.proposer.propose(
            DraftBatch(
                seqs=[None] * rows,
                anchor_ids=input_ids.view(rows, block)[:, 0],
                positions=positions,
                slot_mapping=metadata.slot_mapping,
                block_tables=metadata.block_tables,
                context_lens=metadata.context_lens - block,
            )
        )
        self._forward_prepared(input_ids, positions, metadata, use_graphs=False, observe=True)

    # --- CUDA graphs for the speculative cycle -----------------------------

    @torch.inference_mode()
    def capture_spec_graphs(
        self, max_batch: int, max_model_len: int, keep_outputs: bool = False
    ) -> None:
        """Capture the speculative cycle for each verify length and each size
        in GRAPH_BATCH_SIZES up to ``max_batch`` sequences, when the proposer
        works from the batch's tensors alone (``Proposer.device_only``).

        In inference mode, as :meth:`capture_decode_graphs` explains.

        Without the graph a cycle is the drafter's ~60 kernels, the target's
        piecewise graphs with an eager attention per layer, the context
        writes and the sampling, each launched from the host.

        A graph leaves its results in the runner's static buffers and holds
        nothing else (unless ``keep_outputs``), so the graphs' shared pool is
        as large as the largest cycle, not as their sum.
        """
        proposer, longest = self.proposer, self.block
        if proposer is None or not getattr(proposer, "device_only", False):
            return
        device = self.model.device
        cuda = device.type == "cuda"
        sizes = [size for size in GRAPH_BATCH_SIZES if size < max_batch] + [max_batch]
        width = self._table_width(max_model_len)
        graphs = _SpecGraphs(
            max_batch=max_batch,
            width=width,
            vocab=0,
            host_longs=torch.zeros(2, 3, max_batch * longest, dtype=torch.long, pin_memory=cuda),
            host_ints=torch.zeros(2, max_batch * (1 + width), dtype=torch.int32, pin_memory=cuda),
            longs=torch.zeros(3, max_batch * longest, dtype=torch.long, device=device),
            ints=torch.zeros(max_batch * (1 + width), dtype=torch.int32, device=device),
            cu_seqlens={
                block: torch.arange(max_batch + 1, dtype=torch.int32, device=device) * block
                for block in self.verify_lens
            },
            accepted=torch.zeros(max_batch, dtype=torch.long, device=device),
            graphs={},
            outputs={} if keep_outputs else None,
        )
        self._stage_null_cycles(graphs, longest)

        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle() if cuda else None
        with _no_gc_while_capturing():
            # The longest block and the largest size first: the pool fits all.
            for block in sorted(self.verify_lens, reverse=True):
                self._stage_null_cycles(graphs, block)
                for size in sorted(sizes, reverse=True):
                    key = (size, block)
                    # Eager run first: Triton JIT and cuBLAS setup must not
                    # happen inside the capture.
                    graphs.vocab = self._spec_cycle(graphs, size, block)[0].shape[-1]
                    if not cuda:
                        graphs.graphs[key] = functools.partial(self._record_spec_cycle, graphs, size, block)
                        continue
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        self._record_spec_cycle(graphs, size, block)
                    graphs.graphs[key] = graph.replay
        if cuda:
            torch.cuda.synchronize(device)
        self._spec = graphs
        logger.info(
            "nano-vLLM captured speculative-cycle CUDA graphs for %s sequences, verify rows of %s "
            "tokens, in %.1f s",
            sorted(sizes),
            list(self.verify_lens),
            time.perf_counter() - start,
        )

    def _table_width(self, max_model_len: int) -> int:
        """Block-table width of the graphs' static buffers: a verify row's
        block reaches ``block - 1`` tokens past max_model_len, and a sequence
        keeps those blocks in the plain steps after it."""
        return -(-(max_model_len + self.block - 1) // self.cache.block_size)

    def _stage_null_cycles(self, graphs: _SpecGraphs, block: int) -> None:
        """The static buffers for cycles of ``block`` on the null block,
        every row drawing greedily (a capture's inputs, or a timing's)."""
        self._fill_spec_padding(graphs, 0, 0, block)
        graphs.longs.copy_(graphs.host_longs[0])
        graphs.ints.copy_(graphs.host_ints[0])
        self.sampler.stage([], [], [], 0, slot=0, padded_rows=graphs.max_batch * block)
        if self.draft_sampler is not None:
            self.draft_sampler.stage(
                [], [], [], 0, slot=0, padded_rows=graphs.max_batch * (block - 1)
            )

    def _record_spec_cycle(self, graphs: _SpecGraphs, size: int, block: int) -> None:
        outputs = self._spec_cycle(graphs, size, block)
        if graphs.outputs is not None:
            graphs.outputs[(size, block)] = outputs

    def _spec_cycle(
        self, graphs: _SpecGraphs, size: int, block: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The cycle a graph records for ``size`` sequences of ``block``
        tokens, from the static buffers only: the target's tokens into
        ``sampled``, the accepted counts into ``graphs.accepted``. Returns
        the verify logits and the drafts."""
        proposer = self.proposer
        tokens = size * block
        anchors = graphs.longs[0, :tokens].view(size, block)[:, :1]
        positions, slots = graphs.longs[1, :tokens], graphs.longs[2, :tokens]
        context_lens = graphs.ints[:size]
        tables = graphs.ints[graphs.max_batch :].view(graphs.max_batch, graphs.width)[:size]
        drafts = proposer.propose(
            DraftBatch(
                seqs=[None] * size,
                anchor_ids=anchors[:, 0],
                positions=positions,
                slot_mapping=slots,
                block_tables=tables,
                context_lens=context_lens - block,
                draw=self._graph_draft_draw if self.draft_sampler is not None else None,
            )
        )
        input_ids = torch.cat([anchors, drafts], dim=1).view(-1)
        metadata = BatchMetadata(
            slot_mapping=slots,
            block_tables=tables,
            context_lens=context_lens,
            cu_seqlens_q=graphs.cu_seqlens[block][: size + 1],
            query_lens=[block] * size,
            context_lens_host=[block] * size,
            is_prefill=True,
            logits_indices=None,  # every row
            num_block_rows=size,
            block_len=block,
        )
        if proposer.tap_layers:
            logits, taps = self.model.forward_batch_taps(
                input_ids, positions, self.cache, metadata, proposer.tap_layers
            )
            proposer.observe(taps, positions, slots)
        else:
            logits = self.model.forward_batch(input_ids, positions, self.cache, metadata)
        self.sampler.draw(logits, self.sampled)
        target = self.sampled[:tokens].view(size, block)
        graphs.accepted[:size].copy_(accept_prefix(drafts, target))
        return logits, drafts

    def _fill_spec_padding(self, graphs: _SpecGraphs, start: int, slot: int, block: int) -> None:
        """Rows from ``start`` on: a block of token 0 at position 0 in the
        null block, whose context is that block."""
        null = self.cache.null_block
        longs = graphs.host_longs[slot]
        longs[:2, start * block :] = 0
        longs[2, start * block :] = null * self.cache.block_size
        max_batch = graphs.max_batch
        ints = graphs.host_ints[slot]
        ints[start:max_batch] = block
        ints[max_batch:].view(max_batch, graphs.width)[start:] = null

    @torch.inference_mode()
    def _replay_spec(self, seqs: list[Sequence], bucket: int, slot: int, block: int) -> StepHandle:
        graphs = self._spec
        batch = len(seqs)
        tokens = batch * block
        longs = graphs.host_longs[slot].numpy()
        ints = graphs.host_ints[slot].numpy()
        tables = ints[graphs.max_batch :].reshape(graphs.max_batch, graphs.width)
        null = self.cache.null_block
        for row, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            if start != seq.num_tokens - 1:
                raise ValueError("a speculative row needs all KV but its last token's")
            first = row * block
            longs[0, first] = seq.token_ids[start]
            longs[1, first : first + block] = range(start, start + block)
            longs[2, first : first + block] = [
                self.block_manager.slot_for_position(seq, pos) for pos in range(start, start + block)
            ]
            ints[row] = start + block
            table = seq.block_table
            if len(table) > graphs.width:
                raise ValueError("sequence is longer than max_model_len allows")
            tables[row, : len(table)] = table
            tables[row, len(table) :] = null
        self._fill_spec_padding(graphs, batch, slot, block)
        self.sampler.stage(
            *self._spec_sampling(seqs, batch, block), graphs.vocab, slot=slot,
            padded_rows=bucket * block,
        )
        if self.draft_sampler is not None:
            self.draft_sampler.stage(
                *self._draft_sampling(seqs, block - 1), graphs.vocab, slot=slot,
                padded_rows=bucket * (block - 1),
            )
        graphs.longs.copy_(graphs.host_longs[slot], non_blocking=True)
        graphs.ints.copy_(graphs.host_ints[slot], non_blocking=True)
        graphs.graphs[(bucket, block)]()
        host = self.host_out[slot]
        host[:tokens].copy_(self.sampled[:tokens], non_blocking=True)
        host[tokens : tokens + batch].copy_(graphs.accepted[:batch], non_blocking=True)
        event = self._events[slot]
        if event is not None:
            event.record()
        return StepHandle(tokens, slot, event, spec_seqs=batch, block=block, graph_size=bucket)

    # --- what a step of each graph takes ------------------------------------

    @torch.inference_mode()
    def profile_graphs(self, repeats: int = 4) -> dict[tuple[int, int], float]:
        """Milliseconds of one step of each captured graph, replayed on the
        null block: ``{(sequences, verify length): ms}``, a decode graph
        (with its sampling, without the proposer's context writes it leaves
        pending) being length 1. Empty off CUDA.

        The contexts are empty, so this is the part of a step that depends
        on its rows: the weights' passes, the drafter and the sampling. What
        the sequences' KV adds is measured while serving (spec.SpecPolicy).
        """
        device = self.model.device
        if device.type != "cuda":
            return {}

        def measure(step: Callable[[], None]) -> float:
            step()
            torch.cuda.synchronize(device)
            best = float("inf")
            for _ in range(2):
                start = time.perf_counter()
                for _ in range(repeats):
                    step()
                torch.cuda.synchronize(device)
                best = min(best, (time.perf_counter() - start) / repeats)
            return best * 1e3

        times: dict[tuple[int, int], float] = {}
        start = time.perf_counter()
        if self._graphs is not None:
            decode = self._graphs
            self._fill_padding(decode, 0, slot=0)
            decode.longs.copy_(decode.host_longs[0])
            decode.ints.copy_(decode.host_ints[0])
            self.sampler.stage([], [], [], 0, slot=0, padded_rows=decode.max_batch)
            for size, replay in decode.graphs.items():
                logits = decode.outputs[size]

                def step(replay=replay, logits=logits) -> None:
                    replay()
                    self.sampler.draw(logits, self.sampled, (False, False, False))

                times[(size, 1)] = measure(step)
        if self._spec is not None:
            for block in self.verify_lens:
                self._stage_null_cycles(self._spec, block)
                for (size, length), replay in self._spec.graphs.items():
                    if length == block:
                        times[(size, block)] = measure(replay)
        sizes = sorted({size for size, _ in times})
        lens = sorted({length for _, length in times})
        logger.info(
            "nano-vLLM timed the graphs' steps in %.1f s; ms per step, a row per number of "
            "sequences %s, a column per verify length %s:\n%s",
            time.perf_counter() - start,
            sizes,
            lens,
            "\n".join(
                "  ".join(f"{times[(size, length)]:7.2f}" if (size, length) in times else "      -"
                          for length in lens)
                for size in sizes
            ),
        )
        return times

    # --- CUDA graphs for decode ------------------------------------------

    @torch.inference_mode()
    def capture_decode_graphs(self, max_batch: int, max_model_len: int) -> None:
        """Capture one decode graph per size in GRAPH_BATCH_SIZES up to max_batch.

        Every graph reads the same static buffers, so a step only refills them.
        Padded rows decode token 0 into the null block with a context of 1.

        Every capture and replay in this class runs in inference mode.
        PyTorch keeps the default generator's graph state in tensors made
        when the first live graph is captured and updates them in place at
        each later capture and replay; if that first capture was in
        inference mode (the piecewise graphs', say), an update outside it
        raises. Inside inference mode either kind of tensor can be updated.

        With a proposer that reads the target's taps, a graph leaves its
        tokens' taps among the pending context rows (:class:`_PendingContext`)
        for :meth:`flush_context`. On the CPU (tests) a "graph" is the step
        run eagerly.
        """
        device = self.model.device
        cuda = device.type == "cuda"
        sizes = [size for size in GRAPH_BATCH_SIZES if size < max_batch] + [max_batch]
        width = self._table_width(max_model_len)
        graphs = _DecodeGraphs(
            max_batch=max_batch,
            width=width,
            host_longs=torch.zeros(2, 5, max_batch, dtype=torch.long, pin_memory=cuda),
            host_ints=torch.zeros(2, max_batch * (1 + width), dtype=torch.int32, pin_memory=cuda),
            longs=torch.zeros(5, max_batch, dtype=torch.long, device=device),
            ints=torch.zeros(max_batch * (1 + width), dtype=torch.int32, device=device),
            cu_seqlens=torch.arange(max_batch + 1, dtype=torch.int32, device=device),
            logits_indices=torch.arange(max_batch, dtype=torch.long, device=device),
            graphs={},
            outputs={},
        )
        self._pending = None
        if self.proposer is not None and self.proposer.tap_layers:
            self._pending = self._new_pending_context(max_batch)
        self._fill_padding(graphs, 0, slot=0)
        graphs.longs.copy_(graphs.host_longs[0])
        graphs.ints.copy_(graphs.host_ints[0])

        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle() if cuda else None
        with _no_gc_while_capturing():
            for size in sorted(sizes, reverse=True):  # largest first: the pool fits all
                # Eager run first: Triton JIT and cuBLAS setup must not happen
                # inside the capture.
                self._record_decode_step(graphs, size)
                if not cuda:
                    graphs.graphs[size] = functools.partial(self._record_decode_step, graphs, size)
                    continue
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._record_decode_step(graphs, size)
                graphs.graphs[size] = graph.replay
        if cuda:
            torch.cuda.synchronize(device)
        self._graphs = graphs
        logger.info(
            "nano-vLLM captured decode CUDA graphs for batch sizes %s in %.1f s",
            sorted(sizes),
            time.perf_counter() - start,
        )

    def _new_pending_context(self, max_batch: int) -> _PendingContext:
        """Room for a few steps of the largest decode graph, and one run of
        the proposer's context writes per padded size (their kernels'
        first use must not be mid-serving)."""
        device = self.model.device
        capacity = max(4 * max_batch, 64)
        width = len(self.proposer.tap_layers) * self.model.config.hidden_size
        null_slot = self.cache.null_block * self.cache.block_size
        pending = _PendingContext(
            capacity=capacity,
            taps=torch.zeros(capacity + 1, width, dtype=self.model.dtype, device=device),
            positions=torch.zeros(capacity + 1, dtype=torch.long, device=device),
            slots=torch.full((capacity + 1,), null_slot, dtype=torch.long, device=device),
            sizes=tuple(sorted({min(size, capacity) for size in (16, 64, 256, 1024)} | {capacity})),
        )
        for size in pending.sizes:
            self.proposer.observe(pending.taps[:size], pending.positions[:size], pending.slots[:size])
        return pending

    @torch.inference_mode()
    def flush_context(self) -> None:
        """Give the proposer the context of the tokens the decode graphs
        left pending. Called before a cycle (its drafts read the context),
        before any step that writes context itself, and before a decode
        graph whose rows would not fit or would write a pending slot again."""
        pending = self._pending
        if pending is None or not pending.rows:
            return
        with nvtx_range("flush_context"):
            # Padded to a fixed size: the rows past the pending ones hold the null slot.
            size = next(size for size in pending.sizes if size >= pending.rows)
            self.proposer.observe(pending.taps[:size], pending.positions[:size], pending.slots[:size])
            pending.slots.fill_(self.cache.null_block * self.cache.block_size)
        pending.rows = 0
        pending.host_slots.clear()

    def _record_decode_step(self, graphs: _DecodeGraphs, size: int) -> None:
        graphs.outputs[size] = self._graph_step(graphs, size)

    def _graph_step(self, graphs: _DecodeGraphs, size: int) -> torch.Tensor:
        """The decode step a graph records, from the static buffers only.

        Everything, the gather of in-flight tokens included, must happen in
        here: work done before the capture would be baked in as constants.
        """
        max_batch = graphs.max_batch
        context_lens = graphs.ints[:max_batch]
        tables = graphs.ints[max_batch:].view(max_batch, graphs.width)
        # Rows naming a previous-step row (>= 0) take their input token from
        # ``sampled`` on the device.
        gather = graphs.longs[3, :size]
        input_ids = torch.where(gather >= 0, self.sampled[gather.clamp(min=0)], graphs.longs[0, :size])
        metadata = BatchMetadata(
            slot_mapping=graphs.longs[2, :size],
            block_tables=tables[:size],
            context_lens=context_lens[:size],
            cu_seqlens_q=graphs.cu_seqlens[: size + 1],
            query_lens=[1] * size,
            context_lens_host=[1] * size,
            is_prefill=False,
            logits_indices=graphs.logits_indices[:size],
        )
        positions = graphs.longs[1, :size]
        pending = self._pending
        if pending is None:
            return self.model.forward_batch(input_ids, positions, self.cache, metadata)
        logits, taps = self.model.forward_batch_taps(
            input_ids, positions, self.cache, metadata, self.proposer.tap_layers
        )
        # The proposer's context of these tokens: written by flush_context.
        rows = graphs.longs[4, :size]
        pending.taps.index_copy_(0, rows, taps)
        pending.positions.index_copy_(0, rows, positions)
        pending.slots.index_copy_(0, rows, metadata.slot_mapping)
        return logits

    def _fill_padding(self, graphs: _DecodeGraphs, start: int, slot: int) -> None:
        """Rows from ``start`` on: token 0, null slot, context 1, null blocks;
        their taps go to the pending context's spare row."""
        null = self.cache.null_block
        longs = graphs.host_longs[slot]
        longs[:2, start:] = 0
        longs[2, start:] = null * self.cache.block_size
        longs[3, start:] = -1
        longs[4, start:] = self._pending.capacity if self._pending is not None else 0
        max_batch = graphs.max_batch
        ints = graphs.host_ints[slot]
        ints[start:max_batch] = 1
        tables = ints[max_batch:].view(max_batch, graphs.width)
        tables[start:] = null

    @torch.inference_mode()
    def _replay(
        self, seqs: list[Sequence], bucket: int, gather_rows: Optional[list[int]], slot: int
    ) -> torch.Tensor:
        graphs = self._graphs
        batch = len(seqs)
        longs = graphs.host_longs[slot].numpy()
        ints = graphs.host_ints[slot].numpy()
        tables = ints[graphs.max_batch :].reshape(graphs.max_batch, graphs.width)
        null = self.cache.null_block
        kv_slots = [self.block_manager.slot_for_position(seq, seq.num_cached_tokens) for seq in seqs]
        pending = self._pending
        if pending is not None:
            # The rows join the pending context; a slot already pending (its
            # block was freed and reused, or a discarded step is run again)
            # must be written in order, so what is pending goes first.
            if pending.rows + batch > pending.capacity or not pending.host_slots.isdisjoint(kv_slots):
                self.flush_context()
            longs[4, :batch] = range(pending.rows, pending.rows + batch)
            pending.rows += batch
            pending.host_slots.update(kv_slots)
        for row, seq in enumerate(seqs):
            position = seq.num_cached_tokens
            longs[0, row] = seq.token_ids[position] if position < seq.num_tokens else 0
            longs[1, row] = position
            longs[2, row] = kv_slots[row]
            longs[3, row] = gather_rows[row] if gather_rows is not None else -1
            ints[row] = position + 1
            table = seq.block_table
            if len(table) > graphs.width:
                raise ValueError("sequence is longer than max_model_len allows")
            tables[row, : len(table)] = table
            tables[row, len(table) :] = null
        # Rows past the batch: padding (refilled every time, since the two
        # staging slots alternate).
        self._fill_padding(graphs, batch, slot)
        graphs.longs.copy_(graphs.host_longs[slot], non_blocking=True)
        graphs.ints.copy_(graphs.host_ints[slot], non_blocking=True)
        graphs.graphs[bucket]()
        return graphs.outputs[bucket][:batch]

    # --- piecewise CUDA graphs for the other steps -------------------------

    @torch.inference_mode()
    def capture_piecewise_graphs(self, max_tokens: int, max_rows: int) -> None:
        """Capture the pieces of the forward pass for each size in
        PIECEWISE_TOKEN_SIZES up to ``max_tokens``.

        A step of ``n`` tokens (prefill rows, mixed rows, or decode rows past
        the decode graphs) replays its size's graphs with the attention run
        eagerly between them: about one kernel per layer from the host
        instead of the ~360 of an eager step. On the CPU (tests) the pieces
        are kept as plain functions and "replay" runs them eagerly.
        """
        device = self.model.device
        cuda = device.type == "cuda"
        model, cache = self.model, self.cache
        sizes = [size for size in PIECEWISE_TOKEN_SIZES if size < max_tokens] + [max_tokens]
        tokens = torch.zeros(3, max_tokens, dtype=torch.long, device=device)
        tokens[2].fill_(cache.null_block * cache.block_size)  # capture-time writes: null block
        logits_indices = torch.zeros(max_rows, dtype=torch.long, device=device)
        # Shapes and dtypes of the tensors the pieces hand over, from one eager run.
        query, residual, rope = model.forward_head(tokens[0], tokens[1], cache, tokens[2])
        logits = model.forward_tail(torch.zeros_like(query), residual, logits_indices)
        pieces = _PiecewiseGraphs(
            max_rows=max_rows,
            tokens=tokens,
            logits_indices=logits_indices,
            query=torch.zeros_like(query),
            attn=torch.zeros_like(query),
            residual=torch.zeros_like(residual),
            rope=(torch.zeros_like(rope[0]), torch.zeros_like(rope[1])),
            logits=torch.zeros_like(logits),
            graphs={},
        )
        del query, residual, rope, logits

        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle() if cuda else None
        num_layers = len(model.layers)
        with _no_gc_while_capturing():
            for size in sorted(sizes, reverse=True):  # largest first: the pool fits all
                replays = []
                for index in range(num_layers + 1):
                    piece = functools.partial(self._piece, pieces, index, size)
                    if not cuda:
                        replays.append(piece)
                        continue
                    # Eager run first: Triton JIT and cuBLAS setup must not
                    # happen inside the capture.
                    piece()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        piece()
                    replays.append(graph.replay)
                pieces.graphs[size] = replays
        if cuda:
            torch.cuda.synchronize(device)
        self._pieces = pieces
        logger.info(
            "nano-vLLM captured piecewise CUDA graphs (%d pieces) for token counts %s in %.1f s",
            num_layers + 1,
            sorted(sizes),
            time.perf_counter() - start,
        )

    def _piece(self, pieces: _PiecewiseGraphs, index: int, size: int) -> None:
        """Piece ``index`` of a ``size``-token step, on the static buffers only."""
        model, n = self.model, size
        ids, positions, slots = pieces.tokens[0, :n], pieces.tokens[1, :n], pieces.tokens[2, :n]
        rope = (pieces.rope[0][:n], pieces.rope[1][:n])
        if index == 0:
            query, residual, (cos, sin) = model.forward_head(ids, positions, self.cache, slots)
            rope[0].copy_(cos)
            rope[1].copy_(sin)
        elif index < len(model.layers):
            query, residual = model.forward_between(
                index - 1, pieces.attn[:n], pieces.residual[:n], rope, self.cache, slots
            )
        else:
            rows = min(n, pieces.max_rows)
            logits = model.forward_tail(pieces.attn[:n], pieces.residual[:n], pieces.logits_indices[:rows])
            pieces.logits[:rows].copy_(logits)
            return
        pieces.query[:n].copy_(query)
        pieces.residual[:n].copy_(residual)

    @torch.inference_mode()
    def _replay_pieces(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: BatchMetadata,
        bucket: int,
        taps: Optional[dict[int, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """The logits; ``taps``' keys are layers, filled in with the tokens'
        residual stream after each (piece ``layer + 1`` leaves it in the
        ``residual`` buffer).

        The last piece's graph computes at most ``max_rows`` rows of logits
        (one per sequence); a speculative step with more verify rows than
        that runs the piece eagerly instead.
        """
        pieces = self._pieces
        n, rows = input_ids.numel(), metadata.logits_indices.numel()
        tokens = pieces.tokens
        tokens[0, :n].copy_(input_ids)
        tokens[1, :n].copy_(positions)
        tokens[2, :n].copy_(metadata.slot_mapping)
        if n < bucket:  # padded tokens (stale ids and positions are harmless) write nothing real
            tokens[2, n:bucket].fill_(self.cache.null_block * self.cache.block_size)
        padded_rows = min(bucket, pieces.max_rows)
        if rows <= pieces.max_rows:
            pieces.logits_indices[:rows].copy_(metadata.logits_indices)
        if rows < padded_rows:  # stale indices could point past this size's rows
            pieces.logits_indices[rows:padded_rows].zero_()
        replays = pieces.graphs[bucket]
        query, attn = pieces.query[:n], pieces.attn[:n]
        if taps:
            self.model.check_tap_layers(tuple(taps))
        replays[0]()
        last = len(replays) - 2
        for layer_idx in range(last):
            self.model.attention(layer_idx, query, self.cache, metadata, out=attn)
            replays[layer_idx + 1]()
            if taps and layer_idx in taps:
                taps[layer_idx] = pieces.residual[:n].clone()
        self.model.attention(last, query, self.cache, metadata, out=attn)
        if rows > pieces.max_rows:
            return self.model.forward_tail(attn, pieces.residual[:n], metadata.logits_indices)
        replays[last + 1]()
        return pieces.logits[:rows]
