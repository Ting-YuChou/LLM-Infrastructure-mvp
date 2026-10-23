"""Turn a scheduled batch into one forward pass and sampled tokens."""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .block_manager import BlockManager
from .kv_cache import PagedKVCache
from .metadata import BatchMetadata, build_batch
from .nvtx import nvtx_range
from .sampling import sample_tokens_device
from .sequence import Sequence

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
    # pinned [2, 4, max_batch]: input_ids, positions, slots, gather rows
    host_longs: torch.Tensor
    host_ints: torch.Tensor  # pinned [2, max_batch * (1 + width)]: context_lens, tables
    longs: torch.Tensor
    ints: torch.Tensor
    cu_seqlens: torch.Tensor
    logits_indices: torch.Tensor
    graphs: dict[int, torch.cuda.CUDAGraph]
    outputs: dict[int, torch.Tensor]

    def bucket(self, batch: int) -> Optional[int]:
        return next((size for size in sorted(self.graphs) if size >= batch), None)


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
class StepHandle:
    """A launched step: its sampled tokens arrive in ``host_out[slot][:rows]``."""

    rows: int
    slot: int
    event: Optional[torch.cuda.Event]


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
    """

    def __init__(
        self, model, cache: PagedKVCache, block_manager: BlockManager, max_rows: int = 256
    ):
        self.model = model
        self.cache = cache
        self.block_manager = block_manager
        self._graphs: Optional[_DecodeGraphs] = None
        self._pieces: Optional[_PiecewiseGraphs] = None
        device = model.device
        cuda = device.type == "cuda"
        self.max_rows = max_rows
        self.sampled = torch.zeros(max_rows, dtype=torch.long, device=device)
        self.host_out = torch.zeros(2, max_rows, dtype=torch.long, pin_memory=cuda)
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
    ) -> StepHandle:
        """Enqueue the forward pass and sampling; nothing here waits for the GPU."""
        rows = len(seqs)
        if rows > self.max_rows:
            raise ValueError(f"{rows} rows exceed the runner's {self.max_rows}")
        slot = self._slot
        self._slot ^= 1
        logits = self.forward(seqs, is_prefill, gather_rows=gather_rows, slot=slot)
        with nvtx_range("sample"):
            tokens = sample_tokens_device(logits, [seq.sampling for seq in seqs])
            self.sampled[:rows].copy_(tokens)
            self.host_out[slot, :rows].copy_(tokens, non_blocking=True)
            event = self._events[slot]
            if event is not None:
                event.record()
        return StepHandle(rows, slot, event)

    def collect(self, handle: StepHandle) -> list[int]:
        """Wait for a launched step and return its sampled token ids."""
        with nvtx_range("collect"):
            if handle.event is not None:
                handle.event.synchronize()
            return self.host_out[handle.slot, : handle.rows].tolist()

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
        with nvtx_range("prepare"):
            input_ids, positions, metadata = self.prepare(seqs, is_prefill)
            if gather_rows is not None and any(row >= 0 for row in gather_rows):
                input_ids = self._gather_inputs(input_ids, seqs, gather_rows)
        if use_graphs and self._pieces is not None and len(seqs) <= self._pieces.max_rows:
            bucket = self._pieces.bucket(input_ids.numel())
            if bucket is not None:
                with nvtx_range("pieces"):
                    return self._replay_pieces(input_ids, positions, metadata, bucket)
        with nvtx_range("forward"):
            return self.model.forward_batch(input_ids, positions, self.cache, metadata)

    def prepare(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata]:
        """Pack the rows; ``is_prefill`` says the batch has prefill rows (a mixed
        batch lists its decode rows first), each row follows ``seq.is_prefill``.

        Every row starts at ``num_cached_tokens``. A decode row's input token
        is the one at that position, unless it is still in flight (then a
        placeholder that :meth:`_gather_inputs` replaces).
        """
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        for seq in seqs:
            start = seq.num_cached_tokens
            if seq.is_prefill:
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
        for all batches.
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
            self.model.forward_batch(input_ids, positions, self.cache, metadata)
        torch.cuda.synchronize(self.model.device)
        logger.info(
            "nano-vLLM compiled the eager step's kernels for 0-%d decode rows in %.1f s",
            max_seqs,
            time.perf_counter() - start,
        )

    # --- CUDA graphs for decode ------------------------------------------

    def capture_decode_graphs(self, max_batch: int, max_model_len: int) -> None:
        """Capture one decode graph per size in GRAPH_BATCH_SIZES up to max_batch.

        Every graph reads the same static buffers, so a step only refills them.
        Padded rows decode token 0 into the null block with a context of 1.
        """
        device = self.model.device
        sizes = [size for size in GRAPH_BATCH_SIZES if size < max_batch] + [max_batch]
        width = (max_model_len + self.cache.block_size - 1) // self.cache.block_size
        graphs = _DecodeGraphs(
            max_batch=max_batch,
            width=width,
            host_longs=torch.empty(2, 4, max_batch, dtype=torch.long, pin_memory=True),
            host_ints=torch.empty(
                2, max_batch * (1 + width), dtype=torch.int32, pin_memory=True
            ),
            longs=torch.empty(4, max_batch, dtype=torch.long, device=device),
            ints=torch.empty(max_batch * (1 + width), dtype=torch.int32, device=device),
            cu_seqlens=torch.arange(max_batch + 1, dtype=torch.int32, device=device),
            logits_indices=torch.arange(max_batch, dtype=torch.long, device=device),
            graphs={},
            outputs={},
        )
        self._fill_padding(graphs, 0, slot=0)
        graphs.longs.copy_(graphs.host_longs[0])
        graphs.ints.copy_(graphs.host_ints[0])

        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        with _no_gc_while_capturing():
            for size in sorted(sizes, reverse=True):  # largest first: the pool fits all
                # Eager run first: Triton JIT and cuBLAS setup must not happen
                # inside the capture.
                self._graph_step(graphs, size)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    graphs.outputs[size] = self._graph_step(graphs, size)
                graphs.graphs[size] = graph
        torch.cuda.synchronize(device)
        self._graphs = graphs
        logger.info(
            "nano-vLLM captured decode CUDA graphs for batch sizes %s in %.1f s",
            sorted(sizes),
            time.perf_counter() - start,
        )

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
        return self.model.forward_batch(input_ids, graphs.longs[1, :size], self.cache, metadata)

    def _fill_padding(self, graphs: _DecodeGraphs, start: int, slot: int) -> None:
        """Rows from ``start`` on: token 0, null slot, context 1, null blocks."""
        null = self.cache.null_block
        longs = graphs.host_longs[slot]
        longs[:2, start:] = 0
        longs[2, start:] = null * self.cache.block_size
        longs[3, start:] = -1
        max_batch = graphs.max_batch
        ints = graphs.host_ints[slot]
        ints[start:max_batch] = 1
        tables = ints[max_batch:].view(max_batch, graphs.width)
        tables[start:] = null

    def _replay(
        self, seqs: list[Sequence], bucket: int, gather_rows: Optional[list[int]], slot: int
    ) -> torch.Tensor:
        graphs = self._graphs
        batch = len(seqs)
        longs = graphs.host_longs[slot].numpy()
        ints = graphs.host_ints[slot].numpy()
        tables = ints[graphs.max_batch :].reshape(graphs.max_batch, graphs.width)
        null = self.cache.null_block
        for row, seq in enumerate(seqs):
            position = seq.num_cached_tokens
            longs[0, row] = seq.token_ids[position] if position < seq.num_tokens else 0
            longs[1, row] = position
            longs[2, row] = self.block_manager.slot_for_position(seq, position)
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
        graphs.graphs[bucket].replay()
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
        self, input_ids: torch.Tensor, positions: torch.Tensor, metadata: BatchMetadata, bucket: int
    ) -> torch.Tensor:
        pieces = self._pieces
        n, rows = input_ids.numel(), metadata.num_seqs
        tokens = pieces.tokens
        tokens[0, :n].copy_(input_ids)
        tokens[1, :n].copy_(positions)
        tokens[2, :n].copy_(metadata.slot_mapping)
        if n < bucket:  # padded tokens (stale ids and positions are harmless) write nothing real
            tokens[2, n:bucket].fill_(self.cache.null_block * self.cache.block_size)
        pieces.logits_indices[:rows].copy_(metadata.logits_indices)
        padded_rows = min(bucket, pieces.max_rows)
        if rows < padded_rows:  # stale indices could point past this size's rows
            pieces.logits_indices[rows:padded_rows].zero_()
        replays = pieces.graphs[bucket]
        query, attn = pieces.query[:n], pieces.attn[:n]
        replays[0]()
        for layer_idx in range(len(replays) - 1):
            self.model.attention(layer_idx, query, self.cache, metadata, out=attn)
            replays[layer_idx + 1]()
        return pieces.logits[:rows]
