"""Turn a scheduled batch into one forward pass and sampled tokens."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

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
    replay a CUDA graph once :meth:`capture_decode_graphs` has run.

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
