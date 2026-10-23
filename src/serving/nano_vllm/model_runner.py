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
from .sampling import sample_tokens
from .sequence import Sequence

logger = logging.getLogger(__name__)

# Decode batch sizes that get a CUDA graph; a batch runs in the smallest one
# that fits, with the extra rows padded.
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)


@dataclass
class _DecodeGraphs:
    """Static buffers shared by every captured batch size.

    Rows are padded up to ``max_batch``; each step fills the pinned host
    buffers and copies them over in two transfers. Reusing one pinned buffer
    is safe because every step ends with a host sync (sampling) before the
    next one writes it.
    """

    max_batch: int
    width: int  # block-table width: enough blocks for max_model_len
    host_longs: torch.Tensor  # pinned [3, max_batch]: input_ids, positions, slots
    host_ints: torch.Tensor  # pinned [max_batch * (1 + width)]: context_lens, tables
    longs: torch.Tensor
    ints: torch.Tensor
    cu_seqlens: torch.Tensor
    logits_indices: torch.Tensor
    graphs: dict[int, torch.cuda.CUDAGraph]
    outputs: dict[int, torch.Tensor]

    def bucket(self, batch: int) -> Optional[int]:
        return next((size for size in sorted(self.graphs) if size >= batch), None)


class ModelRunner:
    """Owns the step from scheduler output to sampled token ids.

    The model only needs ``device`` and ``forward_batch(input_ids, positions,
    cache, metadata)`` returning one row of logits per sequence. Decode steps
    replay a CUDA graph once :meth:`capture_decode_graphs` has run.
    """

    def __init__(self, model, cache: PagedKVCache, block_manager: BlockManager):
        self.model = model
        self.cache = cache
        self.block_manager = block_manager
        self._graphs: Optional[_DecodeGraphs] = None

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        logits = self.forward(seqs, is_prefill)
        with nvtx_range("sample"):  # includes waiting for the forward pass
            return sample_tokens(logits, [seq.sampling for seq in seqs])

    def forward(
        self, seqs: list[Sequence], is_prefill: bool, use_graphs: bool = True
    ) -> torch.Tensor:
        """Logits for the last token of each sequence, [len(seqs), vocab]."""
        if use_graphs and not is_prefill and self._graphs is not None:
            bucket = self._graphs.bucket(len(seqs))
            if bucket is not None:
                with nvtx_range("replay"):
                    return self._replay(seqs, bucket)
        with nvtx_range("prepare"):
            input_ids, positions, metadata = self.prepare(seqs, is_prefill)
        with nvtx_range("forward"):
            return self.model.forward_batch(input_ids, positions, self.cache, metadata)

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
            host_longs=torch.empty(3, max_batch, dtype=torch.long, pin_memory=True),
            host_ints=torch.empty(
                max_batch * (1 + width), dtype=torch.int32, pin_memory=True
            ),
            longs=torch.empty(3, max_batch, dtype=torch.long, device=device),
            ints=torch.empty(max_batch * (1 + width), dtype=torch.int32, device=device),
            cu_seqlens=torch.arange(max_batch + 1, dtype=torch.int32, device=device),
            logits_indices=torch.arange(max_batch, dtype=torch.long, device=device),
            graphs={},
            outputs={},
        )
        self._fill_padding(graphs, 0)
        graphs.longs.copy_(graphs.host_longs)
        graphs.ints.copy_(graphs.host_ints)

        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        for size in sorted(sizes, reverse=True):  # largest first: the pool fits all
            input_ids, positions, metadata = self._static_batch(graphs, size)
            # Eager run first: Triton JIT and cuBLAS setup must not happen
            # inside the capture.
            self.model.forward_batch(input_ids, positions, self.cache, metadata)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                graphs.outputs[size] = self.model.forward_batch(
                    input_ids, positions, self.cache, metadata
                )
            graphs.graphs[size] = graph
        torch.cuda.synchronize(device)
        self._graphs = graphs
        logger.info(
            "nano-vLLM captured decode CUDA graphs for batch sizes %s in %.1f s",
            sorted(sizes),
            time.perf_counter() - start,
        )

    def _static_batch(
        self, graphs: _DecodeGraphs, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, BatchMetadata]:
        max_batch = graphs.max_batch
        context_lens = graphs.ints[:max_batch]
        tables = graphs.ints[max_batch:].view(max_batch, graphs.width)
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
        return graphs.longs[0, :size], graphs.longs[1, :size], metadata

    def _fill_padding(self, graphs: _DecodeGraphs, start: int) -> None:
        """Rows from ``start`` on: token 0, null slot, context 1, null blocks."""
        null = self.cache.null_block
        graphs.host_longs[:2, start:] = 0
        graphs.host_longs[2, start:] = null * self.cache.block_size
        max_batch = graphs.max_batch
        graphs.host_ints[start:max_batch] = 1
        tables = graphs.host_ints[max_batch:].view(max_batch, graphs.width)
        tables[start:] = null

    def _replay(self, seqs: list[Sequence], bucket: int) -> torch.Tensor:
        graphs = self._graphs
        batch = len(seqs)
        longs = graphs.host_longs.numpy()
        ints = graphs.host_ints.numpy()
        tables = ints[graphs.max_batch :].reshape(graphs.max_batch, graphs.width)
        null = self.cache.null_block
        for row, seq in enumerate(seqs):
            position = len(seq) - 1
            longs[0, row] = seq.last_token
            longs[1, row] = position
            longs[2, row] = self.block_manager.slot_for_position(seq, position)
            ints[row] = len(seq)
            table = seq.block_table
            if len(table) > graphs.width:
                raise ValueError("sequence is longer than max_model_len allows")
            tables[row, : len(table)] = table
            tables[row, len(table) :] = null
        if batch < bucket:
            self._fill_padding(graphs, batch)
        graphs.longs.copy_(graphs.host_longs, non_blocking=True)
        graphs.ints.copy_(graphs.host_ints, non_blocking=True)
        graphs.graphs[bucket].replay()
        return graphs.outputs[bucket][:batch]
