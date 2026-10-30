"""The engine core: scheduler, KV blocks and model runner, stepped synchronously.

EngineCore knows nothing about text or asyncio. A frontend submits requests
as token ids (:class:`EngineRequest`), and each :meth:`EngineCore.step` returns
the tokens produced (:class:`RequestOutput`). Requests and aborts go through a
thread-safe inbox drained at the start of every step, so a frontend on
another thread (the in-process engine runs steps in a worker thread) or in
another process (:func:`run_core_process`) never touches scheduler state
while a step runs.

With ``async_scheduling`` a step launches the next batch before collecting
the previous one, so the host schedules, prepares and launches step N+1
while the GPU runs step N (vLLM's async scheduling, SGLang's overlap
scheduler). Tokens still in flight feed the next step on the device
(ModelRunner's ``sampled`` buffer); finish checks run one step late, so a
request that just hit EOS may run one extra, discarded, step.
"""

from __future__ import annotations

import logging
import os
import signal
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import torch

from .block_manager import BlockManager
from .config import NanoVLLMConfig
from .kv_cache import PagedKVCache
from .metadata import build_batch
from .model_runner import ModelRunner, StepHandle
from .nvtx import nvtx_range
from .sampling import SamplingOptions
from .scheduler import Row, Scheduler
from .sequence import Sequence, SequenceStatus

logger = logging.getLogger(__name__)

# num_kvcache_blocks: 0 on CPU (nothing to profile).
CPU_KV_BLOCKS = 1024
# GPU memory kept out of the KV budget for the decode CUDA graphs' pool, the
# sampler's buffers (fp32 logits and a sort on its PyTorch path) and
# cuBLAS/Triton workspaces.
KV_RESERVE_BYTES = 512 * 1024**2


@dataclass
class EngineRequest:
    """A request as the core sees it: token ids, no text or stop strings."""

    req_id: int
    prompt_token_ids: list[int]
    sampling: SamplingOptions  # max_tokens resolved; stop strings stay in the frontend
    eos_token_ids: tuple[int, ...] = ()


@dataclass
class RequestOutput:
    """One request's result of a step: a new token, the end, or an error."""

    req_id: int
    token_id: Optional[int] = None
    finish_reason: Optional[str] = None  # set on the request's last output
    error: Optional[BaseException] = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None or self.error is not None


class EngineCore:
    """Continuous batching over one model; drive it with :meth:`step`."""

    def __init__(self, config: NanoVLLMConfig, model) -> None:
        self.config = config
        self.model = model
        if config.num_kvcache_blocks == 0:
            config.num_kvcache_blocks = size_kv_cache(config, model)
        Sequence.block_size = config.kvcache_block_size
        manager_cls = BlockManager
        if config.prefix_cache_backend == "cpp":
            from .radix_cpp import CppBlockManager as manager_cls
        self.block_manager = manager_cls(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            policy=config.scheduling,
        )
        self.paged_cache = model.new_paged_cache(
            num_blocks=config.num_kvcache_blocks, block_size=config.kvcache_block_size
        )
        self.model_runner = ModelRunner(
            model,
            self.paged_cache,
            self.block_manager,
            max_rows=config.max_num_seqs,
            seed=config.seed,
        )
        self.async_scheduling = config.async_scheduling
        self.requests: dict[int, Sequence] = {}
        self._inbox: deque[tuple[str, object]] = deque()
        self._inflight: Optional[tuple[StepHandle, list[Row]]] = None
        self._step_count = 0

    def capture_graphs(self) -> None:
        """Before serving on CUDA: compile the eager step's kernel variants,
        then capture the decode CUDA graphs and the piecewise graphs for the
        other steps (unless enforce_eager)."""
        if self.model.device.type != "cuda":
            return
        self.model_runner.warm_up_eager(self.config.max_num_seqs)
        if self.config.enforce_eager:
            return
        self.model_runner.capture_decode_graphs(
            max_batch=min(self.config.max_num_seqs, self.config.cuda_graph_max_bs),
            max_model_len=self.config.max_model_len,
        )
        max_tokens = min(self.config.cuda_graph_max_tokens, self.config.max_num_batched_tokens)
        if max_tokens > 0:
            self.model_runner.capture_piecewise_graphs(max_tokens, max_rows=self.config.max_num_seqs)

    # --- frontend side (any thread) ------------------------------------------

    def add_request(self, request: EngineRequest) -> None:
        self._inbox.append(("add", request))

    def abort_requests(self, req_ids: list[int]) -> None:
        self._inbox.append(("abort", list(req_ids)))

    def has_work(self) -> bool:
        return (
            bool(self._inbox)
            or not self.scheduler.is_finished()
            or self._inflight is not None
        )

    # --- the step (one thread at a time) --------------------------------------

    def step(self) -> list[RequestOutput]:
        """Apply queued requests and aborts, then run one schedule -> launch ->
        collect -> postprocess round (async: launch this batch, collect the
        previous one). Never raises: failures become error outputs."""
        self._step_count += 1
        with nvtx_range(f"engine_step_{self._step_count}"):
            outputs = self._drain_inbox()
            outputs += self._step()
        return outputs

    def _drain_inbox(self) -> list[RequestOutput]:
        outputs = []
        while self._inbox:
            kind, payload = self._inbox.popleft()
            if kind == "abort":
                for req_id in payload:
                    seq = self.requests.pop(req_id, None)
                    if seq is not None:
                        self.scheduler.abort(seq, "abort")
                continue
            request = payload
            seq = Sequence(
                prompt_token_ids=list(request.prompt_token_ids),
                sampling=request.sampling,
                eos_token_id=request.eos_token_ids,
                ignore_eos=request.sampling.ignore_eos,
                seq_id=request.req_id,
            )
            try:
                self.scheduler.add(seq)
            except ValueError as exc:  # can never fit in the KV cache
                outputs.append(RequestOutput(request.req_id, error=exc))
                continue
            self.requests[request.req_id] = seq
        return outputs

    def _step(self) -> list[RequestOutput]:
        outputs: list[RequestOutput] = []
        try:
            with nvtx_range("schedule"):
                seqs, has_prefill = self.scheduler.schedule()
        except Exception as exc:
            # Scheduler state for queued requests can't be trusted anymore.
            logger.exception("nano-vLLM schedule() failed; failing queued requests")
            return self._collect_inflight() + self._fail(
                [*self.scheduler.running, *self.scheduler.waiting], exc
            )
        launched = None
        if seqs:
            launched, failed = self._launch(seqs, has_prefill)
            outputs += failed
        elif self._inflight is None:
            stuck = self.scheduler.unschedulable_head()
            if stuck is not None:
                outputs += self._fail(
                    [stuck],
                    RuntimeError(
                        "Request cannot be scheduled: it needs more KV blocks "
                        "than are free with nothing left to release"
                    ),
                )
        if self.async_scheduling:
            outputs += self._collect_inflight()
            self._inflight = launched
        elif launched is not None:
            outputs += self._complete(*launched)
        return outputs

    def _launch(
        self, seqs: list[Sequence], has_prefill: bool
    ) -> tuple[Optional[tuple[StepHandle, list[Row]]], list[RequestOutput]]:
        # A decode row whose input token is still in flight reads it from the
        # previous step's row on the device.
        gather = None
        if self.async_scheduling:
            gather = [
                seq.pending_row if not seq.is_prefill and seq.num_inflight else -1
                for seq in seqs
            ]
        try:
            handle = self.model_runner.launch(seqs, has_prefill, gather)
        except Exception as exc:
            logger.exception("nano-vLLM model step failed")
            return None, self._fail(seqs, exc)
        rows = self.scheduler.launched(seqs)
        for index, row in enumerate(rows):
            if row.samples:
                row.seq.pending_row = index
        return (handle, rows), []

    def _collect_inflight(self) -> list[RequestOutput]:
        if self._inflight is None:
            return []
        inflight, self._inflight = self._inflight, None
        return self._complete(*inflight)

    def _complete(self, handle: StepHandle, rows: list[Row]) -> list[RequestOutput]:
        try:
            token_ids = self.model_runner.collect(handle)
            with nvtx_range("postprocess"):
                emitted = self.scheduler.complete(rows, token_ids)
        except Exception as exc:
            logger.exception("nano-vLLM model step failed")
            return self._fail([row.seq for row in rows if not row.seq.is_finished], exc)
        outputs = []
        for seq, token_id in emitted:
            finish = seq.finish_reason if seq.is_finished else None
            outputs.append(RequestOutput(seq.seq_id, token_id, finish))
            if finish is not None:
                self.requests.pop(seq.seq_id, None)
        return outputs

    def _fail(self, seqs: list[Sequence], exc: BaseException) -> list[RequestOutput]:
        outputs = []
        for seq in seqs:
            try:
                self.scheduler.abort(seq, finish_reason="error")
            except Exception:  # still report the failure below
                logger.exception("Failed to release nano-vLLM sequence %s", seq.seq_id)
                seq.status = SequenceStatus.FINISHED
            self.requests.pop(seq.seq_id, None)
            outputs.append(RequestOutput(seq.seq_id, error=exc))
        return outputs

    def fail_all(self, exc: BaseException) -> list[RequestOutput]:
        self._drain_inbox()
        outputs = self._collect_inflight()
        return outputs + self._fail([*self.scheduler.running, *self.scheduler.waiting], exc)

    # --- utilities -------------------------------------------------------------

    def stats(self) -> dict:
        """KV block usage, prefix-cache counters and queue lengths."""
        active = [*self.scheduler.running, *(s for s in self.scheduler.waiting if s.block_table)]
        allocated = sum(len(seq.block_table) for seq in active) * self.block_manager.block_size
        filled = sum(min(seq.num_tokens, len(seq.block_table) * seq.block_size) for seq in active)
        return {
            **self.block_manager.stats_dict(),
            # Allocated slots no token occupies: the tail of each last block.
            "kv_internal_fragmentation": 1 - filled / allocated if allocated else 0.0,
            "num_running": len(self.scheduler.running),
            "num_waiting": len(self.scheduler.waiting),
        }

    @staticmethod
    def start_profile() -> None:
        """Open the cudaProfilerApi capture range (nsys/ncu); no-op off CUDA."""
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStart()

    @staticmethod
    def stop_profile() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()


@torch.inference_mode()
def size_kv_cache(config: NanoVLLMConfig, model) -> int:
    """KV blocks that fit in gpu_memory_utilization of the GPU.

    One eager prefill of max_num_batched_tokens tokens on a scratch cache
    measures the activation peak (and JIT-compiles the prefill kernels); the
    KV cache gets what is left of the budget after everything already on the
    GPU, that peak and KV_RESERVE_BYTES.
    """
    if model.device.type != "cuda":
        return CPU_KV_BLOCKS
    block_size = config.kvcache_block_size
    tokens = min(config.max_num_batched_tokens, config.max_model_len)
    scratch_blocks = -(-tokens // block_size)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    cache = model.new_paged_cache(num_blocks=scratch_blocks, block_size=block_size)
    with_cache = torch.cuda.memory_allocated()
    input_ids, positions, metadata = build_batch(
        input_ids=[0] * tokens,
        positions=list(range(tokens)),
        slots=list(range(tokens)),
        block_tables=[list(range(scratch_blocks))],
        context_lens=[tokens],
        query_lens=[tokens],
        is_prefill=True,
        null_block=cache.null_block,
        num_slots=cache.total_slots,
        device=model.device,
    )
    model.forward_batch(input_ids, positions, cache, metadata)
    torch.cuda.synchronize()
    activation_peak = torch.cuda.max_memory_allocated() - with_cache
    del cache, input_ids, positions, metadata
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    budget = config.gpu_memory_utilization * total - (total - free) - activation_peak - KV_RESERVE_BYTES
    per_block = PagedKVCache.bytes_per_block(
        model.config.num_hidden_layers,
        block_size,
        model.config.num_key_value_heads,
        model.config.head_dim,
        model.dtype,
        model.kv_cache_dtype,
    )
    num_blocks = int(budget // per_block)
    needed = -(-config.max_model_len // block_size)
    if num_blocks < needed:
        raise RuntimeError(
            f"Only {max(num_blocks, 0)} KV blocks fit in gpu_memory_utilization="
            f"{config.gpu_memory_utilization} but one max_model_len sequence needs "
            f"{needed}; raise gpu_memory_utilization or lower max_model_len"
        )
    logger.info(
        "nano-vLLM KV cache: %d blocks (%d tokens, %.2f GiB); %.2f GiB already on the "
        "GPU of %.2f GiB, activation peak %.2f GiB for %d tokens (allocated before: %.2f GiB)",
        num_blocks,
        num_blocks * block_size,
        num_blocks * per_block / 1024**3,
        (total - free) / 1024**3,
        total / 1024**3,
        activation_peak / 1024**3,
        tokens,
        before / 1024**3,
    )
    return num_blocks


def load_model(config: NanoVLLMConfig):
    from .model import NanoLlamaForCausalLM

    return NanoLlamaForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.dtype,
        device=config.device,
        use_triton=config.use_triton,
        attention_backend=config.attention_backend,
        max_model_len=config.max_model_len,
        linear_backend=config.linear_backend,
        weight_quant=config.weight_quant,
        kv_cache_dtype=config.kv_cache_dtype,
        lm_head_quant=config.lm_head_quant,
    )


# --- the core in its own process ---------------------------------------------
#
# Frontend -> core: ("add", EngineRequest), ("abort", [req_id]),
#                   ("utility", call_id, name), ("shutdown",)
# Core -> frontend: ("ready", {"num_blocks": n}), ("outputs", [RequestOutput], stats | None),
#                   ("utility", call_id, result), ("dead", traceback text)

STATS_INTERVAL_S = 0.25


def run_core_process(config: NanoVLLMConfig, inbox, outbox) -> None:
    """Entry point of the engine-core process (multiprocessing spawn target).

    A busy loop: drain the pipe (blocking only when idle), step, send the
    step's outputs. Ctrl-C goes to the frontend, which shuts the core down;
    the core also exits when the frontend's end of the pipe closes.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - core[{os.getpid()}] %(name)s - %(levelname)s - %(message)s",
    )
    try:
        core = EngineCore(config, load_model(config))
        core.capture_graphs()
        outbox.send(("ready", {"num_blocks": core.block_manager.num_blocks}))
        last_stats = 0.0
        while True:
            # Block for the next message only when there is nothing to step.
            while inbox.poll(0 if core.has_work() else None):
                message = inbox.recv()
                kind = message[0]
                if kind == "add":
                    core.add_request(message[1])
                elif kind == "abort":
                    core.abort_requests(message[1])
                elif kind == "utility":
                    outbox.send(("utility", message[1], _utility(core, message[2])))
                elif kind == "shutdown":
                    return
            if not core.has_work():
                continue
            outputs = core.step()
            for output in outputs:
                if output.error is not None:  # a plain exception survives pickling
                    output.error = RuntimeError(f"{type(output.error).__name__}: {output.error}")
            now = time.monotonic()
            idle = not core.has_work()
            stats = None
            if idle or now - last_stats >= STATS_INTERVAL_S:
                stats, last_stats = core.stats(), now
            if outputs or stats is not None:
                outbox.send(("outputs", outputs, stats))
    except (EOFError, BrokenPipeError):
        return  # the frontend went away
    except BaseException:
        try:
            outbox.send(("dead", traceback.format_exc()))
        except Exception:
            pass
        raise


def _utility(core: EngineCore, name: str):
    if name == "stats":
        return core.stats()
    if name == "start_profile":
        core.start_profile()
    elif name == "stop_profile":
        core.stop_profile()
    elif name == "reset_prefix_cache":
        return core.block_manager.reset_prefix_cache()
    return None


def request_from_sequence(seq: Sequence) -> EngineRequest:
    """The core's copy of a frontend sequence (stop strings stay behind)."""
    return EngineRequest(
        req_id=seq.seq_id,
        prompt_token_ids=list(seq.prompt_token_ids),
        sampling=replace(seq.sampling, stop=None),
        eos_token_ids=tuple(sorted(seq.eos_token_ids)),
    )
