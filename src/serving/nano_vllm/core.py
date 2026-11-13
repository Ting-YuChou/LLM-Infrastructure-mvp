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

With a proposer (speculative decoding, spec.py) every step drafts as many
tokens per decode row as :class:`~.spec.SpecPolicy` chooses; a step that
drafts none is a plain step, overlapped as above. How far a cycle moves a
sequence is known only when it is collected, but a cycle that ran as one
CUDA graph leaves that on the device, so with ``async_scheduling`` the next
cycle is launched behind it as well (:meth:`EngineCore._chains`): the
scheduler gives its rows KV slots for wherever the cycle in flight may end,
and a row whose sequence that cycle finishes is discarded. Any other step
after a cycle (a plain one, one with prompt rows) waits for its tokens.
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
from .spec import DFlashProposer, Proposer, SpecPolicy

logger = logging.getLogger(__name__)

# num_kvcache_blocks: 0 on CPU (nothing to profile).
CPU_KV_BLOCKS = 1024
# GPU memory kept out of the KV budget for the decode CUDA graphs' pool, the
# sampler's buffers (fp32 logits and a sort on its PyTorch path) and
# cuBLAS/Triton workspaces.
KV_RESERVE_BYTES = 512 * 1024**2
# The host goes from one step's tokens to the next step's launch in well under
# this; a longer gap means the engine waited for requests.
IDLE_GAP_S = 0.005


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

    def __init__(
        self, config: NanoVLLMConfig, model, proposer: Optional[Proposer] = None
    ) -> None:
        """``proposer``: draft tokens for speculative decoding from something
        other than config.speculative_model (tests script one)."""
        self.config = config
        self.model = model
        drafter = None
        if proposer is None and config.speculative_model:
            drafter = load_drafter(config, model)
        if config.num_kvcache_blocks == 0:
            config.num_kvcache_blocks = size_kv_cache(config, model, drafter)
        if drafter is not None:
            proposer = DFlashProposer(
                drafter,
                drafter.new_paged_cache(config.num_kvcache_blocks, config.kvcache_block_size),
                config.num_speculative_tokens,
            )
        self.proposer = proposer
        num_draft_tokens = proposer.num_draft_tokens if proposer is not None else 0
        # The verify lengths a step can run: the whole block, and under the
        # adaptive policy the configured shorter ones (1: no drafts).
        verify_lens = {num_draft_tokens + 1}
        if config.speculative_policy == "adaptive":
            verify_lens.update(
                length for length in config.speculative_verify_lens if length <= num_draft_tokens
            )
        self.spec_policy = SpecPolicy(tuple(verify_lens)) if proposer is not None else None
        # Decode rows run at each verify length (a row per sequence and step).
        self.spec_rows = dict.fromkeys(sorted(verify_lens), 0) if proposer is not None else {}
        # A cycle can follow one still in flight when both are one graph.
        chain_cycles = (
            config.async_scheduling
            and config.speculative_chain
            and getattr(proposer, "device_only", False)
        )
        if proposer is not None and hasattr(model, "extend_rope_table"):
            # A verify pass reaches num_draft_tokens positions past its
            # anchor, which a cycle in flight leaves up to as many past the
            # last token the host knows.
            model.extend_rope_table(config.max_model_len + 2 * num_draft_tokens)
            if drafter is not None:
                drafter.extend_rope_table(config.max_model_len + 2 * num_draft_tokens)
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
            num_draft_tokens=num_draft_tokens,
            chain_cycles=chain_cycles,
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
            proposer=proposer,
            draft_temperature=config.speculative_draft_temperature,
            verify_lens=tuple(verify_lens),
        )
        self.async_scheduling = config.async_scheduling
        self.chain_cycles = chain_cycles
        self.requests: dict[int, Sequence] = {}
        self._inbox: deque[tuple[str, object]] = deque()
        # The launched step not collected yet: its handle, rows and launch time.
        self._inflight: Optional[tuple[StepHandle, list[Row], float]] = None
        self._last_collected = 0.0  # when the last step's tokens arrived
        self._step_count = 0

    def capture_graphs(self) -> None:
        """Before serving on CUDA: compile the eager step's kernel variants,
        then capture the decode CUDA graphs, with a proposer the speculative
        cycle's, and the piecewise graphs for the other steps (unless
        enforce_eager). The speculative policy gets each graph's step time."""
        if self.model.device.type != "cuda":
            return
        self.model_runner.warm_up_eager(self.config.max_num_seqs)
        if self.config.enforce_eager:
            return
        max_batch = min(self.config.max_num_seqs, self.config.cuda_graph_max_bs)
        # With a proposer: only if a step may run without drafts.
        if self.spec_policy is None or 1 in self.spec_policy.verify_lens:
            self.model_runner.capture_decode_graphs(
                max_batch=max_batch, max_model_len=self.config.max_model_len
            )
        max_tokens = min(self.config.cuda_graph_max_tokens, self.config.max_num_batched_tokens)
        if max_tokens > 0:
            self.model_runner.capture_piecewise_graphs(max_tokens, max_rows=self.config.max_num_seqs)
        if self.proposer is not None:
            self.model_runner.capture_spec_graphs(
                max_batch=max_batch, max_model_len=self.config.max_model_len
            )
            if len(self.spec_policy.verify_lens) > 1:
                self.spec_policy.set_times(self.model_runner.profile_graphs())

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
        draft_tokens, step = 0, {}
        chained = False
        if self.spec_policy is not None:
            # The decode rows the step will have, but for a preemption.
            decode_rows = min(len(self.scheduler.running), self.config.max_num_seqs)
            draft_tokens = self.spec_policy.choose(decode_rows) - 1
            step = {"draft_tokens": draft_tokens}
            cycle_inflight = self._inflight is not None and self._inflight[0].spec_seqs > 0
            chained = bool(draft_tokens) and cycle_inflight and self._chains(decode_rows, draft_tokens + 1)
            if (draft_tokens or cycle_inflight) and not chained:
                # A cycle starts from tokens the host knows, unless it
                # follows one in flight on the device; a plain step never
                # follows a cycle in flight.
                outputs += self._collect_inflight()
        overlap = self.async_scheduling and (not draft_tokens or chained)
        try:
            with nvtx_range("schedule"):
                seqs, has_prefill = self.scheduler.schedule(**step)
        except Exception as exc:
            # Scheduler state for queued requests can't be trusted anymore.
            logger.exception("nano-vLLM schedule() failed; failing queued requests")
            return outputs + self._collect_inflight() + self._fail(
                [*self.scheduler.running, *self.scheduler.waiting], exc
            )
        launched = None
        if seqs:
            launched, failed = self._launch(seqs, has_prefill, draft_tokens, overlap)
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
        if overlap:
            outputs += self._collect_inflight()
            self._inflight = launched
        elif launched is not None:
            if self.chain_cycles and launched[0].chains:
                # Nothing is in flight: the next cycle may follow this one.
                self._inflight = launched
            else:
                outputs += self._complete(*launched)
        return outputs

    def _chains(self, decode_rows: int, block: int) -> bool:
        """Whether a cycle of ``block`` for up to ``decode_rows`` sequences
        can be launched behind the cycle in flight: it must be one graph (no
        prompt rows, which need the host's token ids, and a size the graphs
        cover), and it is not worth a GPU step when every row of the cycle
        in flight may be its sequence's last."""
        if not self.chain_cycles or self.scheduler.waiting:
            return False
        if not self.model_runner.has_spec_graph(decode_rows, block):
            return False
        return not all(
            row.seq.is_finished
            or row.seq.num_completion_tokens + row.drafted + 1 >= row.seq.sampling.max_tokens
            for row in self._inflight[1]
        )

    def _launch(
        self, seqs: list[Sequence], has_prefill: bool, draft_tokens: int = 0, overlap: bool = False
    ) -> tuple[Optional[tuple[StepHandle, list[Row], float]], list[RequestOutput]]:
        # A decode row whose input token is still in flight reads it from the
        # previous step's row on the device (a cycle's row: where the cycle
        # in flight leaves its sequence).
        gather = None
        if overlap:
            gather = [
                seq.pending_row if not seq.is_prefill and seq.num_inflight else -1
                for seq in seqs
            ]
        start = time.perf_counter()
        try:
            handle = self.model_runner.launch(
                seqs, has_prefill, gather, draft_tokens if self.proposer is not None else None
            )
        except Exception as exc:
            logger.exception("nano-vLLM model step failed")
            return None, self._fail(seqs, exc)
        rows = self.scheduler.launched(seqs)
        for index, row in enumerate(rows):
            if row.samples:
                row.seq.pending_row = index
        if self.proposer is not None:
            self.spec_rows[draft_tokens + 1] += sum(not seq.is_prefill for seq in seqs)
        return (handle, rows, start), []

    def _collect_inflight(self) -> list[RequestOutput]:
        if self._inflight is None:
            return []
        inflight, self._inflight = self._inflight, None
        return self._complete(*inflight)

    def _complete(self, handle: StepHandle, rows: list[Row], launched: float) -> list[RequestOutput]:
        try:
            token_ids = self.model_runner.collect(handle)
            if self.spec_policy is not None:
                self._observe_step(handle, rows, token_ids, launched)
            with nvtx_range("postprocess"):
                emitted = self.scheduler.complete(rows, token_ids)
        except Exception as exc:
            logger.exception("nano-vLLM model step failed")
            return self._fail([row.seq for row in rows if not row.seq.is_finished], exc)
        outputs = []
        # A speculative row emits several tokens: the finish goes on the last.
        last = {seq.seq_id: index for index, (seq, _) in enumerate(emitted)}
        for index, (seq, token_id) in enumerate(emitted):
            ends = seq.is_finished and last[seq.seq_id] == index
            finish = seq.finish_reason if ends else None
            outputs.append(RequestOutput(seq.seq_id, token_id, finish))
            if finish is not None:
                self.requests.pop(seq.seq_id, None)
        return outputs

    def _observe_step(
        self, handle: StepHandle, rows: list[Row], token_ids: list, launched: float
    ) -> None:
        """What a collected step tells the speculative policy: the drafts
        each cycle accepted (but for the rows about to be discarded, which
        ran past their sequence's end), and what a step that was one graph
        took.

        A step's time runs from the collection of the step before it: an
        overlapped step starts on the GPU then, and for the others it adds
        the host's work between two steps (finishing one, scheduling the
        next), which a cycle pays once for all its tokens. After an idle
        spell it runs from the launch.
        """
        now = time.perf_counter()
        idle = launched - self._last_collected > IDLE_GAP_S
        started, self._last_collected = (launched if idle else self._last_collected), now
        policy = self.spec_policy
        if handle.spec_seqs:
            policy.update(
                handle.block - 1,
                [
                    len(tokens) - 1
                    for row, tokens in zip(rows, token_ids[: handle.spec_seqs])
                    if not row.seq.is_finished
                ],
            )
        if handle.graph_size:
            policy.record(handle.graph_size, handle.block, (now - started) * 1e3)

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
        spec = {}
        if self.proposer is not None:
            # As vLLM counts them: cycles, and the drafts the target accepted.
            spec = {
                "spec_num_drafts": self.scheduler.spec_cycles,
                "spec_num_draft_tokens": self.scheduler.spec_draft_tokens,
                "spec_num_accepted_tokens": self.scheduler.spec_accepted_tokens,
                # Cycles launched behind one that turned out to end their sequence.
                "spec_num_discarded": self.scheduler.spec_discarded,
                # Decode rows by the verify length of their step (1: no drafts).
                **{f"spec_rows_len_{length}": rows for length, rows in self.spec_rows.items()},
            }
        return {
            **self.block_manager.stats_dict(),
            **spec,
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
def size_kv_cache(config: NanoVLLMConfig, model, drafter=None) -> int:
    """KV blocks that fit in gpu_memory_utilization of the GPU.

    With a ``drafter`` (already loaded) every block also holds its layers'
    K/V in the drafter's pool.

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
    graph_pool = 0
    if drafter is not None:
        # A speculative cycle has logits for every verify and draft row. The
        # cycle's CUDA graphs keep one cycle's activations in their pool for
        # good; a step they do not cover needs the same again while it runs.
        spec_peak = _spec_cycle_peak(config, model, drafter)
        activation_peak = max(activation_peak, spec_peak)
        graph_pool = 0 if config.enforce_eager else spec_peak
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    budget = (
        config.gpu_memory_utilization * total
        - (total - free) - activation_peak - graph_pool - KV_RESERVE_BYTES
    )
    per_block = PagedKVCache.bytes_per_block(
        model.config.num_hidden_layers,
        block_size,
        model.config.num_key_value_heads,
        model.config.head_dim,
        model.dtype,
        model.kv_cache_dtype,
    )
    if drafter is not None:
        per_block += PagedKVCache.bytes_per_block(
            drafter.config.num_hidden_layers,
            block_size,
            drafter.config.num_key_value_heads,
            drafter.config.head_dim,
            drafter.dtype,
            drafter.kv_cache_dtype,
        )
    num_blocks = int(budget // per_block)
    needed = -(-config.max_model_len // block_size)
    if num_blocks < needed:
        raise RuntimeError(
            f"Only {max(num_blocks, 0)} KV blocks fit in gpu_memory_utilization="
            f"{config.gpu_memory_utilization} but one max_model_len sequence needs "
            f"{needed}; raise gpu_memory_utilization or lower max_model_len"
        )
    # Of what is already on the GPU, PyTorch holds the reserved part; reserved
    # but not allocated is cached blocks in segments empty_cache could not release.
    logger.info(
        "nano-vLLM KV cache: %d blocks (%d tokens, %.2f GiB); %.2f GiB already on the "
        "GPU of %.2f GiB (PyTorch reserved %.2f, allocated %.2f), activation peak %.2f GiB "
        "for %d tokens (allocated before: %.2f GiB)",
        num_blocks,
        num_blocks * block_size,
        num_blocks * per_block / 1024**3,
        (total - free) / 1024**3,
        total / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
        torch.cuda.memory_allocated() / 1024**3,
        activation_peak / 1024**3,
        tokens,
        before / 1024**3,
    )
    return num_blocks


def _spec_cycle_peak(config: NanoVLLMConfig, model, drafter) -> int:
    """Peak activation bytes of one eager speculative cycle of max_num_seqs
    sequences, on scratch caches (it also compiles that cycle's kernels)."""
    block_size = config.kvcache_block_size
    proposer = DFlashProposer(
        drafter, drafter.new_paged_cache(1, block_size), config.num_speculative_tokens
    )
    runner = ModelRunner(
        model, model.new_paged_cache(num_blocks=1, block_size=block_size), None,
        max_rows=config.max_num_seqs, proposer=proposer,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    runner.null_spec_cycle(config.max_num_seqs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


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


def load_drafter(config: NanoVLLMConfig, model):
    """config.speculative_model next to the loaded target ``model``."""
    from .dflash import DFlashDrafter

    drafter = DFlashDrafter.from_pretrained(
        config.speculative_model, model, weight_quant=config.speculative_weight_quant
    )
    model.check_tap_layers(tuple(drafter.config.target_layer_ids))
    return drafter


# --- the core in its own process ---------------------------------------------
#
# Frontend -> core: ("add", EngineRequest), ("abort", [req_id]),
#                   ("utility", call_id, name), ("shutdown",)
# Core -> frontend: ("ready", {"num_blocks": n, "lookahead": k}), ("outputs", [RequestOutput], stats | None),
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
        outbox.send((
            "ready",
            {
                "num_blocks": core.block_manager.num_blocks,
                "lookahead": core.scheduler.max_lookahead,
            },
        ))
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
