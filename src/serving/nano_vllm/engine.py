"""Generation engine wrapper for the nano-vLLM server."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

import torch

from .block_manager import BlockManager
from .config import NanoVLLMConfig
from .kv_cache import PagedKVCache
from .metadata import build_batch
from .model import NanoLlamaForCausalLM
from .model_runner import ModelRunner
from .nvtx import nvtx_range
from .scheduler import Scheduler
from .sampling import SamplingOptions, sample_next_token
from .sequence import GenerationEvent, Sequence, SequenceStatus


logger = logging.getLogger(__name__)

# num_kvcache_blocks: 0 on CPU (nothing to profile).
CPU_KV_BLOCKS = 1024
# GPU memory kept out of the KV budget for the decode CUDA graphs' pool, the
# sampler's fp32 logits/sort buffers and cuBLAS/Triton workspaces.
KV_RESERVE_BYTES = 512 * 1024**2


class InvalidRequestError(ValueError):
    """A request the engine rejects up front (maps to HTTP 400)."""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_ids: list[int]
    finish_reason: str = "stop"


@dataclass
class StreamOutput:
    """One streaming step: new text/tokens plus the running request state."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class IncrementalDetokenizer:
    """Decode a growing token list into text deltas.

    Decoding tokens one at a time breaks multi-byte characters (a CJK char is
    often split across byte-level BPE tokens) and SentencePiece leading spaces.
    Instead decode a short window twice -- with and without the new tokens --
    and emit the difference, holding it back while it ends in U+FFFD (an
    incomplete UTF-8 sequence). The window starts on the prompt's last tokens so
    the first completion token gets the right leading space.
    """

    def __init__(self, tokenizer, prompt_token_ids: list[int], context: int = 5):
        self.tokenizer = tokenizer
        self.token_ids = list(prompt_token_ids[-context:])
        self.prefix_offset = 0
        self.read_offset = len(self.token_ids)

    def add(self, token_id: int) -> str:
        self.token_ids.append(token_id)
        prefix_text, new_text = self._decode_window()
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            return self._advance(new_text[len(prefix_text) :])
        return ""

    def flush(self) -> str:
        """Emit whatever is still held back, even an incomplete character."""
        if self.read_offset == len(self.token_ids):
            return ""
        prefix_text, new_text = self._decode_window()
        return self._advance(new_text[len(prefix_text) :])

    def _advance(self, delta: str) -> str:
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.token_ids)
        return delta

    def _decode_window(self) -> tuple[str, str]:
        return (
            self._decode(self.token_ids[self.prefix_offset : self.read_offset]),
            self._decode(self.token_ids[self.prefix_offset :]),
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class _OutputProcessor:
    """Token ids -> text deltas, honoring stop strings while streaming.

    Up to ``len(longest stop) - 1`` trailing characters are withheld until it
    is clear they do not begin a stop string, so a stop string never leaks
    into the stream and generation halts as soon as one completes.
    """

    def __init__(self, tokenizer, prompt_token_ids: list[int], stop: Iterable[str]):
        self.detokenizer = IncrementalDetokenizer(tokenizer, prompt_token_ids)
        self.stop = [marker for marker in stop or () if marker]
        self.hold_back = max((len(marker) for marker in self.stop), default=1) - 1
        self.text = ""
        self.sent = 0
        self.token_ids: list[int] = []
        self.finish_reason: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def add_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self._add_text(self.detokenizer.add(token_id))

    def finish(self, finish_reason: Optional[str]) -> None:
        if self.finished:
            return
        self._add_text(self.detokenizer.flush())
        if not self.finished:
            self.finish_reason = finish_reason or "stop"

    def take_delta(self) -> str:
        if self.finished:
            limit = len(self.text)
        else:
            limit = max(self.sent, len(self.text) - self.hold_back)
        delta = self.text[self.sent : limit]
        self.sent = limit
        return delta

    def _add_text(self, delta: str) -> None:
        if not delta:
            return
        search_from = max(0, len(self.text) - self.hold_back)
        self.text += delta
        cutoff = None
        for marker in self.stop:
            index = self.text.find(marker, search_from)
            if index >= 0:
                cutoff = index if cutoff is None else min(cutoff, index)
        if cutoff is not None:
            self.text = self.text[:cutoff]
            self.finish_reason = "stop"


class NanoLlamaEngine:
    """Single-model decoder with sync fallback and async continuous batching."""

    def __init__(self, config: NanoVLLMConfig):
        self.config = config
        self.model: Optional[NanoLlamaForCausalLM] = None
        self.tokenizer = None
        self.initialized = False
        self.block_manager: Optional[BlockManager] = None
        self.scheduler: Optional[Scheduler] = None
        self.paged_cache: Optional[PagedKVCache] = None
        self.model_runner: Optional[ModelRunner] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_event: Optional[asyncio.Event] = None
        self._scheduler_lock: Optional[asyncio.Lock] = None
        self._stopping = False
        self._eos_token_ids: Optional[frozenset[int]] = None
        # Sequences inside the running model step; aborting one of them only
        # marks it, and postprocess() frees its blocks once the step returns.
        self._inflight_seq_ids: set[int] = set()
        self._step_count = 0

    def initialize(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_path = self.config.tokenizer or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = NanoLlamaForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=self.config.dtype,
            device=self.config.device,
            use_triton=self.config.use_triton,
            attention_backend=self.config.attention_backend,
            max_model_len=self.config.max_model_len,
            linear_backend=self.config.linear_backend,
            weight_quant=self.config.weight_quant,
        )
        self._initialize_batching_state()
        if not self.config.enforce_eager and self.model.device.type == "cuda":
            self.model_runner.capture_decode_graphs(
                max_batch=min(self.config.max_num_seqs, self.config.cuda_graph_max_bs),
                max_model_len=self.config.max_model_len,
            )
        logger.info("nano-vLLM EOS token ids: %s", sorted(self.eos_token_ids))
        self.initialized = True

    async def start(self) -> None:
        if not self.initialized:
            self.initialize()
        self.start_background_loop()

    async def shutdown(self) -> None:
        self._stopping = True
        if self._scheduler_event is not None:
            self._scheduler_event.set()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        if self.scheduler is not None:
            pending = [*self.scheduler.running, *self.scheduler.waiting]
            self._fail_sequences(pending, RuntimeError("nano-vLLM engine shut down"))

    def start_background_loop(self) -> None:
        self._ensure_batching_state()
        if self._scheduler_task is not None and not self._scheduler_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._stopping = False
        self._scheduler_task = loop.create_task(self._scheduler_loop())

    @property
    def eos_token_ids(self) -> frozenset[int]:
        """EOS ids from generation_config.json, config.json and the tokenizer.

        Llama 3 instruct lists several (128001 <|end_of_text|> and 128009
        <|eot_id|>), and tokenizer.eos_token_id alone misses the turn end.
        """
        if self._eos_token_ids is None:
            ids: list[int] = []
            generation_config = Path(self.config.model_name) / "generation_config.json"
            if generation_config.is_file():
                data = json.loads(generation_config.read_text(encoding="utf-8"))
                ids.extend(_token_id_list(data.get("eos_token_id")))
            model_config = getattr(self.model, "config", None)
            ids.extend(_token_id_list(getattr(model_config, "eos_token_id", None)))
            ids.extend(
                _token_id_list(getattr(self.tokenizer, "eos_token_id", None))
            )
            self._eos_token_ids = frozenset(ids)
        return self._eos_token_ids

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling: SamplingOptions,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Single-request decode on the static KV cache (v0.1 reference path)."""
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("nano-vLLM engine is not initialized")

        prompt_ids = self._encode(prompt)
        prompt_tokens = len(prompt_ids)
        max_tokens = self._resolve_max_tokens(prompt_tokens, sampling.max_tokens)
        stop_token_ids = set(sampling.stop_token_ids or ())
        if not sampling.ignore_eos:
            stop_token_ids |= self.eos_token_ids
        processor = _OutputProcessor(self.tokenizer, prompt_ids, sampling.stop or [])
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.device)
        cache = self.model.new_cache(
            batch_size=1, max_seq_len=prompt_tokens + max_tokens
        )

        logits = self.model.forward(input_ids, cache=cache, start_pos=0)
        current_pos = prompt_tokens
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(seed)

        finish_reason = "length"
        for step in range(max_tokens):
            next_token = sample_next_token(
                logits[0, -1],
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                generator=generator,
            )
            processor.add_token(next_token)
            if processor.finished:
                break
            if next_token in stop_token_ids:
                finish_reason = "stop"
                break
            if step == max_tokens - 1:
                break
            token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.model.device
            )
            logits = self.model.forward(
                token_tensor, cache=cache, start_pos=current_pos
            )
            current_pos += 1

        processor.finish(finish_reason)
        return GenerationResult(
            text=processor.take_delta(),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(processor.token_ids),
            token_ids=list(processor.token_ids),
            finish_reason=processor.finish_reason,
        )

    def prepare_request(self, prompt: str, sampling: SamplingOptions) -> Sequence:
        """Tokenize and validate a request without submitting it.

        Raises InvalidRequestError before any response bytes are sent, so the
        server can still answer 400 for streaming requests.
        """
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("nano-vLLM engine is not initialized")
        prompt_ids = self._encode(prompt)
        self._ensure_batching_state()
        kv_capacity = self.block_manager.num_blocks * self.block_manager.block_size
        max_tokens = self._resolve_max_tokens(
            len(prompt_ids), sampling.max_tokens, kv_capacity=kv_capacity
        )
        seq = Sequence(
            prompt_token_ids=prompt_ids,
            sampling=replace(sampling, max_tokens=max_tokens),
            eos_token_id=self.eos_token_ids,
            ignore_eos=sampling.ignore_eos,
        )
        seq.events = asyncio.Queue()
        return seq

    async def stream_request(self, seq: Sequence) -> AsyncIterator[StreamOutput]:
        """Submit a prepared sequence and yield its outputs.

        Closing or cancelling the iterator (client disconnect) aborts the
        sequence and returns its KV blocks; so does hitting a stop string.
        """
        processor = _OutputProcessor(
            self.tokenizer, seq.prompt_token_ids, seq.sampling.stop or []
        )
        self.start_background_loop()
        self._submit_sequence(seq)
        try:
            while not processor.finished:
                event = await seq.events.get()
                if event.error is not None:
                    raise event.error
                new_token_ids = []
                with nvtx_range("detokenize"):
                    if event.token_id is not None:
                        processor.add_token(event.token_id)
                        new_token_ids.append(event.token_id)
                    if event.finished:
                        processor.finish(event.finish_reason)
                yield StreamOutput(
                    text=processor.take_delta(),
                    token_ids=new_token_ids,
                    finished=processor.finished,
                    finish_reason=processor.finish_reason,
                    prompt_tokens=seq.num_prompt_tokens,
                    completion_tokens=len(processor.token_ids),
                )
        finally:
            if not seq.is_finished:
                self.abort(seq, finish_reason=processor.finish_reason or "abort")

    async def generate_async(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ) -> GenerationResult:
        seq = self.prepare_request(prompt, sampling)
        pieces: list[str] = []
        token_ids: list[int] = []
        finish_reason = "stop"
        async with aclosing(self.stream_request(seq)) as outputs:
            async for output in outputs:
                pieces.append(output.text)
                token_ids.extend(output.token_ids)
                if output.finished:
                    finish_reason = output.finish_reason or "stop"
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=seq.num_prompt_tokens,
            completion_tokens=len(token_ids),
            token_ids=token_ids,
            finish_reason=finish_reason,
        )

    async def generate_stream_async(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ) -> AsyncIterator[StreamOutput]:
        seq = self.prepare_request(prompt, sampling)
        async with aclosing(self.stream_request(seq)) as outputs:
            async for output in outputs:
                yield output

    def start_profile(self) -> None:
        """Open the cudaProfilerApi capture range (nsys/ncu); no-op off CUDA."""
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStart()

    def stop_profile(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    def abort(self, seq: Sequence, finish_reason: str = "abort") -> None:
        """Stop generating for ``seq`` and release its KV blocks."""
        if seq.is_finished:
            return
        if seq.seq_id in self._inflight_seq_ids or self.scheduler is None:
            seq.status = SequenceStatus.FINISHED
            seq.finish_reason = seq.finish_reason or finish_reason
            return
        self.scheduler.abort(seq, finish_reason)

    def _encode(self, prompt: str) -> list[int]:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = [int(token_id) for token_id in encoded["input_ids"][0].tolist()]
        if not prompt_ids:
            raise InvalidRequestError("Prompt produced no tokens")
        return prompt_ids

    def _resolve_max_tokens(
        self,
        num_prompt_tokens: int,
        requested: Optional[int],
        kv_capacity: Optional[int] = None,
    ) -> int:
        """Validate lengths; never truncate the prompt or clamp max_tokens."""
        max_model_len = self.config.max_model_len
        if num_prompt_tokens >= max_model_len:
            raise InvalidRequestError(
                f"Prompt has {num_prompt_tokens} tokens but max_model_len is "
                f"{max_model_len}, leaving no room to generate"
            )
        limit = max_model_len
        if kv_capacity is not None:
            if num_prompt_tokens >= kv_capacity:
                raise InvalidRequestError(
                    f"Prompt has {num_prompt_tokens} tokens but the KV cache "
                    f"holds only {kv_capacity}"
                )
            limit = min(limit, kv_capacity)
        if requested is None:
            return min(self.config.max_new_tokens, limit - num_prompt_tokens)
        if requested < 1:
            raise InvalidRequestError("max_tokens must be at least 1")
        if requested > self.config.max_new_tokens:
            raise InvalidRequestError(
                f"max_tokens={requested} exceeds this server's limit of "
                f"{self.config.max_new_tokens} (engine.max_new_tokens / "
                "MAX_NEW_TOKENS)"
            )
        if num_prompt_tokens + requested > max_model_len:
            raise InvalidRequestError(
                f"This model's maximum context length is {max_model_len} tokens, "
                f"but {num_prompt_tokens + requested} were requested "
                f"({num_prompt_tokens} prompt + {requested} max_tokens)"
            )
        if num_prompt_tokens + requested > limit:
            raise InvalidRequestError(
                f"Request needs up to {num_prompt_tokens + requested} KV slots "
                f"but the KV cache holds only {kv_capacity} "
                "(num_kvcache_blocks * kvcache_block_size)"
            )
        return requested

    def _initialize_batching_state(self) -> None:
        if self.model is None:
            return
        if self.config.num_kvcache_blocks == 0:
            self.config.num_kvcache_blocks = self._size_kv_cache()
        Sequence.block_size = self.config.kvcache_block_size
        self.block_manager = BlockManager(
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.kvcache_block_size,
            enable_prefix_cache=self.config.enable_prefix_cache,
        )
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            policy=self.config.scheduling,
        )
        self.paged_cache = self.model.new_paged_cache(
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.kvcache_block_size,
        )
        self.model_runner = ModelRunner(self.model, self.paged_cache, self.block_manager)

    @torch.inference_mode()
    def _size_kv_cache(self) -> int:
        """KV blocks that fit in gpu_memory_utilization of the GPU.

        One eager prefill of max_num_batched_tokens tokens on a scratch cache
        measures the activation peak (and JIT-compiles the prefill kernels);
        the KV cache gets what is left of the budget after everything already
        on the GPU, that peak and KV_RESERVE_BYTES.
        """
        config, model = self.config, self.model
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
        budget = (
            config.gpu_memory_utilization * total
            - (total - free)
            - activation_peak
            - KV_RESERVE_BYTES
        )
        per_block = PagedKVCache.bytes_per_block(
            model.config.num_hidden_layers,
            block_size,
            model.config.num_key_value_heads,
            model.config.head_dim,
            model.dtype,
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

    def stats(self) -> dict:
        """KV block usage, prefix-cache counters and queue lengths."""
        if self.block_manager is None or self.scheduler is None:
            return {}
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

    def _ensure_batching_state(self) -> None:
        if (
            self.scheduler is None
            or self.block_manager is None
            or self.paged_cache is None
        ):
            self._initialize_batching_state()
        if (
            self.scheduler is None
            or self.block_manager is None
            or self.paged_cache is None
        ):
            raise RuntimeError("nano-vLLM batching state is not initialized")
        if self._scheduler_event is None:
            self._scheduler_event = asyncio.Event()
        if self._scheduler_lock is None:
            self._scheduler_lock = asyncio.Lock()

    def _submit_sequence(self, seq: Sequence) -> None:
        self._ensure_batching_state()
        self.scheduler.add(seq)
        self._scheduler_event.set()

    async def _scheduler_loop(self) -> None:
        self._ensure_batching_state()
        while not self._stopping:
            if self.scheduler.is_finished():
                self._scheduler_event.clear()
                await self._scheduler_event.wait()
                continue
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - _step already fails requests
                logger.exception("nano-vLLM scheduler step failed")
                await asyncio.sleep(0)

    async def _step(self) -> None:
        """Run one schedule -> model -> postprocess round.

        Exceptions fail the affected requests instead of escaping: a dead
        scheduler task would leave every current and future request hanging.
        """
        self._step_count += 1
        with nvtx_range(f"engine_step_{self._step_count}"):
            await self._step_inner()

    async def _step_inner(self) -> None:
        async with self._scheduler_lock:
            try:
                with nvtx_range("schedule"):
                    seqs, is_prefill = self.scheduler.schedule()
            except Exception as exc:
                # Scheduler state for queued requests can't be trusted anymore.
                logger.exception("nano-vLLM schedule() failed; failing queued requests")
                self._fail_sequences(
                    [*self.scheduler.running, *self.scheduler.waiting], exc
                )
                return
            if not seqs:
                stuck = self.scheduler.unschedulable_head()
                if stuck is not None:
                    self._fail_sequences(
                        [stuck],
                        RuntimeError(
                            "Request cannot be scheduled: it needs more KV blocks "
                            "than are free with nothing left to release"
                        ),
                    )
            else:
                self._inflight_seq_ids = {seq.seq_id for seq in seqs}
        if not seqs:
            await asyncio.sleep(0)
            return

        try:
            token_ids = await asyncio.to_thread(
                self._run_scheduled_batch,
                seqs,
                is_prefill,
            )
            async with self._scheduler_lock:
                with nvtx_range("postprocess"):
                    emitted = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        except Exception as exc:
            self._fail_sequences(seqs, exc)
            return
        finally:
            self._inflight_seq_ids = set()
        with nvtx_range("publish"):
            self._publish_events(emitted)

    def _run_scheduled_batch(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if self.model_runner is None:
            raise RuntimeError("nano-vLLM engine is not initialized")
        return self.model_runner.run(seqs, is_prefill)

    def _publish_events(self, emitted: list[tuple[Sequence, int]]) -> None:
        for seq, token_id in emitted:
            if seq.events is not None:
                seq.events.put_nowait(
                    GenerationEvent(
                        token_id=token_id,
                        finished=seq.is_finished,
                        finish_reason=seq.finish_reason if seq.is_finished else None,
                    )
                )

    def _fail_sequences(
        self,
        seqs: list[Sequence],
        exc: BaseException,
    ) -> None:
        for seq in seqs:
            try:
                if self.scheduler is not None:
                    self.scheduler.abort(seq, finish_reason="error")
                else:
                    seq.status = SequenceStatus.FINISHED
            except Exception:  # still notify the waiting request below
                logger.exception("Failed to release nano-vLLM sequence %s", seq.seq_id)
            if seq.events is not None:
                seq.events.put_nowait(GenerationEvent(finished=True, error=exc))


def _token_id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(token_id) for token_id in value]
