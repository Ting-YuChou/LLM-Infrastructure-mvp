"""Generation engine frontend for the nano-vLLM server."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import threading
from contextlib import aclosing
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

import torch

from .config import NanoVLLMConfig
from .core import (
    CPU_KV_BLOCKS,
    KV_RESERVE_BYTES,
    EngineCore,
    RequestOutput,
    load_model,
    request_from_sequence,
    run_core_process,
    size_kv_cache,
)
from .nvtx import nvtx_range
from .sampling import SamplingOptions, sample_next_token
from .sequence import GenerationEvent, Sequence, SequenceStatus


logger = logging.getLogger(__name__)

__all__ = [
    "CPU_KV_BLOCKS",
    "KV_RESERVE_BYTES",
    "GenerationResult",
    "InvalidRequestError",
    "NanoLlamaEngine",
    "StreamOutput",
    "size_kv_cache",
]


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
    """Frontend of the nano-vLLM engine around an :class:`EngineCore`.

    The frontend tokenizes, validates, detokenizes and applies stop strings;
    the core schedules and runs the model on token ids. By default the core
    lives in this process and its steps run in a worker thread; with
    ``engine.multiprocess`` it runs in a process of its own
    (:class:`_CoreProcess`), so detokenization, HTTP and SSE never compete
    with the model steps for the GIL.
    """

    def __init__(self, config: NanoVLLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.core: Optional[EngineCore] = None
        self._core_process: Optional[_CoreProcess] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_event: Optional[asyncio.Event] = None
        self._step_future: Optional[asyncio.Future] = None
        self._stopping = False
        self._eos_token_ids: Optional[frozenset[int]] = None
        # The frontend's view of every request the core is still working on.
        self._requests: dict[int, Sequence] = {}

    # The in-process core's parts (tests and benchmarks look at them).
    @property
    def scheduler(self):
        return self.core.scheduler if self.core is not None else None

    @property
    def block_manager(self):
        return self.core.block_manager if self.core is not None else None

    @property
    def paged_cache(self):
        return self.core.paged_cache if self.core is not None else None

    @property
    def model_runner(self):
        return self.core.model_runner if self.core is not None else None

    def initialize(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_path = self.config.tokenizer or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.config.multiprocess:
            self._core_process = _CoreProcess(self.config)
            self._core_process.start()
        else:
            self.model = load_model(self.config)
            self._initialize_batching_state()
            self.core.capture_graphs()
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
        if self._step_future is not None:
            # A step already running in the worker thread cannot be cancelled;
            # let it finish before touching the core.
            await asyncio.wait([self._step_future])
            self._step_future = None
        error = RuntimeError("nano-vLLM engine shut down")
        if self.core is not None:
            self._dispatch(self.core.fail_all(error))
        if self._core_process is not None:
            await asyncio.to_thread(self._core_process.shutdown)
        self._fail_frontend_requests(error)

    def start_background_loop(self) -> None:
        self._ensure_batching_state()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._core_process is not None:
            self._core_process.start_reader(loop, self._on_core_message)
            return
        if self._scheduler_task is not None and not self._scheduler_task.done():
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
            for name in ("generation_config.json", "config.json"):
                path = Path(self.config.model_name) / name
                if path.is_file():
                    data = json.loads(path.read_text(encoding="utf-8"))
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
            raise RuntimeError(
                "nano-vLLM's static path needs an initialized in-process engine"
            )

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
        if not self.initialized or self.tokenizer is None:
            raise RuntimeError("nano-vLLM engine is not initialized")
        prompt_ids = self._encode(prompt)
        self._ensure_batching_state()
        max_tokens = self._resolve_max_tokens(
            len(prompt_ids), sampling.max_tokens, kv_capacity=self.kv_capacity
        )
        seq = Sequence(
            prompt_token_ids=prompt_ids,
            sampling=replace(sampling, max_tokens=max_tokens),
            eos_token_id=self.eos_token_ids,
            ignore_eos=sampling.ignore_eos,
        )
        seq.events = asyncio.Queue()
        return seq

    @property
    def kv_capacity(self) -> int:
        """KV slots in the core's cache."""
        if self.core is not None:
            return self.core.block_manager.num_blocks * self.core.block_manager.block_size
        if self._core_process is not None:
            return self._core_process.num_blocks * self.config.kvcache_block_size
        return self.config.num_kvcache_blocks * self.config.kvcache_block_size

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
        """Open the cudaProfilerApi capture range in the core (nsys/ncu)."""
        if self._core_process is not None:
            self._core_process.utility("start_profile")
        else:
            EngineCore.start_profile()

    def stop_profile(self) -> None:
        if self._core_process is not None:
            self._core_process.utility("stop_profile")
        else:
            EngineCore.stop_profile()

    def stats(self) -> dict:
        """KV block usage, prefix-cache counters and queue lengths (see EngineCore)."""
        if self._core_process is not None:
            return dict(self._core_process.stats)
        return self.core.stats() if self.core is not None else {}

    def abort(self, seq: Sequence, finish_reason: str = "abort") -> None:
        """Stop generating for ``seq``; the core frees its KV blocks."""
        if seq.is_finished:
            return
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = seq.finish_reason or finish_reason
        if self._requests.pop(seq.seq_id, None) is None:
            return
        if self._core_process is not None:
            self._core_process.send(("abort", [seq.seq_id]))
        elif self.core is not None:
            self.core.abort_requests([seq.seq_id])
            self._wake()

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
        self.core = EngineCore(self.config, self.model)

    def _ensure_batching_state(self) -> None:
        if self.core is None and self._core_process is None:
            self._initialize_batching_state()
        if self.core is None and self._core_process is None:
            raise RuntimeError("nano-vLLM batching state is not initialized")
        if self._scheduler_event is None:
            self._scheduler_event = asyncio.Event()

    def _submit_sequence(self, seq: Sequence) -> None:
        self._ensure_batching_state()
        self._requests[seq.seq_id] = seq
        request = request_from_sequence(seq)
        if self._core_process is not None:
            self._core_process.send(("add", request))
        else:
            self.core.add_request(request)
            self._wake()

    def _wake(self) -> None:
        if self._scheduler_event is not None:
            self._scheduler_event.set()

    async def _scheduler_loop(self) -> None:
        """Step the in-process core in a worker thread while it has work."""
        self._ensure_batching_state()
        loop = asyncio.get_running_loop()
        while not self._stopping:
            if not self.core.has_work():
                self._scheduler_event.clear()
                await self._scheduler_event.wait()
                continue
            self._step_future = loop.run_in_executor(None, self.core.step)
            try:
                outputs = await asyncio.shield(self._step_future)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - EngineCore.step reports failures
                logger.exception("nano-vLLM engine step failed")
                await asyncio.sleep(0)
                continue
            self._step_future = None
            self._dispatch(outputs)

    def _on_core_message(self, message: tuple) -> None:
        """A message from the core process (runs on the event loop)."""
        kind = message[0]
        if kind == "outputs":
            if message[2] is not None:
                self._core_process.stats = message[2]
            self._dispatch(message[1])
        elif kind == "dead" and not self._stopping:
            logger.error("nano-vLLM engine core died:\n%s", message[1])
            self.initialized = False
            self._fail_frontend_requests(RuntimeError("nano-vLLM engine core died"))

    def _dispatch(self, outputs: list[RequestOutput]) -> None:
        """Hand a step's outputs to the waiting requests."""
        with nvtx_range("publish"):
            for output in outputs:
                seq = self._requests.get(output.req_id)
                if seq is None:  # aborted by the frontend meanwhile
                    continue
                if output.error is not None:
                    self._requests.pop(output.req_id)
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_reason = seq.finish_reason or "error"
                    seq.events.put_nowait(GenerationEvent(finished=True, error=output.error))
                    continue
                if output.token_id is not None:
                    seq.append_token(output.token_id)
                if output.finish_reason is not None:
                    self._requests.pop(output.req_id)
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_reason = output.finish_reason
                seq.events.put_nowait(
                    GenerationEvent(
                        token_id=output.token_id,
                        finished=output.finish_reason is not None,
                        finish_reason=output.finish_reason,
                    )
                )

    def _fail_frontend_requests(self, exc: BaseException) -> None:
        for seq in list(self._requests.values()):
            seq.status = SequenceStatus.FINISHED
            seq.finish_reason = seq.finish_reason or "error"
            seq.events.put_nowait(GenerationEvent(finished=True, error=exc))
        self._requests.clear()


class _CoreProcess:
    """The engine core in a spawned process, connected by two one-way pipes."""

    def __init__(self, config: NanoVLLMConfig):
        context = multiprocessing.get_context("spawn")
        core_inbox, self._to_core = context.Pipe(duplex=False)
        self._from_core, core_outbox = context.Pipe(duplex=False)
        self.process = context.Process(
            target=run_core_process,
            args=(config, core_inbox, core_outbox),
            name="nano-vllm-core",
            daemon=True,
        )
        self._child_ends = (core_inbox, core_outbox)
        self._send_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self.num_blocks = 0
        self.stats: dict = {}

    def start(self, timeout_s: float = 1800.0) -> None:
        """Start the core and wait until it has loaded the model."""
        self.process.start()
        for end in self._child_ends:
            end.close()  # EOF reaches the core when this process goes away
        if not self._from_core.poll(timeout_s):
            self.process.kill()
            raise RuntimeError("nano-vLLM engine core did not start in time")
        message = self._from_core.recv()
        if message[0] != "ready":
            self.process.join(5)
            detail = message[1] if message[0] == "dead" else message
            raise RuntimeError(f"nano-vLLM engine core failed to start:\n{detail}")
        self.num_blocks = message[1]["num_blocks"]

    def send(self, message: tuple) -> None:
        with self._send_lock:
            self._to_core.send(message)

    def utility(self, name: str) -> None:
        self.send(("utility", 0, name))

    def start_reader(self, loop: asyncio.AbstractEventLoop, on_message) -> None:
        if self._reader is not None:
            return

        def read() -> None:
            while True:
                try:
                    message = self._from_core.recv()
                except (EOFError, OSError):
                    message = ("dead", "the engine core process exited")
                try:
                    if message[0] != "utility":
                        loop.call_soon_threadsafe(on_message, message)
                except RuntimeError:  # the event loop is closed
                    return
                if message[0] == "dead":
                    return

        self._reader = threading.Thread(target=read, name="nano-vllm-core-reader", daemon=True)
        self._reader.start()

    def shutdown(self) -> None:
        try:
            self.send(("shutdown",))
        except (BrokenPipeError, OSError):
            pass
        self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)


def _token_id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(token_id) for token_id in value]
