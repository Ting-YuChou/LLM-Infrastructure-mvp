"""Generation engine wrapper for the nano-vLLM server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import replace
from contextlib import suppress
from typing import Optional

import torch

from .block_manager import BlockManager
from .config import NanoVLLMConfig
from .kv_cache import PagedKVCache
from .model import NanoLlamaForCausalLM
from .scheduler import Scheduler
from .sampling import SamplingOptions, sample_next_token
from .sequence import GenerationEvent, Sequence, SequenceStatus


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_ids: list[int]


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
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_event: Optional[asyncio.Event] = None
        self._scheduler_lock: Optional[asyncio.Lock] = None
        self._stopping = False

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
        )
        self._initialize_batching_state()
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
            with suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None

    def start_background_loop(self) -> None:
        self._ensure_batching_state()
        if self._scheduler_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._stopping = False
        self._scheduler_task = loop.create_task(self._scheduler_loop())

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling: SamplingOptions,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("nano-vLLM engine is not initialized")

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.model.device)
        if input_ids.shape[0] != 1:
            raise ValueError("nano-vLLM v0.1 supports batch size 1 inside the engine")
        input_ids = input_ids[:, -self.config.max_model_len :]
        prompt_tokens = int(input_ids.shape[1])
        max_new_tokens = max(1, min(sampling.max_tokens, self.config.max_new_tokens))
        max_seq_len = min(self.config.max_model_len, prompt_tokens + max_new_tokens)
        cache = self.model.new_cache(batch_size=1, max_seq_len=max_seq_len)

        logits = self.model.forward(input_ids, cache=cache, start_pos=0)
        generated: list[int] = []
        current_pos = prompt_tokens
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(seed)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits[0, -1],
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                generator=generator,
            )
            if self._is_eos(next_token):
                break
            generated.append(next_token)
            if current_pos >= max_seq_len:
                break
            token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.model.device
            )
            logits = self.model.forward(
                token_tensor, cache=cache, start_pos=current_pos
            )
            current_pos += 1

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = self._apply_stop_strings(text, sampling.stop or [])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated),
            token_ids=generated,
        )

    async def generate_async(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ) -> GenerationResult:
        seq = self._create_sequence(prompt, sampling)
        self.start_background_loop()
        self._submit_sequence(seq)
        generated: list[int] = []
        while True:
            event = await seq.events.get()
            if event.error is not None:
                raise event.error
            if event.token_id is not None:
                generated.append(event.token_id)
            if event.finished:
                break

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = self._apply_stop_strings(text, sampling.stop or [])
        return GenerationResult(
            text=text,
            prompt_tokens=seq.num_prompt_tokens,
            completion_tokens=len(generated),
            token_ids=generated,
        )

    async def generate_stream_async(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ):
        seq = self._create_sequence(prompt, sampling)
        self.start_background_loop()
        self._submit_sequence(seq)
        generated: list[int] = []
        while True:
            event = await seq.events.get()
            if event.error is not None:
                raise event.error
            if event.token_id is not None:
                generated.append(event.token_id)
                yield self.tokenizer.decode([event.token_id], skip_special_tokens=True)
            if event.finished:
                break

    def _is_eos(self, token_id: int) -> bool:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return False
        if isinstance(eos, list):
            return token_id in eos
        return token_id == eos

    def _apply_stop_strings(self, text: str, stop: list[str]) -> str:
        cutoff = None
        for marker in stop:
            index = text.find(marker)
            if index >= 0:
                cutoff = index if cutoff is None else min(cutoff, index)
        return text if cutoff is None else text[:cutoff]

    def _initialize_batching_state(self) -> None:
        if self.model is None:
            return
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
        )
        self.paged_cache = self.model.new_paged_cache(
            num_blocks=self.config.num_kvcache_blocks,
            block_size=self.config.kvcache_block_size,
        )

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

    def _create_sequence(self, prompt: str, sampling: SamplingOptions) -> Sequence:
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("nano-vLLM engine is not initialized")
        sampling = replace(
            sampling,
            max_tokens=max(1, min(sampling.max_tokens, self.config.max_new_tokens)),
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"][0].tolist()
        input_ids = input_ids[-self.config.max_model_len :]
        if not input_ids:
            raise ValueError("Prompt produced no tokens")
        seq = Sequence(
            prompt_token_ids=[int(token_id) for token_id in input_ids],
            sampling=sampling,
            eos_token_id=self._first_eos_token_id(),
        )
        seq.events = asyncio.Queue()
        return seq

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
                await asyncio.sleep(0)
                continue

            async with self._scheduler_lock:
                seqs, is_prefill = self.scheduler.schedule()
            if not seqs:
                await asyncio.sleep(0)
                continue

            try:
                token_ids = await asyncio.to_thread(
                    self._run_scheduled_batch,
                    seqs,
                    is_prefill,
                )
                async with self._scheduler_lock:
                    self.scheduler.postprocess(seqs, token_ids, is_prefill)
                await self._publish_events(seqs, token_ids, is_prefill)
            except Exception as exc:
                await self._fail_sequences(seqs, exc)

    def _run_scheduled_batch(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if self.model is None or self.paged_cache is None or self.block_manager is None:
            raise RuntimeError("nano-vLLM engine is not initialized")
        input_ids, positions, slot_mapping, block_tables, context_lens, query_lens = (
            self._prepare_prefill(seqs) if is_prefill else self._prepare_decode(seqs)
        )
        logits = self.model.forward_paged(
            input_ids=input_ids,
            positions=positions,
            cache=self.paged_cache,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_lens=query_lens,
            is_prefill=is_prefill,
        )
        token_ids = []
        offset = 0
        for seq, query_len in zip(seqs, query_lens):
            sample_index = offset + query_len - 1
            token_ids.append(
                sample_next_token(
                    logits[sample_index],
                    temperature=seq.sampling.temperature,
                    top_p=seq.sampling.top_p,
                    top_k=seq.sampling.top_k,
                )
            )
            offset += query_len
        return token_ids

    def _prepare_prefill(
        self,
        seqs: list[Sequence],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
    ]:
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            query_lens.append(seq.num_scheduled_tokens)
            context_lens.append(end)
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            slots.extend(
                self.block_manager.slot_for_position(seq, pos)
                for pos in range(start, end)
            )
        return self._to_runner_tensors(
            input_ids, positions, slots, seqs, context_lens, query_lens
        )

    def _prepare_decode(
        self,
        seqs: list[Sequence],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
    ]:
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        for seq in seqs:
            position = len(seq) - 1
            input_ids.append(seq.last_token)
            positions.append(position)
            slots.append(self.block_manager.slot_for_position(seq, position))
            context_lens.append(len(seq))
            query_lens.append(1)
        return self._to_runner_tensors(
            input_ids, positions, slots, seqs, context_lens, query_lens
        )

    def _to_runner_tensors(
        self,
        input_ids: list[int],
        positions: list[int],
        slots: list[int],
        seqs: list[Sequence],
        context_lens: list[int],
        query_lens: list[int],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
    ]:
        device = self.model.device
        max_blocks = max(len(seq.block_table) for seq in seqs)
        padded_block_tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs
        ]
        return (
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(positions, dtype=torch.long, device=device),
            torch.tensor(slots, dtype=torch.long, device=device),
            torch.tensor(padded_block_tables, dtype=torch.long, device=device),
            torch.tensor(context_lens, dtype=torch.long, device=device),
            query_lens,
        )

    async def _publish_events(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> None:
        for seq, token_id in zip(seqs, token_ids):
            if (
                is_prefill
                and not seq.is_finished
                and seq.num_cached_tokens + 1 < seq.num_tokens
            ):
                continue
            if seq.events is not None:
                await seq.events.put(
                    GenerationEvent(
                        token_id=token_id,
                        finished=seq.is_finished,
                    )
                )

    async def _fail_sequences(
        self,
        seqs: list[Sequence],
        exc: BaseException,
    ) -> None:
        for seq in seqs:
            seq.status = SequenceStatus.FINISHED
            if self.block_manager is not None and seq.block_table:
                self.block_manager.deallocate(seq)
            if self.scheduler is not None:
                with suppress(ValueError):
                    self.scheduler.running.remove(seq)
                with suppress(ValueError):
                    self.scheduler.waiting.remove(seq)
            if seq.events is not None:
                await seq.events.put(GenerationEvent(finished=True, error=exc))

    def _first_eos_token_id(self) -> Optional[int]:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, list):
            return eos[0] if eos else None
        return eos
