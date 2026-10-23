import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.engine import InvalidRequestError, NanoLlamaEngine  # noqa: E402
from serving.nano_vllm.kv_cache import PagedKVCache  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, prompt, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        base = 10 if "alpha" in prompt else 20
        return {"input_ids": torch.tensor([[base, base + 1]])}

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        del skip_special_tokens, kwargs
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class FakePagedModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def new_paged_cache(self, num_blocks, block_size):
        return PagedKVCache(
            num_layers=1,
            num_blocks=num_blocks,
            block_size=block_size,
            num_key_value_heads=1,
            head_dim=1,
            dtype=torch.float32,
            device="cpu",
        )

    def forward_batch(self, input_ids, positions, cache, metadata):
        del positions, cache
        self.calls.append(
            {
                "tokens": int(input_ids.numel()),
                "query_lens": list(metadata.query_lens),
                "is_prefill": metadata.is_prefill,
            }
        )
        # One row per sequence, like the real model with logits_indices.
        logits = torch.zeros(metadata.num_seqs, 128)
        logits[:, 2] = 10.0
        return logits


async def _run_two_requests():
    engine = NanoLlamaEngine(
        NanoVLLMConfig(
            max_new_tokens=1,
            kvcache_block_size=4,
            num_kvcache_blocks=8,
            max_num_seqs=4,
            max_num_batched_tokens=16,
            enable_prefix_cache=False,
            attention_backend="torch",
            eager_load=False,
        )
    )
    engine.tokenizer = FakeTokenizer()
    engine.model = FakePagedModel()
    engine._initialize_batching_state()
    engine.initialized = True
    try:
        results = await asyncio.gather(
            engine.generate_async("alpha prompt", SamplingOptions(max_tokens=1)),
            engine.generate_async("beta prompt", SamplingOptions(max_tokens=1)),
        )
    finally:
        await engine.shutdown()
    return engine.model.calls, results


def test_engine_batches_concurrent_prefill_requests():
    calls, results = asyncio.run(_run_two_requests())

    assert [result.token_ids for result in results] == [[2], [2]]
    assert calls[0]["is_prefill"] is True
    assert calls[0]["query_lens"] == [2, 2]
    assert calls[0]["tokens"] == 4


# --- helpers for the bug-regression tests below -----------------------------


class ListTokenizer:
    """Maps each known prompt string to fixed token ids."""

    eos_token_id = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self, prompts):
        self.prompts = prompts if isinstance(prompts, dict) else {"p": prompts}

    def __call__(self, prompt, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([self.prompts[prompt]])}

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        del skip_special_tokens, kwargs
        return "".join(f" t{token_id}" for token_id in token_ids)


class ByteTokenizer:
    """One token per UTF-8 byte, like byte-level BPE at its most fragmented."""

    eos_token_id = 300
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, prompt, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([list(prompt.encode("utf-8"))])}

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        del skip_special_tokens, kwargs
        data = bytes(token_id for token_id in token_ids if token_id < 256)
        return data.decode("utf-8", errors="replace")


class ScriptedPagedModel(FakePagedModel):
    """Each sequence's logits row argmaxes to next_token(segment_ids, positions)."""

    def __init__(self, next_token, delay: float = 0.0):
        super().__init__()
        self.next_token = next_token
        self.delay = delay

    def forward_batch(self, input_ids, positions, cache, metadata):
        del cache
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(
            {"query_lens": list(metadata.query_lens), "is_prefill": metadata.is_prefill}
        )
        logits = torch.zeros(metadata.num_seqs, 512)
        start = 0
        for row, query_len in enumerate(metadata.query_lens):
            end = start + query_len
            token = self.next_token(
                input_ids[start:end].tolist(), positions[start:end].tolist()
            )
            logits[row, token] = 10.0
            start = end
        return logits


def successor_model(delay: float = 0.0) -> ScriptedPagedModel:
    return ScriptedPagedModel(lambda ids, positions: ids[-1] + 1, delay=delay)


def script_model(script: bytes, prompt_len: int, eos: int = 300):
    def next_token(ids, positions):
        index = positions[-1] - (prompt_len - 1)
        return script[index] if index < len(script) else eos

    return ScriptedPagedModel(next_token)


def _engine(tokenizer, model, **overrides) -> NanoLlamaEngine:
    values = dict(
        max_model_len=64,
        max_new_tokens=32,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        enable_prefix_cache=False,
        attention_backend="torch",
        eager_load=False,
    )
    values.update(overrides)
    engine = NanoLlamaEngine(NanoVLLMConfig(**values))
    engine.tokenizer = tokenizer
    engine.model = model
    engine._initialize_batching_state()
    engine.initialized = True
    return engine


GREEDY = dict(temperature=0.0)


async def _collect(engine, seq):
    outputs = []
    async for output in engine.stream_request(seq):
        outputs.append(output)
    return outputs


async def _wait_for(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for engine"
        await asyncio.sleep(0.005)


def _assert_all_blocks_free(engine):
    assert engine.scheduler.is_finished()
    assert engine.block_manager.num_free_blocks == engine.config.num_kvcache_blocks


# --- bug regressions ----------------------------------------------------------


def test_chunked_prefill_leaving_one_token_streams_only_appended_tokens():
    budget = 8
    prompt = list(range(10, 10 + budget + 1))  # one more than a chunk

    async def scenario():
        engine = _engine(
            ListTokenizer(prompt), successor_model(), max_num_batched_tokens=budget
        )
        try:
            seq = engine.prepare_request("p", SamplingOptions(max_tokens=5, **GREEDY))
            outputs = await _collect(engine, seq)
        finally:
            await engine.shutdown()
        return engine, seq, outputs

    engine, seq, outputs = asyncio.run(scenario())

    streamed = [token for output in outputs for token in output.token_ids]
    assert streamed == seq.completion_token_ids == [19, 20, 21, 22, 23]
    assert [call["query_lens"] for call in engine.model.calls[:2]] == [[8], [1]]
    assert outputs[-1].finish_reason == "length"
    _assert_all_blocks_free(engine)


def test_schedule_exception_fails_requests_but_keeps_loop_alive():
    async def scenario():
        engine = _engine(FakeTokenizer(), FakePagedModel())
        original = engine.scheduler.schedule
        calls = {"count": 0}

        def flaky_schedule():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("scheduler bug")
            return original()

        engine.scheduler.schedule = flaky_schedule
        try:
            with pytest.raises(RuntimeError, match="scheduler bug"):
                await asyncio.wait_for(
                    engine.generate_async("alpha", SamplingOptions(max_tokens=2)), 5
                )
            result = await asyncio.wait_for(
                engine.generate_async("beta", SamplingOptions(max_tokens=2)), 5
            )
        finally:
            await engine.shutdown()
        return engine, result

    engine, result = asyncio.run(scenario())

    assert result.token_ids == [2, 2]
    _assert_all_blocks_free(engine)


def test_background_loop_restarts_if_scheduler_task_died():
    async def scenario():
        engine = _engine(FakeTokenizer(), FakePagedModel())
        engine.start_background_loop()
        dead_task = engine._scheduler_task
        dead_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dead_task
        try:
            result = await asyncio.wait_for(
                engine.generate_async("alpha", SamplingOptions(max_tokens=1)), 5
            )
        finally:
            await engine.shutdown()
        return dead_task, result

    dead_task, result = asyncio.run(scenario())

    assert dead_task.done()
    assert result.token_ids == [2]


def test_requests_that_cannot_fit_are_rejected_not_truncated_or_clamped():
    engine = _engine(
        ListTokenizer({"long": list(range(70)), "short": [1, 2, 3, 4]}),
        FakePagedModel(),
        max_model_len=64,
        max_new_tokens=16,
        num_kvcache_blocks=4,  # 16 KV slots
    )

    with pytest.raises(InvalidRequestError, match="max_model_len"):
        engine.prepare_request("long", SamplingOptions(max_tokens=1))
    with pytest.raises(InvalidRequestError, match="max_new_tokens"):
        engine.prepare_request("short", SamplingOptions(max_tokens=17))
    with pytest.raises(InvalidRequestError, match="KV"):
        engine.prepare_request("short", SamplingOptions(max_tokens=13))

    fitting = engine.prepare_request("short", SamplingOptions(max_tokens=12))
    default = engine.prepare_request("short", SamplingOptions())
    assert fitting.sampling.max_tokens == 12
    assert fitting.prompt_token_ids == [1, 2, 3, 4]
    assert default.sampling.max_tokens == 12  # capped to fit, never above limits


def test_prompt_plus_max_tokens_beyond_max_model_len_is_rejected():
    engine = _engine(
        ListTokenizer(list(range(60))),
        FakePagedModel(),
        max_model_len=64,
        num_kvcache_blocks=64,
    )

    with pytest.raises(InvalidRequestError, match="maximum context length is 64"):
        engine.prepare_request("p", SamplingOptions(max_tokens=5))
    assert engine.prepare_request("p", SamplingOptions()).sampling.max_tokens == 4


def test_eos_ids_from_generation_config_list_and_stop_token_ids(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [97, 98]}), encoding="utf-8"
    )

    async def scenario():
        engine = _engine(
            ListTokenizer([95]), successor_model(), model_name=str(tmp_path)
        )
        try:
            eos = await engine.generate_async(
                "p", SamplingOptions(max_tokens=10, **GREEDY)
            )
            stop_id = await engine.generate_async(
                "p", SamplingOptions(max_tokens=10, stop_token_ids=[96], **GREEDY)
            )
            length = await engine.generate_async(
                "p", SamplingOptions(max_tokens=1, **GREEDY)
            )
        finally:
            await engine.shutdown()
        return engine, eos, stop_id, length

    engine, eos, stop_id, length = asyncio.run(scenario())

    assert engine.eos_token_ids == {97, 98}
    assert (eos.token_ids, eos.finish_reason) == ([96, 97], "stop")
    assert (stop_id.token_ids, stop_id.finish_reason) == ([96], "stop")
    assert (length.token_ids, length.finish_reason) == ([96], "length")


def test_stop_string_halts_streaming_generation():
    prompt = "ab"
    script = b"hello STOP world, this would keep going for a long while"

    async def scenario():
        engine = _engine(ByteTokenizer(), script_model(script, len(prompt)))
        try:
            seq = engine.prepare_request(
                prompt, SamplingOptions(max_tokens=32, stop=["STOP"], **GREEDY)
            )
            outputs = await _collect(engine, seq)
            await _wait_for(engine.scheduler.is_finished)
        finally:
            await engine.shutdown()
        return engine, seq, outputs

    engine, seq, outputs = asyncio.run(scenario())

    assert "".join(output.text for output in outputs) == "hello "
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].completion_tokens == len("hello STOP")
    # Halted right after the stop string, not after max_tokens=32 steps.
    assert seq.num_completion_tokens <= len("hello STOP") + 1
    assert len(engine.model.calls) <= len("hello STOP") + 1
    _assert_all_blocks_free(engine)


def test_streaming_detokenizes_multibyte_utf8_incrementally():
    prompt = "q"
    text = "你好, мир 👋!"

    async def scenario():
        engine = _engine(ByteTokenizer(), script_model(text.encode("utf-8"), 1))
        try:
            seq = engine.prepare_request(
                prompt, SamplingOptions(max_tokens=32, **GREEDY)
            )
            outputs = await _collect(engine, seq)
            result = await engine.generate_async(
                prompt, SamplingOptions(max_tokens=32, **GREEDY)
            )
        finally:
            await engine.shutdown()
        return outputs, result

    outputs, result = asyncio.run(scenario())

    chunks = [output.text for output in outputs if output.text]
    assert "".join(chunks) == text
    assert len(chunks) > 1
    assert not any("�" in chunk for chunk in chunks)
    assert result.text == text
    assert result.finish_reason == "stop"  # ended on the tokenizer EOS id


def test_closing_stream_aborts_sequence_and_frees_kv_blocks():
    async def scenario():
        engine = _engine(ListTokenizer([10, 11, 12]), successor_model(delay=0.005))
        try:
            seq = engine.prepare_request("p", SamplingOptions(max_tokens=30))
            stream = engine.stream_request(seq)
            await stream.__anext__()
            await stream.__anext__()
            await stream.aclose()  # what a client disconnect does to the stream
            await _wait_for(engine.scheduler.is_finished)
            steps_after_abort = len(engine.model.calls)
            await asyncio.sleep(0.05)
        finally:
            await engine.shutdown()
        return engine, seq, steps_after_abort

    engine, seq, steps_after_abort = asyncio.run(scenario())

    assert seq.finish_reason == "abort"
    assert seq.num_completion_tokens < 30
    assert len(engine.model.calls) == steps_after_abort
    _assert_all_blocks_free(engine)


def test_cancelled_consumer_aborts_sequence_and_frees_kv_blocks():
    async def scenario():
        engine = _engine(ListTokenizer([10, 11, 12]), successor_model(delay=0.005))
        try:
            seq = engine.prepare_request("p", SamplingOptions(max_tokens=30))
            task = asyncio.ensure_future(_collect(engine, seq))
            await _wait_for(lambda: seq.num_completion_tokens >= 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await _wait_for(engine.scheduler.is_finished)
        finally:
            await engine.shutdown()
        return engine, seq

    engine, seq = asyncio.run(scenario())

    assert seq.finish_reason == "abort"
    _assert_all_blocks_free(engine)


@pytest.mark.parametrize("scheduling", ["mixed", "prefill_first"])
def test_paged_engine_matches_transformers_greedy_decode(tmp_path, scheduling):
    from transformers import LlamaConfig, LlamaForCausalLM

    from serving.nano_vllm.model import NanoLlamaForCausalLM

    torch.manual_seed(4321)
    hf_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=96,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            max_position_embeddings=128,
            attn_implementation="eager",
        )
    ).eval()
    hf_model.save_pretrained(str(tmp_path))
    eos_ids = {hf_model.config.eos_token_id}

    def hf_greedy(prompt_ids, max_tokens):
        ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(max_tokens):
                token = int(hf_model(torch.tensor([ids])).logits[0, -1].argmax())
                ids.append(token)
                if token in eos_ids:
                    break
        return ids[len(prompt_ids) :]

    prompts = {
        "a": [5, 17, 3, 88, 42, 9, 61, 8, 33, 70, 11],
        "b": [5, 17, 3, 88, 42, 9, 61, 8, 40, 41],  # shares two full blocks
        "c": [7, 7, 50],
    }

    async def scenario():
        model = NanoLlamaForCausalLM.from_pretrained(
            tmp_path, dtype="float32", device="cpu", use_triton=False
        )
        engine = _engine(
            ListTokenizer(prompts),
            model,
            model_name=str(tmp_path),
            max_new_tokens=8,
            max_num_batched_tokens=6,  # forces chunked prefill
            enable_prefix_cache=True,
            scheduling=scheduling,
        )
        sampling = SamplingOptions(max_tokens=8, **GREEDY)
        try:
            first = await asyncio.gather(
                *(engine.generate_async(name, sampling) for name in prompts)
            )
            # Second round reuses prefix-cached blocks that are back in the
            # free list with ref_count 0.
            second = await asyncio.gather(
                *(engine.generate_async(name, sampling) for name in prompts)
            )
        finally:
            await engine.shutdown()
        return engine, first, second

    engine, first, second = asyncio.run(scenario())

    expected = [hf_greedy(ids, 8) for ids in prompts.values()]
    assert [result.token_ids for result in first] == expected
    assert [result.token_ids for result in second] == expected
    _assert_all_blocks_free(engine)


def test_kv_cache_size_zero_means_auto_and_cpu_gets_the_default():
    from serving.nano_vllm.engine import CPU_KV_BLOCKS

    engine = _engine(FakeTokenizer(), FakePagedModel(), num_kvcache_blocks=0)
    assert engine.block_manager.num_blocks == CPU_KV_BLOCKS
    assert engine.config.num_kvcache_blocks == CPU_KV_BLOCKS
