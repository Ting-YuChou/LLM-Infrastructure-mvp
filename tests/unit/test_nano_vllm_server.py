import sys
import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from serving.nano_vllm import server as server_module  # noqa: E402
from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.engine import GenerationResult  # noqa: E402
from serving.nano_vllm.engine import InvalidRequestError, StreamOutput  # noqa: E402
from serving.nano_vllm.server import NanoVLLMServer  # noqa: E402


class FakeNanoEngine:
    def __init__(self):
        self.initialized = False
        self.prompts = []

    def initialize(self):
        self.initialized = True

    def generate(self, prompt, sampling):
        self.prompts.append((prompt, sampling))
        return GenerationResult(
            text=f"nano response for {prompt.split()[0]}",
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=4,
            token_ids=[1, 2, 3, 4],
        )


class FakeAsyncBatchEngine(FakeNanoEngine):
    def __init__(self):
        super().__init__()
        self.async_prompts = []
        self.active = 0
        self.max_active = 0

    async def generate_async(self, prompt, sampling):
        self.async_prompts.append(prompt)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return GenerationResult(
            text=f"batched response for {prompt.split()[0]}",
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=3,
            token_ids=[2, 3, 4],
        )

    async def generate_stream_async(self, prompt, sampling):
        self.async_prompts.append(prompt)
        for token in ("batched", " stream"):
            await asyncio.sleep(0)
            yield token


def _server(eager_load: bool = True) -> NanoVLLMServer:
    return NanoVLLMServer(
        NanoVLLMConfig(
            served_model_name="nano-test",
            model_name="/tmp/not-used",
            eager_load=eager_load,
            enable_metrics=False,
        ),
        engine=FakeNanoEngine(),
    )


def _server_with_engine(engine, eager_load: bool = True) -> NanoVLLMServer:
    return NanoVLLMServer(
        NanoVLLMConfig(
            served_model_name="nano-test",
            model_name="/tmp/not-used",
            eager_load=eager_load,
            enable_metrics=False,
        ),
        engine=engine,
    )


def test_nano_server_health_ready_and_completion_contract():
    server = _server()
    with TestClient(server.app) as client:
        assert client.get("/health").json()["mode"] == "nano-vllm"
        assert client.get("/ready").json()["status"] == "ready"

        response = client.post(
            "/v1/completions",
            json={"prompt": "hello world", "max_tokens": 8, "temperature": 0.0},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["model"] == "nano-test"
    assert data["choices"][0]["text"].startswith("nano response")
    assert data["usage"]["total_tokens"] > 0


def test_nano_server_chat_completion_contract():
    server = _server()
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "say ready"}],
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]


def test_nano_server_reports_not_ready_when_engine_not_loaded():
    server = _server(eager_load=False)
    with TestClient(server.app) as client:
        ready = client.get("/ready")
        completion = client.post("/v1/completions", json={"prompt": "hello"})

    assert ready.status_code == 503
    assert completion.status_code == 503


def test_nano_server_uses_async_batch_generation_for_prompt_lists():
    engine = FakeAsyncBatchEngine()
    server = _server_with_engine(engine)
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": ["alpha prompt", "beta prompt"], "max_tokens": 8},
        )

    assert response.status_code == 200
    data = response.json()
    assert [choice["index"] for choice in data["choices"]] == [0, 1]
    assert engine.async_prompts == ["alpha prompt", "beta prompt"]
    assert engine.prompts == []
    assert engine.max_active == 2


def test_nano_server_streams_from_async_engine():
    engine = FakeAsyncBatchEngine()
    server = _server_with_engine(engine)
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "alpha prompt", "stream": True, "max_tokens": 8},
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "batched" in response.text


class FakeStreamingEngine:
    """Implements the prepare_request/stream_request contract of the engine."""

    def __init__(
        self, chunks=("a", "b"), finish_reason="length", invalid=False, endless=False
    ):
        self.initialized = True
        self.chunks = chunks
        self.finish_reason = finish_reason
        self.invalid = invalid
        self.endless = endless
        self.streams_closed = 0
        self.cancelled = 0

    def initialize(self):
        self.initialized = True

    def prepare_request(self, prompt, sampling):
        if self.invalid:
            raise InvalidRequestError("prompt is longer than max_model_len")
        return {"prompt": prompt, "sampling": sampling}

    async def stream_request(self, seq):
        try:
            if self.endless:
                while True:
                    await asyncio.sleep(0.001)
                    yield StreamOutput(text="x", token_ids=[1])
            for index, chunk in enumerate(self.chunks):
                yield StreamOutput(text=chunk, token_ids=[index])
            yield StreamOutput(
                text="",
                finished=True,
                finish_reason=self.finish_reason,
                prompt_tokens=3,
                completion_tokens=len(self.chunks),
            )
        finally:
            self.streams_closed += 1

    async def generate_async(self, prompt, sampling):
        if self.invalid or prompt == "bad":
            raise InvalidRequestError("prompt is longer than max_model_len")
        if prompt == "slow":
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return GenerationResult(
            text="".join(self.chunks),
            prompt_tokens=3,
            completion_tokens=len(self.chunks),
            token_ids=list(range(len(self.chunks))),
            finish_reason=self.finish_reason,
        )


def _sse_payloads(text):
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: {")
    ]


def test_finish_reason_length_is_reported_for_all_endpoints():
    server = _server_with_engine(FakeStreamingEngine(finish_reason="length"))
    chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
    with TestClient(server.app) as client:
        completion = client.post("/v1/completions", json={"prompt": "hi"}).json()
        chat_completion = client.post("/v1/chat/completions", json=chat).json()
        stream = client.post("/v1/completions", json={"prompt": "hi", "stream": True})
        chat_stream = client.post("/v1/chat/completions", json={**chat, "stream": True})

    assert completion["choices"][0]["finish_reason"] == "length"
    assert chat_completion["choices"][0]["finish_reason"] == "length"
    stream_payloads = _sse_payloads(stream.text)
    assert "".join(p["choices"][0]["text"] for p in stream_payloads) == "ab"
    assert stream_payloads[-1]["choices"][0]["finish_reason"] == "length"
    assert _sse_payloads(chat_stream.text)[-1]["choices"][0]["finish_reason"] == (
        "length"
    )


def test_finish_reason_stop_is_reported_when_engine_stops():
    server = _server_with_engine(FakeStreamingEngine(finish_reason="stop"))
    with TestClient(server.app) as client:
        completion = client.post("/v1/completions", json={"prompt": "hi"}).json()

    assert completion["choices"][0]["finish_reason"] == "stop"


def test_invalid_requests_return_400_for_stream_and_non_stream():
    server = _server_with_engine(FakeStreamingEngine(invalid=True))
    with TestClient(server.app) as client:
        completion = client.post("/v1/completions", json={"prompt": "hi"})
        stream = client.post("/v1/completions", json={"prompt": "hi", "stream": True})
        chat_stream = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert completion.status_code == 400
    assert stream.status_code == 400
    assert chat_stream.status_code == 400
    assert "max_model_len" in stream.json()["detail"]


def test_failed_prompt_cancels_sibling_generations():
    engine = FakeStreamingEngine()
    server = _server_with_engine(engine)
    with TestClient(server.app) as client:
        response = client.post("/v1/completions", json={"prompt": ["slow", "bad"]})
        deadline = time.monotonic() + 2
        while engine.cancelled == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert response.status_code == 400
    assert engine.cancelled == 1


def _record_tracking(monkeypatch):
    events = []
    monkeypatch.setattr(
        server_module, "start_request_tracking", lambda model: events.append("start")
    )
    monkeypatch.setattr(
        server_module,
        "finish_request_tracking",
        lambda **kwargs: events.append(("finish", kwargs["status"])),
    )
    return events


def test_stream_closed_by_client_finishes_tracking_and_closes_engine_stream(
    monkeypatch,
):
    events = _record_tracking(monkeypatch)
    engine = FakeStreamingEngine(endless=True)
    server = _server_with_engine(engine)

    async def scenario():
        body = server._stream_completion(
            lambda: engine.stream_request(None), "nano-test", time.time(), "hi"
        )
        await body.__anext__()
        await body.__anext__()
        await body.aclose()  # what Starlette/GC does after a disconnect

    asyncio.run(scenario())

    assert events == ["start", ("finish", "cancelled")]
    assert engine.streams_closed == 1
    assert server.admission._value == server.config.max_concurrent_requests


def test_cancelled_stream_finishes_tracking_and_closes_engine_stream(monkeypatch):
    events = _record_tracking(monkeypatch)
    engine = FakeStreamingEngine(endless=True)
    server = _server_with_engine(engine)

    async def scenario():
        body = server._stream_chat_completion(
            lambda: engine.stream_request(None), "nano-test", time.time(), "hi"
        )

        async def consume():
            async for _ in body:
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.02)
        task.cancel()  # Starlette cancels the body task on disconnect
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert events == ["start", ("finish", "cancelled")]
    assert engine.streams_closed == 1
    assert server.admission._value == server.config.max_concurrent_requests


def test_stream_with_real_engine_reports_length_and_frees_blocks():
    from test_nano_vllm_engine_batching import ListTokenizer, _engine, successor_model

    engine = _engine(ListTokenizer({"user: hi\nassistant:": [10, 11]}), successor_model())
    server = _server_with_engine(engine, eager_load=False)  # already "loaded"
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "temperature": 0.0,
                "stream": True,
            },
        )

    payloads = _sse_payloads(response.text)
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert content == " t12 t13 t14"
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"
    assert engine.block_manager.num_free_blocks == engine.config.num_kvcache_blocks


def test_ignore_eos_and_usage_chunk_for_benchmark_clients():
    """vllm bench serve sends ignore_eos and stream_options.include_usage."""
    from test_nano_vllm_engine_batching import ListTokenizer, _engine, successor_model

    class EosTokenizer(ListTokenizer):
        eos_token_id = 13  # the successor model reaches it after two tokens

    engine = _engine(EosTokenizer({"go": [10, 11]}), successor_model())
    server = _server_with_engine(engine, eager_load=False)
    body = {"prompt": "go", "max_tokens": 4, "temperature": 0.0, "stream": True}
    with TestClient(server.app) as client:
        stopped = _sse_payloads(client.post("/v1/completions", json=body).text)
        ignored = _sse_payloads(
            client.post(
                "/v1/completions",
                json={**body, "ignore_eos": True, "stream_options": {"include_usage": True}},
            ).text
        )

    # Without ignore_eos: no usage chunk, generation stops at EOS (13; this
    # fake tokenizer does not skip it when decoding).
    assert all("usage" not in p for p in stopped)
    assert "".join(p["choices"][0]["text"] for p in stopped) == " t12 t13"
    assert stopped[-1]["choices"][0]["finish_reason"] == "stop"
    # With it: all four tokens, then a usage chunk with empty choices.
    usage = ignored[-1]
    assert usage["choices"] == []
    assert usage["usage"] == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
    assert ignored[-2]["choices"][0]["finish_reason"] == "length"
    text = "".join(p["choices"][0]["text"] for p in ignored[:-1])
    assert text == " t12 t13 t14 t15"


def test_models_and_profile_routes():
    from test_nano_vllm_engine_batching import ListTokenizer, _engine, successor_model

    with TestClient(_server().app) as client:
        models = client.get("/v1/models").json()
        assert [m["id"] for m in models["data"]] == ["nano-test"]
        assert client.post("/start_profile").status_code == 501  # engine has none

    engine = _engine(ListTokenizer({"go": [10, 11]}), successor_model())
    with TestClient(_server_with_engine(engine, eager_load=False).app) as client:
        assert client.post("/start_profile").status_code == 200  # no-op on CPU
        assert client.post("/stop_profile").status_code == 200


def test_stats_route_and_metrics_report_prefix_cache_hits():
    from test_nano_vllm_engine_batching import ListTokenizer, _engine, successor_model

    prompt = list(range(10, 19))  # two full 4-token blocks + 1
    engine = _engine(
        ListTokenizer({"p": prompt}), successor_model(), enable_prefix_cache=True
    )
    config = NanoVLLMConfig(served_model_name="nano-test", model_name="/tmp/not-used",
                            eager_load=False, enable_metrics=True)
    server = NanoVLLMServer(config, engine=engine)
    body = {"prompt": "p", "max_tokens": 2, "temperature": 0.0}
    with TestClient(server.app) as client:
        client.post("/v1/completions", json=body)
        client.post("/v1/completions", json=body)  # same prompt: 2 blocks hit
        stats = client.get("/stats").json()
        metrics = client.get("/metrics").text

    assert stats["prefix_query_tokens"] == 18
    assert stats["prefix_hit_tokens"] == 8
    assert stats["kv_blocks_used"] == 0 and stats["num_running"] == 0
    assert 'llm_engine_prefix_hit_tokens{model="nano-test"} 8.0' in metrics
