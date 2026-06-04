import sys
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.engine import GenerationResult  # noqa: E402
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
