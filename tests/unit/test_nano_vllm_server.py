import sys
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
