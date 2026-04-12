"""
Mock LLM serving backend for local functional validation.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


class CompletionRequest(BaseModel):
    """Request model for mock completions."""
    model: Optional[str] = None
    prompt: str | List[str]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for mock chat completions."""
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class MockLLMServer:
    """Very small mock backend for end-to-end API validation."""

    def __init__(self) -> None:
        self.model_name = os.getenv("MOCK_MODEL_NAME", "mock-llm-local")
        self.default_tokens = _get_env_int("MOCK_RESPONSE_TOKENS", 96)
        self.delay_ms = _get_env_int("MOCK_RESPONSE_DELAY_MS", 25)
        self.app = FastAPI(
            title="Mock LLM Server",
            description="CPU-friendly stub used for local functional verification",
            version="1.0.0",
        )
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model": self.model_name, "mode": "mock"}

        @self.app.get("/ready")
        async def readiness_check():
            return {"status": "ready", "model": self.model_name, "mode": "mock"}

        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(prompts[0], request.max_tokens),
                    media_type="text/event-stream",
                )

            await self._apply_delay()
            choices = []
            completion_tokens = 0
            for index, prompt in enumerate(prompts):
                text = self._generate_text(prompt, request.max_tokens)
                completion_tokens += self._estimate_tokens(text)
                choices.append(
                    {
                        "text": text,
                        "index": index,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                )

            prompt_tokens = self._estimate_tokens(prompts[0]) if prompts else 0
            response = {
                "id": f"mock-cmpl-{int(time.time() * 1000)}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model or self.model_name,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return JSONResponse(content=response)

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_completion(prompt, request.max_tokens),
                    media_type="text/event-stream",
                )

            await self._apply_delay()
            text = self._generate_text(prompt, request.max_tokens)
            prompt_tokens = self._estimate_tokens(prompt)
            completion_tokens = self._estimate_tokens(text)
            response = {
                "id": f"mock-chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return JSONResponse(content=response)

    async def _stream_completion(self, prompt: str, max_tokens: int):
        text = self._generate_text(prompt, max_tokens)
        for chunk in self._chunk_text(text):
            await self._apply_delay()
            payload = {"choices": [{"text": chunk, "index": 0, "finish_reason": None}]}
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_chat_completion(self, prompt: str, max_tokens: int):
        text = self._generate_text(prompt, max_tokens)
        for chunk in self._chunk_text(text):
            await self._apply_delay()
            payload = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": chunk},
                        "finish_reason": None,
                    }
                ]
            }
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    async def _apply_delay(self) -> None:
        await asyncio.sleep(max(0, self.delay_ms) / 1000.0)

    def _generate_text(self, prompt: str, max_tokens: int) -> str:
        budget = max(1, min(max_tokens, self.default_tokens))
        prompt_words = prompt.split()
        topic = " ".join(prompt_words[: min(8, len(prompt_words))]) or "your request"
        words = (
            f"Mock response for {topic}. "
            f"This backend is running locally for functional verification, "
            f"not GPU performance tuning."
        ).split()

        generated = []
        while len(generated) < budget:
            generated.extend(words)
        return " ".join(generated[:budget])

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunk_size = 8
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = _get_env_int("PORT", 8000)
    server = MockLLMServer()
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
