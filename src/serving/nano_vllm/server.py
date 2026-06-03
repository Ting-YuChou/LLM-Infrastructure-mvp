"""FastAPI server for the educational nano-vLLM backend."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    from src.monitoring.metrics import (
        add_metrics_endpoint,
        finish_request_tracking,
        record_inter_token_latency,
        record_time_to_first_token,
        start_request_tracking,
    )
except ModuleNotFoundError:
    from monitoring.metrics import (  # type: ignore[no-redef]
        add_metrics_endpoint,
        finish_request_tracking,
        record_inter_token_latency,
        record_time_to_first_token,
        start_request_tracking,
    )

from .config import NanoVLLMConfig
from .engine import GenerationResult, NanoLlamaEngine
from .sampling import SamplingOptions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str | List[str]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stream: bool = False
    stop: Optional[List[str]] = None


class NanoVLLMServer:
    """OpenAI-like API wrapper around the nano decoder."""

    def __init__(self, config: NanoVLLMConfig, engine: Any | None = None) -> None:
        self.config = config
        self.engine = engine or NanoLlamaEngine(config)
        self.startup_error: Optional[str] = None
        self.admission = asyncio.Semaphore(max(1, config.max_concurrent_requests))
        self.app = FastAPI(
            title="nano-vLLM Server",
            description="Educational Llama-family backend with PyTorch/Triton kernels",
            version="0.1.0",
            lifespan=self._lifespan,
        )
        if config.enable_metrics:
            add_metrics_endpoint(self.app)
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        del app
        if self.config.eager_load:
            await self._initialize_engine()
        try:
            yield
        finally:
            shutdown = getattr(self.engine, "shutdown", None)
            if shutdown is not None:
                maybe_awaitable = shutdown()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable

    async def _initialize_engine(self) -> None:
        try:
            maybe_awaitable = self.engine.initialize()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
            logger.info("nano-vLLM engine initialized")
        except Exception as exc:  # Keep /health available for diagnostics.
            self.startup_error = str(exc)
            logger.exception("nano-vLLM engine failed to initialize")

    def _register_routes(self) -> None:
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model": self.config.served_model_name,
                "mode": "nano-vllm",
            }

        @self.app.get("/ready")
        async def readiness_check():
            if self.startup_error:
                raise HTTPException(
                    status_code=503,
                    detail=f"nano-vLLM engine is not ready: {self.startup_error}",
                )
            if not getattr(self.engine, "initialized", False):
                raise HTTPException(
                    status_code=503, detail="nano-vLLM engine is not initialized"
                )
            return {
                "status": "ready",
                "model": self.config.served_model_name,
                "mode": "nano-vllm",
            }

        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            self._ensure_ready()
            prompts = (
                [request.prompt] if isinstance(request.prompt, str) else request.prompt
            )
            model_name = request.model or self.config.served_model_name
            sampling = self._completion_sampling(request)

            if request.stream:
                if len(prompts) != 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Streaming completions support a single prompt per request.",
                    )
                request_start = start_request_tracking(model_name)
                return StreamingResponse(
                    self._stream_completion(
                        prompts[0], sampling, model_name, request_start
                    ),
                    media_type="text/event-stream",
                )

            request_start = start_request_tracking(model_name)
            try:
                choices = []
                prompt_tokens = 0
                completion_tokens = 0
                async with self._admitted():
                    results = await asyncio.gather(
                        *(self._generate(prompt, sampling) for prompt in prompts)
                    )
                    for index, result in enumerate(results):
                        prompt_tokens += result.prompt_tokens
                        completion_tokens += result.completion_tokens
                        choices.append(
                            {
                                "text": result.text,
                                "index": index,
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        )

                response = {
                    "id": f"nano-cmpl-{int(time.time() * 1000)}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/completions",
                    start_time=request_start,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                return JSONResponse(content=response)
            except Exception as exc:
                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/completions",
                    start_time=request_start,
                    status="error",
                    error_type=type(exc).__name__,
                )
                raise

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            self._ensure_ready()
            prompt = self._format_chat_prompt(request.messages)
            model_name = request.model or self.config.served_model_name
            sampling = self._chat_sampling(request)

            if request.stream:
                request_start = start_request_tracking(model_name)
                return StreamingResponse(
                    self._stream_chat_completion(
                        prompt, sampling, model_name, request_start
                    ),
                    media_type="text/event-stream",
                )

            request_start = start_request_tracking(model_name)
            try:
                async with self._admitted():
                    result = await self._generate(prompt, sampling)
                response = {
                    "id": f"nano-chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result.text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.prompt_tokens + result.completion_tokens,
                    },
                }
                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/chat/completions",
                    start_time=request_start,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
                return JSONResponse(content=response)
            except Exception as exc:
                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/chat/completions",
                    start_time=request_start,
                    status="error",
                    error_type=type(exc).__name__,
                )
                raise

    def _ensure_ready(self) -> None:
        if self.startup_error:
            raise HTTPException(status_code=503, detail=self.startup_error)
        if not getattr(self.engine, "initialized", False):
            raise HTTPException(
                status_code=503, detail="nano-vLLM engine is not initialized"
            )

    async def _generate(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ) -> GenerationResult:
        generate_async = getattr(self.engine, "generate_async", None)
        if generate_async is not None:
            return await generate_async(prompt, sampling)
        return await asyncio.to_thread(self.engine.generate, prompt, sampling)

    async def _stream_text(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ):
        stream_async = getattr(self.engine, "generate_stream_async", None)
        if stream_async is not None:
            async for chunk in stream_async(prompt, sampling):
                yield chunk
            return
        result = await self._generate(prompt, sampling)
        for chunk in self._chunk_text(result.text):
            yield chunk

    def _completion_sampling(self, request: CompletionRequest) -> SamplingOptions:
        return SamplingOptions(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
        )

    def _chat_sampling(self, request: ChatCompletionRequest) -> SamplingOptions:
        return SamplingOptions(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
        )

    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        return (
            "\n".join(f"{message.role}: {message.content}" for message in messages)
            + "\nassistant:"
        )

    def _format_sse_payload(self, payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _stream_completion(
        self,
        prompt: str,
        sampling: SamplingOptions,
        model_name: str,
        request_start: float,
    ):
        completion_id = f"nano-cmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        result: Optional[GenerationResult] = None
        first_token_seen = False
        last_token_time: Optional[float] = None
        try:
            completion_text = ""
            async with self._admitted():
                async for chunk in self._stream_text(prompt, sampling):
                    completion_text += chunk
                    now = time.time()
                    if not first_token_seen:
                        record_time_to_first_token(model_name, now - request_start)
                        first_token_seen = True
                    elif last_token_time is not None:
                        record_inter_token_latency(model_name, now - last_token_time)
                    last_token_time = now
                    yield self._format_sse_payload(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": chunk,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            prompt_tokens = self._estimate_prompt_tokens(prompt)
            completion_tokens = self._estimate_completion_tokens(completion_text)
            if result is not None:
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
            if not first_token_seen:
                now = time.time()
                record_time_to_first_token(model_name, now - request_start)
            yield self._format_sse_payload(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            yield "data: [DONE]\n\n"
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/completions",
                start_time=request_start,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/completions",
                start_time=request_start,
                status="error",
                error_type=type(exc).__name__,
            )
            raise

    async def _stream_chat_completion(
        self,
        prompt: str,
        sampling: SamplingOptions,
        model_name: str,
        request_start: float,
    ):
        completion_id = f"nano-chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        sent_role = False
        result: Optional[GenerationResult] = None
        try:
            completion_text = ""
            async with self._admitted():
                async for chunk in self._stream_text(prompt, sampling):
                    completion_text += chunk
                    delta = {"content": chunk}
                    if not sent_role:
                        delta["role"] = "assistant"
                        sent_role = True
                    yield self._format_sse_payload(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                        }
                    )
            yield self._format_sse_payload(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/chat/completions",
                start_time=request_start,
                prompt_tokens=(
                    result.prompt_tokens
                    if result is not None
                    else self._estimate_prompt_tokens(prompt)
                ),
                completion_tokens=(
                    result.completion_tokens
                    if result is not None
                    else self._estimate_completion_tokens(completion_text)
                ),
            )
        except Exception as exc:
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/chat/completions",
                start_time=request_start,
                status="error",
                error_type=type(exc).__name__,
            )
            raise

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        chunk_size = 8
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    def _estimate_prompt_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _estimate_completion_tokens(self, text: str) -> int:
        return max(0, len(text.split()))

    def _admitted(self):
        return _SemaphoreContext(self.admission)

    def run(self) -> None:
        uvicorn.run(
            self.app, host=self.config.host, port=self.config.port, log_level="info"
        )


class _SemaphoreContext:
    def __init__(self, semaphore: asyncio.Semaphore):
        self.semaphore = semaphore

    async def __aenter__(self):
        await self.semaphore.acquire()

    async def __aexit__(self, exc_type, exc, traceback):
        self.semaphore.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nano-vLLM server")
    parser.add_argument(
        "--config",
        default=os.getenv("NANO_VLLM_CONFIG", "config/nano_vllm_config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = NanoVLLMServer(NanoVLLMConfig.from_yaml(args.config))
    server.run()


if __name__ == "__main__":
    main()
