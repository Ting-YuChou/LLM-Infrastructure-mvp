"""FastAPI server for the educational nano-vLLM backend."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
import time
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, List, Optional

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
        register_engine_stats,
        start_request_tracking,
    )
except ModuleNotFoundError:
    from monitoring.metrics import (  # type: ignore[no-redef]
        add_metrics_endpoint,
        finish_request_tracking,
        record_inter_token_latency,
        record_time_to_first_token,
        register_engine_stats,
        start_request_tracking,
    )

from .config import NanoVLLMConfig
from .engine import GenerationResult, InvalidRequestError, NanoLlamaEngine
from .engine import StreamOutput
from .sampling import SamplingOptions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str | List[str]
    # None -> engine default (max_new_tokens, capped to fit max_model_len).
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[str | List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[str | List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False


@dataclass
class _StreamState:
    """What a finished stream reports in its final chunk and metrics."""

    text: str = ""
    finish_reason: str = "stop"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


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
            if hasattr(self.engine, "stats"):
                register_engine_stats(config.served_model_name, self.engine.stats)
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

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.served_model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "nano-vllm",
                        "max_model_len": self.config.max_model_len,
                    }
                ],
            }

        @self.app.get("/stats")
        async def engine_stats():
            stats = getattr(self.engine, "stats", None)
            if stats is None:
                raise HTTPException(status_code=501, detail="engine has no stats")
            return stats()

        # nsys --capture-range=cudaProfilerApi records between these calls
        # (vllm bench serve --profile hits the same paths).
        @self.app.post("/start_profile")
        async def start_profile():
            self._profile("start_profile")
            return {"status": "profiling"}

        @self.app.post("/stop_profile")
        async def stop_profile():
            self._profile("stop_profile")
            return {"status": "stopped"}

        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            self._ensure_ready()
            prompts = (
                [request.prompt] if isinstance(request.prompt, str) else request.prompt
            )
            model_name = request.model or self.config.served_model_name
            sampling = self._completion_sampling(request)
            endpoint = "/v1/completions"

            if request.stream:
                if len(prompts) != 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Streaming completions support a single prompt per request.",
                    )
                arrival = time.time()
                open_stream = self._open_stream_or_400(
                    prompts[0], sampling, model_name, endpoint, arrival
                )
                return StreamingResponse(
                    self._stream_completion(
                        open_stream, model_name, arrival, prompts[0],
                        include_usage=_include_usage(request),
                    ),
                    media_type="text/event-stream",
                )

            request_start = start_request_tracking(model_name)
            status, error_type = "error", None
            prompt_tokens = 0
            completion_tokens = 0
            try:
                choices = []
                async with self._admitted():
                    results = await self._generate_all(prompts, sampling)
                    for index, result in enumerate(results):
                        prompt_tokens += result.prompt_tokens
                        completion_tokens += result.completion_tokens
                        choices.append(
                            {
                                "text": result.text,
                                "index": index,
                                "logprobs": None,
                                "finish_reason": _finish_reason(result),
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
                status = "success"
                return JSONResponse(content=response)
            except InvalidRequestError as exc:
                error_type = type(exc).__name__
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                error_type = type(exc).__name__
                raise
            finally:
                finish_request_tracking(
                    model_name=model_name,
                    endpoint=endpoint,
                    start_time=request_start,
                    status=status,
                    error_type=error_type,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            self._ensure_ready()
            prompt = self._format_chat_prompt(request.messages)
            model_name = request.model or self.config.served_model_name
            sampling = self._chat_sampling(request)
            endpoint = "/v1/chat/completions"

            if request.stream:
                arrival = time.time()
                open_stream = self._open_stream_or_400(
                    prompt, sampling, model_name, endpoint, arrival
                )
                return StreamingResponse(
                    self._stream_chat_completion(
                        open_stream, model_name, arrival, prompt,
                        include_usage=_include_usage(request),
                    ),
                    media_type="text/event-stream",
                )

            request_start = start_request_tracking(model_name)
            status, error_type = "error", None
            result: Optional[GenerationResult] = None
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
                            "finish_reason": _finish_reason(result),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.prompt_tokens + result.completion_tokens,
                    },
                }
                status = "success"
                return JSONResponse(content=response)
            except InvalidRequestError as exc:
                error_type = type(exc).__name__
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                error_type = type(exc).__name__
                raise
            finally:
                finish_request_tracking(
                    model_name=model_name,
                    endpoint=endpoint,
                    start_time=request_start,
                    status=status,
                    error_type=error_type,
                    prompt_tokens=result.prompt_tokens if result else 0,
                    completion_tokens=result.completion_tokens if result else 0,
                )

    def _profile(self, method: str) -> None:
        action = getattr(self.engine, method, None)
        if action is None:
            raise HTTPException(status_code=501, detail=f"engine has no {method}")
        action()

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

    async def _generate_all(
        self, prompts: List[str], sampling: SamplingOptions
    ) -> List[GenerationResult]:
        tasks = [
            asyncio.ensure_future(self._generate(prompt, sampling)) for prompt in prompts
        ]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # gather() leaves siblings running; cancel them so they free KV.
            for task in tasks:
                task.cancel()
            raise

    def _open_stream_or_400(
        self,
        prompt: str,
        sampling: SamplingOptions,
        model_name: str,
        endpoint: str,
        arrival: float,
    ) -> Callable[[], AsyncIterator[str | StreamOutput]]:
        """Validate a streaming request before the 200 response starts.

        Returns a factory so nothing is submitted to the engine until the
        response body is actually iterated.
        """
        prepare = getattr(self.engine, "prepare_request", None)
        stream_request = getattr(self.engine, "stream_request", None)
        if prepare is None or stream_request is None:
            return lambda: self._stream_text(prompt, sampling)
        try:
            seq = prepare(prompt, sampling)
        except InvalidRequestError as exc:
            start_request_tracking(model_name)
            finish_request_tracking(
                model_name=model_name,
                endpoint=endpoint,
                start_time=arrival,
                status="error",
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return lambda: stream_request(seq)

    async def _stream_text(
        self,
        prompt: str,
        sampling: SamplingOptions,
    ) -> AsyncIterator[str | StreamOutput]:
        stream_async = getattr(self.engine, "generate_stream_async", None)
        if stream_async is not None:
            async with aclosing(stream_async(prompt, sampling)) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        result = await self._generate(prompt, sampling)
        for chunk in self._chunk_text(result.text):
            yield chunk
        yield StreamOutput(
            text="",
            finished=True,
            finish_reason=_finish_reason(result),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    async def _stream_chunks(
        self,
        open_stream: Callable[[], AsyncIterator[str | StreamOutput]],
        model_name: str,
        arrival: float,
        state: _StreamState,
    ) -> AsyncIterator[str]:
        """Yield non-empty text chunks, recording latency metrics and state."""
        first_token_seen = False
        last_token_time: Optional[float] = None
        async with aclosing(open_stream()) as outputs:
            async for output in outputs:
                if isinstance(output, str):
                    text = output
                else:
                    text = output.text
                    if output.finished:
                        state.finish_reason = output.finish_reason or "stop"
                        state.prompt_tokens = output.prompt_tokens
                        state.completion_tokens = output.completion_tokens
                if not text:
                    continue
                state.text += text
                now = time.time()
                if not first_token_seen:
                    record_time_to_first_token(model_name, now - arrival)
                    first_token_seen = True
                elif last_token_time is not None:
                    record_inter_token_latency(model_name, now - last_token_time)
                last_token_time = now
                yield text
        if not first_token_seen:
            record_time_to_first_token(model_name, time.time() - arrival)

    async def _tracked_stream(
        self,
        open_stream: Callable[[], AsyncIterator[str | StreamOutput]],
        model_name: str,
        endpoint: str,
        arrival: float,
        prompt_text: str,
        format_chunk: Callable[[str], dict],
        format_final: Callable[[str], dict],
        format_usage: Optional[Callable[[dict], dict]] = None,
    ) -> AsyncIterator[str]:
        """SSE body with request tracking that always balances.

        Tracking starts inside the generator and finishes in ``finally``: a
        client disconnect surfaces here as CancelledError or GeneratorExit
        (neither is an Exception), and closing the inner streams aborts the
        engine sequence so its KV blocks are freed.
        """
        start_request_tracking(model_name)
        state = _StreamState()
        status, error_type = "error", None
        try:
            async with self._admitted():
                async with aclosing(
                    self._stream_chunks(open_stream, model_name, arrival, state)
                ) as chunks:
                    async for text in chunks:
                        yield self._format_sse_payload(format_chunk(text))
            yield self._format_sse_payload(format_final(state.finish_reason))
            if format_usage is not None:
                yield self._format_sse_payload(
                    format_usage(self._stream_usage(state, prompt_text))
                )
            yield "data: [DONE]\n\n"
            status = "success"
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            usage = self._stream_usage(state, prompt_text)
            finish_request_tracking(
                model_name=model_name,
                endpoint=endpoint,
                start_time=arrival,
                status=status,
                error_type=error_type,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            )

    def _stream_usage(self, state: _StreamState, prompt_text: str) -> dict:
        prompt_tokens = (
            state.prompt_tokens
            if state.prompt_tokens is not None
            else self._estimate_prompt_tokens(prompt_text)
        )
        completion_tokens = (
            state.completion_tokens
            if state.completion_tokens is not None
            else self._estimate_completion_tokens(state.text)
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _completion_sampling(self, request: CompletionRequest) -> SamplingOptions:
        return SamplingOptions(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=_stop_list(request.stop),
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,
        )

    def _chat_sampling(self, request: ChatCompletionRequest) -> SamplingOptions:
        return SamplingOptions(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=_stop_list(request.stop),
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,
        )

    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        return (
            "\n".join(f"{message.role}: {message.content}" for message in messages)
            + "\nassistant:"
        )

    def _format_sse_payload(self, payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _stream_completion(
        self,
        open_stream: Callable[[], AsyncIterator[str | StreamOutput]],
        model_name: str,
        arrival: float,
        prompt_text: str = "",
        include_usage: bool = False,
    ) -> AsyncIterator[str]:
        completion_id = f"nano-cmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        def chunk(text: str, finish_reason: Optional[str] = None) -> dict:
            return {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }

        return self._tracked_stream(
            open_stream,
            model_name,
            "/v1/completions",
            arrival,
            prompt_text,
            format_chunk=chunk,
            format_final=lambda finish_reason: chunk("", finish_reason),
            format_usage=(
                (lambda usage: {**chunk(""), "choices": [], "usage": usage})
                if include_usage
                else None
            ),
        )

    def _stream_chat_completion(
        self,
        open_stream: Callable[[], AsyncIterator[str | StreamOutput]],
        model_name: str,
        arrival: float,
        prompt_text: str = "",
        include_usage: bool = False,
    ) -> AsyncIterator[str]:
        completion_id = f"nano-chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        sent_role = False

        def chunk(delta: dict, finish_reason: Optional[str] = None) -> dict:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }

        def content_chunk(text: str) -> dict:
            nonlocal sent_role
            delta = {"content": text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            return chunk(delta)

        return self._tracked_stream(
            open_stream,
            model_name,
            "/v1/chat/completions",
            arrival,
            prompt_text,
            format_chunk=content_chunk,
            format_final=lambda finish_reason: chunk({}, finish_reason),
            format_usage=(
                (lambda usage: {**chunk({}), "choices": [], "usage": usage})
                if include_usage
                else None
            ),
        )

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


def _finish_reason(result: GenerationResult) -> str:
    return getattr(result, "finish_reason", None) or "stop"


def _include_usage(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return bool(request.stream_options and request.stream_options.include_usage)


def _stop_list(stop: Optional[str | List[str]]) -> Optional[List[str]]:
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else list(stop)


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
    # Diagnostic (benchmarks/phase2): a shorter GIL switch interval than the
    # default 5 ms shows whether the event loop starves the model thread.
    switch_interval = os.getenv("NANO_SWITCH_INTERVAL")
    if switch_interval:
        sys.setswitchinterval(float(switch_interval))
    server = NanoVLLMServer(NanoVLLMConfig.from_yaml(args.config))
    server.run()


if __name__ == "__main__":
    main()
