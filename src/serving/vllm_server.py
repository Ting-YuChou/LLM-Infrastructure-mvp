"""
vLLM Inference Server
High-performance LLM serving with PagedAttention and continuous batching
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    SamplingParams = Any  # type: ignore[assignment]
    AsyncEngineArgs = None  # type: ignore[assignment]
    AsyncLLMEngine = None  # type: ignore[assignment]
    VLLM_IMPORT_ERROR = exc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models (OpenAI API Compatible)
# ============================================================================

class CompletionRequest(BaseModel):
    """Request model for /v1/completions endpoint"""
    model: Optional[str] = None
    prompt: str | List[str]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = False
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "llama2-7b",
                "prompt": "Explain quantum computing",
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": False,
            }
        }
    }


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for /v1/chat/completions endpoint"""
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    """Completion choice in response"""
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: str


class CompletionResponse(BaseModel):
    """Response model for completions"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VLLMServerConfig:
    """Configuration for vLLM server"""
    model_name: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    tokenizer_revision: Optional[str] = None
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None
    load_format: str = "auto"
    cpu_offload_gb: float = 0.0
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    lora_dtype: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4.0
    enforce_eager: bool = False
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    enable_prefix_caching: bool = True
    scheduling_policy: str = "fcfs"
    enable_chunked_prefill: bool = False
    disable_log_stats: bool = False
    disable_custom_all_reduce: bool = False
    seed: int = 0
    max_logprobs: int = 5
    max_parallel_loading_workers: Optional[int] = None
    distributed_init_method: str = "auto"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VLLMServerConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config.get('model', {})
        server_config = config.get('server', {})
        gpu_config = config.get('gpu', {})
        paged_attention_config = config.get('paged_attention', {})
        batching_config = config.get('batching', {})
        optimization_config = config.get('optimization', {})
        distributed_config = config.get('distributed', {})
        engine_config = config.get('engine', {})
        loading_config = config.get('loading', {})

        return cls(
            model_name=model_config['name'],
            tokenizer=model_config.get('tokenizer'),
            tokenizer_mode=engine_config.get('tokenizer_mode', 'auto'),
            tokenizer_revision=engine_config.get('tokenizer_revision'),
            trust_remote_code=model_config.get('trust_remote_code', False),
            download_dir=model_config.get('download_dir'),
            dtype=model_config.get('dtype', 'auto'),
            quantization=model_config.get('quantization'),
            max_model_len=model_config.get('max_model_len'),
            load_format=loading_config.get('load_format', 'auto'),
            cpu_offload_gb=loading_config.get('cpu_offload_gb', 0.0),
            enable_lora=loading_config.get('enable_lora', False),
            max_loras=loading_config.get('max_loras', 1),
            max_lora_rank=loading_config.get('max_lora_rank', 16),
            lora_dtype=loading_config.get('lora_dtype', 'auto'),
            host=server_config.get('host', '0.0.0.0'),
            port=server_config.get('port', 8000),
            tensor_parallel_size=gpu_config.get('tensor_parallel_size', 1),
            pipeline_parallel_size=gpu_config.get('pipeline_parallel_size', 1),
            gpu_memory_utilization=gpu_config.get('gpu_memory_utilization', 0.9),
            swap_space=gpu_config.get('swap_space', 4.0),
            enforce_eager=gpu_config.get('enforce_eager', False),
            max_num_seqs=gpu_config.get('max_num_seqs', 256),
            max_num_batched_tokens=gpu_config.get('max_num_batched_tokens', 8192),
            block_size=paged_attention_config.get('block_size', 16),
            enable_prefix_caching=paged_attention_config.get('enable_prefix_caching', True),
            scheduling_policy=batching_config.get('scheduling_policy', 'fcfs'),
            enable_chunked_prefill=batching_config.get('enable_chunked_prefill', False),
            disable_log_stats=engine_config.get(
                'disable_log_stats',
                optimization_config.get('disable_log_stats', False),
            ),
            disable_custom_all_reduce=optimization_config.get('disable_custom_all_reduce', False),
            seed=engine_config.get('seed', 0),
            max_logprobs=engine_config.get('max_logprobs', 5),
            max_parallel_loading_workers=engine_config.get('max_parallel_loading_workers'),
            distributed_init_method=distributed_config.get('distributed_init_method', 'auto'),
        )

    def to_async_engine_kwargs(self) -> Dict[str, Any]:
        """Translate config into desired AsyncEngineArgs kwargs."""
        engine_kwargs = {
            "model": self.model_name,
            "tokenizer": self.tokenizer,
            "tokenizer_mode": self.tokenizer_mode,
            "tokenizer_revision": self.tokenizer_revision,
            "trust_remote_code": self.trust_remote_code,
            "download_dir": self.download_dir,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "load_format": self.load_format,
            "cpu_offload_gb": self.cpu_offload_gb,
            "enable_lora": self.enable_lora,
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "lora_dtype": self.lora_dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": self.swap_space,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "block_size": self.block_size,
            "enable_prefix_caching": self.enable_prefix_caching,
            "scheduling_policy": self.scheduling_policy,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "disable_log_stats": self.disable_log_stats,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
            "seed": self.seed,
            "max_logprobs": self.max_logprobs,
            "max_parallel_loading_workers": self.max_parallel_loading_workers,
            "distributed_init_method": self.distributed_init_method,
        }
        return {key: value for key, value in engine_kwargs.items() if value is not None}

    def build_async_engine_kwargs(self, engine_args_cls: Any | None = None) -> Dict[str, Any]:
        """
        Filter desired engine kwargs against the installed AsyncEngineArgs signature.

        This keeps the config forward-compatible while avoiding runtime failures
        when a pinned vLLM version does not support newer options yet.
        """
        desired_kwargs = self.to_async_engine_kwargs()
        target_cls = engine_args_cls or AsyncEngineArgs
        if target_cls is None:
            return desired_kwargs

        supported_kwargs = _get_supported_async_engine_arg_names(target_cls)
        if supported_kwargs is None:
            logger.warning(
                "Could not inspect AsyncEngineArgs signature; applying desired "
                "engine kwargs without compatibility filtering."
            )
            return desired_kwargs

        unsupported_kwargs = sorted(set(desired_kwargs) - supported_kwargs)
        if unsupported_kwargs:
            logger.warning(
                "Skipping unsupported AsyncEngineArgs kwargs for this vLLM "
                "version: %s",
                ", ".join(unsupported_kwargs),
            )

        return {
            key: value
            for key, value in desired_kwargs.items()
            if key in supported_kwargs
        }


def _get_supported_async_engine_arg_names(engine_args_cls: Any) -> Optional[set[str]]:
    """Return supported AsyncEngineArgs parameter names for runtime filtering."""
    try:
        signature = inspect.signature(engine_args_cls)
    except (TypeError, ValueError):
        try:
            signature = inspect.signature(engine_args_cls.__init__)
        except (AttributeError, TypeError, ValueError):
            return None

    parameter_names = set()
    has_var_kwargs = False
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            parameter_names.add(name)

    if has_var_kwargs:
        return None
    return parameter_names


# ============================================================================
# vLLM Inference Engine
# ============================================================================

class VLLMInferenceEngine:
    """Manages vLLM inference engine"""
    
    def __init__(self, config: VLLMServerConfig):
        """
        Initialize vLLM engine
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        
        logger.info(f"Initializing vLLM engine for model: {config.model_name}")
        logger.info(f"Tensor parallel size: {config.tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {config.gpu_memory_utilization}")
    
    async def initialize(self):
        """Initialize the async engine"""
        if VLLM_IMPORT_ERROR is not None or AsyncEngineArgs is None or AsyncLLMEngine is None:
            raise RuntimeError(
                "vLLM is not installed. Install it before starting the serving "
                "stack."
            ) from VLLM_IMPORT_ERROR

        # Configure engine arguments
        engine_kwargs = self.config.build_async_engine_kwargs(AsyncEngineArgs)
        logger.info(
            "Applying AsyncEngineArgs settings: %s",
            ", ".join(sorted(engine_kwargs)),
        )
        engine_args = AsyncEngineArgs(**engine_kwargs)
        
        # Create engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM engine initialized successfully")
    
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams
    ) -> str:
        """
        Generate text from a single prompt
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            
        Returns:
            Generated text
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        # Generate
        results = []
        async for request_output in self.engine.generate(prompt, sampling_params, request_id=None):
            results.append(request_output)
        
        # Get final output
        if results:
            return results[-1].outputs[0].text
        return ""
    
    async def stream_generate(
        self,
        prompt: str,
        sampling_params: SamplingParams
    ):
        """
        Stream generate text from a prompt
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            
        Yields:
            Generated text chunks
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        previous_text = ""
        async for request_output in self.engine.generate(prompt, sampling_params, request_id=None):
            current_text = request_output.outputs[0].text
            if current_text.startswith(previous_text):
                delta = current_text[len(previous_text):]
            else:
                delta = current_text
            previous_text = current_text
            if delta:
                yield delta


# ============================================================================
# FastAPI Server
# ============================================================================

class VLLMServer:
    """FastAPI server for vLLM inference"""
    
    def __init__(self, config: VLLMServerConfig):
        """
        Initialize server
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.app = FastAPI(
            title="vLLM Inference Server",
            description="High-performance LLM inference with PagedAttention",
            version="1.0.0"
        )
        self.engine = VLLMInferenceEngine(config)
        add_metrics_endpoint(self.app)
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info(f"FastAPI server initialized on {config.host}:{config.port}")
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize engine on startup"""
            await self.engine.initialize()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "model": self.config.model_name}
        
        @self.app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint"""
            if self.engine.engine is None:
                raise HTTPException(status_code=503, detail="Engine not ready")
            return {"status": "ready"}
        
        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """
            OpenAI-compatible completions endpoint
            
            Generates text completion from a prompt
            """
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop=request.stop or [],
            )

            model_name = request.model or self.config.model_name

            # Handle streaming
            if request.stream:
                if not isinstance(request.prompt, str):
                    raise HTTPException(
                        status_code=400,
                        detail="Streaming completions support a single prompt per request.",
                    )
                request_start = start_request_tracking(model_name)
                return StreamingResponse(
                    self._stream_completion(
                        prompt=request.prompt,
                        sampling_params=sampling_params,
                        model_name=model_name,
                        request_start=request_start,
                    ),
                    media_type="text/event-stream"
                )

            request_start = start_request_tracking(model_name)
            # Non-streaming generation
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

            try:
                # Generate for all prompts
                outputs = []
                for i, prompt in enumerate(prompts):
                    text = await self.engine.generate(prompt, sampling_params)
                    outputs.append(
                        CompletionChoice(
                            text=text,
                            index=i,
                            finish_reason="stop"
                        )
                    )

                prompt_tokens = sum(self._estimate_tokens(prompt) for prompt in prompts)
                completion_tokens = sum(self._estimate_tokens(choice.text) for choice in outputs)

                # Build response
                response = CompletionResponse(
                    id=f"cmpl-{int(time.time())}",
                    created=int(time.time()),
                    model=model_name,
                    choices=outputs,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                )

                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/completions",
                    start_time=request_start,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                return response
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
            """
            OpenAI-compatible chat completions endpoint
            
            Generates chat completion from messages
            """
            # Convert messages to prompt
            prompt = self._format_chat_prompt(request.messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop or [],
            )

            model_name = request.model or self.config.model_name

            if request.stream:
                request_start = start_request_tracking(model_name)
                return StreamingResponse(
                    self._stream_chat_completion(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        model_name=model_name,
                        request_start=request_start,
                    ),
                    media_type="text/event-stream",
                )

            request_start = start_request_tracking(model_name)

            try:
                # Generate
                text = await self.engine.generate(prompt, sampling_params)
                prompt_tokens = self._estimate_tokens(prompt)
                completion_tokens = self._estimate_tokens(text)

                # Build response
                response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }

                finish_request_tracking(
                    model_name=model_name,
                    endpoint="/v1/chat/completions",
                    start_time=request_start,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
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
    
    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Format chat messages into a prompt string
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt
        """
        # Simple formatting (can be customized per model)
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for metrics and mock usage reporting."""
        return max(1, len(text.split())) if text else 0

    def _format_sse_payload(self, payload: Dict[str, Any]) -> str:
        """Serialize one Server-Sent Event payload as JSON."""
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _stream_completion(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        model_name: str,
        request_start: float,
    ):
        """
        Stream completion chunks
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            
        Yields:
            SSE-formatted completion chunks
        """
        completion_id = f"cmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        prompt_tokens = self._estimate_tokens(prompt)
        completion_text = ""
        first_token_seen = False
        last_token_time: Optional[float] = None

        try:
            async for text in self.engine.stream_generate(prompt, sampling_params):
                now = time.time()
                if not first_token_seen:
                    record_time_to_first_token(model_name, now - request_start)
                    first_token_seen = True
                elif last_token_time is not None:
                    record_inter_token_latency(model_name, now - last_token_time)
                last_token_time = now
                completion_text += text
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                yield self._format_sse_payload(chunk)

            yield self._format_sse_payload(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }],
                }
            )
            yield "data: [DONE]\n\n"
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/completions",
                start_time=request_start,
                prompt_tokens=prompt_tokens,
                completion_tokens=self._estimate_tokens(completion_text),
            )
        except Exception as exc:
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/completions",
                start_time=request_start,
                status="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=self._estimate_tokens(completion_text),
                error_type=type(exc).__name__,
            )
            raise

    async def _stream_chat_completion(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        model_name: str,
        request_start: float,
    ):
        """Stream chat completion chunks using JSON SSE payloads."""
        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        prompt_tokens = self._estimate_tokens(prompt)
        completion_text = ""
        sent_role = False
        first_token_seen = False
        last_token_time: Optional[float] = None

        try:
            async for text in self.engine.stream_generate(prompt, sampling_params):
                now = time.time()
                if not first_token_seen:
                    record_time_to_first_token(model_name, now - request_start)
                    first_token_seen = True
                elif last_token_time is not None:
                    record_inter_token_latency(model_name, now - last_token_time)
                last_token_time = now
                completion_text += text
                delta: Dict[str, str] = {"content": text}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True

                yield self._format_sse_payload(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }],
                    }
                )

            yield self._format_sse_payload(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
            )
            yield "data: [DONE]\n\n"
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/chat/completions",
                start_time=request_start,
                prompt_tokens=prompt_tokens,
                completion_tokens=self._estimate_tokens(completion_text),
            )
        except Exception as exc:
            finish_request_tracking(
                model_name=model_name,
                endpoint="/v1/chat/completions",
                start_time=request_start,
                status="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=self._estimate_tokens(completion_text),
                error_type=type(exc).__name__,
            )
            raise
    
    def run(self):
        """Start the server"""
        logger.info(f"Starting vLLM server on {self.config.host}:{self.config.port}")
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Inference Server")
    parser.add_argument(
        "--config",
        type=str,
        default="config/serving_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name/path (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = VLLMServerConfig.from_yaml(args.config)
    
    # Override with command-line arguments
    if args.model:
        config.model_name = args.model
    if args.port:
        config.port = args.port
    
    # Create and run server
    server = VLLMServer(config)
    server.run()


if __name__ == "__main__":
    main()
