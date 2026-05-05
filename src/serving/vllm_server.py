"""
vLLM Inference Server
High-performance LLM serving with PagedAttention and continuous batching
"""

import os
import yaml
import logging
import asyncio
import inspect
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, fields, is_dataclass
from contextlib import asynccontextmanager, suppress

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

try:
    from src.monitoring import metrics as monitoring_metrics
except ModuleNotFoundError as exc:  # Allows running with src/ directly on PYTHONPATH.
    if exc.name not in {"src", "src.monitoring"}:
        raise
    from monitoring import metrics as monitoring_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _optional_str(value: Any) -> Optional[str]:
    """Normalize empty YAML/env values to None."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _optional_int(value: Any) -> Optional[int]:
    value = _optional_str(value)
    return int(value) if value is not None else None


def _optional_float(value: Any) -> Optional[float]:
    value = _optional_str(value)
    return float(value) if value is not None else None


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)


# ============================================================================
# Request/Response Models (OpenAI API Compatible)
# ============================================================================

class CompletionRequest(BaseModel):
    """Request model for /v1/completions endpoint"""
    model: str
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
    
    class Config:
        schema_extra = {
            "example": {
                "model": "llama2-7b",
                "prompt": "Explain quantum computing",
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": False
            }
        }


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for /v1/chat/completions endpoint"""
    model: str
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
    revision: Optional[str] = None
    tokenizer: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4
    enforce_eager: bool = False
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    enable_prefix_caching: bool = True
    load_format: str = "auto"
    seed: int = 0
    max_concurrent_requests: int = 1000
    enable_metrics: bool = True
    metrics_port: int = 9090
    gpu_metrics_interval: float = 15.0
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VLLMServerConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config.get('model', {})
        server_config = config.get('server', {})
        gpu_config = config.get('gpu', {})
        paged_attention_config = config.get('paged_attention', {})
        loading_config = config.get('loading', {})
        engine_config = config.get('engine', {})
        monitoring_config = config.get('monitoring', {})
        
        return cls(
            model_name=_env('MODEL_NAME', model_config['name']),
            revision=_optional_str(_env('MODEL_REVISION', model_config.get('revision'))),
            tokenizer=_optional_str(_env('TOKENIZER', model_config.get('tokenizer'))),
            tokenizer_revision=_optional_str(
                _env('TOKENIZER_REVISION', engine_config.get('tokenizer_revision'))
            ),
            trust_remote_code=_bool_value(
                _env('TRUST_REMOTE_CODE', model_config.get('trust_remote_code')),
                default=False,
            ),
            download_dir=_optional_str(_env('DOWNLOAD_DIR', model_config.get('download_dir'))),
            dtype=_env('DTYPE', model_config.get('dtype', 'auto')),
            quantization=_optional_str(_env('QUANTIZATION', model_config.get('quantization'))),
            max_model_len=_optional_int(_env('MAX_MODEL_LEN', model_config.get('max_model_len'))),
            host=_env('HOST', server_config.get('host', '0.0.0.0')),
            port=int(_env('PORT', server_config.get('port', 8000))),
            tensor_parallel_size=int(
                _env('TENSOR_PARALLEL_SIZE', gpu_config.get('tensor_parallel_size', 1))
            ),
            pipeline_parallel_size=int(
                _env('PIPELINE_PARALLEL_SIZE', gpu_config.get('pipeline_parallel_size', 1))
            ),
            gpu_memory_utilization=float(
                _env('GPU_MEMORY_UTILIZATION', gpu_config.get('gpu_memory_utilization', 0.9))
            ),
            swap_space=int(_env('SWAP_SPACE', gpu_config.get('swap_space', 4))),
            enforce_eager=_bool_value(
                _env('ENFORCE_EAGER', gpu_config.get('enforce_eager')),
                default=False,
            ),
            max_num_seqs=int(_env('MAX_NUM_SEQS', gpu_config.get('max_num_seqs', 256))),
            max_num_batched_tokens=int(
                _env('MAX_NUM_BATCHED_TOKENS', gpu_config.get('max_num_batched_tokens', 8192))
            ),
            block_size=int(_env('BLOCK_SIZE', paged_attention_config.get('block_size', 16))),
            enable_prefix_caching=_bool_value(
                _env('ENABLE_PREFIX_CACHING', paged_attention_config.get('enable_prefix_caching')),
                default=True,
            ),
            load_format=_env('LOAD_FORMAT', loading_config.get('load_format', 'auto')),
            seed=int(_env('SEED', engine_config.get('seed', 0))),
            max_concurrent_requests=int(
                _env('MAX_CONCURRENT_REQUESTS', server_config.get('max_concurrent_requests', 1000))
            ),
            enable_metrics=_bool_value(
                _env('ENABLE_METRICS', monitoring_config.get('enable_metrics')),
                default=True,
            ),
            metrics_port=int(_env('METRICS_PORT', monitoring_config.get('metrics_port', 9090))),
            gpu_metrics_interval=float(
                _env('GPU_METRICS_INTERVAL', monitoring_config.get('gpu_metrics_interval', 15.0))
            ),
        )


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
        logger.info(f"Quantization: {config.quantization or 'none'}")
        logger.info(f"Dtype: {config.dtype}")
        logger.info(f"Tensor parallel size: {config.tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {config.gpu_memory_utilization}")
    
    async def initialize(self):
        """Initialize the async engine"""
        # Configure engine arguments
        engine_arg_values = self._build_engine_args()
        engine_args = AsyncEngineArgs(**engine_arg_values)
        
        # Create engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM engine initialized successfully")

    def _build_engine_args(self) -> Dict[str, Any]:
        """Build engine arguments and guard against unsupported vLLM versions."""
        engine_arg_values = {
            "model": self.config.model_name,
            "revision": self.config.revision,
            "tokenizer": self.config.tokenizer,
            "tokenizer_revision": self.config.tokenizer_revision,
            "trust_remote_code": self.config.trust_remote_code,
            "download_dir": self.config.download_dir,
            "dtype": self.config.dtype,
            "quantization": self.config.quantization,
            "max_model_len": self.config.max_model_len,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "swap_space": self.config.swap_space,
            "enforce_eager": self.config.enforce_eager,
            "max_num_seqs": self.config.max_num_seqs,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "block_size": self.config.block_size,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "load_format": self.config.load_format,
            "seed": self.config.seed,
        }

        if is_dataclass(AsyncEngineArgs):
            supported_args = {field.name for field in fields(AsyncEngineArgs)}
        else:
            supported_args = set(inspect.signature(AsyncEngineArgs).parameters)

        unsupported_args = sorted(set(engine_arg_values) - supported_args)
        if self.config.quantization and "quantization" in unsupported_args:
            raise RuntimeError(
                "The installed vLLM version does not support the 'quantization' "
                "engine argument. Upgrade vLLM before using AWQ quantization."
            )

        if unsupported_args:
            logger.warning(
                "Skipping unsupported vLLM engine args for this installed version: %s",
                ", ".join(unsupported_args),
            )

        filtered_args = {
            key: value
            for key, value in engine_arg_values.items()
            if key in supported_args and value is not None
        }

        logger.info(
            "Using vLLM engine args: %s",
            {
                key: value
                for key, value in filtered_args.items()
                if key not in {"trust_remote_code"}
            },
        )
        return filtered_args
    
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
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
        async for request_output in self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id or str(uuid.uuid4()),
        ):
            results.append(request_output)
        
        # Get final output
        if results:
            return results[-1].outputs[0].text
        return ""
    
    async def stream_generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
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
        
        async for request_output in self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id or str(uuid.uuid4()),
        ):
            yield request_output.outputs[0].text


# ============================================================================
# FastAPI Server
# ============================================================================

class RequestAdmissionController:
    """Bounded admission queue used for autoscaling metrics."""

    def __init__(self, max_concurrent_requests: int, model_name: str):
        self.model_name = model_name
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))
        self._lock = asyncio.Lock()
        self._queued_requests = 0
        self._active_requests = 0
        self._set_gauges()

    def _set_gauges(self):
        monitoring_metrics.request_queue_size.labels(model=self.model_name).set(
            self._queued_requests
        )
        monitoring_metrics.active_requests.labels(model=self.model_name).set(
            self._active_requests
        )

    @asynccontextmanager
    async def admitted(self):
        queued = False
        acquired = False
        active = False

        async with self._lock:
            self._queued_requests += 1
            queued = True
            self._set_gauges()

        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._queued_requests -= 1
                self._active_requests += 1
                queued = False
                active = True
                self._set_gauges()
            yield
        finally:
            if active:
                self._semaphore.release()
                async with self._lock:
                    self._active_requests -= 1
                    self._set_gauges()
            elif acquired:
                self._semaphore.release()

            if queued:
                async with self._lock:
                    self._queued_requests -= 1
                    self._set_gauges()


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
        self.admission = RequestAdmissionController(
            config.max_concurrent_requests,
            config.model_name,
        )
        self.metrics_server = (
            monitoring_metrics.MetricsServer(config.metrics_port)
            if config.enable_metrics
            else None
        )
        self._gpu_metrics_task: Optional[asyncio.Task] = None
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        if config.enable_metrics:
            monitoring_metrics.add_metrics_endpoint(self.app)

        # Register routes
        self._register_routes()
        
        logger.info(f"FastAPI server initialized on {config.host}:{config.port}")
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize engine on startup"""
            if self.metrics_server is not None:
                self.metrics_server.start()
                if self.config.gpu_metrics_interval > 0:
                    self._gpu_metrics_task = asyncio.create_task(
                        self._run_gpu_metrics_sampler()
                    )
            await self.engine.initialize()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Stop background metric samplers."""
            if self._gpu_metrics_task is not None:
                self._gpu_metrics_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._gpu_metrics_task
        
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
            
            # Handle streaming
            if request.stream:
                stream_prompt = (
                    request.prompt[0]
                    if isinstance(request.prompt, list)
                    else request.prompt
                )
                return StreamingResponse(
                    self._stream_completion(stream_prompt, sampling_params),
                    media_type="text/event-stream"
                )
            
            # Non-streaming generation
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

            endpoint = "/v1/completions"
            start_time = time.time()
            status = "success"

            try:
                async with self.admission.admitted():
                    # Generate for all prompts in the request.
                    outputs = []
                    for i, prompt in enumerate(prompts):
                        text = await self.engine.generate(
                            prompt,
                            sampling_params,
                            request_id=f"cmpl-{uuid.uuid4()}-{i}",
                        )
                        outputs.append(
                            CompletionChoice(
                                text=text,
                                index=i,
                                finish_reason="stop"
                            )
                        )

                prompt_tokens = sum(self._estimate_tokens(prompt) for prompt in prompts)
                completion_tokens = sum(self._estimate_tokens(c.text) for c in outputs)
                monitoring_metrics.track_tokens(
                    self.config.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Build response
                response = CompletionResponse(
                    id=f"cmpl-{int(time.time())}",
                    created=int(time.time()),
                    model=self.config.model_name,
                    choices=outputs,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                )

                return response
            except Exception as e:
                status = "error"
                monitoring_metrics.error_counter.labels(
                    model=self.config.model_name,
                    error_type=type(e).__name__,
                ).inc()
                raise
            finally:
                self._record_request_metrics(
                    endpoint=endpoint,
                    status=status,
                    duration=time.time() - start_time,
                )
        
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
            
            endpoint = "/v1/chat/completions"
            start_time = time.time()
            status = "success"

            try:
                async with self.admission.admitted():
                    text = await self.engine.generate(
                        prompt,
                        sampling_params,
                        request_id=f"chatcmpl-{uuid.uuid4()}",
                    )

                prompt_tokens = self._estimate_tokens(prompt)
                completion_tokens = self._estimate_tokens(text)
                monitoring_metrics.track_tokens(
                    self.config.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Build response
                response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.config.model_name,
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
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                }

                return JSONResponse(content=response)
            except Exception as e:
                status = "error"
                monitoring_metrics.error_counter.labels(
                    model=self.config.model_name,
                    error_type=type(e).__name__,
                ).inc()
                raise
            finally:
                self._record_request_metrics(
                    endpoint=endpoint,
                    status=status,
                    duration=time.time() - start_time,
                )
    
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Use a cheap estimate until tokenizer-accurate accounting is wired."""
        return len(text.split())

    def _record_request_metrics(self, endpoint: str, status: str, duration: float):
        monitoring_metrics.request_duration.labels(
            model=self.config.model_name,
            endpoint=endpoint,
        ).observe(duration)
        monitoring_metrics.request_counter.labels(
            model=self.config.model_name,
            endpoint=endpoint,
            status=status,
        ).inc()

    async def _run_gpu_metrics_sampler(self):
        """Periodically sample visible GPU utilization for Prometheus/HPA."""
        while True:
            monitoring_metrics.collect_gpu_metrics(self.config.model_name)
            await asyncio.sleep(self.config.gpu_metrics_interval)
    
    async def _stream_completion(self, prompt: str, sampling_params: SamplingParams):
        """
        Stream completion chunks
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            
        Yields:
            SSE-formatted completion chunks
        """
        endpoint = "/v1/completions"
        start_time = time.time()
        status = "success"
        latest_text = ""

        try:
            async with self.admission.admitted():
                async for text in self.engine.stream_generate(
                    prompt,
                    sampling_params,
                    request_id=f"cmpl-stream-{uuid.uuid4()}",
                ):
                    latest_text = text
                    # Format as Server-Sent Event
                    chunk = {
                        "choices": [{
                            "text": text,
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {chunk}\n\n"

            monitoring_metrics.track_tokens(
                self.config.model_name,
                prompt_tokens=self._estimate_tokens(prompt),
                completion_tokens=self._estimate_tokens(latest_text),
            )

            # Final chunk
            yield "data: [DONE]\n\n"
        except Exception as e:
            status = "error"
            monitoring_metrics.error_counter.labels(
                model=self.config.model_name,
                error_type=type(e).__name__,
            ).inc()
            raise
        finally:
            self._record_request_metrics(
                endpoint=endpoint,
                status=status,
                duration=time.time() - start_time,
            )
    
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
