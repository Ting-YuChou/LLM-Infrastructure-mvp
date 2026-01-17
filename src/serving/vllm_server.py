"""
vLLM Inference Server
High-performance LLM serving with PagedAttention and continuous batching
"""

import os
import yaml
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

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
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_prefix_caching: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VLLMServerConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            model_name=config['model']['name'],
            host=config['server']['host'],
            port=config['server']['port'],
            tensor_parallel_size=config['gpu']['tensor_parallel_size'],
            gpu_memory_utilization=config['gpu']['gpu_memory_utilization'],
            max_num_seqs=config['gpu']['max_num_seqs'],
            max_num_batched_tokens=config['gpu']['max_num_batched_tokens'],
            enable_prefix_caching=config['paged_attention'].get('enable_prefix_caching', True),
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
        logger.info(f"Tensor parallel size: {config.tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {config.gpu_memory_utilization}")
    
    async def initialize(self):
        """Initialize the async engine"""
        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
            trust_remote_code=False,
        )
        
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
        
        async for request_output in self.engine.generate(prompt, sampling_params, request_id=None):
            yield request_output.outputs[0].text


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
            import time
            
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
                return StreamingResponse(
                    self._stream_completion(request.prompt, sampling_params),
                    media_type="text/event-stream"
                )
            
            # Non-streaming generation
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            
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
            
            # Build response
            response = CompletionResponse(
                id=f"cmpl-{int(time.time())}",
                created=int(time.time()),
                model=self.config.model_name,
                choices=outputs,
                usage={
                    "prompt_tokens": len(prompts[0].split()),  # Rough estimate
                    "completion_tokens": sum(len(c.text.split()) for c in outputs),
                    "total_tokens": len(prompts[0].split()) + sum(len(c.text.split()) for c in outputs)
                }
            )
            
            return response
        
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
            
            # Generate
            text = await self.engine.generate(prompt, sampling_params)
            
            # Build response
            import time
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
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split())
                }
            }
            
            return JSONResponse(content=response)
    
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
    
    async def _stream_completion(self, prompt: str, sampling_params: SamplingParams):
        """
        Stream completion chunks
        
        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            
        Yields:
            SSE-formatted completion chunks
        """
        async for text in self.engine.stream_generate(prompt, sampling_params):
            # Format as Server-Sent Event
            chunk = {
                "choices": [{
                    "text": text,
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {chunk}\n\n"
        
        # Final chunk
        yield "data: [DONE]\n\n"
    
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
