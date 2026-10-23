"""Configuration for the nano-vLLM serving backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


def _env(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class NanoVLLMConfig:
    """Runtime settings for the nano-vLLM backend."""

    model_name: str = "/workspace/models/nano/checkpoint-final"
    served_model_name: str = "nano-llm-local"
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False
    # "auto" follows the checkpoint's torch_dtype (bfloat16 for Llama 3).
    dtype: str = "auto"
    device: str = "auto"
    max_model_len: int = 2048
    max_new_tokens: int = 256
    kvcache_block_size: int = 16
    # 0: as many blocks as fit in gpu_memory_utilization of the GPU after the
    # weights and a profiled forward pass (CPU: 1024).
    num_kvcache_blocks: int = 0
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 512
    # "mixed": each step decodes every running sequence and spends the rest
    # of max_num_batched_tokens on prefill chunks; "prefill_first": prefill
    # steps and decode steps alternate, and a prefill stalls the decodes.
    scheduling: str = "mixed"
    enable_prefix_cache: bool = True
    # "triton": the paged decode/prefill attention and fused RoPE + KV write
    # kernels (benchmarks/phase1); off CUDA they fall back to the PyTorch
    # references, so it is a safe default.
    attention_backend: str = "triton"
    # "triton" / "cuda": a GEMV kernel for linear layers when a batch has at
    # most 16 / 2 rows; "cuda" builds kernels/csrc/gemv_bf16.cu on first use.
    linear_backend: str = "triton"
    # "w4a16": quantize the decoder-layer projections to 4-bit integers at
    # load (RTN, groups of 128 along K); kernels/w4a16.py.
    weight_quant: str = "none"
    use_triton: bool = True
    # Decode steps replay CUDA graphs (one per batch size up to
    # cuda_graph_max_bs) unless enforce_eager; CPU always runs eagerly.
    enforce_eager: bool = False
    cuda_graph_max_bs: int = 64
    # Launch step N+1 before collecting step N, so scheduling and input
    # preparation overlap the GPU; finish checks then lag one step.
    async_scheduling: bool = False
    # Run the engine core (scheduler + model) in its own process; the server
    # process keeps tokenization, detokenization, stop strings and HTTP.
    multiprocess: bool = False
    eager_load: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 16
    enable_metrics: bool = True

    def __post_init__(self) -> None:
        if self.kvcache_block_size <= 0:
            raise ValueError("kvcache_block_size must be positive")
        if self.num_kvcache_blocks < 0:
            raise ValueError("num_kvcache_blocks must be positive, or 0 to size it from memory")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.scheduling not in {"mixed", "prefill_first"}:
            raise ValueError("scheduling must be one of: mixed, prefill_first")
        if self.cuda_graph_max_bs <= 0:
            raise ValueError("cuda_graph_max_bs must be positive")
        if self.attention_backend not in {"torch", "triton", "flash_attn"}:
            raise ValueError("attention_backend must be one of: torch, triton, flash_attn")
        if self.linear_backend not in {"torch", "triton", "cuda"}:
            raise ValueError("linear_backend must be one of: torch, triton, cuda")
        if self.weight_quant not in {"none", "w4a16"}:
            raise ValueError("weight_quant must be one of: none, w4a16")

    @classmethod
    def from_yaml(cls, config_path: str | None) -> "NanoVLLMConfig":
        data: dict[str, Any] = {}
        if config_path:
            path = Path(config_path)
            if path.is_file():
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        model = data.get("model", {})
        server = data.get("server", {})
        engine = data.get("engine", {})
        monitoring = data.get("monitoring", {})

        return cls(
            model_name=str(_env("MODEL_NAME", model.get("name", cls.model_name))),
            served_model_name=str(
                _env(
                    "SERVED_MODEL_NAME",
                    model.get("served_model_name", cls.served_model_name),
                )
            ),
            tokenizer=_optional_str(_env("TOKENIZER", model.get("tokenizer"))),
            trust_remote_code=_bool_value(
                _env("TRUST_REMOTE_CODE", model.get("trust_remote_code")),
                default=False,
            ),
            dtype=str(_env("DTYPE", model.get("dtype", cls.dtype))),
            device=str(_env("DEVICE", engine.get("device", cls.device))),
            max_model_len=int(
                _env("MAX_MODEL_LEN", model.get("max_model_len", cls.max_model_len))
            ),
            max_new_tokens=int(
                _env("MAX_NEW_TOKENS", engine.get("max_new_tokens", cls.max_new_tokens))
            ),
            kvcache_block_size=int(
                _env(
                    "KVCACHE_BLOCK_SIZE",
                    engine.get("kvcache_block_size", cls.kvcache_block_size),
                )
            ),
            num_kvcache_blocks=int(
                _env(
                    "NUM_KVCACHE_BLOCKS",
                    engine.get("num_kvcache_blocks", cls.num_kvcache_blocks),
                )
            ),
            gpu_memory_utilization=float(
                _env(
                    "GPU_MEMORY_UTILIZATION",
                    engine.get("gpu_memory_utilization", cls.gpu_memory_utilization),
                )
            ),
            max_num_seqs=int(
                _env("MAX_NUM_SEQS", engine.get("max_num_seqs", cls.max_num_seqs))
            ),
            max_num_batched_tokens=int(
                _env(
                    "MAX_NUM_BATCHED_TOKENS",
                    engine.get(
                        "max_num_batched_tokens",
                        cls.max_num_batched_tokens,
                    ),
                )
            ),
            scheduling=str(_env("SCHEDULING", engine.get("scheduling", cls.scheduling))),
            enable_prefix_cache=_bool_value(
                _env("ENABLE_PREFIX_CACHE", engine.get("enable_prefix_cache")),
                default=cls.enable_prefix_cache,
            ),
            attention_backend=str(
                _env(
                    "ATTENTION_BACKEND",
                    engine.get("attention_backend", cls.attention_backend),
                )
            ),
            linear_backend=str(
                _env("LINEAR_BACKEND", engine.get("linear_backend", cls.linear_backend))
            ),
            weight_quant=str(
                _env("WEIGHT_QUANT", engine.get("weight_quant", cls.weight_quant))
            ),
            use_triton=_bool_value(
                _env("USE_TRITON", engine.get("use_triton")),
                default=cls.use_triton,
            ),
            enforce_eager=_bool_value(
                _env("ENFORCE_EAGER", engine.get("enforce_eager")),
                default=cls.enforce_eager,
            ),
            cuda_graph_max_bs=int(
                _env(
                    "CUDA_GRAPH_MAX_BS",
                    engine.get("cuda_graph_max_bs", cls.cuda_graph_max_bs),
                )
            ),
            async_scheduling=_bool_value(
                _env("ASYNC_SCHEDULING", engine.get("async_scheduling")),
                default=cls.async_scheduling,
            ),
            multiprocess=_bool_value(
                _env("ENGINE_MULTIPROCESS", engine.get("multiprocess")),
                default=cls.multiprocess,
            ),
            eager_load=_bool_value(
                _env("EAGER_LOAD", engine.get("eager_load")),
                default=cls.eager_load,
            ),
            host=str(_env("HOST", server.get("host", cls.host))),
            port=int(_env("PORT", server.get("port", cls.port))),
            max_concurrent_requests=int(
                _env(
                    "MAX_CONCURRENT_REQUESTS",
                    server.get("max_concurrent_requests", cls.max_concurrent_requests),
                )
            ),
            enable_metrics=_bool_value(
                _env("ENABLE_METRICS", monitoring.get("enable_metrics")),
                default=cls.enable_metrics,
            ),
        )
