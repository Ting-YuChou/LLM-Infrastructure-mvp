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


def _int_tuple(value: Any) -> tuple[int, ...]:
    """A YAML list, or "1,2,4" from the environment."""
    if isinstance(value, str):
        value = [part for part in value.replace(",", " ").split() if part]
    return tuple(int(item) for item in value)


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
    # "python": block_manager.BlockManager; "cpp": the same radix tree and
    # allocator in C++ (csrc/radix_tree.cpp, built with nanobind on first use).
    prefix_cache_backend: str = "python"
    # "triton": the paged decode/prefill attention and fused RoPE + KV write
    # kernels (benchmarks/phase1); off CUDA they fall back to the PyTorch
    # references, so it is a safe default.
    attention_backend: str = "triton"
    # "triton" / "cuda": a GEMV kernel for linear layers when a batch has at
    # most 16 / 2 rows; "cuda" builds kernels/csrc/gemv_bf16.cu on first use.
    linear_backend: str = "triton"
    # "awq": the model path is an AWQ checkpoint (AutoAWQ GEMM format, 4-bit,
    # groups of 128, zero points), as vLLM serves with awq_marlin; such a
    # checkpoint loads as "awq" under "none" too. "w4a16": quantize a
    # full-precision checkpoint's decoder-layer projections to 4 bits at load
    # (RTN, symmetric, groups of 128). Both run the Marlin-style kernel of
    # kernels/w4a16.py.
    weight_quant: str = "none"
    # "int8": lm_head as int8 with a scale per vocab row (kernels/w8a16.py),
    # half the bytes of the largest bf16 read left under 4-bit weights;
    # "auto": model.AUTO_LM_HEAD_QUANT by weight_quant.
    lm_head_quant: str = "auto"
    # "fp8": the paged KV cache holds e4m3 codes plus an fp32 scale per
    # (token, kv head), about half the bytes of bf16 (kernels/fp8.py);
    # "auto": the model dtype.
    kv_cache_dtype: str = "auto"
    use_triton: bool = True
    # Decode steps replay CUDA graphs (one per batch size up to
    # cuda_graph_max_bs) unless enforce_eager; CPU always runs eagerly.
    enforce_eager: bool = False
    cuda_graph_max_bs: int = 64
    # Other steps of up to this many tokens (prefill and mixed steps, decode
    # batches past cuda_graph_max_bs) replay piecewise graphs: everything but
    # the attention, which runs eagerly in between. 0 turns them off.
    cuda_graph_max_tokens: int = 512
    # Launch step N+1 before collecting step N, so scheduling and input
    # preparation overlap the GPU; finish checks then lag one step.
    async_scheduling: bool = True
    # Seeds the sampler's random draws (each comes from this seed, the
    # sequence id and the token's position), like vLLM's --seed.
    seed: int = 0
    # Speculative decoding (spec.py): the path of a drafter checkpoint; None
    # is off. The output is the target's own, whatever the drafter proposes.
    speculative_model: Optional[str] = None
    # "dflash": a DFlash block-diffusion drafter (dflash.py).
    speculative_method: str = "dflash"
    # Drafts per cycle; 0: the drafter's block size - 1.
    num_speculative_tokens: int = 0
    # "w4a16": quantize the drafter's projections to 4 bits at load.
    speculative_weight_quant: str = "none"
    # A sampled request's drafts are drawn like the target's tokens, with the
    # noise of the position each one proposes (seed, sequence id, position),
    # at this fraction of the request's temperature. 1: the request's own
    # temperature (coupled sampling: a draft matches whenever the two models'
    # logits agree closely enough); 0: the drafter's argmax, whatever the
    # request samples with; in between, the noise decides only where the
    # drafter's leading tokens are close. The output is the same for any value.
    # Measured with the public Llama 3.1 8B DFlash drafter at temperature 0.7
    # (benchmarks/phase3, M5): 1 loses 2-7% of the tokens per cycle to the
    # argmax, 0.25 and 0.5 are level with it (-0.3% to +1.0%).
    speculative_draft_temperature: float = 0.25
    # "adaptive": every step picks the verify length Q (a decode row's anchor
    # and Q - 1 drafts) from speculative_verify_lens that yields the most
    # tokens per millisecond, from each CUDA graph's step time measured at
    # startup and the drafts' acceptance rates so far (spec.SpecPolicy);
    # "fixed": every cycle runs the whole block. The output is the same.
    speculative_policy: str = "adaptive"
    # The verify lengths "adaptive" chooses from; the whole block
    # (num_speculative_tokens + 1) always is one, longer ones are dropped.
    # 1 is a plain decode step: with many sequences a pass over the weights
    # is no longer shared by enough rows for drafts to pay.
    speculative_verify_lens: tuple[int, ...] = (1, 2, 3, 4, 6, 10)
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
        if self.prefix_cache_backend not in {"python", "cpp"}:
            raise ValueError("prefix_cache_backend must be one of: python, cpp")
        if self.scheduling not in {"mixed", "prefill_first"}:
            raise ValueError("scheduling must be one of: mixed, prefill_first")
        if self.cuda_graph_max_bs <= 0:
            raise ValueError("cuda_graph_max_bs must be positive")
        if self.cuda_graph_max_tokens < 0:
            raise ValueError("cuda_graph_max_tokens must be 0 (off) or positive")
        if self.attention_backend not in {"torch", "triton", "flash_attn"}:
            raise ValueError("attention_backend must be one of: torch, triton, flash_attn")
        if self.linear_backend not in {"torch", "triton", "cuda"}:
            raise ValueError("linear_backend must be one of: torch, triton, cuda")
        if self.weight_quant not in {"none", "w4a16", "awq"}:
            raise ValueError("weight_quant must be one of: none, w4a16, awq")
        if self.lm_head_quant not in {"auto", "none", "int8", "int8_dense"}:
            raise ValueError("lm_head_quant must be one of: auto, none, int8, int8_dense")
        if self.kv_cache_dtype not in {"auto", "fp8"}:
            raise ValueError("kv_cache_dtype must be one of: auto, fp8")
        if self.kv_cache_dtype == "fp8" and self.attention_backend == "flash_attn":
            raise ValueError("kv_cache_dtype fp8 needs the triton or torch attention backend")
        if self.speculative_method not in {"dflash"}:
            raise ValueError("speculative_method must be: dflash")
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be 0 (the drafter's block) or positive")
        if self.speculative_weight_quant not in {"none", "w4a16"}:
            raise ValueError("speculative_weight_quant must be one of: none, w4a16")
        if not 0.0 <= self.speculative_draft_temperature <= 1.0:
            raise ValueError("speculative_draft_temperature must be between 0 (argmax drafts) and 1")
        if self.speculative_policy not in {"adaptive", "fixed"}:
            raise ValueError("speculative_policy must be one of: adaptive, fixed")
        self.speculative_verify_lens = tuple(int(length) for length in self.speculative_verify_lens)
        if any(length < 1 for length in self.speculative_verify_lens):
            raise ValueError("speculative_verify_lens must be positive")

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
            prefix_cache_backend=str(
                _env(
                    "PREFIX_CACHE_BACKEND",
                    engine.get("prefix_cache_backend", cls.prefix_cache_backend),
                )
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
            lm_head_quant=str(
                _env("LM_HEAD_QUANT", engine.get("lm_head_quant", cls.lm_head_quant))
            ),
            kv_cache_dtype=str(
                _env("KV_CACHE_DTYPE", engine.get("kv_cache_dtype", cls.kv_cache_dtype))
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
            cuda_graph_max_tokens=int(
                _env(
                    "CUDA_GRAPH_MAX_TOKENS",
                    engine.get("cuda_graph_max_tokens", cls.cuda_graph_max_tokens),
                )
            ),
            async_scheduling=_bool_value(
                _env("ASYNC_SCHEDULING", engine.get("async_scheduling")),
                default=cls.async_scheduling,
            ),
            seed=int(_env("SEED", engine.get("seed", cls.seed))),
            speculative_model=_optional_str(
                _env("SPECULATIVE_MODEL", engine.get("speculative_model"))
            ),
            speculative_method=str(
                _env(
                    "SPECULATIVE_METHOD",
                    engine.get("speculative_method", cls.speculative_method),
                )
            ),
            num_speculative_tokens=int(
                _env(
                    "NUM_SPECULATIVE_TOKENS",
                    engine.get("num_speculative_tokens", cls.num_speculative_tokens),
                )
            ),
            speculative_weight_quant=str(
                _env(
                    "SPECULATIVE_WEIGHT_QUANT",
                    engine.get("speculative_weight_quant", cls.speculative_weight_quant),
                )
            ),
            speculative_draft_temperature=float(
                _env(
                    "SPECULATIVE_DRAFT_TEMPERATURE",
                    engine.get("speculative_draft_temperature", cls.speculative_draft_temperature),
                )
            ),
            speculative_policy=str(
                _env(
                    "SPECULATIVE_POLICY",
                    engine.get("speculative_policy", cls.speculative_policy),
                )
            ),
            speculative_verify_lens=_int_tuple(
                _env(
                    "SPECULATIVE_VERIFY_LENS",
                    engine.get("speculative_verify_lens", cls.speculative_verify_lens),
                )
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
