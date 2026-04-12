"""
Unit tests for vLLM serving configuration wiring.
"""

import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.vllm_server import VLLMServerConfig


class FakeAsyncEngineArgs:
    """Signature with broad support for modern performance options."""

    def __init__(
        self,
        model,
        tokenizer=None,
        tokenizer_mode="auto",
        tokenizer_revision=None,
        trust_remote_code=False,
        download_dir=None,
        dtype="auto",
        quantization=None,
        max_model_len=None,
        load_format="auto",
        cpu_offload_gb=0.0,
        enable_lora=False,
        max_loras=1,
        max_lora_rank=16,
        lora_dtype="auto",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.9,
        swap_space=4.0,
        enforce_eager=False,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        block_size=16,
        enable_prefix_caching=True,
        scheduling_policy="fcfs",
        enable_chunked_prefill=False,
        disable_log_stats=False,
        disable_custom_all_reduce=False,
        seed=0,
        max_logprobs=5,
        max_parallel_loading_workers=None,
        distributed_init_method="auto",
    ):
        pass


class NarrowAsyncEngineArgs:
    """Signature representing an older vLLM build with fewer supported args."""

    def __init__(
        self,
        model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        trust_remote_code=False,
    ):
        pass


def test_vllm_server_config_loads_engine_related_settings(tmp_path):
    """Config should hydrate the performance settings already declared in YAML."""
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "/models/final",
                    "tokenizer": "/models/tokenizer",
                    "trust_remote_code": True,
                    "download_dir": "/models/cache",
                    "dtype": "float16",
                    "quantization": "awq",
                    "max_model_len": 4096,
                },
                "server": {"host": "127.0.0.1", "port": 9000},
                "gpu": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 3,
                    "gpu_memory_utilization": 0.85,
                    "swap_space": 8,
                    "enforce_eager": True,
                    "max_num_seqs": 128,
                    "max_num_batched_tokens": 4096,
                },
                "paged_attention": {
                    "block_size": 32,
                    "enable_prefix_caching": False,
                },
                "batching": {
                    "scheduling_policy": "priority",
                    "enable_chunked_prefill": True,
                },
                "optimization": {
                    "disable_log_stats": True,
                    "disable_custom_all_reduce": True,
                },
                "distributed": {"distributed_init_method": "tcp://127.0.0.1:1234"},
                "engine": {
                    "seed": 42,
                    "max_logprobs": 10,
                    "tokenizer_mode": "slow",
                    "tokenizer_revision": "main",
                    "max_parallel_loading_workers": 4,
                },
                "loading": {
                    "load_format": "safetensors",
                    "cpu_offload_gb": 6,
                    "enable_lora": True,
                    "max_loras": 2,
                    "max_lora_rank": 32,
                    "lora_dtype": "float16",
                },
            }
        )
    )

    config = VLLMServerConfig.from_yaml(str(config_path))

    assert config.model_name == "/models/final"
    assert config.tokenizer == "/models/tokenizer"
    assert config.trust_remote_code is True
    assert config.download_dir == "/models/cache"
    assert config.dtype == "float16"
    assert config.quantization == "awq"
    assert config.max_model_len == 4096
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.tensor_parallel_size == 2
    assert config.pipeline_parallel_size == 3
    assert config.gpu_memory_utilization == 0.85
    assert config.swap_space == 8
    assert config.enforce_eager is True
    assert config.max_num_seqs == 128
    assert config.max_num_batched_tokens == 4096
    assert config.block_size == 32
    assert config.enable_prefix_caching is False
    assert config.scheduling_policy == "priority"
    assert config.enable_chunked_prefill is True
    assert config.disable_log_stats is True
    assert config.disable_custom_all_reduce is True
    assert config.seed == 42
    assert config.max_logprobs == 10
    assert config.tokenizer_mode == "slow"
    assert config.tokenizer_revision == "main"
    assert config.max_parallel_loading_workers == 4
    assert config.load_format == "safetensors"
    assert config.cpu_offload_gb == 6
    assert config.enable_lora is True
    assert config.max_loras == 2
    assert config.max_lora_rank == 32
    assert config.lora_dtype == "float16"
    assert config.distributed_init_method == "tcp://127.0.0.1:1234"


def test_build_async_engine_kwargs_passes_supported_performance_options():
    """Supported performance knobs should be forwarded to AsyncEngineArgs."""
    config = VLLMServerConfig(
        model_name="/models/final",
        tokenizer="/models/tokenizer",
        tokenizer_mode="slow",
        tokenizer_revision="main",
        trust_remote_code=True,
        download_dir="/models/cache",
        dtype="float16",
        quantization="awq",
        max_model_len=4096,
        load_format="safetensors",
        cpu_offload_gb=4,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=32,
        lora_dtype="float16",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.88,
        swap_space=10,
        enforce_eager=True,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        block_size=32,
        enable_prefix_caching=False,
        scheduling_policy="priority",
        enable_chunked_prefill=True,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        seed=7,
        max_logprobs=9,
        max_parallel_loading_workers=3,
        distributed_init_method="tcp://127.0.0.1:9999",
    )

    engine_kwargs = config.build_async_engine_kwargs(FakeAsyncEngineArgs)

    assert engine_kwargs == {
        "model": "/models/final",
        "tokenizer": "/models/tokenizer",
        "tokenizer_mode": "slow",
        "tokenizer_revision": "main",
        "trust_remote_code": True,
        "download_dir": "/models/cache",
        "dtype": "float16",
        "quantization": "awq",
        "max_model_len": 4096,
        "load_format": "safetensors",
        "cpu_offload_gb": 4,
        "enable_lora": True,
        "max_loras": 2,
        "max_lora_rank": 32,
        "lora_dtype": "float16",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "gpu_memory_utilization": 0.88,
        "swap_space": 10,
        "enforce_eager": True,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
        "block_size": 32,
        "enable_prefix_caching": False,
        "scheduling_policy": "priority",
        "enable_chunked_prefill": True,
        "disable_log_stats": True,
        "disable_custom_all_reduce": True,
        "seed": 7,
        "max_logprobs": 9,
        "max_parallel_loading_workers": 3,
        "distributed_init_method": "tcp://127.0.0.1:9999",
    }


def test_build_async_engine_kwargs_skips_unsupported_options():
    """Older vLLM versions should receive only the kwargs they support."""
    config = VLLMServerConfig(
        model_name="/models/final",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.88,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        scheduling_policy="priority",
        enable_chunked_prefill=True,
        trust_remote_code=True,
    )

    engine_kwargs = config.build_async_engine_kwargs(NarrowAsyncEngineArgs)

    assert engine_kwargs == {
        "model": "/models/final",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.88,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": False,
        "trust_remote_code": True,
    }
