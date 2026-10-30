"""Minimal Llama-family decoder used by the nano-vLLM backend."""

from __future__ import annotations

import functools
import json
import logging
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F

from .kernels import (
    apply_rope,
    causal_attention,
    fused_add_rms_norm,
    linear,
    W4A16Weight,
    W8A16Weight,
    load_cuda_extension,
    paged_decode_attention,
    paged_prefill_attention,
    rms_norm,
    rope_and_cache_write,
    quantize_w4a16,
    quantize_w8a16,
    rope_cos_sin,
    silu_and_mul,
)
from .kernels.fp8 import KV_CACHE_DTYPES
from .kernels.rotary import compute_inv_freq, validate_rope_scaling
from .kernels.w4a16 import W4A16_CONFIG, awq_unpack, pack_w4a16, w4a16_tiled_supported, w4a16_workspace
from .kv_cache import PagedKVCache, StaticKVCache
from .metadata import BatchMetadata, build_batch

logger = logging.getLogger(__name__)


def torch_dtype_from_string(
    value: str, checkpoint_dtype: Optional[str | torch.dtype] = None
) -> torch.dtype:
    """Resolve a dtype name; "auto" follows the checkpoint's dtype."""
    normalized = value.lower()
    if normalized == "auto":
        if checkpoint_dtype is None:
            raise ValueError(
                "dtype='auto' needs the checkpoint dtype; set DTYPE explicitly "
                "(e.g. bfloat16 for Llama 3)"
            )
        normalized = str(checkpoint_dtype).lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for nano-vLLM: {value}")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@dataclass(frozen=True)
class NanoLlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    torch_dtype: Optional[str] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int | list[int]] = None
    pad_token_id: Optional[int] = None
    # config.json's quantization_config (pre-quantized checkpoints).
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        validate_rope_scaling(self.rope_scaling)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "NanoLlamaConfig":
        path = Path(model_path) / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        model_type = str(data.get("model_type", "llama")).lower()
        if model_type not in {"llama", "mistral"}:
            raise ValueError(
                "nano-vLLM v0.1 supports Llama-family decoder configs only, "
                f"got model_type={model_type!r}"
            )
        heads = int(data["num_attention_heads"])
        # transformers >= 5 writes rope_theta and the scaling keys together
        # under "rope_parameters" instead of "rope_theta" / "rope_scaling".
        rope_parameters = dict(data.get("rope_parameters") or {})
        rope_theta = data.get("rope_theta", rope_parameters.pop("rope_theta", 10000.0))
        rope_scaling = data.get("rope_scaling")
        if not rope_scaling and rope_parameters.get("rope_type", "default") != "default":
            rope_scaling = rope_parameters
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=int(data.get("num_key_value_heads", heads)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_theta),
            rope_scaling=dict(rope_scaling) if rope_scaling else None,
            # LlamaConfig defaults to untied; only tie when the checkpoint says so.
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            # transformers >= 4.56 writes "dtype"; older releases "torch_dtype".
            torch_dtype=data.get("torch_dtype") or data.get("dtype"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            pad_token_id=data.get("pad_token_id"),
            quantization_config=dict(data["quantization_config"]) if data.get("quantization_config") else None,
        )


def _load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        state: Dict[str, torch.Tensor] = {}
        for file_path in safetensor_files:
            state.update(load_file(str(file_path), device="cpu"))
        return state

    bin_path = model_path / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"No .safetensors or pytorch_model.bin weights found in {model_path}"
    )


def _first_floating_dtype(weights: Dict[str, torch.Tensor]) -> Optional[torch.dtype]:
    for tensor in weights.values():
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def _concat_projections(
    weights: Dict[str, torch.Tensor], prefixes: list[str], fused: str
) -> None:
    """Replace several linear layers that read the same input with one."""
    if f"{prefixes[0]}.qweight" in weights:
        # AWQ tensors hold N in dim 1, packed 8 columns per int32 for
        # qweight/qzeros; every part's N is a multiple of 8, so the packed
        # words concatenate as they are.
        for suffix in AWQ_SUFFIXES:
            weights[f"{fused}.{suffix}"] = torch.cat(
                [weights.pop(f"{prefix}.{suffix}") for prefix in prefixes], dim=1
            )
    else:
        weights[f"{fused}.weight"] = torch.cat(
            [weights.pop(f"{prefix}.weight") for prefix in prefixes]
        )
    biases = [weights.pop(f"{prefix}.bias", None) for prefix in prefixes]
    if any(bias is not None for bias in biases):
        if any(bias is None for bias in biases):
            raise ValueError(f"{fused}: either all or none of {prefixes} need a bias")
        weights[f"{fused}.bias"] = torch.cat(biases)


def fuse_projections(
    weights: Dict[str, torch.Tensor], num_layers: int
) -> Dict[str, torch.Tensor]:
    """q/k/v -> ``self_attn.qkv_proj`` and gate/up -> ``mlp.gate_up_proj``.

    One GEMM instead of three (and two) per layer. Done on the checkpoint's
    device (CPU) before weights move, so the GPU never holds both copies.
    """
    weights = dict(weights)
    for layer_idx in range(num_layers):
        attn = f"model.layers.{layer_idx}.self_attn"
        mlp = f"model.layers.{layer_idx}.mlp"
        _concat_projections(
            weights, [f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj"],
            f"{attn}.qkv_proj",
        )
        _concat_projections(
            weights, [f"{mlp}.gate_proj", f"{mlp}.up_proj"], f"{mlp}.gate_up_proj"
        )
    return weights


# "w4a16": RTN-quantize a full-precision checkpoint at load; "awq": load an
# AWQ checkpoint (an AWQ checkpoint always loads as "awq", resolve_weight_quant).
WEIGHT_QUANT_MODES = ("none", "w4a16", "awq")
# The decoder-layer projections; embeddings, norms and lm_head stay in the
# model dtype under either.
QUANTIZED_LINEARS = (
    "self_attn.qkv_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
)
AWQ_SUFFIXES = ("qweight", "qzeros", "scales")
# What nano reads: AutoAWQ's GEMM format, 4 bits, groups of 128, zero points.
AWQ_FORMAT = {"bits": 4, "group_size": 128, "zero_point": True, "version": "gemm"}

# lm_head_quant: "int8" stores lm_head as int8 with a scale per vocab row
# (kernels/w8a16.py); "int8_dense" keeps the dtype weight too, for batches
# past W8A16_DENSE_ROWS rows, where the int8 kernel is slower than cuBLAS;
# "auto" picks per weight_quant from AUTO_LM_HEAD_QUANT: int8_dense under
# 4-bit weights, where lm_head is the largest read of a decode step (runs
# awq-a2, awq-a3: wikitext-2 PPL +0.001% and top-1 99.01% against the bf16
# head; decode 0.36 / 0.36 / 0.30 ms faster at b=1/16/64, the same past 64).
LM_HEAD_QUANT_MODES = ("auto", "none", "int8", "int8_dense")
AUTO_LM_HEAD_QUANT: Dict[str, str] = {"awq": "int8_dense", "w4a16": "int8_dense"}

Linear = Union[torch.Tensor, W4A16Weight, W8A16Weight]


def resolve_lm_head_quant(requested: str, weight_quant: str) -> str:
    """"auto" -> AUTO_LM_HEAD_QUANT[weight_quant] (default "none")."""
    if requested not in LM_HEAD_QUANT_MODES:
        raise ValueError(f"lm_head_quant must be one of: {', '.join(LM_HEAD_QUANT_MODES)}")
    return AUTO_LM_HEAD_QUANT.get(weight_quant, "none") if requested == "auto" else requested


@dataclass(frozen=True)
class AWQTensors:
    """One projection of an AWQ checkpoint, as stored (awq_unpack)."""

    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor


def resolve_weight_quant(requested: str, config: NanoLlamaConfig) -> str:
    """The weight_quant a checkpoint loads with.

    An AWQ checkpoint loads as "awq" whether "awq" or "none" was asked for
    (its weights exist only in 4 bits); "w4a16" on it, or "awq" on a
    full-precision checkpoint, is an error.
    """
    if requested not in WEIGHT_QUANT_MODES:
        raise ValueError(f"weight_quant must be one of: {', '.join(WEIGHT_QUANT_MODES)}")
    quant = config.quantization_config
    if not quant:
        if requested == "awq":
            raise ValueError(
                "weight_quant='awq' needs an AWQ checkpoint (config.json quantization_config "
                "with quant_method 'awq')"
            )
        return requested
    found = {
        "bits": quant.get("bits"),
        "group_size": quant.get("group_size"),
        "zero_point": quant.get("zero_point"),
        "version": str(quant.get("version", "")).lower(),
    }
    if str(quant.get("quant_method", "")).lower() != "awq" or found != AWQ_FORMAT:
        raise ValueError(
            "nano-vLLM loads pre-quantized checkpoints in AutoAWQ's GEMM format only "
            f"({AWQ_FORMAT}); config.json has quantization_config {quant}"
        )
    if requested == "w4a16":
        raise ValueError(
            "weight_quant='w4a16' quantizes a full-precision checkpoint at load; this one "
            "is already AWQ (use 'awq')"
        )
    if requested == "none":
        logger.info("AWQ checkpoint: loading with weight_quant='awq'")
    return "awq"


def _group_awq(weights: Dict[str, Any]) -> Dict[str, Any]:
    """Each AWQ projection's three tensors -> one AWQTensors under ``<name>.weight``."""
    grouped: Dict[str, Any] = {}
    for name, tensor in weights.items():
        base, _, suffix = name.rpartition(".")
        if suffix in AWQ_SUFFIXES and f"{base}.qweight" in weights:
            if suffix == "qweight":
                grouped[f"{base}.weight"] = AWQTensors(
                    tensor, weights[f"{base}.qzeros"], weights[f"{base}.scales"]
                )
        else:
            grouped[name] = tensor
    return grouped


def _clone_weight(weight: Linear) -> Linear:
    """A copy of a loaded weight (a tensor, W4A16Weight or W8A16Weight),
    allocated on the current stream."""
    if isinstance(weight, torch.Tensor):
        return weight.clone()
    return replace(weight, **{
        field.name: value.clone()
        for field in fields(weight)
        if isinstance(value := getattr(weight, field.name), torch.Tensor)
    })


def _build_on_stream(stream: torch.cuda.Stream, build: Callable[[], Linear]) -> Linear:
    """build() with ``stream`` current, then a copy of its result made on the
    current stream.

    Converting a weight at load (the int8 lm_head, RTN quantization, AWQ
    unpacking) makes temporaries of up to 2 GiB (lm_head in fp32). The
    caching allocator carves later allocations, weights included, out of
    their freed blocks, and empty_cache releases only segments with no live
    block, so loading on one stream left ~2 GiB reserved but unallocated,
    which the KV cache never got (run awqserve). A block is reused only by
    the stream that allocated it: the temporaries and the built weight stay
    in segments of ``stream``, empty once the copy is made, and the copies
    fill segments of their own.
    """
    current = torch.cuda.current_stream(stream.device)
    with torch.cuda.stream(stream):
        weight = build()
    stream.synchronize()
    copy = _clone_weight(weight)
    # Done reading before the temporaries' blocks are freed for reuse on stream.
    current.synchronize()
    return copy


@dataclass(frozen=True)
class _LayerWeights:
    """Per-layer tensors, looked up once instead of by name on every call."""

    input_norm: torch.Tensor
    qkv: Linear
    qkv_bias: Optional[torch.Tensor]
    o_proj: Linear
    o_bias: Optional[torch.Tensor]
    post_norm: torch.Tensor
    gate_up: Linear
    gate_up_bias: Optional[torch.Tensor]
    down: Linear
    down_bias: Optional[torch.Tensor]


class NanoLlamaForCausalLM:
    """Small, explicit Llama forward path for learning serving internals."""

    def __init__(
        self,
        config: NanoLlamaConfig,
        weights: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        use_triton: bool = True,
        attention_backend: str = "torch",
        max_model_len: Optional[int] = None,
        linear_backend: str = "torch",
        weight_quant: str = "none",
        kv_cache_dtype: str = "auto",
        lm_head_quant: str = "none",
    ) -> None:
        weight_quant = resolve_weight_quant(weight_quant, config)
        self.lm_head_quant = resolve_lm_head_quant(lm_head_quant, weight_quant)
        if kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of: {', '.join(KV_CACHE_DTYPES)}")
        self.config = config
        self.device = device
        self.dtype = dtype if device.type == "cuda" else torch.float32
        self.use_triton = use_triton
        self.attention_backend = attention_backend
        self.linear_backend = linear_backend
        self.weight_quant = weight_quant
        # The paged caches this model makes (new_paged_cache): "fp8" stores
        # e4m3 codes with a scale per vector (kernels/fp8.py).
        self.kv_cache_dtype = kv_cache_dtype
        if device.type == "cuda":
            # Build the extensions now, never inside a CUDA-graph capture.
            if linear_backend == "cuda":
                load_cuda_extension()
            if weight_quant in ("w4a16", "awq"):
                load_cuda_extension("w4a16")
        self._validate_required_weights(weights)
        weights = _group_awq(fuse_projections(weights, config.num_hidden_layers))
        self.weights = self._load_weights(weights)
        quantized = [w for w in self.weights.values() if isinstance(w, W4A16Weight)]
        if quantized and device.type == "cuda":
            # Split-K buffers, before any CUDA graph is captured.
            w4a16_workspace(device, max(w.shape[0] for w in quantized))
        self.lm_head_weight = self._resolve_lm_head()
        self.layers = [
            self._layer_weights(layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self._qkv_sizes = [q_size, kv_size, kv_size]
        self.inv_freq = compute_inv_freq(
            config.head_dim,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            device=device,
        )
        # Indexed by position in forward_batch instead of recomputing cos/sin
        # in every layer.
        self.rope_cos, self.rope_sin = rope_cos_sin(
            self.inv_freq,
            max_model_len or config.max_position_embeddings,
            self.dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        dtype: str = "auto",
        device: str = "auto",
        use_triton: bool = True,
        attention_backend: str = "torch",
        max_model_len: Optional[int] = None,
        linear_backend: str = "torch",
        weight_quant: str = "none",
        kv_cache_dtype: str = "auto",
        lm_head_quant: str = "none",
    ) -> "NanoLlamaForCausalLM":
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"nano-vLLM model path does not exist: {path}")
        config = NanoLlamaConfig.from_pretrained(path)
        resolved_device = resolve_device(device)
        weights = _load_state_dict(path)
        # Like transformers' torch_dtype="auto": config first, then the weights.
        checkpoint_dtype = config.torch_dtype or _first_floating_dtype(weights)
        return cls(
            config=config,
            weights=weights,
            device=resolved_device,
            dtype=torch_dtype_from_string(dtype, checkpoint_dtype),
            use_triton=use_triton,
            attention_backend=attention_backend,
            max_model_len=max_model_len,
            linear_backend=linear_backend,
            weight_quant=weight_quant,
            kv_cache_dtype=kv_cache_dtype,
            lm_head_quant=lm_head_quant,
        )

    def new_cache(self, batch_size: int, max_seq_len: int) -> StaticKVCache:
        return StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            num_key_value_heads=self.config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def new_paged_cache(
        self,
        num_blocks: int,
        block_size: int,
    ) -> PagedKVCache:
        return PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: StaticKVCache,
        start_pos: int = 0,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        batch_size, seq_len = input_ids.shape
        if batch_size != cache.batch_size:
            raise ValueError("input batch size must match cache batch size")

        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            dtype=torch.long,
            device=self.device,
        )
        x = self.weights["model.embed_tokens.weight"][input_ids.to(self.device)]

        for layer_idx, layer in enumerate(self.layers):
            residual = x
            x_norm = rms_norm(
                x,
                layer.input_norm,
                eps=self.config.rms_norm_eps,
                use_triton=self.use_triton,
            )
            qkv = linear(x_norm, layer.qkv, layer.qkv_bias)
            query, key, value = qkv.split(self._qkv_sizes, dim=-1)
            query = self._shape_heads(query, self.config.num_attention_heads)
            key = self._shape_heads(key, self.config.num_key_value_heads)
            value = self._shape_heads(value, self.config.num_key_value_heads)
            query, key = apply_rope(
                query,
                key,
                positions=positions,
                theta=self.config.rope_theta,
                use_triton=self.use_triton,
                inv_freq=self.inv_freq,
            )

            cache.update(layer_idx, positions, key, value)
            full_key, full_value = cache.read(
                layer_idx, end_position=start_pos + seq_len
            )
            full_key = self._repeat_kv(full_key)
            full_value = self._repeat_kv(full_value)
            attn = causal_attention(
                query,
                full_key,
                full_value,
                query_positions=positions,
                use_triton=self.use_triton,
            )
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            x = residual + linear(attn, layer.o_proj, layer.o_bias)
            x = x + self._mlp(x, layer)

        x = rms_norm(
            x,
            self._weight("model.norm.weight"),
            eps=self.config.rms_norm_eps,
            use_triton=self.use_triton,
        )
        return linear(x, self.lm_head_weight)

    @torch.inference_mode()
    def forward_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache,
        metadata: BatchMetadata,
    ) -> torch.Tensor:
        """Forward over a packed scheduler batch on the paged KV cache.

        Returns logits for the rows in ``metadata.logits_indices`` (all rows
        when it is None). Nothing here reads a device value back to the host.

        The pass is split at each attention: :meth:`forward_head` runs up to
        layer 0's attention, :meth:`forward_between` from layer i's attention
        output to layer i+1's query, :meth:`forward_tail` from the last
        attention output to the logits. Everything but the attention depends
        only on the token count, so the model runner captures these pieces in
        CUDA graphs per token count and runs the attention eagerly between
        them (piecewise graphs, for steps with prefill rows).
        """
        query, residual, rope = self.forward_head(input_ids, positions, cache, metadata.slot_mapping)
        last = len(self.layers) - 1
        for layer_idx in range(last):
            attn = self.attention(layer_idx, query, cache, metadata)
            query, residual = self.forward_between(
                layer_idx, attn, residual, rope, cache, metadata.slot_mapping
            )
        attn = self.attention(last, query, cache, metadata)
        return self.forward_tail(attn, residual, metadata.logits_indices)

    def forward_head(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Embedding through layer 0's query: (query, residual, (cos, sin))."""
        if self.attention_backend not in {"torch", "triton"}:
            raise NotImplementedError(
                "nano-vLLM's paged path supports the torch and triton attention "
                "backends; flash_attn is not implemented."
            )
        rope = (self.rope_cos[positions], self.rope_sin[positions])  # [tokens, head_dim]
        # x is each sublayer's output; fused_add_rms_norm folds it into the
        # residual stream and normalizes for the next sublayer in one pass.
        residual = self.weights["model.embed_tokens.weight"][input_ids]
        x_norm = rms_norm(
            residual, self.layers[0].input_norm, eps=self.config.rms_norm_eps, use_triton=self.use_triton
        )
        return self._query(0, x_norm, rope, cache, slot_mapping), residual, rope

    def forward_between(
        self,
        layer_idx: int,
        attn: torch.Tensor,
        residual: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer ``layer_idx``'s attention output through layer + 1's query."""
        x, residual = self._finish_layer(layer_idx, attn, residual)
        x_norm, residual = fused_add_rms_norm(
            x,
            residual,
            self.layers[layer_idx + 1].input_norm,
            eps=self.config.rms_norm_eps,
            use_triton=self.use_triton,
        )
        return self._query(layer_idx + 1, x_norm, rope, cache, slot_mapping), residual

    def forward_tail(
        self, attn: torch.Tensor, residual: torch.Tensor, logits_indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """The last attention output to logits, for ``logits_indices`` (all rows if None)."""
        x, residual = self._finish_layer(len(self.layers) - 1, attn, residual)
        if logits_indices is not None:
            x = x[logits_indices]
            residual = residual[logits_indices]
        x_norm, _ = fused_add_rms_norm(
            x,
            residual,
            self._weight("model.norm.weight"),
            eps=self.config.rms_norm_eps,
            use_triton=self.use_triton,
        )
        return self._linear(x_norm, self.lm_head_weight)

    def attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        cache: PagedKVCache,
        metadata: BatchMetadata,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Paged attention of layer ``layer_idx`` (into ``out`` when given)."""
        return self._paged_attention(
            query, cache.keys[layer_idx], cache.values[layer_idx], metadata, out=out,
            key_scales=cache.key_scales[layer_idx], value_scales=cache.value_scales[layer_idx],
        )

    def _query(
        self,
        layer_idx: int,
        x_norm: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """QKV projection, RoPE, and the new keys/values written to the cache."""
        layer = self.layers[layer_idx]
        qkv = self._linear(x_norm, layer.qkv, layer.qkv_bias)
        return rope_and_cache_write(
            qkv,
            rope[0],
            rope[1],
            cache.keys[layer_idx],
            cache.values[layer_idx],
            slot_mapping,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            use_triton=self.attention_backend == "triton",
            key_scales=cache.key_scales[layer_idx],
            value_scales=cache.value_scales[layer_idx],
        )

    def _finish_layer(
        self, layer_idx: int, attn: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """o_proj, residual + post-attention norm and the MLP: (x, residual)."""
        layer = self.layers[layer_idx]
        x = self._linear(attn.reshape(attn.shape[0], -1), layer.o_proj, layer.o_bias)
        x_norm, residual = fused_add_rms_norm(
            x, residual, layer.post_norm, eps=self.config.rms_norm_eps, use_triton=self.use_triton
        )
        gate_up = self._linear(x_norm, layer.gate_up, layer.gate_up_bias)
        x = self._linear(
            silu_and_mul(gate_up, use_triton=self.use_triton), layer.down, layer.down_bias
        )
        return x, residual

    def _linear(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear(x, weight, bias, backend=self.linear_backend)

    def _paged_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: BatchMetadata,
        out: Optional[torch.Tensor] = None,
        key_scales: Optional[torch.Tensor] = None,
        value_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_dim)
        use_triton = self.attention_backend == "triton"
        scales = dict(key_scales=key_scales, value_scales=value_scales)
        if metadata.decode_only:
            return paged_decode_attention(
                query,
                key_cache,
                value_cache,
                metadata.block_tables,
                metadata.context_lens,
                scale,
                use_triton=use_triton,
                out=out,
                **scales,
            )
        decode_rows = metadata.num_decode_rows
        if decode_rows == 0:
            return paged_prefill_attention(
                query,
                key_cache,
                value_cache,
                metadata.block_tables,
                metadata.cu_seqlens_q,
                metadata.context_lens,
                metadata.query_lens,
                metadata.context_lens_host,
                scale,
                use_triton=use_triton,
                out=out,
                **scales,
            )
        # Mixed batch: the decode rows come first (one token each, so token i
        # is row i) and take the split-KV decode kernel; the prefill rows
        # after them take the varlen prefill kernel, with their own aligned
        # copies of the per-row fields (see build_batch).
        if out is None:
            out = torch.empty_like(query)
        paged_decode_attention(
            query[:decode_rows],
            key_cache,
            value_cache,
            metadata.block_tables[:decode_rows],
            metadata.context_lens[:decode_rows],
            scale,
            use_triton=use_triton,
            out=out[:decode_rows],
            **scales,
        )
        paged_prefill_attention(
            query[decode_rows:],
            key_cache,
            value_cache,
            metadata.prefill_block_tables,
            metadata.prefill_cu_seqlens_q,
            metadata.prefill_context_lens,
            metadata.query_lens[decode_rows:],
            metadata.context_lens_host[decode_rows:],
            scale,
            use_triton=use_triton,
            out=out[decode_rows:],
            **scales,
        )
        return out

    def _mlp(self, x: torch.Tensor, layer: _LayerWeights) -> torch.Tensor:
        x_norm = rms_norm(
            x,
            layer.post_norm,
            eps=self.config.rms_norm_eps,
            use_triton=self.use_triton,
        )
        gate_up = linear(x_norm, layer.gate_up, layer.gate_up_bias)
        return linear(
            silu_and_mul(gate_up, use_triton=self.use_triton), layer.down, layer.down_bias
        )

    @torch.inference_mode()
    def forward_paged(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        query_lens: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        """Per-row logits for a batch given as tensors (the pre-metadata API).

        Kept for tests and scripts that build batches by hand: it moves the
        tensors to the host, builds BatchMetadata and runs forward_batch over
        every row. block_tables may be padded with -1.
        """
        if input_ids.ndim != 1:
            raise ValueError("forward_paged expects flat input_ids")
        if positions.shape != input_ids.shape:
            raise ValueError("positions must match input_ids shape")
        if slot_mapping.shape != input_ids.shape:
            raise ValueError("slot_mapping must match input_ids shape")
        tables = [
            [int(block) for block in row if int(block) >= 0]
            for row in block_tables.tolist()
        ]
        input_ids_t, positions_t, metadata = build_batch(
            input_ids=input_ids.tolist(),
            positions=positions.tolist(),
            slots=slot_mapping.tolist(),
            block_tables=tables,
            context_lens=context_lens.tolist(),
            query_lens=list(query_lens),
            is_prefill=is_prefill,
            null_block=cache.null_block,
            num_slots=cache.total_slots,
            device=self.device,
            all_logits=True,
        )
        return self.forward_batch(input_ids_t, positions_t, cache, metadata)

    def _resolve_lm_head(self) -> torch.Tensor | W8A16Weight:
        """lm_head.weight (int8 under lm_head_quant, _load_weight), or the
        embedding when tied (then quantized as a copy: the lookup stays exact)."""
        if self.config.tie_word_embeddings:
            embed = self.weights["model.embed_tokens.weight"]
            if self.lm_head_quant == "none":
                return embed
            keep_dense = self.lm_head_quant == "int8_dense"
            if embed.device.type != "cuda":
                return quantize_w8a16(embed, keep_dense=keep_dense)
            # Only the int8 copy is new; the dense one is the embedding itself.
            head = _build_on_stream(torch.cuda.Stream(embed.device), lambda: quantize_w8a16(embed))
            torch.cuda.empty_cache()
            return replace(head, dense=embed) if keep_dense else head
        if "lm_head.weight" not in self.weights:
            raise KeyError(
                "Model checkpoint is missing lm_head.weight and config.json does "
                "not set tie_word_embeddings=true"
            )
        return self.weights["lm_head.weight"]

    def _weight(self, name: str) -> torch.Tensor:
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(f"Missing nano-vLLM weight: {name}") from exc

    def _move_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(device=self.device, dtype=self.dtype)
        return tensor.to(device=self.device)

    def _load_weights(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """_load_weight for every weight; on CUDA each is built on a side
        stream, so its temporaries stay out of the weights' segments
        (_build_on_stream)."""
        if self.device.type != "cuda":
            return {name: self._load_weight(name, tensor) for name, tensor in weights.items()}
        scratch = torch.cuda.Stream(self.device)
        loaded = {
            name: _build_on_stream(scratch, functools.partial(self._load_weight, name, tensor))
            for name, tensor in weights.items()
        }
        torch.cuda.empty_cache()
        return loaded

    def _load_weight(self, name: str, tensor: torch.Tensor | AWQTensors) -> torch.Tensor | W4A16Weight:
        """Move one checkpoint tensor; quantize it here under weight_quant.

        One weight at a time, so the full-precision copy of a quantized
        projection is on the device only while it is being quantized. The
        layout follows W4A16_CONFIG["layout"]; a shape the tiled kernel
        cannot take stays in rows. AWQ projections are unpacked on the
        device and always take the tiled layout (the row-layout kernels have
        no zero points); their fp16 scales take the model dtype.
        """
        if isinstance(tensor, AWQTensors):
            if not name.endswith(QUANTIZED_LINEARS):
                raise ValueError(f"{name}: nano-vLLM reads AWQ weights for the decoder projections only")
            codes, zeros, scales = awq_unpack(
                *(t.to(self.device) for t in (tensor.qweight, tensor.qzeros, tensor.scales))
            )
            return pack_w4a16(codes, scales.to(self.dtype), zeros, layout="tiled")
        moved = self._move_weight(tensor)
        if name == "lm_head.weight" and self.lm_head_quant != "none":
            return quantize_w8a16(moved, keep_dense=self.lm_head_quant == "int8_dense")
        if self.weight_quant == "w4a16" and name.endswith(QUANTIZED_LINEARS):
            tiled = W4A16_CONFIG["layout"] == "tiled" and w4a16_tiled_supported(*moved.shape)
            return quantize_w4a16(moved, layout="tiled" if tiled else "rows")
        return moved

    def _layer_weights(self, layer_idx: int) -> _LayerWeights:
        prefix = f"model.layers.{layer_idx}"
        return _LayerWeights(
            input_norm=self._weight(f"{prefix}.input_layernorm.weight"),
            qkv=self._weight(f"{prefix}.self_attn.qkv_proj.weight"),
            qkv_bias=self.weights.get(f"{prefix}.self_attn.qkv_proj.bias"),
            o_proj=self._weight(f"{prefix}.self_attn.o_proj.weight"),
            o_bias=self.weights.get(f"{prefix}.self_attn.o_proj.bias"),
            post_norm=self._weight(f"{prefix}.post_attention_layernorm.weight"),
            gate_up=self._weight(f"{prefix}.mlp.gate_up_proj.weight"),
            gate_up_bias=self.weights.get(f"{prefix}.mlp.gate_up_proj.bias"),
            down=self._weight(f"{prefix}.mlp.down_proj.weight"),
            down_bias=self.weights.get(f"{prefix}.mlp.down_proj.bias"),
        )

    def _shape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, num_heads, self.config.head_dim).transpose(
            1, 2
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeats == 1:
            return x
        return x.repeat_interleave(repeats, dim=1)

    def _validate_required_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        required = [
            "model.embed_tokens.weight",
            "model.norm.weight",
        ]
        # An AWQ checkpoint stores each projection as qweight/qzeros/scales.
        suffixes = AWQ_SUFFIXES if self.weight_quant == "awq" else ("weight",)
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            required.extend(
                [f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"]
            )
            for proj in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ):
                required.extend(f"{prefix}.{proj}.{suffix}" for suffix in suffixes)
        missing = [name for name in required if name not in weights]
        if missing:
            raise KeyError(
                f"Model checkpoint is missing required Llama weights: {missing[:5]}"
            )
