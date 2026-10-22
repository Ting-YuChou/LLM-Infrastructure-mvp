"""Kernel helpers with PyTorch fallbacks and optional Triton fast paths."""

from .attention import causal_attention, decode_attention
from .linear import linear, load_cuda_extension
from .paged_attention import (
    decode_config,
    decode_num_splits,
    paged_decode_attention,
    paged_decode_attention_reference,
    paged_prefill_attention,
    paged_prefill_attention_reference,
)
from .rmsnorm import is_triton_available as rmsnorm_triton_available
from .rmsnorm import (
    fused_add_rms_norm,
    fused_add_rms_norm_reference,
    rms_norm,
    rms_norm_reference,
)
from .rope_cache import rope_and_cache_write, rope_and_cache_write_reference
from .rotary import apply_rope, apply_rope_cos_sin, rope_cos_sin
from .silu_mul import silu_and_mul, silu_and_mul_reference, silu_mul, silu_mul_reference
from .w4a16 import (
    W4A16Weight,
    dequantize_w4a16,
    quantize_w4a16,
    w4a16_linear,
    w4a16_linear_reference,
)

__all__ = [
    "W4A16Weight",
    "apply_rope",
    "apply_rope_cos_sin",
    "causal_attention",
    "decode_attention",
    "decode_config",
    "decode_num_splits",
    "dequantize_w4a16",
    "fused_add_rms_norm",
    "fused_add_rms_norm_reference",
    "linear",
    "load_cuda_extension",
    "paged_decode_attention",
    "paged_decode_attention_reference",
    "paged_prefill_attention",
    "paged_prefill_attention_reference",
    "quantize_w4a16",
    "rms_norm",
    "rms_norm_reference",
    "rmsnorm_triton_available",
    "rope_and_cache_write",
    "rope_and_cache_write_reference",
    "rope_cos_sin",
    "silu_and_mul",
    "silu_and_mul_reference",
    "silu_mul",
    "silu_mul_reference",
    "w4a16_linear",
    "w4a16_linear_reference",
]
