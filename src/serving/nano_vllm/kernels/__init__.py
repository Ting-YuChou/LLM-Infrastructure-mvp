"""Kernel helpers with PyTorch fallbacks and optional Triton fast paths."""

from .attention import causal_attention, decode_attention
from .paged_attention import (
    paged_decode_attention,
    paged_decode_attention_reference,
    paged_prefill_attention,
    paged_prefill_attention_reference,
)
from .rmsnorm import is_triton_available as rmsnorm_triton_available
from .rmsnorm import rms_norm, rms_norm_reference
from .rotary import apply_rope, apply_rope_cos_sin, rope_cos_sin
from .silu_mul import silu_mul, silu_mul_reference

__all__ = [
    "apply_rope",
    "apply_rope_cos_sin",
    "causal_attention",
    "decode_attention",
    "paged_decode_attention",
    "paged_decode_attention_reference",
    "paged_prefill_attention",
    "paged_prefill_attention_reference",
    "rms_norm",
    "rms_norm_reference",
    "rmsnorm_triton_available",
    "rope_cos_sin",
    "silu_mul",
    "silu_mul_reference",
]
