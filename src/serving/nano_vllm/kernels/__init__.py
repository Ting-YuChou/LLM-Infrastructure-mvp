"""Kernel helpers with PyTorch fallbacks and optional Triton fast paths."""

from .attention import causal_attention, decode_attention
from .rmsnorm import is_triton_available as rmsnorm_triton_available
from .rmsnorm import rms_norm, rms_norm_reference
from .rotary import apply_rope
from .silu_mul import silu_mul, silu_mul_reference

__all__ = [
    "apply_rope",
    "causal_attention",
    "decode_attention",
    "rms_norm",
    "rms_norm_reference",
    "rmsnorm_triton_available",
    "silu_mul",
    "silu_mul_reference",
]
