"""Rotary positional embedding helper.

Uses the HuggingFace Llama layout ("rotate_half"): dimension ``i`` is paired
with dimension ``i + head_dim // 2``. HF checkpoints permute q_proj/k_proj
weights for this layout, so the interleaved (``x[0::2]``, ``x[1::2]``) pairing
from the original Meta reference code produces wrong outputs on them.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch

SUPPORTED_ROPE_TYPES = ("default", "linear", "llama3")


def rope_type_from_scaling(rope_scaling: Optional[Mapping[str, Any]]) -> str:
    if not rope_scaling:
        return "default"
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    return str(rope_type or "default").lower()


def validate_rope_scaling(rope_scaling: Optional[Mapping[str, Any]]) -> None:
    """Raise a clear error for rope_scaling variants nano-vLLM does not model."""
    rope_type = rope_type_from_scaling(rope_scaling)
    if rope_type not in SUPPORTED_ROPE_TYPES:
        raise ValueError(
            f"nano-vLLM does not support rope_scaling type {rope_type!r}; "
            f"supported types: {', '.join(SUPPORTED_ROPE_TYPES)}"
        )
    required = {
        "linear": ("factor",),
        "llama3": (
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        ),
    }.get(rope_type, ())
    missing = [key for key in required if key not in (rope_scaling or {})]
    if missing:
        raise ValueError(
            f"rope_scaling type {rope_type!r} is missing required keys: {missing}"
        )


def compute_inv_freq(
    head_dim: int,
    theta: float = 10000.0,
    rope_scaling: Optional[Mapping[str, Any]] = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return fp32 inverse frequencies, matching transformers' RoPE init."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    validate_rope_scaling(rope_scaling)
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    rope_type = rope_type_from_scaling(rope_scaling)
    if rope_type == "linear":
        return inv_freq / float(rope_scaling["factor"])
    if rope_type == "llama3":
        return _llama3_inv_freq(inv_freq, rope_scaling)
    return inv_freq


def _llama3_inv_freq(
    inv_freq: torch.Tensor, rope_scaling: Mapping[str, Any]
) -> torch.Tensor:
    """Llama 3.1-style frequency scaling (transformers' ``llama3`` rope type)."""
    factor = float(rope_scaling["factor"])
    low_freq_factor = float(rope_scaling["low_freq_factor"])
    high_freq_factor = float(rope_scaling["high_freq_factor"])
    old_context_len = float(rope_scaling["original_max_position_embeddings"])

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    # Long wavelengths are divided by factor, short ones kept, middle smoothed.
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_one(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000.0,
    use_triton: bool = False,
    inv_freq: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors shaped [batch, heads, tokens, head_dim].

    ``inv_freq`` (from :func:`compute_inv_freq`) carries any rope_scaling; when
    omitted, unscaled frequencies are derived from ``theta``.
    """
    del use_triton  # RoPE stays a readable PyTorch helper; there is no Triton path.
    if query.shape[-1] % 2 != 0 or key.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    device = query.device
    dtype = query.dtype
    head_dim = query.shape[-1]
    if inv_freq is None:
        inv_freq = compute_inv_freq(head_dim, theta, device=device)
    positions = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.to(device=device, dtype=torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype).view(1, 1, positions.numel(), -1)
    sin = emb.sin().to(dtype=dtype).view(1, 1, positions.numel(), -1)
    return _apply_rope_one(query, cos, sin), _apply_rope_one(key, cos, sin)


def rope_cos_sin(
    inv_freq: torch.Tensor,
    num_positions: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape [num_positions, head_dim] for positions 0..n-1.

    Computed exactly like :func:`apply_rope` (fp32, then cast), so indexing
    the tables by position gives the same values as recomputing them.
    """
    positions = torch.arange(num_positions, device=inv_freq.device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.to(dtype=torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rope_cos_sin(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate ``x`` with cos/sin already gathered for its positions (broadcast)."""
    return _apply_rope_one(x, cos, sin)
