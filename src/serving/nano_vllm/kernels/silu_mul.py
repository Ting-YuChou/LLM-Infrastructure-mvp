"""Fused SiLU-gate multiplication fallback and optional Triton kernel."""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only in CUDA/Triton environments
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    @triton.jit
    def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        out = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(out_ptr + offsets, out, mask=mask)


def silu_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def silu_mul(
    gate: torch.Tensor, up: torch.Tensor, use_triton: bool = False
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")

    if (
        use_triton and TRITON_AVAILABLE and gate.is_cuda and up.is_cuda
    ):  # pragma: no cover
        gate_flat = gate.contiguous().view(-1)
        up_flat = up.contiguous().view(-1)
        out = torch.empty_like(gate_flat)
        block = 256
        grid = (triton.cdiv(gate_flat.numel(), block),)
        _silu_mul_kernel[grid](gate_flat, up_flat, out, gate_flat.numel(), BLOCK=block)
        return out.view_as(gate)

    return silu_mul_reference(gate, up)
