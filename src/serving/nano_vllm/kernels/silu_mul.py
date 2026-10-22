"""Fused SiLU-gate multiplication fallbacks and optional Triton kernels."""

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
        # Compute in fp32: tl.exp rejects bf16/fp16 inputs, and torch's SiLU
        # also upcasts internally before rounding back.
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_and_mul_kernel(x_ptr, out_ptr, stride_x, stride_out, inter, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < inter
        gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + row * stride_x + inter + cols, mask=mask, other=0.0).to(tl.float32)
        out = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(out_ptr + row * stride_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


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


def silu_and_mul_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return silu_mul_reference(gate, up)


def silu_and_mul(gate_up: torch.Tensor, use_triton: bool = False) -> torch.Tensor:
    """SiLU(gate) * up for the fused gate_up projection output [..., 2 * inter]."""
    if use_triton and TRITON_AVAILABLE and gate_up.is_cuda:  # pragma: no cover
        inter = gate_up.shape[-1] // 2
        flat = gate_up.reshape(-1, 2 * inter)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        out = torch.empty(flat.shape[0], inter, dtype=gate_up.dtype, device=gate_up.device)
        block = 1024
        grid = (flat.shape[0], triton.cdiv(inter, block))
        _silu_and_mul_kernel[grid](flat, out, flat.stride(0), out.stride(0), inter, BLOCK=block)
        return out.view(*gate_up.shape[:-1], inter)

    return silu_and_mul_reference(gate_up)
