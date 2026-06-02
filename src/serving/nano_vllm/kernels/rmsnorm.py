"""RMSNorm fallback and optional Triton kernel."""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only in CUDA/Triton environments
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - default on CPU-only hosts
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        var = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(var + eps) * w
        tl.store(y_ptr + row * n_cols + offsets, y, mask=mask)


def is_triton_available() -> bool:
    return TRITON_AVAILABLE


def rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
    return normalized * weight


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_triton: bool = False,
) -> torch.Tensor:
    """Apply RMSNorm, using Triton only when explicitly requested and available."""
    if (
        use_triton
        and TRITON_AVAILABLE
        and x.is_cuda
        and weight.is_cuda
        and x.shape[-1] <= 8192
    ):  # pragma: no cover - depends on CUDA/Triton
        flat = x.contiguous().view(-1, x.shape[-1])
        output = torch.empty_like(flat)
        block = triton.next_power_of_2(flat.shape[-1])
        _rms_norm_kernel[(flat.shape[0],)](
            flat,
            weight.contiguous(),
            output,
            flat.shape[-1],
            eps,
            BLOCK=block,
        )
        return output.view_as(x)

    return rms_norm_reference(x, weight, eps=eps)
