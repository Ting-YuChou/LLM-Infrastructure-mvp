"""RMSNorm (and residual add + RMSNorm) fallbacks and optional Triton kernels."""

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
        # Square and reduce in fp32: fp16 x*x overflows once |x| exceeds ~256.
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        # Round the normalized value to the output dtype before the weight
        # multiply, like transformers' LlamaRMSNorm.
        out_type = y_ptr.dtype.element_ty
        normalized = (x * tl.rsqrt(var + eps)).to(out_type).to(tl.float32)
        tl.store(y_ptr + row * n_cols + offsets, (normalized * w).to(out_type), mask=mask)

    @triton.jit
    def _fused_add_rms_norm_kernel(
        x_ptr, residual_ptr, w_ptr, y_ptr, residual_out_ptr, n_cols, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row * n_cols + offsets, mask=mask, other=0.0)
        out_type = residual_out_ptr.dtype.element_ty
        # The sum is rounded to the model dtype first (it is the next
        # residual), and the variance is taken of the rounded value.
        total = (x + residual.to(tl.float32)).to(out_type)
        tl.store(residual_out_ptr + row * n_cols + offsets, total, mask=mask)
        total = total.to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(total * total, axis=0) / n_cols
        normalized = (total * tl.rsqrt(var + eps)).to(out_type).to(tl.float32)
        tl.store(y_ptr + row * n_cols + offsets, (normalized * w).to(out_type), mask=mask)


def is_triton_available() -> bool:
    return TRITON_AVAILABLE


def rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # Matches transformers' LlamaRMSNorm: normalize in fp32, then cast back.
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = (x_fp32 * torch.rsqrt(variance + eps)).to(dtype=x.dtype)
    return weight * normalized


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


def fused_add_rms_norm_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = residual + x
    return rms_norm_reference(residual, weight, eps=eps), residual


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rms_norm(residual + x), residual + x) in one pass over the rows.

    A decoder layer adds each sublayer's output to the residual stream and
    normalizes the result for the next sublayer; this does both with one read
    of each input. Inputs are not modified.
    """
    if (
        use_triton
        and TRITON_AVAILABLE
        and x.is_cuda
        and x.shape[-1] <= 8192
    ):  # pragma: no cover - depends on CUDA/Triton
        flat_x = x.contiguous().view(-1, x.shape[-1])
        flat_residual = residual.contiguous().view(-1, x.shape[-1])
        output = torch.empty_like(flat_x)
        residual_out = torch.empty_like(flat_residual)
        _fused_add_rms_norm_kernel[(flat_x.shape[0],)](
            flat_x,
            flat_residual,
            weight.contiguous(),
            output,
            residual_out,
            flat_x.shape[-1],
            eps,
            BLOCK=triton.next_power_of_2(flat_x.shape[-1]),
        )
        return output.view_as(x), residual_out.view_as(residual)

    return fused_add_rms_norm_reference(x, residual, weight, eps=eps)
