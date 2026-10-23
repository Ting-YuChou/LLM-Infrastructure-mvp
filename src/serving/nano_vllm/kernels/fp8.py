"""FP8 (e4m3) KV-cache values without Triton's fp8 types.

Triton 3.6 accepts ``float8e4nv`` only from sm_89 on (its CUDA backend lists
fp8e5 and fp8e4b15 for the A100), and the A100 has no fp8 conversion
instructions anyway. So the kernels see the cache as uint8 and the codec is
integer arithmetic:

- decode: the 7 exponent and mantissa bits go to bits 4-10 of a bf16, the
  sign to bit 15. That bf16 is exactly the e4m3 value times 2**-120
  (subnormals too: bf16 exponent 0 is subnormal like e4m3's), so ``tl.dot``
  takes it as it is and the 2**120 joins the per-token scale.
- encode: ``y = x / scale`` clamped to +-448 and rounded to nearest even on
  e4m3's grid: normal values drop 20 of fp32's 23 mantissa bits, subnormals
  (|y| < 2**-6) round |y| * 512.

The torch functions (torch.float8_e4m3fn conversions) are the reference.
"""

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

FP8_MAX = 448.0
# A decoded bf16 times this is the e4m3 value.
FP8_DECODE_SCALE = 2.0**120


def fp8_encode_reference(y: torch.Tensor) -> torch.Tensor:
    """Values already divided by their scale -> e4m3 bytes (uint8)."""
    return y.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def fp8_decode_reference(bits: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes (uint8) -> their values in fp32."""
    return bits.view(torch.float8_e4m3fn).float()


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    @triton.jit
    def fp8_decode_scaled_bf16(bits):
        """uint8 e4m3 bytes -> bf16 values times 2**-120 (FP8_DECODE_SCALE)."""
        b = bits.to(tl.int32)
        h = ((b & 0x80) << 8) | ((b & 0x7F) << 4)
        return h.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def fp8_encode(y):
        """fp32 values already divided by their scale -> uint8 e4m3 bytes."""
        y = tl.minimum(tl.maximum(y, -448.0), 448.0)
        bits = y.to(tl.int32, bitcast=True)
        sign = (bits >> 24) & 0x80
        a = bits & 0x7FFFFFFF
        # Normal: round to nearest even onto 3 mantissa bits; the carry may
        # move the exponent, which the subtraction rebiases (127 -> 7).
        normal = ((a + 0x7FFFF + ((a >> 20) & 1)) >> 20) - (120 << 3)
        # Subnormal: multiples of 2**-9, nearest even.
        t = tl.abs(y) * 512.0
        f = tl.floor(t)
        r = t - f
        fi = f.to(tl.int32)
        sub = fi + ((r > 0.5) | ((r == 0.5) & ((fi & 1) == 1))).to(tl.int32)
        code = tl.where(a >= (121 << 23), normal, sub)  # 121 << 23: 2**-6
        return (sign | code).to(tl.uint8)
