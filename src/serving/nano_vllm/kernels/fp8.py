"""FP8 (e4m3) KV-cache values without Triton's fp8 types.

Triton 3.6 accepts ``float8e4nv`` only from sm_89 on (its CUDA backend lists
fp8e5 and fp8e4b15 for the A100), and the A100 has no fp8 conversion
instructions anyway. So the kernels see the cache as uint8 and the codec is
integer arithmetic:

- decode: the 7 exponent and mantissa bits go to bits 4-10 of a bf16, the
  sign to bit 15. That bf16 is exactly the e4m3 value times 2**-120
  (subnormals too: bf16 exponent 0 is subnormal like e4m3's), so ``tl.dot``
  takes it as it is and the 2**120 joins the per-token scale. fp32 shares
  bf16's exponent, so the same bits at 20-26 give the same value; the
  kernels use that one when the query is fp32.
- encode: ``y = x / scale`` clamped to +-448 and rounded to nearest even on
  e4m3's grid: normal values drop 20 of fp32's 23 mantissa bits, subnormals
  (|y| < 2**-6) round |y| * 512.

Each (token, kv head) vector of head_dim values has one fp32 scale, its
largest magnitude over 448, so the codes use e4m3's whole range. The
attention kernels never multiply a decoded value by its scale: K's scale
multiplies that token's score and V's scale its probability.

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
# Its square root, applied twice: the attention kernels multiply q by it
# before the dot with decoded keys and the scores by K's scale times it, and
# likewise the probabilities by V's scale times it and the output by it, so
# no product or partial sum comes near fp32's subnormals.
FP8_HALF_SCALE = 2.0**60
KV_CACHE_DTYPES = ("auto", "fp8")


def fp8_encode_reference(y: torch.Tensor) -> torch.Tensor:
    """Values already divided by their scale -> e4m3 bytes (uint8)."""
    return y.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def fp8_decode_reference(bits: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes (uint8) -> their values in fp32."""
    return bits.view(torch.float8_e4m3fn).float()


def fp8_quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., head_dim] -> (float8_e4m3fn codes, fp32 scale per head vector).

    The scale maps the vector's largest magnitude to 448; an all-zero vector
    gets scale 1 (its codes are all zero either way).
    """
    x = x.float()
    amax = x.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    codes = (x / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def fp8_dequantize_reference(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """float8_e4m3fn codes [..., head_dim] and scales [...] -> fp32 values."""
    return codes.float() * scale.unsqueeze(-1)


if TRITON_AVAILABLE:  # pragma: no cover - depends on CUDA/Triton

    _HALF = tl.constexpr(FP8_HALF_SCALE)

    @triton.jit
    def fp8_decode_scaled_bf16(bits):
        """uint8 e4m3 bytes -> bf16 values times 2**-120 (FP8_DECODE_SCALE)."""
        b = bits.to(tl.int32)
        h = ((b & 0x80) << 8) | ((b & 0x7F) << 4)
        return h.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def fp8_decode_scaled_f32(bits):
        """uint8 e4m3 bytes -> fp32 values times 2**-120 (FP8_DECODE_SCALE)."""
        b = bits.to(tl.int32)
        return (((b & 0x80) << 24) | ((b & 0x7F) << 20)).to(tl.float32, bitcast=True)

    @triton.jit
    def fp8_prescale_query(q):
        """q times 2**60 for the dot with decoded keys; anything but fp32 becomes bf16."""
        if q.dtype == tl.float32:
            return q * _HALF
        else:
            return (q.to(tl.float32) * _HALF).to(tl.bfloat16)

    @triton.jit
    def fp8_decode_scaled(bits, q):
        """Decoded keys or values in the dtype of the prescaled query ``q``."""
        if q.dtype == tl.float32:
            return fp8_decode_scaled_f32(bits)
        else:
            return fp8_decode_scaled_bf16(bits)

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
