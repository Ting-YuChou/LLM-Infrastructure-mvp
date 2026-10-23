"""The FP8 KV cache's integer e4m3 codec against torch.float8_e4m3fn.

The A100 has no fp8 conversions and Triton refuses float8e4nv below sm_89,
so kernels/fp8.py encodes and decodes with integer arithmetic; it must agree
with torch bit for bit, ties and subnormals included.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = triton.language

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from serving.nano_vllm.kernels.fp8 import (  # noqa: E402
    FP8_DECODE_SCALE,
    fp8_decode_reference,
    fp8_decode_scaled_bf16,
    fp8_encode,
    fp8_encode_reference,
)


@triton.jit
def _decode_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, fp8_decode_scaled_bf16(tl.load(src_ptr + offs, mask=mask, other=0)), mask=mask)


@triton.jit
def _encode_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, fp8_encode(tl.load(src_ptr + offs, mask=mask, other=0.0)), mask=mask)


def _finite_codes(device):
    codes = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
    return codes[(codes & 0x7F) != 0x7F]  # 0x7F and 0xFF are NaN


def decode(bits):
    out = torch.empty(bits.shape, dtype=torch.bfloat16, device=bits.device)
    _decode_kernel[(triton.cdiv(bits.numel(), 1024),)](bits, out, bits.numel(), BLOCK=1024)
    return out


def encode(y):
    out = torch.empty(y.shape, dtype=torch.uint8, device=y.device)
    _encode_kernel[(triton.cdiv(y.numel(), 1024),)](y, out, y.numel(), BLOCK=1024)
    return out


def test_decode_matches_torch_for_every_code():
    codes = _finite_codes("cuda")
    assert torch.equal(decode(codes).float() * FP8_DECODE_SCALE, fp8_decode_reference(codes))


def test_encode_matches_torch_rounding_ties_subnormals_and_saturation():
    grid = fp8_decode_reference(_finite_codes("cuda")).sort().values
    ties = (grid[1:] + grid[:-1]) / 2  # exact halfway points: round to even
    generator = torch.Generator(device="cuda").manual_seed(0)
    magnitudes = torch.exp2(torch.empty(100_000, device="cuda").uniform_(-14, 10, generator=generator))
    signs = torch.where(torch.rand(100_000, device="cuda", generator=generator) < 0.5, -1.0, 1.0)
    edges = torch.tensor([0.0, -0.0, 448.0, -448.0, 460.0, 464.0, 1e6, -1e6, 2.0**-6, 2.0**-10],
                         device="cuda")
    y = torch.cat([grid, ties, magnitudes * signs, edges])
    assert torch.equal(encode(y), fp8_encode_reference(y))
