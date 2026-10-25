"""Shows the Triton interpreter's bf16 tl.dot is wrong (fp32 and fp16 are exact)."""
import torch, triton, triton.language as tl

@triton.jit
def k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M); n = tl.arange(0, N); kk = tl.arange(0, K)
    a = tl.load(a_ptr + m[:, None] * K + kk[None, :])
    b = tl.load(b_ptr + n[:, None] * K + kk[None, :])
    tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, tl.trans(b)))

for dtype in (torch.float32, torch.bfloat16, torch.float16):
    a = torch.randn(16, 32).to(dtype); b = torch.randn(16, 32).to(dtype)
    c = torch.empty(16, 16, dtype=torch.float32)
    k[(1,)](a, b, c, 16, 16, 32)
    print(dtype, float((c - a.float() @ b.float().T).abs().max()))
