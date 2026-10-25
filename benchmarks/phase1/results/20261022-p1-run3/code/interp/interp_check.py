"""Run nano_vllm's Triton kernels under TRITON_INTERPRET=1 on CPU against the references.

macOS has no Triton wheel, so this runs in a linux/arm64 container:

  docker build -t nano-triton-interp benchmarks/phase1/interp
  docker run --rm -v "$PWD":/repo:ro nano-triton-interp python /repo/benchmarks/phase1/interp/interp_check.py

It launches the @triton.jit kernels directly (the nano_vllm wrappers fall back
to PyTorch on CPU tensors) and catches indexing and masking bugs before a GPU
run. It cannot measure speed, and the interpreter's bf16 tl.dot is broken
(dot_check.py), so kernels that use tl.dot are checked in fp32 and fp16.
"""
import importlib.util
import math
import sys
import traceback

import torch

ROOT = "/repo/src/serving/nano_vllm/kernels"
spec = importlib.util.spec_from_file_location(
    "kernels", f"{ROOT}/__init__.py", submodule_search_locations=[ROOT]
)
K = importlib.util.module_from_spec(spec)
sys.modules["kernels"] = K
spec.loader.exec_module(K)
# By module: kernels/__init__ re-exports functions named like linear and silu_mul.
L, PA, RN, RC, SM = (sys.modules[f"kernels.{m}"] for m in ("linear", "paged_attention", "rmsnorm", "rope_cache", "silu_mul"))

assert PA.TRITON_AVAILABLE, "triton not importable"
torch.manual_seed(0)
results = []


def check(name, out, ref, atol, rtol):
    outs = out if isinstance(out, tuple) else (out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    ok, worst = True, 0.0
    for o, r in zip(outs, refs):
        err = (o.float() - r.float()).abs()
        ok &= bool((err <= atol + rtol * r.float().abs()).all())
        worst = max(worst, float(err.max()))
    results.append((name, ok, worst))
    print(f"{'OK  ' if ok else 'FAIL'} {name}: max err {worst:.3g}", flush=True)


def run(name, fn):
    try:
        fn()
    except Exception:
        traceback.print_exc()
        results.append((name, False, float("nan")))
        print(f"FAIL {name}: exception", flush=True)


def cache(num_blocks, page, kvh, d, dtype):
    shape = (num_blocks + 1, page, kvh, d)
    return torch.randn(shape).to(dtype), torch.randn(shape).to(dtype)


def tables(ctxs, page, num_blocks):
    order = torch.randperm(num_blocks).tolist()
    rows = []
    for c in ctxs:
        need = max(1, (c + page - 1) // page)
        rows.append([order.pop() for _ in range(need)])
    w = max(len(r) for r in rows)
    return torch.tensor([r + [num_blocks] * (w - len(r)) for r in rows], dtype=torch.int32)


def decode_case(heads, page, ctxs, splits, dtype, config=None):
    qh, kvh, d = heads
    nb = sum((c + page - 1) // page for c in ctxs) + 2
    k, v = cache(nb, page, kvh, d, dtype)
    t = tables(ctxs, page, nb)
    q = torch.randn(len(ctxs), qh, d).to(dtype)
    lens = torch.tensor(ctxs, dtype=torch.int32)
    scale = 1 / math.sqrt(d)
    out = PA._paged_decode_attention_triton(
        q, k, v, t, lens, scale, splits, config or PA.decode_config(len(lens))
    )
    ref = PA.paged_decode_attention_reference(q, k, v, t, lens, scale)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    check(f"decode heads={heads} page={page} ctx={ctxs} S={splits} {dtype} {config or ''}", out, ref, tol, tol)


def prefill_case(heads, page, shapes, dtype):
    qh, kvh, d = heads
    qlens = [q for q, _ in shapes]
    ctxs = [c for _, c in shapes]
    nb = sum((c + page - 1) // page for c in ctxs) + 2
    k, v = cache(nb, page, kvh, d, dtype)
    t = tables(ctxs, page, nb)
    q = torch.randn(sum(qlens), qh, d).to(dtype)
    cu = torch.tensor([0] + qlens).cumsum(0).to(torch.int32)
    lens = torch.tensor(ctxs, dtype=torch.int32)
    scale = 1 / math.sqrt(d)
    out = PA._paged_prefill_attention_triton(q, k, v, t, cu, lens, max(qlens), scale, PA.PREFILL_CONFIG)
    ref = PA.paged_prefill_attention_reference(q, k, v, t, qlens, ctxs, scale)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    check(f"prefill heads={heads} page={page} shapes={shapes} {dtype}", out, ref, tol, tol)


def rope_case(heads, n, dtype):
    qh, kvh, d = heads
    inv = K.rotary.compute_inv_freq(d, 500000.0)
    cos_t, sin_t = K.rotary.rope_cos_sin(inv, 256, dtype)
    pos = torch.randint(0, 256, (n,))
    cos, sin = cos_t[pos], sin_t[pos]
    qkv = torch.randn(n, (qh + 2 * kvh) * d).to(dtype)
    nb, page = 8, 4
    slots = torch.randperm(nb * page)[:n]
    slots[-1] = nb * page  # null block
    outs = []
    for fn in (RC._rope_and_cache_write_triton, RC.rope_and_cache_write_reference):
        torch.manual_seed(1)
        k, v = cache(nb, page, kvh, d, dtype)
        q = fn(qkv, cos, sin, k, v, slots, qh, kvh)
        outs.append((q, k, v))
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    check(f"rope_cache heads={heads} n={n} {dtype}", outs[0], outs[1], tol, tol)


def norm_case(rows, n, dtype):
    x = torch.randn(rows, n).to(dtype)
    r = (torch.randn(rows, n) * 4).to(dtype)
    w = (1 + 0.1 * torch.randn(n)).to(dtype)
    out = torch.empty_like(x)
    RN._rms_norm_kernel[(rows,)](x, w, out, n, 1e-5, BLOCK=triton.next_power_of_2(n))
    tol = 1e-5 if dtype == torch.float32 else 3.2e-2
    check(f"rms_norm rows={rows} n={n} {dtype}", out, RN.rms_norm_reference(x, w, 1e-5), tol, 1.6e-2)
    y, ro = torch.empty_like(x), torch.empty_like(r)
    RN._fused_add_rms_norm_kernel[(rows,)](x, r, w, y, ro, n, 1e-5, BLOCK=triton.next_power_of_2(n))
    check(f"fused_add_rms_norm rows={rows} n={n} {dtype}", (y, ro),
          RN.fused_add_rms_norm_reference(x, r, w, 1e-5), tol, 1.6e-2)


def silu_case(rows, inter, dtype):
    gu = (torch.randn(rows, 2 * inter) * 3).to(dtype)
    out = torch.empty(rows, inter, dtype=dtype)
    block = 128
    SM._silu_and_mul_kernel[(rows, triton.cdiv(inter, block))](gu, out, gu.stride(0), out.stride(0), inter, BLOCK=block)
    tol = 1e-5 if dtype == torch.float32 else 3.2e-2
    check(f"silu_and_mul rows={rows} inter={inter} {dtype}", out, SM.silu_and_mul_reference(gu), tol, 1.6e-2)


def gemv_case(m, n, k, split):
    # fp16, not bf16: the interpreter's bf16 tl.dot is broken (dot_check.py).
    x = torch.randn(m, k).to(torch.float16)
    w = (torch.randn(n, k) / k**0.5).to(torch.float16)
    out = L._gemv_triton(x, w, split)
    check(f"gemv m={m} n={n} k={k} split={split}", out, x.float() @ w.float().T, 1.6e-2, 1.6e-2)


import triton  # noqa: E402

LLAMA = (32, 8, 128)
TINY = (4, 2, 16)
for heads in (TINY, LLAMA):
    for page in (4, 16):
        for splits in (1, 4):
            run("decode", lambda: decode_case(heads, page, [1, 15, 17, 70, 0], splits, torch.float32))
    run("decode fp16", lambda: decode_case(heads, 16, [5, 130], 2, torch.float16))
    run("decode large-batch config", lambda: decode_case(
        heads, 16, [1, 129, 300, 0], 2, torch.float32, PA.DECODE_CONFIG_LARGE_BATCH))
    run("prefill", lambda: prefill_case(heads, 4, [(5, 5), (3, 10), (1, 7), (70, 70)], torch.float32))
    run("prefill fp16", lambda: prefill_case(heads, 16, [(20, 40), (65, 65)], torch.float16))
    for n in (1, 5):
        run("rope", lambda: rope_case(heads, n, torch.float32))
    run("rope bf16", lambda: rope_case(heads, 3, torch.bfloat16))
for dtype in (torch.float32, torch.bfloat16):
    run("norm", lambda: norm_case(3, 256, dtype))
    run("silu", lambda: silu_case(3, 200, dtype))
for split in (1, 2):
    run("gemv", lambda: gemv_case(3, 100, 512, split))
run("gemv ragged", lambda: gemv_case(1, 70, 300, 2))

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
