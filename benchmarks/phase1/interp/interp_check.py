"""Run nano_vllm's Triton kernels under TRITON_INTERPRET=1 on CPU against the references.

macOS has no Triton wheel, so this runs in a linux/arm64 container:

  docker build -t nano-triton-interp benchmarks/phase1/interp
  docker run --rm -v "$PWD":/repo:ro nano-triton-interp python /repo/benchmarks/phase1/interp/interp_check.py

INTERP_ONLY=<regex> (docker run -e) runs only the cases whose name matches.

It launches the @triton.jit kernels directly (the nano_vllm wrappers fall back
to PyTorch on CPU tensors) and catches indexing and masking bugs before a GPU
run. It cannot measure speed, and the interpreter's bf16 tl.dot is broken
(dot_check.py), so kernels that use tl.dot are checked in fp32 and fp16.
"""
import importlib.util
import math
import os
import re
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
L, PA, RN, RC, SM, F8, BA = (
    sys.modules[f"kernels.{m}"]
    for m in ("linear", "paged_attention", "rmsnorm", "rope_cache", "silu_mul", "fp8", "block_attention")
)

assert PA.TRITON_AVAILABLE, "triton not importable"
torch.manual_seed(0)
results = []
ONLY = os.environ.get("INTERP_ONLY", "")  # regex on the case names passed to run()


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
    if not re.search(ONLY, name):
        return
    try:
        fn()
    except Exception:
        traceback.print_exc()
        results.append((name, False, float("nan")))
        print(f"FAIL {name}: exception", flush=True)


def cache(num_blocks, page, kvh, d, dtype):
    shape = (num_blocks + 1, page, kvh, d)
    return torch.randn(shape).to(dtype), torch.randn(shape).to(dtype)


def fp8_cache(num_blocks, page, kvh, d):
    """K/V codes (float8_e4m3fn) and their per-vector scales; values span two decades
    (wider makes the scores so large that fp32 rounding alone exceeds the tolerance)."""
    shape = (num_blocks + 1, page, kvh, d)
    out = []
    for _ in range(2):
        x = torch.randn(shape) * torch.logspace(-1, 1, num_blocks + 1).view(-1, 1, 1, 1)
        out += list(F8.fp8_quantize_reference(x))
    return out  # k, k_scale, v, v_scale


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


def mixed_case(heads, page, decode_ctxs, prefill_shapes, dtype):
    """A mixed batch: decode rows into out[:B], prefill rows (own offsets) into out[B:]."""
    qh, kvh, d = heads
    qlens = [1] * len(decode_ctxs) + [q for q, _ in prefill_shapes]
    ctxs = list(decode_ctxs) + [c for _, c in prefill_shapes]
    nb = sum((c + page - 1) // page for c in ctxs) + 2
    k, v = cache(nb, page, kvh, d, dtype)
    t = tables(ctxs, page, nb)
    q = torch.randn(sum(qlens), qh, d).to(dtype)
    lens = torch.tensor(ctxs, dtype=torch.int32)
    b = len(decode_ctxs)
    prefill_cu = torch.tensor([0] + qlens[b:]).cumsum(0).to(torch.int32)
    scale = 1 / math.sqrt(d)
    out = torch.full_like(q, float("nan"))
    PA._paged_decode_attention_triton(
        q[:b], k, v, t[:b], lens[:b], scale, 2, PA.decode_config(b), out[:b]
    )
    PA._paged_prefill_attention_triton(
        q[b:], k, v, t[b:], prefill_cu, lens[b:], max(qlens[b:]), scale, PA.PREFILL_CONFIG, out[b:]
    )
    ref = PA.paged_prefill_attention_reference(q, k, v, t, qlens, ctxs, scale)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    check(f"mixed heads={heads} page={page} decode={decode_ctxs} prefill={prefill_shapes} {dtype}", out, ref, tol, tol)


def fp8_attention_case(heads, page, decode_ctxs, prefill_shapes):
    """fp32 queries over an fp8 cache (fp32 decode: the interpreter's bf16 dot is broken)."""
    qh, kvh, d = heads
    qlens = [1] * len(decode_ctxs) + [q for q, _ in prefill_shapes]
    ctxs = list(decode_ctxs) + [c for _, c in prefill_shapes]
    nb = sum((c + page - 1) // page for c in ctxs) + 2
    k, ks, v, vs = fp8_cache(nb, page, kvh, d)
    t = tables(ctxs, page, nb)
    q = torch.randn(sum(qlens), qh, d)
    lens = torch.tensor(ctxs, dtype=torch.int32)
    b = len(decode_ctxs)
    scale = 1 / math.sqrt(d)
    scales = dict(key_scales=ks, value_scales=vs)
    out = torch.full_like(q, float("nan"))
    for splits in (1, 4):
        PA._paged_decode_attention_triton(
            q[:b], k, v, t[:b], lens[:b], scale, splits, PA.decode_config(b), out[:b], ks, vs
        )
        ref = PA.paged_decode_attention_reference(q[:b], k, v, t[:b], lens[:b], scale, **scales)
        check(f"fp8 decode heads={heads} page={page} ctx={decode_ctxs} S={splits}", out[:b], ref, 1e-4, 1e-4)
    cu = torch.tensor([0] + qlens[b:]).cumsum(0).to(torch.int32)
    PA._paged_prefill_attention_triton(
        q[b:], k, v, t[b:], cu, lens[b:], max(qlens[b:]), scale, PA.PREFILL_CONFIG, out[b:], ks, vs
    )
    ref = PA.paged_prefill_attention_reference(q[b:], k, v, t[b:], qlens[b:], ctxs[b:], scale, **scales)
    check(f"fp8 prefill heads={heads} page={page} shapes={prefill_shapes}", out[b:], ref, 1e-4, 1e-4)


def fp8_rope_case(heads, n, dtype):
    qh, kvh, d = heads
    inv = K.rotary.compute_inv_freq(d, 500000.0)
    cos_t, sin_t = K.rotary.rope_cos_sin(inv, 256, dtype)
    pos = torch.randint(0, 256, (n,))
    cos, sin = cos_t[pos], sin_t[pos]
    qkv = torch.randn(n, (qh + 2 * kvh) * d).to(dtype)
    nb, page = 8, 4
    slots = torch.randperm(nb * page)[:n]
    k = torch.zeros(nb + 1, page, kvh, d, dtype=torch.float8_e4m3fn)
    v = torch.zeros_like(k)
    ks, vs = torch.zeros(nb + 1, page, kvh), torch.zeros(nb + 1, page, kvh)
    q = RC._rope_and_cache_write_triton(qkv, cos, sin, k, v, slots, qh, kvh, ks, vs)
    # Expected: RoPE in fp32 rounded once to the model dtype (what the kernel
    # quantizes), then quantized. The bf16 reference function rotates op by
    # op in bf16, whose cancellation errors are ~1% of a head's largest value.
    kf, vf = torch.zeros(nb + 1, page, kvh, d), torch.zeros(nb + 1, page, kvh, d)
    qf = RC.rope_and_cache_write_reference(qkv.float(), cos.float(), sin.float(), kf, vf, slots, qh, kvh)
    want_k, want_ks = F8.fp8_quantize_reference(kf.view(-1, kvh, d)[slots].to(dtype))
    want_v, want_vs = F8.fp8_quantize_reference(vf.view(-1, kvh, d)[slots].to(dtype))
    rtol = 1e-6 if dtype == torch.float32 else 2.0**-7  # q rounded to dtype: within an ulp
    check(f"fp8 rope_cache q and scales heads={heads} n={n} {dtype}",
          (q, ks.view(-1, kvh)[slots], vs.view(-1, kvh)[slots]), (qf, want_ks, want_vs), 1e-6, rtol)
    # An fp32 ulp of difference (FMA contraction) can move a code by one:
    # same sign and magnitude codes at most 1 apart (codes order by magnitude).
    worst = 0
    for got, want in ((k, want_k), (v, want_v)):
        a = got.view(torch.uint8).view(-1, kvh, d)[slots].int()
        b = want.view(torch.uint8).int()
        distance = ((a & 0x7F) - (b & 0x7F)).abs()
        sign_flip = ((a ^ b) & 0x80).bool() & (((a & 0x7F) > 1) | ((b & 0x7F) > 1))
        worst = max(worst, int(distance.max()), 99 * int(sign_flip.any()))
    results.append((f"fp8 rope_cache codes heads={heads} n={n} {dtype}", worst <= 1, worst))
    print(f"{'OK  ' if worst <= 1 else 'FAIL'} fp8 rope_cache codes heads={heads} n={n} {dtype}: "
          f"max code distance {worst}", flush=True)


def block_case(heads, page, q_max, shapes, splits, dtype, causal, varlen=True, config=None, fp8=False):
    """Block attention: shapes are (q_len, context_len) per sequence, the
    block's tokens counted in the context. Padding query rows hold NaN."""
    qh, kvh, d = heads
    qlens = [q for q, _ in shapes]
    ctxs = [c for _, c in shapes]
    nb = sum((c + page - 1) // page for c in ctxs) + 2
    if fp8:
        k, ks, v, vs = fp8_cache(nb, page, kvh, d)
    else:
        (k, v), ks, vs = cache(nb, page, kvh, d, dtype), None, None
    t = tables(ctxs, page, nb)
    q = torch.randn(len(shapes) * q_max, qh, d).to(dtype)
    padding = torch.tensor([j >= n for n in qlens for j in range(q_max)])
    q[padding] = float("nan")
    lens = torch.tensor(ctxs, dtype=torch.int32)
    q_lens = torch.tensor(qlens, dtype=torch.int32) if varlen else None
    scale = 1 / math.sqrt(d)
    out = BA._paged_block_attention_triton(
        q, k, v, t, lens, q_lens, q_max, scale, causal, splits,
        config or BA.block_config(len(shapes)), None, ks, vs,
    )
    ref = BA.paged_block_attention_reference(q, k, v, t, lens, q_lens, q_max, scale, causal, ks, vs)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    if fp8:  # values up to ~30 and scores of tens: fp32 rounding of a score alone moves an output by 1e-4
        tol = 5e-4
    name = (f"block{' fp8' if fp8 else ''} heads={heads} page={page} Q={q_max} shapes={shapes} S={splits} "
            f"causal={causal} varlen={varlen} {dtype} {config or ''}")
    check(name, out, ref, tol, tol)
    if not bool((out[padding] == 0).all()):
        results.append((name + " padding rows", False, float("nan")))
        print(f"FAIL {name}: padding rows are not zero", flush=True)


def norm_rope_case(heads, n, dtype, with_query):
    qh, kvh, d = heads
    inv = K.rotary.compute_inv_freq(d, 10000.0)
    cos_t, sin_t = K.rotary.rope_cos_sin(inv, 256, dtype)
    pos = torch.randint(0, 256, (n,))
    cos, sin = cos_t[pos], sin_t[pos]
    query = (torch.randn(n, qh * d) * 3).to(dtype) if with_query else None
    kv = (torch.randn(n, 2 * kvh * d) * 3).to(dtype)
    q_norm = (1 + 0.1 * torch.randn(d)).to(dtype)
    k_norm = (1 + 0.1 * torch.randn(d)).to(dtype)
    nb, page = 8, 4
    slots = torch.randperm(nb * page)[:n]
    slots[-1] = nb * page  # null block
    outs = []
    for fn in (RC._norm_rope_and_cache_write_triton, RC.norm_rope_and_cache_write_reference):
        torch.manual_seed(1)
        k, v = cache(nb, page, kvh, d, dtype)
        rotated = fn(query, kv, q_norm, k_norm, 1e-6, cos, sin, k, v, slots, qh, kvh)
        assert (rotated is None) == (not with_query)
        outs.append((k, v) if rotated is None else (rotated, k, v))
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    check(f"norm_rope_cache heads={heads} n={n} query={with_query} {dtype}", outs[0], outs[1], tol, tol)


def fp8_norm_rope_case(heads, n):
    """fp32 inputs into an fp8 cache: codes at most one apart from the reference's."""
    qh, kvh, d = heads
    inv = K.rotary.compute_inv_freq(d, 10000.0)
    cos_t, sin_t = K.rotary.rope_cos_sin(inv, 256, torch.float32)
    pos = torch.randint(0, 256, (n,))
    cos, sin = cos_t[pos], sin_t[pos]
    query = torch.randn(n, qh * d) * 3
    kv = torch.randn(n, 2 * kvh * d) * 3
    q_norm, k_norm = 1 + 0.1 * torch.randn(d), 1 + 0.1 * torch.randn(d)
    nb, page = 8, 4
    slots = torch.randperm(nb * page)[:n]
    got = []
    for fn in (RC._norm_rope_and_cache_write_triton, RC.norm_rope_and_cache_write_reference):
        k = torch.zeros(nb + 1, page, kvh, d, dtype=torch.float8_e4m3fn)
        v = torch.zeros_like(k)
        ks, vs = torch.zeros(nb + 1, page, kvh), torch.zeros(nb + 1, page, kvh)
        rotated = fn(query, kv, q_norm, k_norm, 1e-6, cos, sin, k, v, slots, qh, kvh, ks, vs)
        got.append((rotated, ks.view(-1, kvh)[slots], vs.view(-1, kvh)[slots], k, v))
    check(f"fp8 norm_rope_cache q and scales heads={heads} n={n}", got[0][:3], got[1][:3], 1e-5, 1e-5)
    worst = 0
    for mine, want in zip(got[0][3:], got[1][3:]):
        a = mine.view(torch.uint8).view(-1, kvh, d)[slots].int()
        b = want.view(torch.uint8).view(-1, kvh, d)[slots].int()
        distance = ((a & 0x7F) - (b & 0x7F)).abs()
        sign_flip = ((a ^ b) & 0x80).bool() & (((a & 0x7F) > 1) | ((b & 0x7F) > 1))
        worst = max(worst, int(distance.max()), 99 * int(sign_flip.any()))
    results.append((f"fp8 norm_rope_cache codes heads={heads} n={n}", worst <= 1, worst))
    print(f"{'OK  ' if worst <= 1 else 'FAIL'} fp8 norm_rope_cache codes heads={heads} n={n}: "
          f"max code distance {worst}", flush=True)


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
    for _, config in PA.DECODE_CONFIGS:
        run("decode config", lambda: decode_case(heads, 16, [1, 129, 300, 0], 2, torch.float32, config))
    run("prefill", lambda: prefill_case(heads, 4, [(5, 5), (3, 10), (1, 7), (70, 70)], torch.float32))
    run("prefill fp16", lambda: prefill_case(heads, 16, [(20, 40), (65, 65)], torch.float16))
    run("mixed", lambda: mixed_case(heads, 4, [9, 33], [(5, 12), (20, 20)], torch.float32))
    run("mixed fp16", lambda: mixed_case(heads, 16, [1, 70, 130], [(40, 40)], torch.float16))
    for n in (1, 5):
        run("rope", lambda: rope_case(heads, n, torch.float32))
    run("rope bf16", lambda: rope_case(heads, 3, torch.bfloat16))
    for page in (4, 16):
        run("fp8 attention", lambda: fp8_attention_case(heads, page, [1, 15, 70, 0], [(5, 12), (20, 20), (70, 70)]))
    run("fp8 rope", lambda: fp8_rope_case(heads, 5, torch.float32))
    run("fp8 rope bf16", lambda: fp8_rope_case(heads, 5, torch.bfloat16))
    # Block attention: q_len from 0 to q_max, a block that is the whole
    # context, contexts ending inside, at and across pages and tiles.
    block_shapes = [(5, 5), (5, 13), (3, 10), (0, 6), (1, 8), (4, 16), (5, 70), (2, 150)]
    for causal in (True, False):
        for page in (4, 16):
            for splits in (1, 4):
                run("block", lambda: block_case(heads, page, 5, block_shapes, splits, torch.float32, causal))
        run("block full", lambda: block_case(
            heads, 16, 10, [(10, 10), (10, 37), (10, 300)], 4, torch.float32, causal, varlen=False))
        run("block Q=3", lambda: block_case(heads, 16, 3, [(3, 3), (2, 40), (3, 129)], 2, torch.float32, causal))
        run("block fp16", lambda: block_case(heads, 16, 10, [(10, 20), (7, 130)], 2, torch.float16, causal))
        for config in [c for _, c in BA.BLOCK_CONFIGS] + [
            {"block_n": 64, "num_warps": 8, "num_stages": 1, "block_r": 16},
            {"block_n": 32, "num_warps": 4, "num_stages": 2, "block_r": 32},
        ]:
            for splits in (1, 2, 8):
                run("block config", lambda: block_case(
                    heads, 16, 10, [(10, 10), (4, 129), (10, 300), (0, 0)], splits, torch.float32, causal,
                    config=config))
        for splits in (1, 4):
            run("block fp8", lambda: block_case(
                heads, 16, 5, block_shapes, splits, torch.float32, causal, fp8=True))
    for with_query in (True, False):
        for n in (1, 5):
            run("norm rope", lambda: norm_rope_case(heads, n, torch.float32, with_query))
        run("norm rope bf16", lambda: norm_rope_case(heads, 3, torch.bfloat16, with_query))
    run("fp8 norm rope", lambda: fp8_norm_rope_case(heads, 5))
for dtype in (torch.float32, torch.bfloat16):
    run("norm", lambda: norm_case(3, 256, dtype))
    run("silu", lambda: silu_case(3, 200, dtype))
for split in (1, 2):
    run("gemv", lambda: gemv_case(3, 100, 512, split))
run("gemv ragged", lambda: gemv_case(1, 70, 300, 2))

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
