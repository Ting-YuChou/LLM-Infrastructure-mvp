#!/usr/bin/env python3
"""Kernel micro-benchmarks for Phase 1.

Each kernel registers a shape sweep, an input builder, implementations, a
reference and a bytes (memory-bound) or FLOPs (compute-bound) formula; see
Kernel and the registrations at the bottom. For every shape:
  1. each implementation runs once on fresh inputs and is checked against the
     reference (bf16 tolerance unless the kernel sets its own);
  2. it is timed by replaying a CUDA graph of many calls, so CPU launch
     overhead stays out of the number. Inputs under 2x the L2 (40 MB on an
     A100) are rotated through copies adding up to ~512 MiB, as
     phase0/bandwidth.py does, so the kernel reads HBM rather than L2;
  3. us/call becomes GB/s (or TFLOPs), % of peak (1,555 GB/s and 312 TFLOPs on
     an A100 40GB) and % of the copy bandwidth measured on this GPU.
Writes <out>/kbench.json and <out>/kbench.md; exits 1 if any check failed or,
on the GPU, any implementation could not run.

  python kbench.py --out results/kbench                 # every kernel
  python kbench.py --kernels rms_norm,linear --out d     # a subset
  python kbench.py --list                                # name<TAB>ncu --kernel-name regex
  python kbench.py --ncu-mode --kernels rms_norm         # a few plain calls per shape and
                                                         # impl in cudaProfilerStart/Stop
  python kbench.py --device cpu --out d                  # checks only, no timing (no GPU here)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "phase0"))

from bandwidth import SHAPES as LINEAR_SHAPES  # noqa: E402
from bandwidth import copy_bandwidth  # noqa: E402
# By module: kernels/__init__.py re-exports silu_mul the function over the submodule.
from src.serving.nano_vllm.kernels.rmsnorm import TRITON_AVAILABLE  # noqa: E402
from src.serving.nano_vllm.kernels.linear import linear, load_cuda_extension  # noqa: E402
from src.serving.nano_vllm.kernels.paged_attention import (  # noqa: E402
    paged_decode_attention,
    paged_decode_attention_reference,
    paged_prefill_attention,
    paged_prefill_attention_reference,
)
from src.serving.nano_vllm.kernels.rmsnorm import (  # noqa: E402
    fused_add_rms_norm,
    fused_add_rms_norm_reference,
    rms_norm,
    rms_norm_reference,
)
from src.serving.nano_vllm.kernels.rope_cache import (  # noqa: E402
    rope_and_cache_write,
    rope_and_cache_write_reference,
)
from src.serving.nano_vllm.kernels.silu_mul import (  # noqa: E402
    silu_and_mul,
    silu_and_mul_reference,
    silu_mul,
    silu_mul_reference,
)
from src.serving.nano_vllm.kernels.w4a16 import (  # noqa: E402
    W4A16_GEMV_MAX_ROWS,
    W4A16_MMA_MAX_ROWS,
    W4A16Weight,
    dequantize_w4a16,
    quantize_w4a16,
    unpack_w4a16,
    w4a16_linear,
    w4a16_linear_reference,
)

BF16 = torch.bfloat16
PEAK_TFLOPS = 312.0  # A100 dense bf16
ROTATE_BYTES = 512 * 1024**2  # working set for small inputs, as in bandwidth.py
MAX_COPIES = 4096  # tiny inputs: 4096 copies already outgrow L2
MIN_CALLS = 100  # calls per captured graph
REPLAYS = 20


class Unavailable(Exception):
    """An implementation cannot run here (no Triton, no GPU)."""


class NotApplicable(Unavailable):
    """An implementation is left out of this shape on purpose (too slow to time,
    outside its row limit); unlike Unavailable it does not fail a GPU run."""


@dataclass
class Kernel:
    """One op to benchmark.

    shapes     the sweep; each shape is a dict of named sizes (a str value is a label).
    inputs     (shape, device) -> dict of the tensors (and scalars) of one call.
    impls      name -> factory(inputs) -> zero-arg callable returning the output
               (a tensor or a tuple). The factory does per-input setup (output
               buffers, repacked weights) outside the timed region and raises
               Unavailable when it cannot run here. The first impl is the baseline.
    reference  inputs -> expected output, compared in fp32.
    bytes      shape -> bytes one call must move (memory-bound kernels), or
    flops      shape -> FLOPs of one call (compute-bound kernels).
    ncu        regex for ncu --kernel-name: the kernels the ncu step profiles.
    """

    name: str
    shapes: list[dict]
    inputs: Callable[[dict, str], dict]
    impls: dict[str, Callable[[dict], Callable[[], object]]]
    reference: Callable[[dict], object]
    bytes: Callable[[dict], float] | None = None
    flops: Callable[[dict], float] | None = None
    ncu: str = "."
    atol: float = 2e-2
    rtol: float = 2e-2


KERNELS: dict[str, Kernel] = {}


def register(kernel: Kernel) -> None:
    if (kernel.bytes is None) == (kernel.flops is None):
        raise ValueError(f"{kernel.name}: give exactly one of bytes and flops")
    KERNELS[kernel.name] = kernel


def op_impl(op: Callable[[dict], object]):
    """Factory for an implementation that is just a call on the inputs."""
    return lambda inputs: lambda: op(inputs)


def triton_impl(op: Callable[[dict], object]):
    """Like op_impl, but Unavailable without Triton or off the GPU, where the
    nano_vllm wrappers would silently fall back to their torch reference."""

    def factory(inputs: dict):
        if not TRITON_AVAILABLE or not any(t.is_cuda for t in tensors(inputs)):
            raise Unavailable("needs Triton and a GPU")
        return lambda: op(inputs)

    return factory


def tensors(inputs: dict) -> list[torch.Tensor]:
    return [v for v in inputs.values() if torch.is_tensor(v)]


def as_tuple(out) -> tuple:
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def label(shape: dict) -> str:
    return " ".join(v if isinstance(v, str) else f"{k}={v}" for k, v in shape.items())


def make_inputs(kernel: Kernel, shape: dict, device: str, seed: int | None = None) -> dict:
    if seed is not None:  # the reference and every impl's check see the same values
        torch.manual_seed(seed)
    return kernel.inputs(shape, device)


def compare(outs: tuple, expected: tuple, atol: float, rtol: float) -> dict:
    if len(outs) != len(expected):
        return {"ok": False, "error": f"{len(outs)} outputs, reference has {len(expected)}"}
    ok, worst = True, 0.0
    for out, exp in zip(outs, expected):
        if tuple(out.shape) != tuple(exp.shape):
            return {"ok": False, "error": f"shape {tuple(out.shape)}, reference {tuple(exp.shape)}"}
        out, exp = out.float(), exp.float().to(out.device)
        err = (out - exp).abs()
        ok &= bool((err <= atol + rtol * exp.abs()).all())  # NaN fails
        worst = max(worst, float(err.max()) if err.numel() else 0.0)
    return {"ok": ok, "max_abs_err": worst if math.isfinite(worst) else str(worst)}


def time_us(fns: list[Callable[[], object]]) -> tuple[float, str, int]:
    """us per call: replay a CUDA graph of `calls` calls cycling through fns.
    Falls back to a plain launch loop when an impl cannot be captured."""
    calls = len(fns) * math.ceil(MIN_CALLS / len(fns))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up (Triton JIT, cuBLAS workspace) off the capture stream
        for fn in fns:
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    def loop():
        for c in range(calls):
            fns[c % len(fns)]()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            loop()
        run, mode = graph.replay, "graph"
    except Exception as e:  # e.g. a host sync in the impl
        torch.cuda.synchronize()
        run, mode = loop, f"eager ({type(e).__name__}: {str(e).splitlines()[0][:80]})"
    run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(REPLAYS):
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / calls)
    del graph
    return statistics.median(times), mode, calls


def copies_for(input_bytes: int, l2_bytes: int) -> int:
    if input_bytes >= 2 * l2_bytes:
        return 1
    return min(MAX_COPIES, math.ceil(ROTATE_BYTES / max(input_bytes, 1)))


def bench_kernel(kernel: Kernel, device: str, env: dict) -> list[dict]:
    rows = []
    baseline = next(iter(kernel.impls))
    for shape in kernel.shapes:
        try:
            expected = as_tuple(kernel.reference(make_inputs(kernel, shape, device, seed=0)))
        except Exception as e:
            traceback.print_exc()
            expected = e
        base_us = None
        for impl, factory in kernel.impls.items():
            row = {"kernel": kernel.name, "impl": impl, "shape": shape, "label": label(shape)}
            inputs = fns = None
            try:
                if isinstance(expected, Exception):
                    raise RuntimeError(f"reference failed: {expected}")
                inputs = make_inputs(kernel, shape, device, seed=0)
                row.update(compare(as_tuple(factory(inputs)()), expected, kernel.atol, kernel.rtol))
                input_bytes = sum(t.nbytes for t in tensors(inputs))
                inputs = None
                if device == "cuda" and row["ok"]:
                    copies = copies_for(input_bytes, env["l2_bytes"])
                    fns = [factory(make_inputs(kernel, shape, device)) for _ in range(copies)]
                    us, mode, calls = time_us(fns)
                    row.update(us=us, timing=mode, copies=copies, calls=calls)
                    if kernel.bytes:
                        gbps = kernel.bytes(shape) / (us * 1e-6) / 1e9
                        row.update(
                            GBps=gbps,
                            pct_peak=100 * gbps / env["peak_GBps"],
                            pct_copy=100 * gbps / env["copy_GBps"],
                        )
                    else:
                        tflops = kernel.flops(shape) / (us * 1e-6) / 1e12
                        row.update(TFLOPs=tflops, pct_peak=100 * tflops / env["peak_TFLOPs"])
                    if impl == baseline:
                        base_us = us
                    if base_us:
                        row["x_base"] = base_us / us
            except Unavailable as e:
                row["skipped"] = str(e)
                if isinstance(e, NotApplicable):
                    row["by_design"] = True
            except Exception as e:
                traceback.print_exc()
                row.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
            inputs = fns = None  # free this impl's copies before the next one allocates
            if device == "cuda":
                torch.cuda.empty_cache()
            print("KBENCH " + json.dumps(row), flush=True)
            rows.append(row)
    return rows


def ncu_mode(kernels: list[Kernel], reps: int) -> int:
    """A few plain calls of every impl per shape, each batch between
    cudaProfilerStart/Stop in an NVTX range kernel/impl/shape, for
    ncu --profile-from-start off. Inputs, warm-up and references stay outside."""
    failed = 0
    for kernel in kernels:
        for shape in kernel.shapes:
            for impl, factory in kernel.impls.items():
                name = f"{kernel.name}/{impl}/{label(shape)}"
                fn = None
                try:
                    fn = factory(make_inputs(kernel, shape, "cuda"))
                    fn()  # Triton JIT, cuBLAS heuristics
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                    with torch.cuda.nvtx.range(name):
                        for _ in range(reps):
                            fn()
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    print(f"KBENCH_NCU {name} x{reps}", flush=True)
                except Exception as e:
                    failed += 1
                    print(f"KBENCH_NCU {name} failed: {type(e).__name__}: {e}", flush=True)
                fn = None
                torch.cuda.empty_cache()
    return failed


def fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def markdown(rows: list[dict], env: dict) -> str:
    lines = [f"# kbench on {env['device']}", ""]
    if env["device_type"] == "cuda":
        lines += [
            f"Peak {env['peak_GBps']:,.0f} GB/s HBM and {env['peak_TFLOPs']:.0f} TFLOPs bf16; "
            f"measured copy {env['copy_GBps']:,.0f} GB/s. Time per call from CUDA-graph replay, "
            "inputs rotated past L2. x base: speed relative to the first impl.",
            "",
        ]
    else:
        lines += ["CPU: correctness checks only, no timing.", ""]
    for name in dict.fromkeys(r["kernel"] for r in rows):
        kernel = KERNELS[name]
        bound = "GB/s" if kernel.bytes else "TFLOPs"
        lines += [
            f"## {name}",
            "",
            f"| shape | impl | check | us | {bound} | % peak |"
            + (" % copy |" if kernel.bytes else "")
            + " x base |",
            "|---|---|---|---:|---:|---:|" + ("---:|" if kernel.bytes else "") + "---:|",
        ]
        for r in (r for r in rows if r["kernel"] == name):
            if "skipped" in r:
                check = f"skipped: {r['skipped']}"
            elif r.get("error"):
                check = f"error: {r['error']}"
            else:
                err = r.get("max_abs_err")
                err = f"{err:.1e}" if isinstance(err, float) else err
                check = f"{'ok' if r['ok'] else 'FAIL'} ({err})"
            if r.get("timing", "graph") != "graph":
                check += f", {r['timing']}"
            cells = [
                r["label"], r["impl"], check, fmt(r.get("us"), ".1f"),
                fmt(r.get("GBps") if kernel.bytes else r.get("TFLOPs"), ",.0f" if kernel.bytes else ".1f"),
                fmt(r.get("pct_peak"), ".0f"),
            ]
            if kernel.bytes:
                cells.append(fmt(r.get("pct_copy"), ".0f"))
            cells.append(fmt(r.get("x_base"), ".2f"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernels", default="", help="comma-separated names (default: all)")
    parser.add_argument("--out", type=Path, help="directory for kbench.json and kbench.md")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--list", action="store_true", help="print name<TAB>ncu regex and exit")
    parser.add_argument("--ncu-mode", action="store_true", help="plain calls for ncu, no timing")
    parser.add_argument("--ncu-reps", type=int, default=3)
    parser.add_argument("--bandwidth-json", type=Path, help="phase0 bandwidth.json for copy_GBps")
    parser.add_argument("--peak-gbps", type=float, help="default: 2,039 on an 80GB A100, else 1,555")
    args = parser.parse_args()

    names = [n for n in args.kernels.split(",") if n] or list(KERNELS)
    unknown = [n for n in names if n not in KERNELS]
    if unknown:
        parser.error(f"unknown kernels {unknown}; known: {', '.join(KERNELS)}")
    kernels = [KERNELS[n] for n in names]
    if args.list:
        for kernel in kernels:
            print(f"{kernel.name}\t{kernel.ncu}")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("no CUDA device; use --device cpu to run the checks only")
    if args.ncu_mode:
        if args.device != "cuda":
            parser.error("--ncu-mode needs the GPU")
        sys.exit(1 if ncu_mode(kernels, args.ncu_reps) else 0)
    if args.out is None:
        parser.error("--out is required")

    env = {"device_type": args.device, "torch": torch.__version__}
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        env.update(
            device=props.name,
            l2_bytes=getattr(props, "L2_cache_size", 40 * 1024**2),
            peak_GBps=args.peak_gbps or (2039.0 if "80GB" in props.name else 1555.0),
            peak_TFLOPs=PEAK_TFLOPS,
            copy_GBps=(
                json.loads(args.bandwidth_json.read_text())["copy_GBps"]
                if args.bandwidth_json
                else copy_bandwidth()
            ),
        )
        torch.cuda.empty_cache()
    else:
        env["device"] = "cpu"

    rows = []
    for kernel in kernels:
        rows += bench_kernel(kernel, args.device, env)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "kbench.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    text = markdown(rows, env)
    (args.out / "kbench.md").write_text(text + "\n")
    print(text)
    # On the GPU every impl must run, like the tests step's no-skip rule.
    failed = [
        r
        for r in rows
        if ("skipped" in r and args.device == "cuda" and not r.get("by_design"))
        or not r.get("ok", True)
    ]
    for r in failed:
        print(f"FAILED {r['kernel']} {r['impl']} {r['label']}: {r.get('error') or r.get('skipped') or 'mismatch'}")
    sys.exit(1 if failed else 0)


# --- registrations -----------------------------------------------------------
# Llama 3.1 8B: hidden 4096, intermediate 14336, rms_norm_eps 1e-5.
HIDDEN, INTER, EPS = 4096, 14336, 1e-5
TOKENS = [1, 16, 64, 256, 2048]  # decode batches and one prefill chunk


def randn(*shape: int, device: str, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=BF16).mul_(scale)


register(
    Kernel(
        name="rms_norm",
        shapes=[{"M": m, "N": HIDDEN} for m in TOKENS],
        inputs=lambda s, dev: {
            "x": randn(s["M"], s["N"], device=dev),
            "w": randn(s["N"], device=dev, scale=0.1).add_(1),
        },
        impls={
            "torch": op_impl(lambda i: rms_norm_reference(i["x"], i["w"], EPS)),
            "triton": triton_impl(lambda i: rms_norm(i["x"], i["w"], EPS, use_triton=True)),
        },
        reference=lambda i: rms_norm_reference(i["x"], i["w"], EPS),
        bytes=lambda s: 2 * (2 * s["M"] * s["N"] + s["N"]),  # read x and w, write y
        ncu="_rms_norm_kernel",
    )
)

register(
    Kernel(
        name="silu_mul",
        shapes=[{"M": m, "N": INTER} for m in TOKENS],
        inputs=lambda s, dev: {
            "gate": randn(s["M"], s["N"], device=dev),
            "up": randn(s["M"], s["N"], device=dev),
        },
        impls={
            "torch": op_impl(lambda i: silu_mul_reference(i["gate"], i["up"])),
            "triton": triton_impl(lambda i: silu_mul(i["gate"], i["up"], use_triton=True)),
        },
        reference=lambda i: silu_mul_reference(i["gate"], i["up"]),
        bytes=lambda s: 2 * 3 * s["M"] * s["N"],  # read gate and up, write out
        ncu="_silu_mul_kernel",
    )
)

# The GEMV baseline: cuBLAS through F.linear at the fused decode shapes of
# phase0/bandwidth.py. Weights scaled by 1/sqrt(K) keep outputs O(1), so the
# bf16 tolerance means the same thing for every K.
register(
    Kernel(
        name="linear",
        shapes=[
            {"layer": layer, "M": m, "N": n, "K": k}
            for layer, (n, k) in LINEAR_SHAPES.items()
            for m in (1, 2, 4, 8, 16)
        ],
        inputs=lambda s, dev: {
            "x": randn(s["M"], s["K"], device=dev),
            "w": randn(s["N"], s["K"], device=dev, scale=s["K"] ** -0.5),
        },
        impls={
            "cublas": op_impl(lambda i: F.linear(i["x"], i["w"])),
            "triton": triton_impl(lambda i: linear(i["x"], i["w"], backend="triton")),
            "cuda": lambda i: _cuda_gemv(i),
        },
        reference=lambda i: i["x"].float() @ i["w"].float().t(),
        bytes=lambda s: 2 * (s["N"] * s["K"] + s["M"] * s["K"] + s["M"] * s["N"]),
    )
)

def _cuda_gemv(i: dict):
    if not i["x"].is_cuda:
        raise Unavailable("needs a GPU")
    if i["x"].shape[0] > 8:
        raise NotApplicable("the CUDA GEMV takes up to 8 rows")
    extension = load_cuda_extension()
    if extension is None:
        raise RuntimeError("building kernels/csrc/gemv_bf16.cu failed (see the log)")
    return lambda: extension.gemv_bf16(i["x"], i["w"])


register(
    Kernel(
        name="fused_add_rms_norm",
        shapes=[{"M": m, "N": HIDDEN} for m in TOKENS],
        inputs=lambda s, dev: {
            "x": randn(s["M"], s["N"], device=dev),
            "residual": randn(s["M"], s["N"], device=dev, scale=4.0),
            "w": randn(s["N"], device=dev, scale=0.1).add_(1),
        },
        impls={
            "torch": op_impl(
                lambda i: fused_add_rms_norm_reference(i["x"], i["residual"], i["w"], EPS)
            ),
            "triton": triton_impl(
                lambda i: fused_add_rms_norm(i["x"], i["residual"], i["w"], EPS, use_triton=True)
            ),
        },
        reference=lambda i: fused_add_rms_norm_reference(i["x"], i["residual"], i["w"], EPS),
        # read x, residual and w; write the normed output and the new residual
        bytes=lambda s: 2 * (4 * s["M"] * s["N"] + s["N"]),
        ncu="_fused_add_rms_norm_kernel",
    )
)

register(
    Kernel(
        name="silu_and_mul",
        shapes=[{"M": m, "N": INTER} for m in TOKENS],
        inputs=lambda s, dev: {"gate_up": randn(s["M"], 2 * s["N"], device=dev)},
        impls={
            "torch": op_impl(lambda i: silu_and_mul_reference(i["gate_up"])),
            "triton": triton_impl(lambda i: silu_and_mul(i["gate_up"], use_triton=True)),
        },
        reference=lambda i: silu_and_mul_reference(i["gate_up"]),
        bytes=lambda s: 2 * 3 * s["M"] * s["N"],  # read gate and up, write out
        ncu="_silu_and_mul_kernel",
    )
)

# --- attention and the KV cache (Llama 3.1 8B: 32 q heads, 8 kv heads, 128) ---
Q_HEADS, KV_HEADS, HEAD_DIM, PAGE = 32, 8, 128, 16
SCALE = HEAD_DIM**-0.5


def _rope_inputs(s: dict, dev: str) -> dict:
    n = s["tokens"]
    num_blocks = max(64, 2 * n // PAGE + 1)
    shape = (num_blocks + 1, PAGE, KV_HEADS, HEAD_DIM)
    slots = torch.randperm(num_blocks * PAGE, device=dev)[:n]
    # Real cos/sin rows: both halves equal, which the kernel relies on (it
    # reads the first half only, like rope_cos_sin's tables).
    angles = torch.rand(n, HEAD_DIM // 2, device=dev) * 1000
    angles = torch.cat((angles, angles), dim=-1)
    return {
        "qkv": randn(n, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, device=dev),
        "cos": angles.cos().to(BF16),
        "sin": angles.sin().to(BF16),
        "k_cache": torch.zeros(shape, dtype=BF16, device=dev),
        "v_cache": torch.zeros(shape, dtype=BF16, device=dev),
        "slots": slots,
    }


def _rope_call(fn, **kwargs):
    def call(i):
        q = fn(i["qkv"], i["cos"], i["sin"], i["k_cache"], i["v_cache"], i["slots"],
               Q_HEADS, KV_HEADS, **kwargs)
        flat = KV_HEADS * HEAD_DIM
        # The cache writes are outputs too: compare the written rows.
        return q, i["k_cache"].view(-1, flat)[i["slots"]], i["v_cache"].view(-1, flat)[i["slots"]]
    return call


register(
    Kernel(
        name="rope_cache_write",
        shapes=[{"tokens": n} for n in TOKENS],
        inputs=_rope_inputs,
        impls={
            "torch": op_impl(_rope_call(rope_and_cache_write_reference)),
            "triton": triton_impl(_rope_call(rope_and_cache_write, use_triton=True)),
        },
        reference=_rope_call(rope_and_cache_write_reference),
        # read qkv, cos, sin, slots; write q, k, v
        bytes=lambda s: s["tokens"] * (
            2 * (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM + 2 * 2 * HEAD_DIM + 8
            + 2 * (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
        ),
        ncu="_rope_cache_write_kernel",
    )
)


def _paged_inputs(batch: int, context: int, query_lens: list[int], dev: str) -> dict:
    """Uniform contexts over shuffled blocks, as after allocation churn."""
    blocks_per_seq = (context + PAGE - 1) // PAGE
    num_blocks = batch * blocks_per_seq
    shape = (num_blocks + 1, PAGE, KV_HEADS, HEAD_DIM)
    order = torch.randperm(num_blocks, device=dev, dtype=torch.int32)
    tables = order.view(batch, blocks_per_seq)
    return {
        "q": randn(sum(query_lens), Q_HEADS, HEAD_DIM, device=dev),
        "k_cache": randn(*shape, device=dev),
        "v_cache": randn(*shape, device=dev),
        "tables": tables,
        "context_lens": torch.full((batch,), context, dtype=torch.int32, device=dev),
        "cu_seqlens": torch.tensor([0] + query_lens, device=dev).cumsum(0).to(torch.int32),
        "query_lens": query_lens,
        "context": context,
    }


def _vllm_flash_attn(i: dict, max_query_len: int):
    """vLLM's FA2 fork, called the way vLLM's FlashAttention backend calls it."""
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except Exception as e:  # not installed / built for another GPU
        raise Unavailable(f"vllm_flash_attn: {type(e).__name__}: {e}")
    if not i["q"].is_cuda:
        raise Unavailable("needs a GPU")
    kwargs = dict(
        q=i["q"], k=i["k_cache"], v=i["v_cache"],
        max_seqlen_q=max_query_len, cu_seqlens_q=i["cu_seqlens"],
        max_seqlen_k=i["context"], seqused_k=i["context_lens"],
        softmax_scale=SCALE, causal=True, block_table=i["tables"],
    )
    try:
        flash_attn_varlen_func(**kwargs, fa_version=2)
        return lambda: flash_attn_varlen_func(**kwargs, fa_version=2)
    except TypeError:  # older signature without fa_version
        flash_attn_varlen_func(**kwargs)
        return lambda: flash_attn_varlen_func(**kwargs)


def _small_only(op, too_big: Callable[[dict], bool]):
    """The fp32 PyTorch reference as an impl, skipped where timing it would take
    minutes (it still checks every shape as the reference)."""

    def factory(inputs: dict):
        if too_big(inputs):
            raise NotApplicable("reference too slow to time at this size")
        return lambda: op(inputs)

    return factory


def _decode_bytes(s: dict) -> float:
    return 2 * 2 * s["B"] * s["ctx"] * KV_HEADS * HEAD_DIM  # read K and V once


register(
    Kernel(
        name="paged_decode_attention",
        shapes=[{"B": b, "ctx": c} for b in (1, 16, 64, 128) for c in (128, 1024, 4096)],
        inputs=lambda s, dev: _paged_inputs(s["B"], s["ctx"], [1] * s["B"], dev),
        impls={
            "vllm_fa2": lambda i: _vllm_flash_attn(i, 1),
            "torch": _small_only(
                lambda i: paged_decode_attention_reference(
                    i["q"], i["k_cache"], i["v_cache"], i["tables"], i["context_lens"], SCALE),
                lambda i: i["q"].shape[0] * i["context"] > 64 * 1024,
            ),
            "triton": triton_impl(lambda i: paged_decode_attention(
                i["q"], i["k_cache"], i["v_cache"], i["tables"], i["context_lens"], SCALE,
                use_triton=True)),
        },
        reference=lambda i: paged_decode_attention_reference(
            i["q"], i["k_cache"], i["v_cache"], i["tables"], i["context_lens"], SCALE),
        bytes=_decode_bytes,
        ncu="_paged_decode|flash_fwd",
    )
)


def _prefill_flops(s: dict) -> float:
    q = s["qlen"]  # causal: half of the q x k score matrix, QK^T and PV
    return s["seqs"] * 2 * 2 * (q * (q + 1) / 2) * HEAD_DIM * Q_HEADS


register(
    Kernel(
        name="paged_prefill_attention",
        shapes=[
            {"seqs": n, "qlen": q}
            for n, q in ((1, 512), (1, 1024), (1, 4096), (4, 1024), (16, 256))
        ],
        inputs=lambda s, dev: _paged_inputs(s["seqs"], s["qlen"], [s["qlen"]] * s["seqs"], dev),
        impls={
            "vllm_fa2": lambda i: _vllm_flash_attn(i, max(i["query_lens"])),
            "torch": _small_only(
                lambda i: paged_prefill_attention_reference(
                    i["q"], i["k_cache"], i["v_cache"], i["tables"], i["query_lens"],
                    [i["context"]] * len(i["query_lens"]), SCALE),
                lambda i: len(i["query_lens"]) * i["context"] ** 2 > 2 * 1024**2,
            ),
            "triton": triton_impl(lambda i: paged_prefill_attention(
                i["q"], i["k_cache"], i["v_cache"], i["tables"], i["cu_seqlens"],
                i["context_lens"], i["query_lens"], [i["context"]] * len(i["query_lens"]),
                SCALE, use_triton=True)),
        },
        reference=lambda i: paged_prefill_attention_reference(
            i["q"], i["k_cache"], i["v_cache"], i["tables"], i["query_lens"],
            [i["context"]] * len(i["query_lens"]), SCALE),
        flops=_prefill_flops,
        ncu="_paged_prefill|flash_fwd",
    )
)


# W4A16 (M7). Every impl multiplies by the same 4-bit codes: each factory
# quantizes w with quantize_w4a16 (setup, outside the timed region), and
# vLLM's Marlin gets those codes repacked into its layout. GB/s counts the
# 4-bit weight, its scales, x and y, so the cublas_bf16 row (F.linear on the
# dequantized bf16 weight) reads as the bandwidth W4A16 would need to match it.
def _w4a16_cuda(kernel: str, max_rows: int):
    def factory(i: dict):
        if not i["x"].is_cuda:
            raise Unavailable("needs a GPU")
        if i["x"].shape[0] > max_rows:
            raise NotApplicable(f"the {kernel} kernel takes up to {max_rows} rows")
        if load_cuda_extension("w4a16") is None:
            raise RuntimeError("building kernels/csrc/w4a16.cu failed (see the log)")
        weight = quantize_w4a16(i["w"])
        return lambda: w4a16_linear(i["x"], weight, config={"kernel": kernel})

    return factory


def _marlin_weight(weight: W4A16Weight, device: torch.device):
    """Our codes and scales in the layout of vLLM's gptq_marlin (uint4b8, no act order)."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        get_weight_perm,
        marlin_weights,
    )

    n, k = weight.shape
    codes = unpack_w4a16(weight.qweight).t().contiguous().to(torch.int32)  # [K, N], q + 8
    packed = marlin_weights(codes, k, n, 4, get_weight_perm(4, False))
    scales = marlin_utils.marlin_permute_scales(
        weight.scales.t().contiguous(), k, n, weight.group_size
    )
    return packed.to(device), scales.to(device)


def _vllm_marlin(i: dict):
    if not i["x"].is_cuda:
        raise Unavailable("needs a GPU")
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.scalar_type import scalar_types

    weight = quantize_w4a16(i["w"])
    n, k = weight.shape
    packed, scales = _marlin_weight(weight, i["x"].device)
    workspace = marlin_utils.marlin_make_workspace_new(i["x"].device)
    empty = marlin_utils.marlin_make_empty_g_idx(i["x"].device)  # no zero points, no act order
    return lambda: marlin_utils.apply_gptq_marlin_linear(
        i["x"], packed, scales, empty, empty, empty, workspace, scalar_types.uint4b8,
        n, k, is_k_full=True,
    )


def _cublas_dequantized(i: dict):
    weight = dequantize_w4a16(quantize_w4a16(i["w"]))
    return lambda: F.linear(i["x"], weight)


register(
    Kernel(
        name="w4a16_linear",
        shapes=[
            {"layer": layer, "M": m, "N": n, "K": k}
            for layer, (n, k) in LINEAR_SHAPES.items()
            if layer != "lm_head"  # stays bf16 under weight_quant="w4a16"
            for m in (1, 4, 8, 16, 32, 64)
        ],
        inputs=lambda s, dev: {
            "x": randn(s["M"], s["K"], device=dev),
            "w": randn(s["N"], s["K"], device=dev, scale=s["K"] ** -0.5),
        },
        impls={
            "vllm_marlin": _vllm_marlin,
            "cuda_gemv": _w4a16_cuda("gemv", W4A16_GEMV_MAX_ROWS),
            "cuda_mma": _w4a16_cuda("mma", W4A16_MMA_MAX_ROWS),
            "cuda_wide": _w4a16_cuda("wide", W4A16_MMA_MAX_ROWS),
            "cuda_auto": _w4a16_cuda("auto", W4A16_MMA_MAX_ROWS),
            "cublas_bf16": _cublas_dequantized,
        },
        reference=lambda i: w4a16_linear_reference(i["x"], quantize_w4a16(i["w"])),
        bytes=lambda s: s["N"] * s["K"] // 2 + 2 * s["N"] * s["K"] // 128
        + 2 * s["M"] * (s["K"] + s["N"]),
        ncu="gemv_w4a16|gemm_w4a16|Marlin|marlin",
    )
)


if __name__ == "__main__":
    main()
