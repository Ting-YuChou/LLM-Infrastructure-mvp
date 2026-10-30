# Phase 1: Kernels

Triton/CUDA kernels for nano_vllm, targeting decode of Llama 3.1 8B Instruct (BF16) on one A100. Done in this order:

1. paged decode attention
2. fused RoPE + KV write
3. decode CUDA graph
4. fused add + RMSNorm
5. prefill attention
6. bf16 GEMV (Triton first, then rewritten in CUDA)
7. W4A16

Each milestone runs `tests` → `kbench` → `ncu` → `e2e` on a Colab A100, and the numbers are compared against the [Phase 0](../phase0/README.md) baseline. Failures and fixes are recorded in [LOG.md](LOG.md).

## How to run

```bash
BENCH_PHASE=phase1 bash benchmarks/phase0/colab_phase0.sh
```

Starts a Colab A100, uploads the code, runs every step, downloads the results to `results/<run_id>/`, and shuts the VM down. The summary is in `results/<run_id>/summary.md`.

```bash
BENCH_PHASE=phase1 P1_STEPS="tests kbench" COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh rerun
```

Development loop: runs only the tests and kernel benchmarks on the same VM. The venv, the model and the CUDA toolkit stay on the VM, and `COLAB_KEEP=1` keeps the VM up after the run. Without `P1_STEPS`, `rerun` runs `tests kbench ncu e2e quality summarize`.

- `P1_KERNELS=rms_norm,linear`: kbench and ncu run only these kernels.
- `P1_TUNE=decode,w4a16`: the `tune` step sweeps only these kernels (default all: decode, prefill, gemv, cuda_gemv, w4a16).
- `P1_NANO_CONFIGS`: the nano configs e2e runs, each `label:attention:linear:eager|graph[:weight_quant]`. The default is six configs that stack the milestones in order (torch_eager, triton_eager, triton_graph, gemv_graph, cuda_gemv_graph, w4a16_graph).
- When rerunning with the same `P0_RUN_ID`, labels that ncu and nsys have already profiled are skipped.
- Remember to run `bash benchmarks/phase0/colab_phase0.sh stop` when done.

```bash
python benchmarks/phase1/kbench.py --device cpu --out /tmp/kbench
```

Locally: only correctness checks of the torch implementations; Triton implementations are marked skipped. Takes about 80 seconds. macOS has no Triton wheel and no GPU, so Triton kernels and timing only run on Colab.

## Steps (`run_phase1.sh`)

Each step writes its own `logs/<step>.log` and records its result in `status.tsv`; a failed step does not block the steps after it.

| step | what it does |
|---|---|
| `installs` | vLLM 0.19.1 venv (shared with Phase 0; reused if already on the VM) plus pytest, ninja; nsys; `unsloth/Meta-Llama-3.1-8B-Instruct` |
| `cuda129` | apt-installs the CUDA 12 nvcc, cudart-dev and cccl matching the venv's torch, and writes `/content/p0/cuda129.env` (`CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=8.0`, `TORCH_EXTENSIONS_DIR`), which every later step sources. The system nvcc is 13.0 and cannot build the extensions. Not needed before M6 |
| `tests` | `pytest -q -rs tests/gpu tests/unit/test_nano_vllm_*.py`; any `SKIPPED` line counts as a failure (same rule as the CI training job). If `tests/gpu` does not exist yet, only the unit tests run |
| `kbench` | `kbench.py`: every shape of every kernel is checked against the reference, then timed; writes `kbench/kbench.{json,md}` |
| `ncu` | one ncu capture per kbench kernel (`--kernel-name regex:<registered regex>`, same sections as Phase 0), summarized with Phase 0's `analyze_ncu.py`. `P1_NCU_SHAPES` picks other shapes, `P1_NCU_SECTIONS` adds sections (`SourceCounters` also writes a per-instruction source page) |
| `trace` | `tiled_trace.py`: trace build of `gemm_w4a16_tiled` that breaks down each block's time on cold weights (prologue, each tile, slice end, the last block to finish); writes `trace/trace.{json,md}` and a Gantt chart. Not in the default steps |
| `golden` | `golden.py`: the first time, saves the tiled kernel's output on fixed inputs to `P1_GOLDEN`; later runs compare bit for bit. Not in the default steps |
| `tune` | `tune.py`: sweeps launch configs for each kernel and shape, next to the config the code currently picks; writes `tune/tune.{json,md}`. Not in the default steps |
| `e2e` | one `nano_bench.py` run per `P1_NANO_CONFIGS` entry (`static:1:128:256,paged:1:128:256,paged:16:1024:256,paged:64:1024:256`); `vllm_offline.py` runs the same paged points; nsys captures paged b=1, 16, 64 of the last nano config. When the last config is W4A16, vLLM's AWQ is also captured with nsys, and `gemm_breakdown.py` reads the GEMM time of each projection in both sides' decode |
| `quality` | `quality.py`: on the real 8B, teacher forcing over 2048-token blocks of wikitext-2 test, comparing perplexity and top-1 agreement of the torch path, the Triton kernels, and Triton + W4A16 |
| `summarize` | collects everything into `summary.md` |

## kbench

Each kernel is registered with `register(Kernel(...))`:

| field | content |
|---|---|
| `shapes` | shape sweep; each shape is a dict |
| `inputs(shape, device)` | builds the inputs for one call |
| `impls` | `{name: factory(inputs) -> callable with no arguments}`. The factory does setup outside the timing (allocating outputs, repacking weights). The first implementation is the baseline |
| `reference(inputs)` | the correct answer, compared in fp32 |
| `bytes` or `flops` | bytes moved per call (memory-bound) or FLOPs computed (compute-bound), one or the other |
| `ncu` | `--kernel-name` regex for the ncu step |

Kernels registered so far:

- `rms_norm`, `silu_mul`: the existing Triton versions in `kernels/` against torch, M = 1, 16, 64, 256, 2048.
- `linear`: cuBLAS `F.linear`, Triton GEMV, CUDA GEMV, on the five decode shapes (qkv 6144×4096, o 4096×4096, gate_up 28672×4096, down 4096×14336, lm_head 128256×4096), M = 1, 2, 4, 8, 16.
- `fused_add_rms_norm`, `silu_and_mul`, `rope_cache_write`: the M4 and M2 fused kernels against torch.
- `paged_decode_attention`: vLLM's FA2 (`flash_attn_varlen_func` with block_table), torch reference, Triton split-KV, B = 1, 16, 64, 128, ctx = 128, 1k, 4k.
- `paged_prefill_attention`: vLLM's FA2, torch reference, Triton, in TFLOPs.
- `w4a16_linear`: vLLM's `gptq_marlin`, our CUDA GEMV and tensor-core versions, and cuBLAS on the dequantized bf16 weights; four decoder projections, M = 1–64. All implementations use the same 4-bit codes (Marlin's weights are repacked from our codes with vLLM's helper), so the reference is the same. GB/s is always computed from the 4-bit weight bytes, so the cuBLAS row reads as "how much bandwidth W4A16 needs to match bf16".

**Timing**:

- Each implementation is first checked against the reference with bf16 tolerance (atol = rtol = 2e-2), and only timed if it passes.
- Timing uses CUDA graph replay with at least 100 calls captured in one graph, so launch overhead is not counted.
- When the inputs are smaller than twice L2 (40 MB), several copies are rotated as in `bandwidth.py`, bringing the working set to about 512 MiB, so HBM is measured rather than L2.
- Implementations that cannot be captured (for example with a host sync) fall back to a plain launch loop and are marked `eager` in the table.

**Scoring**:

- memory-bound: GB/s, fraction of the 1,555 GB/s peak, and fraction of this card's measured copy bandwidth. Copy bandwidth is measured once per run with `bandwidth.copy_bandwidth()`; Phase 0 measured 1,380 GB/s.
- compute-bound: TFLOPs and fraction of 312 TFLOPs.

`--ncu-mode` does no timing and no correctness check. Each shape and implementation is warmed up once, then called 3 times between `cudaProfilerStart/Stop`, wrapped in an NVTX range `kernel/impl/shape`.

## Results

All numbers come from a Colab A100-SXM4-40GB with Llama 3.1 8B Instruct. The process (what each run measured, what failed, how it was fixed) is in [LOG.md](LOG.md).

### End-to-end decode

ms is the time per decode step. Inputs are 128 tokens for static and paged b=1, 1024 for b=16 and b=64; outputs are 256 throughout. The nano configs stack one milestone at a time, so each row means "everything up to this milestone".

| config | milestone | run | paged b=1 | paged b=16 | paged b=64 | kernels/step (b=1) |
|---|---|---|---:|---:|---:|---:|
| Phase 0 baseline | - | Phase 0 | 53.63 | 618.92 | - | 1,604 |
| torch_eager | M1 engine refactor | run 3 | 39.60 | 51.12 | 152.52 | |
| triton_eager | + M2 decode attention, RoPE/KV, M4 fusion, M5 prefill | run 3 | 21.56 | 23.00 | 30.17 | |
| triton_graph | + M3 decode CUDA graph | run 3 | 12.61 | 14.87 | 23.48 | |
| gemv_graph | + M6 Triton GEMV | run 4 | 12.32 | **14.63** | 23.77 | 424 |
| cuda_gemv_graph | M6 with CUDA GEMV (1–2 rows) | run 4 | **12.15** | 14.87 | 23.51 | 328 |
| **vLLM 0.19.1 bf16** | reference | run 2–6 | 12.63 | 14.83 | 23.04 | 365 (Phase 0) |
| w4a16_graph | + M7 W4A16 | run 6 | 5.95 | 8.91 | 24.20 | 328 |
| **vLLM 0.19.1 AWQ int4 (awq_marlin)** | reference for M7 | run 6 | 5.55 | 7.87 | 16.23 | |

- **bf16**: nano ties vLLM. 4% faster at b=1, 1% faster at b=16, 2% slower at b=64. e2e run-to-run variation is about 1% (LOG.md, run 4).
- **M2 target** (b=16 < 60 ms in eager): 23.0 ms, met. The M1 refactor alone went from 619 to 51 ms.
- **M3 target** (b=1 ≤ 13.5 ms): 12.6 ms, met.
- **M4 target** (about 330 kernels per step): 328 on the CUDA GEMV path, met; Triton GEMV adds 96 split-K reduce kernels, 424 in total.
- **M6 target** (b=1 about 11 ms): 12.15 ms, not met. The plan assumed "GEMV is much faster than cuBLAS", but cuBLAS under graph replay already reaches 76–94% of peak (Phase 0's 63–73% from CUDA events included launch gaps), so GEMV had only 0–8% to win.
- **M7**: at b=1 W4A16 is 2.0× faster than bf16 and 7% behind vLLM's AWQ Marlin; 13% behind at b=16; 49% behind at b=64 (see the kernel table below).
- The static path deliberately keeps Phase 0's code (per-layer PyTorch attention, F.linear), so it stays around 40 ms and is not listed.
- Prefill time is not compared with vLLM: `vllm_offline.py`'s `prefill_s` does not cover the whole prefill (LOG.md, run 2).

### Kernels

kbench, CUDA graph replay, inputs rotated past L2. The table is from run 5, with every default being the final tuned version.

| kernel | reference | target | measured |
|---|---|---|---|
| paged decode attention | vLLM FA2 split-KV | ≥ FA2 at every (B, ctx); ≥ 80% of peak at large B·ctx | 11 of 12 points ≥ FA2 (1.03–1.44×); B=1/ctx=128 is only 0.52× (only one of S=32 splits has work, plus the reduce kernel, about 0.2 ms more per token). 86–93% of peak at large B·ctx |
| prefill attention | `flash_attn_varlen_func` | ≥ 70% of FA2 TFLOPs at 1k–4k | 1k–4k: 0.80–0.89× (1×4096 is 144 TFLOPs, 46% of peak); 0.68× at 512 |
| bf16 GEMV (M=1) | cuBLAS | ≥ cuBLAS on every shape; o_proj, qkv ≥ 80% of peak | Triton: gate_up 1.06×, down 1.05×, lm_head 1.01×, qkv 0.99×, o_proj 0.97×. CUDA: 1.08×, 1.06×, 1.01×, 1.00×, 0.97×. o_proj 73–74%, qkv 78–79% of peak, both just short |
| W4A16 | vLLM `gptq_marlin` (same 4-bit codes) | weight bandwidth ≥ 80% of peak, within 10% of Marlin | M=1: o_proj 1.33×, qkv 1.06×, gate_up 0.99×, down 0.64×. M=4–16: 0.59–1.07×. M=32–64: 0.48–0.70× (run 6). down_proj and large batches not met: ncu shows our kernel is bound by memory latency (long_scoreboard), with only one set of weights in flight per thread, DRAM 41–52%; Marlin uses a multi-stage `cp.async` pipeline and reaches 62% (LOG.md, run 7) |
| fused add + RMSNorm, silu_and_mul, RoPE + KV write | torch | - | 2.2–7.5× faster than torch. At 256 and 2048 tokens the first two reach 83–94% of peak, RoPE + KV write 42–62% |

### Quality (`quality.py`, wikitext-2 test, 288,627 tokens)

| config | weights GiB | perplexity | top-1 agreement (vs the bf16 torch path) |
|---|---:|---:|---:|
| bf16, PyTorch path | 15.0 | 7.233 | - |
| bf16, Triton kernels | 15.0 | 7.232 | 98.81% |
| W4A16 (RTN, symmetric, groups of 128) | 5.3 | 8.024 | 85.90% |

The plan required bf16 top-1 agreement ≥ 99%; measured 98.81%. Perplexity matches to the third decimal; the difference is only in attention (Triton, like FA2, casts the softmax probabilities to bf16 before multiplying by V), and the disagreements are all near-tie tokens. W4A16's RTN raises perplexity by 11%, the same order of magnitude as published results for Llama 3 8B.

## After Phase 2: Kernel list, round 1 (runs `kernels-r1` to `kernels-r4`, `kernels-serve`)

At the end of Phase 1 the roadmap doc's "Kernel list" had 12 items. Re-evaluated after Phase 2, this round did 5 of them; the rest moved to the next round or were dropped. Branch `claude/inference-kernels`, local only.

| # | item | result |
|---|---|---|
| 10 | async scheduling as the default | Done. Also saves 0.23 / 0.42 / 1.01 ms for W4A16 (b=1/16/64) |
| 12 | sync guard extended to sampled, eager, mixed | Done. No syncs caught on the GPU |
| 5 | measuring the sampling path | Measured. At b=64 nano's sampling costs 0.96 ms more than vLLM's. Round 5's sort-free sampler brings it to 0.13 ms against vLLM's 1.74 |
| 1 | W4A16 GEMV register prefetch | Negative result, reverted. The deeper the prefetch, the slower |
| 3 | FP8 KV cache | Done, opt-in (`kv_cache_dtype: fp8`). 1.7 ms faster at b=64, perplexity +0.2% |
| 2, 6, 7 | Marlin-style W4A16 and its epilogue/prologue fusion | Next round. W4A16 is still 44% behind AWQ at b=64 (#2 is done in round 2, see the next section) |
| 4, 8, 9, 11 | reduce folded into the split kernel, prefill tile order, bf16 CUDA GEMV, CUDA event timing | Dropped: gain too small, TTFT already beats vLLM, not on the default path, not needed after async |

### End-to-end decode (ms/step, b=1 / 16 / 64; b=16 at 4096 tokens)

| config | b=1 | b=16 | b=64 | b=16, 4096 |
|---|---:|---:|---:|---:|
| nano bf16 (async scheduling, now the default) | 12.05 | 14.13 | 22.60 | 18.73 |
| nano bf16 + fp8 KV | **12.01** | **13.81** | **20.87** | 17.16 |
| vLLM bf16 (FA2) | 12.60 | 14.82 | 23.04 | - |
| vLLM bf16 + fp8 KV (FLASHINFER) | 13.05 | 14.50 | 21.30 | - |
| nano W4A16 | 5.68 | 8.48 | 23.27 | 13.01 |
| nano W4A16 + fp8 KV | **5.65** | 8.25 | 21.52 | 11.60 |
| vLLM AWQ (Marlin) | 5.56 | 7.86 | 16.19 | - |
| vLLM AWQ + fp8 KV (FLASHINFER) | 6.05 | **7.55** | **14.44** | - |

Run `kernels-r4` (same VM, same session). vLLM's 4096 points failed (`vllm_offline.py` hard-coded `max_model_len` to 4096; fixed).

### FP8 KV cache

- **How** (`kernels/fp8.py`): Triton 3.6's CUDA backend does not accept `float8e4nv` on sm80, and the A100 has no fp8 conversion instructions, so the kernels treat the cache as uint8 and handle e4m3 with integer arithmetic:
  - **decode**: the 7 exponent and mantissa bits go into bits 4–10 of a bf16, giving "the value × 2^-120", which goes straight into `tl.dot`.
  - **scale**: one fp32 scale per (token, kv head) (amax / 448). The K scale multiplies the scores and the V scale multiplies the probabilities; there is no element-wise dequantization. q is multiplied by 2^60 first and the output by 2^60 at the end, so intermediates stay far from fp32 subnormals.
  - **capacity**: 132 bytes per 128 values against 256 for bf16; automatic KV sizing gives about 1.9× the blocks.
- **vLLM 0.19.1 fp8 KV on the A100**: FLASH_ATTN rejects it; TRITON_ATTN hits the same `fp8e4nv` limit and fails to compile; only FLASHINFER runs, but at b=1 it is 0.45 ms slower than vLLM's own FA2.
- **kernel** (decode attention, µs, bf16 / fp8; fp8 uses its own tuned launch tiers):

  | | B=1 | B=16 | B=64 | B=128 |
  |---|---:|---:|---:|---:|
  | ctx=1024 | 14.4 / 12.0 | 62.3 / 53.9 | 201.0 / 147.3 | 379.9 / 273.2 |
  | ctx=4096 | 20.9 / 19.8 | 204.6 / 156.1 | 770.2 / 541.6 | 1484.9 / 1033.8 |

  Nearly half the bytes, 4–30% less time. fp8 reaches only 60–69% of HBM peak (bf16 86–93%); the integer decode is the new limit.
- **quality** (wikitext-2 test, 288,627 tokens): bf16 perplexity goes from 7.232 to 7.248 (+0.22%), top-1 agreement against bf16 is 97.55%; W4A16 goes from 8.024 to 8.040.
- **online** (run `kernels-serve`, Phase 2's harness, each server sizes its own KV at 0.9 GPU memory):
  - **capacity**: nano from 157,744 to 305,920 tokens (1.94×), vLLM from 159,072 to 326,352.
  - **W2b**: 48 prefixes of 4096 tokens, more in total than the bf16 capacity. fp8 raises the hit rate on both sides from about 48% to 70.6%. nano's req/s goes from 3.33 to 3.99 and TTFT p50 from 405 to 160 ms; vLLM fp8 is 4.16 req/s, 259 ms.
  - **W1 c=64**: ITL p50 is nano fp8 21.00, vLLM fp8 21.38, nano 22.65, vLLM 23.24 ms.
- **Gate G4**: perplexity increase ≤ 1%, passed; top-1 agreement ≥ 98%, 0.45 points short (between bf16 Triton and the torch path, numerical differences alone already give only 98.81%); 2 ms saved per step at b=64, measured 1.73 ms, also just short. Both misses are small, so it stays opt-in and the default stays `auto`, as in vLLM.

### Negative result: GEMV register prefetch

The hypothesis was "each warp issues only one load per iteration, so latency is exposed". Implemented as a register ring that issues the loads 1, 2 or 4 iterations ahead; every depth was slower: down_proj M=1 at depth 0 / 1 / 2 / 4 was 37.7 / 39.2 / 39.4 / 48.3 µs, and e2e W4A16 b=1 at depth 0 / 2 / 4 was 5.74 / 6.00 / 6.32 ms. The ring costs registers and lowers occupancy; depth 0 already has 30% achieved occupancy, so latency was already hidden by other warps. Reverted in e53207f. Marlin is fast not just because of prefetch but because of the persistent grid, the multi-stage `cp.async` pipeline and striped partitioning working together, which needs a full rewrite (next round).

### Sampling path

Every benchmark so far was greedy, but the server defaults to temperature 0.7, top-p 0.9:

| sampled minus greedy (ms/step) | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano | 0.42 | 1.02 | 2.62 |
| vLLM | 0.25 | 1.33 | 1.66 |

nsys (b=16, per step): the innermost scan of `torch.cumsum` along the 128k vocab takes 231 µs, radix sort about 265 µs, two argmaxes 96 µs together (the greedy bf16 argmax is computed every step, with one block per row). Online W1s (c=1/16/64): nano 78 / 855 / 1,711 tok/s, vLLM 75 / 609 / 1,693; ITL p50 at c=64 is 25.31 against 25.03 ms, nano 1.1% slower. Round 5 replaces this path with a sort-free sampler (below).

## Kernel list, round 2: Marlin-style W4A16 (runs `marlin-t1` to `marlin-t2`)

At the end of round 1, W4A16 at b=64 was still 44% slower than vLLM's AWQ (23.27 against 16.19 ms), and kbench showed the gap was almost all in the GEMM. This round wrote our own kernel with Marlin's design, `gemm_w4a16_tiled` (`kernels/csrc/w4a16.cu`), without using vLLM's code; the default layout became tiled. Branch is again `claude/inference-kernels`, local only.

### Design

- **tiled layout** (`kernels/w4a16.py`, repacked at load time):
  - qweight is int32 `[K/16, N/64, 32, 4]`. Every 512 bytes is the B operand of one 16 × 64 tile for mma.m16n8k16; one 16-byte shared load gives a lane the b0/b1 of 8 n8 tiles.
  - scales are laid out as `[K/128, N/64, 8, 8]`, with one lane's 8 columns of scales together.
  - Stored as 4-D, so the old kernel's shape check rejects it.
- **pipeline**:
  - 4 cp.async stages, each holding the x tile (read with ldmatrix after swizzle), the weight tile and the scales;
  - the warps of one n64 column share that stage's k16 rows;
  - after dequantization the scale is applied first (rounding only once), then the mma.
- **persistent striped grid** (Marlin's partitioning):
  - all (column slice, k tile) pairs are numbered in order and each block takes a contiguous range, balanced for any shape;
  - the loader addresses by global tile number, so prefetching across a slice boundary does not drain the pipeline (Marlin drains and refills).
- **cross-block sum**:
  - the blocks sharing a slice each write their fp32 partial sums to a slot, then atomicAdd the slice's counter;
  - the last block to arrive sums in block order and writes the result;
  - the result is deterministic and no block waits for another (Marlin uses a serial lock chain).
  - Synchronous W4A16 no longer needs `splits=1` to be reproducible per token.
- **over 64 rows**: one launch per 64 rows, up to `tiled_max_rows` (128); beyond that, the tiled dequantization kernel plus F.linear. Tiled dequantization gives bitwise the same result as rows, at the same speed (78–80% of peak).

### Catching up with Marlin

| run | change | four-layer sum vs Marlin (M=1 / 16 / 64) |
|---|---|---|
| t1 | first version | 1.21 / 1.15 / 1.11 |
| t1c | incremental addresses, fragment double buffer (across k16 steps and across stages), 256x128 tile | 1.03 / 0.98 / 1.11 |
| t1d | 64x128 tile for MT≥3, slot-sum fences reduced to one | 1.02 / 0.98 / 1.04 |
| t1e | dequantization's AND and OR merged into one `lop3` | **0.96 / 0.96 / 0.98** |

- **ncu (t1b)**:
  - M=1 is instruction-bound, not bandwidth-bound: for the same work the first version issued about 45% more instructions;
  - going from base clock to boost clock, Marlin got 43% faster and we got only 6% faster. (Round 3 correction: ncu was always `--clock-control none`, i.e. boost clock; the 43% is the difference between ncu's cold weights, with the cache flushed each time, and kbench's warm weights left in L2; see round 3.)
- **nvdisasm (t1d)**:
  - nvcc compiles `(w >> 4s) & 0x000F000F | magic` into two LOP3s, 5 instructions per pair of codes;
  - with inline `lop3.b32 …, 0xEA` it is 4;
  - this alone took the M=1 sum from 121 to 115 µs.
- **tune (t1c)**:
  - 256x128 for M ≤ 32 (16 KB of weights per barrier);
  - 64x128 for M=64, and 64x256 when N ≥ 16384 (gate_up);
  - 2 blocks per SM is always slower.

### kernel (kbench, run `marlin-t2-20261023-104627-colab`, µs, tiled / vLLM `gptq_marlin`, same 4-bit codes)

| M | qkv | o_proj | gate_up | down | sum | sum, old kernel |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 17.5 / 19.1 | 15.8 / 17.4 | 52.4 / 59.1 | 28.8 / 23.3 | **114.5** / 118.9 | 126.3 |
| 16 | 18.2 / 21.2 | 16.1 / 19.6 | 62.6 / 65.3 | 31.0 / 27.5 | **127.9** / 133.6 | 149.9 |
| 32 | 23.0 / 27.5 | 20.6 / 22.2 | 72.1 / 72.1 | 41.4 / 37.4 | **157.1** / 159.2 | 258.6 |
| 64 | 40.1 / 45.2 | 33.5 / 35.9 | 105.5 / 113.6 | 73.1 / 63.1 | **252.2** / 257.8 | 449.9 |

down_proj (N=4096, K=14336) is still 1.1–1.3× Marlin at every M; the other three layers match or beat it. (Round 3 correction: kbench measured down_proj with a single copy of the weights, which stayed in L2; cold, down beats Marlin at every M; see round 3.)

### End-to-end decode (run `marlin-t2-20261023-104627-colab`, ms/step)

| config | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16 tiled (new default) | **5.35** | **7.31** | 16.82 |
| nano W4A16 rows (old kernel) | 5.70 | 8.29 | 23.03 |
| vLLM AWQ (awq_marlin) | 5.56 | 7.85 | **16.23** |
| nano bf16 | 12.06 | 14.12 | 22.61 |
| vLLM bf16 | 12.61 | 14.84 | 23.07 |

- b=1 and b=16 are 4% and 7% faster than vLLM AWQ; b=64 went from 42% slower to 4% slower.
- The remaining 0.59 ms at b=64 is not in the GEMMs kbench measures: four-layer sum 252 against 258 µs. Subtracting cuBLAS time from the bf16 e2e, vLLM's AWQ GEMM takes about 224 µs per layer in e2e, less than kbench's `gptq_marlin`; the cause was not investigated.
- nsys (b=1 / b=16): 0.04 ms of GPU idle, 330 kernels per step, GEMM 84% / 66%.

### Gates

- **G1 correctness**: passed.
  - 629 GPU tests passed, no skips, including tiled scheduling, fp16, CUDA graph, and the model, graph and async tests run under tiled;
  - repeated runs of the same config are bitwise identical;
  - tiled and rows dequantization are bitwise identical, so prefill and `quality.py` (teacher forcing goes through dequantization) perplexity is unchanged; quality was not rerun this round.
- **G2 kernel**:
  - four-layer sum ≤ 1.05× Marlin at M=1, 16, 64, passed;
  - no single point slower than 1.25×: down_proj M=8 is 1.29×, not passed.
- **G3 e2e**: b=64 ≤ 17.5, b=16 ≤ 8.1, b=1 ≤ 5.68 ms, all three passed, so the default layout became tiled.

## Kernel list, round 3: down_proj (runs `down-d1` to `down-d3`)

At the end of round 2 kbench showed down_proj at 1.10–1.29× Marlin at every M. This round looked for the cause first, and it was mostly a measurement problem: with cold weights down was already faster than Marlin at M ≤ 32, and the real gap was only at M=64 (b=64 decode). After the fixes, W4A16 decode is faster than vLLM's AWQ at b=1, 16 and 64. Branch is again `claude/inference-kernels`, local only.

### Measurement fixes

- **kbench kept down_proj's weights in L2**: `copies_for` picked the number of rotated weight copies from the size of the input tensors, and the inputs held the bf16 weights (down 117 MB), over 2× L2, so `copies=1` and the 100 calls kept reading the same 29 MB of 4-bit weights, mostly resident in the 40 MB L2.
  - `Kernel.rotate_bytes` now gives the bytes each call actually reads, and W4A16 rotates by the 4-bit weights (18 copies for down, 9 for gate_up).
  - gate_up's 4-bit weights are 59 MB, larger than L2, so in round 2 it was already nearly cold; qkv and o_proj were already rotated; `tune.py` always rotated by the 4-bit size.
  - Warm, Marlin is 43% faster (33.3 → 23.3 µs) while we are only 4–7% faster, so round 2's down gap was almost entirely this.
- **ncu is boost clock**: `ncu_run` has used `--clock-control none` from the start. Round 2's "Marlin is 43% faster going from base clock to boost clock" was actually cold weights versus warm weights (annotated in the previous section and in LOG).
- **Three measurement methods disagree about Marlin**: kbench (cold, back-to-back calls in a CUDA graph) measures Marlin 15–23% slower at M ≥ 16 than AWQ Marlin in decode (down M=64: 81.5 against 66.0 µs), ncu (single call, cache flushed) agrees with decode (68 µs), and our kernel is the same under all three. So comparisons with Marlin use the times in decode: e2e captures both nano and vLLM's AWQ with nsys, and `gemm_breakdown.py` identifies each projection by position (the GEMM after SiLU is down_proj, the three before are gate_up, o_proj, qkv).

### Tools

- **trace build**: `gemm_w4a16_tiled` gets `template <bool kTrace>`; the normal build's SASS is unchanged (compared instruction by instruction with the build before the change). Thread 0 writes `clock64` timestamps to shared memory and writes them back only at the end: entry, first tile arrived, four points at each slice end, exit, plus globaltimer, flags such as last-arriving block, and a finish order taken with an atomic.
- **Observations**:
  - globaltimer only ticks every 1,024 ns, so within a block clock is used, across blocks the finish order.
  - On the same VM, trace runs at M ≥ 8 mostly run at only about 1.08 GHz, and M=1 at 1.3–1.4 GHz; M=1 and M=8 are the same kernel with the same instructions, differing only in how many rows of x are nonzero, which looks like the 400 W power limit. Cross-run comparisons only use rows at the same clock; comparisons between versions use kbench and e2e.
  - On each GEMM's critical path, the prologue (the first tiles arriving from cold DRAM) is about 3–4 µs and the slice end about 4–6 µs; in steady state each tile's time is about its share of DRAM time.

### What was done

| change | commit | result |
|---|---|---|
| stages aligned to 128 bytes | 17a3038 | the static `__shared__ int last_block` pushed dynamic shared memory to address 16, so each weight `cp.async` touched 2× the ideal L2 sectors (ncu SourceCounters: 50% excessive). Moved to the end of the dynamic allocation, down M=1 went from 1.95M to 0.98M sectors, the same as Marlin; kbench four-layer sum −1.2 to −9.2%. Output bitwise identical |
| slice end | 7a0072e | thread 0 reads the counter with `ld.acquire` up front, and if all other blocks have arrived it skips the slot write and the atomic (59–81% of down's last arrivals take this path); otherwise one `atom.acq_rel`; a block's last slice end writes all warps' partial sums at once into an already drained stage. Four-layer sum −0.3 to −2.7%, bitwise identical |
| parallel warp sum | 7e30049, 8537411 | when there is room, the first n−1 warps write their own buffers at once, and the last one adds its own in order. At MT=4 the warp sum went from 2.2 to 1.0 µs, but each tile in the main loop got 5% slower (255 registers), down M=64 72 → 81 µs, so it is kept only for MT 1–2 (about −1%) |
| 128x128 tile for 33–64 rows | de05b0f, eb6b2a0 | each warp computes two k16 rows per stage (64x128 does one), halving the slice end per byte. tune (cold) M=64: qkv 38.2 → 34.7, o_proj 31.6 → 27.9, down 72.0 → 62.4 µs; M=48 down 62.9 → 52.9. gate_up stays at 64x256 |

Two items not done:
- **L2 prefetch of the next GEMM (F5)**: `tiled_trace.py --prefetch-tiles 3` pulls each block's first 3 tiles into L2 before measuring; short GEMMs (qkv, o_proj at M=1) get about 2 µs faster, gate_up and down only 0–1 µs, because during the prologue DRAM is already moving the first 3 tiles of every block. It would need a change to the model's call interface and does not help down, so it was not done.
- **`mma_trans` (F2)**: at boost clock each steady-state tile is no slower than DRAM, so it was not done.

### kernel (kbench, cold, run `down-d3`, µs, tiled / Marlin)

| M | qkv | o_proj | gate_up | down | sum | sum, start of round 3 (`down-d1`) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 16.2 / 18.9 | 14.5 / 17.4 | 51.6 / 59.4 | 30.1 / 33.5 | **112.4** / 129.2 | 116.0 |
| 16 | 16.0 / 21.3 | 15.0 / 19.5 | 54.5 / 65.3 | 31.5 / 36.9 | **117.0** / 143.0 | 125.3 |
| 32 | 19.7 / 27.3 | 18.5 / 21.8 | 65.5 / 71.8 | 38.2 / 48.3 | **141.9** / 169.2 | 157.0 |
| 64 | 34.6 / 44.9 | 27.9 / 35.6 | 101.6 / 111.1 | 62.4 / 81.5 | **226.5** / 273.1 | 253.2 |

(kbench's Marlin is slow at M ≥ 16, see above; for the comparison with Marlin see the next table.)

### GEMM per projection in decode (nsys, run `down-d3`, µs, nano / vLLM AWQ)

| b | qkv | o_proj | gate_up | down | per layer |
|---|---:|---:|---:|---:|---:|
| 1 | 16.6 / 16.4 | 14.8 / 14.6 | 52.3 / 55.6 | **30.4** / 31.0 | **114.2** / 117.6 |
| 16 | 16.4 / 17.8 | 15.2 / 16.3 | 55.0 / 58.7 | **31.7** / 33.3 | **118.3** / 126.0 |
| 64 | 34.6 / 34.8 | 28.3 / 30.4 | 102.0 / 89.6 | **62.0** / 66.0 | 227.0 / 220.8 |

- down_proj at b=1, 16, 64 is 0.98, 0.95, 0.94× vLLM; at the start of round 3 b=64 was 1.09× (71.7 against 65.9, run `down-d2b`).
- At b=64 only gate_up is slower (102.0 against 89.6 µs, about 0.4 ms per step).

### End-to-end decode (run `down-d3`, ms/step)

| config | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16 tiled (round 3) | **5.19** | **7.10** | **15.94** |
| nano W4A16 tiled (round 2, `marlin-t2`) | 5.35 | 7.31 | 16.82 |
| nano W4A16 rows (old kernel) | 5.68 | 8.29 | 22.94 |
| vLLM AWQ (awq_marlin) | 5.55 | 7.85 | 16.19 |
| nano bf16 | 12.04 | 14.12 | 22.57 |
| vLLM bf16 | 12.64 | 14.84 | 23.04 |

W4A16 is 6.5%, 9.5% and 1.5% faster than vLLM AWQ at b=1, 16 and 64; in round 2 b=64 was still 4% slower.

### Gates

- **G0 measurement**: cold down in kbench is 0.85–0.94× Marlin (`down-d1`); in decode b=1/16 is 0.99/0.96× and b=64 is 1.09× (`down-d2b`). Round 2's G2 down single point (1.29× at M=8) is re-judged with the cold numbers: passed.
- **G1 correctness**:
  - 645 GPU tests passed, no skips;
  - after alignment, slice end and parallel sum, golden 16/16 bitwise identical (`down-d2a` to `down-d2d`);
  - the 128x128 default changed the summation order of three shapes at M=64: `down-d3` golden is 13/16 identical; the differences are qkv, o_proj and down at M=64, max difference 0.0078 (1 ulp of bf16 around 1–2), 51–148 elements of each 64×N output; the trace build and the normal build are bitwise identical.
- **G2 kernel (cold)**:
  - down at M=1, 16 ≥ 10% faster than `down-d1`: not passed (−2%, −4%); M=64 is 15% faster;
  - down ≤ 1.05× Marlin at every M: passed (both kbench and decode);
  - four-layer sum no slower than `down-d1` at any M: passed (−3.1 to −10.5%).
- **G3 e2e**: no regression, b=1 5.19 ms ≤ 5.20: passed.

### Not done yet

- gate_up at M=64: 102 against 89.6 µs, the largest remaining gap at b=64.
- Under the power limit the SM clock at M ≥ 8 is about 1.08 GHz (trace); what it is during e2e was not measured.
- Why kbench measures Marlin slower at M ≥ 16 than ncu and decode was not investigated.

## Large batches: b=128, 256 (run `concur-c1`)

After round 3, decode was measured at b=128 and 256. W4A16 ties vLLM AWQ at b=64; past 64 rows the gap grows with batch size, and it all comes from the GEMMs: attention is the same on both sides.

- **How it was run**: nano runs only W4A16 tiled (async), with two kinds of graph:
  - default: decode graph up to 64 rows, piecewise graph for larger batches, attention outside the graph;
  - `fullgraph` (`nano_bench.py --full-graphs`): decode graph up to the largest batch, as in vLLM.
- **context**: b=256 uses 512 prompt tokens (plus 256 output), because 1k of KV does not fit in 40 GB; b=128 runs both lengths.
- **nano's GEMM path**:
  - 65–128 rows: the tiled kernel launches once per 64 rows, each launch reading the whole weight once;
  - over `tiled_max_rows` (128): each projection first dequantizes the whole weight to bf16, then calls cuBLAS.

### End-to-end decode (ms/step, tok/s)

| b / prompt | nano default | nano fullgraph | vLLM AWQ | nano / vLLM | tok/s: nano / vLLM |
|---|---:|---:|---:|---:|---:|
| 64 / 1024 | 16.02 | 16.04 | 16.20 | 0.99 | 3,989 / 3,951 |
| 128 / 1024 | 30.57 | 30.20 | 27.45 | 1.10 | 4,239 / 4,663 |
| 128 / 512 | 24.57 | 24.21 | 21.42 | 1.13 | 5,286 / 5,974 |
| 256 / 512 | 50.44 | 49.99 | 39.04 | 1.28 | 5,121 / 6,558 |

- **graph**: fullgraph is only 0.1–1.6% faster.
- **throughput**: nano actually drops slightly from b=128 to b=256 (5,286 → 5,121 tok/s), while vLLM gains 10%.
- **vLLM bf16 not comparable**: its KV cache holds only 142–144k tokens, so neither b=128/1k (164k) nor b=256/512 (196k) fits and it preempts; these two points are not listed. AWQ has 221–223k, and both points fit.
- **prefill_s not comparable**: vLLM is warmed up with the same prompts first, so the measured run hits the prefix cache.

### GEMM per projection in decode (nsys, µs, nano fullgraph / vLLM AWQ)

| b | qkv | o_proj | gate_up | down | per layer | per step (×32) |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 34.7 / 34.8 | 28.3 / 30.5 | 102.1 / 89.5 | 62.0 / 66.5 | 227.1 / 221.4 | 7.3 / 7.1 ms |
| 128 | 69.1 / 49.8 | 56.7 / 41.6 | 203.7 / 167.1 | 124.7 / 94.7 | 454.2 / 353.1 | 14.5 / 11.3 ms |
| 256 | 110.7 / 80.7 | 78.5 / 59.0 | 556.8 / 326.6 | 262.1 / 166.1 | 1008.0 / 632.4 | 32.3 / 20.2 ms |

(`gemm_breakdown.py` adds consecutive launches of the same projection into one call; over 128 rows the dequantization kernel is counted with the cuBLAS call after it.)

- **Where the gap comes from**:
  - b=128: GEMMs take 3.2 ms/step more, e2e gap 2.75 ms;
  - b=256: GEMMs take 12.0 ms/step more, e2e gap 11.0 ms.
- **Attention is the same**: nano's `_paged_decode_split_kernel` at b=256/512 takes 416 µs per layer, vLLM's `flash_fwd_splitkv_kernel` 419 µs.
- **b=128**: nano is exactly 2× b=64 (two launches, weights read twice); vLLM's Marlin handles all 64-row M blocks in one launch and needs only 1.6×.
- **b=256**:
  - dequantization runs 4 times per layer at 104 µs on average, 418 µs per layer and 13.4 ms per step, all of it extra;
  - cuBLAS itself takes about 590 µs per layer, already faster than Marlin's 632 µs.
  - `down-d2e`'s tune measured 4 tiled launches at M=256 instead: faster for gate_up 416 against 554 µs and down 249 against 258 µs; qkv and o_proj are still faster with dequantization.
- **Still far from the limit**: down at M=256 is 30 GFLOP, 107 µs at the A100's practical ~280 TFLOPS, and reading the weights once takes 21 µs; Marlin takes 166 µs, nano 262 µs.

### What could be done

1. **One launch for 65–256 rows**:
   - how: add an M-block dimension to the tiled kernel's persistent schedule, so blocks of different M blocks process the same weight tile at the same time, and reads after the first come from L2;
   - expected: b=128 goes from 2× down to about Marlin's 1.6×, and b=256 no longer needs dequantization.
   - This is the biggest item.
2. **One weight tile used for 128 rows within a block**: decode the B fragment once, use it twice. MT=4 already uses all 255 registers, so the warps would have to be split into M and K groups.
3. **Stopgap**: over 128 rows, pick the path per projection: tiled for gate_up and down, dequantization for qkv and o_proj. From `down-d2e`'s tune, b=256 saves about 160 µs per layer, about 5 ms per step, but is still slower than vLLM.

Item 1 was done in the next section (so item 3 is not needed); item 2 was not done, because the instruction count per tile is the main cause.

## Kernel list, round 4: large batches (runs `chunk-g0` to `chunk-c7`)

The target is the previous section's b=128 and 256. After three changes, W4A16 decode is 6–9% faster than vLLM's AWQ at every b from 1 to 256: b=256 went from 1.28× to 0.93×, and b=64 also dropped from 15.94 to 15.03 ms. Branch is again `claude/inference-kernels`, local only.

### What was done

| change | commit | result |
|---|---|---|
| one launch for several row blocks | d43d122, c655d29 | over 64 rows, the rows are split into chunks of 16 MT rows (Marlin's `par`): chunk c gets grid / chunks blocks, each striping the whole weight, with its own slots and counters. Block b of every chunk walks the same tiles at the same time, so the weights are read from DRAM once and the other chunks read them from L2. Rows are split evenly (96 rows is two of 48). `tiled_max_rows` went from 128 to 256, so 65–256 rows no longer dequantize; 3 or more chunks use the 64x256 tile. b=256 GEMM per layer 1008 → 723 µs (`chunk-c3`) |
| cheaper load stream | 421dcda | ncu's SourceCounters showed HMMA was only 14% of our executed instructions (Marlin 24%). In the SASS each x tile cp.async was preceded by about 20 instructions of index math: a loop with a runtime bound, signed division, swizzle, 64-bit addresses. Now each thread always copies the same chunk of a few rows a fixed number of rows apart, so the address is computed once; the x and weight pointers advance by one k tile per stage; the next stage's fragments are loaded unconditionally, saving about 20 register copies before the barrier. The main loop for MT 4, 128x128 went from 497 instructions (64 HMMA) to 335. kbench four-layer sum: M=64 −11.5%, M=128 −17.1%, M=256 −21.0% |
| last slice end moved after the main loop | 8bcb99f | the block's last slice end runs after the main loop, so reading the slots four at a time does not crowd the main loop's registers. Doing this inside the loop made MT 3–4 2–4% slower (`chunk-c5`). kbench: M=64 −2.2%, M=128 −1.7% |

None of the three changes the summation order: output up to 64 rows is bitwise identical to before the changes (golden 16/16, saved in `chunk-g0`, checked in `chunk-c2` to `chunk-c7`).

### End-to-end decode (run `chunk-c7`, ms/step)

| b / prompt | nano W4A16 tiled | vLLM AWQ | nano / vLLM | tok/s: nano / vLLM | nano before round 4 |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | **5.24** | 5.55 | 0.94 | 191 / 180 | 5.19 (`down-d3`) |
| 16 / 1024 | **7.11** | 7.85 | 0.91 | 2,249 / 2,039 | 7.10 (`down-d3`) |
| 64 / 1024 | **15.03** | 16.18 | 0.93 | 4,257 / 3,956 | 15.94 (`down-d3`) |
| 128 / 1024 | **25.83** | 27.48 | 0.94 | 4,955 / 4,658 | 30.20 (`concur-c1`) |
| 128 / 512 | **19.86** | 21.43 | 0.93 | 6,447 / 5,973 | 24.21 (`concur-c1`) |
| 256 / 512 | **36.14** | 39.05 | 0.93 | 7,083 / 6,555 | 49.99 (`concur-c1`) |

- nano uses `fullgraph` (decode CUDA graph up to 256 rows), as in vLLM; the default piecewise graph is about 1% slower (`concur-c1`).
- vLLM bf16 preempts at b=128/1k and 256/512, so it is not listed (previous section).

### GEMM per projection in decode (nsys, run `chunk-c7`, µs, nano / vLLM AWQ)

| b | qkv | o_proj | gate_up | down | per layer |
|---|---:|---:|---:|---:|---:|
| 1 | 16.1 / 16.4 | 14.8 / 14.6 | 51.5 / 55.6 | 30.0 / 31.0 | **112.4** / 117.6 |
| 16 | 16.3 / 17.8 | 15.3 / 16.3 | 55.0 / 58.7 | 32.0 / 33.2 | **118.5** / 126.0 |
| 64 | 31.7 / 34.9 | 27.5 / 30.3 | 85.2 / 89.4 | 53.7 / 65.9 | **198.2** / 220.5 |
| 128 | 45.2 / 49.5 | 35.9 / 41.5 | 155.1 / 165.9 | 84.2 / 94.1 | **320.4** / 351.0 |
| 256 | 74.3 / 80.1 | 56.6 / 58.4 | 291.4 / 322.8 | 149.0 / 164.3 | **571.4** / 625.5 |

- Round 3's remaining gate_up M=64 (102.0 against 89.6) is now 85.2 against 89.4.
- Before round 4, b=128 and 256 were 454.2 and 1008.0 µs per layer (`concur-c1`).

### Measurement lessons

- **A Colab rerun can load a stale kernel**: the tarball keeps the local mtimes, and if they are older than the previous build on the VM, ninja does not rebuild (found in the first `chunk-c2` attempt when 68 tests failed). `colab_launcher.py` now sets the mtimes to now after extracting.
- **The trace SM clock differs each time** (1.08–1.41 GHz), so per-tile times must be converted to cycles to compare. Converted, the 128x128 tile at MT 4 went from about 1,900 to about 1,600 cycles, and 64x256 at M=256 is about 1,430 cycles (tensor core floor 1,024).

### Not done yet

- The slice end is still about 25% of the critical path at M=64: 128x128 at MT 3–4 sums 4 warps in order (about 2.3–4.6 µs). A pairwise sum needs one more fp32 tile, and only 29 KB of shared memory is left, so it would cost one stage.
- Each tile is still 40–55% above the tensor core floor: dequant and HMMA are not interleaved (ptxas schedules all of a row's dequant before the HMMA), and there is a barrier every two k16 rows.
- nano's default `cuda_graph_max_bs` is 64; switching to a full graph for large batches is about 1% faster.

## Kernel list, round 5: sort-free sampler (runs `sampler-s1`, `sampler-s2`, `sampler-serve`)

Round 1 measured the sampling path (item 5). With the server's default temperature 0.7 and top-p 0.9, nano BF16 paid 0.42 / 1.02 / 2.62 ms per step for sampling at b=1/16/64, against vLLM's 0.25 / 1.33 / 1.66. This round replaces that path with CUDA kernels that never sort. Sampling now costs nano 0.02 / 0.06 / 0.13 ms, against vLLM's 0.26 / 1.33 / 1.74 in the same run. Branch `claude/fast-sampler`, local only.

### How vLLM 0.19.1 samples

Read from v0.19.1's source, default settings, on the A100:

- The logits are converted to fp32 and divided by the temperature in place.
- `apply_top_k_top_p`: for B ≥ 8, a Triton pivot search with one program per row (grid min(SMs, B), up to 18 passes over the row); for B < 8, a full sort and a cumsum in PyTorch.
- Softmax, then `random_sample`: `exponential_()`, a division and an argmax.
- Greedy rows take the argmax of the fp32 copy. FlashInfer's sampler is off unless `VLLM_USE_FLASHINFER_SAMPLER=1`.

Neither engine splits a row across blocks, and both write full-vocabulary fp32 intermediates several times per step.

### Design (`src/serving/nano_vllm/kernels/csrc/sampler.cu`)

- **No sort**: bf16 and fp16 logits have a 16-bit key with the same order, and dividing by a positive temperature keeps that order, so any top-k or top-p cutoff is one key value. A histogram over the key's high 10 bits finds the bin holding the cutoff; a second histogram over that bin's low 6 bits finds the exact key.
- **Every batch size fills the GPU**: each row is split into chunks of 2,048, 4,096 or 8,192 tokens (fewer rows, smaller chunks), on a (chunks, rows) grid. The chunks of a row meet in a per-row workspace through global atomics. The row's last block to finish (a ticket, as in `w4a16.cu`) completes the row and re-zeroes the workspace, so one workspace serves every call and every CUDA graph.
- **Five kernels**, each launched only when some row needs it:
  - stats: max and argmax. Greedy rows end here; an all-greedy batch launches nothing else.
  - hist: 1,024 bins of counts (top-k) and probability mass (top-p). Masses are 2^37 fixed point, so their sums are exact integers in any order.
  - refine: 64 bins inside the boundary bin, which give the exact cutoff key. A second refine runs only for rows with both top-k and top-p whose top-p cutoff lies in a higher bin than top-k's.
  - draw: an exponential race over the kept tokens, the argmax of v − log E with E ~ Exp(1). E comes from Philox4x32-10 keyed by (engine seed, sequence id, position, token index), so a request's tokens do not depend on the batch it runs in.
- **Ties stay with the cutoff**: top-k keeps a token when fewer than k tokens are larger, and top-p keeps one when the mass of the larger tokens is below p of the total, so the kept set depends only on the values. `sampling.keep_mask`, the CPU and no-nvcc path, uses the same rule.
- **No host sync**: each step uploads the rows' settings in one copy from pinned staging with two slots, like the decode graph's inputs.

### S1: shared u64 atomics are CAS loops on sm_80

In `sampler-s1` sampling already cost less than vLLM's at every b, but `sample_hist_kernel` took 1,382 µs per step at b=64. The SASS showed why: on sm_80 only 32-bit integer shared-memory atomics (add, min, max) are native. u64 `atomicAdd` and `atomicMax`, and f32 `atomicAdd`, compile to compare-and-swap loops (`ATOMS.CAST.SPIN`). The key's high 8 bits are the sign and 7 exponent bits, so each of the 256 bins spans a factor of 4 and a row's tokens crowd a few of them; the loops serialized. The fix (a926db2):

- masses go into shared memory as two u32 halves, split at bit 19, and into global memory as u64, where `RED.E.ADD.64` is native;
- 1,024 first-level bins instead of 256;
- with top-p alone, tokens more than ln(V / (1 − p)) nats below the max are never kept (all of them together hold under 1 − p of the max token's mass), so they add to the total from a register instead of the histogram;
- the draw became a kernel of its own, so no shared u64 max is left.

Sampler kernels per decode step (nsys, W4A16 sampled, µs):

| b | `sampler-s1` | `sampler-s2` | hist alone |
|---:|---:|---:|---:|
| 1 | 96 | 26 | 72 → 10 |
| 16 | 385 | 67 | 323 → 26 |
| 64 | 1,548 | 172 | 1,382 → 68 |

In `sampler-s1` every sampled kbench row errored as well. The PyTorch impl uploaded pinned tensors inside the timed call, which a CUDA graph cannot capture, and the failed capture left the CUDA RNG in capture mode for every later call. The settings are now uploaded once, outside the timed call.

### kernel (kbench, run `sampler-s2`, µs, nano / vLLM)

Logits [B, 128,256] bf16, "peaked" (randn × 4 plus a few spikes), except the flat column (randn × 1, the largest nuclei):

| B | greedy | T 0.7, p 0.9 | T 0.7, p 0.9, flat | T 1 (vLLM default) | T 0.7, p 0.9, k 50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.7 / 41.1 | 32.8 / 269.0 | 35.0 / 270.8 | 21.1 / 103.5 | 39.9 / 304.8 |
| 16 | 19.6 / 51.0 | 51.9 / 550.9 | 62.8 / 1,456.7 | 36.9 / 136.7 | 62.9 / 426.2 |
| 64 | 30.5 / 105.8 | 104.5 / 882.2 | 149.1 / 2,064.3 | 90.2 / 362.9 | 128.0 / 719.0 |
| 256 | 79.0 / 296.3 | 301.1 / 2,577.1 | 459.5 / 7,154.4 | 273.7 / 1,296.0 | 382.4 / 2,292.5 |

- nano is faster at all 8 batch sizes from 1 to 256, in every setting and both distributions (the flat rows of T 1 and k 50 are in the summary): 2.6–3.8× for greedy, 3.5–30× when sampling.
- Every drawn token passed the check: it is in the kept set of a float64 reference (at p + 1e-4), and greedy rows equal the argmax.
- The PyTorch path (`sample_rows`, the same rule with a sort) takes 0.3 ms at B=1 and 10.7 ms at B=256.

### End-to-end decode (run `sampler-s2`, ms/step)

| config | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16, sampled | **5.11** | **7.11** | **15.13** |
| vLLM AWQ, sampled | 5.81 | 9.33 | 18.11 |
| nano BF16, sampled | **12.03** | **14.14** | **22.68** |
| vLLM BF16, sampled | 12.87 | 16.15 | 24.76 |
| nano W4A16, greedy | **5.09** | **7.06** | **15.00** |
| vLLM AWQ, greedy | 5.54 | 7.84 | 16.20 |
| nano BF16, greedy | **12.01** | **14.08** | **22.55** |
| vLLM BF16, greedy | 12.61 | 14.82 | 23.02 |

Sampled minus greedy:

| ms/step | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16 | **0.02** | **0.05** | **0.13** |
| vLLM AWQ | 0.27 | 1.49 | 1.91 |
| nano BF16 | **0.02** | **0.06** | **0.13** |
| vLLM BF16 | 0.26 | 1.33 | 1.74 |
| nano BF16 before this round (`kernels-r1`) | 0.42 | 1.02 | 2.62 |

nano's sampled decode is 12% / 24% / 16% faster than vLLM AWQ's at b=1/16/64, and 7% / 12% / 8% faster than vLLM BF16's. Greedy also uses the stats kernel instead of PyTorch's bf16 argmax (kbench B=1: 15.7 against 52.0 µs). Against vLLM on the same VM, greedy W4A16 went from 0.94 / 0.91 / 0.93 of vLLM's time (`chunk-c7`, fullgraph) to 0.92 / 0.90 / 0.93 (piecewise graph here).

### Online (run `sampler-serve`, Phase 2's harness, BF16)

`vllm bench serve`, random 1024-token prompts, 256 output tokens. nano runs as in `kernels-serve`: core process, async scheduling, 2,048-token chunks, KV sized from 0.9 of GPU memory.

| | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| W1s output tok/s, nano / vLLM | **81** / 75 | **904** / 612 | **1,833** / 1,692 |
| W1s ITL p50 (ms), nano / vLLM | **12.05** / 13.05 | **14.16** / 16.27 | **22.73** / 25.00 |
| W1 (greedy) ITL p50 (ms), nano / vLLM | **12.03** / 12.78 | **14.08** / 14.83 | **22.62** / 23.22 |

- Sampling adds 0.02 / 0.08 / 0.11 ms to nano's ITL p50 and 0.27 / 1.44 / 1.78 ms to vLLM's.
- In `kernels-serve`, nano's W1s ITL p50 at c=64 was 25.31 ms, 1.1% behind vLLM. It is now 22.73 ms, 9% ahead.
- vLLM's sampled throughput at c=16 is again far below its greedy one (612 against 937 tok/s, TTFT p50 498 ms), as in `kernels-serve`. This was not investigated.

### Gates

- **G1, correctness**: passed. 777 tests passed with no skips. They cover:
  - kept sets between a float64 reference's at p − 1e-4 and at p + 1e-4, for every setting × distribution (peaked, flat, ties, −inf), 1 to 256 rows, vocabularies of 128,256 and 1,000, and fp16;
  - both top-k + top-p branches;
  - greedy equal to `torch.argmax`, first index on ties;
  - a chi-square test of the drawn distribution;
  - determinism, CUDA graph replay, and the sync guard on the sampled path.
- **G2, kernel**: faster than vLLM at every B, setting and distribution, passed. Greedy faster than nano's old argmax, passed. **Missed**: b=64 at (0.7, 0.9) in ≤ 60 µs; measured 104.5 µs peaked and 149.1 flat.
- **G3, end to end**: sampled minus greedy below vLLM's at b=1/16/64, and nano sampled faster than vLLM sampled in BF16 and W4A16, both passed. **Missed**: the ≤ 0.1 ms target at b=64, measured 0.13 (b=1 and 16 are 0.02–0.06). There is no same-VM greedy baseline with the old sampler. Greedy's ratio to vLLM did not get worse (above).
- **G4, online**: passed. W1s ITL p50 is at or below vLLM's, and throughput at or above, at c=1, 16 and 64.

### Not done yet

At b=64 the four kernels take 172 µs per step, while a pass that only reads the logits (stats) takes 23. The benchmarks' prompts are random tokens, which likely gives high-entropy rows (the kernels take longer here than on kbench's flat rows), and two kernels pay for that:

- **hist (68 µs)**: many of a row's tokens land in the same few bins, so lanes of a warp still collide on shared u32 atomics, which replay. Warp-aggregated atomics (`__match_any_sync`, then `__reduce_add_sync`) would issue one atomic per distinct bin per warp.
- **draw (52 µs)**: with a large nucleus, almost every warp step has a kept token, so all lanes pay for Philox (two calls per 8 tokens), `log1pf` and `__logf`, about 50 instructions per token. An inverse-CDF draw would replace that with one exp per token: one random number per row, then a search over per-chunk kept masses.

## Kernel list, round 6: AWQ weights, int8 lm_head, large-M prefill (runs `awq-a1` to `awq-a3`)

Until this round nano's W4A16 was its own RTN quantization at load: symmetric, groups of 128, no zero points. On wikitext-2 that cost 10.9% perplexity (8.023 against BF16's 7.232), and vLLM's AWQ side ran a different checkpoint (`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`), so the two engines never had the same weights. nano now loads that AutoAWQ checkpoint itself, and two speedups were tried on top: an int8 lm_head and a W4A16 kernel for prefill-sized batches. Branch `claude/nano-awq`, local only.

### What was done

- **AWQ loading** (`model.py`, `kernels/w4a16.py`): `weight_quant: awq`, chosen on its own for a checkpoint whose `quantization_config` is AutoAWQ GEMM, 4 bits, groups of 128, zero points (anything else is refused). `awq_unpack` undoes AutoAWQ's packing (nibble order 0, 2, 4, 6, 1, 3, 5, 7 along N), and the codes go into the tiled layout `gemm_w4a16_tiled` already reads. q/k/v and gate/up are concatenated while still packed; fp16 scales become the model's dtype.
- **Zero points in the tiled kernel** (`csrc/w4a16.cu`): one byte per column and group, beside the scales in each stage. A tile turns them into (magic + z) pairs with `__byte_perm`, and `dequant_pair` subtracts that instead of a constant, so q − z is exact before the scale. RTN runs the same kernel with z = 8, bit-identical to before. Like the scales before zero points, a tile's scales and zero words are read with its first row's fragments, after the barrier that publishes the stage (read at the top of each tile instead, the first mma waited on shared memory: +7–8% GEMM time at 64–128 rows).
- **int8 lm_head** (`kernels/w8a16.py`): symmetric int8 per vocab row with fp32 scales, a Triton kernel that converts to bf16 inside `tl.dot` and scales the output. Up to 64 rows it reads half the bytes of bf16; past that it reaches only ~130 TFLOPs, so `lm_head_quant: int8_dense`, the default under 4-bit weights, also keeps the bf16 weight for batches past 64 rows.
- **Prefill past 256 rows**: `gemm_w4a16_tiled` now takes up to 512 rows (`tiled_max_rows`). `gemm_w4a16_large`, a kernel for 64·WM-row chunks with 64×64 per warp, ran at ~180 TFLOPs at 1,024–2,048 rows, slower than dequantizing and calling cuBLAS (230–250), so it is opt-in (`"large": True`) and dequantize + cuBLAS stays the path past 512 rows.
- **Tools**: kbench's `w4a16_linear` uses asymmetric weights and adds vLLM's AWQ Marlin (repacked as vLLM 0.19.1 does) and the large kernel; `awq_dequant` checks a real layer against vLLM's `awq_dequantize_triton` bit for bit; `lm_head` compares int8 configs with bf16. `quality.py` takes `--awq-model` and an lm_head field, and `quality_vllm.py` scores vLLM on the same chunks.

### Quality (wikitext-2 test, 288,627 tokens in chunks of 2,048, run `awq-a1`)

| config | weights | perplexity | top-1 agreement vs nano BF16 |
|---|---|---:|---:|
| nano BF16 | bf16 | 7.232 | - |
| vLLM BF16 | bf16 | 7.232 | 98.81% |
| nano W4A16 (RTN, before) | RTN at load | 8.023 (+10.9%) | 85.91% |
| nano AWQ | AutoAWQ checkpoint | **7.562** (+4.6%) | 90.10% |
| vLLM AWQ (`awq_marlin`) | AutoAWQ checkpoint | 7.560 (+4.5%) | 90.13% |
| nano AWQ + int8 lm_head | | 7.563 | 90.06% |

- nano AWQ is within 0.04% of vLLM AWQ, and the first layer's dequantized weights are identical to vLLM's.
- The int8 lm_head agrees with the bf16 head on 99.01% of top-1 tokens, at +0.001% perplexity (run `awq-a2`).

### kernel (kbench, run `awq-a3`, cold, µs)

Decode shapes, nano tiled (AWQ) / vLLM AWQ Marlin:

| projection | M=1 | M=16 | M=64 | M=128 | M=256 |
|---|---:|---:|---:|---:|---:|
| qkv | 18.1 / 19.1 | 16.0 / 17.7 | 32.8 / 35.8 | 47.4 / 49.9 | 75.7 / 78.7 |
| o_proj | 14.5 / 14.8 | 14.9 / 16.1 | 27.7 / 28.3 | 36.7 / 41.4 | 56.6 / 56.5 |
| gate_up | 52.7 / 54.9 | 55.7 / 59.4 | 90.8 / 91.4 | 164.3 / 171.7 | 309.6 / 330.6 |
| down | 30.8 / 30.0 | 32.3 / 32.4 | 56.9 / 65.3 | 91.4 / 96.2 | 158.7 / 169.7 |

Prefill shapes, a layer's four projections (large: the faster of its two tiles):

| rows | tiled | large kernel | dequantize + cuBLAS | vLLM Marlin | cuBLAS BF16 |
|---:|---:|---:|---:|---:|---:|
| 384 | **865** | 1,041 | 1,171 | 921 | 784 |
| 512 | **1,171** | 1,300 | 1,392 | 1,212 | 1,014 |
| 768 | - | 1,874 | 1,836 | **1,778** | 1,469 |
| 1,024 | - | 2,450 | **2,288** | 2,355 | 1,921 |
| 2,048 | - | 4,948 | **3,971** | 4,685 | 3,597 |

- nano's path (tiled up to 512 rows, dequantize + cuBLAS past that) is faster than Marlin everywhere but 768 rows, where it is 3% slower.
- The large kernel reaches ~180 TFLOPs at 2,048 rows, cuBLAS BF16 ~250.

lm_head (run `awq-a2`): int8 369–386 µs up to 16 rows against bf16's 713–736, and 524 against 779 at 64 rows; at 128 and 256 rows 1,124 and 2,111 against cuBLAS's 810 and 1,168.

### End-to-end decode (run `awq-a3`, ms/step, greedy)

| config | b=1 | b=16 | b=64 | b=128 | b=256 |
|---|---:|---:|---:|---:|---:|
| nano AWQ, `int8_dense` lm_head (default) | **4.80** | **6.77** | **14.95** | **26.61** | **37.07** |
| nano AWQ, bf16 lm_head | 5.16 | 7.13 | 15.25 | 26.60 | 37.04 |
| vLLM AWQ | 5.55 | 7.86 | 16.20 | 27.35 | 38.88 |
| nano RTN, before zero points (`sampler-s2`) | 5.09 | 7.06 | 15.00 | - | - |

nano now takes 0.87 / 0.86 / 0.92 / 0.97 / 0.95 of vLLM AWQ's time with the same weights; with RTN it was 0.92 / 0.90 / 0.93 at b=1/16/64. GEMM per layer in decode (nsys, µs, nano / vLLM): b=1 113.9 / 117.7, b=16 119.5 / 126.1, b=64 205.3 / 220.2, b=128 332.4 / 349.5.

### Gates

- **G1, correctness**: passed. 962 tests passed with no skips (AWQ unpacking and loading, zero points in every tiled shape and both dtypes, the large kernel, int8 lm_head, CUDA graphs); a real layer matches vLLM bit for bit; perplexity is within ±0.5% of vLLM AWQ; the large kernel's perplexity is within 0.01 of the dequantize path.
- **G2, AWQ decode at most 1% slower than RTN**: **missed**, +1.4% / +1.0% / +1.7% at b=1/16/64 with the bf16 head. At b=1 the step is bandwidth-bound, and the zero points add 1.5% to the bytes read (64 bytes of weights, 2 of scale and 1 of zero point per column and group). At b=64 the GEMM is still 3.7% slower, from decoding zero points once per k16 row at MT 3–4, where a tile's pairs did not fit in registers. Still faster than vLLM AWQ at every b.
- **G3, int8 lm_head**: passed. Quality as above; 0.36 ms faster at b=1, and with `int8_dense` no batch size is slower.
- **G4, large-M kernel**: **missed**. The large kernel is 7% slower than dequantize + cuBLAS at 1,024 rows and 25% at 2,048; prefill gains only from the tiled kernel up to 512 rows (−26% at 384 rows, −16% at 512). Online, the uncached 1024-token TTFT at c=1 did not change (89.1 ms, Phase 2's `awqserve`), since a 1,024-row step still dequantizes.
- **G5, online** (Phase 2's `awqserve`): nano beats vLLM AWQ on throughput, TTFT p50 and ITL p50 in W1, W1u, W1s and W3. **Missed** with 2,048-token chunks: W1's ITL p99 at c=16 (86.65 against 75.20 ms) and W3's ITL p99 (12.2 against 11.6). With nano's default 512-token chunks the former is 44.85 ms, but W3's ITL p99 (29.3 ms) and long-prompt TTFT (433 against 399 ms) fall behind.

### Not done yet

- **A faster prefill kernel**: the large kernel lacks the tiled kernel's register double buffer (it spilled), and each B fragment is dequantized by every warp row of the block (2× at 128×256). Dequantizing a stage into shared memory once, then running a plain BF16 mma loop over it, may get closer to cuBLAS.
- **int8 lm_head past 64 rows**: a CUDA kernel in the tiled kernel's style would let `int8` drop the bf16 copy (0.5 GB).
- **Zero points as nibbles**: half the zero-point bytes, recovering about 0.7% at b=1.
- **KV capacity after loading AWQ**: done after the round (Phase 2's `kvfrag`). The memory left reserved after loading came from the int8 lm_head's fp32 temporaries, plus 0.76 GiB from the projections with RTN or AWQ, not from AWQ unpacking alone. Weights are now built on a side stream, and nano keeps 230,736 KV tokens (`awqserve`: 215,424; vLLM: 237,824).

## Files

| file | content |
|---|---|
| `run_phase1.sh` | main flow. The step runner, installs, nsys and ncu functions come from `../phase0/lib.sh` |
| `kbench.py` | kernel micro-benchmark: registration, correctness checks, CUDA graph timing, JSON and markdown |
| `tune.py` | sweep of launch configs, reusing kbench's timing |
| `tiled_trace.py` | trace build of `gemm_w4a16_tiled`: per-block time breakdown, the last block to finish, `--prefetch-tiles` to measure the upper bound of L2 prefetch; over 64 rows the same schedule is applied per chunk |
| `golden.py` | bit-for-bit comparison of the tiled kernel's output (saved once, checked afterwards) |
| `gemm_breakdown.py` | reads the GEMM time of each projection in decode from nsys's SQLite, for both nano and vLLM; consecutive launches of the same projection are merged into one, and a `:prelude` regex (for example dequantization) is counted with the next GEMM |
| `quality.py` | perplexity and top-1 agreement on the 8B; AWQ configs load `--awq-model` |
| `quality_vllm.py` | vLLM's perplexity and top-1 on `quality.py`'s chunks (vLLM venv), added to the same table |
| `interp/` | runs `TRITON_INTERPRET=1` locally in Docker (linux/arm64) to check the logic of the Triton kernels: `interp_check.py`, and `dot_check.py`, which shows the interpreter's bf16 `tl.dot` is wrong |
| `summarize.py` | collects one run into `summary.md` |
| `results/<run_id>/` | `summary.md`, `status.tsv`, `logs/`, `tests/`, `kbench/`, `tune/`, `ncu/`, `nsys/`, `offline/`, `quality/`, `trace/`, `environment.txt`, and `code/`, a copy of the code for that run |

The Colab driver and launcher reuse Phase 0's `colab_phase0.sh` and `colab_launcher.py`, switched with `BENCH_PHASE=phase1`. Besides `benchmarks/phase0`, `benchmarks/phase1` and `src/serving/nano_vllm`, the tarball carries `tests/unit/test_nano_vllm_*.py`, `tiny_llama.py`, and the files outside nano_vllm that these tests import or read: `src/monitoring/metrics.py`, `src/api/model_routing.py`, `config/model_routing.nano.json`. `tests/gpu/` is included too if it exists.

## Limitations

- Only run on a Colab A100-SXM4-40GB. On 80GB cards the peak switches to 2,039 GB/s automatically; other GPUs need `--peak-gbps`.
- Each point is measured once: kbench takes the median of 20 replays, and e2e runs each config once.
- The name `cuda129` comes from the plan, but what is installed is the same CUDA version as the venv's torch. Phase 0 recorded vLLM 0.19.1's torch as `2.10.0+cu128`, so 12.8 is installed by default; to force 12.9, set `P1_CUDA_VERSION=12.9`.
