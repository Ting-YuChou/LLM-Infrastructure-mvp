# nano_vllm

An LLM inference engine written from scratch, with its own Triton and CUDA
kernels. It serves Llama 3.1 8B Instruct behind an OpenAI-compatible API, and
on one A100 it decodes faster than vLLM 0.19.1 in BF16, with an FP8 KV cache,
and with 4-bit weights, greedy or sampled.

All numbers below come from a Google Colab A100-SXM4-40GB with Llama 3.1 8B
Instruct. Each comparison was measured in a single run on a single VM. Nothing
has been measured on other GPUs.

## Results

### 4-bit weights: the same AWQ checkpoint as vLLM

nano loads vLLM's AutoAWQ checkpoint
(`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`) and runs it through its
own W4A16 GEMM; vLLM runs it through Marlin (`awq_marlin`). Offline decode, ms
per step, CUDA graphs on both sides:

| batch / prompt | nano AWQ | vLLM AWQ | speedup |
|---|---:|---:|---:|
| 1 / 128 | 4.80 | 5.55 | 1.16× |
| 16 / 1024 | 6.77 | 7.86 | 1.16× |
| 64 / 1024 | 14.95 | 16.20 | 1.08× |
| 128 / 1024 | 26.61 | 27.35 | 1.03× |
| 256 / 512 | 37.07 | 38.88 | 1.05× |

- Same quality: wikitext-2 perplexity 7.562 against vLLM's 7.560 (BF16:
  7.232; nano's earlier round-to-nearest W4A16 reached 8.023), and a real
  layer dequantizes to exactly vLLM's values.
- The GEMMs alone (nsys, inside the decode graph) run 1.03–1.07× faster than
  Marlin per layer from b=1 to 128.
- Up to 64 rows lm_head runs in int8 with a scale per vocab row, 0.30–0.36 ms
  less per step, at +0.001% perplexity and 99.01% top-1 agreement with BF16;
  past 64 rows it uses the BF16 weight.

Online (`vllm bench serve`, random 1024-token prompts, 256 output tokens),
nano serves 204 / 1,679 / 2,107 tok/s at c=1/16/64 against vLLM AWQ's
172 / 1,502 / 1,941, with ITL p50 4.85 / 6.79 / 15.01 ms against
5.75 / 7.88 / 16.35. An uncached 1024-token prompt's TTFT p50 is 89 ms
against 95 at c=1, and 350 ms against 645 at c=16.

### BF16 and FP8 KV cache

Offline decode, ms per step:

| | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano BF16 | **12.05** | **14.13** | **22.60** |
| vLLM BF16 (FlashAttention 2) | 12.60 | 14.82 | 23.04 |
| nano FP8 KV | **12.01** | **13.81** | **20.87** |
| vLLM FP8 KV (FlashInfer) | 13.05 | 14.50 | 21.30 |

FP8 KV holds 1.94× as many tokens (157,744 → 305,920) and raises wikitext-2
perplexity by only 0.22%. The A100 has no FP8 instructions, and Triton 3.6
rejects `float8e4nv` on sm80 (as does vLLM's Triton backend), so the kernels
decode e4m3 with integer ops. Under shared-prefix load that overflows the BF16
cache, nano's TTFT p50 is 160 ms against vLLM FP8's 259 ms.

### Sampling (temperature 0.7, top-p 0.9)

The server's default sampling, offline decode, ms per step:

| | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| nano W4A16 (round-to-nearest weights) | **5.11** | **7.11** | **15.13** |
| vLLM AWQ | 5.81 | 9.33 | 18.11 |
| nano BF16 | **12.03** | **14.14** | **22.68** |
| vLLM BF16 | 12.87 | 16.15 | 24.76 |

Sampling adds 0.02–0.13 ms to nano's decode step and 0.26–1.91 ms to vLLM's.
nano's sampler never sorts. It finds the exact top-k and top-p cutoffs with
radix histograms over the logits' 16-bit keys, splits each row across many
blocks, and draws with an exponential race from Philox, so a request's tokens
do not depend on the batch it runs in. On its own it runs 3.5–30× faster than
vLLM's sampling path at every batch size from 1 to 256.

Online with the same sampling (`vllm bench serve`, random 1024-token prompts,
BF16), nano's ITL p50 is 12.05 / 14.16 / 22.73 ms at c=1/16/64 against vLLM's
13.05 / 16.27 / 25.00, and nano serves 1,833 tok/s at c=64 against 1,692.

### Online serving vs vLLM and SGLang (BF16)

Load from `vllm bench serve`; random 1024-token prompts, 256 output tokens.

| | nano | vLLM | SGLang |
|---|---:|---:|---:|
| throughput, c=64 (tok/s) | **1,997** | 1,976 | 1,284 |
| TPOT p50, c=1 (ms) | **12.09** | 12.81 | 12.60 |
| TTFT p50, c=1 (ms) | **22.4** | 26.6 | – |
| TTFT p50, c=1, uncached prompt (ms) | **76.1** | 81.8 | – |
| shared 2k prefix: req/s · TTFT p50 | **7.84 · 165 ms** | 7.32 · 270 ms | 7.20 · 206 ms |
| ITL p50 with 4k prompts arriving (ms) | **16.36** | 17.22 | 17.05 |

The TTFT rows come from a later run with piecewise CUDA graphs; vLLM was
measured in that same run, and SGLang was not part of it.

### Kernels and starting point

- **Paged decode attention** (Triton): faster than vLLM's FA2 split-KV at 11 of
  12 (batch, context) points, up to 1.44×, and 86–93% of HBM peak at large
  batch × context.
- **Starting point**: the first version of nano took 44.1 ms per token at b=1
  and 619 ms per decode step at b=16. It now takes 12.05 and 14.13 ms.

## What's inside

Engine:

- Continuous batching, with chunked prefill mixed into decode steps (`scheduler.py`)
- Paged KV cache sized from free GPU memory, plus a radix-tree prefix cache
  written in C++ with nanobind (`block_manager.py`, `csrc/radix_tree.cpp`)
- The engine core runs in its own process with async scheduling: step N+1
  launches before step N's tokens reach the host (`core.py`, `engine.py`)
- CUDA graphs: full graphs for decode, and piecewise graphs for prefill and
  mixed steps, with attention run eagerly between the pieces (`model_runner.py`)
- An OpenAI-compatible server (`server.py`)

Kernels (`kernels/`):

- `csrc/w4a16.cu`: a Marlin-style W4A16 GEMM.
  - Weights are stored pre-laid-out as `mma.sync` fragments, AWQ's zero points
    as one byte per column and group.
  - A 4-stage `cp.async` pipeline feeds a persistent striped grid.
  - Results are reduced deterministically, with no spin locks.
  - One launch handles up to 512 rows: the rows are split into chunks that
    share weight tiles through L2. Larger batches dequantize and call cuBLAS.
- `w8a16.py`: the int8 lm_head (Triton).
- `csrc/sampler.cu`: sort-free sampling. Top-k / top-p cutoffs come from
  radix histograms over 16-bit logit keys, and tokens are drawn with an
  exponential race keyed by (seed, sequence, position).
- `paged_attention.py`: paged decode attention (split-KV) and prefill attention, in Triton
- `fp8.py`: the FP8 (e4m3) KV cache codec for A100
- `rope_cache.py`, `rmsnorm.py`, `silu_mul.py`: fused RoPE + KV write, fused
  residual add + RMSNorm, and SiLU-and-mul
- `linear.py`, `csrc/gemv_bf16.cu`: the BF16 decode GEMV, in Triton and in CUDA

## Run it

```bash
python -m src.serving.nano_vllm.server --config config/nano_vllm_config.yaml
```

Point `model.name` at a Llama checkpoint. An AutoAWQ checkpoint (4 bits, groups
of 128, GEMM format) is detected and loaded as such; `engine.weight_quant: w4a16`
quantizes a BF16 checkpoint to 4 bits at load instead. `engine.kv_cache_dtype:
fp8` turns on the FP8 KV cache.

The benchmarks run on Colab through a single driver script, which starts a VM,
runs the steps and downloads the results:

```bash
BENCH_PHASE=phase1 bash benchmarks/phase0/colab_phase0.sh   # kernels, offline decode
BENCH_PHASE=phase2 bash benchmarks/phase0/colab_phase0.sh   # online serving
```

Tests: `pytest tests/unit/test_nano_vllm_*.py` runs on CPU; `pytest tests/gpu`
needs a GPU.

## References

| result | data | write-up |
|---|---|---|
| AWQ decode and per-layer GEMMs | [awq-a3 summary](../../../benchmarks/phase1/results/awq-a3-20261030-054204-colab/summary.md) | [phase1 README](../../../benchmarks/phase1/README.md), round 6 |
| AWQ quality (nano, vLLM, RTN) | [awq-a1 quality](../../../benchmarks/phase1/results/awq-a1-20261030-034248-colab/quality/quality.md) | same |
| int8 lm_head quality | [awq-a2 quality](../../../benchmarks/phase1/results/awq-a2-20261030-045241-colab/quality/quality.md) | same |
| AWQ online vs vLLM AWQ | [awqserve summary](../../../benchmarks/phase2/results/awqserve-20261030-060945-colab/summary.md) | [phase2 README](../../../benchmarks/phase2/README.md), AWQ online |
| Sampling: kernel and offline decode | [sampler-s2 summary](../../../benchmarks/phase1/results/sampler-s2-20261028-120816-colab/summary.md) | [phase1 README](../../../benchmarks/phase1/README.md), round 5 |
| Sampling online (W1s) | [sampler-serve summary](../../../benchmarks/phase2/results/sampler-serve-20261028-124708-colab/summary.md) | same |
| BF16 / FP8 KV offline decode | [kernels-r4 summary](../../../benchmarks/phase1/results/kernels-r4-20261023-072114-colab/summary.md) | [phase1 README](../../../benchmarks/phase1/README.md), round 1 |
| FP8 KV quality | [kernels-r2 quality](../../../benchmarks/phase1/results/kernels-r2-20261023-065546-colab/quality/quality.md) | same |
| FP8 KV online (capacity, shared prefixes) | [kernels-serve summary](../../../benchmarks/phase2/results/kernels-serve-20261023-075342-colab/summary.md) | same |
| Online serving vs vLLM / SGLang | [final summary](../../../benchmarks/phase2/results/final-20261023-030032-colab/summary.md) | [phase2 README](../../../benchmarks/phase2/README.md) |
| TTFT with piecewise CUDA graphs | [ttft summary](../../../benchmarks/phase2/results/ttft-20261023-041614-colab/summary.md) | [phase2 README](../../../benchmarks/phase2/README.md) |
| Decode attention kernel | [run5 kbench](../../../benchmarks/phase1/results/20261022-p1-run5/kbench/kbench.md) | [phase1 README](../../../benchmarks/phase1/README.md) |
| Baseline and roofline | [phase0 summary](../../../benchmarks/phase0/results/20261022-colab-a100-40gb/summary.md) | [phase0 README](../../../benchmarks/phase0/README.md) |

Every run, including failures and fixes, is logged in `LOG.md` in each
`benchmarks/phase*/` directory.
