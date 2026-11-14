# nano_vllm

An LLM inference engine written from scratch, with its own Triton and CUDA
kernels. It serves Llama 3.1 8B Instruct behind an OpenAI-compatible API, and
on one A100 it decodes faster than vLLM 0.19.1 in BF16, with an FP8 KV cache,
and with 4-bit weights, greedy or sampled. With the public DFlash drafter it
does lossless speculative decoding, and on the AWQ checkpoint it serves more
tokens per second than vLLM 0.33.0 with the same drafter at every load measured.

All numbers below come from a Google Colab A100-SXM4-40GB with Llama 3.1 8B
Instruct. Each comparison was measured in a single run on a single VM, except
the speculative decoding baselines, which come from an earlier VM of the same
type. Nothing has been measured on other GPUs.

## Results

### Speculative decoding (DFlash drafter)

nano loads `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`, the drafter vLLM and
SGLang load: 5 layers that draft a block of 9 tokens in one forward. The
target verifies the block in one step and keeps the drafts up to its first
disagreement, so the output is the target's own. Real prompts sent to
`/v1/completions`, requests stop at EOS or 256 tokens, output tok/s at
temperature 0 (vLLM 0.33.0, SGLang 0.5.23):

| workload | AWQ: nano + DFlash | nano, no drafter | vLLM + DFlash | BF16: nano + DFlash | vLLM + DFlash | SGLang + DFlash |
|---|---:|---:|---:|---:|---:|---:|
| alpaca c=1 | **455** | 207 | 380 | **224** | 220 | 196 |
| alpaca c=16 | **3252** | 2538 | 2084 | **2041** | 1910 | 1652 |
| alpaca c=64 | 5946 | **5978** | 2943 | **5190** | 3327 | 2915 |
| humaneval c=1 | **677** | 206 | 589 | **344** | 322 | 295 |
| humaneval c=16 | **3607** | 2523 | 2950 | 2772 | **2806** | 2456 |
| humaneval c=64 | **5697** | 5150 | 4204 | **5537** | 4576 | 4168 |
| gsm8k c=1 | **511** | 206 | 434 | **259** | 256 | 222 |
| long context c=1 | **304** | 177 | 245 | **174** | 161 | 157 |

- **AWQ target**: above vLLM + DFlash at all 16 (workload, load) points, by
  1.09× to 2.02×, and never below 0.99× of nano without a drafter. With the
  drafter's weights quantized to 4 bits at load, c=1 rises to 512 / 764 / 576
  / 333 tok/s.
- **BF16 target**: above SGLang everywhere. Against vLLM it is level (0.98× to
  1.02×) where every step verifies the whole block, and 1.04× to 1.56× ahead
  elsewhere.
- **Sampled requests** (temperature 0.7, top-p 0.9) give the tokens of the same
  seed without a drafter: the drafter draws with the target's noise for the
  same position. AWQ c=1: 436 tok/s on alpaca against vLLM's 371, 667 on
  humaneval against 556.
- **High load**: at c=64 a plain decode step yields as many tokens per
  millisecond as a cycle, and nano runs plain steps there. vLLM keeps
  drafting and falls below its own server without a drafter (AWQ alpaca
  c=64: 2943 with DFlash, 5092 without).
- **Cost**: with a drafter, TTFT p50 on alpaca (AWQ) is 9.6 / 28.4 / 41.7 /
  56.6 ms at c=1/4/16/64, against 9.7 / 15.9 / 18.9 / 47.5 without one (vLLM
  AWQ + DFlash: 24.1 / 31.0 / 71.8 / 214.3).

Where it comes from:

- A cycle (draft, draw, verify, draw, accept) is one CUDA graph: 7.19 ms for
  one AWQ sequence with 10 verify rows, against 4.80 ms for a plain decode step.
- Verify attention reads a sequence's KV once for all of its rows and query
  heads (`kernels/block_attention.py`).
- Each step picks its verify length (1, 3, 5 or 10 rows per sequence) from the
  graphs' step times measured at startup and the acceptance at each draft
  depth so far; a length of 1 is a plain decode step.
- A cycle computes its positions, KV slots and next anchor on the device, so
  the next cycle is launched before the one in flight reaches the host
  (1.01× to 1.09× serving).

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
- Lossless speculative decoding with a DFlash drafter: the drafter and its
  own KV pool (`dflash.py`), cycle graphs chained behind the cycle in flight
  (`model_runner.py`), and the policy that picks each step's verify length
  (`spec.py`)
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
- `block_attention.py`: attention for a block of rows per sequence, causal for
  the target's verify rows and bidirectional for the drafter, in Triton
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
fp8` turns on the FP8 KV cache. `engine.speculative_model` (a DFlash drafter,
for example `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`) turns on speculative
decoding; `engine.speculative_weight_quant: w4a16` quantizes the drafter to 4
bits at load.

The benchmarks run on Colab through a single driver script, which starts a VM,
runs the steps and downloads the results:

```bash
BENCH_PHASE=phase1 bash benchmarks/phase0/colab_phase0.sh   # kernels, offline decode
BENCH_PHASE=phase2 bash benchmarks/phase0/colab_phase0.sh   # online serving
BENCH_PHASE=phase3 bash benchmarks/phase0/colab_phase0.sh   # speculative decoding
```

Tests: `pytest tests/unit/test_nano_vllm_*.py` runs on CPU; `pytest tests/gpu`
needs a GPU.

## References

| result | data | write-up |
|---|---|---|
| Speculative decoding: nano with DFlash | [m7-fix summary](../../../benchmarks/phase3/results/m7-fix-20261113-120835-colab/summary.md) | [phase3 README](../../../benchmarks/phase3/README.md), M7 |
| Speculative decoding: vLLM and SGLang baselines | [m0 summary](../../../benchmarks/phase3/results/m0-20261111-074458-colab/summary.md) | [phase3 README](../../../benchmarks/phase3/README.md), M0 |
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
