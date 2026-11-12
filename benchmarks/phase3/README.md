# Phase 3: speculative decoding

Roadmap Phase 3 for `src/serving/nano_vllm/`: lossless speculative decoding
with the public DFlash drafter (`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`,
the checkpoint vLLM and SGLang load), measured against both on one Colab
A100-SXM4-40GB with Llama 3.1 8B Instruct (BF16 and the AWQ INT4 checkpoint).

| milestone | change |
|---|---|
| M0 | baselines on real prompts (vLLM and SGLang with and without DFlash, nano as it is) and the drafter's acceptance study |
| M1 | the DFlash drafter in nano, reference block attention |
| M2 | synchronous greedy cycle on the existing kernels: output equal to no speculation |
| M3 | block attention kernel: one KV read for all of a sequence's verify rows |
| M4 | the cycle as CUDA graphs, 4-bit drafter |

## How to run

```bash
BENCH_PHASE=phase3 COLAB_SESSION=phase3 COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh          # new VM
BENCH_PHASE=phase3 COLAB_SESSION=phase3 COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh rerun    # same VM
```

`run_phase3.sh` documents the workloads, the steps and the `P3_*` variables.
Each run writes `summary.md` in its results directory.

## Results

### M0 (run `m0-20261111-074458-colab`, Colab A100 40GB, vLLM 0.33.0, SGLang 0.5.23)

Full tables: `results/m0-20261111-074458-colab/summary.md`. Requests stop at
EOS or 256 tokens; DFlash servers use the public drafter with block 10.

Output tok/s, temperature 0, one client (c=1):

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| vLLM BF16 | 78 | 78 | 78 | 74 |
| vLLM BF16 + DFlash | 220 | 322 | 256 | 161 |
| SGLang BF16 | 79 | 79 | 79 | 75 |
| SGLang BF16 + DFlash | 196 | 295 | 222 | 157 |
| nano BF16 (no speculation) | 83 | 83 | 83 | 79 |
| vLLM AWQ | 175 | 174 | 174 | 150 |
| vLLM AWQ + DFlash | 380 | 589 | 434 | 245 |
| nano AWQ (no speculation) | 207 | 206 | 205 | 178 |

Output tok/s, temperature 0, c=16 and c=64:

| server | alpaca c=16 | alpaca c=64 | humaneval c=16 | humaneval c=64 |
|---|---:|---:|---:|---:|
| vLLM BF16 + DFlash | 1910 | 3327 | 2806 | 4576 |
| SGLang BF16 + DFlash | 1652 | 2915 | 2456 | 4168 |
| vLLM AWQ | 2112 | 5092 | 2113 | 4563 |
| vLLM AWQ + DFlash | 2084 | 2943 | 2950 | 4204 |
| nano AWQ (no speculation) | 2567 | 6012 | 2516 | 5159 |

Tokens per cycle (acceptance study, greedy, block 10, 16 prompts per
dataset, 8 for long):

| target / drafter | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| BF16 / BF16 | 3.56 | 5.21 | 4.14 | 3.05 |
| BF16 / 4-bit RTN | 3.49 | 5.13 | 4.16 | 2.97 |
| AWQ / BF16 | 3.47 | 5.12 | 4.08 | 2.83 |
| AWQ / 4-bit RTN | 3.45 | 5.08 | 4.03 | 2.82 |

M0 gate (projected nano c=1 at least 1.2x the best DFlash baseline, 4-bit
drafter; the cycle is an estimate until M3 measures it):

| target | alpaca | humaneval | gsm8k | long | gate |
|---|---:|---:|---:|---:|---|
| AWQ | 1.40x | 1.33x | 1.43x | 1.76x | met |
| BF16 | 1.11x | 1.12x | 1.14x | 1.29x | not met |

On the BF16 target the verify pass (12 ms, the same weights read by every
engine) is 84% of the cycle, so the projected margin is what the draft and
the cycle overhead leave: about 1.3 ms of vLLM's 15.5 ms.

### M1 and M2 on the GPU (runs `m2-smoke-20261112-041533-colab`, `m2-20261112-043725-colab`)

| check | result |
|---|---|
| drafter parity with the checkpoint's code (top-1, 200 blocks, gate 99%) | 98.89%: not met; every disagreement is within one bf16 step of a tie |
| tests/gpu + nano unit tests | 1039 passed, none skipped |
| emitted tokens that are not the teacher-forced argmax, with the drafter | 0.14-0.40%, largest margin 0.25 (without: 0.17-0.43%, 0.25) |
| tokens per cycle against vLLM + DFlash on the same prompts (c=1) | AWQ 3.39 / 5.36 / 3.98 / 3.04 vs 3.38 / 5.44 / 3.90 / 3.04; BF16 3.48 / 5.35 / 4.03 / 3.04 vs 3.53 / 5.28 / 4.15 / 3.03 |
| TTFT with the drafter | at most 2.6% higher |

Output tok/s at c=1 after M2 (the cycle is not optimized yet: about 16 ms on
AWQ, 24 ms on BF16):

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano AWQ | 207 | 205 | 205 | 183 |
| nano AWQ + DFlash (M2) | 207 | 316 | 238 | 144 |
| vLLM AWQ + DFlash | 380 | 589 | 434 | 245 |
| nano BF16 | 83 | 83 | 83 | 80 |
| nano BF16 + DFlash (M2) | 142 | 212 | 164 | 108 |
| vLLM BF16 + DFlash | 220 | 322 | 256 | 161 |
