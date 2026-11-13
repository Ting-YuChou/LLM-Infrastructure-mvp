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
| M5 | sampled requests: drafts drawn with the target's noise (coupled sampling), a keyed PyTorch reference of the sampler |

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

### M3 (run `m3-20261112-085116-colab`, repeated as `m3-20261112-092848-colab`)

Verify attention for a block of 10 against the attention of a plain decode
step at the same context (microseconds per call, bf16; the gate is 1.3x for
one sequence and 1.5x for 64):

| sequences, context | decode | vLLM flash-attn | prefill kernel (M2) | block kernel | block / decode |
|---|---:|---:|---:|---:|---:|
| 1, 256 | 16.7 | 11.4 | 20.7 | 13.3 | 0.79 |
| 1, 1024 | 14.4 | 29.6 | 65.3 | 14.6 | 1.02 |
| 1, 4096 | 20.8 | 103.8 | 243.2 | 26.4 | 1.27 |
| 16, 1024 | 61.5 | 115.0 | 229.6 | 75.6 | 1.23 |
| 64, 256 | 57.8 | 126.1 | 240.3 | 76.2 | 1.32 |
| 64, 1024 | 198.7 | 477.4 | 835.2 | 248.5 | 1.25 |
| 64, 4096 | 767.1 | 2499.6 | 3248.2 | 926.6 | 1.21 |

| check | result |
|---|---|
| gate G2 | met on all six shapes |
| drafter norm + RoPE + KV write, one kernel | 5-17 us against 44-206 us in PyTorch |
| tests/gpu + nano unit tests | 1287 passed, none skipped |
| emitted tokens that are not the teacher-forced argmax, with the drafter | 0.17-0.50%, largest margin 0.25 (without: 0.17-0.43%, 0.25) |
| nano AWQ without a drafter, c=1 | 206 / 204 / 204 / 177 (M0: 207 / 206 / 205 / 178) |

Output tok/s at c=1 after M3 (a cycle is about 14 ms on AWQ and 20 ms on
BF16; the verify pass is about 5 ms of the AWQ cycle, the rest is M4's):

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano AWQ + DFlash (M2) | 207 | 316 | 238 | 144 |
| nano AWQ + DFlash (M3) | 246 | 359 | 276 | 182 |
| vLLM AWQ + DFlash | 380 | 589 | 434 | 245 |
| nano BF16 + DFlash (M2) | 142 | 212 | 164 | 108 |
| nano BF16 + DFlash (M3) | 172 | 266 | 203 | 138 |
| vLLM BF16 + DFlash | 220 | 322 | 256 | 161 |

AWQ + DFlash at c=16 / c=64: alpaca 2055 / 2739 and humaneval 2898 / 4007,
against vLLM AWQ + DFlash's 2084 / 2943 and 2950 / 4204 and nano AWQ
without a drafter's 2567 / 6012 and 2516 / 5159. The block length is still
fixed at 10 (M6).

### M4 (run `m4-20261112-214147-colab`; baselines from the M0 run)

The whole cycle (draft, verify, the drafter's context writes, sampling,
acceptance) is one CUDA graph per number of sequences. A cycle as the host
sees it, one sequence (median ms):

| target | plain decode step | cycle, BF16 drafter | cycle, 4-bit drafter | M3 |
|---|---:|---:|---:|---:|
| AWQ | 5.00 | 7.75 | 6.68 | about 14 |
| BF16 | 12.29 | 15.45 | 14.50 | about 20 |

Output tok/s at c=1, temperature 0:

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano AWQ (no drafter) | 207 | 206 | 205 | 177 |
| vLLM AWQ + DFlash | 380 | 589 | 434 | 245 |
| nano AWQ + DFlash | 428 | 641 | 484 | 289 |
| nano AWQ + DFlash, 4-bit drafter | 477 | 714 | 547 | 316 |
| SGLang BF16 + DFlash | 196 | 295 | 222 | 157 |
| vLLM BF16 + DFlash | 220 | 322 | 256 | 161 |
| nano BF16 + DFlash | 216 | 333 | 254 | 170 |
| nano BF16 + DFlash, 4-bit drafter | 228 | 357 | 272 | 180 |

| gate | result |
|---|---|
| G3: an AWQ cycle at batch 1 within 8.0 ms | 7.75 ms (BF16 drafter), 6.68 ms (4-bit): met |
| G4: above vLLM's and SGLang's DFlash at c=1, chat and code, both targets | 4-bit drafter: met on every dataset and target (1.21-1.29x on AWQ, 1.03-1.12x on BF16). Same BF16 drafter: met on AWQ (1.09-1.18x); level with vLLM on BF16 (0.98-1.06x), above SGLang |
| tests/gpu + nano unit tests | 1303 passed, none skipped |
| emitted tokens that are not the teacher-forced argmax, with the drafter | 0.17-0.50%, largest margin 0.25 (without: 0.17-0.43%, 0.25) |
| tokens per cycle with the 4-bit drafter | 1.3-3.1% fewer on AWQ, under 1.5% fewer on BF16 |

Not yet: with a block of 10 for every sequence, AWQ + DFlash at c=16 / c=64
(alpaca 2229 / 2739, humaneval 2966 / 3946) is about vLLM AWQ + DFlash's and
below nano without a drafter (2567 / 6012, 2516 / 5159); sampled requests
draft greedily (M5); steps with a prompt row run outside the graph.

### M5 (runs `m5-20261112-231250-colab`, `m5-sweep-20261112-235755-colab`; baselines from the M0 run)

A sampled request was already lossless (a verify row draws with the noise
of its position); M5 draws the drafts with the same noise, at
`speculative_draft_temperature` times the request's temperature. Tokens per
cycle at c=1, temperature 0.7, top-p 0.9, one seed:

| target, dataset | 0 (argmax drafts) | 0.25 | 0.5 | 0.75 | 1 (coupled) |
|---|---:|---:|---:|---:|---:|
| AWQ, alpaca | 3.40 | 3.42 | 3.41 | 3.35 | 3.16 |
| AWQ, humaneval | 5.14 | 5.19 | 5.16 | 5.16 | 4.95 |
| BF16, alpaca | 3.47 | 3.46 | 3.44 | 3.39 | 3.35 |
| BF16, humaneval | 5.21 | 5.25 | 5.26 | 5.23 | 5.12 |

Output tok/s at c=1, temperature 0.7, top-p 0.9 (alpaca / humaneval):

| server | tok/s |
|---|---|
| vLLM AWQ + DFlash | 371 / 556 |
| nano AWQ + DFlash, drafts at 0.25 | 411 / 628 |
| nano AWQ + DFlash, 4-bit drafter, coupled | 448 / 684 |
| SGLang BF16 + DFlash | 186 / 287 |
| vLLM BF16 + DFlash | 208 / 310 |
| nano BF16 + DFlash, drafts at 0.25 | 219 / 327 |
| nano BF16 + DFlash, 4-bit drafter, coupled | 219 / 332 |

| gate | result |
|---|---|
| tokens per cycle at temperature 0.7 at least those of argmax drafts | coupled at the request's temperature: not met (2-7% fewer). At 0.25, the default: level (-0.3% to +1.0%) |
| sampled c=1 above both baselines | met in every configuration measured (1.01x to 1.23x) |
| one seed, with and without a drafter: the same tokens (CPU) | met, for any draft temperature, eager steps and graphs |
| tests/gpu + nano unit tests | 1324 passed, none skipped |
