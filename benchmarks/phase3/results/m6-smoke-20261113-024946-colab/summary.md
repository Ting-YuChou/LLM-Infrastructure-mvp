# Phase 3 run m6-smoke-20261113-024946-colab

```
load generator / nano: vllm 0.19.1
vllm baseline: 0.33.0 | torch 2.14.0
sglang baseline: 0.5.23 | torch 2.14.1
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 53 |
| install_study | 0 | 16 |
| install_vllm_spec | 0 | 658 |
| install_sglang_spec | 0 | 31 |
| download_model | 0 | 45 |
| download_w4_model | 0 | 20 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 53 |
| workloads | 0 | 45 |
| tests | 0 | 309 |
| spec_check | 0 | 285 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle). rows by length: decode rows per verify length of their step (1: no drafts).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean | rows by length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 194 | 4.99 | 4.99 | - |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1973 | 6.28 | 5.94 | - |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 473 | 7.74 | 7.78 | 10:749 |
| awq spec b=16 | 3059 | 9 | 0.29% | 0.250 | 0 | 0 | 3.11 | 2760 | 13.68 | 12.59 | 4:892 6:83 10:8 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 4.98 | 4.99 | 4.98 | 5.03 |
| awq spec | 7.73 | 7.74 | 7.71 | 7.88 |

## Step time of each CUDA graph (ms, empty context)

One replay of the graph for that many sequences and verify length Q (1: a decode step with its sampling; with a drafter it also writes the drafter's context), on the null block.

**awq plain**

| sequences | Q=1 |
|---|---:|
| 1 | 4.62 |
| 2 | 4.69 |
| 4 | 4.71 |
| 8 | 4.77 |
| 16 | 4.88 |

**awq spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.35 | 6.90 | 6.93 | 6.96 | 7.05 | 7.17 |
| 2 | 5.59 | 6.98 | 7.05 | 7.09 | 7.21 | 8.20 |
| 4 | 5.57 | 6.99 | 7.07 | 7.13 | 8.02 | 9.27 |
| 8 | 5.65 | 7.09 | 7.84 | 8.10 | 9.35 | 14.02 |
| 16 | 5.75 | 7.91 | 9.23 | 10.27 | 14.11 | 22.56 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.40x | 1.84x | 2.14x | 2.46x | 2.62x |
| 2 | 1.00x | 1.45x | 1.90x | 2.20x | 2.51x | 2.40x |
| 4 | 1.00x | 1.44x | 1.88x | 2.18x | 2.25x | 2.12x |
| 8 | 1.00x | 1.44x | 1.72x | 1.95x | 1.96x | 1.42x |
| 16 | 1.00x | 1.32x | 1.49x | 1.56x | 1.32x | 0.90x |

## M4 and M5 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.74 ms (met).

