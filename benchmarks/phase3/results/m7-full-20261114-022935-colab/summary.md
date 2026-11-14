# Phase 3 run m7-full-20261114-022935-colab

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
| install_venv | 0 | 52 |
| install_study | 0 | 19 |
| install_vllm_spec | 0 | 44 |
| install_sglang_spec | 0 | 33 |
| download_model | 0 | 45 |
| download_w4_model | 0 | 16 |
| download_drafter | 0 | 7 |
| cuda129 | 0 | 43 |
| workloads | 0 | 45 |
| tests | 0 | 1049 |
| spec_check | 0 | 309 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle). rows by length: decode rows per verify length of their step (1: no drafts).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean | rows by length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 | 5.00 | 5.01 | - |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1964 | 6.30 | 5.96 | - |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.01 | 474 | 7.72 | 7.77 | 10:748 |
| awq spec b=16 | 3072 | 9 | 0.29% | 0.250 | 0 | 0 | 3.12 | 2749 | 13.47 | 12.13 | 1:64 4:894 6:49 10:23 |
| awq spec, async b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 490 | 7.34 | 7.24 | 10:759 |
| awq spec, async b=16 | 3048 | 13 | 0.43% | 0.250 | 0 | 0 | 3.09 | 2877 | 12.61 | 11.49 | 4:915 6:80 10:6 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 4.99 | 5.01 | 4.99 | 5.05 |
| awq spec | 7.71 | 7.72 | 7.71 | 7.85 |
| awq spec, async | 7.33 | 7.34 | 7.33 | 7.46 |

## Step time of each CUDA graph (ms, empty context)

One replay of the graph for that many sequences and verify length Q (1: a decode step with its sampling; with a drafter it also writes the drafter's context), on the null block.

**awq plain**

| sequences | Q=1 |
|---|---:|
| 1 | 4.62 |
| 2 | 4.68 |
| 4 | 4.70 |
| 8 | 4.76 |
| 16 | 4.88 |

**awq spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.63 | 6.91 | 6.94 | 6.98 | 7.07 | 7.20 |
| 2 | 4.69 | 7.00 | 7.06 | 7.11 | 7.23 | 8.21 |
| 4 | 4.72 | 7.00 | 7.08 | 7.14 | 8.02 | 9.27 |
| 8 | 4.79 | 7.12 | 7.85 | 8.11 | 9.34 | 14.02 |
| 16 | 4.89 | 7.92 | 9.24 | 10.28 | 14.06 | 22.45 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.21x | 1.59x | 1.85x | 2.12x | 2.26x |
| 2 | 1.00x | 1.21x | 1.59x | 1.84x | 2.10x | 2.01x |
| 4 | 1.00x | 1.22x | 1.59x | 1.84x | 1.90x | 1.79x |
| 8 | 1.00x | 1.22x | 1.46x | 1.65x | 1.66x | 1.20x |
| 16 | 1.00x | 1.12x | 1.27x | 1.33x | 1.13x | 0.77x |

**awq spec, async**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.63 | 6.91 | 6.95 | 6.98 | 7.07 | 7.20 |
| 2 | 4.69 | 7.00 | 7.07 | 7.11 | 7.23 | 8.16 |
| 4 | 4.72 | 7.01 | 7.08 | 7.14 | 8.02 | 9.26 |
| 8 | 4.78 | 7.11 | 7.85 | 8.09 | 9.33 | 14.00 |
| 16 | 4.89 | 7.91 | 9.23 | 10.26 | 14.05 | 22.38 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.21x | 1.59x | 1.85x | 2.12x | 2.26x |
| 2 | 1.00x | 1.21x | 1.59x | 1.84x | 2.10x | 2.02x |
| 4 | 1.00x | 1.22x | 1.59x | 1.84x | 1.91x | 1.79x |
| 8 | 1.00x | 1.22x | 1.46x | 1.65x | 1.66x | 1.20x |
| 16 | 1.00x | 1.12x | 1.27x | 1.33x | 1.13x | 0.77x |

## M4 and M5 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.72 ms (met); spec, async: 7.34 ms (met).

## M7: cycles launched behind the cycle in flight

The engine core in one process (spec_check.py), each cycle collected before the next (sync) and launched behind the one in flight (async). Same tokens: requests whose tokens are the sync run's (greedy streams can split at a near-tie when a discarded row changes a step's size). Discarded: rows that landed after their sequence had ended.

| target, drafter, batch | sync tok/s | async tok/s | async / sync | sync ms/step | async ms/step | same tokens | discarded |
|---|---:|---:|---:|---:|---:|---:|---:|
| awq spec b=1 | 474 | 490 | 1.033x | 7.72 | 7.34 | 16 of 16 | 10 |
| awq spec b=16 | 2749 | 2877 | 1.047x | 13.47 | 12.61 | 6 of 16 | 13 |

