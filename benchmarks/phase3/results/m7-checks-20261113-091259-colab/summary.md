# Phase 3 run m7-checks-20261113-091259-colab

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
| tests | 0 | 971 |
| spec_check | 0 | 486 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle). rows by length: decode rows per verify length of their step (1: no drafts).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean | rows by length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 | 5.00 | 5.01 | - |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1964 | 6.29 | 5.97 | - |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.01 | 475 | 7.72 | 7.76 | 10:748 |
| awq spec b=16 | 3072 | 9 | 0.29% | 0.250 | 0 | 0 | 3.12 | 2761 | 13.31 | 12.09 | 1:64 4:894 6:49 10:23 |
| awq spec, async b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 490 | 7.34 | 7.23 | 10:759 |
| awq spec, async b=16 | 3048 | 13 | 0.43% | 0.250 | 0 | 0 | 3.09 | 2876 | 12.63 | 11.50 | 4:915 6:80 10:6 |
| awq spec, w4a16 drafter b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 3.94 | 526 | 6.75 | 6.80 | 10:764 |
| awq spec, w4a16 drafter b=16 | 3073 | 12 | 0.39% | 0.250 | 0 | 0 | 3.10 | 2944 | 12.87 | 11.85 | 3:16 4:895 6:71 10:12 |
| awq spec, w4a16 drafter, async b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 3.94 | 547 | 6.37 | 6.29 | 10:774 |
| awq spec, w4a16 drafter, async b=16 | 3068 | 12 | 0.39% | 0.250 | 0 | 0 | 3.05 | 3072 | 12.09 | 10.94 | 3:32 4:893 6:96 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 81 | 12.23 | 12.23 | - |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 869 | 13.61 | 13.31 | - |
| bf16 spec b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.99 | 247 | 15.42 | 15.45 | 10:752 |
| bf16 spec b=16 | 2938 | 9 | 0.31% | 0.250 | 0 | 0 | 3.87 | 1721 | 23.57 | 22.69 | 6:295 10:462 |
| bf16 spec, async b=1 | 3005 | 4 | 0.13% | 0.250 | 0 | 0 | 4.00 | 251 | 15.01 | 14.75 | 10:757 |
| bf16 spec, async b=16 | 2940 | 9 | 0.31% | 0.250 | 0 | 0 | 3.90 | 1774 | 22.74 | 21.17 | 6:420 10:347 |
| bf16 spec, w4a16 drafter b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 263 | 14.43 | 14.47 | 10:753 |
| bf16 spec, w4a16 drafter b=16 | 2972 | 8 | 0.27% | 0.250 | 0 | 0 | 3.59 | 1781 | 23.00 | 21.03 | 6:722 10:102 |
| bf16 spec, w4a16 drafter, async b=1 | 3005 | 4 | 0.13% | 0.250 | 0 | 0 | 3.99 | 267 | 14.04 | 13.80 | 10:759 |
| bf16 spec, w4a16 drafter, async b=16 | 2969 | 8 | 0.27% | 0.250 | 0 | 0 | 3.62 | 1848 | 22.21 | 20.24 | 6:723 10:106 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 5.00 | 5.00 | 5.00 | 5.06 |
| awq spec | 7.71 | 7.72 | 7.71 | 7.84 |
| awq spec, async | 7.32 | 7.33 | 7.32 | 7.45 |
| awq spec, w4a16 drafter | 6.74 | 6.75 | 6.74 | 6.89 |
| awq spec, w4a16 drafter, async | 6.36 | 6.37 | 6.36 | 6.48 |
| bf16 plain | 12.22 | 12.22 | 12.22 | 12.27 |
| bf16 spec | 15.40 | 15.41 | 15.39 | 15.52 |
| bf16 spec, async | 14.99 | 15.01 | 15.00 | 15.13 |
| bf16 spec, w4a16 drafter | 14.42 | 14.43 | 14.41 | 14.55 |
| bf16 spec, w4a16 drafter, async | 14.03 | 14.04 | 14.03 | 14.16 |

## Step time of each CUDA graph (ms, empty context)

One replay of the graph for that many sequences and verify length Q (1: a decode step with its sampling; with a drafter it also writes the drafter's context), on the null block.

**awq plain**

| sequences | Q=1 |
|---|---:|
| 1 | 4.63 |
| 2 | 4.67 |
| 4 | 4.70 |
| 8 | 4.76 |
| 16 | 4.88 |

**awq spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.63 | 6.92 | 6.94 | 6.98 | 7.07 | 7.19 |
| 2 | 4.69 | 7.01 | 7.07 | 7.11 | 7.23 | 8.21 |
| 4 | 4.71 | 7.00 | 7.09 | 7.16 | 8.02 | 9.26 |
| 8 | 4.78 | 7.11 | 7.85 | 8.10 | 9.33 | 14.01 |
| 16 | 4.89 | 7.91 | 9.22 | 10.26 | 14.04 | 22.43 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.21x | 1.59x | 1.85x | 2.12x | 2.26x |
| 2 | 1.00x | 1.21x | 1.58x | 1.84x | 2.10x | 2.01x |
| 4 | 1.00x | 1.22x | 1.59x | 1.84x | 1.90x | 1.79x |
| 8 | 1.00x | 1.22x | 1.46x | 1.65x | 1.66x | 1.20x |
| 16 | 1.00x | 1.12x | 1.27x | 1.33x | 1.13x | 0.77x |

**awq spec, async**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.62 | 6.92 | 6.94 | 6.99 | 7.07 | 7.19 |
| 2 | 4.68 | 7.01 | 7.06 | 7.11 | 7.23 | 8.17 |
| 4 | 4.71 | 7.01 | 7.09 | 7.15 | 8.02 | 9.27 |
| 8 | 4.78 | 7.12 | 7.85 | 8.10 | 9.33 | 14.01 |
| 16 | 4.89 | 7.91 | 9.21 | 10.27 | 14.07 | 22.45 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.21x | 1.59x | 1.85x | 2.12x | 2.26x |
| 2 | 1.00x | 1.21x | 1.59x | 1.84x | 2.10x | 2.02x |
| 4 | 1.00x | 1.22x | 1.59x | 1.84x | 1.90x | 1.79x |
| 8 | 1.00x | 1.22x | 1.46x | 1.65x | 1.66x | 1.20x |
| 16 | 1.00x | 1.12x | 1.27x | 1.33x | 1.13x | 0.77x |

**awq spec, w4a16 drafter**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.63 | 5.93 | 5.96 | 6.00 | 6.08 | 6.22 |
| 2 | 4.69 | 6.02 | 6.09 | 6.13 | 6.25 | 7.22 |
| 4 | 4.71 | 6.03 | 6.11 | 6.18 | 7.08 | 8.50 |
| 8 | 4.78 | 6.14 | 6.90 | 7.16 | 8.57 | 13.50 |
| 16 | 4.89 | 6.97 | 8.46 | 9.69 | 13.50 | 22.44 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.41x | 1.86x | 2.16x | 2.47x | 2.62x |
| 2 | 1.00x | 1.41x | 1.84x | 2.13x | 2.43x | 2.28x |
| 4 | 1.00x | 1.42x | 1.84x | 2.13x | 2.16x | 1.95x |
| 8 | 1.00x | 1.41x | 1.66x | 1.86x | 1.81x | 1.25x |
| 16 | 1.00x | 1.27x | 1.38x | 1.41x | 1.17x | 0.77x |

**awq spec, w4a16 drafter, async**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.63 | 5.93 | 5.96 | 6.00 | 6.09 | 6.22 |
| 2 | 4.69 | 6.02 | 6.08 | 6.14 | 6.25 | 7.22 |
| 4 | 4.71 | 6.03 | 6.12 | 6.18 | 7.08 | 8.51 |
| 8 | 4.78 | 6.15 | 6.91 | 7.16 | 8.58 | 13.50 |
| 16 | 4.89 | 6.98 | 8.47 | 9.69 | 13.51 | 22.45 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.41x | 1.86x | 2.16x | 2.47x | 2.62x |
| 2 | 1.00x | 1.41x | 1.84x | 2.13x | 2.43x | 2.29x |
| 4 | 1.00x | 1.41x | 1.84x | 2.13x | 2.15x | 1.95x |
| 8 | 1.00x | 1.41x | 1.65x | 1.86x | 1.81x | 1.25x |
| 16 | 1.00x | 1.27x | 1.38x | 1.41x | 1.17x | 0.77x |

**bf16 plain**

| sequences | Q=1 |
|---|---:|
| 1 | 11.83 |
| 2 | 11.88 |
| 4 | 11.94 |
| 8 | 12.04 |
| 16 | 12.22 |

**bf16 spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.83 | 14.46 | 14.52 | 14.56 | 14.67 | 14.85 |
| 2 | 11.89 | 14.58 | 14.67 | 14.75 | 14.92 | 16.76 |
| 4 | 11.95 | 14.62 | 14.73 | 14.85 | 16.61 | 16.17 |
| 8 | 12.06 | 14.83 | 16.52 | 16.65 | 18.35 | 19.87 |
| 16 | 12.25 | 16.53 | 18.19 | 18.42 | 20.09 | 23.04 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.48x | 1.95x | 2.27x | 2.61x | 2.80x |
| 2 | 1.00x | 1.48x | 1.94x | 2.25x | 2.58x | 2.50x |
| 4 | 1.00x | 1.48x | 1.94x | 2.25x | 2.33x | 2.60x |
| 8 | 1.00x | 1.47x | 1.75x | 2.02x | 2.13x | 2.14x |
| 16 | 1.00x | 1.34x | 1.61x | 1.86x | 1.98x | 1.87x |

**bf16 spec, async**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.84 | 14.46 | 14.52 | 14.55 | 14.68 | 14.86 |
| 2 | 11.88 | 14.57 | 14.67 | 14.75 | 14.92 | 16.77 |
| 4 | 11.95 | 14.62 | 14.73 | 14.86 | 16.62 | 16.17 |
| 8 | 12.07 | 14.83 | 16.52 | 16.66 | 18.36 | 19.87 |
| 16 | 12.24 | 16.53 | 18.20 | 18.40 | 20.12 | 23.07 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.48x | 1.95x | 2.27x | 2.61x | 2.80x |
| 2 | 1.00x | 1.48x | 1.94x | 2.25x | 2.58x | 2.49x |
| 4 | 1.00x | 1.48x | 1.94x | 2.24x | 2.33x | 2.60x |
| 8 | 1.00x | 1.47x | 1.75x | 2.02x | 2.13x | 2.14x |
| 16 | 1.00x | 1.34x | 1.61x | 1.86x | 1.97x | 1.87x |

**bf16 spec, w4a16 drafter**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.83 | 13.48 | 13.55 | 13.58 | 13.70 | 13.88 |
| 2 | 11.88 | 13.60 | 13.69 | 13.78 | 13.94 | 15.82 |
| 4 | 11.94 | 13.65 | 13.78 | 13.89 | 15.67 | 15.42 |
| 8 | 12.07 | 13.87 | 15.58 | 15.73 | 17.63 | 19.38 |
| 16 | 12.24 | 15.61 | 17.47 | 17.85 | 19.59 | 23.13 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.59x | 2.09x | 2.43x | 2.80x | 3.00x |
| 2 | 1.00x | 1.58x | 2.07x | 2.41x | 2.76x | 2.64x |
| 4 | 1.00x | 1.58x | 2.07x | 2.40x | 2.47x | 2.73x |
| 8 | 1.00x | 1.57x | 1.85x | 2.14x | 2.22x | 2.19x |
| 16 | 1.00x | 1.42x | 1.67x | 1.91x | 2.02x | 1.86x |

**bf16 spec, w4a16 drafter, async**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.83 | 13.49 | 13.54 | 13.59 | 13.71 | 13.89 |
| 2 | 11.88 | 13.61 | 13.70 | 13.79 | 13.96 | 15.83 |
| 4 | 11.95 | 13.66 | 13.77 | 13.89 | 15.68 | 15.42 |
| 8 | 12.06 | 13.87 | 15.58 | 15.73 | 17.62 | 19.38 |
| 16 | 12.25 | 15.61 | 17.47 | 17.86 | 19.58 | 23.15 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.59x | 2.09x | 2.43x | 2.80x | 3.00x |
| 2 | 1.00x | 1.58x | 2.07x | 2.40x | 2.76x | 2.64x |
| 4 | 1.00x | 1.58x | 2.07x | 2.40x | 2.47x | 2.73x |
| 8 | 1.00x | 1.57x | 1.85x | 2.14x | 2.22x | 2.19x |
| 16 | 1.00x | 1.42x | 1.68x | 1.91x | 2.03x | 1.86x |

## M4 and M5 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.72 ms (met); spec, async: 7.34 ms (met); spec, w4a16 drafter: 6.75 ms (met); spec, w4a16 drafter, async: 6.37 ms (met).

## M7: cycles launched behind the cycle in flight

The engine core in one process (spec_check.py), each cycle collected before the next (sync) and launched behind the one in flight (async). Same tokens: requests whose tokens are the sync run's (greedy streams can split at a near-tie when a discarded row changes a step's size). Discarded: rows that landed after their sequence had ended.

| target, drafter, batch | sync tok/s | async tok/s | async / sync | sync ms/step | async ms/step | same tokens | discarded |
|---|---:|---:|---:|---:|---:|---:|---:|
| awq spec b=1 | 475 | 490 | 1.033x | 7.72 | 7.34 | 16 of 16 | 10 |
| awq spec b=16 | 2761 | 2876 | 1.042x | 13.31 | 12.63 | 6 of 16 | 13 |
| awq spec, w4a16 drafter b=1 | 526 | 547 | 1.041x | 6.75 | 6.37 | 16 of 16 | 10 |
| awq spec, w4a16 drafter b=16 | 2944 | 3072 | 1.044x | 12.87 | 12.09 | 9 of 16 | 14 |
| bf16 spec b=1 | 247 | 251 | 1.017x | 15.42 | 15.01 | 15 of 16 | 9 |
| bf16 spec b=16 | 1721 | 1774 | 1.030x | 23.57 | 22.74 | 9 of 16 | 14 |
| bf16 spec, w4a16 drafter b=1 | 263 | 267 | 1.016x | 14.43 | 14.04 | 15 of 16 | 9 |
| bf16 spec, w4a16 drafter b=16 | 1781 | 1848 | 1.037x | 23.00 | 22.21 | 8 of 16 | 13 |

