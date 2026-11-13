# Phase 3 run m6-checks-20261113-044714-colab

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
| install_venv | 0 | 306 |
| install_study | 0 | 29 |
| install_vllm_spec | 0 | 208 |
| install_sglang_spec | 0 | 93 |
| download_model | 0 | 43 |
| download_w4_model | 0 | 15 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 24 |
| workloads | 0 | 30 |
| tests | 1 | 993 |
| spec_check | 0 | 427 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle). rows by length: decode rows per verify length of their step (1: no drafts).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean | rows by length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 | 5.00 | 5.00 | - |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1968 | 6.28 | 5.96 | - |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.01 | 475 | 7.71 | 7.75 | 10:748 |
| awq spec b=16 | 3072 | 9 | 0.29% | 0.250 | 0 | 0 | 3.12 | 2750 | 13.52 | 12.13 | 1:64 4:894 6:49 10:23 |
| awq spec, w4a16 drafter b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 3.94 | 527 | 6.74 | 6.78 | 10:764 |
| awq spec, w4a16 drafter b=16 | 3073 | 12 | 0.39% | 0.250 | 0 | 0 | 3.10 | 2931 | 12.99 | 11.90 | 3:16 4:895 6:69 10:14 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 81 | 12.21 | 12.22 | - |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 870 | 13.59 | 13.30 | - |
| bf16 spec b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.99 | 248 | 15.39 | 15.42 | 10:752 |
| bf16 spec b=16 | 2926 | 8 | 0.27% | 0.250 | 0 | 0 | 3.87 | 1695 | 23.60 | 22.35 | 6:325 10:430 |
| bf16 spec, w4a16 drafter b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 263 | 14.42 | 14.46 | 10:753 |
| bf16 spec, w4a16 drafter b=16 | 2972 | 8 | 0.27% | 0.250 | 0 | 0 | 3.59 | 1782 | 23.02 | 21.03 | 6:722 10:102 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 4.99 | 5.00 | 4.99 | 5.05 |
| awq spec | 7.69 | 7.71 | 7.70 | 7.83 |
| awq spec, w4a16 drafter | 6.73 | 6.74 | 6.72 | 6.85 |
| bf16 plain | 12.21 | 12.21 | 12.20 | 12.26 |
| bf16 spec | 15.37 | 15.38 | 15.37 | 15.50 |
| bf16 spec, w4a16 drafter | 14.41 | 14.41 | 14.40 | 14.55 |

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
| 1 | 4.63 | 6.88 | 6.91 | 6.95 | 7.04 | 7.17 |
| 2 | 4.69 | 6.97 | 7.04 | 7.08 | 7.19 | 8.17 |
| 4 | 4.71 | 6.97 | 7.06 | 7.12 | 7.99 | 9.25 |
| 8 | 4.78 | 7.08 | 7.82 | 8.07 | 9.32 | 14.02 |
| 16 | 4.89 | 7.88 | 9.21 | 10.24 | 14.07 | 22.50 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.22x | 1.60x | 1.86x | 2.13x | 2.27x |
| 2 | 1.00x | 1.22x | 1.59x | 1.85x | 2.11x | 2.02x |
| 4 | 1.00x | 1.22x | 1.59x | 1.85x | 1.91x | 1.79x |
| 8 | 1.00x | 1.22x | 1.46x | 1.65x | 1.66x | 1.20x |
| 16 | 1.00x | 1.12x | 1.27x | 1.33x | 1.13x | 0.76x |

**awq spec, w4a16 drafter**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.63 | 5.91 | 5.94 | 5.98 | 6.06 | 6.20 |
| 2 | 4.69 | 6.00 | 6.06 | 6.12 | 6.23 | 7.19 |
| 4 | 4.71 | 6.00 | 6.10 | 6.16 | 7.06 | 8.48 |
| 8 | 4.78 | 6.13 | 6.89 | 7.15 | 8.57 | 13.51 |
| 16 | 4.89 | 6.96 | 8.45 | 9.68 | 13.53 | 22.50 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.42x | 1.86x | 2.16x | 2.47x | 2.63x |
| 2 | 1.00x | 1.42x | 1.85x | 2.14x | 2.44x | 2.30x |
| 4 | 1.00x | 1.42x | 1.85x | 2.13x | 2.16x | 1.96x |
| 8 | 1.00x | 1.41x | 1.66x | 1.87x | 1.81x | 1.25x |
| 16 | 1.00x | 1.27x | 1.38x | 1.41x | 1.17x | 0.77x |

**bf16 plain**

| sequences | Q=1 |
|---|---:|
| 1 | 11.83 |
| 2 | 11.89 |
| 4 | 11.94 |
| 8 | 12.05 |
| 16 | 12.23 |

**bf16 spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.84 | 14.44 | 14.49 | 14.54 | 14.66 | 14.82 |
| 2 | 11.89 | 14.56 | 14.64 | 14.73 | 14.90 | 16.74 |
| 4 | 11.95 | 14.60 | 14.72 | 14.83 | 16.59 | 16.17 |
| 8 | 12.07 | 14.80 | 16.49 | 16.63 | 18.35 | 19.87 |
| 16 | 12.24 | 16.50 | 18.19 | 18.40 | 20.11 | 23.09 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.48x | 1.95x | 2.27x | 2.62x | 2.81x |
| 2 | 1.00x | 1.48x | 1.94x | 2.25x | 2.59x | 2.50x |
| 4 | 1.00x | 1.48x | 1.94x | 2.25x | 2.33x | 2.60x |
| 8 | 1.00x | 1.48x | 1.75x | 2.02x | 2.13x | 2.14x |
| 16 | 1.00x | 1.34x | 1.61x | 1.86x | 1.97x | 1.87x |

**bf16 spec, w4a16 drafter**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.83 | 13.47 | 13.51 | 13.56 | 13.67 | 13.85 |
| 2 | 11.89 | 13.59 | 13.67 | 13.76 | 13.91 | 15.80 |
| 4 | 11.95 | 13.63 | 13.75 | 13.86 | 15.65 | 15.40 |
| 8 | 12.07 | 13.82 | 15.57 | 15.71 | 17.58 | 19.38 |
| 16 | 12.24 | 15.58 | 17.44 | 17.84 | 19.57 | 23.21 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.59x | 2.09x | 2.43x | 2.80x | 3.01x |
| 2 | 1.00x | 1.58x | 2.08x | 2.41x | 2.77x | 2.65x |
| 4 | 1.00x | 1.59x | 2.08x | 2.41x | 2.47x | 2.73x |
| 8 | 1.00x | 1.58x | 1.85x | 2.14x | 2.22x | 2.19x |
| 16 | 1.00x | 1.42x | 1.68x | 1.91x | 2.03x | 1.86x |

## M4 and M5 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.71 ms (met); spec, w4a16 drafter: 6.74 ms (met).

