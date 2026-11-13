# Phase 3 run m6-checks-20261113-032556-colab

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
| install_venv | 0 | 331 |
| install_study | 0 | 100 |
| install_vllm_spec | 0 | 258 |
| install_sglang_spec | 0 | 80 |
| download_model | 0 | 43 |
| download_w4_model | 0 | 15 |
| download_drafter | 0 | 5 |
| cuda129 | 0 | 25 |
| workloads | 0 | 30 |
| tests | 0 | 1055 |
| spec_check | 0 | 429 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle). rows by length: decode rows per verify length of their step (1: no drafts).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean | rows by length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 191 | 5.03 | 5.05 | - |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1954 | 6.32 | 6.00 | - |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 469 | 7.83 | 7.84 | 10:749 |
| awq spec b=16 | 3059 | 9 | 0.29% | 0.250 | 0 | 0 | 3.11 | 2752 | 13.71 | 12.63 | 4:892 6:83 10:8 |
| awq spec, w4a16 drafter b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 3.94 | 522 | 6.83 | 6.85 | 10:764 |
| awq spec, w4a16 drafter b=16 | 3073 | 12 | 0.39% | 0.250 | 0 | 0 | 3.09 | 2884 | 13.27 | 12.09 | 3:16 4:895 6:81 10:3 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 80 | 12.37 | 12.36 | - |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 862 | 13.70 | 13.42 | - |
| bf16 spec b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 244 | 15.60 | 15.61 | 10:753 |
| bf16 spec b=16 | 2931 | 7 | 0.24% | 0.250 | 0 | 0 | 3.87 | 1687 | 23.89 | 22.79 | 6:326 10:430 |
| bf16 spec, w4a16 drafter b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 260 | 14.64 | 14.65 | 10:752 |
| bf16 spec, w4a16 drafter b=16 | 2967 | 11 | 0.37% | 0.250 | 0 | 0 | 3.66 | 1745 | 23.29 | 21.44 | 6:708 10:98 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 5.02 | 5.03 | 5.01 | 5.08 |
| awq spec | 7.77 | 7.80 | 7.79 | 7.91 |
| awq spec, w4a16 drafter | 6.77 | 6.79 | 6.78 | 6.92 |
| bf16 plain | 12.31 | 12.37 | 12.38 | 12.40 |
| bf16 spec | 15.54 | 15.60 | 15.57 | 15.71 |
| bf16 spec, w4a16 drafter | 14.59 | 14.65 | 14.60 | 14.72 |

## Step time of each CUDA graph (ms, empty context)

One replay of the graph for that many sequences and verify length Q (1: a decode step with its sampling; with a drafter it also writes the drafter's context), on the null block.

**awq plain**

| sequences | Q=1 |
|---|---:|
| 1 | 4.64 |
| 2 | 4.69 |
| 4 | 4.71 |
| 8 | 4.77 |
| 16 | 4.89 |

**awq spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.64 | 6.91 | 6.93 | 6.98 | 7.05 | 7.18 |
| 2 | 4.70 | 6.99 | 7.05 | 7.10 | 7.22 | 8.19 |
| 4 | 5.23 | 6.99 | 7.07 | 7.13 | 8.01 | 9.27 |
| 8 | 5.40 | 7.10 | 7.84 | 8.10 | 9.34 | 14.06 |
| 16 | 5.49 | 7.90 | 9.22 | 10.26 | 14.09 | 22.53 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.22x | 1.60x | 1.86x | 2.13x | 2.28x |
| 2 | 1.00x | 1.22x | 1.59x | 1.85x | 2.11x | 2.02x |
| 4 | 1.00x | 1.35x | 1.77x | 2.05x | 2.12x | 1.99x |
| 8 | 1.00x | 1.38x | 1.65x | 1.86x | 1.88x | 1.35x |
| 16 | 1.00x | 1.26x | 1.42x | 1.49x | 1.26x | 0.86x |

**awq spec, w4a16 drafter**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.22 | 5.93 | 5.95 | 5.98 | 6.07 | 6.21 |
| 2 | 5.29 | 6.01 | 6.07 | 6.12 | 6.24 | 7.21 |
| 4 | 5.28 | 6.02 | 6.11 | 6.16 | 7.07 | 8.51 |
| 8 | 5.36 | 6.13 | 6.90 | 7.15 | 8.60 | 13.53 |
| 16 | 5.51 | 6.97 | 8.48 | 9.70 | 13.55 | 22.55 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.59x | 2.09x | 2.43x | 2.79x | 2.96x |
| 2 | 1.00x | 1.60x | 2.08x | 2.41x | 2.75x | 2.59x |
| 4 | 1.00x | 1.59x | 2.07x | 2.39x | 2.42x | 2.19x |
| 8 | 1.00x | 1.58x | 1.86x | 2.09x | 2.02x | 1.40x |
| 16 | 1.00x | 1.43x | 1.55x | 1.58x | 1.32x | 0.86x |

**bf16 plain**

| sequences | Q=1 |
|---|---:|
| 1 | 11.83 |
| 2 | 11.90 |
| 4 | 11.96 |
| 8 | 12.07 |
| 16 | 12.24 |

**bf16 spec**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.85 | 14.46 | 14.51 | 14.56 | 14.66 | 14.84 |
| 2 | 11.91 | 14.57 | 14.67 | 14.75 | 14.90 | 16.76 |
| 4 | 11.97 | 14.61 | 14.74 | 14.83 | 16.61 | 16.17 |
| 8 | 12.09 | 14.81 | 16.52 | 16.66 | 18.36 | 19.89 |
| 16 | 12.68 | 16.53 | 18.21 | 18.43 | 20.11 | 23.11 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.48x | 1.95x | 2.27x | 2.62x | 2.81x |
| 2 | 1.00x | 1.48x | 1.94x | 2.25x | 2.59x | 2.50x |
| 4 | 1.00x | 1.48x | 1.94x | 2.25x | 2.33x | 2.61x |
| 8 | 1.00x | 1.48x | 1.75x | 2.02x | 2.13x | 2.14x |
| 16 | 1.00x | 1.39x | 1.66x | 1.92x | 2.04x | 1.93x |

**bf16 spec, w4a16 drafter**

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.85 | 13.48 | 13.52 | 13.57 | 13.70 | 13.87 |
| 2 | 11.90 | 13.60 | 13.69 | 13.78 | 13.94 | 15.82 |
| 4 | 11.97 | 13.65 | 13.76 | 13.88 | 15.67 | 15.42 |
| 8 | 12.10 | 13.84 | 15.59 | 15.72 | 17.62 | 19.39 |
| 16 | 12.76 | 15.60 | 17.48 | 17.88 | 19.59 | 23.20 |

Tokens per ms over a plain step's, at the acceptance of alpaca (M0 study):

| sequences | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.00x | 1.59x | 2.09x | 2.44x | 2.80x | 3.01x |
| 2 | 1.00x | 1.58x | 2.08x | 2.41x | 2.77x | 2.65x |
| 4 | 1.00x | 1.59x | 2.08x | 2.41x | 2.47x | 2.73x |
| 8 | 1.00x | 1.58x | 1.86x | 2.15x | 2.22x | 2.20x |
| 16 | 1.00x | 1.48x | 1.74x | 1.99x | 2.11x | 1.94x |

## M4 and M5 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.83 ms (met); spec, w4a16 drafter: 6.83 ms (met).

