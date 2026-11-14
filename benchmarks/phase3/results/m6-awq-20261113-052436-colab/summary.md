# Phase 3 run m6-awq-20261113-052436-colab

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
| tests | 0 | 227 |
| serve | 0 | 1521 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq | 42 | 163 | 99 | 1223 |
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano_awq_dflash4 | 42 | 163 | 99 | 1223 |
| nano_awq_dflashf | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 207 | 753 | 2429 | 6001 |
| nano_awq_dflash | 423 | 1294 | 3045 | 5971 |
| nano_awq_dflash4 | 480 | 1491 | 3218 | 5786 |
| nano_awq_dflashf | 434 | 1243 | 2263 | 2744 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.79 | 4.93 | 5.33 | 9.44 |
| nano_awq_dflash | 2.24 | 2.77 | 4.21 | 9.62 |
| nano_awq_dflash4 | 1.95 | 2.46 | 4.14 | 9.95 |
| nano_awq_dflashf | 2.15 | 2.83 | 6.06 | 21.68 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 9.6 | 15.9 | 20.8 | 48.1 |
| nano_awq_dflash | 9.6 | 20.5 | 30.2 | 68.5 |
| nano_awq_dflash4 | 9.4 | 19.4 | 29.0 | 40.1 |
| nano_awq_dflashf | 9.7 | 22.3 | 47.9 | 166.0 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.04x | 1.72x | 1.25x | 0.99x |
| nano_awq_dflash4 | 2.31x | 1.98x | 1.32x | 0.96x |
| nano_awq_dflashf | 2.09x | 1.65x | 0.93x | 0.46x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 206 | 737 | 2532 | 5158 |
| nano_awq_dflash | 641 | 1786 | 3434 | 5523 |
| nano_awq_dflash4 | 716 | 1956 | 3529 | 5425 |
| nano_awq_dflashf | 645 | 1793 | 3030 | 3974 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.80 | 5.05 | 5.73 | 9.92 |
| nano_awq_dflash | 1.50 | 1.95 | 4.21 | 9.59 |
| nano_awq_dflash4 | 1.34 | 1.88 | 4.00 | 9.74 |
| nano_awq_dflashf | 1.48 | 1.97 | 4.67 | 13.88 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 16.5 | 25.2 | 155.1 | 123.1 |
| nano_awq_dflash | 16.5 | 32.2 | 53.3 | 129.2 |
| nano_awq_dflash4 | 16.6 | 30.4 | 58.2 | 132.8 |
| nano_awq_dflashf | 16.9 | 32.2 | 72.7 | 209.7 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.12x | 2.42x | 1.36x | 1.07x |
| nano_awq_dflash4 | 3.48x | 2.65x | 1.39x | 1.05x |
| nano_awq_dflashf | 3.13x | 2.43x | 1.20x | 0.77x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2209 |
| nano_awq_dflash | 486 | 3170 |
| nano_awq_dflash4 | 545 | 3254 |
| nano_awq_dflashf | 490 | 2439 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.80 | 5.75 |
| nano_awq_dflash | 2.01 | 4.37 |
| nano_awq_dflash4 | 1.77 | 4.25 |
| nano_awq_dflashf | 2.00 | 5.80 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 15.8 | 24.9 |
| nano_awq_dflash | 14.4 | 39.1 |
| nano_awq_dflash4 | 14.4 | 37.8 |
| nano_awq_dflashf | 14.8 | 57.0 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.36x | 1.44x |
| nano_awq_dflash4 | 2.65x | 1.47x |
| nano_awq_dflashf | 2.38x | 1.10x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 178 | 869 |
| nano_awq_dflash | 289 | 934 |
| nano_awq_dflash4 | 318 | 939 |
| nano_awq_dflashf | 292 | 805 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.85 | 15.79 |
| nano_awq_dflash | 2.64 | 14.22 |
| nano_awq_dflash4 | 2.32 | 14.17 |
| nano_awq_dflashf | 2.60 | 15.50 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 103.5 | 210.7 |
| nano_awq_dflash | 105.2 | 293.4 |
| nano_awq_dflash4 | 105.3 | 315.6 |
| nano_awq_dflashf | 105.4 | 257.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.63x | 1.08x |
| nano_awq_dflash4 | 1.79x | 1.08x |
| nano_awq_dflashf | 1.64x | 0.93x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2528 |
| nano_awq_dflash | 410 | 3042 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.81 | 5.39 |
| nano_awq_dflash | 2.38 | 4.52 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.9 | 24.7 |
| nano_awq_dflash | 9.7 | 29.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.99x | 1.20x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2775 |
| nano_awq_dflash | 630 | 3293 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.83 | 5.52 |
| nano_awq_dflash | 1.48 | 4.25 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.4 | 34.4 |
| nano_awq_dflash | 16.8 | 61.2 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.06x | 1.19x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1333 | 2.44 | 3.44 |
| nano_awq_dflash | alpaca_c4 | 1473 | 2.29 | 3.29 |
| nano_awq_dflash | alpaca_c16 | 3940 | 1.91 | 2.91 |
| nano_awq_dflash | alpaca_c64 | 1886 | 1.47 | 2.47 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1020 | 4.54 | 5.54 |
| nano_awq_dflash | humaneval_c16 | 3572 | 3.49 | 4.49 |
| nano_awq_dflash | humaneval_c64 | 11560 | 2.47 | 3.47 |
| nano_awq_dflash | gsm8k_c1 | 1145 | 2.96 | 3.96 |
| nano_awq_dflash | gsm8k_c16 | 3284 | 2.14 | 3.14 |
| nano_awq_dflash | long_c1 | 1110 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 3357 | 1.63 | 2.63 |
| nano_awq_dflash | alpaca_c1s | 1355 | 2.34 | 3.34 |
| nano_awq_dflash | alpaca_c16s | 4113 | 1.79 | 2.79 |
| nano_awq_dflash | humaneval_c1s | 1088 | 4.28 | 5.28 |
| nano_awq_dflash | humaneval_c16s | 3813 | 3.23 | 4.23 |
| nano_awq_dflash4 | alpaca_c1 | 1345 | 2.41 | 3.41 |
| nano_awq_dflash4 | alpaca_c4 | 1504 | 2.25 | 3.25 |
| nano_awq_dflash4 | alpaca_c16 | 3978 | 1.86 | 2.86 |
| nano_awq_dflash4 | alpaca_c64 | 8520 | 1.24 | 2.24 |
| nano_awq_dflash4 | humaneval_c1 | 1107 | 4.26 | 5.26 |
| nano_awq_dflash4 | humaneval_c4 | 1023 | 4.54 | 5.54 |
| nano_awq_dflash4 | humaneval_c16 | 3658 | 3.39 | 4.39 |
| nano_awq_dflash4 | humaneval_c64 | 11050 | 2.58 | 3.58 |
| nano_awq_dflash4 | gsm8k_c1 | 1161 | 2.91 | 3.91 |
| nano_awq_dflash4 | gsm8k_c16 | 3290 | 2.12 | 3.12 |
| nano_awq_dflash4 | long_c1 | 1126 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 3314 | 1.62 | 2.62 |
| nano_awq_dflashf | alpaca_c1 | 1309 | 2.50 | 3.50 |
| nano_awq_dflashf | alpaca_c4 | 1345 | 2.59 | 3.59 |
| nano_awq_dflashf | alpaca_c16 | 3189 | 2.67 | 3.67 |
| nano_awq_dflashf | alpaca_c64 | 14595 | 2.50 | 3.50 |
| nano_awq_dflashf | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflashf | humaneval_c4 | 1019 | 4.54 | 5.54 |
| nano_awq_dflashf | humaneval_c16 | 2915 | 4.58 | 5.58 |
| nano_awq_dflashf | humaneval_c64 | 7336 | 4.59 | 5.59 |
| nano_awq_dflashf | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflashf | gsm8k_c16 | 2506 | 3.18 | 4.18 |
| nano_awq_dflashf | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflashf | long_c16 | 2902 | 2.06 | 3.06 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.44 |
| nano_awq_dflash | alpaca_c4 | 0% | 0% | 0% | 0% | 97% | 3% | 3.29 |
| nano_awq_dflash | alpaca_c16 | 2% | 0% | 0% | 90% | 8% | 0% | 2.88 |
| nano_awq_dflash | alpaca_c64 | 96% | 2% | 0% | 2% | 0% | 0% | 1.06 |
| nano_awq_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.32 |
| nano_awq_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.54 |
| nano_awq_dflash | humaneval_c16 | 3% | 0% | 0% | 6% | 89% | 2% | 4.37 |
| nano_awq_dflash | humaneval_c64 | 5% | 1% | 0% | 91% | 3% | 0% | 3.34 |
| nano_awq_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.96 |
| nano_awq_dflash | gsm8k_c16 | 3% | 0% | 0% | 92% | 4% | 1% | 3.07 |
| nano_awq_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.02 |
| nano_awq_dflash | long_c16 | 4% | 0% | 0% | 93% | 3% | 0% | 2.57 |
| nano_awq_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.34 |
| nano_awq_dflash | alpaca_c16s | 3% | 0% | 0% | 93% | 4% | 0% | 2.73 |
| nano_awq_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.28 |
| nano_awq_dflash | humaneval_c16s | 2% | 0% | 0% | 15% | 81% | 2% | 4.17 |
| nano_awq_dflash4 | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.41 |
| nano_awq_dflash4 | alpaca_c4 | 0% | 0% | 0% | 15% | 83% | 2% | 3.25 |
| nano_awq_dflash4 | alpaca_c16 | 2% | 0% | 0% | 95% | 2% | 0% | 2.82 |
| nano_awq_dflash4 | alpaca_c64 | 79% | 9% | 10% | 2% | 0% | 0% | 1.26 |
| nano_awq_dflash4 | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.26 |
| nano_awq_dflash4 | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.54 |
| nano_awq_dflash4 | humaneval_c16 | 3% | 0% | 0% | 12% | 83% | 2% | 4.28 |
| nano_awq_dflash4 | humaneval_c64 | 11% | 0% | 0% | 80% | 9% | 0% | 3.30 |
| nano_awq_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.91 |
| nano_awq_dflash4 | gsm8k_c16 | 4% | 0% | 0% | 92% | 3% | 1% | 3.04 |
| nano_awq_dflash4 | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.98 |
| nano_awq_dflash4 | long_c16 | 5% | 0% | 0% | 93% | 1% | 0% | 2.53 |
| nano_awq_dflashf | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_awq_dflashf | alpaca_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 3.59 |
| nano_awq_dflashf | alpaca_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 3.67 |
| nano_awq_dflashf | alpaca_c64 | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_awq_dflashf | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.32 |
| nano_awq_dflashf | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.54 |
| nano_awq_dflashf | humaneval_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 5.58 |
| nano_awq_dflashf | humaneval_c64 | 0% | 0% | 0% | 0% | 0% | 100% | 5.59 |
| nano_awq_dflashf | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.97 |
| nano_awq_dflashf | gsm8k_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 4.18 |
| nano_awq_dflashf | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.02 |
| nano_awq_dflashf | long_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 3.06 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 423 | vllmw4_dflash 380 | 1.11x | met |
| alpaca / nano_awq_dflash4 | 480 | vllmw4_dflash 380 | 1.26x | met |
| alpaca / nano_awq_dflashf | 434 | vllmw4_dflash 380 | 1.14x | met |
| humaneval / nano_awq_dflash | 641 | vllmw4_dflash 589 | 1.09x | met |
| humaneval / nano_awq_dflash4 | 716 | vllmw4_dflash 589 | 1.21x | met |
| humaneval / nano_awq_dflashf | 645 | vllmw4_dflash 589 | 1.09x | met |
| gsm8k / nano_awq_dflash | 486 | vllmw4_dflash 434 | 1.12x | met |
| gsm8k / nano_awq_dflash4 | 545 | vllmw4_dflash 434 | 1.25x | met |
| gsm8k / nano_awq_dflashf | 490 | vllmw4_dflash 434 | 1.13x | met |
| long / nano_awq_dflash | 289 | vllmw4_dflash 245 | 1.18x | met |
| long / nano_awq_dflash4 | 318 | vllmw4_dflash 245 | 1.30x | met |
| long / nano_awq_dflashf | 292 | vllmw4_dflash 245 | 1.19x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 410 | vllmw4_dflash 371 | 1.11x | met |
| humaneval / nano_awq_dflash | 630 | vllmw4_dflash 556 | 1.13x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_awq_dflash | 423 | 207 | 2.04x | vllmw4_dflash 380 | 1.11x |
| alpaca c=4 / nano_awq_dflash | 1294 | 753 | 1.72x | vllmw4_dflash 1211 | 1.07x |
| alpaca c=16 / nano_awq_dflash | 3045 | 2429 | 1.25x | vllmw4_dflash 2084 | 1.46x |
| alpaca c=64 / nano_awq_dflash | 5971 | 6001 | 0.99x | vllmw4_dflash 2943 | 2.03x |
| alpaca c=1 / nano_awq_dflash4 | 480 | 207 | 2.31x | vllmw4_dflash 380 | 1.26x |
| alpaca c=4 / nano_awq_dflash4 | 1491 | 753 | 1.98x | vllmw4_dflash 1211 | 1.23x |
| alpaca c=16 / nano_awq_dflash4 | 3218 | 2429 | 1.32x | vllmw4_dflash 2084 | 1.54x |
| alpaca c=64 / nano_awq_dflash4 | 5786 | 6001 | 0.96x NOT met | vllmw4_dflash 2943 | 1.97x |
| alpaca c=1 / nano_awq_dflashf | 434 | 207 | 2.09x | vllmw4_dflash 380 | 1.14x |
| alpaca c=4 / nano_awq_dflashf | 1243 | 753 | 1.65x | vllmw4_dflash 1211 | 1.03x |
| alpaca c=16 / nano_awq_dflashf | 2263 | 2429 | 0.93x NOT met | vllmw4_dflash 2084 | 1.09x |
| alpaca c=64 / nano_awq_dflashf | 2744 | 6001 | 0.46x NOT met | vllmw4_dflash 2943 | 0.93x NOT met |
| humaneval c=1 / nano_awq_dflash | 641 | 206 | 3.12x | vllmw4_dflash 589 | 1.09x |
| humaneval c=4 / nano_awq_dflash | 1786 | 737 | 2.42x | vllmw4_dflash 1701 | 1.05x |
| humaneval c=16 / nano_awq_dflash | 3434 | 2532 | 1.36x | vllmw4_dflash 2950 | 1.16x |
| humaneval c=64 / nano_awq_dflash | 5523 | 5158 | 1.07x | vllmw4_dflash 4204 | 1.31x |
| humaneval c=1 / nano_awq_dflash4 | 716 | 206 | 3.48x | vllmw4_dflash 589 | 1.21x |
| humaneval c=4 / nano_awq_dflash4 | 1956 | 737 | 2.65x | vllmw4_dflash 1701 | 1.15x |
| humaneval c=16 / nano_awq_dflash4 | 3529 | 2532 | 1.39x | vllmw4_dflash 2950 | 1.20x |
| humaneval c=64 / nano_awq_dflash4 | 5425 | 5158 | 1.05x | vllmw4_dflash 4204 | 1.29x |
| humaneval c=1 / nano_awq_dflashf | 645 | 206 | 3.13x | vllmw4_dflash 589 | 1.09x |
| humaneval c=4 / nano_awq_dflashf | 1793 | 737 | 2.43x | vllmw4_dflash 1701 | 1.05x |
| humaneval c=16 / nano_awq_dflashf | 3030 | 2532 | 1.20x | vllmw4_dflash 2950 | 1.03x |
| humaneval c=64 / nano_awq_dflashf | 3974 | 5158 | 0.77x NOT met | vllmw4_dflash 4204 | 0.95x NOT met |
| gsm8k c=1 / nano_awq_dflash | 486 | 206 | 2.36x | vllmw4_dflash 434 | 1.12x |
| gsm8k c=16 / nano_awq_dflash | 3170 | 2209 | 1.44x | vllmw4_dflash 2312 | 1.37x |
| gsm8k c=1 / nano_awq_dflash4 | 545 | 206 | 2.65x | vllmw4_dflash 434 | 1.25x |
| gsm8k c=16 / nano_awq_dflash4 | 3254 | 2209 | 1.47x | vllmw4_dflash 2312 | 1.41x |
| gsm8k c=1 / nano_awq_dflashf | 490 | 206 | 2.38x | vllmw4_dflash 434 | 1.13x |
| gsm8k c=16 / nano_awq_dflashf | 2439 | 2209 | 1.10x | vllmw4_dflash 2312 | 1.05x |
| long c=1 / nano_awq_dflash | 289 | 178 | 1.63x | vllmw4_dflash 245 | 1.18x |
| long c=16 / nano_awq_dflash | 934 | 869 | 1.08x | vllmw4_dflash 750 | 1.25x |
| long c=1 / nano_awq_dflash4 | 318 | 178 | 1.79x | vllmw4_dflash 245 | 1.30x |
| long c=16 / nano_awq_dflash4 | 939 | 869 | 1.08x | vllmw4_dflash 750 | 1.25x |
| long c=1 / nano_awq_dflashf | 292 | 178 | 1.64x | vllmw4_dflash 245 | 1.19x |
| long c=16 / nano_awq_dflashf | 805 | 869 | 0.93x NOT met | vllmw4_dflash 750 | 1.07x |
| alpaca sampled c=1 / nano_awq_dflash | 410 | 206 | 1.99x | vllmw4_dflash 371 | 1.11x |
| alpaca sampled c=16 / nano_awq_dflash | 3042 | 2528 | 1.20x | vllmw4_dflash 2040 | 1.49x |
| humaneval sampled c=1 / nano_awq_dflash | 630 | 206 | 3.06x | vllmw4_dflash 556 | 1.13x |
| humaneval sampled c=16 / nano_awq_dflash | 3293 | 2775 | 1.19x | vllmw4_dflash 2869 | 1.15x |

- nano_awq_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_awq_dflash4: at least 0.98x its own server without a drafter on 11 of 12 workloads; above the best DFlash baseline on 12 of 12.
- nano_awq_dflashf: at least 0.98x its own server without a drafter on 8 of 12 workloads; above the best DFlash baseline on 10 of 12.

