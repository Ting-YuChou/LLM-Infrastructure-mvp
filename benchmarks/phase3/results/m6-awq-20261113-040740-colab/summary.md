# Phase 3 run m6-awq-20261113-040740-colab

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
| serve | 0 | 1584 |
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
| nano_awq | 207 | 748 | 2491 | 6014 |
| nano_awq_dflash | 415 | 1253 | 2967 | 5092 |
| nano_awq_dflash4 | 464 | 1402 | 3063 | 5813 |
| nano_awq_dflashf | 427 | 1175 | 2194 | 2741 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.79 | 4.94 | 5.30 | 9.39 |
| nano_awq_dflash | 2.21 | 2.91 | 4.60 | 11.42 |
| nano_awq_dflash4 | 2.02 | 2.58 | 4.32 | 9.99 |
| nano_awq_dflashf | 2.18 | 2.96 | 6.23 | 21.58 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 10.1 | 16.1 | 24.0 | 49.2 |
| nano_awq_dflash | 10.2 | 26.8 | 33.3 | 70.6 |
| nano_awq_dflash4 | 10.3 | 25.3 | 33.1 | 56.7 |
| nano_awq_dflashf | 10.1 | 28.3 | 51.6 | 165.4 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.00x | 1.68x | 1.19x | 0.85x |
| nano_awq_dflash4 | 2.24x | 1.88x | 1.23x | 0.97x |
| nano_awq_dflashf | 2.06x | 1.57x | 0.88x | 0.46x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 206 | 735 | 2527 | 5125 |
| nano_awq_dflash | 629 | 1703 | 3262 | 5239 |
| nano_awq_dflash4 | 704 | 1830 | 3389 | 5316 |
| nano_awq_dflashf | 633 | 1717 | 2939 | 3919 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.81 | 5.06 | 5.73 | 10.00 |
| nano_awq_dflash | 1.53 | 2.07 | 4.41 | 10.23 |
| nano_awq_dflash4 | 1.36 | 1.98 | 4.18 | 9.99 |
| nano_awq_dflashf | 1.52 | 2.06 | 4.83 | 13.83 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 16.8 | 26.5 | 125.3 | 130.6 |
| nano_awq_dflash | 17.3 | 36.1 | 58.8 | 136.0 |
| nano_awq_dflash4 | 17.3 | 34.3 | 61.6 | 140.8 |
| nano_awq_dflashf | 17.5 | 36.3 | 75.7 | 209.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.06x | 2.32x | 1.29x | 1.02x |
| nano_awq_dflash4 | 3.42x | 2.49x | 1.34x | 1.04x |
| nano_awq_dflashf | 3.08x | 2.34x | 1.16x | 0.76x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2173 |
| nano_awq_dflash | 478 | 2965 |
| nano_awq_dflash4 | 535 | 3056 |
| nano_awq_dflashf | 482 | 2385 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.80 | 5.79 |
| nano_awq_dflash | 2.06 | 4.67 |
| nano_awq_dflash4 | 1.79 | 4.51 |
| nano_awq_dflashf | 2.02 | 5.84 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 15.9 | 24.3 |
| nano_awq_dflash | 14.9 | 42.7 |
| nano_awq_dflash4 | 15.1 | 38.9 |
| nano_awq_dflashf | 15.4 | 60.4 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.33x | 1.36x |
| nano_awq_dflash4 | 2.60x | 1.41x |
| nano_awq_dflashf | 2.34x | 1.10x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 178 | 858 |
| nano_awq_dflash | 283 | 911 |
| nano_awq_dflash4 | 311 | 926 |
| nano_awq_dflashf | 284 | 789 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.86 | 15.46 |
| nano_awq_dflash | 2.71 | 14.65 |
| nano_awq_dflash4 | 2.42 | 14.41 |
| nano_awq_dflashf | 2.69 | 16.68 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 104.9 | 240.4 |
| nano_awq_dflash | 106.6 | 240.3 |
| nano_awq_dflash4 | 106.5 | 229.3 |
| nano_awq_dflashf | 106.2 | 245.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.59x | 1.06x |
| nano_awq_dflash4 | 1.75x | 1.08x |
| nano_awq_dflashf | 1.60x | 0.92x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2522 |
| nano_awq_dflash | 408 | 2919 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.82 | 5.41 |
| nano_awq_dflash | 2.34 | 4.71 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 10.1 | 23.1 |
| nano_awq_dflash | 10.3 | 33.0 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.98x | 1.16x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2779 |
| nano_awq_dflash | 617 | 3213 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.83 | 5.53 |
| nano_awq_dflash | 1.51 | 4.51 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.9 | 32.1 |
| nano_awq_dflash | 17.5 | 63.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.00x | 1.16x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1333 | 2.44 | 3.44 |
| nano_awq_dflash | alpaca_c4 | 1440 | 2.36 | 3.36 |
| nano_awq_dflash | alpaca_c16 | 3935 | 1.90 | 2.90 |
| nano_awq_dflash | alpaca_c64 | 20789 | 1.44 | 2.44 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1019 | 4.54 | 5.54 |
| nano_awq_dflash | humaneval_c16 | 3603 | 3.48 | 4.48 |
| nano_awq_dflash | humaneval_c64 | 11746 | 2.47 | 3.47 |
| nano_awq_dflash | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflash | gsm8k_c16 | 3316 | 2.12 | 3.12 |
| nano_awq_dflash | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 3376 | 1.64 | 2.64 |
| nano_awq_dflash | alpaca_c1s | 1354 | 2.35 | 3.35 |
| nano_awq_dflash | alpaca_c16s | 4151 | 1.80 | 2.80 |
| nano_awq_dflash | humaneval_c1s | 1090 | 4.27 | 5.27 |
| nano_awq_dflash | humaneval_c16s | 3766 | 3.30 | 4.30 |
| nano_awq_dflash4 | alpaca_c1 | 1343 | 2.41 | 3.41 |
| nano_awq_dflash4 | alpaca_c4 | 1465 | 2.30 | 3.30 |
| nano_awq_dflash4 | alpaca_c16 | 4018 | 1.87 | 2.87 |
| nano_awq_dflash4 | alpaca_c64 | 4342 | 1.14 | 2.14 |
| nano_awq_dflash4 | humaneval_c1 | 1108 | 4.25 | 5.25 |
| nano_awq_dflash4 | humaneval_c4 | 1025 | 4.53 | 5.53 |
| nano_awq_dflash4 | humaneval_c16 | 3648 | 3.44 | 4.44 |
| nano_awq_dflash4 | humaneval_c64 | 10482 | 2.76 | 3.76 |
| nano_awq_dflash4 | gsm8k_c1 | 1162 | 2.90 | 3.90 |
| nano_awq_dflash4 | gsm8k_c16 | 3324 | 2.11 | 3.11 |
| nano_awq_dflash4 | long_c1 | 1123 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 3428 | 1.62 | 2.62 |
| nano_awq_dflashf | alpaca_c1 | 1309 | 2.50 | 3.50 |
| nano_awq_dflashf | alpaca_c4 | 1345 | 2.59 | 3.59 |
| nano_awq_dflashf | alpaca_c16 | 3113 | 2.70 | 3.70 |
| nano_awq_dflashf | alpaca_c64 | 14591 | 2.50 | 3.50 |
| nano_awq_dflashf | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflashf | humaneval_c4 | 1019 | 4.54 | 5.54 |
| nano_awq_dflashf | humaneval_c16 | 2885 | 4.63 | 5.63 |
| nano_awq_dflashf | humaneval_c64 | 7343 | 4.59 | 5.59 |
| nano_awq_dflashf | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflashf | gsm8k_c16 | 2507 | 3.20 | 4.20 |
| nano_awq_dflashf | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflashf | long_c16 | 2943 | 2.03 | 3.03 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.44 |
| nano_awq_dflash | alpaca_c4 | 0% | 0% | 0% | 0% | 95% | 5% | 3.36 |
| nano_awq_dflash | alpaca_c16 | 0% | 0% | 0% | 93% | 7% | 0% | 2.90 |
| nano_awq_dflash | alpaca_c64 | 1% | 3% | 86% | 10% | 0% | 0% | 2.42 |
| nano_awq_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.32 |
| nano_awq_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.54 |
| nano_awq_dflash | humaneval_c16 | 0% | 0% | 0% | 6% | 91% | 3% | 4.48 |
| nano_awq_dflash | humaneval_c64 | 0% | 0% | 0% | 96% | 3% | 0% | 3.47 |
| nano_awq_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.97 |
| nano_awq_dflash | gsm8k_c16 | 0% | 0% | 0% | 95% | 4% | 2% | 3.12 |
| nano_awq_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.02 |
| nano_awq_dflash | long_c16 | 0% | 0% | 0% | 96% | 4% | 0% | 2.64 |
| nano_awq_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.35 |
| nano_awq_dflash | alpaca_c16s | 0% | 0% | 0% | 95% | 5% | 0% | 2.80 |
| nano_awq_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.27 |
| nano_awq_dflash | humaneval_c16s | 0% | 0% | 0% | 11% | 87% | 3% | 4.30 |
| nano_awq_dflash4 | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.41 |
| nano_awq_dflash4 | alpaca_c4 | 0% | 0% | 0% | 0% | 98% | 2% | 3.30 |
| nano_awq_dflash4 | alpaca_c16 | 0% | 0% | 0% | 95% | 5% | 0% | 2.87 |
| nano_awq_dflash4 | alpaca_c64 | 91% | 6% | 1% | 2% | 0% | 0% | 1.11 |
| nano_awq_dflash4 | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.25 |
| nano_awq_dflash4 | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.53 |
| nano_awq_dflash4 | humaneval_c16 | 0% | 0% | 0% | 11% | 87% | 3% | 4.44 |
| nano_awq_dflash4 | humaneval_c64 | 11% | 0% | 0% | 67% | 22% | 0% | 3.45 |
| nano_awq_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.90 |
| nano_awq_dflash4 | gsm8k_c16 | 0% | 0% | 0% | 95% | 4% | 1% | 3.11 |
| nano_awq_dflash4 | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.98 |
| nano_awq_dflash4 | long_c16 | 0% | 0% | 0% | 98% | 2% | 0% | 2.62 |
| nano_awq_dflashf | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_awq_dflashf | alpaca_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 3.59 |
| nano_awq_dflashf | alpaca_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 3.70 |
| nano_awq_dflashf | alpaca_c64 | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_awq_dflashf | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.32 |
| nano_awq_dflashf | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.54 |
| nano_awq_dflashf | humaneval_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 5.63 |
| nano_awq_dflashf | humaneval_c64 | 0% | 0% | 0% | 0% | 0% | 100% | 5.59 |
| nano_awq_dflashf | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.97 |
| nano_awq_dflashf | gsm8k_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 4.20 |
| nano_awq_dflashf | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.02 |
| nano_awq_dflashf | long_c16 | 0% | 0% | 0% | 0% | 0% | 100% | 3.03 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 415 | vllmw4_dflash 380 | 1.09x | met |
| alpaca / nano_awq_dflash4 | 464 | vllmw4_dflash 380 | 1.22x | met |
| alpaca / nano_awq_dflashf | 427 | vllmw4_dflash 380 | 1.13x | met |
| humaneval / nano_awq_dflash | 629 | vllmw4_dflash 589 | 1.07x | met |
| humaneval / nano_awq_dflash4 | 704 | vllmw4_dflash 589 | 1.19x | met |
| humaneval / nano_awq_dflashf | 633 | vllmw4_dflash 589 | 1.07x | met |
| gsm8k / nano_awq_dflash | 478 | vllmw4_dflash 434 | 1.10x | met |
| gsm8k / nano_awq_dflash4 | 535 | vllmw4_dflash 434 | 1.23x | met |
| gsm8k / nano_awq_dflashf | 482 | vllmw4_dflash 434 | 1.11x | met |
| long / nano_awq_dflash | 283 | vllmw4_dflash 245 | 1.16x | met |
| long / nano_awq_dflash4 | 311 | vllmw4_dflash 245 | 1.27x | met |
| long / nano_awq_dflashf | 284 | vllmw4_dflash 245 | 1.16x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 408 | vllmw4_dflash 371 | 1.10x | met |
| humaneval / nano_awq_dflash | 617 | vllmw4_dflash 556 | 1.11x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_awq_dflash | 415 | 207 | 2.00x | vllmw4_dflash 380 | 1.09x |
| alpaca c=4 / nano_awq_dflash | 1253 | 748 | 1.68x | vllmw4_dflash 1211 | 1.03x |
| alpaca c=16 / nano_awq_dflash | 2967 | 2491 | 1.19x | vllmw4_dflash 2084 | 1.42x |
| alpaca c=64 / nano_awq_dflash | 5092 | 6014 | 0.85x NOT met | vllmw4_dflash 2943 | 1.73x |
| alpaca c=1 / nano_awq_dflash4 | 464 | 207 | 2.24x | vllmw4_dflash 380 | 1.22x |
| alpaca c=4 / nano_awq_dflash4 | 1402 | 748 | 1.88x | vllmw4_dflash 1211 | 1.16x |
| alpaca c=16 / nano_awq_dflash4 | 3063 | 2491 | 1.23x | vllmw4_dflash 2084 | 1.47x |
| alpaca c=64 / nano_awq_dflash4 | 5813 | 6014 | 0.97x NOT met | vllmw4_dflash 2943 | 1.97x |
| alpaca c=1 / nano_awq_dflashf | 427 | 207 | 2.06x | vllmw4_dflash 380 | 1.13x |
| alpaca c=4 / nano_awq_dflashf | 1175 | 748 | 1.57x | vllmw4_dflash 1211 | 0.97x NOT met |
| alpaca c=16 / nano_awq_dflashf | 2194 | 2491 | 0.88x NOT met | vllmw4_dflash 2084 | 1.05x |
| alpaca c=64 / nano_awq_dflashf | 2741 | 6014 | 0.46x NOT met | vllmw4_dflash 2943 | 0.93x NOT met |
| humaneval c=1 / nano_awq_dflash | 629 | 206 | 3.06x | vllmw4_dflash 589 | 1.07x |
| humaneval c=4 / nano_awq_dflash | 1703 | 735 | 2.32x | vllmw4_dflash 1701 | 1.00x |
| humaneval c=16 / nano_awq_dflash | 3262 | 2527 | 1.29x | vllmw4_dflash 2950 | 1.11x |
| humaneval c=64 / nano_awq_dflash | 5239 | 5125 | 1.02x | vllmw4_dflash 4204 | 1.25x |
| humaneval c=1 / nano_awq_dflash4 | 704 | 206 | 3.42x | vllmw4_dflash 589 | 1.19x |
| humaneval c=4 / nano_awq_dflash4 | 1830 | 735 | 2.49x | vllmw4_dflash 1701 | 1.08x |
| humaneval c=16 / nano_awq_dflash4 | 3389 | 2527 | 1.34x | vllmw4_dflash 2950 | 1.15x |
| humaneval c=64 / nano_awq_dflash4 | 5316 | 5125 | 1.04x | vllmw4_dflash 4204 | 1.26x |
| humaneval c=1 / nano_awq_dflashf | 633 | 206 | 3.08x | vllmw4_dflash 589 | 1.07x |
| humaneval c=4 / nano_awq_dflashf | 1717 | 735 | 2.34x | vllmw4_dflash 1701 | 1.01x |
| humaneval c=16 / nano_awq_dflashf | 2939 | 2527 | 1.16x | vllmw4_dflash 2950 | 1.00x NOT met |
| humaneval c=64 / nano_awq_dflashf | 3919 | 5125 | 0.76x NOT met | vllmw4_dflash 4204 | 0.93x NOT met |
| gsm8k c=1 / nano_awq_dflash | 478 | 206 | 2.33x | vllmw4_dflash 434 | 1.10x |
| gsm8k c=16 / nano_awq_dflash | 2965 | 2173 | 1.36x | vllmw4_dflash 2312 | 1.28x |
| gsm8k c=1 / nano_awq_dflash4 | 535 | 206 | 2.60x | vllmw4_dflash 434 | 1.23x |
| gsm8k c=16 / nano_awq_dflash4 | 3056 | 2173 | 1.41x | vllmw4_dflash 2312 | 1.32x |
| gsm8k c=1 / nano_awq_dflashf | 482 | 206 | 2.34x | vllmw4_dflash 434 | 1.11x |
| gsm8k c=16 / nano_awq_dflashf | 2385 | 2173 | 1.10x | vllmw4_dflash 2312 | 1.03x |
| long c=1 / nano_awq_dflash | 283 | 178 | 1.59x | vllmw4_dflash 245 | 1.16x |
| long c=16 / nano_awq_dflash | 911 | 858 | 1.06x | vllmw4_dflash 750 | 1.22x |
| long c=1 / nano_awq_dflash4 | 311 | 178 | 1.75x | vllmw4_dflash 245 | 1.27x |
| long c=16 / nano_awq_dflash4 | 926 | 858 | 1.08x | vllmw4_dflash 750 | 1.23x |
| long c=1 / nano_awq_dflashf | 284 | 178 | 1.60x | vllmw4_dflash 245 | 1.16x |
| long c=16 / nano_awq_dflashf | 789 | 858 | 0.92x NOT met | vllmw4_dflash 750 | 1.05x |
| alpaca sampled c=1 / nano_awq_dflash | 408 | 206 | 1.98x | vllmw4_dflash 371 | 1.10x |
| alpaca sampled c=16 / nano_awq_dflash | 2919 | 2522 | 1.16x | vllmw4_dflash 2040 | 1.43x |
| humaneval sampled c=1 / nano_awq_dflash | 617 | 206 | 3.00x | vllmw4_dflash 556 | 1.11x |
| humaneval sampled c=16 / nano_awq_dflash | 3213 | 2779 | 1.16x | vllmw4_dflash 2869 | 1.12x |

- nano_awq_dflash: at least 0.98x its own server without a drafter on 15 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_awq_dflash4: at least 0.98x its own server without a drafter on 11 of 12 workloads; above the best DFlash baseline on 12 of 12.
- nano_awq_dflashf: at least 0.98x its own server without a drafter on 8 of 12 workloads; above the best DFlash baseline on 8 of 12.

