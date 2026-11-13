# Phase 3 run m7-awq-20261113-093737-colab

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
| serve | 0 | 1597 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq | 42 | 163 | 99 | 1223 |
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano_awq_dflashs | 42 | 163 | 99 | 1223 |
| nano_awq_dflash4 | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 207 | 747 | 2567 | 6001 |
| nano_awq_dflash | 451 | 1371 | 3240 | 5576 |
| nano_awq_dflashs | 426 | 1361 | 3116 | 5962 |
| nano_awq_dflash4 | 511 | 1567 | 3458 | 6040 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.79 | 4.93 | 5.35 | 9.43 |
| nano_awq_dflash | 2.08 | 2.59 | 4.08 | 10.46 |
| nano_awq_dflashs | 2.22 | 2.73 | 4.33 | 9.66 |
| nano_awq_dflash4 | 1.87 | 2.22 | 3.92 | 9.65 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 9.9 | 16.0 | 23.3 | 50.1 |
| nano_awq_dflash | 9.6 | 28.8 | 41.8 | 81.1 |
| nano_awq_dflashs | 9.4 | 20.3 | 29.0 | 58.3 |
| nano_awq_dflash4 | 9.7 | 26.7 | 40.3 | 58.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.18x | 1.83x | 1.26x | 0.93x |
| nano_awq_dflashs | 2.05x | 1.82x | 1.21x | 0.99x |
| nano_awq_dflash4 | 2.46x | 2.10x | 1.35x | 1.01x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 206 | 737 | 2530 | 5192 |
| nano_awq_dflash | 678 | 1866 | 3588 | 5664 |
| nano_awq_dflashs | 640 | 1799 | 3414 | 5486 |
| nano_awq_dflash4 | 764 | 2083 | 3695 | 5712 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.81 | 5.05 | 5.71 | 9.85 |
| nano_awq_dflash | 1.41 | 1.82 | 3.97 | 9.08 |
| nano_awq_dflashs | 1.51 | 1.95 | 4.26 | 9.62 |
| nano_awq_dflash4 | 1.25 | 1.74 | 3.77 | 9.19 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 16.7 | 24.8 | 111.8 | 134.8 |
| nano_awq_dflash | 16.7 | 42.0 | 76.5 | 133.9 |
| nano_awq_dflashs | 16.5 | 32.0 | 61.6 | 135.8 |
| nano_awq_dflash4 | 16.7 | 41.0 | 67.2 | 148.4 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.29x | 2.53x | 1.42x | 1.09x |
| nano_awq_dflashs | 3.11x | 2.44x | 1.35x | 1.06x |
| nano_awq_dflash4 | 3.71x | 2.82x | 1.46x | 1.10x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2226 |
| nano_awq_dflash | 510 | 3254 |
| nano_awq_dflashs | 483 | 2983 |
| nano_awq_dflash4 | 577 | 3391 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.80 | 5.77 |
| nano_awq_dflash | 1.89 | 4.16 |
| nano_awq_dflashs | 2.02 | 4.48 |
| nano_awq_dflash4 | 1.65 | 4.03 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 15.8 | 24.5 |
| nano_awq_dflash | 18.4 | 48.9 |
| nano_awq_dflashs | 14.5 | 36.2 |
| nano_awq_dflash4 | 17.4 | 46.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.48x | 1.46x |
| nano_awq_dflashs | 2.35x | 1.34x |
| nano_awq_dflash4 | 2.81x | 1.52x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 178 | 870 |
| nano_awq_dflash | 304 | 946 |
| nano_awq_dflashs | 290 | 933 |
| nano_awq_dflash4 | 334 | 947 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.86 | 15.79 |
| nano_awq_dflash | 2.45 | 14.14 |
| nano_awq_dflashs | 2.61 | 14.20 |
| nano_awq_dflash4 | 2.17 | 13.57 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 104.2 | 219.3 |
| nano_awq_dflash | 105.9 | 252.8 |
| nano_awq_dflashs | 105.4 | 229.1 |
| nano_awq_dflash4 | 105.8 | 312.1 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.71x | 1.09x |
| nano_awq_dflashs | 1.63x | 1.07x |
| nano_awq_dflash4 | 1.88x | 1.09x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2548 |
| nano_awq_dflash | 436 | 3148 |
| nano_awq_dflashs | 412 | 3016 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.81 | 5.39 |
| nano_awq_dflash | 2.22 | 4.29 |
| nano_awq_dflashs | 2.37 | 4.53 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.7 | 20.8 |
| nano_awq_dflash | 10.1 | 42.4 |
| nano_awq_dflashs | 9.6 | 29.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.11x | 1.24x |
| nano_awq_dflashs | 2.00x | 1.18x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2801 |
| nano_awq_dflash | 667 | 3494 |
| nano_awq_dflashs | 627 | 3378 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.83 | 5.51 |
| nano_awq_dflash | 1.41 | 3.93 |
| nano_awq_dflashs | 1.49 | 4.22 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.5 | 31.2 |
| nano_awq_dflash | 16.9 | 77.4 |
| nano_awq_dflashs | 16.8 | 57.7 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.24x | 1.25x |
| nano_awq_dflashs | 3.04x | 1.21x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1332 | 2.44 | 3.44 |
| nano_awq_dflash | alpaca_c4 | 1464 | 2.30 | 3.30 |
| nano_awq_dflash | alpaca_c16 | 3941 | 1.90 | 2.90 |
| nano_awq_dflash | alpaca_c64 | 23832 | 1.10 | 2.10 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1019 | 4.54 | 5.54 |
| nano_awq_dflash | humaneval_c16 | 3654 | 3.42 | 4.42 |
| nano_awq_dflash | humaneval_c64 | 11694 | 2.47 | 3.47 |
| nano_awq_dflash | gsm8k_c1 | 1145 | 2.96 | 3.96 |
| nano_awq_dflash | gsm8k_c16 | 3230 | 2.14 | 3.14 |
| nano_awq_dflash | long_c1 | 1110 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 3336 | 1.65 | 2.65 |
| nano_awq_dflash | alpaca_c1s | 1355 | 2.34 | 3.34 |
| nano_awq_dflash | alpaca_c16s | 4111 | 1.82 | 2.82 |
| nano_awq_dflash | humaneval_c1s | 1088 | 4.28 | 5.28 |
| nano_awq_dflash | humaneval_c16s | 3869 | 3.19 | 4.19 |
| nano_awq_dflashs | alpaca_c1 | 1333 | 2.44 | 3.44 |
| nano_awq_dflashs | alpaca_c4 | 1504 | 2.30 | 3.30 |
| nano_awq_dflashs | alpaca_c16 | 3943 | 1.89 | 2.89 |
| nano_awq_dflashs | alpaca_c64 | 1498 | 1.36 | 2.36 |
| nano_awq_dflashs | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflashs | humaneval_c4 | 1020 | 4.54 | 5.54 |
| nano_awq_dflashs | humaneval_c16 | 3611 | 3.45 | 4.45 |
| nano_awq_dflashs | humaneval_c64 | 11400 | 2.50 | 3.50 |
| nano_awq_dflashs | gsm8k_c1 | 1145 | 2.96 | 3.96 |
| nano_awq_dflashs | gsm8k_c16 | 3252 | 2.11 | 3.11 |
| nano_awq_dflashs | long_c1 | 1110 | 2.02 | 3.02 |
| nano_awq_dflashs | long_c16 | 3347 | 1.64 | 2.64 |
| nano_awq_dflashs | alpaca_c1s | 1355 | 2.34 | 3.34 |
| nano_awq_dflashs | alpaca_c16s | 4063 | 1.81 | 2.81 |
| nano_awq_dflashs | humaneval_c1s | 1088 | 4.28 | 5.28 |
| nano_awq_dflashs | humaneval_c16s | 3662 | 3.41 | 4.41 |
| nano_awq_dflash4 | alpaca_c1 | 1352 | 2.39 | 3.39 |
| nano_awq_dflash4 | alpaca_c4 | 1484 | 2.33 | 3.33 |
| nano_awq_dflash4 | alpaca_c16 | 3999 | 1.88 | 2.88 |
| nano_awq_dflash4 | alpaca_c64 | 1990 | 1.44 | 2.44 |
| nano_awq_dflash4 | humaneval_c1 | 1107 | 4.26 | 5.26 |
| nano_awq_dflash4 | humaneval_c4 | 1007 | 4.59 | 5.59 |
| nano_awq_dflash4 | humaneval_c16 | 3744 | 3.28 | 4.28 |
| nano_awq_dflash4 | humaneval_c64 | 11783 | 2.42 | 3.42 |
| nano_awq_dflash4 | gsm8k_c1 | 1161 | 2.91 | 3.91 |
| nano_awq_dflash4 | gsm8k_c16 | 3270 | 2.12 | 3.12 |
| nano_awq_dflash4 | long_c1 | 1126 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 3331 | 1.61 | 2.61 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.42 |
| nano_awq_dflash | alpaca_c4 | 1% | 0% | 0% | 0% | 96% | 3% | 3.23 |
| nano_awq_dflash | alpaca_c16 | 3% | 0% | 0% | 92% | 5% | 0% | 2.83 |
| nano_awq_dflash | alpaca_c64 | 4% | 56% | 32% | 8% | 0% | 0% | 2.05 |
| nano_awq_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.29 |
| nano_awq_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.45 |
| nano_awq_dflash | humaneval_c16 | 0% | 0% | 0% | 14% | 84% | 2% | 4.39 |
| nano_awq_dflash | humaneval_c64 | 2% | 0% | 0% | 95% | 2% | 0% | 3.40 |
| nano_awq_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.89 |
| nano_awq_dflash | gsm8k_c16 | 2% | 0% | 0% | 95% | 2% | 2% | 3.05 |
| nano_awq_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_awq_dflash | long_c16 | 3% | 0% | 0% | 95% | 2% | 0% | 2.56 |
| nano_awq_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.32 |
| nano_awq_dflash | alpaca_c16s | 2% | 0% | 0% | 94% | 4% | 0% | 2.77 |
| nano_awq_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.24 |
| nano_awq_dflash | humaneval_c16s | 0% | 0% | 0% | 25% | 72% | 2% | 4.16 |
| nano_awq_dflashs | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.44 |
| nano_awq_dflashs | alpaca_c4 | 0% | 0% | 0% | 1% | 97% | 2% | 3.30 |
| nano_awq_dflashs | alpaca_c16 | 3% | 0% | 0% | 90% | 6% | 1% | 2.83 |
| nano_awq_dflashs | alpaca_c64 | 97% | 2% | 0% | 1% | 0% | 0% | 1.04 |
| nano_awq_dflashs | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.32 |
| nano_awq_dflashs | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.54 |
| nano_awq_dflashs | humaneval_c16 | 3% | 0% | 0% | 8% | 87% | 2% | 4.33 |
| nano_awq_dflashs | humaneval_c64 | 7% | 0% | 0% | 88% | 5% | 0% | 3.33 |
| nano_awq_dflashs | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.96 |
| nano_awq_dflashs | gsm8k_c16 | 3% | 0% | 0% | 92% | 4% | 1% | 3.03 |
| nano_awq_dflashs | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.02 |
| nano_awq_dflashs | long_c16 | 4% | 0% | 0% | 92% | 4% | 0% | 2.59 |
| nano_awq_dflashs | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.34 |
| nano_awq_dflashs | alpaca_c16s | 3% | 0% | 0% | 93% | 4% | 0% | 2.75 |
| nano_awq_dflashs | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.28 |
| nano_awq_dflashs | humaneval_c16s | 2% | 0% | 0% | 6% | 90% | 2% | 4.35 |
| nano_awq_dflash4 | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.37 |
| nano_awq_dflash4 | alpaca_c4 | 0% | 0% | 0% | 0% | 98% | 2% | 3.30 |
| nano_awq_dflash4 | alpaca_c16 | 3% | 0% | 0% | 94% | 3% | 0% | 2.80 |
| nano_awq_dflash4 | alpaca_c64 | 96% | 2% | 0% | 2% | 0% | 0% | 1.06 |
| nano_awq_dflash4 | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.23 |
| nano_awq_dflash4 | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.49 |
| nano_awq_dflash4 | humaneval_c16 | 2% | 0% | 0% | 25% | 72% | 2% | 4.20 |
| nano_awq_dflash4 | humaneval_c64 | 5% | 0% | 5% | 90% | 1% | 0% | 3.30 |
| nano_awq_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.84 |
| nano_awq_dflash4 | gsm8k_c16 | 4% | 0% | 0% | 92% | 3% | 1% | 3.00 |
| nano_awq_dflash4 | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.91 |
| nano_awq_dflash4 | long_c16 | 4% | 0% | 0% | 94% | 1% | 0% | 2.52 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 451 | vllmw4_dflash 380 | 1.19x | met |
| alpaca / nano_awq_dflash4 | 511 | vllmw4_dflash 380 | 1.34x | met |
| alpaca / nano_awq_dflashs | 426 | vllmw4_dflash 380 | 1.12x | met |
| humaneval / nano_awq_dflash | 678 | vllmw4_dflash 589 | 1.15x | met |
| humaneval / nano_awq_dflash4 | 764 | vllmw4_dflash 589 | 1.30x | met |
| humaneval / nano_awq_dflashs | 640 | vllmw4_dflash 589 | 1.09x | met |
| gsm8k / nano_awq_dflash | 510 | vllmw4_dflash 434 | 1.18x | met |
| gsm8k / nano_awq_dflash4 | 577 | vllmw4_dflash 434 | 1.33x | met |
| gsm8k / nano_awq_dflashs | 483 | vllmw4_dflash 434 | 1.11x | met |
| long / nano_awq_dflash | 304 | vllmw4_dflash 245 | 1.24x | met |
| long / nano_awq_dflash4 | 334 | vllmw4_dflash 245 | 1.36x | met |
| long / nano_awq_dflashs | 290 | vllmw4_dflash 245 | 1.19x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 436 | vllmw4_dflash 371 | 1.18x | met |
| alpaca / nano_awq_dflashs | 412 | vllmw4_dflash 371 | 1.11x | met |
| humaneval / nano_awq_dflash | 667 | vllmw4_dflash 556 | 1.20x | met |
| humaneval / nano_awq_dflashs | 627 | vllmw4_dflash 556 | 1.13x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_awq_dflash | 451 | 207 | 2.18x | vllmw4_dflash 380 | 1.19x |
| alpaca c=4 / nano_awq_dflash | 1371 | 747 | 1.83x | vllmw4_dflash 1211 | 1.13x |
| alpaca c=16 / nano_awq_dflash | 3240 | 2567 | 1.26x | vllmw4_dflash 2084 | 1.55x |
| alpaca c=64 / nano_awq_dflash | 5576 | 6001 | 0.93x NOT met | vllmw4_dflash 2943 | 1.89x |
| alpaca c=1 / nano_awq_dflash4 | 511 | 207 | 2.46x | vllmw4_dflash 380 | 1.34x |
| alpaca c=4 / nano_awq_dflash4 | 1567 | 747 | 2.10x | vllmw4_dflash 1211 | 1.29x |
| alpaca c=16 / nano_awq_dflash4 | 3458 | 2567 | 1.35x | vllmw4_dflash 2084 | 1.66x |
| alpaca c=64 / nano_awq_dflash4 | 6040 | 6001 | 1.01x | vllmw4_dflash 2943 | 2.05x |
| alpaca c=1 / nano_awq_dflashs | 426 | 207 | 2.05x | vllmw4_dflash 380 | 1.12x |
| alpaca c=4 / nano_awq_dflashs | 1361 | 747 | 1.82x | vllmw4_dflash 1211 | 1.12x |
| alpaca c=16 / nano_awq_dflashs | 3116 | 2567 | 1.21x | vllmw4_dflash 2084 | 1.50x |
| alpaca c=64 / nano_awq_dflashs | 5962 | 6001 | 0.99x | vllmw4_dflash 2943 | 2.03x |
| humaneval c=1 / nano_awq_dflash | 678 | 206 | 3.29x | vllmw4_dflash 589 | 1.15x |
| humaneval c=4 / nano_awq_dflash | 1866 | 737 | 2.53x | vllmw4_dflash 1701 | 1.10x |
| humaneval c=16 / nano_awq_dflash | 3588 | 2530 | 1.42x | vllmw4_dflash 2950 | 1.22x |
| humaneval c=64 / nano_awq_dflash | 5664 | 5192 | 1.09x | vllmw4_dflash 4204 | 1.35x |
| humaneval c=1 / nano_awq_dflash4 | 764 | 206 | 3.71x | vllmw4_dflash 589 | 1.30x |
| humaneval c=4 / nano_awq_dflash4 | 2083 | 737 | 2.82x | vllmw4_dflash 1701 | 1.22x |
| humaneval c=16 / nano_awq_dflash4 | 3695 | 2530 | 1.46x | vllmw4_dflash 2950 | 1.25x |
| humaneval c=64 / nano_awq_dflash4 | 5712 | 5192 | 1.10x | vllmw4_dflash 4204 | 1.36x |
| humaneval c=1 / nano_awq_dflashs | 640 | 206 | 3.11x | vllmw4_dflash 589 | 1.09x |
| humaneval c=4 / nano_awq_dflashs | 1799 | 737 | 2.44x | vllmw4_dflash 1701 | 1.06x |
| humaneval c=16 / nano_awq_dflashs | 3414 | 2530 | 1.35x | vllmw4_dflash 2950 | 1.16x |
| humaneval c=64 / nano_awq_dflashs | 5486 | 5192 | 1.06x | vllmw4_dflash 4204 | 1.30x |
| gsm8k c=1 / nano_awq_dflash | 510 | 206 | 2.48x | vllmw4_dflash 434 | 1.18x |
| gsm8k c=16 / nano_awq_dflash | 3254 | 2226 | 1.46x | vllmw4_dflash 2312 | 1.41x |
| gsm8k c=1 / nano_awq_dflash4 | 577 | 206 | 2.81x | vllmw4_dflash 434 | 1.33x |
| gsm8k c=16 / nano_awq_dflash4 | 3391 | 2226 | 1.52x | vllmw4_dflash 2312 | 1.47x |
| gsm8k c=1 / nano_awq_dflashs | 483 | 206 | 2.35x | vllmw4_dflash 434 | 1.11x |
| gsm8k c=16 / nano_awq_dflashs | 2983 | 2226 | 1.34x | vllmw4_dflash 2312 | 1.29x |
| long c=1 / nano_awq_dflash | 304 | 178 | 1.71x | vllmw4_dflash 245 | 1.24x |
| long c=16 / nano_awq_dflash | 946 | 870 | 1.09x | vllmw4_dflash 750 | 1.26x |
| long c=1 / nano_awq_dflash4 | 334 | 178 | 1.88x | vllmw4_dflash 245 | 1.36x |
| long c=16 / nano_awq_dflash4 | 947 | 870 | 1.09x | vllmw4_dflash 750 | 1.26x |
| long c=1 / nano_awq_dflashs | 290 | 178 | 1.63x | vllmw4_dflash 245 | 1.19x |
| long c=16 / nano_awq_dflashs | 933 | 870 | 1.07x | vllmw4_dflash 750 | 1.24x |
| alpaca sampled c=1 / nano_awq_dflash | 436 | 206 | 2.11x | vllmw4_dflash 371 | 1.18x |
| alpaca sampled c=16 / nano_awq_dflash | 3148 | 2548 | 1.24x | vllmw4_dflash 2040 | 1.54x |
| alpaca sampled c=1 / nano_awq_dflashs | 412 | 206 | 2.00x | vllmw4_dflash 371 | 1.11x |
| alpaca sampled c=16 / nano_awq_dflashs | 3016 | 2548 | 1.18x | vllmw4_dflash 2040 | 1.48x |
| humaneval sampled c=1 / nano_awq_dflash | 667 | 206 | 3.24x | vllmw4_dflash 556 | 1.20x |
| humaneval sampled c=16 / nano_awq_dflash | 3494 | 2801 | 1.25x | vllmw4_dflash 2869 | 1.22x |
| humaneval sampled c=1 / nano_awq_dflashs | 627 | 206 | 3.04x | vllmw4_dflash 556 | 1.13x |
| humaneval sampled c=16 / nano_awq_dflashs | 3378 | 2801 | 1.21x | vllmw4_dflash 2869 | 1.18x |

- nano_awq_dflash: at least 0.98x its own server without a drafter on 15 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_awq_dflash4: at least 0.98x its own server without a drafter on 12 of 12 workloads; above the best DFlash baseline on 12 of 12.
- nano_awq_dflashs: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 16 of 16.

**Serving: the same server with SPECULATIVE_CHAIN=0 (each cycle collected) and as it is**

| workload / server | collected tok/s | chained tok/s | chained / collected | collected TTFT p50 (ms) | chained TTFT p50 (ms) |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_awq_dflash | 426 | 451 | 1.061x | 9.4 | 9.6 |
| alpaca c=4 / nano_awq_dflash | 1361 | 1371 | 1.008x | 20.3 | 28.8 |
| alpaca c=16 / nano_awq_dflash | 3116 | 3240 | 1.040x | 29.0 | 41.8 |
| alpaca c=64 / nano_awq_dflash | 5962 | 5576 | 0.935x | 58.3 | 81.1 |
| humaneval c=1 / nano_awq_dflash | 640 | 678 | 1.059x | 16.5 | 16.7 |
| humaneval c=4 / nano_awq_dflash | 1799 | 1866 | 1.038x | 32.0 | 42.0 |
| humaneval c=16 / nano_awq_dflash | 3414 | 3588 | 1.051x | 61.6 | 76.5 |
| humaneval c=64 / nano_awq_dflash | 5486 | 5664 | 1.033x | 135.8 | 133.9 |
| gsm8k c=1 / nano_awq_dflash | 483 | 510 | 1.057x | 14.5 | 18.4 |
| gsm8k c=16 / nano_awq_dflash | 2983 | 3254 | 1.091x | 36.2 | 48.9 |
| long c=1 / nano_awq_dflash | 290 | 304 | 1.050x | 105.4 | 105.9 |
| long c=16 / nano_awq_dflash | 933 | 946 | 1.014x | 229.1 | 252.8 |
| alpaca sampled c=1 / nano_awq_dflash | 412 | 436 | 1.058x | 9.6 | 10.1 |
| alpaca sampled c=16 / nano_awq_dflash | 3016 | 3148 | 1.044x | 29.6 | 42.4 |
| humaneval sampled c=1 / nano_awq_dflash | 627 | 667 | 1.064x | 16.8 | 16.9 |
| humaneval sampled c=16 / nano_awq_dflash | 3378 | 3494 | 1.034x | 57.7 | 77.4 |

