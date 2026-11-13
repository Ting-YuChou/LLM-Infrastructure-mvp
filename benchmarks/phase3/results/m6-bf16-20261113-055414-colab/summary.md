# Phase 3 run m6-bf16-20261113-055414-colab

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
| serve | 0 | 1651 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |
| nano_dflash4 | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 83 | 311 | 1048 | 3218 |
| nano_dflash | 218 | 759 | 2034 | 4933 |
| nano_dflash4 | 228 | 785 | 2025 | 4929 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 12.00 | 12.16 | 12.72 | 16.96 |
| nano_dflash | 4.55 | 4.72 | 6.60 | 11.02 |
| nano_dflash4 | 4.27 | 4.62 | 6.58 | 11.15 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 17.8 | 38.6 | 43.0 | 59.2 |
| nano_dflash | 17.8 | 37.7 | 47.7 | 77.5 |
| nano_dflash4 | 17.7 | 36.4 | 46.4 | 77.9 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_dflash | 2.62x | 2.44x | 1.94x | 1.53x |
| nano_dflash4 | 2.74x | 2.52x | 1.93x | 1.53x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 83 | 301 | 1165 | 2952 |
| nano_dflash | 334 | 1111 | 2768 | 5394 |
| nano_dflash4 | 356 | 1168 | 2734 | 5249 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 12.01 | 12.25 | 13.02 | 17.49 |
| nano_dflash | 2.84 | 3.29 | 4.90 | 9.43 |
| nano_dflash4 | 2.59 | 3.23 | 5.08 | 9.79 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 22.2 | 43.3 | 118.3 | 114.1 |
| nano_dflash | 22.0 | 42.1 | 68.5 | 143.5 |
| nano_dflash4 | 22.0 | 40.7 | 73.8 | 131.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_dflash | 4.03x | 3.69x | 2.38x | 1.83x |
| nano_dflash4 | 4.29x | 3.88x | 2.35x | 1.78x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 83 | 986 |
| nano_dflash | 254 | 2191 |
| nano_dflash4 | 270 | 2214 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.01 | 13.01 |
| nano_dflash | 3.96 | 6.11 |
| nano_dflash4 | 3.74 | 6.05 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 29.7 | 44.2 |
| nano_dflash | 21.4 | 54.4 |
| nano_dflash4 | 21.5 | 52.0 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 3.07x | 2.22x |
| nano_dflash4 | 3.26x | 2.24x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 79 | 663 |
| nano_dflash | 170 | 871 |
| nano_dflash4 | 180 | 882 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.06 | 21.06 |
| nano_dflash | 5.25 | 15.61 |
| nano_dflash4 | 4.80 | 15.01 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 95.5 | 150.7 |
| nano_dflash | 92.4 | 204.7 |
| nano_dflash4 | 92.8 | 190.3 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 2.17x | 1.31x |
| nano_dflash4 | 2.28x | 1.33x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 83 | 1080 |
| nano_dflash | 221 | 1922 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.02 | 12.76 |
| nano_dflash | 4.40 | 6.74 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 18.0 | 43.4 |
| nano_dflash | 18.0 | 49.4 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 2.67x | 1.78x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 83 | 1169 |
| nano_dflash | 312 | 2609 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.04 | 13.13 |
| nano_dflash | 3.08 | 5.38 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 22.6 | 76.0 |
| nano_dflash | 22.2 | 69.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 3.78x | 2.23x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_dflash | alpaca_c1 | 1347 | 2.46 | 3.46 |
| nano_dflash | alpaca_c4 | 1332 | 2.66 | 3.66 |
| nano_dflash | alpaca_c16 | 3164 | 2.59 | 3.59 |
| nano_dflash | alpaca_c64 | 17732 | 1.88 | 2.88 |
| nano_dflash | humaneval_c1 | 1020 | 4.43 | 5.43 |
| nano_dflash | humaneval_c4 | 1029 | 4.35 | 5.35 |
| nano_dflash | humaneval_c16 | 2880 | 4.59 | 5.59 |
| nano_dflash | humaneval_c64 | 9291 | 3.34 | 4.34 |
| nano_dflash | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash | gsm8k_c16 | 2822 | 2.76 | 3.76 |
| nano_dflash | long_c1 | 1144 | 2.03 | 3.03 |
| nano_dflash | long_c16 | 3053 | 1.93 | 2.93 |
| nano_dflash | alpaca_c1s | 1304 | 2.52 | 3.52 |
| nano_dflash | alpaca_c16s | 3357 | 2.47 | 3.47 |
| nano_dflash | humaneval_c1s | 1139 | 4.07 | 5.07 |
| nano_dflash | humaneval_c16s | 3042 | 4.34 | 5.34 |
| nano_dflash4 | alpaca_c1 | 1374 | 2.40 | 3.40 |
| nano_dflash4 | alpaca_c4 | 1365 | 2.58 | 3.58 |
| nano_dflash4 | alpaca_c16 | 3311 | 2.43 | 3.43 |
| nano_dflash4 | alpaca_c64 | 17792 | 1.86 | 2.86 |
| nano_dflash4 | humaneval_c1 | 1023 | 4.41 | 5.41 |
| nano_dflash4 | humaneval_c4 | 1058 | 4.31 | 5.31 |
| nano_dflash4 | humaneval_c16 | 2870 | 4.55 | 5.55 |
| nano_dflash4 | humaneval_c64 | 9334 | 3.32 | 4.32 |
| nano_dflash4 | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash4 | gsm8k_c16 | 2886 | 2.69 | 3.69 |
| nano_dflash4 | long_c1 | 1147 | 2.02 | 3.02 |
| nano_dflash4 | long_c16 | 3082 | 1.91 | 2.91 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|
| nano_dflash | alpaca_c1 | 0% | 0% | 100% | 3.46 |
| nano_dflash | alpaca_c4 | 0% | 1% | 99% | 3.66 |
| nano_dflash | alpaca_c16 | 0% | 74% | 26% | 3.59 |
| nano_dflash | alpaca_c64 | 96% | 2% | 2% | 2.88 |
| nano_dflash | humaneval_c1 | 0% | 0% | 100% | 5.43 |
| nano_dflash | humaneval_c4 | 0% | 0% | 100% | 5.35 |
| nano_dflash | humaneval_c16 | 0% | 0% | 100% | 5.59 |
| nano_dflash | humaneval_c64 | 23% | 73% | 4% | 4.34 |
| nano_dflash | gsm8k_c1 | 0% | 0% | 100% | 4.06 |
| nano_dflash | gsm8k_c16 | 0% | 82% | 18% | 3.76 |
| nano_dflash | long_c1 | 0% | 0% | 100% | 3.03 |
| nano_dflash | long_c16 | 0% | 97% | 3% | 2.93 |
| nano_dflash | alpaca_c1s | 0% | 0% | 100% | 3.52 |
| nano_dflash | alpaca_c16s | 0% | 73% | 27% | 3.47 |
| nano_dflash | humaneval_c1s | 0% | 0% | 100% | 5.07 |
| nano_dflash | humaneval_c16s | 0% | 0% | 100% | 5.34 |
| nano_dflash4 | alpaca_c1 | 0% | 0% | 100% | 3.40 |
| nano_dflash4 | alpaca_c4 | 0% | 2% | 98% | 3.58 |
| nano_dflash4 | alpaca_c16 | 0% | 94% | 6% | 3.43 |
| nano_dflash4 | alpaca_c64 | 96% | 4% | 0% | 2.86 |
| nano_dflash4 | humaneval_c1 | 0% | 0% | 100% | 5.41 |
| nano_dflash4 | humaneval_c4 | 0% | 0% | 100% | 5.31 |
| nano_dflash4 | humaneval_c16 | 0% | 0% | 100% | 5.55 |
| nano_dflash4 | humaneval_c64 | 18% | 79% | 3% | 4.32 |
| nano_dflash4 | gsm8k_c1 | 0% | 0% | 100% | 4.06 |
| nano_dflash4 | gsm8k_c16 | 0% | 94% | 6% | 3.69 |
| nano_dflash4 | long_c1 | 0% | 0% | 100% | 3.02 |
| nano_dflash4 | long_c16 | 2% | 95% | 3% | 2.91 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_dflash | 218 | vllm_dflash 220 | 0.99x | NOT met |
| alpaca / nano_dflash4 | 228 | vllm_dflash 220 | 1.03x | met |
| humaneval / nano_dflash | 334 | vllm_dflash 322 | 1.04x | met |
| humaneval / nano_dflash4 | 356 | vllm_dflash 322 | 1.10x | met |
| gsm8k / nano_dflash | 254 | vllm_dflash 256 | 0.99x | NOT met |
| gsm8k / nano_dflash4 | 270 | vllm_dflash 256 | 1.06x | met |
| long / nano_dflash | 170 | vllm_dflash 161 | 1.06x | met |
| long / nano_dflash4 | 180 | vllm_dflash 161 | 1.12x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_dflash | 221 | vllm_dflash 208 | 1.07x | met |
| humaneval / nano_dflash | 312 | vllm_dflash 310 | 1.01x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_dflash | 218 | 83 | 2.62x | vllm_dflash 220 | 0.99x NOT met |
| alpaca c=4 / nano_dflash | 759 | 311 | 2.44x | vllm_dflash 770 | 0.99x NOT met |
| alpaca c=16 / nano_dflash | 2034 | 1048 | 1.94x | vllm_dflash 1910 | 1.06x |
| alpaca c=64 / nano_dflash | 4933 | 3218 | 1.53x | vllm_dflash 3327 | 1.48x |
| alpaca c=1 / nano_dflash4 | 228 | 83 | 2.74x | vllm_dflash 220 | 1.03x |
| alpaca c=4 / nano_dflash4 | 785 | 311 | 2.52x | vllm_dflash 770 | 1.02x |
| alpaca c=16 / nano_dflash4 | 2025 | 1048 | 1.93x | vllm_dflash 1910 | 1.06x |
| alpaca c=64 / nano_dflash4 | 4929 | 3218 | 1.53x | vllm_dflash 3327 | 1.48x |
| humaneval c=1 / nano_dflash | 334 | 83 | 4.03x | vllm_dflash 322 | 1.04x |
| humaneval c=4 / nano_dflash | 1111 | 301 | 3.69x | vllm_dflash 1120 | 0.99x NOT met |
| humaneval c=16 / nano_dflash | 2768 | 1165 | 2.38x | vllm_dflash 2806 | 0.99x NOT met |
| humaneval c=64 / nano_dflash | 5394 | 2952 | 1.83x | vllm_dflash 4576 | 1.18x |
| humaneval c=1 / nano_dflash4 | 356 | 83 | 4.29x | vllm_dflash 322 | 1.10x |
| humaneval c=4 / nano_dflash4 | 1168 | 301 | 3.88x | vllm_dflash 1120 | 1.04x |
| humaneval c=16 / nano_dflash4 | 2734 | 1165 | 2.35x | vllm_dflash 2806 | 0.97x NOT met |
| humaneval c=64 / nano_dflash4 | 5249 | 2952 | 1.78x | vllm_dflash 4576 | 1.15x |
| gsm8k c=1 / nano_dflash | 254 | 83 | 3.07x | vllm_dflash 256 | 0.99x NOT met |
| gsm8k c=16 / nano_dflash | 2191 | 986 | 2.22x | vllm_dflash 2229 | 0.98x NOT met |
| gsm8k c=1 / nano_dflash4 | 270 | 83 | 3.26x | vllm_dflash 256 | 1.06x |
| gsm8k c=16 / nano_dflash4 | 2214 | 986 | 2.24x | vllm_dflash 2229 | 0.99x NOT met |
| long c=1 / nano_dflash | 170 | 79 | 2.17x | vllm_dflash 161 | 1.06x |
| long c=16 / nano_dflash | 871 | 663 | 1.31x | vllm_dflash 838 | 1.04x |
| long c=1 / nano_dflash4 | 180 | 79 | 2.28x | vllm_dflash 161 | 1.12x |
| long c=16 / nano_dflash4 | 882 | 663 | 1.33x | vllm_dflash 838 | 1.05x |
| alpaca sampled c=1 / nano_dflash | 221 | 83 | 2.67x | vllm_dflash 208 | 1.07x |
| alpaca sampled c=16 / nano_dflash | 1922 | 1080 | 1.78x | vllm_dflash 1817 | 1.06x |
| humaneval sampled c=1 / nano_dflash | 312 | 83 | 3.78x | vllm_dflash 310 | 1.01x |
| humaneval sampled c=16 / nano_dflash | 2609 | 1169 | 2.23x | vllm_dflash 2714 | 0.96x NOT met |

- nano_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 9 of 16.
- nano_dflash4: at least 0.98x its own server without a drafter on 12 of 12 workloads; above the best DFlash baseline on 10 of 12.

