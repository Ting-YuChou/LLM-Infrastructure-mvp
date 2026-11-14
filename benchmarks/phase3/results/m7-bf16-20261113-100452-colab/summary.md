# Phase 3 run m7-bf16-20261113-100452-colab

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
| serve | 0 | 2073 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |
| nano_dflashs | 42 | 163 | 99 | 1223 |
| nano_dflash4 | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 83 | 311 | 1051 | 3272 |
| nano_dflash | 223 | 778 | 2064 | 5219 |
| nano_dflashs | 218 | 758 | 1976 | 4937 |
| nano_dflash4 | 236 | 805 | 2101 | 5075 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 11.99 | 12.16 | 12.73 | 16.98 |
| nano_dflash | 4.43 | 4.54 | 6.53 | 10.62 |
| nano_dflashs | 4.55 | 4.73 | 6.76 | 11.10 |
| nano_dflash4 | 4.11 | 4.40 | 6.34 | 10.82 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 17.9 | 38.5 | 42.3 | 60.1 |
| nano_dflash | 17.9 | 53.8 | 68.3 | 99.3 |
| nano_dflashs | 17.6 | 37.8 | 50.0 | 79.7 |
| nano_dflash4 | 17.8 | 51.7 | 66.4 | 99.7 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_dflash | 2.68x | 2.50x | 1.96x | 1.59x |
| nano_dflashs | 2.63x | 2.44x | 1.88x | 1.51x |
| nano_dflash4 | 2.84x | 2.59x | 2.00x | 1.55x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 83 | 296 | 1159 | 2943 |
| nano_dflash | 344 | 1124 | 2839 | 5587 |
| nano_dflashs | 333 | 1108 | 2763 | 5347 |
| nano_dflash4 | 366 | 1193 | 2791 | 5614 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 12.01 | 12.26 | 13.07 | 17.59 |
| nano_dflash | 2.75 | 3.20 | 4.78 | 9.02 |
| nano_dflashs | 2.85 | 3.30 | 5.02 | 9.45 |
| nano_dflash4 | 2.50 | 3.13 | 4.80 | 9.15 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano | 22.1 | 43.3 | 101.3 | 111.8 |
| nano_dflash | 22.1 | 58.3 | 95.7 | 139.1 |
| nano_dflashs | 22.1 | 42.2 | 69.0 | 137.8 |
| nano_dflash4 | 22.1 | 56.0 | 98.3 | 135.0 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_dflash | 4.15x | 3.80x | 2.45x | 1.90x |
| nano_dflashs | 4.02x | 3.74x | 2.38x | 1.82x |
| nano_dflash4 | 4.42x | 4.02x | 2.41x | 1.91x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 83 | 976 |
| nano_dflash | 259 | 2240 |
| nano_dflashs | 251 | 2160 |
| nano_dflash4 | 276 | 2201 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.00 | 12.99 |
| nano_dflash | 3.83 | 5.92 |
| nano_dflashs | 3.98 | 6.19 |
| nano_dflash4 | 3.62 | 5.85 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 29.9 | 44.0 |
| nano_dflash | 32.8 | 73.5 |
| nano_dflashs | 21.7 | 55.3 |
| nano_dflash4 | 31.8 | 73.2 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 3.13x | 2.29x |
| nano_dflashs | 3.04x | 2.21x |
| nano_dflash4 | 3.34x | 2.25x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 79 | 662 |
| nano_dflash | 174 | 888 |
| nano_dflashs | 170 | 863 |
| nano_dflash4 | 184 | 886 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.06 | 21.20 |
| nano_dflash | 5.08 | 15.12 |
| nano_dflashs | 5.26 | 15.48 |
| nano_dflash4 | 4.64 | 15.16 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 95.4 | 158.5 |
| nano_dflash | 100.0 | 234.6 |
| nano_dflashs | 92.6 | 198.2 |
| nano_dflash4 | 99.1 | 225.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 2.21x | 1.34x |
| nano_dflashs | 2.16x | 1.30x |
| nano_dflash4 | 2.34x | 1.34x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 83 | 1073 |
| nano_dflash | 227 | 1974 |
| nano_dflashs | 221 | 1981 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.02 | 12.77 |
| nano_dflash | 4.26 | 6.59 |
| nano_dflashs | 4.40 | 6.77 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 18.1 | 42.2 |
| nano_dflash | 18.0 | 68.9 |
| nano_dflashs | 18.0 | 48.8 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 2.74x | 1.84x |
| nano_dflashs | 2.66x | 1.85x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 83 | 1166 |
| nano_dflash | 322 | 2731 |
| nano_dflashs | 313 | 2656 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 12.03 | 13.20 |
| nano_dflash | 2.98 | 4.94 |
| nano_dflashs | 3.07 | 5.25 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano | 22.2 | 66.1 |
| nano_dflash | 22.3 | 97.3 |
| nano_dflashs | 22.1 | 70.8 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 3.89x | 2.34x |
| nano_dflashs | 3.78x | 2.28x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_dflash | alpaca_c1 | 1355 | 2.44 | 3.44 |
| nano_dflash | alpaca_c4 | 1346 | 2.64 | 3.64 |
| nano_dflash | alpaca_c16 | 3184 | 2.55 | 3.55 |
| nano_dflash | alpaca_c64 | 17738 | 1.88 | 2.88 |
| nano_dflash | humaneval_c1 | 1020 | 4.43 | 5.43 |
| nano_dflash | humaneval_c4 | 1048 | 4.27 | 5.27 |
| nano_dflash | humaneval_c16 | 2867 | 4.61 | 5.61 |
| nano_dflash | humaneval_c64 | 9759 | 3.13 | 4.13 |
| nano_dflash | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash | gsm8k_c16 | 2784 | 2.73 | 3.73 |
| nano_dflash | long_c1 | 1144 | 2.03 | 3.03 |
| nano_dflash | long_c16 | 3041 | 1.96 | 2.96 |
| nano_dflash | alpaca_c1s | 1304 | 2.52 | 3.52 |
| nano_dflash | alpaca_c16s | 3439 | 2.37 | 3.37 |
| nano_dflash | humaneval_c1s | 1139 | 4.07 | 5.07 |
| nano_dflash | humaneval_c16s | 2980 | 4.45 | 5.45 |
| nano_dflashs | alpaca_c1 | 1347 | 2.46 | 3.46 |
| nano_dflashs | alpaca_c4 | 1332 | 2.66 | 3.66 |
| nano_dflashs | alpaca_c16 | 3142 | 2.60 | 3.60 |
| nano_dflashs | alpaca_c64 | 17649 | 1.89 | 2.89 |
| nano_dflashs | humaneval_c1 | 1020 | 4.43 | 5.43 |
| nano_dflashs | humaneval_c4 | 1029 | 4.35 | 5.35 |
| nano_dflashs | humaneval_c16 | 2882 | 4.59 | 5.59 |
| nano_dflashs | humaneval_c64 | 9373 | 3.30 | 4.30 |
| nano_dflashs | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflashs | gsm8k_c16 | 2751 | 2.83 | 3.83 |
| nano_dflashs | long_c1 | 1144 | 2.03 | 3.03 |
| nano_dflashs | long_c16 | 3003 | 1.95 | 2.95 |
| nano_dflashs | alpaca_c1s | 1304 | 2.52 | 3.52 |
| nano_dflashs | alpaca_c16s | 3383 | 2.46 | 3.46 |
| nano_dflashs | humaneval_c1s | 1139 | 4.07 | 5.07 |
| nano_dflashs | humaneval_c16s | 3021 | 4.37 | 5.37 |
| nano_dflash4 | alpaca_c1 | 1366 | 2.42 | 3.42 |
| nano_dflash4 | alpaca_c4 | 1347 | 2.62 | 3.62 |
| nano_dflash4 | alpaca_c16 | 3272 | 2.43 | 3.43 |
| nano_dflash4 | alpaca_c64 | 17693 | 1.86 | 2.86 |
| nano_dflash4 | humaneval_c1 | 1023 | 4.41 | 5.41 |
| nano_dflash4 | humaneval_c4 | 1044 | 4.35 | 5.35 |
| nano_dflash4 | humaneval_c16 | 2878 | 4.58 | 5.58 |
| nano_dflash4 | humaneval_c64 | 10938 | 2.67 | 3.67 |
| nano_dflash4 | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash4 | gsm8k_c16 | 2848 | 2.69 | 3.69 |
| nano_dflash4 | long_c1 | 1147 | 2.02 | 3.02 |
| nano_dflash4 | long_c16 | 3105 | 1.90 | 2.90 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|
| nano_dflash | alpaca_c1 | 0% | 0% | 0% | 100% | 3.43 |
| nano_dflash | alpaca_c4 | 0% | 0% | 1% | 99% | 3.59 |
| nano_dflash | alpaca_c16 | 0% | 0% | 77% | 23% | 3.50 |
| nano_dflash | alpaca_c64 | 1% | 95% | 2% | 1% | 2.84 |
| nano_dflash | humaneval_c1 | 0% | 0% | 0% | 100% | 5.39 |
| nano_dflash | humaneval_c4 | 0% | 0% | 0% | 100% | 5.18 |
| nano_dflash | humaneval_c16 | 0% | 0% | 0% | 100% | 5.54 |
| nano_dflash | humaneval_c64 | 0% | 42% | 54% | 4% | 4.10 |
| nano_dflash | gsm8k_c1 | 0% | 0% | 0% | 100% | 3.99 |
| nano_dflash | gsm8k_c16 | 0% | 0% | 87% | 13% | 3.67 |
| nano_dflash | long_c1 | 0% | 0% | 0% | 100% | 2.96 |
| nano_dflash | long_c16 | 0% | 0% | 97% | 3% | 2.92 |
| nano_dflash | alpaca_c1s | 0% | 0% | 0% | 100% | 3.50 |
| nano_dflash | alpaca_c16s | 0% | 0% | 93% | 7% | 3.33 |
| nano_dflash | humaneval_c1s | 0% | 0% | 0% | 100% | 5.03 |
| nano_dflash | humaneval_c16s | 0% | 0% | 0% | 100% | 5.39 |
| nano_dflashs | alpaca_c1 | 0% | 0% | 0% | 100% | 3.46 |
| nano_dflashs | alpaca_c4 | 0% | 0% | 1% | 99% | 3.66 |
| nano_dflashs | alpaca_c16 | 0% | 0% | 66% | 34% | 3.60 |
| nano_dflashs | alpaca_c64 | 0% | 96% | 3% | 1% | 2.89 |
| nano_dflashs | humaneval_c1 | 0% | 0% | 0% | 100% | 5.43 |
| nano_dflashs | humaneval_c4 | 0% | 0% | 0% | 100% | 5.35 |
| nano_dflashs | humaneval_c16 | 0% | 0% | 0% | 100% | 5.59 |
| nano_dflashs | humaneval_c64 | 0% | 25% | 72% | 3% | 4.30 |
| nano_dflashs | gsm8k_c1 | 0% | 0% | 0% | 100% | 4.06 |
| nano_dflashs | gsm8k_c16 | 0% | 0% | 67% | 33% | 3.83 |
| nano_dflashs | long_c1 | 0% | 0% | 0% | 100% | 3.03 |
| nano_dflashs | long_c16 | 2% | 0% | 96% | 3% | 2.91 |
| nano_dflashs | alpaca_c1s | 0% | 0% | 0% | 100% | 3.52 |
| nano_dflashs | alpaca_c16s | 0% | 0% | 73% | 27% | 3.46 |
| nano_dflashs | humaneval_c1s | 0% | 0% | 0% | 100% | 5.07 |
| nano_dflashs | humaneval_c16s | 0% | 0% | 0% | 100% | 5.37 |
| nano_dflash4 | alpaca_c1 | 0% | 0% | 0% | 100% | 3.40 |
| nano_dflash4 | alpaca_c4 | 0% | 0% | 1% | 99% | 3.57 |
| nano_dflash4 | alpaca_c16 | 0% | 0% | 93% | 7% | 3.38 |
| nano_dflash4 | alpaca_c64 | 1% | 95% | 3% | 0% | 2.82 |
| nano_dflash4 | humaneval_c1 | 0% | 0% | 0% | 100% | 5.37 |
| nano_dflash4 | humaneval_c4 | 0% | 0% | 0% | 100% | 5.25 |
| nano_dflash4 | humaneval_c16 | 0% | 0% | 0% | 100% | 5.50 |
| nano_dflash4 | humaneval_c64 | 0% | 78% | 19% | 2% | 3.65 |
| nano_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 100% | 3.99 |
| nano_dflash4 | gsm8k_c16 | 0% | 0% | 95% | 5% | 3.63 |
| nano_dflash4 | long_c1 | 0% | 0% | 0% | 100% | 2.96 |
| nano_dflash4 | long_c16 | 0% | 3% | 94% | 3% | 2.86 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_dflash | 223 | vllm_dflash 220 | 1.01x | met |
| alpaca / nano_dflash4 | 236 | vllm_dflash 220 | 1.07x | met |
| alpaca / nano_dflashs | 218 | vllm_dflash 220 | 0.99x | NOT met |
| humaneval / nano_dflash | 344 | vllm_dflash 322 | 1.07x | met |
| humaneval / nano_dflash4 | 366 | vllm_dflash 322 | 1.14x | met |
| humaneval / nano_dflashs | 333 | vllm_dflash 322 | 1.03x | met |
| gsm8k / nano_dflash | 259 | vllm_dflash 256 | 1.01x | met |
| gsm8k / nano_dflash4 | 276 | vllm_dflash 256 | 1.08x | met |
| gsm8k / nano_dflashs | 251 | vllm_dflash 256 | 0.98x | NOT met |
| long / nano_dflash | 174 | vllm_dflash 161 | 1.08x | met |
| long / nano_dflash4 | 184 | vllm_dflash 161 | 1.15x | met |
| long / nano_dflashs | 170 | vllm_dflash 161 | 1.06x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_dflash | 227 | vllm_dflash 208 | 1.09x | met |
| alpaca / nano_dflashs | 221 | vllm_dflash 208 | 1.06x | met |
| humaneval / nano_dflash | 322 | vllm_dflash 310 | 1.04x | met |
| humaneval / nano_dflashs | 313 | vllm_dflash 310 | 1.01x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_dflash | 223 | 83 | 2.68x | vllm_dflash 220 | 1.01x |
| alpaca c=4 / nano_dflash | 778 | 311 | 2.50x | vllm_dflash 770 | 1.01x |
| alpaca c=16 / nano_dflash | 2064 | 1051 | 1.96x | vllm_dflash 1910 | 1.08x |
| alpaca c=64 / nano_dflash | 5219 | 3272 | 1.59x | vllm_dflash 3327 | 1.57x |
| alpaca c=1 / nano_dflash4 | 236 | 83 | 2.84x | vllm_dflash 220 | 1.07x |
| alpaca c=4 / nano_dflash4 | 805 | 311 | 2.59x | vllm_dflash 770 | 1.05x |
| alpaca c=16 / nano_dflash4 | 2101 | 1051 | 2.00x | vllm_dflash 1910 | 1.10x |
| alpaca c=64 / nano_dflash4 | 5075 | 3272 | 1.55x | vllm_dflash 3327 | 1.53x |
| alpaca c=1 / nano_dflashs | 218 | 83 | 2.63x | vllm_dflash 220 | 0.99x NOT met |
| alpaca c=4 / nano_dflashs | 758 | 311 | 2.44x | vllm_dflash 770 | 0.98x NOT met |
| alpaca c=16 / nano_dflashs | 1976 | 1051 | 1.88x | vllm_dflash 1910 | 1.03x |
| alpaca c=64 / nano_dflashs | 4937 | 3272 | 1.51x | vllm_dflash 3327 | 1.48x |
| humaneval c=1 / nano_dflash | 344 | 83 | 4.15x | vllm_dflash 322 | 1.07x |
| humaneval c=4 / nano_dflash | 1124 | 296 | 3.80x | vllm_dflash 1120 | 1.00x |
| humaneval c=16 / nano_dflash | 2839 | 1159 | 2.45x | vllm_dflash 2806 | 1.01x |
| humaneval c=64 / nano_dflash | 5587 | 2943 | 1.90x | vllm_dflash 4576 | 1.22x |
| humaneval c=1 / nano_dflash4 | 366 | 83 | 4.42x | vllm_dflash 322 | 1.14x |
| humaneval c=4 / nano_dflash4 | 1193 | 296 | 4.02x | vllm_dflash 1120 | 1.06x |
| humaneval c=16 / nano_dflash4 | 2791 | 1159 | 2.41x | vllm_dflash 2806 | 0.99x NOT met |
| humaneval c=64 / nano_dflash4 | 5614 | 2943 | 1.91x | vllm_dflash 4576 | 1.23x |
| humaneval c=1 / nano_dflashs | 333 | 83 | 4.02x | vllm_dflash 322 | 1.03x |
| humaneval c=4 / nano_dflashs | 1108 | 296 | 3.74x | vllm_dflash 1120 | 0.99x NOT met |
| humaneval c=16 / nano_dflashs | 2763 | 1159 | 2.38x | vllm_dflash 2806 | 0.98x NOT met |
| humaneval c=64 / nano_dflashs | 5347 | 2943 | 1.82x | vllm_dflash 4576 | 1.17x |
| gsm8k c=1 / nano_dflash | 259 | 83 | 3.13x | vllm_dflash 256 | 1.01x |
| gsm8k c=16 / nano_dflash | 2240 | 976 | 2.29x | vllm_dflash 2229 | 1.00x |
| gsm8k c=1 / nano_dflash4 | 276 | 83 | 3.34x | vllm_dflash 256 | 1.08x |
| gsm8k c=16 / nano_dflash4 | 2201 | 976 | 2.25x | vllm_dflash 2229 | 0.99x NOT met |
| gsm8k c=1 / nano_dflashs | 251 | 83 | 3.04x | vllm_dflash 256 | 0.98x NOT met |
| gsm8k c=16 / nano_dflashs | 2160 | 976 | 2.21x | vllm_dflash 2229 | 0.97x NOT met |
| long c=1 / nano_dflash | 174 | 79 | 2.21x | vllm_dflash 161 | 1.08x |
| long c=16 / nano_dflash | 888 | 662 | 1.34x | vllm_dflash 838 | 1.06x |
| long c=1 / nano_dflash4 | 184 | 79 | 2.34x | vllm_dflash 161 | 1.15x |
| long c=16 / nano_dflash4 | 886 | 662 | 1.34x | vllm_dflash 838 | 1.06x |
| long c=1 / nano_dflashs | 170 | 79 | 2.16x | vllm_dflash 161 | 1.06x |
| long c=16 / nano_dflashs | 863 | 662 | 1.30x | vllm_dflash 838 | 1.03x |
| alpaca sampled c=1 / nano_dflash | 227 | 83 | 2.74x | vllm_dflash 208 | 1.09x |
| alpaca sampled c=16 / nano_dflash | 1974 | 1073 | 1.84x | vllm_dflash 1817 | 1.09x |
| alpaca sampled c=1 / nano_dflashs | 221 | 83 | 2.66x | vllm_dflash 208 | 1.06x |
| alpaca sampled c=16 / nano_dflashs | 1981 | 1073 | 1.85x | vllm_dflash 1817 | 1.09x |
| humaneval sampled c=1 / nano_dflash | 322 | 83 | 3.89x | vllm_dflash 310 | 1.04x |
| humaneval sampled c=16 / nano_dflash | 2731 | 1166 | 2.34x | vllm_dflash 2714 | 1.01x |
| humaneval sampled c=1 / nano_dflashs | 313 | 83 | 3.78x | vllm_dflash 310 | 1.01x |
| humaneval sampled c=16 / nano_dflashs | 2656 | 1166 | 2.28x | vllm_dflash 2714 | 0.98x NOT met |

- nano_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_dflash4: at least 0.98x its own server without a drafter on 12 of 12 workloads; above the best DFlash baseline on 10 of 12.
- nano_dflashs: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 9 of 16.

**Serving: the same server with SPECULATIVE_CHAIN=0 (each cycle collected) and as it is**

| workload / server | collected tok/s | chained tok/s | chained / collected | collected TTFT p50 (ms) | chained TTFT p50 (ms) |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_dflash | 218 | 223 | 1.020x | 17.6 | 17.9 |
| alpaca c=4 / nano_dflash | 758 | 778 | 1.026x | 37.8 | 53.8 |
| alpaca c=16 / nano_dflash | 1976 | 2064 | 1.045x | 50.0 | 68.3 |
| alpaca c=64 / nano_dflash | 4937 | 5219 | 1.057x | 79.7 | 99.3 |
| humaneval c=1 / nano_dflash | 333 | 344 | 1.034x | 22.1 | 22.1 |
| humaneval c=4 / nano_dflash | 1108 | 1124 | 1.015x | 42.2 | 58.3 |
| humaneval c=16 / nano_dflash | 2763 | 2839 | 1.028x | 69.0 | 95.7 |
| humaneval c=64 / nano_dflash | 5347 | 5587 | 1.045x | 137.8 | 139.1 |
| gsm8k c=1 / nano_dflash | 251 | 259 | 1.029x | 21.7 | 32.8 |
| gsm8k c=16 / nano_dflash | 2160 | 2240 | 1.037x | 55.3 | 73.5 |
| long c=1 / nano_dflash | 170 | 174 | 1.022x | 92.6 | 100.0 |
| long c=16 / nano_dflash | 863 | 888 | 1.029x | 198.2 | 234.6 |
| alpaca sampled c=1 / nano_dflash | 221 | 227 | 1.028x | 18.0 | 18.0 |
| alpaca sampled c=16 / nano_dflash | 1981 | 1974 | 0.996x | 48.8 | 68.9 |
| humaneval sampled c=1 / nano_dflash | 313 | 322 | 1.027x | 22.1 | 22.3 |
| humaneval sampled c=16 / nano_dflash | 2656 | 2731 | 1.028x | 70.8 | 97.3 |

