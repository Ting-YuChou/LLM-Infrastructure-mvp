# Phase 3 run m7-fix-20261113-104018-colab

```
load generator / nano: vllm 0.19.1
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 447: /content/p0/venvs/vllm_spec/bin/python: No such file or directory
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 448: /content/p0/venvs/sglang_spec/bin/python: No such file or directory
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 93 |
| install_study | 0 | 35 |
| download_model | 0 | 44 |
| download_w4_model | 0 | 15 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 22 |
| workloads | 0 | 28 |
| tests | 0 | 324 |
| serve | 0 | 1726 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq | 42 | 163 | 99 | 1223 |
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano_awq_dflash4 | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 206 | 751 | 2556 | 5981 |
| nano_awq_dflash | 455 | 1367 | 3213 | 5726 |
| nano_awq_dflash4 | 514 | 1573 | 3422 | 6033 |
| nano_dflash | 223 | 789 | 2064 | 5126 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.81 | 4.95 | 5.36 | 9.51 |
| nano_awq_dflash | 2.09 | 2.58 | 4.09 | 10.08 |
| nano_awq_dflash4 | 1.84 | 2.27 | 3.88 | 9.70 |
| nano_dflash | 4.42 | 4.51 | 6.51 | 10.68 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 9.6 | 15.9 | 19.7 | 49.0 |
| nano_awq_dflash | 9.7 | 28.8 | 41.1 | 78.4 |
| nano_awq_dflash4 | 9.7 | 26.7 | 39.6 | 54.8 |
| nano_dflash | 18.0 | 53.8 | 68.7 | 100.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.21x | 1.82x | 1.26x | 0.96x |
| nano_awq_dflash4 | 2.49x | 2.09x | 1.34x | 1.01x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 205 | 732 | 2508 | 5098 |
| nano_awq_dflash | 677 | 1836 | 3594 | 5835 |
| nano_awq_dflash4 | 766 | 2037 | 3692 | 5720 |
| nano_dflash | 344 | 1129 | 2815 | 5590 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.83 | 5.09 | 5.77 | 9.97 |
| nano_awq_dflash | 1.41 | 1.82 | 4.04 | 9.27 |
| nano_awq_dflash4 | 1.24 | 1.75 | 3.87 | 9.15 |
| nano_dflash | 2.75 | 3.14 | 4.79 | 9.28 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 16.7 | 26.7 | 147.4 | 134.0 |
| nano_awq_dflash | 16.8 | 41.4 | 72.9 | 157.0 |
| nano_awq_dflash4 | 16.8 | 38.5 | 62.2 | 138.3 |
| nano_dflash | 22.2 | 58.8 | 96.2 | 130.2 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.31x | 2.51x | 1.43x | 1.14x |
| nano_awq_dflash4 | 3.74x | 2.78x | 1.47x | 1.12x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 205 | 2192 |
| nano_awq_dflash | 511 | 3191 |
| nano_awq_dflash4 | 578 | 3320 |
| nano_dflash | 259 | 2202 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.82 | 5.82 |
| nano_awq_dflash | 1.89 | 4.32 |
| nano_awq_dflash4 | 1.65 | 4.14 |
| nano_dflash | 3.83 | 5.94 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 15.9 | 24.2 |
| nano_awq_dflash | 18.4 | 41.7 |
| nano_awq_dflash4 | 17.2 | 40.3 |
| nano_dflash | 33.0 | 76.2 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.50x | 1.46x |
| nano_awq_dflash4 | 2.82x | 1.51x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 177 | 859 |
| nano_awq_dflash | 305 | 944 |
| nano_awq_dflash4 | 335 | 961 |
| nano_dflash | 174 | 882 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.88 | 15.58 |
| nano_awq_dflash | 2.45 | 13.66 |
| nano_awq_dflash4 | 2.16 | 13.48 |
| nano_dflash | 5.08 | 15.41 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 104.7 | 212.5 |
| nano_awq_dflash | 105.9 | 343.9 |
| nano_awq_dflash4 | 106.0 | 313.7 |
| nano_dflash | 100.6 | 227.1 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.72x | 1.10x |
| nano_awq_dflash4 | 1.89x | 1.12x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 205 | 2485 |
| nano_awq_dflash | 436 | 3172 |
| nano_dflash | 227 | 1994 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.84 | 5.42 |
| nano_awq_dflash | 2.21 | 4.24 |
| nano_dflash | 4.26 | 6.54 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.7 | 19.8 |
| nano_awq_dflash | 10.3 | 42.2 |
| nano_dflash | 18.1 | 69.1 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.13x | 1.28x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 205 | 2795 |
| nano_awq_dflash | 669 | 3470 |
| nano_dflash | 322 | 2718 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.85 | 5.54 |
| nano_awq_dflash | 1.39 | 4.06 |
| nano_dflash | 2.98 | 5.02 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.8 | 28.0 |
| nano_awq_dflash | 16.8 | 72.9 |
| nano_dflash | 22.6 | 98.0 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.26x | 1.24x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1323 | 2.46 | 3.46 |
| nano_awq_dflash | alpaca_c4 | 1447 | 2.32 | 3.32 |
| nano_awq_dflash | alpaca_c16 | 3909 | 1.89 | 2.89 |
| nano_awq_dflash | alpaca_c64 | 12071 | 1.11 | 2.11 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1005 | 4.54 | 5.54 |
| nano_awq_dflash | humaneval_c16 | 3597 | 3.41 | 4.41 |
| nano_awq_dflash | humaneval_c64 | 11053 | 2.47 | 3.47 |
| nano_awq_dflash | gsm8k_c1 | 1145 | 2.96 | 3.96 |
| nano_awq_dflash | gsm8k_c16 | 3019 | 2.14 | 3.14 |
| nano_awq_dflash | long_c1 | 1110 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 3208 | 1.66 | 2.66 |
| nano_awq_dflash | alpaca_c1s | 1355 | 2.34 | 3.34 |
| nano_awq_dflash | alpaca_c16s | 4124 | 1.82 | 2.82 |
| nano_awq_dflash | humaneval_c1s | 1088 | 4.28 | 5.28 |
| nano_awq_dflash | humaneval_c16s | 3835 | 3.21 | 4.21 |
| nano_awq_dflash4 | alpaca_c1 | 1343 | 2.41 | 3.41 |
| nano_awq_dflash4 | alpaca_c4 | 1499 | 2.31 | 3.31 |
| nano_awq_dflash4 | alpaca_c16 | 4000 | 1.85 | 2.85 |
| nano_awq_dflash4 | alpaca_c64 | 2573 | 1.32 | 2.32 |
| nano_awq_dflash4 | humaneval_c1 | 1107 | 4.26 | 5.26 |
| nano_awq_dflash4 | humaneval_c4 | 1015 | 4.55 | 5.55 |
| nano_awq_dflash4 | humaneval_c16 | 3795 | 3.12 | 4.12 |
| nano_awq_dflash4 | humaneval_c64 | 11335 | 2.43 | 3.43 |
| nano_awq_dflash4 | gsm8k_c1 | 1161 | 2.91 | 3.91 |
| nano_awq_dflash4 | gsm8k_c16 | 3161 | 2.12 | 3.12 |
| nano_awq_dflash4 | long_c1 | 1126 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 3311 | 1.59 | 2.59 |
| nano_dflash | alpaca_c1 | 1355 | 2.44 | 3.44 |
| nano_dflash | alpaca_c4 | 1301 | 2.67 | 3.67 |
| nano_dflash | alpaca_c16 | 3125 | 2.61 | 3.61 |
| nano_dflash | alpaca_c64 | 17523 | 1.87 | 2.87 |
| nano_dflash | humaneval_c1 | 1020 | 4.43 | 5.43 |
| nano_dflash | humaneval_c4 | 1043 | 4.37 | 5.37 |
| nano_dflash | humaneval_c16 | 2889 | 4.57 | 5.57 |
| nano_dflash | humaneval_c64 | 9343 | 3.25 | 4.25 |
| nano_dflash | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash | gsm8k_c16 | 2828 | 2.73 | 3.73 |
| nano_dflash | long_c1 | 1144 | 2.03 | 3.03 |
| nano_dflash | long_c16 | 2932 | 1.96 | 2.96 |
| nano_dflash | alpaca_c1s | 1304 | 2.52 | 3.52 |
| nano_dflash | alpaca_c16s | 3497 | 2.33 | 3.33 |
| nano_dflash | humaneval_c1s | 1139 | 4.07 | 5.07 |
| nano_dflash | humaneval_c16s | 2978 | 4.41 | 5.41 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.44 |
| nano_awq_dflash | alpaca_c4 | 1% | 0% | 0% | 0% | 97% | 2% | 3.26 |
| nano_awq_dflash | alpaca_c16 | 4% | 0% | 0% | 90% | 6% | 0% | 2.79 |
| nano_awq_dflash | alpaca_c64 | 68% | 19% | 11% | 2% | 0% | 0% | 1.35 |
| nano_awq_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.29 |
| nano_awq_dflash | humaneval_c4 | 2% | 0% | 0% | 0% | 0% | 98% | 5.34 |
| nano_awq_dflash | humaneval_c16 | 6% | 0% | 0% | 14% | 79% | 2% | 4.18 |
| nano_awq_dflash | humaneval_c64 | 19% | 0% | 0% | 80% | 1% | 0% | 2.99 |
| nano_awq_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.89 |
| nano_awq_dflash | gsm8k_c16 | 18% | 0% | 0% | 78% | 1% | 2% | 2.71 |
| nano_awq_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_awq_dflash | long_c16 | 12% | 0% | 0% | 86% | 2% | 0% | 2.43 |
| nano_awq_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.32 |
| nano_awq_dflash | alpaca_c16s | 2% | 0% | 0% | 94% | 4% | 0% | 2.76 |
| nano_awq_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.24 |
| nano_awq_dflash | humaneval_c16s | 1% | 0% | 0% | 22% | 75% | 2% | 4.15 |
| nano_awq_dflash4 | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.39 |
| nano_awq_dflash4 | alpaca_c4 | 0% | 0% | 0% | 15% | 82% | 2% | 3.26 |
| nano_awq_dflash4 | alpaca_c16 | 3% | 0% | 0% | 95% | 2% | 0% | 2.78 |
| nano_awq_dflash4 | alpaca_c64 | 95% | 2% | 2% | 2% | 0% | 0% | 1.07 |
| nano_awq_dflash4 | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.23 |
| nano_awq_dflash4 | humaneval_c4 | 1% | 0% | 0% | 0% | 0% | 99% | 5.43 |
| nano_awq_dflash4 | humaneval_c16 | 11% | 0% | 0% | 35% | 54% | 1% | 3.75 |
| nano_awq_dflash4 | humaneval_c64 | 15% | 0% | 2% | 82% | 1% | 0% | 3.06 |
| nano_awq_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.84 |
| nano_awq_dflash4 | gsm8k_c16 | 15% | 0% | 0% | 81% | 4% | 1% | 2.76 |
| nano_awq_dflash4 | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.91 |
| nano_awq_dflash4 | long_c16 | 12% | 0% | 0% | 87% | 1% | 0% | 2.38 |
| nano_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.43 |
| nano_dflash | alpaca_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 3.60 |
| nano_dflash | alpaca_c16 | 1% | 0% | 0% | 0% | 67% | 32% | 3.54 |
| nano_dflash | alpaca_c64 | 2% | 0% | 0% | 94% | 2% | 1% | 2.81 |
| nano_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.39 |
| nano_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.27 |
| nano_dflash | humaneval_c16 | 1% | 0% | 0% | 0% | 0% | 99% | 5.45 |
| nano_dflash | humaneval_c64 | 7% | 0% | 0% | 27% | 63% | 3% | 4.01 |
| nano_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.99 |
| nano_dflash | gsm8k_c16 | 1% | 0% | 0% | 0% | 83% | 15% | 3.64 |
| nano_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_dflash | long_c16 | 10% | 0% | 0% | 0% | 88% | 2% | 2.73 |
| nano_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_dflash | alpaca_c16s | 1% | 0% | 0% | 0% | 94% | 5% | 3.28 |
| nano_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.03 |
| nano_dflash | humaneval_c16s | 1% | 0% | 0% | 0% | 0% | 99% | 5.29 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 455 | vllmw4_dflash 380 | 1.20x | met |
| alpaca / nano_awq_dflash4 | 514 | vllmw4_dflash 380 | 1.35x | met |
| alpaca / nano_dflash | 223 | vllm_dflash 220 | 1.01x | met |
| humaneval / nano_awq_dflash | 677 | vllmw4_dflash 589 | 1.15x | met |
| humaneval / nano_awq_dflash4 | 766 | vllmw4_dflash 589 | 1.30x | met |
| humaneval / nano_dflash | 344 | vllm_dflash 322 | 1.07x | met |
| gsm8k / nano_awq_dflash | 511 | vllmw4_dflash 434 | 1.18x | met |
| gsm8k / nano_awq_dflash4 | 578 | vllmw4_dflash 434 | 1.33x | met |
| gsm8k / nano_dflash | 259 | vllm_dflash 256 | 1.01x | met |
| long / nano_awq_dflash | 305 | vllmw4_dflash 245 | 1.25x | met |
| long / nano_awq_dflash4 | 335 | vllmw4_dflash 245 | 1.37x | met |
| long / nano_dflash | 174 | vllm_dflash 161 | 1.08x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 436 | vllmw4_dflash 371 | 1.18x | met |
| alpaca / nano_dflash | 227 | vllm_dflash 208 | 1.09x | met |
| humaneval / nano_awq_dflash | 669 | vllmw4_dflash 556 | 1.20x | met |
| humaneval / nano_dflash | 322 | vllm_dflash 310 | 1.04x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_awq_dflash | 455 | 206 | 2.21x | vllmw4_dflash 380 | 1.20x |
| alpaca c=4 / nano_awq_dflash | 1367 | 751 | 1.82x | vllmw4_dflash 1211 | 1.13x |
| alpaca c=16 / nano_awq_dflash | 3213 | 2556 | 1.26x | vllmw4_dflash 2084 | 1.54x |
| alpaca c=64 / nano_awq_dflash | 5726 | 5981 | 0.96x NOT met | vllmw4_dflash 2943 | 1.95x |
| alpaca c=1 / nano_awq_dflash4 | 514 | 206 | 2.49x | vllmw4_dflash 380 | 1.35x |
| alpaca c=4 / nano_awq_dflash4 | 1573 | 751 | 2.09x | vllmw4_dflash 1211 | 1.30x |
| alpaca c=16 / nano_awq_dflash4 | 3422 | 2556 | 1.34x | vllmw4_dflash 2084 | 1.64x |
| alpaca c=64 / nano_awq_dflash4 | 6033 | 5981 | 1.01x | vllmw4_dflash 2943 | 2.05x |
| alpaca c=1 / nano_dflash | 223 | 83 | 2.68x | vllm_dflash 220 | 1.01x |
| alpaca c=4 / nano_dflash | 789 | 311 | 2.54x | vllm_dflash 770 | 1.02x |
| alpaca c=16 / nano_dflash | 2064 | 1056 | 1.95x | vllm_dflash 1910 | 1.08x |
| alpaca c=64 / nano_dflash | 5126 | 3258 | 1.57x | vllm_dflash 3327 | 1.54x |
| humaneval c=1 / nano_awq_dflash | 677 | 205 | 3.31x | vllmw4_dflash 589 | 1.15x |
| humaneval c=4 / nano_awq_dflash | 1836 | 732 | 2.51x | vllmw4_dflash 1701 | 1.08x |
| humaneval c=16 / nano_awq_dflash | 3594 | 2508 | 1.43x | vllmw4_dflash 2950 | 1.22x |
| humaneval c=64 / nano_awq_dflash | 5835 | 5098 | 1.14x | vllmw4_dflash 4204 | 1.39x |
| humaneval c=1 / nano_awq_dflash4 | 766 | 205 | 3.74x | vllmw4_dflash 589 | 1.30x |
| humaneval c=4 / nano_awq_dflash4 | 2037 | 732 | 2.78x | vllmw4_dflash 1701 | 1.20x |
| humaneval c=16 / nano_awq_dflash4 | 3692 | 2508 | 1.47x | vllmw4_dflash 2950 | 1.25x |
| humaneval c=64 / nano_awq_dflash4 | 5720 | 5098 | 1.12x | vllmw4_dflash 4204 | 1.36x |
| humaneval c=1 / nano_dflash | 344 | 83 | 4.15x | vllm_dflash 322 | 1.07x |
| humaneval c=4 / nano_dflash | 1129 | 296 | 3.81x | vllm_dflash 1120 | 1.01x |
| humaneval c=16 / nano_dflash | 2815 | 1166 | 2.41x | vllm_dflash 2806 | 1.00x |
| humaneval c=64 / nano_dflash | 5590 | 2943 | 1.90x | vllm_dflash 4576 | 1.22x |
| gsm8k c=1 / nano_awq_dflash | 511 | 205 | 2.50x | vllmw4_dflash 434 | 1.18x |
| gsm8k c=16 / nano_awq_dflash | 3191 | 2192 | 1.46x | vllmw4_dflash 2312 | 1.38x |
| gsm8k c=1 / nano_awq_dflash4 | 578 | 205 | 2.82x | vllmw4_dflash 434 | 1.33x |
| gsm8k c=16 / nano_awq_dflash4 | 3320 | 2192 | 1.51x | vllmw4_dflash 2312 | 1.44x |
| gsm8k c=1 / nano_dflash | 259 | 83 | 3.13x | vllm_dflash 256 | 1.01x |
| gsm8k c=16 / nano_dflash | 2202 | 976 | 2.26x | vllm_dflash 2229 | 0.99x NOT met |
| long c=1 / nano_awq_dflash | 305 | 177 | 1.72x | vllmw4_dflash 245 | 1.25x |
| long c=16 / nano_awq_dflash | 944 | 859 | 1.10x | vllmw4_dflash 750 | 1.26x |
| long c=1 / nano_awq_dflash4 | 335 | 177 | 1.89x | vllmw4_dflash 245 | 1.37x |
| long c=16 / nano_awq_dflash4 | 961 | 859 | 1.12x | vllmw4_dflash 750 | 1.28x |
| long c=1 / nano_dflash | 174 | 79 | 2.21x | vllm_dflash 161 | 1.08x |
| long c=16 / nano_dflash | 882 | 670 | 1.32x | vllm_dflash 838 | 1.05x |
| alpaca sampled c=1 / nano_awq_dflash | 436 | 205 | 2.13x | vllmw4_dflash 371 | 1.18x |
| alpaca sampled c=16 / nano_awq_dflash | 3172 | 2485 | 1.28x | vllmw4_dflash 2040 | 1.55x |
| alpaca sampled c=1 / nano_dflash | 227 | 83 | 2.74x | vllm_dflash 208 | 1.09x |
| alpaca sampled c=16 / nano_dflash | 1994 | 1076 | 1.85x | vllm_dflash 1817 | 1.10x |
| humaneval sampled c=1 / nano_awq_dflash | 669 | 205 | 3.26x | vllmw4_dflash 556 | 1.20x |
| humaneval sampled c=16 / nano_awq_dflash | 3470 | 2795 | 1.24x | vllmw4_dflash 2869 | 1.21x |
| humaneval sampled c=1 / nano_dflash | 322 | 83 | 3.89x | vllm_dflash 310 | 1.04x |
| humaneval sampled c=16 / nano_dflash | 2718 | 1167 | 2.33x | vllm_dflash 2714 | 1.00x |

- nano_awq_dflash: at least 0.98x its own server without a drafter on 15 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_awq_dflash4: at least 0.98x its own server without a drafter on 12 of 12 workloads; above the best DFlash baseline on 12 of 12.
- nano_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 15 of 16.

