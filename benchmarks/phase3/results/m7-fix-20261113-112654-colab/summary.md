# Phase 3 run m7-fix-20261113-112654-colab

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
| install_venv | 0 | 53 |
| install_study | 0 | 19 |
| download_model | 0 | 45 |
| download_w4_model | 0 | 17 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 47 |
| workloads | 0 | 42 |
| tests | 0 | 329 |
| serve | 0 | 1748 |
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
| nano_awq | 206 | 745 | 2525 | 6012 |
| nano_awq_dflash | 455 | 1364 | 3221 | 5891 |
| nano_awq_dflash4 | 512 | 1543 | 3381 | 5760 |
| nano_dflash | 224 | 753 | 2042 | 5125 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.82 | 4.96 | 5.37 | 9.46 |
| nano_awq_dflash | 2.08 | 2.57 | 4.10 | 9.89 |
| nano_awq_dflash4 | 1.87 | 2.29 | 3.95 | 10.11 |
| nano_dflash | 4.43 | 4.51 | 6.51 | 10.65 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 9.9 | 15.9 | 19.5 | 48.5 |
| nano_awq_dflash | 9.7 | 29.0 | 42.2 | 53.9 |
| nano_awq_dflash4 | 9.9 | 26.4 | 40.0 | 59.0 |
| nano_dflash | 18.0 | 53.5 | 69.8 | 102.3 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.21x | 1.83x | 1.28x | 0.98x |
| nano_awq_dflash4 | 2.48x | 2.07x | 1.34x | 0.96x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 204 | 730 | 2511 | 5123 |
| nano_awq_dflash | 677 | 1905 | 3612 | 5762 |
| nano_awq_dflash4 | 765 | 2028 | 3666 | 5708 |
| nano_dflash | 344 | 1122 | 2814 | 5564 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.83 | 5.09 | 5.76 | 9.92 |
| nano_awq_dflash | 1.41 | 1.82 | 3.94 | 9.20 |
| nano_awq_dflash4 | 1.25 | 1.74 | 3.82 | 9.29 |
| nano_dflash | 2.75 | 3.19 | 4.81 | 9.04 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 16.6 | 26.8 | 156.8 | 128.6 |
| nano_awq_dflash | 16.8 | 43.8 | 73.5 | 125.2 |
| nano_awq_dflash4 | 16.7 | 40.2 | 69.0 | 125.0 |
| nano_dflash | 23.0 | 59.1 | 93.3 | 181.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.31x | 2.61x | 1.44x | 1.12x |
| nano_awq_dflash4 | 3.74x | 2.78x | 1.46x | 1.11x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 204 | 2200 |
| nano_awq_dflash | 511 | 3234 |
| nano_awq_dflash4 | 578 | 3303 |
| nano_dflash | 258 | 2189 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.83 | 5.81 |
| nano_awq_dflash | 1.89 | 4.23 |
| nano_awq_dflash4 | 1.65 | 4.03 |
| nano_dflash | 3.84 | 6.01 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 15.7 | 25.2 |
| nano_awq_dflash | 18.4 | 48.8 |
| nano_awq_dflash4 | 17.5 | 46.2 |
| nano_dflash | 33.0 | 77.3 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.50x | 1.47x |
| nano_awq_dflash4 | 2.83x | 1.50x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 177 | 856 |
| nano_awq_dflash | 304 | 938 |
| nano_awq_dflash4 | 334 | 953 |
| nano_dflash | 174 | 878 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.88 | 15.63 |
| nano_awq_dflash | 2.45 | 13.34 |
| nano_awq_dflash4 | 2.16 | 13.68 |
| nano_dflash | 5.08 | 14.98 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 104.5 | 206.6 |
| nano_awq_dflash | 107.0 | 462.3 |
| nano_awq_dflash4 | 106.5 | 320.4 |
| nano_dflash | 100.3 | 252.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.72x | 1.10x |
| nano_awq_dflash4 | 1.89x | 1.11x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 205 | 2514 |
| nano_awq_dflash | 436 | 3168 |
| nano_dflash | 227 | 1971 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.84 | 5.41 |
| nano_awq_dflash | 2.21 | 4.24 |
| nano_dflash | 4.26 | 6.59 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.6 | 24.6 |
| nano_awq_dflash | 10.5 | 42.1 |
| nano_dflash | 18.2 | 72.3 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.13x | 1.26x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 205 | 2776 |
| nano_awq_dflash | 667 | 3513 |
| nano_dflash | 321 | 2732 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.86 | 5.54 |
| nano_awq_dflash | 1.39 | 4.00 |
| nano_dflash | 2.97 | 4.98 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.5 | 30.8 |
| nano_awq_dflash | 17.4 | 81.4 |
| nano_dflash | 23.1 | 105.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.26x | 1.27x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1321 | 2.47 | 3.47 |
| nano_awq_dflash | alpaca_c4 | 1452 | 2.30 | 3.30 |
| nano_awq_dflash | alpaca_c16 | 3876 | 1.91 | 2.91 |
| nano_awq_dflash | alpaca_c64 | 6415 | 1.16 | 2.16 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1008 | 4.59 | 5.59 |
| nano_awq_dflash | humaneval_c16 | 3664 | 3.39 | 4.39 |
| nano_awq_dflash | humaneval_c64 | 11126 | 2.46 | 3.46 |
| nano_awq_dflash | gsm8k_c1 | 1145 | 2.96 | 3.96 |
| nano_awq_dflash | gsm8k_c16 | 3308 | 2.10 | 3.10 |
| nano_awq_dflash | long_c1 | 1110 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 3356 | 1.64 | 2.64 |
| nano_awq_dflash | alpaca_c1s | 1355 | 2.34 | 3.34 |
| nano_awq_dflash | alpaca_c16s | 4082 | 1.82 | 2.82 |
| nano_awq_dflash | humaneval_c1s | 1088 | 4.28 | 5.28 |
| nano_awq_dflash | humaneval_c16s | 3858 | 3.20 | 4.20 |
| nano_awq_dflash4 | alpaca_c1 | 1352 | 2.39 | 3.39 |
| nano_awq_dflash4 | alpaca_c4 | 1479 | 2.28 | 3.28 |
| nano_awq_dflash4 | alpaca_c16 | 3977 | 1.85 | 2.85 |
| nano_awq_dflash4 | alpaca_c64 | 23067 | 0.88 | 1.88 |
| nano_awq_dflash4 | humaneval_c1 | 1107 | 4.26 | 5.26 |
| nano_awq_dflash4 | humaneval_c4 | 1020 | 4.52 | 5.52 |
| nano_awq_dflash4 | humaneval_c16 | 3806 | 3.23 | 4.23 |
| nano_awq_dflash4 | humaneval_c64 | 11023 | 2.43 | 3.43 |
| nano_awq_dflash4 | gsm8k_c1 | 1161 | 2.91 | 3.91 |
| nano_awq_dflash4 | gsm8k_c16 | 3281 | 2.08 | 3.08 |
| nano_awq_dflash4 | long_c1 | 1126 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 3372 | 1.61 | 2.61 |
| nano_dflash | alpaca_c1 | 1347 | 2.46 | 3.46 |
| nano_dflash | alpaca_c4 | 1308 | 2.66 | 3.66 |
| nano_dflash | alpaca_c16 | 3155 | 2.58 | 3.58 |
| nano_dflash | alpaca_c64 | 17617 | 1.88 | 2.88 |
| nano_dflash | humaneval_c1 | 1020 | 4.43 | 5.43 |
| nano_dflash | humaneval_c4 | 1035 | 4.32 | 5.32 |
| nano_dflash | humaneval_c16 | 2868 | 4.60 | 5.60 |
| nano_dflash | humaneval_c64 | 9390 | 3.27 | 4.27 |
| nano_dflash | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash | gsm8k_c16 | 2852 | 2.73 | 3.73 |
| nano_dflash | long_c1 | 1144 | 2.03 | 3.03 |
| nano_dflash | long_c16 | 3037 | 1.93 | 2.93 |
| nano_dflash | alpaca_c1s | 1304 | 2.52 | 3.52 |
| nano_dflash | alpaca_c16s | 3397 | 2.43 | 3.43 |
| nano_dflash | humaneval_c1s | 1139 | 4.07 | 5.07 |
| nano_dflash | humaneval_c16s | 2990 | 4.44 | 5.44 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.45 |
| nano_awq_dflash | alpaca_c4 | 1% | 0% | 0% | 0% | 97% | 2% | 3.24 |
| nano_awq_dflash | alpaca_c16 | 5% | 0% | 0% | 89% | 6% | 0% | 2.80 |
| nano_awq_dflash | alpaca_c64 | 85% | 8% | 4% | 2% | 0% | 0% | 1.17 |
| nano_awq_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.29 |
| nano_awq_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.47 |
| nano_awq_dflash | humaneval_c16 | 1% | 0% | 0% | 17% | 81% | 1% | 4.32 |
| nano_awq_dflash | humaneval_c64 | 17% | 0% | 0% | 82% | 1% | 0% | 3.02 |
| nano_awq_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.89 |
| nano_awq_dflash | gsm8k_c16 | 1% | 0% | 0% | 94% | 4% | 1% | 3.03 |
| nano_awq_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_awq_dflash | long_c16 | 3% | 0% | 0% | 95% | 2% | 0% | 2.57 |
| nano_awq_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.32 |
| nano_awq_dflash | alpaca_c16s | 1% | 0% | 0% | 94% | 5% | 0% | 2.77 |
| nano_awq_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.24 |
| nano_awq_dflash | humaneval_c16s | 1% | 0% | 0% | 25% | 73% | 2% | 4.14 |
| nano_awq_dflash4 | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.37 |
| nano_awq_dflash4 | alpaca_c4 | 1% | 0% | 0% | 5% | 93% | 1% | 3.23 |
| nano_awq_dflash4 | alpaca_c16 | 3% | 0% | 0% | 95% | 2% | 0% | 2.76 |
| nano_awq_dflash4 | alpaca_c64 | 25% | 68% | 5% | 2% | 0% | 0% | 1.65 |
| nano_awq_dflash4 | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.23 |
| nano_awq_dflash4 | humaneval_c4 | 1% | 0% | 0% | 0% | 0% | 99% | 5.38 |
| nano_awq_dflash4 | humaneval_c16 | 1% | 0% | 0% | 32% | 66% | 1% | 4.16 |
| nano_awq_dflash4 | humaneval_c64 | 20% | 0% | 3% | 75% | 1% | 0% | 2.93 |
| nano_awq_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.84 |
| nano_awq_dflash4 | gsm8k_c16 | 3% | 0% | 0% | 93% | 3% | 1% | 2.98 |
| nano_awq_dflash4 | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.91 |
| nano_awq_dflash4 | long_c16 | 5% | 0% | 0% | 94% | 1% | 0% | 2.51 |
| nano_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.45 |
| nano_dflash | alpaca_c4 | 1% | 0% | 0% | 0% | 2% | 97% | 3.59 |
| nano_dflash | alpaca_c16 | 2% | 0% | 0% | 0% | 72% | 26% | 3.47 |
| nano_dflash | alpaca_c64 | 3% | 0% | 0% | 93% | 2% | 2% | 2.81 |
| nano_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.39 |
| nano_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.21 |
| nano_dflash | humaneval_c16 | 1% | 0% | 0% | 0% | 0% | 99% | 5.46 |
| nano_dflash | humaneval_c64 | 2% | 0% | 0% | 28% | 68% | 3% | 4.18 |
| nano_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.99 |
| nano_dflash | gsm8k_c16 | 1% | 0% | 0% | 0% | 86% | 13% | 3.63 |
| nano_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_dflash | long_c16 | 3% | 0% | 0% | 0% | 96% | 2% | 2.85 |
| nano_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_dflash | alpaca_c16s | 1% | 0% | 0% | 0% | 77% | 23% | 3.38 |
| nano_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.03 |
| nano_dflash | humaneval_c16s | 1% | 0% | 0% | 0% | 0% | 99% | 5.34 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 455 | vllmw4_dflash 380 | 1.20x | met |
| alpaca / nano_awq_dflash4 | 512 | vllmw4_dflash 380 | 1.35x | met |
| alpaca / nano_dflash | 224 | vllm_dflash 220 | 1.02x | met |
| humaneval / nano_awq_dflash | 677 | vllmw4_dflash 589 | 1.15x | met |
| humaneval / nano_awq_dflash4 | 765 | vllmw4_dflash 589 | 1.30x | met |
| humaneval / nano_dflash | 344 | vllm_dflash 322 | 1.07x | met |
| gsm8k / nano_awq_dflash | 511 | vllmw4_dflash 434 | 1.18x | met |
| gsm8k / nano_awq_dflash4 | 578 | vllmw4_dflash 434 | 1.33x | met |
| gsm8k / nano_dflash | 258 | vllm_dflash 256 | 1.01x | met |
| long / nano_awq_dflash | 304 | vllmw4_dflash 245 | 1.24x | met |
| long / nano_awq_dflash4 | 334 | vllmw4_dflash 245 | 1.36x | met |
| long / nano_dflash | 174 | vllm_dflash 161 | 1.08x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 436 | vllmw4_dflash 371 | 1.18x | met |
| alpaca / nano_dflash | 227 | vllm_dflash 208 | 1.09x | met |
| humaneval / nano_awq_dflash | 667 | vllmw4_dflash 556 | 1.20x | met |
| humaneval / nano_dflash | 321 | vllm_dflash 310 | 1.04x | met |

## M6 gate: with a drafter at every concurrency

Output tok/s against the same nano server without a drafter in this run (at least 0.98x) and against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab) (above).

| workload / server | tok/s | no drafter | over no drafter | best baseline | over baseline |
|---|---:|---:|---:|---:|---:|
| alpaca c=1 / nano_awq_dflash | 455 | 206 | 2.21x | vllmw4_dflash 380 | 1.20x |
| alpaca c=4 / nano_awq_dflash | 1364 | 745 | 1.83x | vllmw4_dflash 1211 | 1.13x |
| alpaca c=16 / nano_awq_dflash | 3221 | 2525 | 1.28x | vllmw4_dflash 2084 | 1.55x |
| alpaca c=64 / nano_awq_dflash | 5891 | 6012 | 0.98x NOT met | vllmw4_dflash 2943 | 2.00x |
| alpaca c=1 / nano_awq_dflash4 | 512 | 206 | 2.48x | vllmw4_dflash 380 | 1.35x |
| alpaca c=4 / nano_awq_dflash4 | 1543 | 745 | 2.07x | vllmw4_dflash 1211 | 1.27x |
| alpaca c=16 / nano_awq_dflash4 | 3381 | 2525 | 1.34x | vllmw4_dflash 2084 | 1.62x |
| alpaca c=64 / nano_awq_dflash4 | 5760 | 6012 | 0.96x NOT met | vllmw4_dflash 2943 | 1.96x |
| alpaca c=1 / nano_dflash | 224 | 83 | 2.70x | vllm_dflash 220 | 1.02x |
| alpaca c=4 / nano_dflash | 753 | 311 | 2.42x | vllm_dflash 770 | 0.98x NOT met |
| alpaca c=16 / nano_dflash | 2042 | 1056 | 1.93x | vllm_dflash 1910 | 1.07x |
| alpaca c=64 / nano_dflash | 5125 | 3258 | 1.57x | vllm_dflash 3327 | 1.54x |
| humaneval c=1 / nano_awq_dflash | 677 | 204 | 3.31x | vllmw4_dflash 589 | 1.15x |
| humaneval c=4 / nano_awq_dflash | 1905 | 730 | 2.61x | vllmw4_dflash 1701 | 1.12x |
| humaneval c=16 / nano_awq_dflash | 3612 | 2511 | 1.44x | vllmw4_dflash 2950 | 1.22x |
| humaneval c=64 / nano_awq_dflash | 5762 | 5123 | 1.12x | vllmw4_dflash 4204 | 1.37x |
| humaneval c=1 / nano_awq_dflash4 | 765 | 204 | 3.74x | vllmw4_dflash 589 | 1.30x |
| humaneval c=4 / nano_awq_dflash4 | 2028 | 730 | 2.78x | vllmw4_dflash 1701 | 1.19x |
| humaneval c=16 / nano_awq_dflash4 | 3666 | 2511 | 1.46x | vllmw4_dflash 2950 | 1.24x |
| humaneval c=64 / nano_awq_dflash4 | 5708 | 5123 | 1.11x | vllmw4_dflash 4204 | 1.36x |
| humaneval c=1 / nano_dflash | 344 | 83 | 4.14x | vllm_dflash 322 | 1.07x |
| humaneval c=4 / nano_dflash | 1122 | 296 | 3.79x | vllm_dflash 1120 | 1.00x |
| humaneval c=16 / nano_dflash | 2814 | 1166 | 2.41x | vllm_dflash 2806 | 1.00x |
| humaneval c=64 / nano_dflash | 5564 | 2943 | 1.89x | vllm_dflash 4576 | 1.22x |
| gsm8k c=1 / nano_awq_dflash | 511 | 204 | 2.50x | vllmw4_dflash 434 | 1.18x |
| gsm8k c=16 / nano_awq_dflash | 3234 | 2200 | 1.47x | vllmw4_dflash 2312 | 1.40x |
| gsm8k c=1 / nano_awq_dflash4 | 578 | 204 | 2.83x | vllmw4_dflash 434 | 1.33x |
| gsm8k c=16 / nano_awq_dflash4 | 3303 | 2200 | 1.50x | vllmw4_dflash 2312 | 1.43x |
| gsm8k c=1 / nano_dflash | 258 | 83 | 3.12x | vllm_dflash 256 | 1.01x |
| gsm8k c=16 / nano_dflash | 2189 | 976 | 2.24x | vllm_dflash 2229 | 0.98x NOT met |
| long c=1 / nano_awq_dflash | 304 | 177 | 1.72x | vllmw4_dflash 245 | 1.24x |
| long c=16 / nano_awq_dflash | 938 | 856 | 1.10x | vllmw4_dflash 750 | 1.25x |
| long c=1 / nano_awq_dflash4 | 334 | 177 | 1.89x | vllmw4_dflash 245 | 1.36x |
| long c=16 / nano_awq_dflash4 | 953 | 856 | 1.11x | vllmw4_dflash 750 | 1.27x |
| long c=1 / nano_dflash | 174 | 79 | 2.21x | vllm_dflash 161 | 1.08x |
| long c=16 / nano_dflash | 878 | 670 | 1.31x | vllm_dflash 838 | 1.05x |
| alpaca sampled c=1 / nano_awq_dflash | 436 | 205 | 2.13x | vllmw4_dflash 371 | 1.18x |
| alpaca sampled c=16 / nano_awq_dflash | 3168 | 2514 | 1.26x | vllmw4_dflash 2040 | 1.55x |
| alpaca sampled c=1 / nano_dflash | 227 | 83 | 2.73x | vllm_dflash 208 | 1.09x |
| alpaca sampled c=16 / nano_dflash | 1971 | 1076 | 1.83x | vllm_dflash 1817 | 1.08x |
| humaneval sampled c=1 / nano_awq_dflash | 667 | 205 | 3.26x | vllmw4_dflash 556 | 1.20x |
| humaneval sampled c=16 / nano_awq_dflash | 3513 | 2776 | 1.27x | vllmw4_dflash 2869 | 1.22x |
| humaneval sampled c=1 / nano_dflash | 321 | 83 | 3.88x | vllm_dflash 310 | 1.04x |
| humaneval sampled c=16 / nano_dflash | 2732 | 1167 | 2.34x | vllm_dflash 2714 | 1.01x |

- nano_awq_dflash: at least 0.98x its own server without a drafter on 15 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_awq_dflash4: at least 0.98x its own server without a drafter on 11 of 12 workloads; above the best DFlash baseline on 12 of 12.
- nano_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 14 of 16.

