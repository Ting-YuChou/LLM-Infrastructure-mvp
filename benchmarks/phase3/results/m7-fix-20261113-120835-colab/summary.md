# Phase 3 run m7-fix-20261113-120835-colab

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
| install_venv | 0 | 306 |
| install_study | 0 | 54 |
| download_model | 0 | 42 |
| download_w4_model | 0 | 15 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 25 |
| workloads | 0 | 26 |
| tests | 0 | 316 |
| serve | 0 | 1709 |
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
| nano_awq | 207 | 748 | 2538 | 5978 |
| nano_awq_dflash | 455 | 1393 | 3252 | 5946 |
| nano_awq_dflash4 | 512 | 1550 | 3367 | 5974 |
| nano_dflash | 224 | 754 | 2041 | 5190 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.79 | 4.93 | 5.34 | 9.49 |
| nano_awq_dflash | 2.08 | 2.60 | 4.08 | 9.78 |
| nano_awq_dflash4 | 1.84 | 2.34 | 4.06 | 9.75 |
| nano_dflash | 4.43 | 4.51 | 6.48 | 10.56 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 9.7 | 15.9 | 18.9 | 47.5 |
| nano_awq_dflash | 9.6 | 28.4 | 41.7 | 56.6 |
| nano_awq_dflash4 | 9.6 | 26.6 | 39.8 | 65.9 |
| nano_dflash | 17.9 | 53.4 | 69.1 | 102.7 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.19x | 1.86x | 1.28x | 0.99x |
| nano_awq_dflash4 | 2.47x | 2.07x | 1.33x | 1.00x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 206 | 737 | 2523 | 5150 |
| nano_awq_dflash | 677 | 1848 | 3607 | 5697 |
| nano_awq_dflash4 | 764 | 2034 | 3771 | 5668 |
| nano_dflash | 344 | 1125 | 2772 | 5537 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 4.80 | 5.05 | 5.72 | 9.90 |
| nano_awq_dflash | 1.41 | 1.82 | 3.98 | 9.14 |
| nano_awq_dflash4 | 1.25 | 1.75 | 3.80 | 9.33 |
| nano_dflash | 2.75 | 3.18 | 4.87 | 9.28 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq | 16.6 | 26.7 | 150.3 | 142.7 |
| nano_awq_dflash | 16.8 | 42.1 | 73.8 | 148.2 |
| nano_awq_dflash4 | 16.8 | 41.8 | 70.2 | 142.5 |
| nano_dflash | 22.2 | 58.8 | 97.0 | 132.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.29x | 2.51x | 1.43x | 1.11x |
| nano_awq_dflash4 | 3.71x | 2.76x | 1.49x | 1.10x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2206 |
| nano_awq_dflash | 511 | 3223 |
| nano_awq_dflash4 | 576 | 3380 |
| nano_dflash | 259 | 2242 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.80 | 5.79 |
| nano_awq_dflash | 1.89 | 4.21 |
| nano_awq_dflash4 | 1.65 | 4.02 |
| nano_dflash | 3.83 | 5.94 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 15.9 | 24.3 |
| nano_awq_dflash | 18.5 | 51.7 |
| nano_awq_dflash4 | 17.4 | 46.2 |
| nano_dflash | 32.5 | 74.1 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.48x | 1.46x |
| nano_awq_dflash4 | 2.80x | 1.53x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 177 | 856 |
| nano_awq_dflash | 304 | 943 |
| nano_awq_dflash4 | 333 | 948 |
| nano_dflash | 174 | 868 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.86 | 15.47 |
| nano_awq_dflash | 2.46 | 13.41 |
| nano_awq_dflash4 | 2.17 | 13.53 |
| nano_dflash | 5.08 | 15.37 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 105.0 | 203.6 |
| nano_awq_dflash | 106.6 | 409.8 |
| nano_awq_dflash4 | 106.7 | 342.2 |
| nano_dflash | 101.6 | 222.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.71x | 1.10x |
| nano_awq_dflash4 | 1.88x | 1.11x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2526 |
| nano_awq_dflash | 436 | 3113 |
| nano_dflash | 227 | 1957 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.81 | 5.39 |
| nano_awq_dflash | 2.22 | 4.28 |
| nano_dflash | 4.26 | 6.52 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.6 | 21.4 |
| nano_awq_dflash | 9.8 | 41.5 |
| nano_dflash | 18.0 | 69.7 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.11x | 1.23x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 206 | 2810 |
| nano_awq_dflash | 667 | 3452 |
| nano_dflash | 321 | 2673 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 4.83 | 5.52 |
| nano_awq_dflash | 1.39 | 4.05 |
| nano_dflash | 2.98 | 5.11 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq | 9.5 | 32.3 |
| nano_awq_dflash | 17.0 | 72.8 |
| nano_dflash | 23.3 | 91.9 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.24x | 1.23x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1321 | 2.47 | 3.47 |
| nano_awq_dflash | alpaca_c4 | 1439 | 2.34 | 3.34 |
| nano_awq_dflash | alpaca_c16 | 3907 | 1.91 | 2.91 |
| nano_awq_dflash | alpaca_c64 | 1834 | 1.26 | 2.26 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1018 | 4.54 | 5.54 |
| nano_awq_dflash | humaneval_c16 | 3623 | 3.43 | 4.43 |
| nano_awq_dflash | humaneval_c64 | 11216 | 2.45 | 3.45 |
| nano_awq_dflash | gsm8k_c1 | 1145 | 2.96 | 3.96 |
| nano_awq_dflash | gsm8k_c16 | 3198 | 2.15 | 3.15 |
| nano_awq_dflash | long_c1 | 1110 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 3325 | 1.65 | 2.65 |
| nano_awq_dflash | alpaca_c1s | 1355 | 2.34 | 3.34 |
| nano_awq_dflash | alpaca_c16s | 4170 | 1.80 | 2.80 |
| nano_awq_dflash | humaneval_c1s | 1088 | 4.28 | 5.28 |
| nano_awq_dflash | humaneval_c16s | 3909 | 3.13 | 4.13 |
| nano_awq_dflash4 | alpaca_c1 | 1349 | 2.40 | 3.40 |
| nano_awq_dflash4 | alpaca_c4 | 1455 | 2.31 | 3.31 |
| nano_awq_dflash4 | alpaca_c16 | 4004 | 1.85 | 2.85 |
| nano_awq_dflash4 | alpaca_c64 | 2503 | 1.31 | 2.31 |
| nano_awq_dflash4 | humaneval_c1 | 1107 | 4.26 | 5.26 |
| nano_awq_dflash4 | humaneval_c4 | 1021 | 4.52 | 5.52 |
| nano_awq_dflash4 | humaneval_c16 | 4017 | 2.97 | 3.97 |
| nano_awq_dflash4 | humaneval_c64 | 10853 | 2.43 | 3.43 |
| nano_awq_dflash4 | gsm8k_c1 | 1161 | 2.91 | 3.91 |
| nano_awq_dflash4 | gsm8k_c16 | 3328 | 2.12 | 3.12 |
| nano_awq_dflash4 | long_c1 | 1126 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 3368 | 1.60 | 2.60 |
| nano_dflash | alpaca_c1 | 1347 | 2.46 | 3.46 |
| nano_dflash | alpaca_c4 | 1308 | 2.66 | 3.66 |
| nano_dflash | alpaca_c16 | 3158 | 2.57 | 3.57 |
| nano_dflash | alpaca_c64 | 17570 | 1.88 | 2.88 |
| nano_dflash | humaneval_c1 | 1020 | 4.43 | 5.43 |
| nano_dflash | humaneval_c4 | 1035 | 4.32 | 5.32 |
| nano_dflash | humaneval_c16 | 2861 | 4.56 | 5.56 |
| nano_dflash | humaneval_c64 | 9674 | 3.14 | 4.14 |
| nano_dflash | gsm8k_c1 | 1064 | 3.06 | 4.06 |
| nano_dflash | gsm8k_c16 | 2800 | 2.78 | 3.78 |
| nano_dflash | long_c1 | 1144 | 2.03 | 3.03 |
| nano_dflash | long_c16 | 3032 | 1.94 | 2.94 |
| nano_dflash | alpaca_c1s | 1304 | 2.52 | 3.52 |
| nano_dflash | alpaca_c16s | 3467 | 2.35 | 3.35 |
| nano_dflash | humaneval_c1s | 1139 | 4.07 | 5.07 |
| nano_dflash | humaneval_c16s | 3026 | 4.35 | 5.35 |

## Verify lengths the policy ran

The share of a server's decode rows (a row per sequence and step) at each verify length: 1 is a plain decode step, Q a cycle of the anchor and Q - 1 drafts. Tokens per row: what a decode row yielded on average.

| server | run | Q=1 | Q=2 | Q=3 | Q=4 | Q=6 | Q=10 | tokens per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.45 |
| nano_awq_dflash | alpaca_c4 | 1% | 0% | 0% | 0% | 97% | 2% | 3.28 |
| nano_awq_dflash | alpaca_c16 | 5% | 0% | 0% | 90% | 5% | 0% | 2.79 |
| nano_awq_dflash | alpaca_c64 | 96% | 2% | 1% | 1% | 0% | 0% | 1.05 |
| nano_awq_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.29 |
| nano_awq_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.42 |
| nano_awq_dflash | humaneval_c16 | 3% | 0% | 0% | 13% | 82% | 2% | 4.29 |
| nano_awq_dflash | humaneval_c64 | 16% | 0% | 0% | 83% | 1% | 0% | 3.06 |
| nano_awq_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.89 |
| nano_awq_dflash | gsm8k_c16 | 5% | 0% | 0% | 91% | 3% | 1% | 3.01 |
| nano_awq_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_awq_dflash | long_c16 | 6% | 0% | 0% | 92% | 2% | 0% | 2.52 |
| nano_awq_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.32 |
| nano_awq_dflash | alpaca_c16s | 2% | 0% | 0% | 93% | 5% | 0% | 2.73 |
| nano_awq_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.24 |
| nano_awq_dflash | humaneval_c16s | 1% | 0% | 0% | 29% | 69% | 1% | 4.08 |
| nano_awq_dflash4 | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.38 |
| nano_awq_dflash4 | alpaca_c4 | 1% | 0% | 0% | 5% | 93% | 2% | 3.26 |
| nano_awq_dflash4 | alpaca_c16 | 4% | 0% | 0% | 94% | 2% | 0% | 2.76 |
| nano_awq_dflash4 | alpaca_c64 | 95% | 2% | 2% | 2% | 0% | 0% | 1.07 |
| nano_awq_dflash4 | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.23 |
| nano_awq_dflash4 | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.40 |
| nano_awq_dflash4 | humaneval_c16 | 3% | 0% | 0% | 54% | 42% | 1% | 3.87 |
| nano_awq_dflash4 | humaneval_c64 | 24% | 1% | 2% | 72% | 1% | 0% | 2.83 |
| nano_awq_dflash4 | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.84 |
| nano_awq_dflash4 | gsm8k_c16 | 1% | 0% | 0% | 94% | 5% | 0% | 3.06 |
| nano_awq_dflash4 | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.91 |
| nano_awq_dflash4 | long_c16 | 6% | 0% | 0% | 92% | 1% | 0% | 2.48 |
| nano_dflash | alpaca_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.45 |
| nano_dflash | alpaca_c4 | 1% | 0% | 0% | 0% | 2% | 97% | 3.59 |
| nano_dflash | alpaca_c16 | 2% | 0% | 0% | 0% | 76% | 22% | 3.47 |
| nano_dflash | alpaca_c64 | 3% | 0% | 0% | 94% | 2% | 1% | 2.81 |
| nano_dflash | humaneval_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 5.39 |
| nano_dflash | humaneval_c4 | 0% | 0% | 0% | 0% | 0% | 100% | 5.21 |
| nano_dflash | humaneval_c16 | 1% | 0% | 0% | 0% | 0% | 99% | 5.42 |
| nano_dflash | humaneval_c64 | 2% | 0% | 0% | 40% | 54% | 4% | 4.05 |
| nano_dflash | gsm8k_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 3.99 |
| nano_dflash | gsm8k_c16 | 1% | 0% | 0% | 0% | 86% | 13% | 3.69 |
| nano_dflash | long_c1 | 0% | 0% | 0% | 0% | 0% | 100% | 2.96 |
| nano_dflash | long_c16 | 2% | 0% | 0% | 0% | 95% | 3% | 2.86 |
| nano_dflash | alpaca_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 3.50 |
| nano_dflash | alpaca_c16s | 1% | 0% | 0% | 0% | 95% | 4% | 3.28 |
| nano_dflash | humaneval_c1s | 0% | 0% | 0% | 0% | 0% | 100% | 5.03 |
| nano_dflash | humaneval_c16s | 1% | 0% | 0% | 0% | 0% | 99% | 5.23 |

## M4 and M5 gates

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 455 | vllmw4_dflash 380 | 1.20x | met |
| alpaca / nano_awq_dflash4 | 512 | vllmw4_dflash 380 | 1.35x | met |
| alpaca / nano_dflash | 224 | vllm_dflash 220 | 1.02x | met |
| humaneval / nano_awq_dflash | 677 | vllmw4_dflash 589 | 1.15x | met |
| humaneval / nano_awq_dflash4 | 764 | vllmw4_dflash 589 | 1.30x | met |
| humaneval / nano_dflash | 344 | vllm_dflash 322 | 1.07x | met |
| gsm8k / nano_awq_dflash | 511 | vllmw4_dflash 434 | 1.18x | met |
| gsm8k / nano_awq_dflash4 | 576 | vllmw4_dflash 434 | 1.33x | met |
| gsm8k / nano_dflash | 259 | vllm_dflash 256 | 1.01x | met |
| long / nano_awq_dflash | 304 | vllmw4_dflash 245 | 1.24x | met |
| long / nano_awq_dflash4 | 333 | vllmw4_dflash 245 | 1.36x | met |
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
| alpaca c=1 / nano_awq_dflash | 455 | 207 | 2.19x | vllmw4_dflash 380 | 1.20x |
| alpaca c=4 / nano_awq_dflash | 1393 | 748 | 1.86x | vllmw4_dflash 1211 | 1.15x |
| alpaca c=16 / nano_awq_dflash | 3252 | 2538 | 1.28x | vllmw4_dflash 2084 | 1.56x |
| alpaca c=64 / nano_awq_dflash | 5946 | 5978 | 0.99x | vllmw4_dflash 2943 | 2.02x |
| alpaca c=1 / nano_awq_dflash4 | 512 | 207 | 2.47x | vllmw4_dflash 380 | 1.35x |
| alpaca c=4 / nano_awq_dflash4 | 1550 | 748 | 2.07x | vllmw4_dflash 1211 | 1.28x |
| alpaca c=16 / nano_awq_dflash4 | 3367 | 2538 | 1.33x | vllmw4_dflash 2084 | 1.62x |
| alpaca c=64 / nano_awq_dflash4 | 5974 | 5978 | 1.00x | vllmw4_dflash 2943 | 2.03x |
| alpaca c=1 / nano_dflash | 224 | 83 | 2.70x | vllm_dflash 220 | 1.02x |
| alpaca c=4 / nano_dflash | 754 | 311 | 2.43x | vllm_dflash 770 | 0.98x NOT met |
| alpaca c=16 / nano_dflash | 2041 | 1056 | 1.93x | vllm_dflash 1910 | 1.07x |
| alpaca c=64 / nano_dflash | 5190 | 3258 | 1.59x | vllm_dflash 3327 | 1.56x |
| humaneval c=1 / nano_awq_dflash | 677 | 206 | 3.29x | vllmw4_dflash 589 | 1.15x |
| humaneval c=4 / nano_awq_dflash | 1848 | 737 | 2.51x | vllmw4_dflash 1701 | 1.09x |
| humaneval c=16 / nano_awq_dflash | 3607 | 2523 | 1.43x | vllmw4_dflash 2950 | 1.22x |
| humaneval c=64 / nano_awq_dflash | 5697 | 5150 | 1.11x | vllmw4_dflash 4204 | 1.36x |
| humaneval c=1 / nano_awq_dflash4 | 764 | 206 | 3.71x | vllmw4_dflash 589 | 1.30x |
| humaneval c=4 / nano_awq_dflash4 | 2034 | 737 | 2.76x | vllmw4_dflash 1701 | 1.20x |
| humaneval c=16 / nano_awq_dflash4 | 3771 | 2523 | 1.49x | vllmw4_dflash 2950 | 1.28x |
| humaneval c=64 / nano_awq_dflash4 | 5668 | 5150 | 1.10x | vllmw4_dflash 4204 | 1.35x |
| humaneval c=1 / nano_dflash | 344 | 83 | 4.15x | vllm_dflash 322 | 1.07x |
| humaneval c=4 / nano_dflash | 1125 | 296 | 3.80x | vllm_dflash 1120 | 1.00x |
| humaneval c=16 / nano_dflash | 2772 | 1166 | 2.38x | vllm_dflash 2806 | 0.99x NOT met |
| humaneval c=64 / nano_dflash | 5537 | 2943 | 1.88x | vllm_dflash 4576 | 1.21x |
| gsm8k c=1 / nano_awq_dflash | 511 | 206 | 2.48x | vllmw4_dflash 434 | 1.18x |
| gsm8k c=16 / nano_awq_dflash | 3223 | 2206 | 1.46x | vllmw4_dflash 2312 | 1.39x |
| gsm8k c=1 / nano_awq_dflash4 | 576 | 206 | 2.80x | vllmw4_dflash 434 | 1.33x |
| gsm8k c=16 / nano_awq_dflash4 | 3380 | 2206 | 1.53x | vllmw4_dflash 2312 | 1.46x |
| gsm8k c=1 / nano_dflash | 259 | 83 | 3.13x | vllm_dflash 256 | 1.01x |
| gsm8k c=16 / nano_dflash | 2242 | 976 | 2.30x | vllm_dflash 2229 | 1.01x |
| long c=1 / nano_awq_dflash | 304 | 177 | 1.71x | vllmw4_dflash 245 | 1.24x |
| long c=16 / nano_awq_dflash | 943 | 856 | 1.10x | vllmw4_dflash 750 | 1.26x |
| long c=1 / nano_awq_dflash4 | 333 | 177 | 1.88x | vllmw4_dflash 245 | 1.36x |
| long c=16 / nano_awq_dflash4 | 948 | 856 | 1.11x | vllmw4_dflash 750 | 1.26x |
| long c=1 / nano_dflash | 174 | 79 | 2.21x | vllm_dflash 161 | 1.08x |
| long c=16 / nano_dflash | 868 | 670 | 1.30x | vllm_dflash 838 | 1.04x |
| alpaca sampled c=1 / nano_awq_dflash | 436 | 206 | 2.11x | vllmw4_dflash 371 | 1.18x |
| alpaca sampled c=16 / nano_awq_dflash | 3113 | 2526 | 1.23x | vllmw4_dflash 2040 | 1.53x |
| alpaca sampled c=1 / nano_dflash | 227 | 83 | 2.74x | vllm_dflash 208 | 1.09x |
| alpaca sampled c=16 / nano_dflash | 1957 | 1076 | 1.82x | vllm_dflash 1817 | 1.08x |
| humaneval sampled c=1 / nano_awq_dflash | 667 | 206 | 3.24x | vllmw4_dflash 556 | 1.20x |
| humaneval sampled c=16 / nano_awq_dflash | 3452 | 2810 | 1.23x | vllmw4_dflash 2869 | 1.20x |
| humaneval sampled c=1 / nano_dflash | 321 | 83 | 3.89x | vllm_dflash 310 | 1.04x |
| humaneval sampled c=16 / nano_dflash | 2673 | 1167 | 2.29x | vllm_dflash 2714 | 0.99x NOT met |

- nano_awq_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 16 of 16.
- nano_awq_dflash4: at least 0.98x its own server without a drafter on 12 of 12 workloads; above the best DFlash baseline on 12 of 12.
- nano_dflash: at least 0.98x its own server without a drafter on 16 of 16 workloads; above the best DFlash baseline on 13 of 16.

