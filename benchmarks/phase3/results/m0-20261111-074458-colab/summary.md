# Phase 3 run m0-20261111-074458-colab

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
| accept | 0 | 1233 |
| serve | 0 | 4754 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| vllm | 42 | 163 | 99 | 1223 |
| vllm_dflash | 42 | 163 | 99 | 1223 |
| vllmw4 | 42 | 163 | 99 | 1223 |
| vllmw4_dflash | 42 | 163 | 99 | 1223 |
| sglang | 42 | 163 | 99 | 1223 |
| sglang_dflash | 42 | 163 | 99 | 1223 |
| nano | 42 | 163 | 99 | 1223 |
| nano_awq | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm | 78 | 284 | 1001 | 3148 |
| vllm_dflash | 220 | 770 | 1910 | 3327 |
| vllmw4 | 175 | 638 | 2112 | 5092 |
| vllmw4_dflash | 380 | 1211 | 2084 | 2943 |
| sglang | 79 | 291 | 943 | 1615 |
| sglang_dflash | 196 | 670 | 1652 | 2915 |
| nano | 83 | 311 | 1056 | 3258 |
| nano_awq | 207 | 753 | 2567 | 6012 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm | 12.71 | 12.95 | 13.49 | 17.56 |
| vllm_dflash | 4.41 | 4.49 | 6.64 | 16.96 |
| vllmw4 | 5.64 | 5.85 | 6.49 | 11.21 |
| vllmw4_dflash | 2.43 | 2.86 | 6.00 | 19.85 |
| sglang | 12.52 | 12.97 | 14.21 | 37.68 |
| sglang_dflash | 4.96 | 5.15 | 7.99 | 14.73 |
| nano | 12.00 | 12.16 | 12.70 | 16.98 |
| nano_awq | 4.80 | 4.94 | 5.32 | 9.48 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm | 23.0 | 40.0 | 44.0 | 63.5 |
| vllm_dflash | 32.0 | 52.5 | 73.8 | 191.1 |
| vllmw4 | 18.6 | 20.1 | 27.8 | 79.5 |
| vllmw4_dflash | 24.1 | 31.0 | 71.8 | 214.3 |
| sglang | 28.9 | 45.7 | 57.0 | 100.7 |
| sglang_dflash | 39.4 | 57.3 | 71.5 | 1058.4 |
| nano | 17.9 | 38.6 | 42.8 | 60.9 |
| nano_awq | 9.6 | 15.9 | 24.0 | 47.7 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm_dflash | 2.82x | 2.71x | 1.91x | 1.06x |
| vllmw4_dflash | 2.17x | 1.90x | 0.99x | 0.58x |
| sglang_dflash | 2.48x | 2.30x | 1.75x | 1.81x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm | 78 | 281 | 1097 | 2907 |
| vllm_dflash | 322 | 1120 | 2806 | 4576 |
| vllmw4 | 174 | 605 | 2113 | 4563 |
| vllmw4_dflash | 589 | 1701 | 2950 | 4204 |
| sglang | 79 | 283 | 1075 | 1545 |
| sglang_dflash | 295 | 985 | 2456 | 4168 |
| nano | 83 | 296 | 1166 | 2943 |
| nano_awq | 206 | 734 | 2516 | 5159 |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm | 12.74 | 13.03 | 13.77 | 18.15 |
| vllm_dflash | 2.87 | 3.24 | 4.89 | 11.59 |
| vllmw4 | 5.68 | 5.99 | 7.04 | 11.48 |
| vllmw4_dflash | 1.59 | 2.06 | 4.72 | 12.87 |
| sglang | 12.55 | 13.11 | 14.14 | 37.27 |
| sglang_dflash | 3.17 | 3.65 | 5.71 | 9.69 |
| nano | 12.01 | 12.26 | 13.02 | 17.57 |
| nano_awq | 4.81 | 5.06 | 5.86 | 9.92 |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm | 27.2 | 44.6 | 116.9 | 133.4 |
| vllm_dflash | 35.5 | 57.1 | 90.8 | 210.7 |
| vllmw4 | 22.2 | 30.4 | 97.5 | 137.5 |
| vllmw4_dflash | 25.5 | 42.2 | 96.3 | 227.2 |
| sglang | 29.0 | 47.6 | 138.7 | 190.4 |
| sglang_dflash | 40.6 | 58.9 | 74.4 | 877.7 |
| nano | 21.9 | 43.4 | 117.6 | 87.0 |
| nano_awq | 16.5 | 26.7 | 127.3 | 134.4 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| vllm_dflash | 4.13x | 3.98x | 2.56x | 1.57x |
| vllmw4_dflash | 3.39x | 2.81x | 1.40x | 0.92x |
| sglang_dflash | 3.73x | 3.48x | 2.29x | 2.70x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 78 | 1006 |
| vllm_dflash | 256 | 2229 |
| vllmw4 | 174 | 1845 |
| vllmw4_dflash | 434 | 2312 |
| sglang | 79 | 879 |
| sglang_dflash | 222 | 1748 |
| nano | 83 | 976 |
| nano_awq | 205 | 2238 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 12.70 | 13.80 |
| vllm_dflash | 3.91 | 6.03 |
| vllmw4 | 5.64 | 7.01 |
| vllmw4_dflash | 2.15 | 5.89 |
| sglang | 12.53 | 14.72 |
| sglang_dflash | 4.35 | 7.57 |
| nano | 12.01 | 12.98 |
| nano_awq | 4.81 | 5.79 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 32.2 | 45.9 |
| vllm_dflash | 35.3 | 81.7 |
| vllmw4 | 21.4 | 30.9 |
| vllmw4_dflash | 24.5 | 81.4 |
| sglang | 35.6 | 47.2 |
| sglang_dflash | 40.5 | 72.8 |
| nano | 29.7 | 44.4 |
| nano_awq | 15.8 | 24.5 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 3.28x | 2.22x |
| vllmw4_dflash | 2.49x | 1.25x |
| sglang_dflash | 2.81x | 1.99x |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 74 | 645 |
| vllm_dflash | 161 | 838 |
| vllmw4 | 150 | 766 |
| vllmw4_dflash | 245 | 750 |
| sglang | 75 | 614 |
| sglang_dflash | 157 | 797 |
| nano | 79 | 670 |
| nano_awq | 178 | 870 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 12.86 | 21.85 |
| vllm_dflash | 5.51 | 15.56 |
| vllmw4 | 5.78 | 18.00 |
| vllmw4_dflash | 3.12 | 15.95 |
| sglang | 12.66 | 23.19 |
| sglang_dflash | 5.68 | 16.29 |
| nano | 12.06 | 21.19 |
| nano_awq | 4.86 | 15.24 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 98.7 | 178.3 |
| vllm_dflash | 102.3 | 287.3 |
| vllmw4 | 116.5 | 215.9 |
| vllmw4_dflash | 118.3 | 371.9 |
| sglang | 100.6 | 182.4 |
| sglang_dflash | 106.8 | 240.7 |
| nano | 95.3 | 167.5 |
| nano_awq | 103.8 | 198.2 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 2.17x | 1.30x |
| vllmw4_dflash | 1.63x | 0.98x |
| sglang_dflash | 2.10x | 1.30x |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 78 | 1024 |
| vllm_dflash | 208 | 1817 |
| vllmw4 | 172 | 2072 |
| vllmw4_dflash | 371 | 2040 |
| sglang | 79 | 948 |
| sglang_dflash | 186 | 1613 |
| nano | 83 | 1076 |
| nano_awq | 206 | 2540 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 12.78 | 13.60 |
| vllm_dflash | 4.49 | 6.87 |
| vllmw4 | 5.72 | 6.64 |
| vllmw4_dflash | 2.47 | 6.61 |
| sglang | 12.58 | 14.27 |
| sglang_dflash | 5.00 | 8.50 |
| nano | 12.02 | 12.77 |
| nano_awq | 4.82 | 5.38 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 22.0 | 44.4 |
| vllm_dflash | 32.1 | 76.7 |
| vllmw4 | 18.9 | 30.8 |
| vllmw4_dflash | 24.8 | 73.9 |
| sglang | 29.2 | 50.0 |
| sglang_dflash | 40.4 | 74.2 |
| nano | 17.9 | 44.8 |
| nano_awq | 9.4 | 25.4 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 2.67x | 1.77x |
| vllmw4_dflash | 2.15x | 0.98x |
| sglang_dflash | 2.36x | 1.70x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 78 | 1088 |
| vllm_dflash | 310 | 2714 |
| vllmw4 | 172 | 2283 |
| vllmw4_dflash | 556 | 2869 |
| sglang | 79 | 1084 |
| sglang_dflash | 287 | 2377 |
| nano | 83 | 1167 |
| nano_awq | 206 | 2803 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 12.82 | 13.93 |
| vllm_dflash | 3.05 | 5.14 |
| vllmw4 | 5.76 | 6.78 |
| vllmw4_dflash | 1.67 | 4.84 |
| sglang | 12.60 | 13.94 |
| sglang_dflash | 3.24 | 5.97 |
| nano | 12.04 | 13.13 |
| nano_awq | 4.83 | 5.53 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm | 26.2 | 118.4 |
| vllm_dflash | 35.9 | 92.3 |
| vllmw4 | 19.4 | 45.2 |
| vllmw4_dflash | 25.3 | 95.7 |
| sglang | 30.3 | 158.8 |
| sglang_dflash | 40.5 | 78.8 |
| nano | 22.5 | 90.2 |
| nano_awq | 9.4 | 30.6 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 3.99x | 2.49x |
| vllmw4_dflash | 3.23x | 1.26x |
| sglang_dflash | 3.65x | 2.19x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| vllm_dflash | alpaca_c1 | 1317 | 2.53 | 3.53 |
| vllm_dflash | alpaca_c4 | 1294 | 2.69 | 3.69 |
| vllm_dflash | alpaca_c16 | 3001 | 2.78 | 3.78 |
| vllm_dflash | alpaca_c64 | 14247 | 2.61 | 3.61 |
| vllm_dflash | humaneval_c1 | 1074 | 4.28 | 5.28 |
| vllm_dflash | humaneval_c4 | 1042 | 4.35 | 5.35 |
| vllm_dflash | humaneval_c16 | 2893 | 4.55 | 5.55 |
| vllm_dflash | humaneval_c64 | 7314 | 4.52 | 5.52 |
| vllm_dflash | gsm8k_c1 | 1037 | 3.15 | 4.15 |
| vllm_dflash | gsm8k_c16 | 2525 | 3.15 | 4.15 |
| vllm_dflash | long_c1 | 1135 | 2.03 | 3.03 |
| vllm_dflash | long_c16 | 2919 | 2.08 | 3.08 |
| vllm_dflash | alpaca_c1s | 1438 | 2.35 | 3.35 |
| vllm_dflash | alpaca_c16s | 3140 | 2.71 | 3.71 |
| vllm_dflash | humaneval_c1s | 1124 | 4.13 | 5.13 |
| vllm_dflash | humaneval_c16s | 2993 | 4.42 | 5.42 |
| vllmw4_dflash | alpaca_c1 | 1358 | 2.38 | 3.38 |
| vllmw4_dflash | alpaca_c4 | 1385 | 2.49 | 3.49 |
| vllmw4_dflash | alpaca_c16 | 3146 | 2.71 | 3.71 |
| vllmw4_dflash | alpaca_c64 | 14655 | 2.50 | 3.50 |
| vllmw4_dflash | humaneval_c1 | 1072 | 4.44 | 5.44 |
| vllmw4_dflash | humaneval_c4 | 1026 | 4.51 | 5.51 |
| vllmw4_dflash | humaneval_c16 | 2921 | 4.56 | 5.56 |
| vllmw4_dflash | humaneval_c64 | 7349 | 4.57 | 5.57 |
| vllmw4_dflash | gsm8k_c1 | 1124 | 2.90 | 3.90 |
| vllmw4_dflash | gsm8k_c16 | 2555 | 3.11 | 4.11 |
| vllmw4_dflash | long_c1 | 1084 | 2.04 | 3.04 |
| vllmw4_dflash | long_c16 | 2885 | 2.05 | 3.05 |
| vllmw4_dflash | alpaca_c1s | 1421 | 2.33 | 3.33 |
| vllmw4_dflash | alpaca_c16s | 3322 | 2.53 | 3.53 |
| vllmw4_dflash | humaneval_c1s | 1117 | 4.18 | 5.18 |
| vllmw4_dflash | humaneval_c16s | 2936 | 4.52 | 5.52 |

## Acceptance study (accept_study.py)

Tokens per cycle (accepted drafts + the bonus token), `drafter-draw-block` per column.

**bf16 target, greedy**

| dataset | bf16-greedy-b10 | bf16-greedy-b6 | bf16-greedy-b4 | rtn4-greedy-b10 |
|---|---:|---:|---:|---:|
| alpaca | 3.56 | 3.31 | 2.86 | 3.49 |
| humaneval | 5.21 | 4.26 | 3.37 | 5.13 |
| gsm8k | 4.14 | 3.70 | 3.06 | 4.16 |
| long | 3.05 | 2.93 | 2.60 | 2.97 |

**bf16 target, sampled**

| dataset | bf16-greedy-b10 | bf16-coupled-b10 | bf16-independent-b10 | bf16-coupled-b6 | bf16-coupled-b4 | rtn4-coupled-b10 |
|---|---:|---:|---:|---:|---:|---:|
| alpaca | 3.38 | 3.36 | 3.19 | 3.14 | 2.77 | 3.28 |
| humaneval | 5.09 | 5.03 | 4.92 | 4.26 | 3.33 | 5.03 |
| gsm8k | 3.92 | 3.91 | 3.77 | 3.56 | 3.00 | 3.88 |
| long | 3.06 | 3.10 | 2.87 | 2.93 | 2.69 | 3.02 |

**awq target, greedy**

| dataset | bf16-greedy-b10 | bf16-greedy-b6 | bf16-greedy-b4 | rtn4-greedy-b10 |
|---|---:|---:|---:|---:|
| alpaca | 3.47 | 3.25 | 2.79 | 3.45 |
| humaneval | 5.12 | 4.28 | 3.32 | 5.08 |
| gsm8k | 4.08 | 3.67 | 3.05 | 4.03 |
| long | 2.83 | 2.77 | 2.54 | 2.82 |

**awq target, sampled**

| dataset | bf16-greedy-b10 | bf16-coupled-b10 | bf16-independent-b10 | bf16-coupled-b6 | bf16-coupled-b4 | rtn4-coupled-b10 |
|---|---:|---:|---:|---:|---:|---:|
| alpaca | 3.33 | 3.26 | 3.11 | 3.04 | 2.68 | 3.20 |
| humaneval | 4.83 | 4.73 | 4.63 | 4.00 | 3.23 | 4.68 |
| gsm8k | 3.77 | 3.80 | 3.54 | 3.51 | 2.95 | 3.72 |
| long | 2.75 | 2.89 | 2.65 | 2.75 | 2.51 | 2.73 |

Survival by depth (the share of cycles that accept at least that many drafts), bf16 drafter, greedy drafts, block 10; the last columns: how often the drafter's second choice was the target's token at the first rejected depth, and how many prompts' replay matched the live loop.

| stream | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 2nd | replay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 greedy alpaca | 0.79 | 0.58 | 0.43 | 0.29 | 0.20 | 0.13 | 0.08 | 0.06 | 0.03 | 0.39 | 16/16 |
| bf16 greedy humaneval | 0.90 | 0.77 | 0.66 | 0.54 | 0.44 | 0.34 | 0.27 | 0.21 | 0.16 | 0.49 | 16/16 |
| bf16 greedy gsm8k | 0.84 | 0.67 | 0.50 | 0.38 | 0.28 | 0.20 | 0.15 | 0.11 | 0.07 | 0.43 | 16/16 |
| bf16 greedy long | 0.72 | 0.52 | 0.34 | 0.22 | 0.14 | 0.09 | 0.04 | 0.02 | 0.00 | 0.36 | 8/8 |
| bf16 sampled alpaca | 0.78 | 0.56 | 0.39 | 0.25 | 0.18 | 0.12 | 0.07 | 0.04 | 0.02 | 0.39 | 16/16 |
| bf16 sampled humaneval | 0.87 | 0.73 | 0.62 | 0.52 | 0.42 | 0.35 | 0.28 | 0.21 | 0.16 | 0.50 | 16/16 |
| bf16 sampled gsm8k | 0.81 | 0.63 | 0.47 | 0.36 | 0.27 | 0.17 | 0.12 | 0.09 | 0.05 | 0.42 | 16/16 |
| bf16 sampled long | 0.76 | 0.53 | 0.34 | 0.20 | 0.13 | 0.07 | 0.04 | 0.01 | 0.01 | 0.33 | 8/8 |
| awq greedy alpaca | 0.81 | 0.58 | 0.40 | 0.27 | 0.18 | 0.12 | 0.08 | 0.05 | 0.03 | 0.39 | 16/16 |
| awq greedy humaneval | 0.87 | 0.72 | 0.63 | 0.52 | 0.44 | 0.36 | 0.29 | 0.21 | 0.16 | 0.51 | 16/16 |
| awq greedy gsm8k | 0.84 | 0.67 | 0.50 | 0.37 | 0.27 | 0.20 | 0.14 | 0.09 | 0.06 | 0.43 | 16/16 |
| awq greedy long | 0.73 | 0.49 | 0.27 | 0.17 | 0.10 | 0.05 | 0.03 | 0.01 | 0.00 | 0.38 | 8/8 |
| awq sampled alpaca | 0.78 | 0.56 | 0.40 | 0.25 | 0.16 | 0.10 | 0.06 | 0.03 | 0.02 | 0.38 | 16/16 |
| awq sampled humaneval | 0.85 | 0.70 | 0.59 | 0.47 | 0.38 | 0.30 | 0.25 | 0.19 | 0.15 | 0.47 | 16/16 |
| awq sampled gsm8k | 0.81 | 0.61 | 0.45 | 0.33 | 0.23 | 0.17 | 0.10 | 0.07 | 0.05 | 0.37 | 16/16 |
| awq sampled long | 0.69 | 0.43 | 0.28 | 0.15 | 0.09 | 0.07 | 0.04 | 0.02 | 0.01 | 0.37 | 8/8 |

## M0 gate: projected nano at c=1

Cycle estimate: 1.08 x nano's own TPOT at c=1 (verify) + 2.1 ms (bf16 drafter) or 1.0 ms (4-bit drafter) + 0.3 ms; tokens per cycle from the acceptance study (greedy, block 10). The baseline is the faster DFlash server's measured output tok/s at c=1.

| target / dataset / drafter | nano TPOT (ms) | tokens/cycle | cycle (ms) | projected tok/s | best DFlash tok/s | ratio |
|---|---:|---:|---:|---:|---:|---:|
| bf16 / alpaca / bf16 drafter | 12.00 | 3.56 | 15.36 | 231 | 220 | 1.05x |
| bf16 / alpaca / rtn4 drafter | 12.00 | 3.49 | 14.26 | 245 | 220 | 1.11x |
| bf16 / humaneval / bf16 drafter | 12.01 | 5.21 | 15.37 | 339 | 322 | 1.05x |
| bf16 / humaneval / rtn4 drafter | 12.01 | 5.13 | 14.27 | 360 | 322 | 1.12x |
| bf16 / gsm8k / bf16 drafter | 12.01 | 4.14 | 15.37 | 270 | 256 | 1.05x |
| bf16 / gsm8k / rtn4 drafter | 12.01 | 4.16 | 14.27 | 292 | 256 | 1.14x |
| bf16 / long / bf16 drafter | 12.06 | 3.05 | 15.42 | 198 | 161 | 1.23x |
| bf16 / long / rtn4 drafter | 12.06 | 2.97 | 14.32 | 207 | 161 | 1.29x |
| awq / alpaca / bf16 drafter | 4.80 | 3.47 | 7.58 | 457 | 380 | 1.20x |
| awq / alpaca / rtn4 drafter | 4.80 | 3.45 | 6.48 | 532 | 380 | 1.40x |
| awq / humaneval / bf16 drafter | 4.81 | 5.12 | 7.59 | 674 | 589 | 1.14x |
| awq / humaneval / rtn4 drafter | 4.81 | 5.08 | 6.49 | 782 | 589 | 1.33x |
| awq / gsm8k / bf16 drafter | 4.81 | 4.08 | 7.59 | 537 | 434 | 1.24x |
| awq / gsm8k / rtn4 drafter | 4.81 | 4.03 | 6.49 | 621 | 434 | 1.43x |
| awq / long / bf16 drafter | 4.86 | 2.83 | 7.65 | 370 | 245 | 1.51x |
| awq / long / rtn4 drafter | 4.86 | 2.82 | 6.55 | 431 | 245 | 1.76x |

Gate (>= 1.2x on every row with the 4-bit drafter): NOT met, lowest 1.11x.

