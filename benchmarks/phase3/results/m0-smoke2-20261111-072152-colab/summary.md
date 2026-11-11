# Phase 3 run m0-smoke2-20261111-072152-colab

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
| install_venv | 0 | 4 |
| install_study | 0 | 3 |
| install_vllm_spec | 0 | 5 |
| install_sglang_spec | 0 | 8 |
| serve | 0 | 1331 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| vllm_dflash | 42 | 163 | 99 | 1223 |
| vllmw4_dflash | 42 | 163 | 99 | 1223 |
| sglang_dflash | 42 | 163 | 99 | 1223 |
| nano_awq | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 204 | 1743 |
| vllmw4_dflash | 377 | 1913 |
| sglang_dflash | 180 | 1411 |
| nano_awq | 206 | 2529 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 4.56 | 7.19 |
| vllmw4_dflash | 2.39 | 6.83 |
| sglang_dflash | 5.33 | 9.15 |
| nano_awq | 4.82 | 5.40 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 33.8 | 76.8 |
| vllmw4_dflash | 27.0 | 73.2 |
| sglang_dflash | 42.9 | 78.1 |
| nano_awq | 9.8 | 19.8 |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 313 | 2619 |
| vllmw4_dflash | 551 | 2827 |
| sglang_dflash | 277 | 2305 |
| nano_awq | 205 | 2511 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 3.12 | 5.34 |
| vllmw4_dflash | 1.69 | 4.93 |
| sglang_dflash | 3.29 | 5.99 |
| nano_awq | 4.83 | 5.78 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| vllm_dflash | 36.6 | 98.9 |
| vllmw4_dflash | 28.1 | 96.9 |
| sglang_dflash | 42.7 | 78.8 |
| nano_awq | 17.1 | 155.2 |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| vllm_dflash | alpaca_c1s | 1401 | 2.31 | 3.31 |
| vllm_dflash | alpaca_c16s | 3090 | 2.66 | 3.66 |
| vllm_dflash | humaneval_c1s | 1114 | 4.19 | 5.19 |
| vllm_dflash | humaneval_c16s | 3124 | 4.20 | 5.20 |
| vllmw4_dflash | alpaca_c1s | 1329 | 2.41 | 3.41 |
| vllmw4_dflash | alpaca_c16s | 3267 | 2.52 | 3.52 |
| vllmw4_dflash | humaneval_c1s | 1115 | 4.16 | 5.16 |
| vllmw4_dflash | humaneval_c16s | 3022 | 4.39 | 5.39 |

