# Phase 3 run m5-sweep-20261112-235755-colab

```
load generator / nano: vllm 0.19.1
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 441: /content/p0/venvs/vllm_spec/bin/python: No such file or directory
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 442: /content/p0/venvs/sglang_spec/bin/python: No such file or directory
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 435 |
| install_study | 0 | 85 |
| download_model | 0 | 43 |
| download_w4_model | 0 | 15 |
| download_drafter | 0 | 7 |
| cuda129 | 0 | 24 |
| workloads | 0 | 28 |
| tests | 0 | 222 |
| serve | 0 | 954 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq_dflash_t25 | 42 | 163 | 99 | 1223 |
| nano_dflash_t25 | 42 | 163 | 99 | 1223 |
| nano_awq_dflash_t50 | 42 | 163 | 99 | 1223 |
| nano_dflash_t50 | 42 | 163 | 99 | 1223 |
| nano_awq_dflash_t75 | 42 | 163 | 99 | 1223 |
| nano_dflash_t75 | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash_t25 | 411 | 2131 |
| nano_dflash_t25 | 219 | 1886 |
| nano_awq_dflash_t50 | 423 | 2123 |
| nano_dflash_t50 | 218 | 1909 |
| nano_awq_dflash_t75 | 417 | 2140 |
| nano_dflash_t75 | 215 | 1884 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash_t25 | 2.26 | 6.62 |
| nano_dflash_t25 | 4.42 | 7.05 |
| nano_awq_dflash_t50 | 2.28 | 6.74 |
| nano_dflash_t50 | 4.42 | 7.24 |
| nano_awq_dflash_t75 | 2.29 | 6.62 |
| nano_dflash_t75 | 4.49 | 7.37 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash_t25 | 9.8 | 49.8 |
| nano_dflash_t25 | 18.0 | 55.3 |
| nano_awq_dflash_t50 | 9.7 | 51.9 |
| nano_dflash_t50 | 17.8 | 54.9 |
| nano_awq_dflash_t75 | 9.9 | 49.5 |
| nano_dflash_t75 | 18.1 | 56.2 |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash_t25 | 628 | 2991 |
| nano_dflash_t25 | 327 | 2688 |
| nano_awq_dflash_t50 | 625 | 2960 |
| nano_dflash_t50 | 328 | 2620 |
| nano_awq_dflash_t75 | 623 | 2991 |
| nano_dflash_t75 | 326 | 2622 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash_t25 | 1.51 | 4.65 |
| nano_dflash_t25 | 3.00 | 4.98 |
| nano_awq_dflash_t50 | 1.54 | 4.73 |
| nano_dflash_t50 | 2.92 | 5.11 |
| nano_awq_dflash_t75 | 1.54 | 4.66 |
| nano_dflash_t75 | 2.93 | 5.16 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash_t25 | 16.9 | 73.2 |
| nano_dflash_t25 | 22.2 | 73.7 |
| nano_awq_dflash_t50 | 17.0 | 75.7 |
| nano_dflash_t50 | 22.3 | 72.1 |
| nano_awq_dflash_t75 | 16.9 | 71.8 |
| nano_dflash_t75 | 22.3 | 72.1 |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash_t25 | alpaca_c1s | 1346 | 2.42 | 3.42 |
| nano_awq_dflash_t25 | alpaca_c16s | 3373 | 2.51 | 3.51 |
| nano_awq_dflash_t25 | humaneval_c1s | 1110 | 4.19 | 5.19 |
| nano_awq_dflash_t25 | humaneval_c16s | 2887 | 4.60 | 5.60 |
| nano_dflash_t25 | alpaca_c1s | 1328 | 2.46 | 3.46 |
| nano_dflash_t25 | alpaca_c16s | 3155 | 2.62 | 3.62 |
| nano_dflash_t25 | humaneval_c1s | 1093 | 4.25 | 5.25 |
| nano_dflash_t25 | humaneval_c16s | 2985 | 4.42 | 5.42 |
| nano_awq_dflash_t50 | alpaca_c1s | 1349 | 2.41 | 3.41 |
| nano_awq_dflash_t50 | alpaca_c16s | 3250 | 2.53 | 3.53 |
| nano_awq_dflash_t50 | humaneval_c1s | 1115 | 4.16 | 5.16 |
| nano_awq_dflash_t50 | humaneval_c16s | 2962 | 4.53 | 5.53 |
| nano_dflash_t50 | alpaca_c1s | 1336 | 2.44 | 3.44 |
| nano_dflash_t50 | alpaca_c16s | 3187 | 2.62 | 3.62 |
| nano_dflash_t50 | humaneval_c1s | 1090 | 4.26 | 5.26 |
| nano_dflash_t50 | humaneval_c16s | 3024 | 4.33 | 5.33 |
| nano_awq_dflash_t75 | alpaca_c1s | 1370 | 2.35 | 3.35 |
| nano_awq_dflash_t75 | alpaca_c16s | 3349 | 2.47 | 3.47 |
| nano_awq_dflash_t75 | humaneval_c1s | 1114 | 4.16 | 5.16 |
| nano_awq_dflash_t75 | humaneval_c16s | 2924 | 4.55 | 5.55 |
| nano_dflash_t75 | alpaca_c1s | 1352 | 2.39 | 3.39 |
| nano_dflash_t75 | alpaca_c16s | 3143 | 2.66 | 3.66 |
| nano_dflash_t75 | humaneval_c1s | 1095 | 4.23 | 5.23 |
| nano_dflash_t75 | humaneval_c16s | 2990 | 4.35 | 5.35 |

