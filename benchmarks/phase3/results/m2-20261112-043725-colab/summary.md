# Phase 3 run m2-20261112-043725-colab

```
load generator / nano: vllm 0.19.1
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 416: /content/p0/venvs/vllm_spec/bin/python: No such file or directory
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 417: /content/p0/venvs/sglang_spec/bin/python: No such file or directory
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | 0 | 513 |
| spec_check | 0 | 322 |
| serve | 0 | 1071 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 207 | 736 | 1722 | 2259 |
| nano | 83 | - | - | - |
| nano_dflash | 142 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 4.71 | 4.79 | 7.98 | 26.01 |
| nano | 12.00 | - | - | - |
| nano_dflash | 6.89 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 9.8 | 34.2 | 60.9 | 197.6 |
| nano | 17.7 | - | - | - |
| nano_dflash | 17.2 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_dflash | 1.71x | - | - | - |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 316 | 1081 | 2367 | 3128 |
| nano | 83 | - | - | - |
| nano_dflash | 212 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.06 | 3.25 | 5.99 | 17.64 |
| nano | 12.01 | - | - | - |
| nano_dflash | 4.43 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 17.0 | 44.0 | 89.3 | 230.6 |
| nano | 22.2 | - | - | - |
| nano_dflash | 22.6 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_dflash | 2.56x | - | - | - |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 238 | 1830 |
| nano | 83 | - |
| nano_dflash | 164 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 4.23 | 7.58 |
| nano | 12.01 | - |
| nano_dflash | 6.07 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 14.8 | 73.4 |
| nano | 25.3 | - |
| nano_dflash | 22.0 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 1.98x | - |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 144 | 592 |
| nano | 80 | - |
| nano_dflash | 108 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 6.06 | 22.62 |
| nano | 12.06 | - |
| nano_dflash | 8.65 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 105.1 | 294.1 |
| nano | 94.1 | - |
| nano_dflash | 91.6 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_dflash | 1.36x | - |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 200 | 1650 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 4.64 | 8.57 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 10.1 | 62.6 |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 300 | 2219 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.25 | 6.36 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 17.4 | 89.7 |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1354 | 2.39 | 3.39 |
| nano_awq_dflash | alpaca_c4 | 1362 | 2.55 | 3.55 |
| nano_awq_dflash | alpaca_c16 | 3178 | 2.67 | 3.67 |
| nano_awq_dflash | alpaca_c64 | 14656 | 2.50 | 3.50 |
| nano_awq_dflash | humaneval_c1 | 1085 | 4.34 | 5.34 |
| nano_awq_dflash | humaneval_c4 | 1011 | 4.53 | 5.53 |
| nano_awq_dflash | humaneval_c16 | 2880 | 4.63 | 5.63 |
| nano_awq_dflash | humaneval_c64 | 7363 | 4.57 | 5.57 |
| nano_awq_dflash | gsm8k_c1 | 1162 | 2.95 | 3.95 |
| nano_awq_dflash | gsm8k_c16 | 2486 | 3.16 | 4.16 |
| nano_awq_dflash | long_c1 | 1089 | 2.03 | 3.03 |
| nano_awq_dflash | long_c16 | 2846 | 2.11 | 3.11 |
| nano_awq_dflash | alpaca_c1s | 1373 | 2.32 | 3.32 |
| nano_awq_dflash | alpaca_c16s | 3303 | 2.53 | 3.53 |
| nano_awq_dflash | humaneval_c1s | 1124 | 4.18 | 5.18 |
| nano_awq_dflash | humaneval_c16s | 3012 | 4.41 | 5.41 |
| nano_dflash | alpaca_c1 | 1335 | 2.48 | 3.48 |
| nano_dflash | humaneval_c1 | 1055 | 4.35 | 5.35 |
| nano_dflash | gsm8k_c1 | 1056 | 3.03 | 4.03 |
| nano_dflash | long_c1 | 1141 | 2.04 | 3.04 |

## Completions at temperature 0 (texts.py)

Completions with identical text: a server's 16-at-a-time pass against its own one-at-a-time pass (logits depend slightly on the batch size), and a drafter server's passes against the one-at-a-time pass of the same server without a drafter. Last column: output tok/s of the one-at-a-time pass.

| server | c16 = own c1 | c1 = base c1 | c16 = base c1 | c1 tok/s |
|---|---:|---:|---:|---:|
| nano_dflash | 9/32 | 12/32 | 16/32 | 152 |
| nano | 12/32 | - | - | 82 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's.

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1964 |
| awq spec b=1 | 3039 | 9 | 0.30% | 0.125 | 0 | 0 | 3.95 | 230 |
| awq spec b=16 | 3007 | 12 | 0.40% | 0.250 | 0 | 0 | 4.03 | 1305 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 81 |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 870 |
| bf16 spec b=1 | 2947 | 4 | 0.14% | 0.125 | 0 | 0 | 4.09 | 164 |
| bf16 spec b=16 | 2907 | 6 | 0.21% | 0.125 | 0 | 0 | 4.02 | 1107 |

