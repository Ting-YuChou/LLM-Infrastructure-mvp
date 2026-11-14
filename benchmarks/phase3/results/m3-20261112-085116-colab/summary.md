# Phase 3 run m3-20261112-085116-colab

```
load generator / nano: vllm 0.19.1
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 437: /content/p0/venvs/vllm_spec/bin/python: No such file or directory
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 438: /content/p0/venvs/sglang_spec/bin/python: No such file or directory
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 53 |
| install_study | 0 | 15 |
| download_model | 0 | 44 |
| download_w4_model | 0 | 16 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 49 |
| workloads | 0 | 42 |
| tests | 0 | 774 |
| spec_check | 0 | 324 |
| kbench | 0 | 60 |
| serve | 0 | 668 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |
| nano_awq | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 246 | 963 | 2055 | 2739 |
| nano_dflash | 172 | - | - | - |
| nano_awq | 206 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.92 | 3.56 | 6.69 | 21.81 |
| nano_dflash | 5.68 | - | - | - |
| nano_awq | 4.83 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 9.6 | 28.3 | 53.5 | 165.9 |
| nano_dflash | 17.9 | - | - | - |
| nano_awq | 9.7 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 1.19x | - | - | - |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 359 | 1362 | 2898 | 4007 |
| nano_dflash | 266 | - | - | - |
| nano_awq | 204 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.72 | 2.51 | 4.86 | 13.87 |
| nano_dflash | 3.59 | - | - | - |
| nano_awq | 4.84 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 17.1 | 36.8 | 74.5 | 182.8 |
| nano_dflash | 22.2 | - | - | - |
| nano_awq | 16.6 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 1.76x | - | - | - |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 276 | 2257 |
| nano_dflash | 203 | - |
| nano_awq | 204 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.59 | 6.16 |
| nano_dflash | 4.98 | - |
| nano_awq | 4.84 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 14.6 | 60.4 |
| nano_dflash | 21.6 | - |
| nano_awq | 15.8 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.35x | - |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 182 | 788 |
| nano_dflash | 138 | - |
| nano_awq | 177 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 4.67 | 16.27 |
| nano_dflash | 6.57 | - |
| nano_awq | 4.89 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 105.4 | 278.5 |
| nano_dflash | 92.2 | - |
| nano_awq | 104.2 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.03x | - |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1309 | 2.50 | 3.50 |
| nano_awq_dflash | alpaca_c4 | 1351 | 2.57 | 3.57 |
| nano_awq_dflash | alpaca_c16 | 3126 | 2.70 | 3.70 |
| nano_awq_dflash | alpaca_c64 | 14539 | 2.52 | 3.52 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1030 | 4.49 | 5.49 |
| nano_awq_dflash | humaneval_c16 | 2897 | 4.61 | 5.61 |
| nano_awq_dflash | humaneval_c64 | 7313 | 4.61 | 5.61 |
| nano_awq_dflash | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflash | gsm8k_c16 | 2445 | 3.20 | 4.20 |
| nano_awq_dflash | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 2902 | 2.06 | 3.06 |
| nano_dflash | alpaca_c1 | 1361 | 2.43 | 3.43 |
| nano_dflash | humaneval_c1 | 1022 | 4.42 | 5.42 |
| nano_dflash | gsm8k_c1 | 1050 | 3.08 | 4.08 |
| nano_dflash | long_c1 | 1143 | 2.02 | 3.02 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's.

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1960 |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 275 |
| awq spec b=16 | 3040 | 8 | 0.26% | 0.250 | 0 | 0 | 4.01 | 1901 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 81 |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 870 |
| bf16 spec b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 196 |
| bf16 spec b=16 | 2917 | 5 | 0.17% | 0.250 | 0 | 0 | 4.06 | 1558 |

## Verify attention (kbench paged_block_attention)

Microseconds per call for Q causal queries after each sequence's context, bf16. decode: the decode kernel on one token per sequence at the same context, what a plain decode step pays. vllm_fa2: vLLM's flash-attention. prefill: nano's varlen kernel, the verify path before M3. block: the block kernel.

| shape | decode | vllm_fa2 | prefill | block | block / decode |
|---|---:|---:|---:|---:|---:|
| B=1 Q=10 ctx=256 | 16.7 | 11.4 | 20.7 | 13.3 | 0.79 |
| B=1 Q=10 ctx=1024 | 14.4 | 29.6 | 65.3 | 14.6 | 1.02 |
| B=1 Q=10 ctx=4096 | 20.8 | 103.8 | 243.2 | 26.4 | 1.27 |
| B=4 Q=10 ctx=256 | 14.8 | 16.4 | 27.2 | 14.5 | 0.98 |
| B=4 Q=10 ctx=1024 | 20.9 | 45.4 | 87.0 | 26.7 | 1.28 |
| B=4 Q=10 ctx=4096 | 62.5 | 159.1 | 324.5 | 76.3 | 1.22 |
| B=16 Q=10 ctx=256 | 21.8 | 37.4 | 69.0 | 25.0 | 1.15 |
| B=16 Q=10 ctx=1024 | 61.5 | 115.0 | 229.6 | 75.6 | 1.23 |
| B=16 Q=10 ctx=4096 | 206.8 | 439.9 | 875.0 | 274.6 | 1.33 |
| B=64 Q=10 ctx=256 | 57.8 | 126.1 | 240.3 | 76.2 | 1.32 |
| B=64 Q=10 ctx=1024 | 198.7 | 477.4 | 835.2 | 248.5 | 1.25 |
| B=64 Q=10 ctx=4096 | 767.1 | 2499.6 | 3248.2 | 926.6 | 1.21 |
| B=1 Q=3 ctx=1024 | 14.4 | 29.6 | 65.1 | 12.6 | 0.87 |
| B=1 Q=5 ctx=1024 | 14.4 | 29.7 | 65.1 | 13.7 | 0.95 |
| B=16 Q=3 ctx=1024 | 62.0 | 114.2 | 229.2 | 66.2 | 1.07 |
| B=16 Q=5 ctx=1024 | 61.7 | 115.4 | 229.4 | 70.8 | 1.15 |
| B=64 Q=3 ctx=1024 | 199.7 | 474.2 | 831.8 | 222.1 | 1.11 |
| B=64 Q=5 ctx=1024 | 199.6 | 475.5 | 832.9 | 211.9 | 1.06 |

Gate (block <= 1.3x decode at B=1, <= 1.5x at B=64, Q=10): met: B=1 Q=10 ctx=256 0.79x (limit 1.3), B=1 Q=10 ctx=1024 1.02x (limit 1.3), B=1 Q=10 ctx=4096 1.27x (limit 1.3), B=64 Q=10 ctx=256 1.32x (limit 1.5), B=64 Q=10 ctx=1024 1.25x (limit 1.5), B=64 Q=10 ctx=4096 1.21x (limit 1.5).

## Drafter norm + RoPE + KV write (kbench norm_rope_cache_write)

| shape | torch us | triton us |
|---|---:|---:|
| tokens=1 query=1 | 76.2 | 9.6 |
| tokens=10 query=1 | 82.3 | 6.8 |
| tokens=160 query=1 | 106.3 | 9.6 |
| tokens=640 query=1 | 205.9 | 17.1 |
| tokens=1 query=0 | 43.7 | 8.5 |
| tokens=10 query=0 | 48.0 | 5.1 |
| tokens=160 query=0 | 56.1 | 7.4 |
| tokens=640 query=0 | 77.2 | 10.9 |

