# Phase 3 run m3-20261112-092848-colab

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
| install_venv | 0 | 61 |
| install_study | 0 | 15 |
| download_model | 0 | 45 |
| download_w4_model | 0 | 16 |
| download_drafter | 0 | 7 |
| cuda129 | 0 | 43 |
| workloads | 0 | 42 |
| tests | 0 | 780 |
| spec_check | 0 | 322 |
| kbench | 0 | 59 |
| serve | 0 | 662 |

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
| nano_awq_dflash | 247 | 961 | 1962 | 2750 |
| nano_dflash | 172 | - | - | - |
| nano_awq | 207 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.80 | 3.56 | 6.50 | 21.52 |
| nano_dflash | 5.64 | - | - | - |
| nano_awq | 4.79 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 9.3 | 28.6 | 52.9 | 162.6 |
| nano_dflash | 17.9 | - | - | - |
| nano_awq | 9.5 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 1.19x | - | - | - |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 368 | 1385 | 2911 | 3986 |
| nano_dflash | 267 | - | - | - |
| nano_awq | 206 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.66 | 2.44 | 4.86 | 13.73 |
| nano_dflash | 3.59 | - | - | - |
| nano_awq | 4.81 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 16.8 | 36.3 | 75.0 | 200.7 |
| nano_dflash | 22.0 | - | - | - |
| nano_awq | 16.5 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 1.79x | - | - | - |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 278 | 2257 |
| nano_dflash | 204 | - |
| nano_awq | 206 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 3.57 | 5.98 |
| nano_dflash | 4.96 | - |
| nano_awq | 4.80 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 14.6 | 60.0 |
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
| nano_awq_dflash | 185 | 786 |
| nano_dflash | 139 | - |
| nano_awq | 178 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 4.59 | 16.79 |
| nano_dflash | 6.56 | - |
| nano_awq | 4.85 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 104.6 | 247.5 |
| nano_dflash | 91.7 | - |
| nano_awq | 104.0 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.04x | - |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1309 | 2.50 | 3.50 |
| nano_awq_dflash | alpaca_c4 | 1351 | 2.57 | 3.57 |
| nano_awq_dflash | alpaca_c16 | 3184 | 2.67 | 3.67 |
| nano_awq_dflash | alpaca_c64 | 14599 | 2.50 | 3.50 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1030 | 4.49 | 5.49 |
| nano_awq_dflash | humaneval_c16 | 2897 | 4.61 | 5.61 |
| nano_awq_dflash | humaneval_c64 | 7325 | 4.60 | 5.60 |
| nano_awq_dflash | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflash | gsm8k_c16 | 2473 | 3.23 | 4.23 |
| nano_awq_dflash | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 2923 | 2.05 | 3.05 |
| nano_dflash | alpaca_c1 | 1361 | 2.43 | 3.43 |
| nano_dflash | humaneval_c1 | 1022 | 4.42 | 5.42 |
| nano_dflash | gsm8k_c1 | 1050 | 3.08 | 4.08 |
| nano_dflash | long_c1 | 1143 | 2.02 | 3.02 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's.

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1973 |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 281 |
| awq spec b=16 | 3040 | 8 | 0.26% | 0.250 | 0 | 0 | 4.01 | 1925 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 81 |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 870 |
| bf16 spec b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 198 |
| bf16 spec b=16 | 2917 | 5 | 0.17% | 0.250 | 0 | 0 | 4.06 | 1565 |

## Verify attention (kbench paged_block_attention)

Microseconds per call for Q causal queries after each sequence's context, bf16. decode: the decode kernel on one token per sequence at the same context, what a plain decode step pays. vllm_fa2: vLLM's flash-attention. prefill: nano's varlen kernel, the verify path before M3. block: the block kernel.

| shape | decode | vllm_fa2 | prefill | block | block / decode |
|---|---:|---:|---:|---:|---:|
| B=1 Q=10 ctx=256 | 16.8 | 11.3 | 20.5 | 13.1 | 0.78 |
| B=1 Q=10 ctx=1024 | 14.3 | 29.5 | 64.9 | 14.6 | 1.02 |
| B=1 Q=10 ctx=4096 | 20.8 | 103.6 | 241.8 | 26.4 | 1.27 |
| B=4 Q=10 ctx=256 | 14.7 | 16.4 | 27.2 | 14.4 | 0.98 |
| B=4 Q=10 ctx=1024 | 20.9 | 45.3 | 87.0 | 26.8 | 1.28 |
| B=4 Q=10 ctx=4096 | 62.8 | 159.0 | 324.7 | 76.6 | 1.22 |
| B=16 Q=10 ctx=256 | 21.8 | 37.4 | 69.0 | 25.0 | 1.15 |
| B=16 Q=10 ctx=1024 | 61.2 | 115.3 | 229.7 | 75.7 | 1.24 |
| B=16 Q=10 ctx=4096 | 206.9 | 437.9 | 874.9 | 273.2 | 1.32 |
| B=64 Q=10 ctx=256 | 58.3 | 126.2 | 240.4 | 76.3 | 1.31 |
| B=64 Q=10 ctx=1024 | 198.9 | 477.0 | 835.4 | 248.3 | 1.25 |
| B=64 Q=10 ctx=4096 | 764.9 | 2495.0 | 3220.5 | 926.9 | 1.21 |
| B=1 Q=3 ctx=1024 | 14.4 | 29.5 | 64.7 | 12.5 | 0.87 |
| B=1 Q=5 ctx=1024 | 14.3 | 29.5 | 64.7 | 13.6 | 0.95 |
| B=16 Q=3 ctx=1024 | 61.8 | 114.2 | 229.2 | 66.3 | 1.07 |
| B=16 Q=5 ctx=1024 | 61.6 | 114.5 | 229.4 | 70.6 | 1.15 |
| B=64 Q=3 ctx=1024 | 199.8 | 470.2 | 832.0 | 221.3 | 1.11 |
| B=64 Q=5 ctx=1024 | 200.0 | 472.1 | 833.1 | 210.6 | 1.05 |

Gate (block <= 1.3x decode at B=1, <= 1.5x at B=64, Q=10): met: B=1 Q=10 ctx=256 0.78x (limit 1.3), B=1 Q=10 ctx=1024 1.02x (limit 1.3), B=1 Q=10 ctx=4096 1.27x (limit 1.3), B=64 Q=10 ctx=256 1.31x (limit 1.5), B=64 Q=10 ctx=1024 1.25x (limit 1.5), B=64 Q=10 ctx=4096 1.21x (limit 1.5).

## Drafter norm + RoPE + KV write (kbench norm_rope_cache_write)

| shape | torch us | triton us |
|---|---:|---:|
| tokens=1 query=1 | 76.7 | 9.9 |
| tokens=10 query=1 | 81.2 | 6.6 |
| tokens=160 query=1 | 106.1 | 9.5 |
| tokens=640 query=1 | 204.9 | 16.9 |
| tokens=1 query=0 | 44.7 | 8.9 |
| tokens=10 query=0 | 47.7 | 5.1 |
| tokens=160 query=0 | 55.7 | 7.2 |
| tokens=640 query=0 | 77.0 | 10.8 |

