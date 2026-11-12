# Phase 3 run m3-kernel-20261112-064729-colab

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
| install_venv | 0 | 183 |
| install_study | 0 | 69 |
| download_model | 0 | 43 |
| download_w4_model | 0 | 14 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 23 |
| tests | 1 | 478 |
| kbench | 0 | 64 |
| tune | 0 | 165 |

## Verify attention (kbench paged_block_attention)

Microseconds per call for Q causal queries after each sequence's context, bf16. decode: the decode kernel on one token per sequence at the same context, what a plain decode step pays. vllm_fa2: vLLM's flash-attention. prefill: nano's varlen kernel, the verify path before M3. block: the block kernel.

| shape | decode | vllm_fa2 | prefill | block | block / decode |
|---|---:|---:|---:|---:|---:|
| B=1 Q=10 ctx=256 | 16.9 | 11.3 | 20.6 | 18.3 | 1.08 |
| B=1 Q=10 ctx=1024 | 14.4 | 29.6 | 65.1 | 19.8 | 1.38 |
| B=1 Q=10 ctx=4096 | 20.9 | 104.1 | 242.6 | 44.5 | 2.13 |
| B=4 Q=10 ctx=256 | 14.8 | 16.4 | 27.3 | 29.8 | 2.02 |
| B=4 Q=10 ctx=1024 | 21.0 | 45.4 | 87.2 | 50.0 | 2.38 |
| B=4 Q=10 ctx=4096 | 62.7 | 159.4 | 325.2 | 130.2 | 2.08 |
| B=16 Q=10 ctx=256 | 21.8 | 37.6 | 69.2 | 56.5 | 2.59 |
| B=16 Q=10 ctx=1024 | 61.4 | 114.3 | 230.2 | 111.3 | 1.81 |
| B=16 Q=10 ctx=4096 | 208.8 | 434.4 | 877.4 | 305.6 | 1.46 |
| B=64 Q=10 ctx=256 | 57.8 | 125.2 | 241.0 | 96.3 | 1.67 |
| B=64 Q=10 ctx=1024 | 198.8 | 474.7 | 837.1 | 296.8 | 1.49 |
| B=64 Q=10 ctx=4096 | 764.6 | 2499.9 | 3213.0 | 1144.6 | 1.50 |
| B=1 Q=3 ctx=1024 | 14.4 | 29.6 | 64.9 | 14.5 | 1.01 |
| B=1 Q=5 ctx=1024 | 14.4 | 29.6 | 65.0 | 17.1 | 1.19 |
| B=16 Q=3 ctx=1024 | 61.9 | 114.3 | 229.7 | 69.2 | 1.12 |
| B=16 Q=5 ctx=1024 | 61.3 | 114.5 | 229.9 | 77.6 | 1.27 |
| B=64 Q=3 ctx=1024 | 199.8 | 470.6 | 833.3 | 205.6 | 1.03 |
| B=64 Q=5 ctx=1024 | 199.5 | 472.4 | 833.9 | 211.8 | 1.06 |

Gate (block <= 1.3x decode at B=1, <= 1.5x at B=64, Q=10): NOT met: B=1 Q=10 ctx=256 1.08x (limit 1.3), B=1 Q=10 ctx=1024 1.38x (limit 1.3), B=1 Q=10 ctx=4096 2.13x (limit 1.3), B=64 Q=10 ctx=256 1.67x (limit 1.5), B=64 Q=10 ctx=1024 1.49x (limit 1.5), B=64 Q=10 ctx=4096 1.50x (limit 1.5).

## Drafter norm + RoPE + KV write (kbench norm_rope_cache_write)

| shape | torch us | triton us |
|---|---:|---:|
| tokens=1 query=1 | 77.8 | 9.8 |
| tokens=10 query=1 | 81.8 | 6.7 |
| tokens=160 query=1 | 106.8 | 9.6 |
| tokens=640 query=1 | 206.1 | 16.9 |
| tokens=1 query=0 | 44.5 | 8.9 |
| tokens=10 query=0 | 48.0 | 5.1 |
| tokens=160 query=0 | 55.9 | 7.3 |
| tokens=640 query=0 | 77.4 | 10.8 |

## Block kernel launch settings (tune.py)

| shape | best config | best us | current us | current / best |
|---|---:|---:|---:|---:|
| B=1 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 8, 'num_stages': 1, 'splits_shift': 2} | 18.5 | 19.7 | 1.06 |
| B=1 Q=10 ctx=4096 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 32.0 | 44.4 | 1.39 |
| B=4 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 39.2 | 50.1 | 1.28 |
| B=16 Q=10 ctx=1024 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 1} | 89.4 | 110.9 | 1.24 |
| B=16 Q=10 ctx=4096 | {'block_n': 32, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 1} | 284.6 | 305.5 | 1.07 |
| B=64 Q=10 ctx=1024 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 247.9 | 298.6 | 1.20 |
| B=64 Q=10 ctx=4096 | {'block_n': 64, 'num_warps': 4, 'num_stages': 2, 'splits_shift': 0} | 929.2 | 1134.4 | 1.22 |

