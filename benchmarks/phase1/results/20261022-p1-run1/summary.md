# Phase 1 results: run 20261022-p1-run1

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_nsys | ok | 0 |
| download_model | ok | 70 |
| install_vllm | ok | 84 |
| tests | **1** | 52 |
| kbench | **1** | 126 |
| environment | ok | 3 |

## Environment

```
run_id: 20261022-p1-run1
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=unset; nvcc: Build cuda_13.0.r13.0/compiler.36424714_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Tests

```
11 failed, 198 passed, 1 warning in 50.21s
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,378 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.0 | 1 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.9 | 9 | 1 | 1 | 5.90 |
| M=16 N=4096 | torch | ok (0.0e+00) | 28.5 | 9 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.4 | 114 | 7 | 8 | 11.96 |
| M=64 N=4096 | torch | ok (0.0e+00) | 30.7 | 34 | 2 | 3 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 2.8 | 383 | 25 | 28 | 11.11 |
| M=256 N=4096 | torch | ok (0.0e+00) | 42.9 | 98 | 6 | 7 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 4.2 | 1,006 | 65 | 73 | 10.28 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 257.2 | 131 | 8 | 9 | 1.00 |
| M=2048 N=4096 | triton | ok (0.0e+00) | 24.2 | 1,388 | 89 | 101 | 10.64 |

### silu_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.19 |
| M=16 N=14336 | torch | ok (0.0e+00) | 5.0 | 274 | 18 | 20 | 1.00 |
| M=16 N=14336 | triton | ok (3.1e-02) | 2.7 | 511 | 33 | 37 | 1.87 |
| M=64 N=14336 | torch | ok (0.0e+00) | 7.8 | 704 | 45 | 51 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.8 | 1,137 | 73 | 82 | 1.62 |
| M=256 N=14336 | torch | ok (0.0e+00) | 22.1 | 995 | 64 | 72 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 13.6 | 1,622 | 104 | 118 | 1.63 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 219.2 | 804 | 52 | 58 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 132.1 | 1,334 | 86 | 97 | 1.66 |

### linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | cublas | ok (7.8e-03) | 40.9 | 1,230 | 79 | 89 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | triton | ok (7.8e-03) | 42.2 | 1,192 | 77 | 86 | 0.97 |
| qkv_proj M=2 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,233 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | triton | ok (1.3e-02) | 42.9 | 1,175 | 76 | 85 | 0.95 |
| qkv_proj M=4 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.0 | 1,230 | 79 | 89 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | triton | ok (1.3e-02) | 43.5 | 1,160 | 75 | 84 | 0.94 |
| qkv_proj M=8 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.1 | 1,227 | 79 | 89 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | triton | ok (1.3e-02) | 44.2 | 1,142 | 73 | 83 | 0.93 |
| qkv_proj M=16 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.4 | 1,222 | 79 | 89 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | triton | ok (1.3e-02) | 45.1 | 1,123 | 72 | 81 | 0.92 |
| o_proj M=1 N=4096 K=4096 | cublas | ok (7.8e-03) | 28.5 | 1,176 | 76 | 85 | 1.00 |
| o_proj M=1 N=4096 K=4096 | triton | ok (7.8e-03) | 31.0 | 1,082 | 70 | 78 | 0.92 |
| o_proj M=2 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,170 | 75 | 85 | 1.00 |
| o_proj M=2 N=4096 K=4096 | triton | ok (1.3e-02) | 31.7 | 1,060 | 68 | 77 | 0.91 |
| o_proj M=4 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.8 | 1,167 | 75 | 85 | 1.00 |
| o_proj M=4 N=4096 K=4096 | triton | ok (1.3e-02) | 31.9 | 1,054 | 68 | 76 | 0.90 |
| o_proj M=8 N=4096 K=4096 | cublas | ok (1.3e-02) | 29.0 | 1,163 | 75 | 84 | 1.00 |
| o_proj M=8 N=4096 K=4096 | triton | ok (1.3e-02) | 32.2 | 1,045 | 67 | 76 | 0.90 |
| o_proj M=16 N=4096 K=4096 | cublas | ok (1.3e-02) | 29.0 | 1,165 | 75 | 85 | 1.00 |
| o_proj M=16 N=4096 K=4096 | triton | ok (1.3e-02) | 32.6 | 1,038 | 67 | 75 | 0.89 |
| gate_up_proj M=1 N=28672 K=4096 | cublas | ok (1.7e-02) | 178.4 | 1,317 | 85 | 96 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | triton | ok (1.3e-02) | 171.2 | 1,373 | 88 | 100 | 1.04 |
| gate_up_proj M=2 N=28672 K=4096 | cublas | ok (1.3e-02) | 179.2 | 1,312 | 84 | 95 | 1.00 |
| gate_up_proj M=2 N=28672 K=4096 | triton | ok (1.3e-02) | 171.6 | 1,369 | 88 | 99 | 1.04 |
| gate_up_proj M=4 N=28672 K=4096 | cublas | ok (1.3e-02) | 180.2 | 1,305 | 84 | 95 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | triton | ok (1.3e-02) | 172.1 | 1,366 | 88 | 99 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cublas | ok (1.5e-02) | 181.4 | 1,298 | 83 | 94 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | triton | ok (1.5e-02) | 173.8 | 1,355 | 87 | 98 | 1.04 |
| gate_up_proj M=16 N=28672 K=4096 | cublas | ok (1.5e-02) | 182.9 | 1,290 | 83 | 94 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | triton | ok (1.5e-02) | 175.8 | 1,342 | 86 | 97 | 1.04 |
| down_proj M=1 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.0 | 1,263 | 81 | 92 | 1.00 |
| down_proj M=1 N=4096 K=14336 | triton | ok (7.8e-03) | 91.3 | 1,287 | 83 | 93 | 1.02 |
| down_proj M=2 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.2 | 1,261 | 81 | 92 | 1.00 |
| down_proj M=2 N=4096 K=14336 | triton | ok (7.8e-03) | 92.3 | 1,273 | 82 | 92 | 1.01 |
| down_proj M=4 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.5 | 1,258 | 81 | 91 | 1.00 |
| down_proj M=4 N=4096 K=14336 | triton | ok (7.8e-03) | 92.8 | 1,267 | 81 | 92 | 1.01 |
| down_proj M=8 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.6 | 1,257 | 81 | 91 | 1.00 |
| down_proj M=8 N=4096 K=14336 | triton | ok (7.8e-03) | 93.7 | 1,257 | 81 | 91 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cublas | ok (1.3e-02) | 93.9 | 1,257 | 81 | 91 | 1.00 |
| down_proj M=16 N=4096 K=14336 | triton | ok (1.3e-02) | 95.2 | 1,239 | 80 | 90 | 0.99 |
| lm_head M=1 N=128256 K=4096 | cublas | ok (1.3e-02) | 720.8 | 1,458 | 94 | 106 | 1.00 |
| lm_head M=1 N=128256 K=4096 | triton | ok (1.3e-02) | 709.3 | 1,482 | 95 | 107 | 1.02 |
| lm_head M=2 N=128256 K=4096 | cublas | ok (1.4e-02) | 723.2 | 1,453 | 93 | 105 | 1.00 |
| lm_head M=2 N=128256 K=4096 | triton | ok (1.4e-02) | 712.0 | 1,476 | 95 | 107 | 1.02 |
| lm_head M=4 N=128256 K=4096 | cublas | ok (1.4e-02) | 728.4 | 1,444 | 93 | 105 | 1.00 |
| lm_head M=4 N=128256 K=4096 | triton | ok (1.4e-02) | 717.3 | 1,466 | 94 | 106 | 1.02 |
| lm_head M=8 N=128256 K=4096 | cublas | ok (1.5e-02) | 733.2 | 1,436 | 92 | 104 | 1.00 |
| lm_head M=8 N=128256 K=4096 | triton | ok (1.5e-02) | 722.0 | 1,458 | 94 | 106 | 1.02 |
| lm_head M=16 N=128256 K=4096 | cublas | ok (1.5e-02) | 743.0 | 1,420 | 91 | 103 | 1.00 |
| lm_head M=16 N=128256 K=4096 | triton | ok (1.5e-02) | 731.6 | 1,442 | 93 | 105 | 1.02 |

### fused_add_rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.6 | 2 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.7 | 15 | 1 | 1 | 6.60 |
| M=16 N=4096 | torch | ok (0.0e+00) | 29.6 | 18 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.7 | 194 | 12 | 14 | 10.76 |
| M=64 N=4096 | torch | ok (0.0e+00) | 32.1 | 66 | 4 | 5 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 3.3 | 648 | 42 | 47 | 9.88 |
| M=256 N=4096 | torch | ok (0.0e+00) | 45.7 | 184 | 12 | 13 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 6.1 | 1,379 | 89 | 100 | 7.50 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 298.1 | 225 | 14 | 16 | 1.00 |
| M=2048 N=4096 | triton | ok (1.6e-02) | 52.0 | 1,290 | 83 | 94 | 5.73 |

### silu_and_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 45 | 3 | 3 | 2.18 |
| M=16 N=14336 | torch | ok (0.0e+00) | 9.1 | 151 | 10 | 11 | 1.00 |
| M=16 N=14336 | triton | ok (6.2e-02) | 2.7 | 516 | 33 | 37 | 3.43 |
| M=64 N=14336 | torch | ok (0.0e+00) | 14.9 | 369 | 24 | 27 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.9 | 1,129 | 73 | 82 | 3.06 |
| M=256 N=14336 | torch | ok (0.0e+00) | 38.6 | 570 | 37 | 41 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 15.0 | 1,467 | 94 | 106 | 2.57 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 292.3 | 603 | 39 | 44 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 132.1 | 1,334 | 86 | 97 | 2.21 |

### rope_cache_write

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| tokens=1 | torch | ok (0.0e+00) | 37.8 | 1 | 0 | 0 | 1.00 |
| tokens=1 | triton | FAIL (1.3e+01) | - | - | - | - | - |
| tokens=16 | torch | ok (0.0e+00) | 43.3 | 9 | 1 | 1 | 1.00 |
| tokens=16 | triton | FAIL (1.4e+01) | - | - | - | - | - |
| tokens=64 | torch | ok (0.0e+00) | 47.3 | 34 | 2 | 2 | 1.00 |
| tokens=64 | triton | FAIL (1.9e+01) | - | - | - | - | - |
| tokens=256 | torch | ok (0.0e+00) | 63.6 | 101 | 6 | 7 | 1.00 |
| tokens=256 | triton | FAIL (1.8e+01) | - | - | - | - | - |
| tokens=2048 | torch | ok (0.0e+00) | 281.0 | 183 | 12 | 13 | 1.00 |
| tokens=2048 | triton | FAIL (2.1e+01) | - | - | - | - | - |

### paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 6.9 | 76 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.7 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 12.0 | 44 | 3 | 3 | 0.58 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.7 | 267 | 17 | 19 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.9 | 49 | 3 | 4 | 0.18 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 13.6 | 308 | 20 | 22 | 1.15 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.2 | 555 | 36 | 40 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 228.1 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 22.4 | 748 | 48 | 54 | 1.35 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.1 | 363 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 126.2 | 66 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 18.1 | 464 | 30 | 34 | 1.28 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 75.7 | 887 | 57 | 64 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 933.8 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 64.4 | 1,042 | 67 | 76 | 1.17 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 225.1 | 1,192 | 77 | 87 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3578.1 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 222.1 | 1,209 | 78 | 88 | 1.01 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.3 | 757 | 49 | 55 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 482.6 | 70 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 33.7 | 996 | 64 | 72 | 1.32 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 220.8 | 1,216 | 78 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3586.2 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 221.6 | 1,211 | 78 | 88 | 1.00 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 813.4 | 1,320 | 85 | 96 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 861.6 | 1,246 | 80 | 90 | 0.94 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 72.9 | 921 | 59 | 67 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 925.0 | 73 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 59.9 | 1,120 | 72 | 81 | 1.22 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 404.4 | 1,328 | 85 | 96 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 388.8 | 1,381 | 89 | 100 | 1.04 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1524.5 | 1,409 | 91 | 102 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1510.3 | 1,422 | 91 | 103 | 1.01 |

### paged_prefill_attention

| shape | impl | check | us | TFLOPs | % peak | x base |
|---|---|---|---:|---:|---:|---:|
| seqs=1 qlen=512 | vllm_fa2 | ok (1.6e-02) | 35.2 | 61.0 | 20 | 1.00 |
| seqs=1 qlen=512 | torch | ok (0.0e+00) | 665.7 | 3.2 | 1 | 0.05 |
| seqs=1 qlen=512 | triton | ok (1.6e-02) | 52.8 | 40.8 | 13 | 0.67 |
| seqs=1 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 84.2 | 102.1 | 33 | 1.00 |
| seqs=1 qlen=1024 | torch | ok (0.0e+00) | 2203.0 | 3.9 | 1 | 0.04 |
| seqs=1 qlen=1024 | triton | ok (1.6e-02) | 141.0 | 61.0 | 20 | 0.60 |
| seqs=1 qlen=4096 | vllm_fa2 | ok (1.6e-02) | 847.0 | 162.3 | 52 | 1.00 |
| seqs=1 qlen=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=1 qlen=4096 | triton | ok (1.6e-02) | 1529.5 | 89.9 | 29 | 0.55 |
| seqs=4 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 260.9 | 131.8 | 42 | 1.00 |
| seqs=4 qlen=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=4 qlen=1024 | triton | ok (1.6e-02) | 435.2 | 79.0 | 25 | 0.60 |
| seqs=16 qlen=256 | vllm_fa2 | ok (1.6e-02) | 113.7 | 75.9 | 24 | 1.00 |
| seqs=16 qlen=256 | torch | ok (0.0e+00) | 3449.1 | 2.5 | 1 | 0.03 |
| seqs=16 qlen=256 | triton | ok (1.6e-02) | 163.4 | 52.8 | 17 | 0.70 |


