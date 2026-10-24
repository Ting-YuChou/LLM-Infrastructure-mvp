# Phase 1 results: run down-d3-20261024-011151-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 307 |
| kbench | ok | 269 |
| tune | ok | 67 |
| trace | ok | 19 |
| golden | **1** | 4 |
| e2e | ok | 593 |
| environment | ok | 2 |

## Environment

```
run_id: down-d3-20261024-011151-colab
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=/usr/local/cuda-12.8; nvcc: Build cuda_12.8.r12.8/compiler.35583870_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Tests

```
645 passed, 1 warning in 305.01s (0:05:05)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,379 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.9 | 688 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.0 | 591 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.3 | 641 | 41 | 46 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.3 | 609 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 730 | 47 | 53 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.2 | 802 | 52 | 58 | 1.16 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 38.3 | 341 | 22 | 25 | 0.49 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.4 | 640 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.7 | 576 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.4 | 641 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.4 | 795 | 51 | 58 | 1.15 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 694 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 56.9 | 231 | 15 | 17 | 0.33 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 640 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.2 | 542 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 821 | 53 | 59 | 1.18 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 320 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.3 | 626 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.4 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.8 | 479 | 31 | 35 | 0.77 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 831 | 53 | 60 | 1.33 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.3 | 499 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.2 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.2 | 339 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.7 | 690 | 44 | 50 | 1.38 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.6 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 34.6 | 413 | 27 | 30 | 1.30 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.4 | 497 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 13.2 | 656 | 42 | 48 | 1.32 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 555 | 36 | 40 | 1.12 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 532 | 34 | 39 | 1.07 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 658 | 42 | 48 | 1.32 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 14.5 | 597 | 38 | 43 | 1.20 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 501 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 22.3 | 391 | 25 | 28 | 0.78 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 490 | 31 | 35 | 0.98 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.5 | 601 | 39 | 44 | 1.20 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 304 | 20 | 22 | 0.61 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 504 | 32 | 37 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.6 | 177 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.2 | 541 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.1 | 437 | 28 | 32 | 0.87 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.2 | 541 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.6 | 602 | 39 | 44 | 1.20 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.5 | 458 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.6 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 511 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.4 | 365 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.0 | 595 | 38 | 43 | 1.30 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.8 | 420 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.8 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 18.5 | 496 | 32 | 36 | 1.18 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.6 | 273 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.6 | 137 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 27.9 | 348 | 22 | 25 | 1.28 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.18 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.4 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.4 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 926 | 60 | 67 | 0.91 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 85.8 | 707 | 45 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.4 | 1,020 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 51.6 | 1,175 | 76 | 85 | 1.15 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.0 | 341 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.5 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.7 | 512 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.7 | 701 | 45 | 51 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 52.2 | 1,164 | 75 | 84 | 1.14 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.0 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.1 | 1,016 | 65 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 227.0 | 269 | 17 | 20 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.3 | 907 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.0 | 694 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.3 | 908 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.3 | 1,146 | 74 | 83 | 1.13 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.2 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.3 | 943 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.9 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.7 | 859 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.2 | 676 | 43 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.7 | 859 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.5 | 1,129 | 73 | 82 | 1.20 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.0 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.8 | 872 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.9 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.8 | 461 | 30 | 33 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 65.5 | 957 | 62 | 69 | 1.10 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.6 | 292 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 111.1 | 583 | 37 | 42 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.4 | 210 | 14 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.7 | 328 | 21 | 24 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.7 | 328 | 21 | 24 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 101.6 | 637 | 41 | 46 | 1.09 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 264.9 | 244 | 16 | 18 | 0.42 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 905 | 58 | 66 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 38.4 | 790 | 51 | 57 | 0.87 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.2 | 736 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.6 | 664 | 43 | 48 | 0.73 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.4 | 790 | 51 | 57 | 0.87 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.1 | 1,008 | 65 | 73 | 1.11 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.1 | 329 | 21 | 24 | 0.36 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 905 | 58 | 66 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 65.9 | 462 | 30 | 33 | 0.51 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.1 | 647 | 42 | 47 | 0.71 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.9 | 727 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.1 | 1,009 | 65 | 73 | 1.11 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.3 | 330 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.8 | 905 | 58 | 66 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.2 | 174 | 11 | 13 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.9 | 713 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.5 | 630 | 41 | 46 | 0.70 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.9 | 713 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.6 | 999 | 64 | 72 | 1.10 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.5 | 331 | 21 | 24 | 0.37 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.9 | 836 | 54 | 61 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 340.9 | 91 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.2 | 684 | 44 | 50 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.2 | 580 | 37 | 42 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.5 | 981 | 63 | 71 | 1.17 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.1 | 332 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.3 | 652 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 661.7 | 48 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.7 | 300 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 38.2 | 822 | 53 | 60 | 1.26 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.8 | 312 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.5 | 400 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.5 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 62.4 | 523 | 34 | 38 | 1.31 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.5 | 328 | 21 | 24 | 0.82 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 16.38 | 0.00 | 3.00 | 1.16 / 1.21 | 0.00 / 1.86 (2.41) | 3.07 | 0.00 + 4.09 + 9.18 + 2.55 + 0.08 |
| qkv_proj M=8 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.89 | 1.46 / 1.46 | 0.00 / 2.50 (3.14) | 3.07 | 0.00 + 3.89 + 11.65 + 3.19 + 0.10 |
| qkv_proj M=16 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.52 | 1.48 / 1.74 | 0.00 / 2.66 (3.47) | 3.07 | 0.00 + 3.50 + 11.74 + 3.56 + 0.10 |
| qkv_proj M=64 | 128x128, 103 × 15 | 43.01 | 1.02 | 2.07 | 1.70 / 1.85 | 5.41 / 5.63 (8.19) | 10.24 | 0.00 + 2.08 + 25.66 + 14.65 + 0.10 |
| o_proj M=1 | 256x128, 103 × 5 | 14.34 | 0.00 | 3.17 | 1.14 / 1.30 | 1.87 / 1.77 (1.96) | 4.10 | 0.00 + 3.89 + 5.44 + 3.72 + 0.08 |
| o_proj M=8 | 256x128, 103 × 5 | 16.38 | 0.00 | 3.08 | 1.42 / 1.56 | 2.45 / 2.33 (2.50) | 5.12 | 0.00 + 4.31 + 7.03 + 4.91 + 0.10 |
| o_proj M=16 | 256x128, 103 × 5 | 17.41 | 0.00 | 2.70 | 1.45 / 1.87 | 2.56 / 2.54 (3.40) | 5.12 | 0.00 + 3.70 + 7.19 + 5.59 + 0.10 |
| o_proj M=64 | 128x128, 103 × 10 | 31.74 | 0.00 | 1.90 | 1.53 / 1.85 | 4.92 / 5.35 (8.52) | 10.24 | 0.00 + 1.90 + 15.22 + 14.15 + 0.10 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 50.18 | 0.00 | 3.34 | 1.16 / 1.34 | 0.99 / 1.74 (1.74) | 5.12 | 0.00 + 3.58 + 41.01 + 5.33 + 0.07 |
| gate_up_proj M=8 | 256x128, 106 × 34 | 62.46 | 0.00 | 3.34 | 1.50 / 1.61 | 1.29 / 2.20 (2.23) | 4.10 | 0.00 + 4.99 + 50.73 + 6.09 + 0.10 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 62.46 | 0.00 | 2.89 | 1.51 / 1.92 | 1.35 / 2.40 (2.42) | 4.10 | 0.00 + 4.06 + 51.27 + 6.58 + 0.09 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 131.07 | 0.00 | 1.99 | 1.65 / 1.27 | 5.33 / 11.00 (11.02) | 10.24 | 0.00 + 2.32 + 110.77 + 17.79 + 0.10 |
| down_proj M=1 | 256x128, 106 × 17 | 30.72 | 0.00 | 3.39 | 1.26 / 1.34 | 2.26 / 1.98 (2.36) | 5.12 | 0.00 + 4.10 + 21.34 + 4.92 + 0.08 |
| down_proj M=8 | 256x128, 106 × 17 | 31.74 | 0.00 | 2.93 | 1.30 / 1.61 | 2.42 / 2.04 (2.54) | 5.12 | 0.00 + 3.65 + 22.08 + 5.14 + 0.09 |
| down_proj M=16 | 256x128, 106 × 17 | 35.84 | 0.00 | 2.73 | 1.50 / 1.92 | 2.60 / 2.48 (3.43) | 6.14 | 0.00 + 3.30 + 25.58 + 6.76 + 0.10 |
| down_proj M=64 | 128x128, 106 × 34 | 77.82 | 0.00 | 2.08 | 1.72 / 1.90 | 5.37 / 5.41 (9.21) | 13.31 | 0.00 + 2.20 + 57.84 + 17.17 + 0.10 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | last, saw it early | SM GHz (runs) | globaltimer tick |
|---|---:|---:|---:|---:|---:|---:|
| qkv_proj M=1 | 0.59 | 1.04 | 0.72 | 27% | 1.30–1.36 | 1024 ns |
| qkv_proj M=8 | 0.73 | 1.41 | 0.93 | 6% | 1.05–1.09 | 1024 ns |
| qkv_proj M=16 | 0.73 | 1.53 | 1.14 | 4% | 1.06–1.09 | 1024 ns |
| qkv_proj M=64 | 2.19 | 2.71 | 5.23 | 88% | 1.07–1.09 | 1024 ns |
| o_proj M=1 | 0.57 | 0.99 | 0.89 | 50% | 1.35–1.43 | 1024 ns |
| o_proj M=8 | 0.73 | 1.34 | 1.24 | 53% | 1.04–1.44 | 1024 ns |
| o_proj M=16 | 0.74 | 1.46 | 1.80 | 50% | 1.04–1.10 | 1024 ns |
| o_proj M=64 | 2.01 | 2.63 | 4.82 | 50% | 1.17–1.20 | 1024 ns |
| gate_up_proj M=1 | 0.46 | 0.61 | 0.90 | 100% | 1.39–1.42 | 1024 ns |
| gate_up_proj M=8 | 0.59 | 0.81 | 1.18 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=16 | 0.60 | 0.86 | 1.37 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=64 | 0.97 | 2.41 | 9.03 | 100% | 1.09–1.10 | 1024 ns |
| down_proj M=1 | 0.64 | 1.11 | 1.00 | 59% | 1.23–1.27 | 1024 ns |
| down_proj M=8 | 0.64 | 1.18 | 1.09 | 62% | 1.08–1.26 | 1024 ns |
| down_proj M=16 | 0.73 | 1.45 | 1.81 | 62% | 1.07–1.09 | 1024 ns |
| down_proj M=64 | 2.18 | 2.71 | 5.43 | 62% | 1.08–1.09 | 1024 ns |

## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| bf16_async | paged | 1 | 128 | 256 | 12.04 | 83 | bf16 | 12.64 | 79 | 0.95x |
| bf16_async | paged | 16 | 1024 | 256 | 14.12 | 1,133 | bf16 | 14.84 | 1,078 | 0.95x |
| bf16_async | paged | 64 | 1024 | 256 | 22.57 | 2,835 | bf16 | 23.04 | 2,778 | 0.98x |
| w4a16_rows | paged | 1 | 128 | 256 | 5.68 | 176 | awq_marlin | 5.55 | 180 | 1.02x |
| w4a16_rows | paged | 16 | 1024 | 256 | 8.29 | 1,931 | awq_marlin | 7.85 | 2,039 | 1.06x |
| w4a16_rows | paged | 64 | 1024 | 256 | 22.94 | 2,790 | awq_marlin | 16.19 | 3,953 | 1.42x |
| w4a16_tiled | paged | 1 | 128 | 256 | 5.19 | 193 | awq_marlin | 5.55 | 180 | 0.93x |
| w4a16_tiled | paged | 16 | 1024 | 256 | 7.10 | 2,252 | awq_marlin | 7.85 | 2,039 | 0.91x |
| w4a16_tiled | paged | 64 | 1024 | 256 | 15.94 | 4,015 | awq_marlin | 16.19 | 3,953 | 0.98x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_b1 | 5.26 | 5.22 | 0.04 | 330 | 2 | gemm 83%, attention 8%, norm 3%, rope 2% |
| nano_w4a16_tiled_b16 | 7.02 | 6.98 | 0.04 | 332 | 2 | gemm 65%, attention 29%, norm 3%, rope 1% |
| nano_w4a16_tiled_b64 | 15.44 | 15.38 | 0.06 | 300 | 2 | gemm 52%, attention 45%, norm 1%, activation 1% |
| vllmw4_b1 | 5.66 | 5.59 | 0.07 | 366 | 15 | other 68%, gemm 13%, attention 11%, norm 4% |
| vllmw4_b16 | 7.81 | 7.73 | 0.08 | 431 | 15 | other 52%, attention 29%, gemm 10%, elementwise/copy 4% |
| vllmw4_b64 | 15.81 | 15.72 | 0.09 | 431 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

