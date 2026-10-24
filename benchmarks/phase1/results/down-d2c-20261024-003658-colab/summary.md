# Phase 1 results: run down-d2c-20261024-003658-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 104 |
| kbench | ok | 268 |
| trace | ok | 20 |
| golden | ok | 3 |
| e2e | ok | 534 |
| environment | ok | 3 |

## Environment

```
run_id: down-d2c-20261024-003658-colab
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
213 passed in 102.78s (0:01:42)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,387 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.9 | 689 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.0 | 590 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.3 | 641 | 41 | 46 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.4 | 607 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 729 | 47 | 53 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.2 | 804 | 52 | 58 | 1.17 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 690 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 45.9 | 284 | 18 | 21 | 0.41 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.4 | 641 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.6 | 577 | 37 | 42 | 0.84 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.3 | 799 | 51 | 58 | 1.16 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 695 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 72.2 | 182 | 12 | 13 | 0.26 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 640 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.3 | 541 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.6 | 639 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 820 | 53 | 59 | 1.18 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 320 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.2 | 627 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.4 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.7 | 480 | 31 | 35 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 624 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 830 | 53 | 60 | 1.32 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.3 | 499 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.1 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.1 | 340 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.7 | 691 | 44 | 50 | 1.38 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.4 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.7 | 224 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 41.0 | 348 | 22 | 25 | 1.09 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.3 | 500 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.4 | 528 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 556 | 36 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 531 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 656 | 42 | 47 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 14.6 | 595 | 38 | 43 | 1.19 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.3 | 505 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.9 | 313 | 20 | 23 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 489 | 31 | 35 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.5 | 600 | 39 | 43 | 1.19 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 304 | 20 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.5 | 503 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 60.4 | 145 | 9 | 10 | 0.29 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.1 | 437 | 28 | 32 | 0.87 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.6 | 601 | 39 | 43 | 1.19 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.5 | 457 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.7 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 511 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.4 | 365 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.9 | 599 | 39 | 43 | 1.31 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.8 | 421 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.7 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 257 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 18.5 | 496 | 32 | 36 | 1.18 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.6 | 272 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.3 | 138 | 9 | 10 | 0.51 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 34.2 | 284 | 18 | 20 | 1.04 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 324 | 21 | 23 | 1.19 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.1 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.4 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.5 | 925 | 59 | 67 | 0.90 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 86.0 | 705 | 45 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.5 | 1,020 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 51.4 | 1,179 | 76 | 85 | 1.15 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.4 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.6 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.6 | 513 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.4 | 916 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.8 | 701 | 45 | 51 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 52.2 | 1,166 | 75 | 84 | 1.14 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.2 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.2 | 1,014 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 227.3 | 269 | 17 | 19 | 0.27 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.2 | 908 | 58 | 66 | 0.90 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.0 | 694 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.2 | 909 | 58 | 66 | 0.90 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.2 | 1,148 | 74 | 83 | 1.13 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.1 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.4 | 942 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.9 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.7 | 860 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.1 | 677 | 44 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.7 | 859 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.5 | 1,131 | 73 | 82 | 1.20 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.9 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.8 | 873 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.7 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.7 | 462 | 30 | 33 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.3 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 65.6 | 956 | 61 | 69 | 1.10 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.6 | 292 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.3 | 572 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.3 | 210 | 14 | 15 | 0.37 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.5 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.8 | 327 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 99.9 | 648 | 42 | 47 | 1.13 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.3 | 244 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 904 | 58 | 65 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 47.6 | 636 | 41 | 46 | 0.70 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.3 | 735 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.5 | 666 | 43 | 48 | 0.74 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.4 | 790 | 51 | 57 | 0.87 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.0 | 1,011 | 65 | 73 | 1.12 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.8 | 330 | 21 | 24 | 0.37 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 906 | 58 | 65 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 65.9 | 462 | 30 | 33 | 0.51 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 728 | 47 | 52 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 46.9 | 649 | 42 | 47 | 0.72 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.8 | 727 | 47 | 52 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.1 | 1,011 | 65 | 73 | 1.12 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.2 | 330 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.7 | 907 | 58 | 65 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.4 | 173 | 11 | 13 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.9 | 712 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.6 | 630 | 40 | 45 | 0.69 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.9 | 712 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.5 | 1,002 | 64 | 72 | 1.10 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.4 | 331 | 21 | 24 | 0.36 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.8 | 838 | 54 | 60 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 340.9 | 91 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.2 | 682 | 44 | 49 | 0.81 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.1 | 582 | 37 | 42 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.4 | 982 | 63 | 71 | 1.17 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.8 | 333 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.3 | 651 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 663.1 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.9 | 300 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.67 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 434 | 28 | 31 | 0.67 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 38.1 | 825 | 53 | 59 | 1.27 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.5 | 313 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.6 | 400 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.7 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 81.3 | 401 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.4 | 328 | 21 | 24 | 0.82 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 16.38 | 0.00 | 2.96 | 1.15 / 1.21 | 0.00 / 1.82 (2.37) | 3.07 | 0.00 + 4.05 + 9.10 + 2.48 + 0.08 |
| qkv_proj M=8 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.90 | 1.46 / 1.46 | 0.00 / 2.50 (3.15) | 3.07 | 0.00 + 3.87 + 11.63 + 3.24 + 0.10 |
| qkv_proj M=16 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.53 | 1.48 / 1.74 | 0.00 / 2.66 (3.48) | 3.07 | 0.00 + 3.52 + 11.73 + 3.55 + 0.10 |
| qkv_proj M=64 | 64x128, 106 × 29 | 52.22 | 0.00 | 1.75 | 1.24 / 0.94 | 4.88 / 5.18 (7.84) | 10.24 | 0.00 + 1.78 + 35.95 + 14.48 + 0.11 |
| o_proj M=1 | 256x128, 103 × 5 | 14.34 | 0.00 | 3.17 | 1.13 / 1.30 | 1.87 / 1.77 (1.80) | 4.10 | 0.00 + 3.80 + 5.41 + 4.11 + 0.08 |
| o_proj M=8 | 256x128, 103 × 5 | 17.41 | 0.00 | 3.12 | 1.44 / 1.57 | 2.49 / 2.37 (2.70) | 5.12 | 0.00 + 4.29 + 7.12 + 5.00 + 0.10 |
| o_proj M=16 | 256x128, 103 × 5 | 17.41 | 0.00 | 2.71 | 1.45 / 1.87 | 2.56 / 2.54 (3.50) | 5.12 | 0.00 + 3.74 + 7.16 + 5.72 + 0.10 |
| o_proj M=64 | 64x128, 108 × 19 | 34.30 | 0.00 | 1.42 | 0.98 / 0.96 | 3.74 / 4.41 (8.69) | 11.26 | 0.00 + 1.45 + 18.76 + 14.51 + 0.09 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 50.18 | 0.00 | 3.34 | 1.16 / 1.34 | 0.99 / 1.74 (1.75) | 5.12 | 0.00 + 3.92 + 40.72 + 5.47 + 0.08 |
| gate_up_proj M=8 | 256x128, 106 × 34 | 62.46 | 0.00 | 3.34 | 1.49 / 1.61 | 1.28 / 2.20 (2.23) | 4.10 | 0.00 + 4.82 + 50.63 + 6.14 + 0.10 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 62.46 | 0.00 | 2.89 | 1.51 / 1.93 | 1.35 / 2.39 (2.42) | 4.10 | 0.00 + 4.50 + 51.22 + 6.54 + 0.09 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 129.02 | 0.00 | 1.98 | 1.62 / 1.27 | 5.30 / 10.90 (10.91) | 10.24 | 0.00 + 2.31 + 108.71 + 17.69 + 0.10 |
| down_proj M=1 | 256x128, 106 × 17 | 34.82 | 0.00 | 3.57 | 1.44 / 1.34 | 2.40 / 2.27 (2.73) | 5.12 | 0.00 + 4.34 + 24.52 + 5.26 + 0.10 |
| down_proj M=8 | 256x128, 106 × 17 | 35.84 | 0.00 | 3.16 | 1.49 / 1.61 | 2.51 / 2.34 (2.96) | 6.14 | 0.00 + 3.88 + 25.28 + 5.71 + 0.10 |
| down_proj M=16 | 256x128, 106 × 17 | 35.84 | 0.00 | 2.70 | 1.50 / 1.93 | 2.60 / 2.47 (3.51) | 6.14 | 0.00 + 3.35 + 25.52 + 6.61 + 0.10 |
| down_proj M=64 | 64x128, 107 × 67 | 103.94 | 0.00 | 1.75 | 1.25 / 0.95 | 4.77 / 5.17 (9.25) | 15.36 | 0.00 + 1.72 + 83.63 + 18.51 + 0.11 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | last, saw it early | SM GHz (runs) | globaltimer tick |
|---|---:|---:|---:|---:|---:|---:|
| qkv_proj M=1 | 0.59 | 1.02 | 0.71 | 29% | 1.32–1.37 | 1024 ns |
| qkv_proj M=8 | 0.73 | 1.42 | 0.92 | 6% | 1.06–1.09 | 1024 ns |
| qkv_proj M=16 | 0.73 | 1.54 | 1.14 | 4% | 1.06–1.09 | 1024 ns |
| qkv_proj M=64 | 1.05 | 3.21 | 5.58 | 92% | 1.08–1.41 | 1024 ns |
| o_proj M=1 | 0.57 | 0.99 | 0.88 | 53% | 1.35–1.43 | 1024 ns |
| o_proj M=8 | 0.74 | 1.37 | 1.26 | 50% | 1.04–1.09 | 1024 ns |
| o_proj M=16 | 0.74 | 1.46 | 1.80 | 48% | 1.04–1.09 | 1024 ns |
| o_proj M=64 | 0.85 | 2.72 | 5.55 | 44% | 1.07–1.42 | 1024 ns |
| gate_up_proj M=1 | 0.46 | 0.61 | 0.90 | 100% | 1.39–1.42 | 1024 ns |
| gate_up_proj M=8 | 0.60 | 0.81 | 1.19 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=16 | 0.59 | 0.86 | 1.37 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=64 | 0.96 | 2.40 | 8.96 | 100% | 1.09–1.10 | 1024 ns |
| down_proj M=1 | 0.72 | 1.27 | 1.15 | 56% | 1.07–1.10 | 1024 ns |
| down_proj M=8 | 0.73 | 1.34 | 1.26 | 59% | 1.07–1.09 | 1024 ns |
| down_proj M=16 | 0.73 | 1.44 | 1.81 | 62% | 1.06–1.09 | 1024 ns |
| down_proj M=64 | 1.05 | 3.30 | 6.86 | 81% | 1.09–1.10 | 1024 ns |

## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| w4a16_tiled | static | 1 | 128 | 256 | 42.55 | 24 | awq_marlin | 5.54 | 180 | 7.68x |
| w4a16_tiled | paged | 1 | 128 | 256 | 5.28 | 189 | awq_marlin | 5.54 | 180 | 0.95x |
| w4a16_tiled | paged | 16 | 1024 | 256 | 7.10 | 2,253 | awq_marlin | 7.85 | 2,037 | 0.90x |
| w4a16_tiled | paged | 64 | 1024 | 256 | 16.93 | 3,781 | awq_marlin | 16.18 | 3,955 | 1.05x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_b1 | 5.27 | 5.22 | 0.04 | 330 | 2 | gemm 83%, attention 8%, norm 3%, rope 2% |
| nano_w4a16_tiled_b16 | 7.02 | 6.98 | 0.05 | 332 | 2 | gemm 65%, attention 29%, norm 3%, rope 1% |
| nano_w4a16_tiled_b64 | 16.41 | 16.38 | 0.04 | 300 | 2 | gemm 55%, attention 42%, norm 1%, activation 1% |
| vllmw4_b1 | 5.66 | 5.58 | 0.07 | 365 | 15 | other 68%, gemm 13%, attention 11%, norm 4% |
| vllmw4_b16 | 7.82 | 7.74 | 0.08 | 430 | 15 | other 52%, attention 29%, gemm 10%, elementwise/copy 4% |
| vllmw4_b64 | 15.80 | 15.71 | 0.08 | 429 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

