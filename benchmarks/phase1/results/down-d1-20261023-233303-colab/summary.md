# Phase 1 results: run down-d1-20261023-233303-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_nsys | ok | 0 |
| download_model | ok | 59 |
| install_vllm | ok | 92 |
| cuda129 | ok | 23 |
| tests | ok | 102 |
| kbench | ok | 274 |
| ncu | ok | 305 |
| trace | ok | 20 |
| golden | ok | 4 |
| environment | ok | 2 |

## Environment

```
run_id: down-d1-20261023-233303-colab
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
213 passed in 100.31s (0:01:40)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,384 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.9 | 689 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.0 | 591 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.2 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.4 | 609 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 730 | 47 | 53 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 17.4 | 747 | 48 | 54 | 1.08 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 37.4 | 349 | 22 | 25 | 0.51 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.4 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.7 | 576 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 17.4 | 749 | 48 | 54 | 1.08 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 694 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 56.8 | 231 | 15 | 17 | 0.33 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.3 | 542 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 17.8 | 739 | 48 | 53 | 1.06 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 320 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.3 | 626 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.4 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 624 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.7 | 480 | 31 | 35 | 0.77 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 18.0 | 739 | 48 | 53 | 1.18 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.4 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.3 | 499 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.3 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 22.6 | 603 | 39 | 44 | 1.21 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.6 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.7 | 224 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 39.7 | 360 | 23 | 26 | 1.13 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.3 | 502 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 14.8 | 586 | 38 | 42 | 1.17 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 17.0 | 509 | 33 | 37 | 1.01 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 531 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 657 | 42 | 48 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 15.6 | 556 | 36 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.3 | 505 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.9 | 313 | 20 | 23 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 489 | 31 | 35 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.6 | 559 | 36 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 304 | 20 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 504 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.4 | 178 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 540 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.1 | 437 | 28 | 32 | 0.87 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.4 | 569 | 37 | 41 | 1.13 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.4 | 460 | 30 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.6 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.5 | 364 | 23 | 26 | 0.79 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 511 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 558 | 36 | 40 | 1.21 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.8 | 420 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.8 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 257 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 20.4 | 449 | 29 | 32 | 1.07 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.6 | 273 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.3 | 138 | 9 | 10 | 0.51 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 33.0 | 294 | 19 | 21 | 1.08 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.19 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.0 | 1,027 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.5 | 1,020 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 927 | 60 | 67 | 0.90 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 86.1 | 704 | 45 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.5 | 1,020 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 52.3 | 1,158 | 74 | 84 | 1.13 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.2 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.7 | 1,018 | 65 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.6 | 513 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.2 | 918 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.3 | 704 | 45 | 51 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.0 | 1,127 | 72 | 81 | 1.11 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.9 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.1 | 1,017 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 226.9 | 269 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.3 | 908 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.0 | 694 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.4 | 907 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.7 | 1,116 | 72 | 81 | 1.10 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.2 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.2 | 945 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 454.0 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.6 | 860 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.1 | 677 | 44 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.7 | 860 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 58.5 | 1,053 | 68 | 76 | 1.12 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.9 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 68.8 | 910 | 59 | 66 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.7 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.6 | 462 | 30 | 33 | 0.51 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 72.9 | 859 | 55 | 62 | 0.94 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.5 | 292 | 19 | 21 | 0.32 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 110.6 | 585 | 38 | 42 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.3 | 210 | 14 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.6 | 328 | 21 | 24 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.7 | 328 | 21 | 24 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 107.2 | 604 | 39 | 44 | 1.03 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.0 | 244 | 16 | 18 | 0.42 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 903 | 58 | 65 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 38.3 | 792 | 51 | 57 | 0.88 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.3 | 735 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.5 | 666 | 43 | 48 | 0.74 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.3 | 792 | 51 | 57 | 0.88 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.7 | 988 | 64 | 71 | 1.09 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.7 | 330 | 21 | 24 | 0.37 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 907 | 58 | 66 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 73.3 | 415 | 27 | 30 | 0.46 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.0 | 647 | 42 | 47 | 0.71 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.2 | 974 | 63 | 70 | 1.07 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.1 | 330 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.7 | 906 | 58 | 65 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.9 | 173 | 11 | 12 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.9 | 713 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.5 | 630 | 41 | 46 | 0.69 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.9 | 713 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.5 | 969 | 62 | 70 | 1.07 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.3 | 331 | 21 | 24 | 0.37 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.9 | 836 | 54 | 60 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 341.3 | 90 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.2 | 580 | 37 | 42 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.3 | 681 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 32.8 | 940 | 60 | 68 | 1.12 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.8 | 333 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.3 | 652 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 662.6 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.8 | 300 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 41.1 | 765 | 49 | 55 | 1.17 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.5 | 313 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.5 | 401 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.7 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.1 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.3 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 73.3 | 445 | 29 | 32 | 1.11 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.4 | 328 | 21 | 24 | 0.82 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 20.99 | 0.00 | 3.69 | 1.50 / 1.20 | 0.00 / 2.55 (3.17) | 4.10 | 0.00 + 5.25 + 11.95 + 3.24 + 0.10 |
| qkv_proj M=8 | 256x128, 96 × 8 | 21.50 | 0.00 | 3.36 | 1.55 / 1.45 | 0.00 / 2.63 (3.28) | 4.10 | 0.00 + 4.93 + 13.13 + 3.29 + 0.10 |
| qkv_proj M=16 | 256x128, 96 × 8 | 23.55 | 1.02 | 3.41 | 1.63 / 1.74 | 0.00 / 2.80 (3.57) | 6.14 | 1.02 + 5.23 + 14.42 + 3.64 + 0.10 |
| qkv_proj M=64 | 64x128, 106 × 29 | 53.76 | 0.00 | 1.88 | 1.19 / 0.94 | 4.98 / 5.36 (10.38) | 14.34 | 0.00 + 2.04 + 34.73 + 16.52 + 0.11 |
| o_proj M=1 | 256x128, 103 × 5 | 15.36 | 0.00 | 3.39 | 1.16 / 1.29 | 1.68 / 1.69 (2.63) | 8.19 | 0.00 + 4.05 + 6.53 + 4.33 + 0.08 |
| o_proj M=8 | 256x128, 103 × 5 | 15.36 | 0.00 | 3.00 | 1.18 / 1.56 | 1.76 / 1.74 (2.73) | 8.19 | 0.00 + 3.92 + 6.29 + 4.56 + 0.08 |
| o_proj M=16 | 256x128, 103 × 5 | 20.48 | 1.02 | 3.29 | 1.55 / 1.86 | 2.35 / 2.33 (4.24) | 11.26 | 0.51 + 4.67 + 7.85 + 6.97 + 0.10 |
| o_proj M=64 | 64x128, 108 × 19 | 44.03 | 0.00 | 1.88 | 1.19 / 0.96 | 4.90 / 5.54 (11.99) | 18.43 | 0.00 + 1.88 + 23.20 + 18.29 + 0.11 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 53.25 | 0.00 | 3.73 | 1.23 / 1.33 | 1.03 / 2.64 (2.67) | 30.72 | 0.00 + 4.60 + 42.81 + 6.22 + 0.08 |
| gate_up_proj M=8 | 256x128, 106 × 34 | 67.58 | 0.00 | 3.51 | 1.61 / 1.60 | 1.32 / 3.42 (3.44) | 38.91 | 0.00 + 4.17 + 55.69 + 7.18 + 0.10 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 71.68 | 0.00 | 3.36 | 1.72 / 1.92 | 1.40 / 3.76 (3.79) | 43.01 | 0.00 + 4.73 + 59.07 + 7.78 + 0.10 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 139.26 | 0.00 | 2.29 | 1.71 / 1.26 | 5.12 / 15.59 (15.61) | 14.34 | 0.00 + 2.51 + 114.63 + 22.10 + 0.10 |
| down_proj M=1 | 256x128, 106 × 17 | 37.89 | 0.00 | 4.01 | 1.55 / 1.33 | 2.23 / 2.15 (3.41) | 20.99 | 0.00 + 5.38 + 26.24 + 5.96 + 0.10 |
| down_proj M=8 | 256x128, 106 × 17 | 37.89 | 0.00 | 3.42 | 1.59 / 1.60 | 2.36 / 2.19 (3.53) | 21.50 | 0.00 + 4.19 + 27.51 + 6.42 + 0.10 |
| down_proj M=16 | 256x128, 106 × 17 | 40.96 | 0.00 | 3.28 | 1.69 / 1.92 | 2.45 / 2.29 (4.26) | 23.55 | 0.00 + 4.29 + 29.37 + 7.27 + 0.10 |
| down_proj M=64 | 64x128, 107 × 67 | 100.35 | 0.00 | 1.83 | 1.19 / 0.95 | 4.91 / 5.28 (11.51) | 14.34 | 0.00 + 1.94 + 80.10 + 18.57 + 0.10 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | globaltimer tick |
|---|---:|---:|---:|---:|
| qkv_proj M=1 | 0.63 | 1.40 | 0.84 | 16384 ns |
| qkv_proj M=8 | 0.63 | 1.47 | 0.87 | 16384 ns |
| qkv_proj M=16 | 0.63 | 1.55 | 1.10 | 17408 ns |
| qkv_proj M=64 | 1.69 | 2.92 | 5.45 | 37888 ns |
| o_proj M=1 | 0.45 | 1.03 | 0.85 | 6144 ns |
| o_proj M=8 | 0.45 | 1.09 | 0.91 | 6144 ns |
| o_proj M=16 | 0.59 | 1.49 | 1.80 | 8192 ns |
| o_proj M=64 | 1.69 | 3.08 | 7.02 | 24576 ns |
| gate_up_proj M=1 | 0.45 | 1.30 | 0.89 | 21504 ns |
| gate_up_proj M=8 | 0.58 | 1.55 | 1.17 | 26624 ns |
| gate_up_proj M=16 | 0.58 | 1.70 | 1.39 | 28672 ns |
| gate_up_proj M=64 | 0.60 | 4.50 | 9.40 | 123904 ns |
| down_proj M=1 | 0.58 | 1.35 | 1.12 | 16384 ns |
| down_proj M=8 | 0.58 | 1.41 | 1.18 | 15360 ns |
| down_proj M=16 | 0.58 | 1.50 | 1.82 | 16384 ns |
| down_proj M=64 | 1.68 | 2.97 | 6.69 | 84992 ns |

## Kernels under ncu

### ncu kbench_w4a16_linear: 87 kernels, 5349 us, 560 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 4>` | gemm | 6 | 15 | 133.6 | 17 | 267 | 19 | 12/12 | 191 | 75 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 4, 2>` | gemm | 3 | 13 | 239.7 | 9 | 144 | 16 | 12/12 | 188 | 85 | long_scoreboard, wait |
| `void <unnamed>::gemv_w4a16_kernel<__nv_bfloat16, 1, 2>` | gemm | 18 | 13 | 38.9 | 52 | 891 | 55 | 35/62 | 48 | 14 | long_scoreboard, not_selected |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 1, 4>` | gemm | 15 | 13 | 44.7 | 46 | 760 | 28 | 15/19 | 130 | 25 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 1>` | gemm | 12 | 12 | 52.1 | 40 | 664 | 24 | 12/25 | 124 | 23 | long_scoreboard, wait |
| `void <unnamed>::gemv_w4a16_kernel<__nv_bfloat16, 8, 2>` | gemm | 3 | 10 | 182.3 | 12 | 179 | 30 | 23/25 | 104 | 45 | long_scoreboard, selected |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 1, 25` | gemm | 12 | 8 | 35.9 | 56 | 956 | 29 | 12/12 | 193 | 56 | wait, selected |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 12 | 8 | 35.3 | 58 | 968 | 27 | 12/12 | 128 | 16 | long_scoreboard, selected |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 4, 64` | gemm | 3 | 4 | 75.6 | 30 | 463 | 32 | 12/12 | 255 | 80 | wait, selected |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 3 | 4 | 68.0 | 33 | 519 | 36 | 6/6 | 255 | 72 | wait, selected |

