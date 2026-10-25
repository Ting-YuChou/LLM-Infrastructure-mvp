# Phase 1 results: run down-d2b-20261024-001228-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 103 |
| kbench | ok | 268 |
| trace | ok | 35 |
| golden | ok | 4 |
| e2e | ok | 629 |
| environment | ok | 3 |

## Environment

```
run_id: down-d2b-20261024-001228-colab
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
213 passed in 101.14s (0:01:41)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,387 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.8 | 690 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 17.9 | 727 | 47 | 52 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.3 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.4 | 608 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.9 | 727 | 47 | 52 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.1 | 805 | 52 | 58 | 1.17 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 36.3 | 360 | 23 | 26 | 0.52 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.3 | 643 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.7 | 576 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 643 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.3 | 802 | 52 | 58 | 1.16 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 318 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.0 | 693 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 72.2 | 182 | 12 | 13 | 0.26 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.3 | 541 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 824 | 53 | 59 | 1.19 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.2 | 627 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.4 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 624 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.7 | 480 | 31 | 35 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.3 | 818 | 53 | 59 | 1.30 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.3 | 499 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.1 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.2 | 340 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.1 | 340 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.9 | 684 | 44 | 49 | 1.37 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.1 | 324 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.9 | 196 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 37.8 | 378 | 24 | 27 | 1.19 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.3 | 501 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.4 | 528 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 555 | 36 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 531 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 656 | 42 | 47 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 14.7 | 590 | 38 | 43 | 1.18 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 502 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.9 | 312 | 20 | 23 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 488 | 31 | 35 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.8 | 591 | 38 | 43 | 1.18 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 303 | 20 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 504 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 54.3 | 162 | 10 | 12 | 0.32 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 540 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.2 | 436 | 28 | 31 | 0.86 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.0 | 587 | 38 | 42 | 1.16 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.5 | 458 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.6 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 511 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.5 | 365 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.3 | 582 | 37 | 42 | 1.27 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.8 | 420 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.8 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 18.5 | 497 | 32 | 36 | 1.18 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.5 | 273 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.3 | 138 | 9 | 10 | 0.51 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 31.6 | 307 | 20 | 22 | 1.12 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.18 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.1 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.4 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 927 | 60 | 67 | 0.90 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 85.9 | 706 | 45 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.4 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 51.6 | 1,176 | 76 | 85 | 1.15 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.2 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.4 | 1,024 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.8 | 512 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.7 | 701 | 45 | 51 | 0.68 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 52.3 | 1,162 | 75 | 84 | 1.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.1 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.2 | 1,015 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 227.2 | 269 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.3 | 908 | 58 | 65 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.9 | 695 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.3 | 908 | 58 | 65 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.2 | 1,147 | 74 | 83 | 1.13 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.1 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.4 | 943 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.9 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.7 | 860 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.1 | 676 | 43 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.7 | 860 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.7 | 1,126 | 72 | 81 | 1.19 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.0 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.8 | 872 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.6 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.6 | 462 | 30 | 33 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 552 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 66.4 | 943 | 61 | 68 | 1.08 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.4 | 292 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 106.4 | 609 | 39 | 44 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.3 | 210 | 14 | 15 | 0.35 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.6 | 328 | 21 | 24 | 0.54 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.8 | 327 | 21 | 24 | 0.54 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 101.6 | 637 | 41 | 46 | 1.05 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.4 | 244 | 16 | 18 | 0.40 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 903 | 58 | 65 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 38.3 | 792 | 51 | 57 | 0.88 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.2 | 735 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.5 | 666 | 43 | 48 | 0.74 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.4 | 790 | 51 | 57 | 0.87 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.3 | 1,001 | 64 | 72 | 1.11 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.8 | 330 | 21 | 24 | 0.37 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 906 | 58 | 65 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 65.9 | 462 | 30 | 33 | 0.51 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 728 | 47 | 52 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 46.9 | 648 | 42 | 47 | 0.72 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.8 | 728 | 47 | 52 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.5 | 998 | 64 | 72 | 1.10 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.2 | 330 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.8 | 905 | 58 | 65 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 175.9 | 174 | 11 | 13 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.9 | 712 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.5 | 630 | 41 | 45 | 0.70 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.8 | 714 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.7 | 995 | 64 | 72 | 1.10 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.3 | 331 | 21 | 24 | 0.37 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.9 | 836 | 54 | 60 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 340.2 | 91 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.1 | 581 | 37 | 42 | 0.70 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.7 | 973 | 63 | 70 | 1.16 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.9 | 332 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.3 | 651 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 662.3 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.9 | 300 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 38.6 | 816 | 52 | 59 | 1.25 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.4 | 313 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.6 | 400 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.5 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.1 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 72.0 | 453 | 29 | 33 | 1.13 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.5 | 328 | 21 | 24 | 0.82 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 19.46 | 0.00 | 3.71 | 1.37 / 1.21 | 0.00 / 2.16 (2.92) | 3.07 | 0.00 + 4.60 + 10.94 + 3.17 + 0.11 |
| qkv_proj M=1, first 3 tiles in L2 | 256x128, 96 × 8 | 17.41 | 0.00 | 2.47 | 1.37 / 1.21 | 0.00 / 2.43 (2.97) | 2.05 | 0.00 + 2.75 + 10.93 + 3.18 + 0.11 |
| qkv_proj M=8 | 256x128, 96 × 8 | 18.43 | 1.02 | 2.86 | 1.42 / 1.45 | 0.00 / 2.36 (3.14) | 3.07 | 0.00 + 3.86 + 11.26 + 3.26 + 0.11 |
| qkv_proj M=8, first 3 tiles in L2 | 256x128, 96 × 8 | 17.41 | 0.00 | 2.17 | 1.40 / 1.45 | 0.00 / 2.57 (3.14) | 2.05 | 0.00 + 2.44 + 11.20 + 3.33 + 0.15 |
| qkv_proj M=16 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.50 | 1.44 / 1.74 | 0.00 / 2.72 (3.77) | 3.07 | 0.00 + 3.38 + 11.53 + 3.96 + 0.11 |
| qkv_proj M=16, first 3 tiles in L2 | 256x128, 96 × 8 | 18.43 | 0.00 | 1.87 | 1.44 / 1.74 | 0.00 / 3.08 (3.83) | 2.05 | 0.00 + 1.89 + 11.55 + 4.41 + 0.15 |
| qkv_proj M=64 | 64x128, 106 × 29 | 51.20 | 0.00 | 1.65 | 1.17 / 0.94 | 5.24 / 5.73 (8.19) | 10.24 | 0.00 + 1.68 + 34.09 + 15.35 + 0.11 |
| qkv_proj M=64, first 3 tiles in L2 | 64x128, 106 × 29 | 51.20 | 1.02 | 1.63 | 1.17 / 0.94 | 5.21 / 5.74 (8.14) | 10.24 | 0.00 + 1.66 + 33.99 + 15.41 + 0.11 |
| o_proj M=1 | 256x128, 103 × 5 | 14.34 | 0.00 | 3.71 | 1.06 / 1.29 | 2.07 / 1.72 (1.86) | 5.12 | 0.00 + 4.81 + 5.28 + 3.90 + 0.08 |
| o_proj M=1, first 3 tiles in L2 | 256x128, 103 × 5 | 12.29 | 0.00 | 1.98 | 1.07 / 1.29 | 2.13 / 1.73 (1.80) | 4.10 | 0.00 + 2.08 + 5.37 + 4.72 + 0.08 |
| o_proj M=8 | 256x128, 103 × 5 | 17.41 | 0.00 | 3.07 | 1.39 / 1.56 | 2.75 / 2.32 (2.49) | 5.12 | 0.00 + 4.27 + 6.96 + 5.27 + 0.11 |
| o_proj M=8, first 3 tiles in L2 | 256x128, 103 × 5 | 15.36 | 0.00 | 2.17 | 1.38 / 1.56 | 2.72 / 2.31 (2.39) | 4.10 | 0.00 + 2.30 + 6.97 + 5.64 + 0.11 |
| o_proj M=16 | 256x128, 103 × 5 | 16.38 | 0.00 | 2.55 | 1.29 / 1.86 | 2.66 / 2.35 (2.87) | 6.14 | 0.00 + 3.67 + 6.34 + 5.70 + 0.10 |
| o_proj M=16, first 3 tiles in L2 | 256x128, 103 × 5 | 15.36 | 0.00 | 1.80 | 1.37 / 1.86 | 2.88 / 2.60 (3.01) | 5.12 | 0.00 + 1.96 + 6.88 + 6.46 + 0.10 |
| o_proj M=64 | 64x128, 108 × 19 | 41.98 | 1.02 | 1.67 | 1.18 / 0.96 | 5.17 / 6.04 (10.86) | 12.29 | 0.00 + 1.64 + 22.56 + 17.73 + 0.10 |
| o_proj M=64, first 3 tiles in L2 | 64x128, 108 × 19 | 32.77 | 0.00 | 1.29 | 0.92 / 0.96 | 4.01 / 4.74 (8.52) | 10.24 | 0.00 + 1.33 + 17.47 + 13.84 + 0.08 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 50.18 | 0.00 | 3.89 | 1.12 / 1.33 | 1.09 / 1.73 (1.74) | 5.12 | 0.00 + 4.65 + 39.39 + 5.74 + 0.08 |
| gate_up_proj M=1, first 3 tiles in L2 | 256x128, 106 × 34 | 59.39 | 0.51 | 2.43 | 1.40 / 1.33 | 1.40 / 2.17 (2.18) | 4.10 | 0.00 + 2.57 + 48.27 + 8.17 + 0.10 |
| gate_up_proj M=8 | 256x128, 106 × 34 | 61.44 | 0.00 | 3.24 | 1.45 / 1.60 | 1.41 / 2.26 (2.28) | 4.10 | 0.00 + 4.64 + 49.26 + 6.63 + 0.10 |
| gate_up_proj M=8, first 3 tiles in L2 | 256x128, 106 × 34 | 60.42 | 0.00 | 2.16 | 1.45 / 1.60 | 1.40 / 2.25 (2.27) | 4.10 | 0.00 + 2.26 + 49.34 + 8.11 + 0.10 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 62.46 | 0.00 | 2.86 | 1.48 / 1.92 | 1.48 / 2.57 (2.59) | 4.10 | 0.00 + 3.69 + 50.62 + 7.25 + 0.10 |
| gate_up_proj M=16, first 3 tiles in L2 | 256x128, 106 × 34 | 61.44 | 0.00 | 1.81 | 1.47 / 1.92 | 1.48 / 2.57 (2.60) | 4.10 | 0.00 + 1.92 + 49.91 + 8.67 + 0.10 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 131.07 | 0.00 | 1.98 | 1.64 / 1.26 | 5.34 / 11.05 (11.05) | 9.73 | 0.00 + 2.36 + 110.26 + 17.80 + 0.10 |
| gate_up_proj M=64, first 3 tiles in L2 | 64x256, 107 × 67 | 113.66 | 0.00 | 1.47 | 1.43 / 1.26 | 4.66 / 9.89 (9.90) | 8.19 | 0.00 + 1.48 + 96.01 + 15.79 + 0.09 |
| down_proj M=1 | 256x128, 106 × 17 | 34.82 | 1.02 | 4.08 | 1.40 / 1.33 | 2.65 / 2.17 (2.35) | 6.14 | 0.00 + 5.11 + 23.80 + 5.49 + 0.11 |
| down_proj M=1, first 3 tiles in L2 | 256x128, 106 × 17 | 34.82 | 0.00 | 2.45 | 1.43 / 1.33 | 3.16 / 2.16 (2.29) | 6.14 | 0.00 + 2.49 + 23.87 + 7.83 + 0.11 |
| down_proj M=8 | 256x128, 106 × 17 | 34.82 | 0.00 | 3.09 | 1.45 / 1.60 | 2.80 / 2.28 (2.47) | 6.14 | 0.00 + 3.85 + 24.67 + 6.13 + 0.11 |
| down_proj M=8, first 3 tiles in L2 | 256x128, 106 × 17 | 35.84 | 0.00 | 2.18 | 1.46 / 1.60 | 3.33 / 2.27 (2.44) | 7.17 | 0.00 + 2.27 + 24.65 + 7.99 + 0.11 |
| down_proj M=16 | 256x128, 106 × 17 | 35.84 | 0.00 | 2.68 | 1.47 / 1.92 | 3.00 / 2.55 (3.11) | 7.17 | 0.00 + 3.32 + 25.12 + 7.10 + 0.10 |
| down_proj M=16, first 3 tiles in L2 | 256x128, 106 × 17 | 35.84 | 0.00 | 1.83 | 1.48 / 1.92 | 3.60 / 2.55 (3.04) | 7.17 | 0.00 + 1.97 + 24.93 + 8.69 + 0.10 |
| down_proj M=64 | 64x128, 107 × 67 | 99.33 | 0.00 | 1.67 | 1.19 / 0.95 | 5.19 / 5.72 (9.71) | 13.31 | 0.00 + 1.64 + 79.82 + 17.57 + 0.10 |
| down_proj M=64, first 3 tiles in L2 | 64x128, 107 × 67 | 76.80 | 0.00 | 1.28 | 0.92 / 0.95 | 4.05 / 4.44 (7.96) | 10.24 | 0.00 + 1.33 + 61.98 + 13.72 + 0.08 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | last, saw it early | SM GHz (runs) | globaltimer tick |
|---|---:|---:|---:|---:|---:|---:|
| qkv_proj M=1 | 0.84 | 1.02 | 1.01 | 27% | 1.06–1.09 | 1024 ns |
| qkv_proj M=1, first 3 tiles in L2 | 0.83 | 1.11 | 1.01 | 0% | 1.04–1.10 | 1024 ns |
| qkv_proj M=8 | 0.84 | 1.22 | 1.09 | 8% | 1.05–1.10 | 1024 ns |
| qkv_proj M=8, first 3 tiles in L2 | 0.83 | 1.23 | 1.09 | 0% | 1.05–1.11 | 1024 ns |
| qkv_proj M=16 | 0.84 | 1.51 | 1.43 | 2% | 1.05–1.09 | 1024 ns |
| qkv_proj M=16, first 3 tiles in L2 | 0.84 | 1.56 | 1.43 | 0% | 1.05–1.09 | 1024 ns |
| qkv_proj M=64 | 2.14 | 2.63 | 5.30 | 81% | 1.08–1.11 | 1024 ns |
| qkv_proj M=64, first 3 tiles in L2 | 2.14 | 2.63 | 5.26 | 83% | 1.09–1.11 | 1024 ns |
| o_proj M=1 | 0.67 | 0.80 | 1.04 | 56% | 1.33–1.41 | 1024 ns |
| o_proj M=1, first 3 tiles in L2 | 0.68 | 0.80 | 1.06 | 78% | 1.33–1.40 | 1024 ns |
| o_proj M=8 | 0.86 | 1.13 | 1.50 | 59% | 1.04–1.10 | 1024 ns |
| o_proj M=8, first 3 tiles in L2 | 0.85 | 1.13 | 1.49 | 78% | 1.02–1.09 | 1024 ns |
| o_proj M=16 | 0.78 | 1.24 | 1.90 | 56% | 1.15–1.22 | 1024 ns |
| o_proj M=16, first 3 tiles in L2 | 0.84 | 1.37 | 2.07 | 78% | 1.03–1.21 | 1024 ns |
| o_proj M=64 | 2.18 | 2.80 | 6.35 | 34% | 1.07–1.10 | 1024 ns |
| o_proj M=64, first 3 tiles in L2 | 1.69 | 2.16 | 4.96 | 31% | 1.38–1.41 | 1024 ns |
| gate_up_proj M=1 | 0.75 | 0.38 | 1.02 | 100% | 1.39–1.41 | 1024 ns |
| gate_up_proj M=1, first 3 tiles in L2 | 0.97 | 0.49 | 1.26 | 100% | 1.08–1.09 | 1024 ns |
| gate_up_proj M=8 | 0.97 | 0.51 | 1.42 | 100% | 1.08–1.09 | 1024 ns |
| gate_up_proj M=8, first 3 tiles in L2 | 0.97 | 0.51 | 1.42 | 100% | 1.08–1.09 | 1024 ns |
| gate_up_proj M=16 | 0.97 | 0.62 | 1.70 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=16, first 3 tiles in L2 | 0.97 | 0.63 | 1.70 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=64 | 0.98 | 2.41 | 9.09 | 100% | 1.09–1.09 | 1024 ns |
| gate_up_proj M=64, first 3 tiles in L2 | 0.86 | 2.10 | 8.19 | 100% | 1.25–1.26 | 1024 ns |
| down_proj M=1 | 0.85 | 1.01 | 1.39 | 72% | 1.07–1.09 | 1024 ns |
| down_proj M=1, first 3 tiles in L2 | 0.84 | 1.01 | 1.38 | 88% | 1.07–1.10 | 1024 ns |
| down_proj M=8 | 0.85 | 1.11 | 1.52 | 78% | 1.07–1.10 | 1024 ns |
| down_proj M=8, first 3 tiles in L2 | 0.84 | 1.11 | 1.51 | 91% | 1.07–1.09 | 1024 ns |
| down_proj M=16 | 0.84 | 1.36 | 2.12 | 81% | 1.06–1.09 | 1024 ns |
| down_proj M=16, first 3 tiles in L2 | 0.84 | 1.36 | 2.10 | 91% | 1.07–1.09 | 1024 ns |
| down_proj M=64 | 2.17 | 2.80 | 6.23 | 59% | 1.09–1.09 | 1024 ns |
| down_proj M=64, first 3 tiles in L2 | 1.68 | 2.25 | 4.85 | 50% | 1.39–1.41 | 1024 ns |

## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| w4a16_tiled | static | 1 | 128 | 256 | 42.80 | 23 | awq_marlin | 5.54 | 180 | 7.72x |
| w4a16_tiled | paged | 1 | 128 | 256 | 5.24 | 191 | awq_marlin | 5.54 | 180 | 0.95x |
| w4a16_tiled | paged | 16 | 1024 | 256 | 7.14 | 2,241 | awq_marlin | 7.85 | 2,038 | 0.91x |
| w4a16_tiled | paged | 64 | 1024 | 256 | 16.50 | 3,880 | awq_marlin | 16.18 | 3,956 | 1.02x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_b1 | 5.27 | 5.22 | 0.04 | 330 | 2 | gemm 83%, attention 8%, norm 3%, rope 2% |
| nano_w4a16_tiled_b16 | 7.05 | 7.01 | 0.04 | 332 | 2 | gemm 65%, attention 29%, norm 3%, rope 1% |
| nano_w4a16_tiled_b64 | 15.99 | 15.95 | 0.04 | 300 | 2 | gemm 54%, attention 43%, norm 1%, activation 1% |
| vllmw4_b1 | 5.67 | 5.59 | 0.08 | 366 | 15 | other 68%, gemm 13%, attention 11%, norm 4% |
| vllmw4_b16 | 7.81 | 7.73 | 0.08 | 429 | 15 | other 52%, attention 30%, gemm 10%, elementwise/copy 4% |
| vllmw4_b64 | 15.81 | 15.73 | 0.08 | 429 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

