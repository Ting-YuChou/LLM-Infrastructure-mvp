# Phase 1 results: run marlin-t1d-20261023-103133-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 94 |
| kbench | ok | 86 |
| environment | ok | 3 |

## Environment

```
run_id: marlin-t1d-20261023-103133-colab
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
207 passed in 92.46s (0:01:32)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,387 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 19.1 | 680 | 44 | 49 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.2 | 586 | 38 | 42 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 25.1 | 517 | 33 | 37 | 0.76 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.8 | 595 | 38 | 43 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.9 | 726 | 47 | 52 | 1.07 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 18.5 | 703 | 45 | 51 | 1.04 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 40.9 | 318 | 20 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 683 | 44 | 49 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 46.2 | 282 | 18 | 20 | 0.41 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 21.2 | 616 | 40 | 44 | 0.90 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 23.2 | 563 | 36 | 41 | 0.82 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 21.2 | 617 | 40 | 44 | 0.90 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 18.6 | 703 | 45 | 51 | 1.03 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 318 | 20 | 23 | 0.47 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 687 | 44 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 57.1 | 230 | 15 | 17 | 0.34 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 21.3 | 617 | 40 | 44 | 0.90 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.7 | 532 | 34 | 38 | 0.77 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 21.4 | 615 | 40 | 44 | 0.90 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 18.7 | 704 | 45 | 51 | 1.03 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.2 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.2 | 627 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 112.0 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 22.1 | 602 | 39 | 43 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 28.2 | 472 | 30 | 34 | 0.75 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 22.1 | 601 | 39 | 43 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.1 | 696 | 45 | 50 | 1.11 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 320 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.6 | 495 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 222.8 | 61 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 31.1 | 439 | 28 | 32 | 0.89 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.5 | 336 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.5 | 336 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 23.5 | 579 | 37 | 42 | 1.17 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 45.2 | 316 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 93.5 | 153 | 10 | 11 | 0.48 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 65.0 | 220 | 14 | 16 | 0.69 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 65.1 | 219 | 14 | 16 | 0.69 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 43.5 | 329 | 21 | 24 | 1.04 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.06 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.5 | 494 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 15.8 | 549 | 35 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 19.2 | 451 | 29 | 32 | 0.91 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 19.4 | 447 | 29 | 32 | 0.90 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.3 | 654 | 42 | 47 | 1.32 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 16.3 | 532 | 34 | 38 | 1.08 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.7 | 302 | 19 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.6 | 495 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 28.1 | 310 | 20 | 22 | 0.63 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.8 | 518 | 33 | 37 | 1.05 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 18.3 | 477 | 31 | 34 | 0.96 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.8 | 518 | 33 | 37 | 1.05 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.3 | 533 | 34 | 38 | 1.08 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 301 | 19 | 22 | 0.61 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.7 | 497 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 61.7 | 142 | 9 | 10 | 0.29 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.1 | 513 | 33 | 37 | 1.03 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.6 | 427 | 27 | 31 | 0.86 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.2 | 512 | 33 | 37 | 1.03 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.2 | 543 | 35 | 39 | 1.09 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 303 | 19 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.7 | 452 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 92.0 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 18.2 | 489 | 31 | 35 | 1.08 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.7 | 360 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 18.2 | 489 | 31 | 35 | 1.08 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.6 | 537 | 35 | 39 | 1.19 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.1 | 306 | 20 | 22 | 0.68 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.2 | 414 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 176.9 | 52 | 3 | 4 | 0.13 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.7 | 257 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.7 | 257 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.7 | 257 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 21.2 | 432 | 28 | 31 | 1.04 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.4 | 312 | 20 | 23 | 0.75 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.8 | 271 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 71.4 | 136 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.8 | 168 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.8 | 168 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 35.6 | 272 | 18 | 20 | 1.01 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.19 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.2 | 1,025 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.6 | 1,017 | 65 | 73 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 68.9 | 880 | 57 | 63 | 0.86 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 86.4 | 701 | 45 | 51 | 0.68 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.6 | 1,016 | 65 | 73 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 55.7 | 1,088 | 70 | 78 | 1.06 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 177.5 | 341 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.0 | 1,030 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 119.6 | 509 | 33 | 37 | 0.49 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 70.2 | 867 | 56 | 62 | 0.84 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.4 | 696 | 45 | 50 | 0.68 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 70.1 | 867 | 56 | 62 | 0.84 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 55.0 | 1,105 | 71 | 80 | 1.07 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.9 | 340 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.8 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 229.2 | 267 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.1 | 859 | 55 | 62 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.9 | 687 | 44 | 50 | 0.67 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.1 | 859 | 55 | 62 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 57.0 | 1,071 | 69 | 77 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.6 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.2 | 944 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 458.4 | 134 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 74.1 | 831 | 53 | 60 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 92.2 | 668 | 43 | 48 | 0.71 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 74.1 | 832 | 53 | 60 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 62.4 | 988 | 64 | 71 | 1.05 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.3 | 338 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.8 | 872 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 916.8 | 68 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 137.6 | 455 | 29 | 33 | 0.52 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 118.4 | 529 | 34 | 38 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 118.1 | 530 | 34 | 38 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 74.0 | 847 | 54 | 61 | 0.97 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 215.6 | 291 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.7 | 570 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 312.3 | 207 | 13 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 204.3 | 317 | 20 | 23 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 204.3 | 317 | 20 | 23 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 107.8 | 601 | 39 | 43 | 1.05 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.5 | 244 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.6 | 1,284 | 83 | 93 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 36.3 | 836 | 54 | 60 | 0.65 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 37.9 | 800 | 51 | 58 | 0.62 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 40.3 | 752 | 48 | 54 | 0.59 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 35.4 | 857 | 55 | 62 | 0.67 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.7 | 987 | 63 | 71 | 0.77 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.0 | 326 | 21 | 23 | 0.25 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.5 | 1,293 | 83 | 93 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 64.0 | 475 | 31 | 34 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 38.3 | 795 | 51 | 57 | 0.61 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 41.8 | 728 | 47 | 52 | 0.56 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 38.3 | 795 | 51 | 57 | 0.61 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.8 | 987 | 63 | 71 | 0.76 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.3 | 326 | 21 | 24 | 0.25 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.7 | 1,290 | 83 | 93 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 159.6 | 192 | 12 | 14 | 0.15 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 39.2 | 781 | 50 | 56 | 0.61 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 43.5 | 702 | 45 | 51 | 0.54 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 39.2 | 781 | 50 | 56 | 0.61 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.2 | 979 | 63 | 71 | 0.76 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.7 | 326 | 21 | 24 | 0.25 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.9 | 1,107 | 71 | 80 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 315.3 | 98 | 6 | 7 | 0.09 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 41.9 | 737 | 47 | 53 | 0.67 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.6 | 649 | 42 | 47 | 0.59 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 41.9 | 737 | 47 | 53 | 0.67 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 32.9 | 940 | 60 | 68 | 0.85 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 94.1 | 328 | 21 | 24 | 0.30 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.3 | 843 | 54 | 61 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 628.3 | 50 | 3 | 4 | 0.06 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 99.4 | 316 | 20 | 23 | 0.38 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 71.0 | 443 | 28 | 32 | 0.53 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 71.1 | 443 | 28 | 32 | 0.53 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 42.1 | 747 | 48 | 54 | 0.89 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 103.0 | 306 | 20 | 22 | 0.36 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 63.0 | 518 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 238.4 | 137 | 9 | 10 | 0.26 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 130.9 | 249 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 80.9 | 403 | 26 | 29 | 0.78 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 98.7 | 331 | 21 | 24 | 0.64 |


