# Phase 1 results: run marlin-t1-20261023-092256-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_nsys | ok | 0 |
| download_model | ok | 70 |
| install_vllm | ok | 89 |
| cuda129 | ok | 36 |
| tests | ok | 96 |
| kbench | ok | 97 |
| environment | ok | 3 |

## Environment

```
run_id: marlin-t1-20261023-092256-colab
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
197 passed in 94.48s (0:01:34)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,381 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.8 | 690 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.1 | 588 | 38 | 43 | 0.85 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 25.1 | 518 | 33 | 37 | 0.75 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 26.1 | 498 | 32 | 36 | 0.72 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.9 | 727 | 47 | 53 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 21.0 | 619 | 40 | 45 | 0.90 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.1 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 690 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 36.4 | 359 | 23 | 26 | 0.52 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 21.1 | 620 | 40 | 45 | 0.90 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 23.1 | 565 | 36 | 41 | 0.82 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 21.1 | 620 | 40 | 45 | 0.90 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 20.8 | 627 | 40 | 45 | 0.91 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.2 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.0 | 693 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 57.0 | 231 | 15 | 17 | 0.33 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 21.3 | 618 | 40 | 45 | 0.89 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.6 | 534 | 34 | 39 | 0.77 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 21.3 | 617 | 40 | 45 | 0.89 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 21.0 | 626 | 40 | 45 | 0.90 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.3 | 318 | 20 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.4 | 622 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.4 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 22.1 | 603 | 39 | 44 | 0.97 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 28.1 | 473 | 30 | 34 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 22.1 | 603 | 39 | 44 | 0.97 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 22.0 | 605 | 39 | 44 | 0.97 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.6 | 320 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.4 | 498 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.5 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 31.0 | 440 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.0 | 341 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.0 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 27.2 | 501 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 45.0 | 317 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 80.9 | 177 | 11 | 13 | 0.56 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 64.5 | 222 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 64.6 | 221 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.9 | 333 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.4 | 499 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.5 | 526 | 34 | 38 | 1.05 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 19.9 | 436 | 28 | 32 | 0.87 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 20.0 | 434 | 28 | 31 | 0.87 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 656 | 42 | 47 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 16.3 | 533 | 34 | 39 | 1.07 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.6 | 303 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 501 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 23.9 | 364 | 23 | 26 | 0.73 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.8 | 518 | 33 | 38 | 1.03 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 18.2 | 479 | 31 | 35 | 0.96 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.8 | 518 | 33 | 38 | 1.03 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.2 | 537 | 35 | 39 | 1.07 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 302 | 19 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 504 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.5 | 177 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.3 | 507 | 33 | 37 | 1.01 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.3 | 433 | 28 | 31 | 0.86 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.3 | 508 | 33 | 37 | 1.01 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.4 | 536 | 34 | 39 | 1.06 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.4 | 459 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 116.6 | 76 | 5 | 6 | 0.17 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 18.1 | 492 | 32 | 36 | 1.07 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.5 | 364 | 23 | 26 | 0.79 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 18.1 | 492 | 32 | 36 | 1.07 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 17.3 | 514 | 33 | 37 | 1.12 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.1 | 307 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.0 | 417 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 176.3 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.6 | 258 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.4 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.4 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 21.8 | 422 | 27 | 31 | 1.01 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.4 | 312 | 20 | 23 | 0.75 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.7 | 272 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 71.4 | 136 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.7 | 168 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.7 | 168 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 40.1 | 242 | 16 | 18 | 0.89 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.19 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 58.8 | 1,031 | 66 | 75 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.5 | 1,019 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 68.3 | 888 | 57 | 64 | 0.86 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 87.2 | 695 | 45 | 50 | 0.67 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.3 | 1,022 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 68.4 | 886 | 57 | 64 | 0.86 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.5 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.3 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 119.2 | 510 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 69.5 | 876 | 56 | 63 | 0.85 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.1 | 699 | 45 | 51 | 0.68 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 69.4 | 876 | 56 | 63 | 0.85 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 67.7 | 898 | 58 | 65 | 0.88 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.8 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.5 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 227.0 | 269 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 70.6 | 866 | 56 | 63 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.0 | 694 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 70.6 | 865 | 56 | 63 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 68.0 | 898 | 58 | 65 | 0.88 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.3 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.1 | 946 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.9 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 73.8 | 835 | 54 | 60 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.2 | 676 | 43 | 49 | 0.71 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 73.8 | 835 | 54 | 60 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 73.7 | 836 | 54 | 61 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.8 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.9 | 871 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.7 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 137.1 | 457 | 29 | 33 | 0.52 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 117.6 | 533 | 34 | 39 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 117.4 | 533 | 34 | 39 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 88.4 | 709 | 46 | 51 | 0.81 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 215.3 | 291 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.3 | 572 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 310.7 | 208 | 13 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 203.5 | 318 | 20 | 23 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 203.7 | 318 | 20 | 23 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 120.6 | 537 | 35 | 39 | 0.94 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 262.4 | 247 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,302 | 84 | 94 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 36.1 | 840 | 54 | 61 | 0.65 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 37.7 | 804 | 52 | 58 | 0.62 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 39.8 | 762 | 49 | 55 | 0.59 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 35.7 | 849 | 55 | 61 | 0.65 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 37.0 | 818 | 53 | 59 | 0.63 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.1 | 333 | 21 | 24 | 0.26 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.5 | 1,295 | 83 | 94 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 63.5 | 479 | 31 | 35 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 38.1 | 799 | 51 | 58 | 0.62 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 41.1 | 740 | 48 | 54 | 0.57 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 38.1 | 799 | 51 | 58 | 0.62 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 36.9 | 825 | 53 | 60 | 0.64 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.6 | 332 | 21 | 24 | 0.26 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.6 | 1,295 | 83 | 94 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 157.8 | 194 | 12 | 14 | 0.15 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 39.2 | 779 | 50 | 56 | 0.60 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 42.9 | 712 | 46 | 52 | 0.55 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 39.3 | 778 | 50 | 56 | 0.60 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 37.1 | 824 | 53 | 60 | 0.64 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.5 | 334 | 21 | 24 | 0.26 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.7 | 1,113 | 72 | 81 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 314.0 | 98 | 6 | 7 | 0.09 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.0 | 735 | 47 | 53 | 0.66 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.8 | 646 | 42 | 47 | 0.58 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.0 | 735 | 47 | 53 | 0.66 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 40.2 | 768 | 49 | 56 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.9 | 336 | 22 | 24 | 0.30 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.0 | 850 | 55 | 62 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 613.6 | 51 | 3 | 4 | 0.06 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 98.7 | 319 | 21 | 23 | 0.38 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 70.8 | 444 | 29 | 32 | 0.52 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 70.9 | 444 | 29 | 32 | 0.52 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 48.9 | 644 | 41 | 47 | 0.76 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.3 | 317 | 20 | 23 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 62.7 | 520 | 33 | 38 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.2 | 138 | 9 | 10 | 0.26 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 130.7 | 250 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 78.8 | 414 | 27 | 30 | 0.80 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 98.8 | 330 | 21 | 24 | 0.63 |

### w4a16_dequant

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj N=6144 K=4096 | rows | ok (0.0e+00) | 52.5 | 1,206 | 78 | 87 | 1.00 |
| qkv_proj N=6144 K=4096 | tiled | ok (0.0e+00) | 52.5 | 1,205 | 78 | 87 | 1.00 |
| o_proj N=4096 K=4096 | rows | ok (0.0e+00) | 35.5 | 1,187 | 76 | 86 | 1.00 |
| o_proj N=4096 K=4096 | tiled | ok (0.0e+00) | 34.3 | 1,230 | 79 | 89 | 1.04 |
| gate_up_proj N=28672 K=4096 | rows | ok (0.0e+00) | 238.8 | 1,237 | 80 | 90 | 1.00 |
| gate_up_proj N=28672 K=4096 | tiled | ok (0.0e+00) | 241.4 | 1,224 | 79 | 89 | 0.99 |
| down_proj N=4096 K=14336 | rows | ok (0.0e+00) | 120.3 | 1,228 | 79 | 89 | 1.00 |
| down_proj N=4096 K=14336 | tiled | ok (0.0e+00) | 120.9 | 1,222 | 79 | 88 | 1.00 |


