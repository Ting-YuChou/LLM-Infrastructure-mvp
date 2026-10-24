# Phase 1 results: run down-d2a-20261023-235500-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 101 |
| kbench | ok | 268 |
| ncu | ok | 106 |
| trace | ok | 19 |
| golden | ok | 4 |
| environment | ok | 2 |

## Environment

```
run_id: down-d2a-20261023-235500-colab
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
213 passed in 99.56s (0:01:39)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,385 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.9 | 688 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.0 | 591 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.2 | 643 | 41 | 46 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.3 | 609 | 39 | 44 | 0.89 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 730 | 47 | 53 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.3 | 798 | 51 | 58 | 1.16 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 690 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 36.3 | 360 | 23 | 26 | 0.52 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.3 | 643 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.7 | 576 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.3 | 799 | 51 | 58 | 1.16 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 694 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 56.7 | 232 | 15 | 17 | 0.33 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 640 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.3 | 541 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.2 | 810 | 52 | 58 | 1.17 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 320 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.3 | 626 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.3 | 119 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.7 | 479 | 31 | 35 | 0.77 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.2 | 821 | 53 | 59 | 1.31 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.3 | 500 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.3 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.2 | 339 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.2 | 339 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.6 | 694 | 45 | 50 | 1.39 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.6 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.7 | 224 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 39.3 | 363 | 23 | 26 | 1.14 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.4 | 498 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.4 | 528 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 556 | 36 | 40 | 1.12 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 532 | 34 | 38 | 1.07 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 657 | 42 | 47 | 1.32 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 15.2 | 571 | 37 | 41 | 1.15 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 501 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 22.3 | 391 | 25 | 28 | 0.78 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 489 | 31 | 35 | 0.98 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 554 | 36 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.1 | 579 | 37 | 42 | 1.16 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 304 | 20 | 22 | 0.61 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.4 | 504 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 50.0 | 176 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.1 | 437 | 28 | 32 | 0.87 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.0 | 586 | 38 | 42 | 1.16 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.4 | 460 | 30 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.6 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.4 | 365 | 23 | 26 | 0.79 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 511 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.1 | 589 | 38 | 43 | 1.28 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.8 | 420 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 176.0 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 19.0 | 483 | 31 | 35 | 1.15 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.4 | 312 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.5 | 273 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.2 | 138 | 9 | 10 | 0.51 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 32.5 | 299 | 19 | 22 | 1.09 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 324 | 21 | 23 | 1.18 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 58.9 | 1,029 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.4 | 1,020 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 927 | 60 | 67 | 0.90 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 85.3 | 711 | 46 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.4 | 1,021 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 52.5 | 1,156 | 74 | 83 | 1.12 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.1 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.7 | 1,019 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.9 | 512 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.4 | 916 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.0 | 699 | 45 | 50 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.4 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.4 | 1,138 | 73 | 82 | 1.12 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.4 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.2 | 1,014 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 227.0 | 269 | 17 | 19 | 0.27 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.3 | 908 | 58 | 66 | 0.90 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.2 | 693 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.2 | 908 | 58 | 66 | 0.90 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.4 | 1,122 | 72 | 81 | 1.11 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.5 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 63.8 | 966 | 62 | 70 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 454.1 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.6 | 860 | 55 | 62 | 0.89 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 90.6 | 680 | 44 | 49 | 0.70 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.7 | 859 | 55 | 62 | 0.89 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 56.1 | 1,099 | 71 | 79 | 1.14 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.2 | 336 | 22 | 24 | 0.35 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.9 | 872 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.8 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.6 | 462 | 30 | 33 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 65.7 | 953 | 61 | 69 | 1.09 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.4 | 292 | 19 | 21 | 0.34 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 91.3 | 709 | 46 | 51 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.3 | 210 | 14 | 15 | 0.30 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.7 | 328 | 21 | 24 | 0.46 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.7 | 327 | 21 | 24 | 0.46 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 104.1 | 622 | 40 | 45 | 0.88 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 264.4 | 245 | 16 | 18 | 0.35 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 903 | 58 | 65 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 47.7 | 636 | 41 | 46 | 0.70 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.2 | 735 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.5 | 667 | 43 | 48 | 0.74 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.3 | 792 | 51 | 57 | 0.88 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.6 | 990 | 64 | 71 | 1.10 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.9 | 330 | 21 | 24 | 0.37 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 908 | 58 | 66 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 65.9 | 462 | 30 | 33 | 0.51 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 727 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 46.8 | 650 | 42 | 47 | 0.72 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.8 | 987 | 63 | 71 | 1.09 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.0 | 331 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.8 | 904 | 58 | 65 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.4 | 173 | 11 | 13 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.9 | 712 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.4 | 631 | 41 | 46 | 0.70 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.9 | 713 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.1 | 982 | 63 | 71 | 1.09 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.3 | 331 | 21 | 24 | 0.37 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.0 | 834 | 54 | 60 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 340.9 | 91 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.1 | 581 | 37 | 42 | 0.70 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 32.1 | 962 | 62 | 69 | 1.15 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.8 | 333 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.3 | 652 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 663.4 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.8 | 300 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.67 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 38.2 | 823 | 53 | 59 | 1.26 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.6 | 313 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 75.6 | 432 | 28 | 31 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.5 | 137 | 9 | 10 | 0.32 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.58 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.2 | 249 | 16 | 18 | 0.58 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 72.9 | 448 | 29 | 32 | 1.04 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.3 | 329 | 21 | 24 | 0.76 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 16.38 | 0.00 | 3.07 | 1.12 / 1.21 | 0.00 / 2.03 (2.54) | 3.07 | 0.00 + 4.36 + 8.86 + 2.69 + 0.08 |
| qkv_proj M=8 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.81 | 1.45 / 1.46 | 0.00 / 2.64 (3.30) | 3.07 | 0.00 + 3.95 + 11.53 + 3.33 + 0.10 |
| qkv_proj M=16 | 256x128, 96 × 8 | 19.46 | 0.00 | 2.46 | 1.46 / 1.75 | 0.00 / 2.86 (3.63) | 3.07 | 0.00 + 3.52 + 11.62 + 3.68 + 0.10 |
| qkv_proj M=64 | 64x128, 106 × 29 | 53.25 | 0.00 | 1.66 | 1.17 / 0.94 | 4.93 / 5.40 (10.29) | 12.29 | 0.00 + 1.82 + 33.91 + 16.53 + 0.11 |
| o_proj M=1 | 256x128, 103 × 5 | 14.34 | 0.00 | 3.37 | 1.11 / 1.30 | 1.68 / 1.68 (2.62) | 5.12 | 0.00 + 4.04 + 5.92 + 4.36 + 0.08 |
| o_proj M=8 | 256x128, 103 × 5 | 17.41 | 0.00 | 3.04 | 1.40 / 1.57 | 2.22 / 2.22 (3.49) | 6.14 | 0.00 + 4.30 + 6.92 + 5.74 + 0.10 |
| o_proj M=16 | 256x128, 103 × 5 | 16.38 | 0.00 | 2.55 | 1.28 / 1.87 | 2.05 / 2.05 (3.74) | 6.14 | 0.00 + 3.59 + 6.32 + 5.82 + 0.09 |
| o_proj M=64 | 64x128, 108 × 19 | 40.96 | 1.02 | 1.62 | 1.13 / 0.96 | 4.72 / 5.50 (11.61) | 13.31 | 0.00 + 1.62 + 21.62 + 17.51 + 0.10 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 51.20 | 0.00 | 3.48 | 1.15 / 1.34 | 1.03 / 2.61 (2.64) | 5.12 | 0.00 + 3.63 + 41.20 + 5.59 + 0.07 |
| gate_up_proj M=8 | 256x128, 106 × 34 | 62.46 | 0.00 | 3.25 | 1.47 / 1.61 | 1.33 / 3.38 (3.41) | 5.12 | 0.00 + 4.10 + 50.69 + 7.13 + 0.10 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 63.49 | 0.00 | 2.87 | 1.49 / 1.93 | 1.41 / 3.75 (3.78) | 5.12 | 0.00 + 4.26 + 50.74 + 7.60 + 0.10 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 135.17 | 0.00 | 1.99 | 1.65 / 1.27 | 5.02 / 15.59 (15.60) | 14.34 | 0.00 + 2.20 + 110.60 + 21.88 + 0.10 |
| down_proj M=1 | 256x128, 106 × 17 | 30.72 | 1.02 | 3.54 | 1.22 / 1.34 | 1.86 / 1.84 (2.92) | 5.12 | 0.00 + 4.26 + 20.68 + 5.15 + 0.08 |
| down_proj M=8 | 256x128, 106 × 17 | 35.84 | 0.00 | 3.13 | 1.47 / 1.61 | 2.25 / 2.18 (3.52) | 6.14 | 0.00 + 3.80 + 25.05 + 6.37 + 0.10 |
| down_proj M=16 | 256x128, 106 × 17 | 36.86 | 0.00 | 2.76 | 1.49 / 1.93 | 2.32 / 2.29 (4.28) | 7.17 | 0.00 + 3.45 + 25.41 + 7.37 + 0.10 |
| down_proj M=64 | 64x128, 107 × 67 | 100.35 | 0.00 | 1.67 | 1.18 / 0.95 | 4.94 / 5.38 (11.57) | 14.34 | 0.00 + 1.63 + 79.23 + 18.80 + 0.17 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | SM GHz (runs) | globaltimer tick |
|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 | 0.50 | 1.11 | 0.67 | 1.33–1.40 | 1024 ns |
| qkv_proj M=8 | 0.63 | 1.48 | 0.88 | 1.05–1.10 | 1024 ns |
| qkv_proj M=16 | 0.63 | 1.59 | 1.12 | 1.05–1.08 | 1024 ns |
| qkv_proj M=64 | 1.67 | 2.91 | 5.42 | 1.08–1.41 | 1024 ns |
| o_proj M=1 | 0.47 | 1.03 | 0.85 | 1.34–1.43 | 1024 ns |
| o_proj M=8 | 0.60 | 1.39 | 1.18 | 1.04–1.11 | 1024 ns |
| o_proj M=16 | 0.53 | 1.31 | 1.58 | 1.17–1.24 | 1024 ns |
| o_proj M=64 | 1.67 | 3.04 | 6.91 | 1.10–1.24 | 1024 ns |
| gate_up_proj M=1 | 0.45 | 1.21 | 0.88 | 1.39–1.41 | 1024 ns |
| gate_up_proj M=8 | 0.57 | 1.46 | 1.17 | 1.08–1.10 | 1024 ns |
| gate_up_proj M=16 | 0.57 | 1.57 | 1.38 | 1.08–1.09 | 1024 ns |
| gate_up_proj M=64 | 0.57 | 4.44 | 9.41 | 1.09–1.10 | 1024 ns |
| down_proj M=1 | 0.51 | 1.14 | 0.94 | 1.24–1.28 | 1024 ns |
| down_proj M=8 | 0.58 | 1.40 | 1.19 | 1.07–1.09 | 1024 ns |
| down_proj M=16 | 0.60 | 1.48 | 1.83 | 1.06–1.09 | 1024 ns |
| down_proj M=64 | 1.68 | 3.03 | 6.80 | 1.08–1.09 | 1024 ns |

## Kernels under ncu

### ncu kbench_w4a16_linear: 51 kernels, 3033 us, 444 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 4>` | gemm | 6 | 27 | 134.3 | 17 | 265 | 19 | 12/12 | 191 | 75 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 4, 2>` | gemm | 3 | 24 | 239.5 | 9 | 144 | 16 | 12/12 | 188 | 84 | long_scoreboard, wait |
| `void <unnamed>::gemv_w4a16_kernel<__nv_bfloat16, 1, 2>` | gemm | 12 | 12 | 29.7 | 48 | 762 | 52 | 31/62 | 48 | 11 | long_scoreboard, not_selected |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 4, 64` | gemm | 3 | 7 | 75.6 | 30 | 462 | 32 | 13/12 | 255 | 60 | wait, selected |
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 1>` | gemm | 6 | 7 | 35.3 | 41 | 647 | 26 | 13/25 | 124 | 17 | long_scoreboard, wait |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 3 | 7 | 68.1 | 33 | 518 | 36 | 6/6 | 255 | 72 | wait, selected |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 1, 4>` | gemm | 6 | 7 | 33.1 | 43 | 687 | 27 | 13/19 | 130 | 11 | long_scoreboard, wait |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 6 | 5 | 26.1 | 54 | 869 | 26 | 12/12 | 128 | 11 | long_scoreboard, selected |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 1, 25` | gemm | 6 | 5 | 25.7 | 55 | 882 | 30 | 12/12 | 193 | 8 | wait, selected |

