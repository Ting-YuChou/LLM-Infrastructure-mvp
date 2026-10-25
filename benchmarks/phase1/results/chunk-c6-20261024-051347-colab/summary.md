# Phase 1 results: run chunk-c6-20261024-051347-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 120 |
| kbench | ok | 354 |
| trace | ok | 20 |
| golden | ok | 3 |
| environment | ok | 2 |

## Environment

```
run_id: chunk-c6-20261024-051347-colab
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
276 passed in 118.69s (0:01:58)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,380 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.8 | 692 | 45 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 21.9 | 593 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.2 | 643 | 41 | 47 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.4 | 609 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 730 | 47 | 53 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 15.8 | 822 | 53 | 60 | 1.19 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.8 | 695 | 45 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 45.6 | 286 | 18 | 21 | 0.41 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.3 | 642 | 41 | 47 | 0.92 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.7 | 576 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 642 | 41 | 47 | 0.92 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.1 | 814 | 52 | 59 | 1.17 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.8 | 698 | 45 | 51 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 56.7 | 232 | 15 | 17 | 0.33 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.2 | 542 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.9 | 824 | 53 | 60 | 1.18 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.1 | 630 | 41 | 46 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.2 | 120 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.8 | 479 | 31 | 35 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 626 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.8 | 842 | 54 | 61 | 1.34 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.4 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.2 | 501 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.2 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 442 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.2 | 339 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.2 | 339 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.8 | 690 | 44 | 50 | 1.38 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.1 | 324 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.8 | 319 | 21 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.4 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 224 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 31.8 | 449 | 29 | 33 | 1.41 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| qkv_proj M=128 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 62.1 | 251 | 16 | 18 | 1.00 |
| qkv_proj M=128 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 45.5 | 343 | 22 | 25 | 1.36 |
| qkv_proj M=128 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 55.9 | 279 | 18 | 20 | 1.11 |
| qkv_proj M=256 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 99.5 | 183 | 12 | 13 | 1.00 |
| qkv_proj M=256 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 74.1 | 246 | 16 | 18 | 1.34 |
| qkv_proj M=256 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 61.3 | 297 | 19 | 22 | 1.62 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.2 | 503 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 13.2 | 657 | 42 | 48 | 1.30 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 557 | 36 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 532 | 34 | 39 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 657 | 42 | 48 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 14.6 | 595 | 38 | 43 | 1.18 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.60 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.3 | 504 | 32 | 37 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 22.3 | 391 | 25 | 28 | 0.78 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.8 | 553 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 490 | 31 | 35 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.6 | 596 | 38 | 43 | 1.18 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 304 | 20 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.2 | 510 | 33 | 37 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.3 | 178 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 540 | 35 | 39 | 1.06 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.1 | 437 | 28 | 32 | 0.86 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.3 | 540 | 35 | 39 | 1.06 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.6 | 600 | 39 | 44 | 1.18 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.2 | 463 | 30 | 34 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.6 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.5 | 365 | 23 | 26 | 0.79 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.8 | 603 | 39 | 44 | 1.30 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 308 | 20 | 22 | 0.66 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.7 | 423 | 27 | 31 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.9 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.4 | 259 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 18.6 | 492 | 32 | 36 | 1.16 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.3 | 275 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.2 | 138 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 27.0 | 360 | 23 | 26 | 1.31 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.18 |
| o_proj M=128 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 52.6 | 204 | 13 | 15 | 1.00 |
| o_proj M=128 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 35.6 | 302 | 19 | 22 | 1.48 |
| o_proj M=128 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.3 | 260 | 17 | 19 | 1.27 |
| o_proj M=256 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 71.4 | 180 | 12 | 13 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 55.6 | 231 | 15 | 17 | 1.28 |
| o_proj M=256 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 54.2 | 237 | 15 | 17 | 1.32 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.3 | 1,023 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.4 | 1,020 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 927 | 60 | 67 | 0.91 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 85.6 | 709 | 46 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.4 | 1,020 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 51.9 | 1,169 | 75 | 85 | 1.14 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.1 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.7 | 1,019 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.7 | 513 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.4 | 916 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.7 | 701 | 45 | 51 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 52.9 | 1,150 | 74 | 83 | 1.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.9 | 338 | 22 | 25 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.2 | 1,015 | 65 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 226.8 | 269 | 17 | 20 | 0.27 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.3 | 908 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.9 | 695 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.3 | 908 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.6 | 1,140 | 73 | 83 | 1.12 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.3 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.4 | 942 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.7 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.7 | 859 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.3 | 675 | 43 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.7 | 859 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.8 | 1,125 | 72 | 82 | 1.19 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.2 | 336 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 66.8 | 937 | 60 | 68 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.0 | 69 | 4 | 5 | 0.07 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.6 | 462 | 30 | 33 | 0.49 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.5 | 552 | 36 | 40 | 0.59 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.59 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 60.3 | 1,039 | 67 | 75 | 1.11 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 213.4 | 294 | 19 | 21 | 0.31 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.2 | 572 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.3 | 210 | 14 | 15 | 0.37 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.3 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.2 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 87.3 | 741 | 48 | 54 | 1.30 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.1 | 244 | 16 | 18 | 0.43 |
| gate_up_proj M=128 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 168.7 | 409 | 26 | 30 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 159.3 | 433 | 28 | 31 | 1.06 |
| gate_up_proj M=128 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 270.8 | 255 | 16 | 18 | 0.62 |
| gate_up_proj M=256 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 325.5 | 238 | 15 | 17 | 1.00 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 300.5 | 257 | 17 | 19 | 1.08 |
| gate_up_proj M=256 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 320.2 | 241 | 16 | 18 | 1.02 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 903 | 58 | 65 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 38.9 | 779 | 50 | 56 | 0.86 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.2 | 736 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.6 | 665 | 43 | 48 | 0.74 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.2 | 793 | 51 | 57 | 0.88 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.4 | 998 | 64 | 72 | 1.10 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.7 | 331 | 21 | 24 | 0.37 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 906 | 58 | 66 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 73.8 | 412 | 27 | 30 | 0.45 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.7 | 729 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 46.9 | 649 | 42 | 47 | 0.72 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.7 | 729 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.8 | 988 | 64 | 72 | 1.09 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.0 | 331 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.8 | 903 | 58 | 65 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.2 | 173 | 11 | 13 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.8 | 714 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.5 | 631 | 41 | 46 | 0.70 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.8 | 714 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.1 | 984 | 63 | 71 | 1.09 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.3 | 331 | 21 | 24 | 0.37 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.9 | 838 | 54 | 61 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 341.2 | 90 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.1 | 685 | 44 | 50 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.0 | 582 | 37 | 42 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.1 | 685 | 44 | 50 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.8 | 969 | 62 | 70 | 1.16 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.4 | 334 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.1 | 654 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 662.6 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.6 | 301 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 36.3 | 867 | 56 | 63 | 1.33 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.3 | 314 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.3 | 401 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.4 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.1 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.1 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 54.2 | 603 | 39 | 44 | 1.50 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.2 | 329 | 21 | 24 | 0.82 |
| down_proj M=128 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 121.5 | 288 | 19 | 21 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 85.8 | 408 | 26 | 30 | 1.42 |
| down_proj M=128 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 111.2 | 315 | 20 | 23 | 1.09 |
| down_proj M=256 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 187.7 | 212 | 14 | 15 | 1.00 |
| down_proj M=256 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 150.6 | 264 | 17 | 19 | 1.25 |
| down_proj M=256 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 144.5 | 275 | 18 | 20 | 1.30 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 17.41 | 0.00 | 3.78 | 1.13 / 1.74 | 0.00 / 2.05 (2.65) | 3.07 | 0.00 + 4.64 + 9.01 + 2.91 + 0.08 |
| qkv_proj M=16 | 256x128, 96 × 8 | 18.43 | 0.00 | 2.74 | 1.32 / 1.74 | 0.00 / 2.57 (3.42) | 4.10 | 0.00 + 3.27 + 10.94 + 3.96 + 0.09 |
| qkv_proj M=64 | 128x128, 103 × 15 | 41.98 | 0.00 | 1.83 | 1.51 / 1.85 | 7.42 / 5.77 (7.98) | 12.29 | 0.00 + 2.49 + 22.52 + 16.66 + 0.09 |
| qkv_proj M=128 | 128x128, 106 × 29 | 54.27 | 0.00 | 1.53 | 1.30 / 0.95 | 6.51 / 6.09 (6.12) | 10.24 | 0.00 + 1.34 + 38.20 + 13.78 + 0.08 |
| qkv_proj M=256 | 64x256, 108 × 57 | 80.90 | 0.00 | 1.32 | 1.13 / 0.32 | 4.65 / 7.40 (7.50) | 11.26 | 0.00 + 1.39 + 64.16 + 14.65 + 0.06 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 50.18 | 0.00 | 4.09 | 1.10 / 1.93 | 1.00 / 1.71 (1.73) | 7.17 | 0.00 + 4.66 + 39.43 + 6.23 + 0.07 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 55.30 | 0.00 | 2.75 | 1.28 / 1.93 | 1.36 / 2.34 (2.36) | 6.14 | 0.00 + 3.43 + 45.01 + 7.07 + 0.09 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 119.81 | 0.00 | 1.81 | 1.45 / 1.27 | 5.96 / 11.18 (11.20) | 12.29 | 0.00 + 2.01 + 97.82 + 19.41 + 0.07 |
| gate_up_proj M=128 | 64x256, 108 × 133 | 221.18 | 0.00 | 1.77 | 1.45 / 0.64 | 4.43 / 11.62 (11.63) | 10.24 | 0.00 + 1.74 + 192.98 + 25.89 + 0.07 |
| gate_up_proj M=256 | 64x256, 108 × 266 | 343.04 | 0.00 | 1.47 | 1.00 / 0.32 | 3.54 / 2.92 (0.00) | 6.14 | 0.00 + 1.58 + 232.64 + 15.44 + 93.45 |
| down_proj M=1 | 256x128, 106 × 17 | 29.70 | 0.00 | 4.08 | 1.09 / 1.93 | 3.07 / 1.82 (1.84) | 7.17 | 0.00 + 4.58 + 18.89 + 5.54 + 0.07 |
| down_proj M=16 | 256x128, 106 × 17 | 33.79 | 0.00 | 2.66 | 1.32 / 1.93 | 3.21 / 2.61 (3.03) | 7.17 | 0.00 + 3.45 + 22.20 + 7.87 + 0.09 |
| down_proj M=64 | 128x128, 106 × 34 | 71.68 | 0.00 | 1.80 | 1.47 / 1.91 | 7.41 / 5.73 (8.75) | 16.38 | 0.00 + 1.78 + 50.48 + 19.39 + 0.10 |
| down_proj M=128 | 128x128, 108 × 67 | 116.74 | 0.00 | 1.72 | 1.46 / 0.97 | 7.37 / 6.68 (8.02) | 14.34 | 0.00 + 1.80 + 98.17 + 16.08 + 0.09 |
| down_proj M=256 | 64x256, 108 × 133 | 212.99 | 0.00 | 1.60 | 1.44 / 0.32 | 5.98 / 9.05 (11.74) | 18.43 | 0.00 + 1.72 + 192.52 + 18.79 + 0.07 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | last, saw it early | SM GHz (runs) | globaltimer tick |
|---|---:|---:|---:|---:|---:|---:|
| qkv_proj M=1 | 0.65 | 1.28 | 0.76 | 40% | 1.16–1.21 | 1024 ns |
| qkv_proj M=16 | 0.71 | 1.63 | 1.04 | 23% | 1.06–1.10 | 1024 ns |
| qkv_proj M=64 | 2.33 | 2.42 | 4.91 | 92% | 1.08–1.10 | 1024 ns |
| qkv_proj M=128 | 2.10 | 1.98 | 3.50 | 100% | 1.23–1.24 | 1024 ns |
| qkv_proj M=256 | 0.83 | 2.36 | 5.96 | 100% | 1.40–1.41 | 1024 ns |
| gate_up_proj M=1 | 0.48 | 0.69 | 0.86 | 100% | 1.39–1.42 | 1024 ns |
| gate_up_proj M=16 | 0.62 | 0.97 | 1.25 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=64 | 2.31 | 2.03 | 9.20 | 100% | 1.09–1.10 | 1024 ns |
| gate_up_proj M=128 | 2.32 | 2.02 | 9.61 | 100% | 1.09–1.10 | 1024 ns |
| gate_up_proj M=256 | 1.90 | 2.58 | 0.00 | 0% | 1.35–1.35 | 1024 ns |
| down_proj M=1 | 0.56 | 1.10 | 0.86 | 66% | 1.38–1.42 | 1024 ns |
| down_proj M=16 | 0.72 | 1.59 | 1.70 | 69% | 1.07–1.10 | 1024 ns |
| down_proj M=64 | 2.29 | 2.76 | 5.73 | 94% | 1.08–1.10 | 1024 ns |
| down_proj M=128 | 2.33 | 2.32 | 5.06 | 100% | 1.09–1.10 | 1024 ns |
| down_proj M=256 | 1.04 | 3.17 | 9.71 | 91% | 1.09–1.10 | 1024 ns |

