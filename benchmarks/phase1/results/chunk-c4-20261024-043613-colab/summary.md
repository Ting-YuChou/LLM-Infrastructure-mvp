# Phase 1 results: run chunk-c4-20261024-043613-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 208 |
| kbench | ok | 355 |
| ncu | ok | 85 |
| trace | ok | 19 |
| golden | ok | 4 |
| e2e | ok | 901 |
| environment | ok | 2 |

## Environment

```
run_id: chunk-c4-20261024-043613-colab
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
711 passed, 1 warning in 205.90s (0:03:25)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,378 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.8 | 693 | 45 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 21.9 | 593 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.2 | 642 | 41 | 47 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.4 | 609 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 730 | 47 | 53 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 15.9 | 819 | 53 | 59 | 1.18 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.8 | 693 | 45 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 38.5 | 339 | 22 | 25 | 0.49 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.3 | 643 | 41 | 47 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.6 | 577 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 643 | 41 | 47 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.9 | 820 | 53 | 59 | 1.18 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 318 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.9 | 697 | 45 | 51 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 60.3 | 218 | 14 | 16 | 0.31 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 642 | 41 | 47 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.3 | 542 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 47 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.9 | 824 | 53 | 60 | 1.18 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.2 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.1 | 629 | 40 | 46 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.2 | 120 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 626 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.8 | 479 | 31 | 35 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 626 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.7 | 846 | 54 | 61 | 1.34 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.2 | 501 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.2 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.7 | 692 | 44 | 50 | 1.38 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.64 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.7 | 319 | 21 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.5 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.8 | 224 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.8 | 224 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 33.7 | 423 | 27 | 31 | 1.33 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| qkv_proj M=128 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 62.1 | 251 | 16 | 18 | 1.00 |
| qkv_proj M=128 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 47.9 | 325 | 21 | 24 | 1.30 |
| qkv_proj M=128 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 56.3 | 277 | 18 | 20 | 1.10 |
| qkv_proj M=256 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 99.5 | 183 | 12 | 13 | 1.00 |
| qkv_proj M=256 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 75.7 | 241 | 15 | 17 | 1.31 |
| qkv_proj M=256 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 61.1 | 298 | 19 | 22 | 1.63 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.3 | 500 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.3 | 532 | 34 | 39 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 557 | 36 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 532 | 34 | 39 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.1 | 660 | 42 | 48 | 1.32 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 14.5 | 599 | 39 | 43 | 1.20 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.2 | 508 | 33 | 37 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.8 | 314 | 20 | 23 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 556 | 36 | 40 | 1.09 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.9 | 488 | 31 | 35 | 0.96 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.09 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.5 | 600 | 39 | 44 | 1.18 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 304 | 20 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.3 | 508 | 33 | 37 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.4 | 178 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 539 | 35 | 39 | 1.06 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.2 | 434 | 28 | 31 | 0.85 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.2 | 543 | 35 | 39 | 1.07 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.5 | 606 | 39 | 44 | 1.19 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.2 | 464 | 30 | 34 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.5 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.10 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.4 | 365 | 24 | 27 | 0.79 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 513 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.8 | 602 | 39 | 44 | 1.30 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 308 | 20 | 22 | 0.66 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.7 | 423 | 27 | 31 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.9 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.4 | 259 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 257 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 257 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 18.6 | 493 | 32 | 36 | 1.16 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.4 | 312 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.4 | 274 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.4 | 138 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 29.0 | 335 | 22 | 24 | 1.22 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 324 | 21 | 23 | 1.18 |
| o_proj M=128 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 52.6 | 204 | 13 | 15 | 1.00 |
| o_proj M=128 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 37.6 | 286 | 18 | 21 | 1.40 |
| o_proj M=128 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.3 | 260 | 17 | 19 | 1.27 |
| o_proj M=256 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 71.4 | 180 | 12 | 13 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 56.7 | 227 | 15 | 16 | 1.26 |
| o_proj M=256 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 54.2 | 237 | 15 | 17 | 1.32 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.2 | 1,023 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.3 | 1,022 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 927 | 60 | 67 | 0.91 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 85.7 | 708 | 46 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.4 | 1,020 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 51.7 | 1,172 | 75 | 85 | 1.15 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.1 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.6 | 1,021 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.6 | 513 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.2 | 919 | 59 | 67 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.5 | 703 | 45 | 51 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.2 | 919 | 59 | 67 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 52.7 | 1,155 | 74 | 84 | 1.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.6 | 339 | 22 | 25 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.0 | 1,017 | 65 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 226.8 | 269 | 17 | 20 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.2 | 909 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.9 | 695 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.2 | 909 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.5 | 1,141 | 73 | 83 | 1.12 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.0 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.2 | 945 | 61 | 69 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.5 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.6 | 861 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 90.7 | 679 | 44 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.6 | 860 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.6 | 1,129 | 73 | 82 | 1.19 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.0 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.6 | 875 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 906.9 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.5 | 462 | 30 | 34 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.3 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.3 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 60.8 | 1,031 | 66 | 75 | 1.18 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.6 | 292 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.1 | 573 | 37 | 42 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.4 | 210 | 14 | 15 | 0.37 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.4 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.5 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 86.5 | 749 | 48 | 54 | 1.31 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 264.7 | 245 | 16 | 18 | 0.43 |
| gate_up_proj M=128 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 169.9 | 406 | 26 | 29 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 157.6 | 437 | 28 | 32 | 1.08 |
| gate_up_proj M=128 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 271.7 | 254 | 16 | 18 | 0.63 |
| gate_up_proj M=256 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 325.3 | 238 | 15 | 17 | 1.00 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 297.4 | 260 | 17 | 19 | 1.09 |
| gate_up_proj M=256 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 320.0 | 242 | 16 | 18 | 1.02 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 905 | 58 | 66 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 43.5 | 697 | 45 | 51 | 0.77 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.2 | 736 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.6 | 665 | 43 | 48 | 0.74 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.2 | 793 | 51 | 58 | 0.88 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.3 | 999 | 64 | 72 | 1.10 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.9 | 330 | 21 | 24 | 0.36 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 908 | 58 | 66 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 65.7 | 463 | 30 | 34 | 0.51 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 46.9 | 649 | 42 | 47 | 0.71 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.7 | 991 | 64 | 72 | 1.09 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.0 | 331 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.7 | 906 | 58 | 66 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.6 | 173 | 11 | 13 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.8 | 714 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.5 | 630 | 41 | 46 | 0.70 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.9 | 713 | 46 | 52 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.0 | 988 | 64 | 72 | 1.09 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.4 | 331 | 21 | 24 | 0.37 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.8 | 840 | 54 | 61 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 342.1 | 90 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.1 | 684 | 44 | 50 | 0.81 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.3 | 579 | 37 | 42 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.1 | 684 | 44 | 50 | 0.81 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.7 | 974 | 63 | 71 | 1.16 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.7 | 333 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.1 | 653 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 663.6 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.7 | 301 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 36.8 | 856 | 55 | 62 | 1.31 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.6 | 313 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.3 | 401 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.6 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.6 | 248 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.6 | 248 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 55.7 | 586 | 38 | 43 | 1.46 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.2 | 329 | 21 | 24 | 0.82 |
| down_proj M=128 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 121.6 | 288 | 19 | 21 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 88.9 | 394 | 25 | 29 | 1.37 |
| down_proj M=128 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 111.7 | 313 | 20 | 23 | 1.09 |
| down_proj M=256 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 166.7 | 238 | 15 | 17 | 1.00 |
| down_proj M=256 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 153.0 | 260 | 17 | 19 | 1.09 |
| down_proj M=256 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 144.6 | 275 | 18 | 20 | 1.15 |


## gemm_w4a16_tiled trace

| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qkv_proj M=1 | 256x128, 96 × 8 | 16.38 | 0.00 | 4.01 | 1.09 / 1.74 | 0.00 / 2.06 (2.65) | 3.07 | 0.00 + 4.58 + 8.75 + 2.96 + 0.09 |
| qkv_proj M=16 | 256x128, 96 × 8 | 18.43 | 0.00 | 2.86 | 1.25 / 1.74 | 0.00 / 2.50 (3.41) | 4.10 | 0.00 + 4.01 + 9.92 + 3.90 + 0.09 |
| qkv_proj M=64 | 128x128, 103 × 15 | 43.01 | 0.00 | 2.05 | 1.49 / 1.85 | 7.35 / 7.69 (10.29) | 12.29 | 0.00 + 2.67 + 22.22 + 18.04 + 0.10 |
| qkv_proj M=128 | 128x128, 106 × 29 | 53.25 | 0.00 | 1.61 | 1.24 / 0.95 | 6.13 / 7.33 (7.36) | 10.24 | 0.00 + 1.68 + 35.96 + 15.29 + 0.08 |
| qkv_proj M=256 | 64x256, 108 × 57 | 75.78 | 0.00 | 1.30 | 1.01 / 0.32 | 4.61 / 7.63 (7.74) | 13.31 | 0.00 + 1.27 + 57.58 + 17.16 + 0.07 |
| gate_up_proj M=1 | 256x128, 106 × 34 | 50.18 | 0.00 | 4.24 | 1.09 / 1.92 | 0.97 / 1.73 (1.74) | 8.19 | 0.00 + 4.86 + 38.88 + 6.39 + 0.07 |
| gate_up_proj M=16 | 256x128, 106 × 34 | 55.30 | 1.02 | 2.88 | 1.25 / 1.92 | 1.32 / 2.37 (2.40) | 6.14 | 0.00 + 3.16 + 44.13 + 7.18 + 0.09 |
| gate_up_proj M=64 | 64x256, 107 × 67 | 109.57 | 0.00 | 1.77 | 1.30 / 1.27 | 5.93 / 11.54 (11.56) | 12.29 | 0.00 + 1.75 + 87.59 + 19.70 + 0.10 |
| gate_up_proj M=128 | 64x256, 108 × 133 | 200.70 | 0.00 | 1.72 | 1.29 / 0.64 | 4.43 / 12.02 (12.02) | 9.22 | 0.00 + 1.79 + 171.85 + 26.32 + 0.10 |
| gate_up_proj M=256 | 64x256, 108 × 266 | 307.20 | 0.00 | 1.38 | 0.88 / 0.32 | 3.51 / 3.48 (0.00) | 6.14 | 0.00 + 1.46 + 201.91 + 15.30 + 87.31 |
| down_proj M=1 | 256x128, 106 × 17 | 29.70 | 0.00 | 4.19 | 1.08 / 1.92 | 3.02 / 1.77 (1.87) | 7.17 | 0.00 + 4.85 + 18.65 + 5.51 + 0.07 |
| down_proj M=16 | 256x128, 106 × 17 | 32.77 | 0.00 | 2.76 | 1.28 / 1.92 | 3.15 / 2.49 (3.08) | 8.19 | 0.00 + 3.30 + 21.81 + 7.79 + 0.09 |
| down_proj M=64 | 128x128, 106 × 34 | 72.70 | 0.00 | 1.99 | 1.47 / 1.90 | 7.34 / 7.65 (10.99) | 14.34 | 0.00 + 2.03 + 50.13 + 19.94 + 0.10 |
| down_proj M=128 | 128x128, 108 × 67 | 120.83 | 0.00 | 1.85 | 1.47 / 0.97 | 7.29 / 8.38 (10.72) | 16.38 | 0.00 + 1.89 + 98.03 + 20.68 + 0.10 |
| down_proj M=256 | 64x256, 108 × 133 | 155.65 | 0.00 | 1.30 | 1.02 / 0.32 | 4.67 / 7.15 (12.37) | 15.36 | 0.00 + 1.25 + 135.05 + 19.78 + 0.08 |

Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.

| shape | sum warps | handoff | last block's sum | last, saw it early | SM GHz (runs) | globaltimer tick |
|---|---:|---:|---:|---:|---:|---:|
| qkv_proj M=1 | 0.66 | 1.21 | 0.85 | 42% | 1.15–1.20 | 1024 ns |
| qkv_proj M=16 | 0.71 | 1.50 | 1.14 | 23% | 1.06–1.09 | 1024 ns |
| qkv_proj M=64 | 4.67 | 2.31 | 5.19 | 70% | 1.07–1.10 | 1024 ns |
| qkv_proj M=128 | 3.90 | 1.81 | 3.23 | 100% | 1.29–1.31 | 1024 ns |
| qkv_proj M=256 | 1.91 | 2.29 | 5.76 | 99% | 1.40–1.41 | 1024 ns |
| gate_up_proj M=1 | 0.45 | 0.62 | 0.92 | 100% | 1.39–1.42 | 1024 ns |
| gate_up_proj M=16 | 0.59 | 0.88 | 1.37 | 100% | 1.08–1.10 | 1024 ns |
| gate_up_proj M=64 | 2.46 | 1.56 | 8.98 | 100% | 1.09–1.10 | 1024 ns |
| gate_up_proj M=128 | 2.42 | 1.56 | 9.44 | 100% | 1.09–1.11 | 1024 ns |
| gate_up_proj M=256 | 1.91 | 2.49 | 0.00 | 0% | 1.11–1.38 | 1024 ns |
| down_proj M=1 | 0.56 | 1.01 | 0.93 | 66% | 1.38–1.43 | 1024 ns |
| down_proj M=16 | 0.72 | 1.47 | 1.85 | 72% | 1.07–1.10 | 1024 ns |
| down_proj M=64 | 4.65 | 2.34 | 5.28 | 64% | 1.08–1.10 | 1024 ns |
| down_proj M=128 | 4.64 | 2.20 | 5.65 | 84% | 1.09–1.10 | 1024 ns |
| down_proj M=256 | 1.93 | 2.38 | 10.28 | 91% | 1.39–1.41 | 1024 ns |

## Kernels under ncu

### ncu kbench_w4a16_linear: 21 kernels, 2510 us, 315 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 4>` | gemm | 6 | 33 | 136.2 | 17 | 262 | 19 | 12/12 | 191 | 75 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 4, 2>` | gemm | 3 | 29 | 240.1 | 9 | 144 | 16 | 12/12 | 188 | 85 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 4, 12` | gemm | 6 | 18 | 74.6 | 33 | 486 | 48 | 12/12 | 245 | 67 | wait, math_pipe_throttle |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 3 | 13 | 106.4 | 31 | 473 | 50 | 12/12 | 255 | 59 | wait, math_pipe_throttle |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 3 | 8 | 68.7 | 33 | 511 | 36 | 6/6 | 255 | 73 | wait, selected |

## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| w4a16_tiled_full | paged | 64 | 1024 | 256 | 15.20 | 4,211 | awq_marlin | 16.17 | 3,957 | 0.94x |
| w4a16_tiled_full | paged | 128 | 1024 | 256 | 26.05 | 4,913 | awq_marlin | 27.48 | 4,657 | 0.95x |
| w4a16_tiled_full | paged | 128 | 512 | 256 | 20.08 | 6,376 | awq_marlin | 21.43 | 5,972 | 0.94x |
| w4a16_tiled_full | paged | 256 | 512 | 256 | 36.16 | 7,080 | awq_marlin | 39.07 | 6,552 | 0.93x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_full_b128 | 24.84 | 24.77 | 0.07 | 300 | 2 | attention 52%, gemm 45%, norm 1%, activation 1% |
| nano_w4a16_tiled_full_b256 | 33.93 | 33.84 | 0.09 | 300 | 2 | gemm 57%, attention 40%, activation 1%, norm 1% |
| nano_w4a16_tiled_full_b64 | 14.65 | 14.60 | 0.04 | 300 | 2 | gemm 50%, attention 47%, norm 1%, activation 1% |
| vllmw4_b128 | 26.30 | 26.22 | 0.08 | 431 | 15 | attention 49%, other 43%, gemm 3%, elementwise/copy 2% |
| vllmw4_b256 | 36.70 | 36.62 | 0.08 | 430 | 14 | other 55%, attention 37%, gemm 3%, elementwise/copy 3% |
| vllmw4_b64 | 15.82 | 15.74 | 0.08 | 431 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

