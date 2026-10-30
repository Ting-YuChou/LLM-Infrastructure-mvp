# Phase 1 results: run awq-a3-20261030-054204-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 294 |
| kbench | ok | 212 |
| e2e | ok | 1008 |
| environment | ok | 2 |

## Environment

```
run_id: awq-a3-20261030-054204-colab
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
962 passed, 1 warning in 292.62s (0:04:52)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,380 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 19.1 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 18.1 | 729 | 47 | 53 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 18.1 | 730 | 47 | 53 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 85.3 | 155 | 10 | 11 | 0.22 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.0 | 322 | 21 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.2 | 818 | 53 | 59 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 16.1 | 825 | 53 | 60 | 1.01 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 16.1 | 825 | 53 | 60 | 1.01 |
| qkv_proj M=4 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 86.5 | 153 | 10 | 11 | 0.19 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.0 | 323 | 21 | 23 | 0.40 |
| qkv_proj M=8 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.3 | 820 | 53 | 59 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 16.1 | 830 | 53 | 60 | 1.01 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 16.1 | 831 | 53 | 60 | 1.01 |
| qkv_proj M=8 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 86.8 | 154 | 10 | 11 | 0.19 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.1 | 324 | 21 | 23 | 0.40 |
| qkv_proj M=16 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 17.7 | 763 | 49 | 55 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.0 | 844 | 54 | 61 | 1.11 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 16.0 | 845 | 54 | 61 | 1.11 |
| qkv_proj M=16 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 87.5 | 154 | 10 | 11 | 0.20 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.5 | 325 | 21 | 24 | 0.43 |
| qkv_proj M=32 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 21.2 | 653 | 42 | 47 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 19.6 | 706 | 45 | 51 | 1.08 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 19.6 | 705 | 45 | 51 | 1.08 |
| qkv_proj M=32 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 87.7 | 158 | 10 | 11 | 0.24 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.3 | 327 | 21 | 24 | 0.50 |
| qkv_proj M=64 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 35.8 | 405 | 26 | 29 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 32.8 | 442 | 28 | 32 | 1.09 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 32.8 | 441 | 28 | 32 | 1.09 |
| qkv_proj M=64 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 66.1 | 219 | 14 | 16 | 0.54 |
| qkv_proj M=64 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 80.4 | 180 | 12 | 13 | 0.44 |
| qkv_proj M=64 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 88.8 | 163 | 10 | 12 | 0.40 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.8 | 339 | 22 | 25 | 0.84 |
| qkv_proj M=128 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 49.9 | 317 | 20 | 23 | 1.00 |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 47.4 | 333 | 21 | 24 | 1.05 |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 47.4 | 333 | 21 | 24 | 1.05 |
| qkv_proj M=128 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 88.1 | 179 | 12 | 13 | 0.57 |
| qkv_proj M=128 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 89.7 | 176 | 11 | 13 | 0.56 |
| qkv_proj M=128 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 103.8 | 152 | 10 | 11 | 0.48 |
| qkv_proj M=128 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 56.4 | 280 | 18 | 20 | 0.88 |
| qkv_proj M=256 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 78.7 | 234 | 15 | 17 | 1.00 |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 75.7 | 243 | 16 | 18 | 1.04 |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 75.7 | 243 | 16 | 18 | 1.04 |
| qkv_proj M=256 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 111.2 | 166 | 11 | 12 | 0.71 |
| qkv_proj M=256 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 108.3 | 170 | 11 | 12 | 0.73 |
| qkv_proj M=256 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 106.7 | 173 | 11 | 13 | 0.74 |
| qkv_proj M=256 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 61.2 | 301 | 19 | 22 | 1.29 |
| qkv_proj M=384 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 113.0 | 186 | 12 | 14 | 1.00 |
| qkv_proj M=384 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 106.3 | 198 | 13 | 14 | 1.06 |
| qkv_proj M=384 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 106.4 | 198 | 13 | 14 | 1.06 |
| qkv_proj M=384 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 147.1 | 143 | 9 | 10 | 0.77 |
| qkv_proj M=384 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 172.2 | 122 | 8 | 9 | 0.66 |
| qkv_proj M=384 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 138.1 | 152 | 10 | 11 | 0.82 |
| qkv_proj M=384 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 96.1 | 219 | 14 | 16 | 1.18 |
| qkv_proj M=512 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 140.6 | 168 | 11 | 12 | 1.00 |
| qkv_proj M=512 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 141.4 | 167 | 11 | 12 | 0.99 |
| qkv_proj M=512 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 141.6 | 167 | 11 | 12 | 0.99 |
| qkv_proj M=512 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 181.0 | 131 | 8 | 9 | 0.78 |
| qkv_proj M=512 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 176.3 | 134 | 9 | 10 | 0.80 |
| qkv_proj M=512 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 148.7 | 159 | 10 | 12 | 0.95 |
| qkv_proj M=512 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 110.1 | 215 | 14 | 16 | 1.28 |
| qkv_proj M=768 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 204.8 | 141 | 9 | 10 | 1.00 |
| qkv_proj M=768 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=768 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=768 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 249.0 | 116 | 7 | 8 | 0.82 |
| qkv_proj M=768 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 240.7 | 120 | 8 | 9 | 0.85 |
| qkv_proj M=768 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 244.6 | 118 | 8 | 9 | 0.84 |
| qkv_proj M=768 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 203.8 | 142 | 9 | 10 | 1.00 |
| qkv_proj M=1024 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 270.2 | 126 | 8 | 9 | 1.00 |
| qkv_proj M=1024 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=1024 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=1024 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 323.2 | 106 | 7 | 8 | 0.84 |
| qkv_proj M=1024 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 302.3 | 113 | 7 | 8 | 0.89 |
| qkv_proj M=1024 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 256.2 | 133 | 9 | 10 | 1.05 |
| qkv_proj M=1024 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 215.9 | 158 | 10 | 11 | 1.25 |
| qkv_proj M=2048 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 540.2 | 102 | 7 | 7 | 1.00 |
| qkv_proj M=2048 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=2048 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=2048 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 609.8 | 90 | 6 | 7 | 0.89 |
| qkv_proj M=2048 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 578.0 | 95 | 6 | 7 | 0.93 |
| qkv_proj M=2048 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 470.7 | 117 | 8 | 8 | 1.15 |
| qkv_proj M=2048 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 428.4 | 129 | 8 | 9 | 1.26 |
| o_proj M=1 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.8 | 595 | 38 | 43 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.5 | 605 | 39 | 44 | 1.02 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.6 | 605 | 39 | 44 | 1.02 |
| o_proj M=1 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 52.7 | 167 | 11 | 12 | 0.28 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.6 | 308 | 20 | 22 | 0.52 |
| o_proj M=4 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.7 | 602 | 39 | 44 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.6 | 605 | 39 | 44 | 1.01 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.6 | 605 | 39 | 44 | 1.01 |
| o_proj M=4 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.4 | 166 | 11 | 12 | 0.28 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.8 | 308 | 20 | 22 | 0.51 |
| o_proj M=8 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.7 | 607 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.6 | 608 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.7 | 607 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.7 | 166 | 11 | 12 | 0.27 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.9 | 309 | 20 | 22 | 0.51 |
| o_proj M=16 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.1 | 563 | 36 | 41 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.9 | 606 | 39 | 44 | 1.08 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.9 | 606 | 39 | 44 | 1.08 |
| o_proj M=16 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.8 | 168 | 11 | 12 | 0.30 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 29.0 | 312 | 20 | 23 | 0.55 |
| o_proj M=32 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 17.0 | 547 | 35 | 40 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 18.4 | 505 | 32 | 37 | 0.92 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 18.4 | 505 | 32 | 37 | 0.92 |
| o_proj M=32 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 54.1 | 172 | 11 | 12 | 0.31 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 29.3 | 317 | 20 | 23 | 0.58 |
| o_proj M=64 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 28.3 | 348 | 22 | 25 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 27.7 | 355 | 23 | 26 | 1.02 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 27.7 | 354 | 23 | 26 | 1.02 |
| o_proj M=64 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 52.6 | 187 | 12 | 14 | 0.54 |
| o_proj M=64 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 61.1 | 161 | 10 | 12 | 0.46 |
| o_proj M=64 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 54.6 | 180 | 12 | 13 | 0.52 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 30.0 | 328 | 21 | 24 | 0.94 |
| o_proj M=128 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 41.4 | 263 | 17 | 19 | 1.00 |
| o_proj M=128 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 36.7 | 296 | 19 | 21 | 1.13 |
| o_proj M=128 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 36.8 | 296 | 19 | 21 | 1.13 |
| o_proj M=128 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 70.5 | 154 | 10 | 11 | 0.59 |
| o_proj M=128 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 70.8 | 154 | 10 | 11 | 0.58 |
| o_proj M=128 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 64.6 | 168 | 11 | 12 | 0.64 |
| o_proj M=128 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.5 | 262 | 17 | 19 | 1.00 |
| o_proj M=256 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 56.5 | 230 | 15 | 17 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 56.6 | 229 | 15 | 17 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 56.8 | 229 | 15 | 17 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 91.1 | 142 | 9 | 10 | 0.62 |
| o_proj M=256 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 88.2 | 147 | 9 | 11 | 0.64 |
| o_proj M=256 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 78.0 | 166 | 11 | 12 | 0.72 |
| o_proj M=256 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 54.2 | 239 | 15 | 17 | 1.04 |
| o_proj M=384 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 78.9 | 191 | 12 | 14 | 1.00 |
| o_proj M=384 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 75.3 | 200 | 13 | 15 | 1.05 |
| o_proj M=384 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 75.3 | 200 | 13 | 15 | 1.05 |
| o_proj M=384 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 111.9 | 135 | 9 | 10 | 0.70 |
| o_proj M=384 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 128.0 | 118 | 8 | 9 | 0.62 |
| o_proj M=384 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 81.5 | 185 | 12 | 13 | 0.97 |
| o_proj M=384 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 58.3 | 259 | 17 | 19 | 1.35 |
| o_proj M=512 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 102.2 | 168 | 11 | 12 | 1.00 |
| o_proj M=512 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 98.8 | 174 | 11 | 13 | 1.03 |
| o_proj M=512 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 98.7 | 174 | 11 | 13 | 1.03 |
| o_proj M=512 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 135.2 | 127 | 8 | 9 | 0.76 |
| o_proj M=512 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 130.7 | 131 | 8 | 10 | 0.78 |
| o_proj M=512 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 115.6 | 149 | 10 | 11 | 0.88 |
| o_proj M=512 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 92.2 | 186 | 12 | 13 | 1.11 |
| o_proj M=768 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 140.8 | 152 | 10 | 11 | 1.00 |
| o_proj M=768 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=768 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=768 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 180.7 | 118 | 8 | 9 | 0.78 |
| o_proj M=768 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 175.8 | 122 | 8 | 9 | 0.80 |
| o_proj M=768 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 132.9 | 161 | 10 | 12 | 1.06 |
| o_proj M=768 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 109.1 | 196 | 13 | 14 | 1.29 |
| o_proj M=1024 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 187.0 | 137 | 9 | 10 | 1.00 |
| o_proj M=1024 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=1024 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=1024 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 232.1 | 110 | 7 | 8 | 0.81 |
| o_proj M=1024 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 218.5 | 117 | 8 | 8 | 0.86 |
| o_proj M=1024 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 226.9 | 113 | 7 | 8 | 0.82 |
| o_proj M=1024 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 203.3 | 126 | 8 | 9 | 0.92 |
| o_proj M=2048 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 354.7 | 119 | 8 | 9 | 1.00 |
| o_proj M=2048 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=2048 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=2048 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 440.4 | 96 | 6 | 7 | 0.81 |
| o_proj M=2048 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 404.6 | 105 | 7 | 8 | 0.88 |
| o_proj M=2048 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 337.6 | 125 | 8 | 9 | 1.05 |
| o_proj M=2048 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 314.3 | 135 | 9 | 10 | 1.13 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 54.9 | 1,120 | 72 | 81 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 52.7 | 1,168 | 75 | 85 | 1.04 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 52.6 | 1,169 | 75 | 85 | 1.04 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 423.8 | 145 | 9 | 11 | 0.13 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 178.1 | 346 | 22 | 25 | 0.31 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 55.4 | 1,114 | 72 | 81 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 53.7 | 1,151 | 74 | 83 | 1.03 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 53.7 | 1,151 | 74 | 83 | 1.03 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 424.4 | 145 | 9 | 11 | 0.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 180.1 | 343 | 22 | 25 | 0.31 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 55.9 | 1,109 | 71 | 80 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 54.5 | 1,137 | 73 | 82 | 1.03 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 54.4 | 1,140 | 73 | 83 | 1.03 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 425.8 | 146 | 9 | 11 | 0.13 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 181.3 | 342 | 22 | 25 | 0.31 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 59.4 | 1,053 | 68 | 76 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 55.7 | 1,123 | 72 | 81 | 1.07 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 55.6 | 1,125 | 72 | 82 | 1.07 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 426.9 | 146 | 9 | 11 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 183.0 | 342 | 22 | 25 | 0.32 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 61.2 | 1,038 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 61.1 | 1,040 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 61.4 | 1,036 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 456.2 | 139 | 9 | 10 | 0.13 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 213.6 | 298 | 19 | 22 | 0.29 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 91.4 | 718 | 46 | 52 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 90.8 | 723 | 47 | 52 | 1.01 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 90.8 | 723 | 46 | 52 | 1.01 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 182.9 | 359 | 23 | 26 | 0.50 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 296.0 | 222 | 14 | 16 | 0.31 |
| gate_up_proj M=64 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 508.3 | 129 | 8 | 9 | 0.18 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 264.8 | 248 | 16 | 18 | 0.35 |
| gate_up_proj M=128 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 171.7 | 407 | 26 | 30 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 164.3 | 425 | 27 | 31 | 1.05 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 164.3 | 425 | 27 | 31 | 1.04 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 203.4 | 343 | 22 | 25 | 0.84 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 310.8 | 225 | 14 | 16 | 0.55 |
| gate_up_proj M=128 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 515.7 | 135 | 9 | 10 | 0.33 |
| gate_up_proj M=128 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 268.3 | 260 | 17 | 19 | 0.64 |
| gate_up_proj M=256 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 330.6 | 237 | 15 | 17 | 1.00 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 309.6 | 253 | 16 | 18 | 1.07 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 309.9 | 253 | 16 | 18 | 1.07 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 358.5 | 218 | 14 | 16 | 0.92 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 349.6 | 224 | 14 | 16 | 0.95 |
| gate_up_proj M=256 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 555.1 | 141 | 9 | 10 | 0.60 |
| gate_up_proj M=256 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 323.0 | 242 | 16 | 18 | 1.02 |
| gate_up_proj M=384 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 486.2 | 178 | 11 | 13 | 1.00 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 455.1 | 190 | 12 | 14 | 1.07 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 455.2 | 190 | 12 | 14 | 1.07 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 510.1 | 170 | 11 | 12 | 0.95 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 629.6 | 138 | 9 | 10 | 0.77 |
| gate_up_proj M=384 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 646.7 | 134 | 9 | 10 | 0.75 |
| gate_up_proj M=384 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 428.7 | 202 | 13 | 15 | 1.13 |
| gate_up_proj M=512 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 645.6 | 147 | 9 | 11 | 1.00 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 620.8 | 153 | 10 | 11 | 1.04 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 620.9 | 153 | 10 | 11 | 1.04 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 671.4 | 142 | 9 | 10 | 0.96 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 655.4 | 145 | 9 | 11 | 0.99 |
| gate_up_proj M=512 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 756.0 | 126 | 8 | 9 | 0.85 |
| gate_up_proj M=512 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 540.5 | 176 | 11 | 13 | 1.19 |
| gate_up_proj M=768 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 969.5 | 115 | 7 | 8 | 1.00 |
| gate_up_proj M=768 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=768 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=768 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 979.9 | 114 | 7 | 8 | 0.99 |
| gate_up_proj M=768 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 959.3 | 117 | 7 | 8 | 1.01 |
| gate_up_proj M=768 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 986.7 | 113 | 7 | 8 | 0.98 |
| gate_up_proj M=768 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 785.6 | 142 | 9 | 10 | 1.23 |
| gate_up_proj M=1024 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 1285.4 | 100 | 6 | 7 | 1.00 |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 1333.0 | 96 | 6 | 7 | 0.96 |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 1260.4 | 102 | 7 | 7 | 1.02 |
| gate_up_proj M=1024 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 1193.8 | 108 | 7 | 8 | 1.08 |
| gate_up_proj M=1024 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 992.0 | 130 | 8 | 9 | 1.30 |
| gate_up_proj M=2048 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 2587.5 | 76 | 5 | 5 | 1.00 |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 2803.3 | 70 | 4 | 5 | 0.92 |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 2555.7 | 77 | 5 | 6 | 1.01 |
| gate_up_proj M=2048 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 2096.9 | 93 | 6 | 7 | 1.23 |
| gate_up_proj M=2048 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 1885.8 | 104 | 7 | 8 | 1.37 |
| down_proj M=1 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.0 | 1,025 | 66 | 74 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 30.8 | 1,000 | 64 | 72 | 0.98 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 30.8 | 1,000 | 64 | 73 | 0.98 |
| down_proj M=1 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 213.7 | 144 | 9 | 10 | 0.14 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 91.8 | 335 | 22 | 24 | 0.33 |
| down_proj M=4 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.1 | 1,027 | 66 | 74 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 31.1 | 993 | 64 | 72 | 0.97 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 31.1 | 993 | 64 | 72 | 0.97 |
| down_proj M=4 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 214.5 | 144 | 9 | 10 | 0.14 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 92.0 | 336 | 22 | 24 | 0.33 |
| down_proj M=8 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.2 | 1,028 | 66 | 74 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 31.5 | 984 | 63 | 71 | 0.96 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 31.5 | 984 | 63 | 71 | 0.96 |
| down_proj M=8 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 215.8 | 144 | 9 | 10 | 0.14 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 92.3 | 336 | 22 | 24 | 0.33 |
| down_proj M=16 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 32.4 | 967 | 62 | 70 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 32.3 | 970 | 62 | 70 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 32.3 | 970 | 62 | 70 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 216.2 | 145 | 9 | 11 | 0.15 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.7 | 338 | 22 | 24 | 0.35 |
| down_proj M=32 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 37.6 | 850 | 55 | 62 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 37.1 | 860 | 55 | 62 | 1.01 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 37.0 | 862 | 55 | 62 | 1.01 |
| down_proj M=32 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 219.2 | 146 | 9 | 11 | 0.17 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 100.2 | 319 | 20 | 23 | 0.38 |
| down_proj M=64 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 65.3 | 507 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 56.9 | 581 | 37 | 42 | 1.15 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 56.8 | 583 | 37 | 42 | 1.15 |
| down_proj M=64 N=4096 K=14336 | cuda_large_128x256 | ok (7.8e-03) | 108.4 | 305 | 20 | 22 | 0.60 |
| down_proj M=64 N=4096 K=14336 | cuda_large_256x128 | ok (7.8e-03) | 161.4 | 205 | 13 | 15 | 0.40 |
| down_proj M=64 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 219.6 | 151 | 10 | 11 | 0.30 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 99.4 | 333 | 21 | 24 | 0.66 |
| down_proj M=128 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 96.2 | 369 | 24 | 27 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 91.4 | 388 | 25 | 28 | 1.05 |
| down_proj M=128 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 91.6 | 387 | 25 | 28 | 1.05 |
| down_proj M=128 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 129.9 | 273 | 18 | 20 | 0.74 |
| down_proj M=128 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 172.2 | 206 | 13 | 15 | 0.56 |
| down_proj M=128 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 231.0 | 154 | 10 | 11 | 0.42 |
| down_proj M=128 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 111.6 | 318 | 20 | 23 | 0.86 |
| down_proj M=256 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 169.7 | 237 | 15 | 17 | 1.00 |
| down_proj M=256 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 158.7 | 253 | 16 | 18 | 1.07 |
| down_proj M=256 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 158.4 | 254 | 16 | 18 | 1.07 |
| down_proj M=256 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 199.8 | 201 | 13 | 15 | 0.85 |
| down_proj M=256 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 192.9 | 208 | 13 | 15 | 0.88 |
| down_proj M=256 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 257.4 | 156 | 10 | 11 | 0.66 |
| down_proj M=256 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 149.2 | 269 | 17 | 20 | 1.14 |
| down_proj M=384 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 242.9 | 185 | 12 | 13 | 1.00 |
| down_proj M=384 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 230.3 | 195 | 13 | 14 | 1.05 |
| down_proj M=384 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 227.9 | 197 | 13 | 14 | 1.07 |
| down_proj M=384 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 271.6 | 165 | 11 | 12 | 0.89 |
| down_proj M=384 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 328.0 | 137 | 9 | 10 | 0.74 |
| down_proj M=384 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 304.8 | 147 | 9 | 11 | 0.80 |
| down_proj M=384 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 200.5 | 224 | 14 | 16 | 1.21 |
| down_proj M=512 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 323.3 | 153 | 10 | 11 | 1.00 |
| down_proj M=512 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 309.6 | 160 | 10 | 12 | 1.04 |
| down_proj M=512 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 309.6 | 160 | 10 | 12 | 1.04 |
| down_proj M=512 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 349.3 | 142 | 9 | 10 | 0.93 |
| down_proj M=512 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 337.7 | 147 | 9 | 11 | 0.96 |
| down_proj M=512 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 372.0 | 133 | 9 | 10 | 0.87 |
| down_proj M=512 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 271.0 | 183 | 12 | 13 | 1.19 |
| down_proj M=768 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 463.2 | 127 | 8 | 9 | 1.00 |
| down_proj M=768 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=768 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=768 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 507.9 | 116 | 7 | 8 | 0.91 |
| down_proj M=768 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 497.8 | 119 | 8 | 9 | 0.93 |
| down_proj M=768 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 471.9 | 125 | 8 | 9 | 0.98 |
| down_proj M=768 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 370.3 | 159 | 10 | 12 | 1.25 |
| down_proj M=1024 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 612.2 | 112 | 7 | 8 | 1.00 |
| down_proj M=1024 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=1024 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=1024 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 685.6 | 100 | 6 | 7 | 0.89 |
| down_proj M=1024 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 668.6 | 102 | 7 | 7 | 0.92 |
| down_proj M=1024 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 611.3 | 112 | 7 | 8 | 1.00 |
| down_proj M=1024 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 509.6 | 134 | 9 | 10 | 1.20 |
| down_proj M=2048 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 1203.0 | 88 | 6 | 6 | 1.00 |
| down_proj M=2048 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=2048 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=2048 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 1403.5 | 76 | 5 | 5 | 0.86 |
| down_proj M=2048 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 1410.1 | 75 | 5 | 5 | 0.85 |
| down_proj M=2048 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 1066.2 | 100 | 6 | 7 | 1.13 |
| down_proj M=2048 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 968.0 | 110 | 7 | 8 | 1.24 |


## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| awq | paged | 1 | 128 | 256 | 5.16 | 194 | awq_marlin | 5.55 | 180 | 0.93x |
| awq | paged | 16 | 1024 | 256 | 7.13 | 2,244 | awq_marlin | 7.86 | 2,037 | 0.91x |
| awq | paged | 64 | 1024 | 256 | 15.25 | 4,198 | awq_marlin | 16.20 | 3,950 | 0.94x |
| awq | paged | 128 | 1024 | 256 | 26.60 | 4,813 | awq_marlin | 27.35 | 4,680 | 0.97x |
| awq | paged | 256 | 512 | 256 | 37.04 | 6,911 | awq_marlin | 38.88 | 6,584 | 0.95x |
| awq_int8dense | paged | 1 | 128 | 256 | 4.80 | 208 | awq_marlin | 5.55 | 180 | 0.87x |
| awq_int8dense | paged | 16 | 1024 | 256 | 6.77 | 2,365 | awq_marlin | 7.86 | 2,037 | 0.86x |
| awq_int8dense | paged | 64 | 1024 | 256 | 14.95 | 4,280 | awq_marlin | 16.20 | 3,950 | 0.92x |
| awq_int8dense | paged | 128 | 1024 | 256 | 26.61 | 4,811 | awq_marlin | 27.35 | 4,680 | 0.97x |
| awq_int8dense | paged | 256 | 512 | 256 | 37.07 | 6,907 | awq_marlin | 38.88 | 6,584 | 0.95x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_awq_b1 | 5.21 | 5.16 | 0.05 | 330 | 2 | gemm 84%, attention 8%, norm 4%, rope 2% |
| nano_awq_b128 | 25.46 | 25.19 | 0.27 | 300 | 70 | attention 52%, gemm 46%, norm 1%, activation 1% |
| nano_awq_b16 | 7.02 | 6.97 | 0.04 | 332 | 2 | gemm 65%, attention 29%, norm 3%, rope 2% |
| nano_awq_b64 | 14.73 | 14.68 | 0.05 | 300 | 2 | gemm 50%, attention 47%, norm 1%, activation 1% |
| vllmw4_b1 | 5.66 | 5.59 | 0.07 | 366 | 15 | other 68%, gemm 13%, attention 11%, norm 4% |
| vllmw4_b128 | 26.22 | 26.14 | 0.08 | 429 | 15 | attention 49%, other 43%, gemm 3%, elementwise/copy 2% |
| vllmw4_b16 | 7.82 | 7.74 | 0.08 | 429 | 15 | other 52%, attention 29%, gemm 10%, elementwise/copy 4% |
| vllmw4_b64 | 15.80 | 15.72 | 0.09 | 431 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

