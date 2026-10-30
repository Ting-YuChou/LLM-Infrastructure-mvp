# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,381 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 19.1 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 19.1 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 19.1 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 85.3 | 155 | 10 | 11 | 0.22 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.0 | 321 | 21 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.2 | 819 | 53 | 59 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 16.1 | 825 | 53 | 60 | 1.01 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 16.1 | 826 | 53 | 60 | 1.01 |
| qkv_proj M=4 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 86.5 | 153 | 10 | 11 | 0.19 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.0 | 324 | 21 | 23 | 0.40 |
| qkv_proj M=8 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.3 | 818 | 53 | 59 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 16.0 | 835 | 54 | 60 | 1.02 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 16.0 | 835 | 54 | 60 | 1.02 |
| qkv_proj M=8 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 86.8 | 154 | 10 | 11 | 0.19 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.2 | 324 | 21 | 23 | 0.40 |
| qkv_proj M=16 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 17.7 | 764 | 49 | 55 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.0 | 844 | 54 | 61 | 1.11 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 15.9 | 847 | 54 | 61 | 1.11 |
| qkv_proj M=16 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 87.5 | 154 | 10 | 11 | 0.20 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.5 | 325 | 21 | 24 | 0.43 |
| qkv_proj M=32 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 21.1 | 654 | 42 | 47 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 19.7 | 701 | 45 | 51 | 1.07 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 19.7 | 701 | 45 | 51 | 1.07 |
| qkv_proj M=32 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 87.7 | 158 | 10 | 11 | 0.24 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.2 | 328 | 21 | 24 | 0.50 |
| qkv_proj M=64 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 35.7 | 406 | 26 | 29 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 33.0 | 439 | 28 | 32 | 1.08 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 33.0 | 439 | 28 | 32 | 1.08 |
| qkv_proj M=64 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 65.6 | 221 | 14 | 16 | 0.54 |
| qkv_proj M=64 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 80.0 | 181 | 12 | 13 | 0.45 |
| qkv_proj M=64 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 89.1 | 163 | 10 | 12 | 0.40 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.9 | 338 | 22 | 24 | 0.83 |
| qkv_proj M=128 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 49.9 | 317 | 20 | 23 | 1.00 |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 49.0 | 323 | 21 | 23 | 1.02 |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 49.0 | 323 | 21 | 23 | 1.02 |
| qkv_proj M=128 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 88.1 | 179 | 12 | 13 | 0.57 |
| qkv_proj M=128 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 90.0 | 176 | 11 | 13 | 0.55 |
| qkv_proj M=128 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 103.0 | 153 | 10 | 11 | 0.48 |
| qkv_proj M=128 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 56.7 | 279 | 18 | 20 | 0.88 |
| qkv_proj M=256 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 78.7 | 234 | 15 | 17 | 1.00 |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 79.7 | 231 | 15 | 17 | 0.99 |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 79.6 | 231 | 15 | 17 | 0.99 |
| qkv_proj M=256 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 111.2 | 166 | 11 | 12 | 0.71 |
| qkv_proj M=256 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 108.3 | 170 | 11 | 12 | 0.73 |
| qkv_proj M=256 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 106.7 | 173 | 11 | 12 | 0.74 |
| qkv_proj M=256 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 61.3 | 301 | 19 | 22 | 1.29 |
| qkv_proj M=384 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 113.0 | 186 | 12 | 13 | 1.00 |
| qkv_proj M=384 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 111.2 | 189 | 12 | 14 | 1.02 |
| qkv_proj M=384 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 111.0 | 189 | 12 | 14 | 1.02 |
| qkv_proj M=384 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 147.3 | 143 | 9 | 10 | 0.77 |
| qkv_proj M=384 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 172.2 | 122 | 8 | 9 | 0.66 |
| qkv_proj M=384 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 138.2 | 152 | 10 | 11 | 0.82 |
| qkv_proj M=384 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 96.1 | 219 | 14 | 16 | 1.18 |
| qkv_proj M=512 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 140.6 | 168 | 11 | 12 | 1.00 |
| qkv_proj M=512 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 148.3 | 160 | 10 | 12 | 0.95 |
| qkv_proj M=512 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 148.3 | 160 | 10 | 12 | 0.95 |
| qkv_proj M=512 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 181.2 | 131 | 8 | 9 | 0.78 |
| qkv_proj M=512 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 176.1 | 134 | 9 | 10 | 0.80 |
| qkv_proj M=512 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 148.7 | 159 | 10 | 12 | 0.95 |
| qkv_proj M=512 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 110.1 | 215 | 14 | 16 | 1.28 |
| qkv_proj M=768 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 204.8 | 141 | 9 | 10 | 1.00 |
| qkv_proj M=768 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=768 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=768 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 249.1 | 116 | 7 | 8 | 0.82 |
| qkv_proj M=768 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 240.7 | 120 | 8 | 9 | 0.85 |
| qkv_proj M=768 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 244.6 | 118 | 8 | 9 | 0.84 |
| qkv_proj M=768 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 204.1 | 142 | 9 | 10 | 1.00 |
| qkv_proj M=1024 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 270.3 | 126 | 8 | 9 | 1.00 |
| qkv_proj M=1024 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=1024 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=1024 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 323.0 | 106 | 7 | 8 | 0.84 |
| qkv_proj M=1024 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 302.1 | 113 | 7 | 8 | 0.89 |
| qkv_proj M=1024 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 256.6 | 133 | 9 | 10 | 1.05 |
| qkv_proj M=1024 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 215.9 | 158 | 10 | 11 | 1.25 |
| qkv_proj M=2048 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 540.2 | 102 | 7 | 7 | 1.00 |
| qkv_proj M=2048 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=2048 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=2048 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 610.1 | 90 | 6 | 7 | 0.89 |
| qkv_proj M=2048 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 579.2 | 95 | 6 | 7 | 0.93 |
| qkv_proj M=2048 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 470.7 | 117 | 8 | 8 | 1.15 |
| qkv_proj M=2048 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 428.4 | 129 | 8 | 9 | 1.26 |
| o_proj M=1 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.8 | 596 | 38 | 43 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.6 | 604 | 39 | 44 | 1.01 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.6 | 604 | 39 | 44 | 1.01 |
| o_proj M=1 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.3 | 165 | 11 | 12 | 0.28 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.6 | 308 | 20 | 22 | 0.52 |
| o_proj M=4 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.7 | 601 | 39 | 44 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.7 | 602 | 39 | 44 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.7 | 602 | 39 | 44 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.0 | 167 | 11 | 12 | 0.28 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.7 | 308 | 20 | 22 | 0.51 |
| o_proj M=8 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.7 | 606 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.7 | 608 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.7 | 608 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.3 | 167 | 11 | 12 | 0.28 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.9 | 309 | 20 | 22 | 0.51 |
| o_proj M=16 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.0 | 565 | 36 | 41 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.9 | 607 | 39 | 44 | 1.07 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.9 | 607 | 39 | 44 | 1.07 |
| o_proj M=16 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.7 | 169 | 11 | 12 | 0.30 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 29.0 | 312 | 20 | 23 | 0.55 |
| o_proj M=32 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 17.0 | 546 | 35 | 40 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 18.7 | 497 | 32 | 36 | 0.91 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 18.7 | 498 | 32 | 36 | 0.91 |
| o_proj M=32 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.7 | 173 | 11 | 13 | 0.32 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 29.3 | 317 | 20 | 23 | 0.58 |
| o_proj M=64 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 28.3 | 348 | 22 | 25 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 27.7 | 355 | 23 | 26 | 1.02 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 27.7 | 355 | 23 | 26 | 1.02 |
| o_proj M=64 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 52.9 | 186 | 12 | 13 | 0.54 |
| o_proj M=64 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 61.1 | 161 | 10 | 12 | 0.46 |
| o_proj M=64 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 54.5 | 180 | 12 | 13 | 0.52 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 30.0 | 327 | 21 | 24 | 0.94 |
| o_proj M=128 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 41.4 | 263 | 17 | 19 | 1.00 |
| o_proj M=128 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 37.5 | 290 | 19 | 21 | 1.10 |
| o_proj M=128 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 37.5 | 290 | 19 | 21 | 1.10 |
| o_proj M=128 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 71.4 | 152 | 10 | 11 | 0.58 |
| o_proj M=128 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 69.7 | 156 | 10 | 11 | 0.59 |
| o_proj M=128 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 65.3 | 166 | 11 | 12 | 0.63 |
| o_proj M=128 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.5 | 262 | 17 | 19 | 1.00 |
| o_proj M=256 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 56.7 | 229 | 15 | 17 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 59.4 | 218 | 14 | 16 | 0.95 |
| o_proj M=256 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 59.5 | 218 | 14 | 16 | 0.95 |
| o_proj M=256 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 90.7 | 143 | 9 | 10 | 0.62 |
| o_proj M=256 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 88.5 | 147 | 9 | 11 | 0.64 |
| o_proj M=256 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 78.0 | 166 | 11 | 12 | 0.73 |
| o_proj M=256 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 54.2 | 239 | 15 | 17 | 1.04 |
| o_proj M=384 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 78.9 | 191 | 12 | 14 | 1.00 |
| o_proj M=384 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 79.2 | 190 | 12 | 14 | 1.00 |
| o_proj M=384 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 79.3 | 190 | 12 | 14 | 1.00 |
| o_proj M=384 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 111.9 | 135 | 9 | 10 | 0.71 |
| o_proj M=384 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 127.5 | 118 | 8 | 9 | 0.62 |
| o_proj M=384 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 81.3 | 185 | 12 | 13 | 0.97 |
| o_proj M=384 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 58.9 | 256 | 16 | 19 | 1.34 |
| o_proj M=512 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 102.2 | 168 | 11 | 12 | 1.00 |
| o_proj M=512 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 103.0 | 167 | 11 | 12 | 0.99 |
| o_proj M=512 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 103.0 | 167 | 11 | 12 | 0.99 |
| o_proj M=512 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 135.1 | 127 | 8 | 9 | 0.76 |
| o_proj M=512 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 130.5 | 132 | 8 | 10 | 0.78 |
| o_proj M=512 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 115.3 | 149 | 10 | 11 | 0.89 |
| o_proj M=512 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 92.3 | 186 | 12 | 13 | 1.11 |
| o_proj M=768 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 140.7 | 152 | 10 | 11 | 1.00 |
| o_proj M=768 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=768 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=768 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 180.7 | 118 | 8 | 9 | 0.78 |
| o_proj M=768 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 175.5 | 122 | 8 | 9 | 0.80 |
| o_proj M=768 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 133.0 | 161 | 10 | 12 | 1.06 |
| o_proj M=768 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 108.8 | 196 | 13 | 14 | 1.29 |
| o_proj M=1024 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 187.5 | 136 | 9 | 10 | 1.00 |
| o_proj M=1024 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=1024 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=1024 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 232.0 | 110 | 7 | 8 | 0.81 |
| o_proj M=1024 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 218.3 | 117 | 8 | 8 | 0.86 |
| o_proj M=1024 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 226.9 | 113 | 7 | 8 | 0.83 |
| o_proj M=1024 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 203.3 | 126 | 8 | 9 | 0.92 |
| o_proj M=2048 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 354.7 | 119 | 8 | 9 | 1.00 |
| o_proj M=2048 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=2048 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=2048 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 440.1 | 96 | 6 | 7 | 0.81 |
| o_proj M=2048 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 404.6 | 105 | 7 | 8 | 0.88 |
| o_proj M=2048 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 337.7 | 125 | 8 | 9 | 1.05 |
| o_proj M=2048 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 314.3 | 135 | 9 | 10 | 1.13 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 54.9 | 1,121 | 72 | 81 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 52.6 | 1,169 | 75 | 85 | 1.04 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 52.6 | 1,169 | 75 | 85 | 1.04 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 423.3 | 145 | 9 | 11 | 0.13 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 178.4 | 345 | 22 | 25 | 0.31 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 55.4 | 1,115 | 72 | 81 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 53.6 | 1,151 | 74 | 83 | 1.03 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 53.7 | 1,149 | 74 | 83 | 1.03 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 424.1 | 146 | 9 | 11 | 0.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 180.1 | 343 | 22 | 25 | 0.31 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 56.0 | 1,106 | 71 | 80 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 54.5 | 1,139 | 73 | 82 | 1.03 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 54.5 | 1,137 | 73 | 82 | 1.03 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 425.1 | 146 | 9 | 11 | 0.13 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 181.2 | 342 | 22 | 25 | 0.31 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 59.6 | 1,049 | 67 | 76 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 55.7 | 1,123 | 72 | 81 | 1.07 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 55.7 | 1,122 | 72 | 81 | 1.07 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 426.6 | 147 | 9 | 11 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 183.1 | 341 | 22 | 25 | 0.33 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 61.0 | 1,041 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 61.2 | 1,039 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 61.2 | 1,039 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 454.9 | 140 | 9 | 10 | 0.13 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 214.2 | 297 | 19 | 21 | 0.28 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 91.8 | 716 | 46 | 52 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 92.6 | 709 | 46 | 51 | 0.99 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 92.5 | 710 | 46 | 51 | 0.99 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 182.9 | 359 | 23 | 26 | 0.50 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 296.1 | 222 | 14 | 16 | 0.31 |
| gate_up_proj M=64 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 505.7 | 130 | 8 | 9 | 0.18 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 264.9 | 248 | 16 | 18 | 0.35 |
| gate_up_proj M=128 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 172.2 | 406 | 26 | 29 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 168.5 | 415 | 27 | 30 | 1.02 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 168.6 | 414 | 27 | 30 | 1.02 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 203.5 | 343 | 22 | 25 | 0.85 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 310.6 | 225 | 14 | 16 | 0.55 |
| gate_up_proj M=128 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 515.4 | 136 | 9 | 10 | 0.33 |
| gate_up_proj M=128 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 269.2 | 259 | 17 | 19 | 0.64 |
| gate_up_proj M=256 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 331.2 | 236 | 15 | 17 | 1.00 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 317.7 | 246 | 16 | 18 | 1.04 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 317.7 | 246 | 16 | 18 | 1.04 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 358.7 | 218 | 14 | 16 | 0.92 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 349.6 | 224 | 14 | 16 | 0.95 |
| gate_up_proj M=256 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 555.1 | 141 | 9 | 10 | 0.60 |
| gate_up_proj M=256 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 323.0 | 242 | 16 | 18 | 1.03 |
| gate_up_proj M=384 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 486.1 | 178 | 11 | 13 | 1.00 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 466.6 | 186 | 12 | 13 | 1.04 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 466.6 | 186 | 12 | 13 | 1.04 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 512.9 | 169 | 11 | 12 | 0.95 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 629.5 | 138 | 9 | 10 | 0.77 |
| gate_up_proj M=384 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 646.9 | 134 | 9 | 10 | 0.75 |
| gate_up_proj M=384 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 428.7 | 202 | 13 | 15 | 1.13 |
| gate_up_proj M=512 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 650.3 | 146 | 9 | 11 | 1.00 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 630.3 | 151 | 10 | 11 | 1.03 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 636.4 | 149 | 10 | 11 | 1.02 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 671.1 | 142 | 9 | 10 | 0.97 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 655.2 | 145 | 9 | 10 | 0.99 |
| gate_up_proj M=512 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 756.0 | 126 | 8 | 9 | 0.86 |
| gate_up_proj M=512 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 540.4 | 176 | 11 | 13 | 1.20 |
| gate_up_proj M=768 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 969.8 | 115 | 7 | 8 | 1.00 |
| gate_up_proj M=768 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=768 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=768 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 980.0 | 114 | 7 | 8 | 0.99 |
| gate_up_proj M=768 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 959.3 | 117 | 7 | 8 | 1.01 |
| gate_up_proj M=768 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 986.5 | 113 | 7 | 8 | 0.98 |
| gate_up_proj M=768 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 785.7 | 142 | 9 | 10 | 1.23 |
| gate_up_proj M=1024 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 1282.1 | 100 | 6 | 7 | 1.00 |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 1332.9 | 96 | 6 | 7 | 0.96 |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 1260.4 | 102 | 7 | 7 | 1.02 |
| gate_up_proj M=1024 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 1193.3 | 108 | 7 | 8 | 1.07 |
| gate_up_proj M=1024 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 987.9 | 130 | 8 | 9 | 1.30 |
| gate_up_proj M=2048 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 2588.5 | 76 | 5 | 5 | 1.00 |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 2806.8 | 70 | 4 | 5 | 0.92 |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 2558.9 | 76 | 5 | 6 | 1.01 |
| gate_up_proj M=2048 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 2099.4 | 93 | 6 | 7 | 1.23 |
| gate_up_proj M=2048 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 1892.2 | 103 | 7 | 7 | 1.37 |
| down_proj M=1 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.2 | 1,018 | 65 | 74 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 30.7 | 1,001 | 64 | 72 | 0.98 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 30.8 | 1,000 | 64 | 72 | 0.98 |
| down_proj M=1 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 216.0 | 142 | 9 | 10 | 0.14 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 91.9 | 335 | 22 | 24 | 0.33 |
| down_proj M=4 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.2 | 1,021 | 66 | 74 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 31.1 | 992 | 64 | 72 | 0.97 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 31.2 | 991 | 64 | 72 | 0.97 |
| down_proj M=4 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 214.6 | 144 | 9 | 10 | 0.14 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 92.1 | 336 | 22 | 24 | 0.33 |
| down_proj M=8 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.3 | 1,025 | 66 | 74 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 31.5 | 984 | 63 | 71 | 0.96 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 31.5 | 984 | 63 | 71 | 0.96 |
| down_proj M=8 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 214.7 | 145 | 9 | 10 | 0.14 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 92.3 | 336 | 22 | 24 | 0.33 |
| down_proj M=16 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 32.6 | 961 | 62 | 70 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 32.3 | 969 | 62 | 70 | 1.01 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 32.3 | 969 | 62 | 70 | 1.01 |
| down_proj M=16 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 215.0 | 146 | 9 | 11 | 0.15 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.6 | 338 | 22 | 24 | 0.35 |
| down_proj M=32 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 37.5 | 850 | 55 | 62 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 37.1 | 861 | 55 | 62 | 1.01 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 37.1 | 860 | 55 | 62 | 1.01 |
| down_proj M=32 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 219.4 | 145 | 9 | 11 | 0.17 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 100.5 | 317 | 20 | 23 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 65.3 | 507 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 59.5 | 556 | 36 | 40 | 1.10 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 59.5 | 556 | 36 | 40 | 1.10 |
| down_proj M=64 N=4096 K=14336 | cuda_large_128x256 | ok (7.8e-03) | 107.9 | 307 | 20 | 22 | 0.61 |
| down_proj M=64 N=4096 K=14336 | cuda_large_256x128 | ok (7.8e-03) | 161.2 | 205 | 13 | 15 | 0.41 |
| down_proj M=64 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 219.4 | 151 | 10 | 11 | 0.30 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 99.4 | 333 | 21 | 24 | 0.66 |
| down_proj M=128 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 96.1 | 369 | 24 | 27 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 95.8 | 370 | 24 | 27 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 95.8 | 370 | 24 | 27 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 128.6 | 276 | 18 | 20 | 0.75 |
| down_proj M=128 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 172.3 | 206 | 13 | 15 | 0.56 |
| down_proj M=128 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 230.4 | 154 | 10 | 11 | 0.42 |
| down_proj M=128 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 111.6 | 318 | 20 | 23 | 0.86 |
| down_proj M=256 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 169.8 | 237 | 15 | 17 | 1.00 |
| down_proj M=256 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 166.9 | 241 | 15 | 17 | 1.02 |
| down_proj M=256 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 166.9 | 241 | 15 | 17 | 1.02 |
| down_proj M=256 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 199.9 | 201 | 13 | 15 | 0.85 |
| down_proj M=256 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 193.2 | 208 | 13 | 15 | 0.88 |
| down_proj M=256 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 257.5 | 156 | 10 | 11 | 0.66 |
| down_proj M=256 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 147.9 | 272 | 17 | 20 | 1.15 |
| down_proj M=384 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 242.9 | 185 | 12 | 13 | 1.00 |
| down_proj M=384 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 240.5 | 187 | 12 | 14 | 1.01 |
| down_proj M=384 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 240.5 | 187 | 12 | 14 | 1.01 |
| down_proj M=384 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 271.4 | 165 | 11 | 12 | 0.90 |
| down_proj M=384 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 327.5 | 137 | 9 | 10 | 0.74 |
| down_proj M=384 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 305.3 | 147 | 9 | 11 | 0.80 |
| down_proj M=384 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 200.5 | 224 | 14 | 16 | 1.21 |
| down_proj M=512 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 323.3 | 153 | 10 | 11 | 1.00 |
| down_proj M=512 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 326.2 | 152 | 10 | 11 | 0.99 |
| down_proj M=512 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 326.2 | 152 | 10 | 11 | 0.99 |
| down_proj M=512 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 349.6 | 142 | 9 | 10 | 0.92 |
| down_proj M=512 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 338.2 | 147 | 9 | 11 | 0.96 |
| down_proj M=512 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 372.3 | 133 | 9 | 10 | 0.87 |
| down_proj M=512 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 269.6 | 184 | 12 | 13 | 1.20 |
| down_proj M=768 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 462.3 | 128 | 8 | 9 | 1.00 |
| down_proj M=768 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=768 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=768 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 507.7 | 116 | 7 | 8 | 0.91 |
| down_proj M=768 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 498.3 | 118 | 8 | 9 | 0.93 |
| down_proj M=768 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 471.4 | 125 | 8 | 9 | 0.98 |
| down_proj M=768 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 370.4 | 159 | 10 | 12 | 1.25 |
| down_proj M=1024 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 612.1 | 112 | 7 | 8 | 1.00 |
| down_proj M=1024 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=1024 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=1024 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 684.8 | 100 | 6 | 7 | 0.89 |
| down_proj M=1024 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 668.2 | 102 | 7 | 7 | 0.92 |
| down_proj M=1024 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 611.2 | 112 | 7 | 8 | 1.00 |
| down_proj M=1024 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 509.6 | 134 | 9 | 10 | 1.20 |
| down_proj M=2048 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 1203.8 | 88 | 6 | 6 | 1.00 |
| down_proj M=2048 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=2048 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=2048 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 1402.9 | 76 | 5 | 5 | 0.86 |
| down_proj M=2048 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 1412.5 | 75 | 5 | 5 | 0.85 |
| down_proj M=2048 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 1068.2 | 99 | 6 | 7 | 1.13 |
| down_proj M=2048 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 968.7 | 110 | 7 | 8 | 1.24 |

## lm_head

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=128256 K=4096 | bf16_nano | ok (1.6e-02) | 712.6 | 738 | 47 | 53 | 1.00 |
| M=1 N=128256 K=4096 | int8 | ok (1.6e-02) | 369.4 | 1,424 | 92 | 103 | 1.93 |
| M=1 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 623.2 | 844 | 54 | 61 | 1.14 |
| M=1 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 931.9 | 565 | 36 | 41 | 0.76 |
| M=1 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 1074.9 | 489 | 31 | 35 | 0.66 |
| M=1 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 492.3 | 1,069 | 69 | 77 | 1.45 |
| M=1 N=128256 K=4096 | cublas_bf16 | ok (1.6e-02) | 720.7 | 730 | 47 | 53 | 0.99 |
| M=4 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 721.0 | 731 | 47 | 53 | 1.00 |
| M=4 N=128256 K=4096 | int8 | ok (1.6e-02) | 375.0 | 1,405 | 90 | 102 | 1.92 |
| M=4 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 626.0 | 842 | 54 | 61 | 1.15 |
| M=4 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 1186.6 | 444 | 29 | 32 | 0.61 |
| M=4 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 1068.3 | 493 | 32 | 36 | 0.67 |
| M=4 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 490.7 | 1,074 | 69 | 78 | 1.47 |
| M=4 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 728.8 | 723 | 46 | 52 | 0.99 |
| M=16 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 735.8 | 720 | 46 | 52 | 1.00 |
| M=16 N=128256 K=4096 | int8 | ok (1.6e-02) | 386.4 | 1,372 | 88 | 99 | 1.90 |
| M=16 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 632.8 | 838 | 54 | 61 | 1.16 |
| M=16 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 1196.7 | 443 | 28 | 32 | 0.61 |
| M=16 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 1071.7 | 495 | 32 | 36 | 0.69 |
| M=16 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 494.9 | 1,071 | 69 | 78 | 1.49 |
| M=16 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 743.0 | 713 | 46 | 52 | 0.99 |
| M=32 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 767.5 | 696 | 45 | 50 | 1.00 |
| M=32 N=128256 K=4096 | int8 | ok (1.6e-02) | 506.2 | 1,055 | 68 | 76 | 1.52 |
| M=32 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 643.6 | 830 | 53 | 60 | 1.19 |
| M=32 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 1193.9 | 448 | 29 | 32 | 0.64 |
| M=32 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 1085.1 | 492 | 32 | 36 | 0.71 |
| M=32 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 507.1 | 1,054 | 68 | 76 | 1.51 |
| M=32 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 766.1 | 697 | 45 | 50 | 1.00 |
| M=64 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 778.6 | 697 | 45 | 50 | 1.00 |
| M=64 N=128256 K=4096 | int8 | ok (1.6e-02) | 523.5 | 1,037 | 67 | 75 | 1.49 |
| M=64 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 656.2 | 827 | 53 | 60 | 1.19 |
| M=64 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 1195.2 | 454 | 29 | 33 | 0.65 |
| M=64 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 1089.7 | 498 | 32 | 36 | 0.71 |
| M=64 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 523.6 | 1,037 | 67 | 75 | 1.49 |
| M=64 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 777.7 | 698 | 45 | 51 | 1.00 |
| M=128 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 813.7 | 688 | 44 | 50 | 1.00 |
| M=128 N=128256 K=4096 | int8 | ok (1.6e-02) | 1123.9 | 498 | 32 | 36 | 0.72 |
| M=128 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 1059.0 | 529 | 34 | 38 | 0.77 |
| M=128 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 1199.8 | 467 | 30 | 34 | 0.68 |
| M=128 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 1087.3 | 515 | 33 | 37 | 0.75 |
| M=128 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 967.3 | 579 | 37 | 42 | 0.84 |
| M=128 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 810.4 | 691 | 44 | 50 | 1.00 |
| M=256 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 1166.1 | 509 | 33 | 37 | 1.00 |
| M=256 N=128256 K=4096 | int8 | ok (1.6e-02) | 2110.8 | 281 | 18 | 20 | 0.55 |
| M=256 N=128256 K=4096 | int8_m64_n256_w8 | ok (1.6e-02) | 2005.8 | 296 | 19 | 21 | 0.58 |
| M=256 N=128256 K=4096 | int8_m128_n256_w8 | ok (1.6e-02) | 2342.5 | 253 | 16 | 18 | 0.50 |
| M=256 N=128256 K=4096 | int8_m128_k128_w8 | ok (1.6e-02) | 2059.1 | 288 | 19 | 21 | 0.57 |
| M=256 N=128256 K=4096 | int8_m64_k128 | ok (1.6e-02) | 1891.6 | 314 | 20 | 23 | 0.62 |
| M=256 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 1168.0 | 508 | 33 | 37 | 1.00 |

