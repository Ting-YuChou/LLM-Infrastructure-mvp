# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,384 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 19.1 | 692 | 45 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 20.0 | 661 | 43 | 48 | 0.95 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 20.0 | 661 | 43 | 48 | 0.95 |
| qkv_proj M=1 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 86.4 | 153 | 10 | 11 | 0.22 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.0 | 322 | 21 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.3 | 815 | 52 | 59 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 16.4 | 807 | 52 | 58 | 0.99 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 16.4 | 808 | 52 | 58 | 0.99 |
| qkv_proj M=4 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 87.3 | 152 | 10 | 11 | 0.19 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.0 | 323 | 21 | 23 | 0.40 |
| qkv_proj M=8 N=6144 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 16.2 | 822 | 53 | 59 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (7.8e-03) | 16.3 | 818 | 53 | 59 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 16.3 | 817 | 53 | 59 | 0.99 |
| qkv_proj M=8 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | dequant_cublas | ok (7.8e-03) | 87.2 | 153 | 10 | 11 | 0.19 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (7.8e-03) | 41.2 | 324 | 21 | 23 | 0.39 |
| qkv_proj M=16 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 17.8 | 759 | 49 | 55 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 16.3 | 827 | 53 | 60 | 1.09 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 16.3 | 826 | 53 | 60 | 1.09 |
| qkv_proj M=16 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 88.1 | 153 | 10 | 11 | 0.20 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.5 | 326 | 21 | 24 | 0.43 |
| qkv_proj M=32 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 21.2 | 653 | 42 | 47 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 20.4 | 678 | 44 | 49 | 1.04 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 20.4 | 677 | 44 | 49 | 1.04 |
| qkv_proj M=32 N=6144 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 88.1 | 157 | 10 | 11 | 0.24 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.2 | 328 | 21 | 24 | 0.50 |
| qkv_proj M=64 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 35.6 | 407 | 26 | 29 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 33.5 | 433 | 28 | 31 | 1.06 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 33.5 | 433 | 28 | 31 | 1.06 |
| qkv_proj M=64 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 65.5 | 221 | 14 | 16 | 0.54 |
| qkv_proj M=64 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 84.4 | 172 | 11 | 12 | 0.42 |
| qkv_proj M=64 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 89.1 | 163 | 10 | 12 | 0.40 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.8 | 338 | 22 | 24 | 0.83 |
| qkv_proj M=128 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 50.2 | 315 | 20 | 23 | 1.00 |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 49.7 | 318 | 20 | 23 | 1.01 |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 49.7 | 318 | 20 | 23 | 1.01 |
| qkv_proj M=128 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 86.3 | 183 | 12 | 13 | 0.58 |
| qkv_proj M=128 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 93.5 | 169 | 11 | 12 | 0.54 |
| qkv_proj M=128 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 103.1 | 153 | 10 | 11 | 0.49 |
| qkv_proj M=128 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 56.0 | 282 | 18 | 20 | 0.90 |
| qkv_proj M=256 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 79.3 | 232 | 15 | 17 | 1.00 |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 80.6 | 228 | 15 | 17 | 0.98 |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 80.6 | 229 | 15 | 17 | 0.98 |
| qkv_proj M=256 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 110.6 | 167 | 11 | 12 | 0.72 |
| qkv_proj M=256 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 111.7 | 165 | 11 | 12 | 0.71 |
| qkv_proj M=256 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 108.1 | 170 | 11 | 12 | 0.73 |
| qkv_proj M=256 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 61.1 | 301 | 19 | 22 | 1.30 |
| qkv_proj M=384 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 113.6 | 185 | 12 | 13 | 1.00 |
| qkv_proj M=384 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| qkv_proj M=384 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 113.1 | 186 | 12 | 13 | 1.00 |
| qkv_proj M=384 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 145.9 | 144 | 9 | 10 | 0.78 |
| qkv_proj M=384 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 179.9 | 117 | 8 | 8 | 0.63 |
| qkv_proj M=384 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 138.0 | 152 | 10 | 11 | 0.82 |
| qkv_proj M=384 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 96.3 | 218 | 14 | 16 | 1.18 |
| qkv_proj M=512 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 141.2 | 168 | 11 | 12 | 1.00 |
| qkv_proj M=512 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| qkv_proj M=512 N=6144 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 151.2 | 156 | 10 | 11 | 0.93 |
| qkv_proj M=512 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 179.4 | 132 | 8 | 10 | 0.79 |
| qkv_proj M=512 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 183.7 | 129 | 8 | 9 | 0.77 |
| qkv_proj M=512 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 149.4 | 158 | 10 | 11 | 0.95 |
| qkv_proj M=512 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 109.3 | 217 | 14 | 16 | 1.29 |
| qkv_proj M=1024 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 268.6 | 127 | 8 | 9 | 1.00 |
| qkv_proj M=1024 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| qkv_proj M=1024 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=1024 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 320.1 | 107 | 7 | 8 | 0.84 |
| qkv_proj M=1024 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 318.8 | 107 | 7 | 8 | 0.84 |
| qkv_proj M=1024 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 255.3 | 134 | 9 | 10 | 1.05 |
| qkv_proj M=1024 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 214.5 | 159 | 10 | 11 | 1.25 |
| qkv_proj M=2048 N=6144 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 537.5 | 103 | 7 | 7 | 1.00 |
| qkv_proj M=2048 N=6144 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| qkv_proj M=2048 N=6144 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| qkv_proj M=2048 N=6144 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 602.4 | 91 | 6 | 7 | 0.89 |
| qkv_proj M=2048 N=6144 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 605.4 | 91 | 6 | 7 | 0.89 |
| qkv_proj M=2048 N=6144 K=4096 | dequant_cublas | ok (1.6e-02) | 468.4 | 118 | 8 | 8 | 1.15 |
| qkv_proj M=2048 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 428.8 | 129 | 8 | 9 | 1.25 |
| o_proj M=1 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.6 | 601 | 39 | 43 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 14.8 | 593 | 38 | 43 | 0.99 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.9 | 592 | 38 | 43 | 0.98 |
| o_proj M=1 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.5 | 164 | 11 | 12 | 0.27 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.6 | 308 | 20 | 22 | 0.51 |
| o_proj M=4 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.6 | 607 | 39 | 44 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 15.0 | 591 | 38 | 43 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 14.9 | 592 | 38 | 43 | 0.98 |
| o_proj M=4 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.5 | 165 | 11 | 12 | 0.27 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.7 | 308 | 20 | 22 | 0.51 |
| o_proj M=8 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 14.6 | 609 | 39 | 44 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 15.0 | 596 | 38 | 43 | 0.98 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 15.0 | 596 | 38 | 43 | 0.98 |
| o_proj M=8 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 53.7 | 166 | 11 | 12 | 0.27 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 28.9 | 309 | 20 | 22 | 0.51 |
| o_proj M=16 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 15.9 | 567 | 36 | 41 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 15.2 | 596 | 38 | 43 | 1.05 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 15.2 | 596 | 38 | 43 | 1.05 |
| o_proj M=16 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 54.2 | 167 | 11 | 12 | 0.29 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 29.0 | 312 | 20 | 23 | 0.55 |
| o_proj M=32 N=4096 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 17.1 | 544 | 35 | 39 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (7.8e-03) | 19.2 | 484 | 31 | 35 | 0.89 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 19.2 | 484 | 31 | 35 | 0.89 |
| o_proj M=32 N=4096 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | dequant_cublas | ok (7.8e-03) | 54.3 | 171 | 11 | 12 | 0.31 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (7.8e-03) | 29.3 | 317 | 20 | 23 | 0.58 |
| o_proj M=64 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 28.3 | 348 | 22 | 25 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 28.1 | 350 | 23 | 25 | 1.01 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 28.1 | 350 | 22 | 25 | 1.01 |
| o_proj M=64 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 52.7 | 187 | 12 | 13 | 0.54 |
| o_proj M=64 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 63.5 | 155 | 10 | 11 | 0.45 |
| o_proj M=64 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 55.4 | 177 | 11 | 13 | 0.51 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 30.0 | 328 | 21 | 24 | 0.94 |
| o_proj M=128 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 41.6 | 262 | 17 | 19 | 1.00 |
| o_proj M=128 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 38.1 | 285 | 18 | 21 | 1.09 |
| o_proj M=128 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 38.1 | 285 | 18 | 21 | 1.09 |
| o_proj M=128 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 70.6 | 154 | 10 | 11 | 0.59 |
| o_proj M=128 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 71.7 | 152 | 10 | 11 | 0.58 |
| o_proj M=128 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 65.1 | 167 | 11 | 12 | 0.64 |
| o_proj M=128 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.4 | 263 | 17 | 19 | 1.00 |
| o_proj M=256 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 56.7 | 229 | 15 | 17 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 59.6 | 218 | 14 | 16 | 0.95 |
| o_proj M=256 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 59.7 | 217 | 14 | 16 | 0.95 |
| o_proj M=256 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 91.0 | 143 | 9 | 10 | 0.62 |
| o_proj M=256 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 90.6 | 143 | 9 | 10 | 0.63 |
| o_proj M=256 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 79.1 | 164 | 11 | 12 | 0.72 |
| o_proj M=256 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 54.4 | 238 | 15 | 17 | 1.04 |
| o_proj M=384 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 79.3 | 190 | 12 | 14 | 1.00 |
| o_proj M=384 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| o_proj M=384 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 80.4 | 187 | 12 | 14 | 0.99 |
| o_proj M=384 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 110.9 | 136 | 9 | 10 | 0.72 |
| o_proj M=384 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 133.6 | 113 | 7 | 8 | 0.59 |
| o_proj M=384 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 81.6 | 185 | 12 | 13 | 0.97 |
| o_proj M=384 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 58.5 | 257 | 17 | 19 | 1.35 |
| o_proj M=512 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 102.4 | 168 | 11 | 12 | 1.00 |
| o_proj M=512 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| o_proj M=512 N=4096 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 105.0 | 164 | 11 | 12 | 0.97 |
| o_proj M=512 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 134.0 | 128 | 8 | 9 | 0.76 |
| o_proj M=512 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 135.9 | 126 | 8 | 9 | 0.75 |
| o_proj M=512 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 116.1 | 148 | 10 | 11 | 0.88 |
| o_proj M=512 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 91.4 | 188 | 12 | 14 | 1.12 |
| o_proj M=1024 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 186.3 | 137 | 9 | 10 | 1.00 |
| o_proj M=1024 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| o_proj M=1024 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=1024 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 230.1 | 111 | 7 | 8 | 0.81 |
| o_proj M=1024 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 228.9 | 112 | 7 | 8 | 0.81 |
| o_proj M=1024 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 229.2 | 112 | 7 | 8 | 0.81 |
| o_proj M=1024 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 204.3 | 125 | 8 | 9 | 0.91 |
| o_proj M=2048 N=4096 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 352.9 | 120 | 8 | 9 | 1.00 |
| o_proj M=2048 N=4096 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| o_proj M=2048 N=4096 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| o_proj M=2048 N=4096 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 434.9 | 97 | 6 | 7 | 0.81 |
| o_proj M=2048 N=4096 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 423.3 | 100 | 6 | 7 | 0.83 |
| o_proj M=2048 N=4096 K=4096 | dequant_cublas | ok (1.6e-02) | 335.8 | 126 | 8 | 9 | 1.05 |
| o_proj M=2048 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 312.7 | 135 | 9 | 10 | 1.13 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 54.9 | 1,120 | 72 | 81 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 52.4 | 1,175 | 76 | 85 | 1.05 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 52.5 | 1,173 | 75 | 85 | 1.05 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 422.9 | 146 | 9 | 11 | 0.13 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 177.7 | 346 | 22 | 25 | 0.31 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 55.5 | 1,113 | 72 | 80 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 53.5 | 1,153 | 74 | 83 | 1.04 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 53.5 | 1,155 | 74 | 83 | 1.04 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 424.8 | 145 | 9 | 10 | 0.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 180.3 | 342 | 22 | 25 | 0.31 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 56.1 | 1,105 | 71 | 80 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 54.4 | 1,139 | 73 | 82 | 1.03 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 54.4 | 1,140 | 73 | 82 | 1.03 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 426.0 | 146 | 9 | 11 | 0.13 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 181.3 | 342 | 22 | 25 | 0.31 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_awq_marlin | ok (7.8e-03) | 59.5 | 1,050 | 68 | 76 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (7.8e-03) | 55.6 | 1,125 | 72 | 81 | 1.07 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled_512 | ok (7.8e-03) | 55.5 | 1,127 | 72 | 81 | 1.07 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | dequant_cublas | ok (7.8e-03) | 427.2 | 146 | 9 | 11 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (7.8e-03) | 183.0 | 342 | 22 | 25 | 0.33 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 61.1 | 1,040 | 67 | 75 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 61.6 | 1,032 | 66 | 75 | 0.99 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 61.8 | 1,029 | 66 | 74 | 0.99 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 457.1 | 139 | 9 | 10 | 0.13 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 213.3 | 298 | 19 | 22 | 0.29 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 91.4 | 719 | 46 | 52 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 92.7 | 708 | 46 | 51 | 0.99 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 92.7 | 709 | 46 | 51 | 0.99 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 180.7 | 363 | 23 | 26 | 0.51 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 316.8 | 207 | 13 | 15 | 0.29 |
| gate_up_proj M=64 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 509.8 | 129 | 8 | 9 | 0.18 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 264.1 | 249 | 16 | 18 | 0.35 |
| gate_up_proj M=128 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 171.0 | 408 | 26 | 30 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 171.1 | 408 | 26 | 30 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 170.9 | 409 | 26 | 30 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 201.8 | 346 | 22 | 25 | 0.85 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 330.6 | 211 | 14 | 15 | 0.52 |
| gate_up_proj M=128 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 515.7 | 135 | 9 | 10 | 0.33 |
| gate_up_proj M=128 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 270.1 | 259 | 17 | 19 | 0.63 |
| gate_up_proj M=256 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 328.3 | 238 | 15 | 17 | 1.00 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 321.3 | 244 | 16 | 18 | 1.02 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 321.2 | 244 | 16 | 18 | 1.02 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 354.0 | 221 | 14 | 16 | 0.93 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 361.5 | 216 | 14 | 16 | 0.91 |
| gate_up_proj M=256 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 556.9 | 141 | 9 | 10 | 0.59 |
| gate_up_proj M=256 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 320.6 | 244 | 16 | 18 | 1.02 |
| gate_up_proj M=384 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 485.9 | 178 | 11 | 13 | 1.00 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| gate_up_proj M=384 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 476.6 | 182 | 12 | 13 | 1.02 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 501.6 | 173 | 11 | 12 | 0.97 |
| gate_up_proj M=384 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 664.3 | 130 | 8 | 9 | 0.73 |
| gate_up_proj M=384 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 643.8 | 135 | 9 | 10 | 0.75 |
| gate_up_proj M=384 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 426.4 | 203 | 13 | 15 | 1.14 |
| gate_up_proj M=512 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 645.4 | 147 | 9 | 11 | 1.00 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| gate_up_proj M=512 N=28672 K=4096 | cuda_tiled_512 | ok (1.6e-02) | 645.3 | 147 | 9 | 11 | 1.00 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 655.2 | 145 | 9 | 10 | 0.98 |
| gate_up_proj M=512 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 676.1 | 141 | 9 | 10 | 0.95 |
| gate_up_proj M=512 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 751.4 | 126 | 8 | 9 | 0.86 |
| gate_up_proj M=512 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 534.8 | 178 | 11 | 13 | 1.21 |
| gate_up_proj M=1024 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 1270.1 | 101 | 7 | 7 | 1.00 |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 1302.0 | 99 | 6 | 7 | 0.98 |
| gate_up_proj M=1024 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 1301.7 | 99 | 6 | 7 | 0.98 |
| gate_up_proj M=1024 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 1184.8 | 109 | 7 | 8 | 1.07 |
| gate_up_proj M=1024 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 979.0 | 131 | 8 | 9 | 1.30 |
| gate_up_proj M=2048 N=28672 K=4096 | vllm_awq_marlin | ok (1.6e-02) | 2552.7 | 77 | 5 | 6 | 1.00 |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_large_128x256 | ok (1.6e-02) | 2766.5 | 71 | 5 | 5 | 0.92 |
| gate_up_proj M=2048 N=28672 K=4096 | cuda_large_256x128 | ok (1.6e-02) | 2636.0 | 74 | 5 | 5 | 0.97 |
| gate_up_proj M=2048 N=28672 K=4096 | dequant_cublas | ok (1.6e-02) | 2080.1 | 94 | 6 | 7 | 1.23 |
| gate_up_proj M=2048 N=28672 K=4096 | cublas_bf16 | ok (1.6e-02) | 1872.7 | 104 | 7 | 8 | 1.36 |
| down_proj M=1 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 29.7 | 1,036 | 67 | 75 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 30.7 | 1,001 | 64 | 72 | 0.97 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 30.8 | 1,000 | 64 | 72 | 0.97 |
| down_proj M=1 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 213.9 | 144 | 9 | 10 | 0.14 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 91.7 | 336 | 22 | 24 | 0.32 |
| down_proj M=4 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.1 | 1,026 | 66 | 74 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 31.1 | 993 | 64 | 72 | 0.97 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 31.1 | 993 | 64 | 72 | 0.97 |
| down_proj M=4 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 214.4 | 144 | 9 | 10 | 0.14 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 91.9 | 336 | 22 | 24 | 0.33 |
| down_proj M=8 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 30.4 | 1,022 | 66 | 74 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 31.5 | 985 | 63 | 71 | 0.96 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 31.5 | 984 | 63 | 71 | 0.96 |
| down_proj M=8 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 214.4 | 145 | 9 | 10 | 0.14 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 92.2 | 336 | 22 | 24 | 0.33 |
| down_proj M=16 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 32.5 | 964 | 62 | 70 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 32.2 | 972 | 62 | 70 | 1.01 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 32.2 | 973 | 63 | 70 | 1.01 |
| down_proj M=16 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 214.9 | 146 | 9 | 11 | 0.15 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.6 | 338 | 22 | 24 | 0.35 |
| down_proj M=32 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 37.7 | 847 | 54 | 61 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 37.3 | 855 | 55 | 62 | 1.01 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 37.4 | 853 | 55 | 62 | 1.01 |
| down_proj M=32 N=4096 K=14336 | cuda_large_128x256 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | cuda_large_256x128 | skipped: gemm_w4a16_large is timed from 64 rows | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 219.1 | 146 | 9 | 11 | 0.17 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 100.6 | 317 | 20 | 23 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_awq_marlin | ok (7.8e-03) | 65.3 | 506 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (7.8e-03) | 60.2 | 550 | 35 | 40 | 1.09 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled_512 | ok (7.8e-03) | 60.0 | 551 | 35 | 40 | 1.09 |
| down_proj M=64 N=4096 K=14336 | cuda_large_128x256 | ok (7.8e-03) | 106.8 | 310 | 20 | 22 | 0.61 |
| down_proj M=64 N=4096 K=14336 | cuda_large_256x128 | ok (7.8e-03) | 169.2 | 196 | 13 | 14 | 0.39 |
| down_proj M=64 N=4096 K=14336 | dequant_cublas | ok (7.8e-03) | 219.6 | 151 | 10 | 11 | 0.30 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (7.8e-03) | 99.4 | 333 | 21 | 24 | 0.66 |
| down_proj M=128 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 96.4 | 368 | 24 | 27 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 97.1 | 365 | 23 | 26 | 0.99 |
| down_proj M=128 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 97.1 | 365 | 23 | 26 | 0.99 |
| down_proj M=128 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 128.9 | 275 | 18 | 20 | 0.75 |
| down_proj M=128 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 180.2 | 197 | 13 | 14 | 0.53 |
| down_proj M=128 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 232.1 | 153 | 10 | 11 | 0.42 |
| down_proj M=128 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 111.2 | 319 | 21 | 23 | 0.87 |
| down_proj M=256 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 169.4 | 237 | 15 | 17 | 1.00 |
| down_proj M=256 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 165.5 | 243 | 16 | 18 | 1.02 |
| down_proj M=256 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 165.5 | 243 | 16 | 18 | 1.02 |
| down_proj M=256 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 197.4 | 204 | 13 | 15 | 0.86 |
| down_proj M=256 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 202.2 | 199 | 13 | 14 | 0.84 |
| down_proj M=256 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 256.9 | 156 | 10 | 11 | 0.66 |
| down_proj M=256 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 144.5 | 278 | 18 | 20 | 1.17 |
| down_proj M=384 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 243.0 | 185 | 12 | 13 | 1.00 |
| down_proj M=384 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| down_proj M=384 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 241.1 | 186 | 12 | 13 | 1.01 |
| down_proj M=384 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 268.3 | 167 | 11 | 12 | 0.91 |
| down_proj M=384 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 344.7 | 130 | 8 | 9 | 0.70 |
| down_proj M=384 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 303.5 | 148 | 10 | 11 | 0.80 |
| down_proj M=384 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 199.0 | 226 | 15 | 16 | 1.22 |
| down_proj M=512 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 321.0 | 155 | 10 | 11 | 1.00 |
| down_proj M=512 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| down_proj M=512 N=4096 K=14336 | cuda_tiled_512 | ok (1.6e-02) | 323.3 | 153 | 10 | 11 | 0.99 |
| down_proj M=512 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 341.0 | 145 | 9 | 11 | 0.94 |
| down_proj M=512 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 349.3 | 142 | 9 | 10 | 0.92 |
| down_proj M=512 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 369.9 | 134 | 9 | 10 | 0.87 |
| down_proj M=512 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 265.7 | 187 | 12 | 13 | 1.21 |
| down_proj M=1024 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 607.2 | 113 | 7 | 8 | 1.00 |
| down_proj M=1024 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| down_proj M=1024 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=1024 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 671.9 | 102 | 7 | 7 | 0.90 |
| down_proj M=1024 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 685.8 | 100 | 6 | 7 | 0.89 |
| down_proj M=1024 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 606.4 | 113 | 7 | 8 | 1.00 |
| down_proj M=1024 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 506.4 | 135 | 9 | 10 | 1.20 |
| down_proj M=2048 N=4096 K=14336 | vllm_awq_marlin | ok (1.6e-02) | 1193.6 | 89 | 6 | 6 | 1.00 |
| down_proj M=2048 N=4096 K=14336 | cuda_tiled | skipped: gemm_w4a16_tiled takes up to 256 rows | - | - | - | - | - |
| down_proj M=2048 N=4096 K=14336 | cuda_tiled_512 | skipped: gemm_w4a16_tiled takes up to 512 rows | - | - | - | - | - |
| down_proj M=2048 N=4096 K=14336 | cuda_large_128x256 | ok (1.6e-02) | 1378.2 | 77 | 5 | 6 | 0.87 |
| down_proj M=2048 N=4096 K=14336 | cuda_large_256x128 | ok (1.6e-02) | 1451.3 | 73 | 5 | 5 | 0.82 |
| down_proj M=2048 N=4096 K=14336 | dequant_cublas | ok (1.6e-02) | 1056.7 | 101 | 6 | 7 | 1.13 |
| down_proj M=2048 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 958.0 | 111 | 7 | 8 | 1.25 |

## w4a16_dequant

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj N=6144 K=4096 | tiled | ok (0.0e+00) | 52.9 | 1,201 | 77 | 87 | 1.00 |
| o_proj N=4096 K=4096 | tiled | ok (0.0e+00) | 34.6 | 1,225 | 79 | 88 | 1.00 |
| gate_up_proj N=28672 K=4096 | tiled | ok (0.0e+00) | 243.2 | 1,219 | 78 | 88 | 1.00 |
| down_proj N=4096 K=14336 | tiled | ok (0.0e+00) | 121.5 | 1,220 | 78 | 88 | 1.00 |

## awq_dequant

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj N=6144 K=4096 | vllm_triton | ok (0.0e+00) | 53.7 | 1,182 | 76 | 85 | 1.00 |
| qkv_proj N=6144 K=4096 | nano | ok (0.0e+00) | 53.1 | 1,197 | 77 | 86 | 1.01 |
| o_proj N=4096 K=4096 | vllm_triton | ok (0.0e+00) | 34.7 | 1,221 | 79 | 88 | 1.00 |
| o_proj N=4096 K=4096 | nano | ok (0.0e+00) | 34.7 | 1,218 | 78 | 88 | 1.00 |
| gate_up_proj N=28672 K=4096 | vllm_triton | ok (0.0e+00) | 243.4 | 1,217 | 78 | 88 | 1.00 |
| gate_up_proj N=28672 K=4096 | nano | ok (0.0e+00) | 243.0 | 1,220 | 78 | 88 | 1.00 |
| down_proj N=4096 K=14336 | vllm_triton | ok (0.0e+00) | 121.7 | 1,217 | 78 | 88 | 1.00 |
| down_proj N=4096 K=14336 | nano | ok (0.0e+00) | 121.7 | 1,218 | 78 | 88 | 1.00 |

## lm_head

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=128256 K=4096 | bf16_nano | ok (1.6e-02) | 712.9 | 738 | 47 | 53 | 1.00 |
| M=1 N=128256 K=4096 | int8 | ok (1.6e-02) | 369.3 | 1,425 | 92 | 103 | 1.93 |
| M=1 N=128256 K=4096 | int8_k128 | ok (1.6e-02) | 380.4 | 1,383 | 89 | 100 | 1.87 |
| M=1 N=128256 K=4096 | int8_n64 | ok (1.6e-02) | 402.0 | 1,309 | 84 | 95 | 1.77 |
| M=1 N=128256 K=4096 | cublas_bf16 | ok (1.6e-02) | 720.5 | 730 | 47 | 53 | 0.99 |
| M=4 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 721.0 | 731 | 47 | 53 | 1.00 |
| M=4 N=128256 K=4096 | int8 | ok (1.6e-02) | 375.7 | 1,402 | 90 | 101 | 1.92 |
| M=4 N=128256 K=4096 | int8_k128 | ok (1.6e-02) | 381.9 | 1,380 | 89 | 100 | 1.89 |
| M=4 N=128256 K=4096 | int8_n64 | ok (1.6e-02) | 409.0 | 1,288 | 83 | 93 | 1.76 |
| M=4 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 728.6 | 723 | 47 | 52 | 0.99 |
| M=16 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 735.8 | 720 | 46 | 52 | 1.00 |
| M=16 N=128256 K=4096 | int8 | ok (1.6e-02) | 386.2 | 1,372 | 88 | 99 | 1.91 |
| M=16 N=128256 K=4096 | int8_k128 | ok (1.6e-02) | 401.6 | 1,320 | 85 | 95 | 1.83 |
| M=16 N=128256 K=4096 | int8_n64 | ok (1.6e-02) | 401.1 | 1,321 | 85 | 95 | 1.83 |
| M=16 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 742.8 | 714 | 46 | 52 | 0.99 |
| M=64 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 778.7 | 697 | 45 | 50 | 1.00 |
| M=64 N=128256 K=4096 | int8 | ok (1.6e-02) | 641.5 | 846 | 54 | 61 | 1.21 |
| M=64 N=128256 K=4096 | int8_k128 | ok (1.6e-02) | 526.3 | 1,031 | 66 | 74 | 1.48 |
| M=64 N=128256 K=4096 | int8_n64 | ok (1.6e-02) | 712.2 | 762 | 49 | 55 | 1.09 |
| M=64 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 779.5 | 696 | 45 | 50 | 1.00 |
| M=256 N=128256 K=4096 | bf16_nano | ok (3.1e-02) | 1156.9 | 513 | 33 | 37 | 1.00 |
| M=256 N=128256 K=4096 | int8 | ok (1.6e-02) | 2526.2 | 235 | 15 | 17 | 0.46 |
| M=256 N=128256 K=4096 | int8_k128 | ok (1.6e-02) | 2000.3 | 297 | 19 | 21 | 0.58 |
| M=256 N=128256 K=4096 | int8_n64 | ok (1.6e-02) | 2752.7 | 216 | 14 | 16 | 0.42 |
| M=256 N=128256 K=4096 | cublas_bf16 | ok (3.1e-02) | 1164.1 | 510 | 33 | 37 | 0.99 |

