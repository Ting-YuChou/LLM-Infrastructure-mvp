# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,378 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

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

