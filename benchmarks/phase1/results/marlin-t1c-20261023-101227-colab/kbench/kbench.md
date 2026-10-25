# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,381 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 19.1 | 681 | 44 | 49 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.2 | 586 | 38 | 42 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 25.2 | 517 | 33 | 37 | 0.76 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 22.5 | 579 | 37 | 42 | 0.85 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.9 | 725 | 47 | 52 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 18.6 | 697 | 45 | 50 | 1.02 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 684 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 37.5 | 348 | 22 | 25 | 0.51 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 21.6 | 603 | 39 | 44 | 0.88 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 23.5 | 555 | 36 | 40 | 0.81 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 21.2 | 617 | 40 | 45 | 0.90 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 18.8 | 694 | 45 | 50 | 1.02 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 318 | 20 | 23 | 0.47 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 689 | 44 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 72.6 | 181 | 12 | 13 | 0.26 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 21.4 | 614 | 40 | 44 | 0.89 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.7 | 532 | 34 | 39 | 0.77 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 21.3 | 617 | 40 | 45 | 0.90 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 18.9 | 694 | 45 | 50 | 1.01 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.2 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.2 | 627 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 141.1 | 94 | 6 | 7 | 0.15 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 22.1 | 602 | 39 | 44 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 28.3 | 471 | 30 | 34 | 0.75 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 22.1 | 601 | 39 | 44 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.2 | 694 | 45 | 50 | 1.11 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.5 | 495 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 221.7 | 61 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 31.1 | 439 | 28 | 32 | 0.89 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.3 | 338 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.2 | 339 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 24.2 | 564 | 36 | 41 | 1.14 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 45.2 | 316 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 73.6 | 194 | 12 | 14 | 0.61 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 65.1 | 220 | 14 | 16 | 0.69 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 65.2 | 219 | 14 | 16 | 0.69 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 43.9 | 325 | 21 | 24 | 1.03 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.9 | 333 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.4 | 499 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.5 | 524 | 34 | 38 | 1.05 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 19.9 | 435 | 28 | 32 | 0.87 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.8 | 515 | 33 | 37 | 1.03 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 655 | 42 | 47 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 16.5 | 527 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.6 | 303 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.5 | 498 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 28.1 | 310 | 20 | 22 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 20.0 | 435 | 28 | 31 | 0.87 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 21.8 | 401 | 26 | 29 | 0.81 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.8 | 520 | 33 | 38 | 1.04 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.5 | 529 | 34 | 38 | 1.06 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 303 | 20 | 22 | 0.61 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.5 | 502 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.7 | 177 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.1 | 513 | 33 | 37 | 1.02 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.5 | 428 | 27 | 31 | 0.85 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.1 | 513 | 33 | 37 | 1.02 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.2 | 541 | 35 | 39 | 1.08 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 304 | 20 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.8 | 451 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 92.4 | 96 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 18.2 | 490 | 31 | 35 | 1.09 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.7 | 360 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 18.2 | 490 | 31 | 35 | 1.09 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.6 | 536 | 34 | 39 | 1.19 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 308 | 20 | 22 | 0.68 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.2 | 413 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 177.2 | 52 | 3 | 4 | 0.13 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.7 | 257 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.8 | 257 | 16 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.8 | 256 | 16 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 21.7 | 423 | 27 | 31 | 1.02 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.76 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.9 | 270 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 71.4 | 136 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.5 | 169 | 11 | 12 | 0.63 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.5 | 169 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 40.0 | 242 | 16 | 18 | 0.90 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.20 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 58.8 | 1,031 | 66 | 75 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.7 | 1,015 | 65 | 73 | 0.98 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 68.8 | 881 | 57 | 64 | 0.85 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 87.5 | 693 | 45 | 50 | 0.67 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.7 | 1,015 | 65 | 73 | 0.98 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 55.9 | 1,084 | 70 | 78 | 1.05 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 177.6 | 341 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.4 | 1,025 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 119.0 | 511 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 70.1 | 868 | 56 | 63 | 0.85 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 87.1 | 698 | 45 | 51 | 0.68 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 70.0 | 868 | 56 | 63 | 0.85 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 55.8 | 1,089 | 70 | 79 | 1.06 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.1 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.7 | 1,023 | 66 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 229.2 | 266 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.1 | 859 | 55 | 62 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.5 | 690 | 44 | 50 | 0.67 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.1 | 859 | 55 | 62 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 57.6 | 1,060 | 68 | 77 | 1.04 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.3 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.2 | 945 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 458.3 | 134 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 74.4 | 828 | 53 | 60 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 91.3 | 675 | 43 | 49 | 0.71 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 74.4 | 828 | 53 | 60 | 0.88 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 62.0 | 993 | 64 | 72 | 1.05 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.1 | 336 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 72.1 | 869 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 916.7 | 68 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 137.5 | 456 | 29 | 33 | 0.52 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 118.0 | 531 | 34 | 38 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 118.1 | 531 | 34 | 38 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 77.3 | 811 | 52 | 59 | 0.93 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 216.3 | 290 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 100.9 | 642 | 41 | 46 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 312.4 | 207 | 13 | 15 | 0.32 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 204.5 | 317 | 20 | 23 | 0.49 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 204.7 | 316 | 20 | 23 | 0.49 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 112.5 | 576 | 37 | 42 | 0.90 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 262.6 | 247 | 16 | 18 | 0.38 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,303 | 84 | 94 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 35.2 | 862 | 55 | 62 | 0.66 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 37.8 | 802 | 52 | 58 | 0.62 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 39.9 | 761 | 49 | 55 | 0.58 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 35.9 | 844 | 54 | 61 | 0.65 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.9 | 982 | 63 | 71 | 0.75 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.0 | 333 | 21 | 24 | 0.26 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,307 | 84 | 95 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 63.5 | 479 | 31 | 35 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 38.2 | 796 | 51 | 58 | 0.61 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 41.5 | 733 | 47 | 53 | 0.56 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 38.3 | 795 | 51 | 58 | 0.61 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.0 | 981 | 63 | 71 | 0.75 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.4 | 333 | 21 | 24 | 0.25 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.4 | 1,304 | 84 | 94 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 160.6 | 190 | 12 | 14 | 0.15 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 39.3 | 778 | 50 | 56 | 0.60 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 43.3 | 706 | 45 | 51 | 0.54 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 39.3 | 779 | 50 | 56 | 0.60 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.8 | 961 | 62 | 70 | 0.74 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.7 | 333 | 21 | 24 | 0.26 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.6 | 1,116 | 72 | 81 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 319.6 | 97 | 6 | 7 | 0.09 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.3 | 730 | 47 | 53 | 0.65 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.6 | 649 | 42 | 47 | 0.58 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.2 | 731 | 47 | 53 | 0.66 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 33.1 | 931 | 60 | 67 | 0.83 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.6 | 337 | 22 | 24 | 0.30 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.4 | 841 | 54 | 61 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 630.4 | 50 | 3 | 4 | 0.06 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 97.8 | 322 | 21 | 23 | 0.38 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 71.1 | 442 | 28 | 32 | 0.53 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 71.2 | 442 | 28 | 32 | 0.53 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 43.9 | 716 | 46 | 52 | 0.85 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.0 | 315 | 20 | 23 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 63.0 | 518 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 238.3 | 137 | 9 | 10 | 0.26 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.4 | 248 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.6 | 248 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 76.1 | 429 | 28 | 31 | 0.83 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 98.6 | 331 | 21 | 24 | 0.64 |

## w4a16_dequant

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj N=6144 K=4096 | rows | ok (0.0e+00) | 52.5 | 1,207 | 78 | 87 | 1.00 |
| qkv_proj N=6144 K=4096 | tiled | ok (0.0e+00) | 52.5 | 1,206 | 78 | 87 | 1.00 |
| o_proj N=4096 K=4096 | rows | ok (0.0e+00) | 35.6 | 1,187 | 76 | 86 | 1.00 |
| o_proj N=4096 K=4096 | tiled | ok (0.0e+00) | 34.4 | 1,228 | 79 | 89 | 1.03 |
| gate_up_proj N=28672 K=4096 | rows | ok (0.0e+00) | 238.9 | 1,237 | 80 | 90 | 1.00 |
| gate_up_proj N=28672 K=4096 | tiled | ok (0.0e+00) | 241.6 | 1,223 | 79 | 89 | 0.99 |
| down_proj N=4096 K=14336 | rows | ok (0.0e+00) | 120.2 | 1,229 | 79 | 89 | 1.00 |
| down_proj N=4096 K=14336 | tiled | ok (0.0e+00) | 120.7 | 1,224 | 79 | 89 | 1.00 |

