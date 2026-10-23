# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,380 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 19.1 | 681 | 44 | 49 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.2 | 586 | 38 | 42 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 23.8 | 547 | 35 | 40 | 0.80 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.5 | 605 | 39 | 44 | 0.89 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 18.0 | 723 | 46 | 52 | 1.06 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 17.5 | 741 | 48 | 54 | 1.09 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 683 | 44 | 49 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 46.2 | 283 | 18 | 20 | 0.41 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.5 | 638 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.8 | 573 | 37 | 42 | 0.84 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.4 | 640 | 41 | 46 | 0.94 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 17.6 | 742 | 48 | 54 | 1.09 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 318 | 20 | 23 | 0.47 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.2 | 686 | 44 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 72.7 | 181 | 12 | 13 | 0.26 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.6 | 639 | 41 | 46 | 0.93 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.4 | 539 | 35 | 39 | 0.79 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.93 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 17.9 | 734 | 47 | 53 | 1.07 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.3 | 318 | 20 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.2 | 627 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 143.2 | 93 | 6 | 7 | 0.15 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 624 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 28.0 | 476 | 31 | 34 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.4 | 622 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 18.2 | 730 | 47 | 53 | 1.16 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 320 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.5 | 495 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 221.8 | 61 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.8 | 443 | 28 | 32 | 0.89 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.4 | 337 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.5 | 337 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 23.0 | 594 | 38 | 43 | 1.20 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 45.2 | 316 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 87.9 | 163 | 10 | 12 | 0.51 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 64.1 | 223 | 14 | 16 | 0.71 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 64.2 | 223 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 40.1 | 356 | 23 | 26 | 1.13 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.9 | 333 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.4 | 498 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.5 | 524 | 34 | 38 | 1.05 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 554 | 36 | 40 | 1.11 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.4 | 527 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.3 | 654 | 42 | 47 | 1.31 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 15.8 | 550 | 35 | 40 | 1.10 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.6 | 303 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.5 | 497 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 28.1 | 310 | 20 | 22 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 18.7 | 466 | 30 | 34 | 0.94 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.9 | 486 | 31 | 35 | 0.98 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.8 | 552 | 36 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.8 | 550 | 35 | 40 | 1.11 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 303 | 20 | 22 | 0.61 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.6 | 498 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 54.5 | 161 | 10 | 12 | 0.32 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 506 | 33 | 37 | 1.02 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.3 | 433 | 28 | 31 | 0.87 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.2 | 541 | 35 | 39 | 1.09 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 15.6 | 563 | 36 | 41 | 1.13 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 304 | 20 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.6 | 454 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 112.7 | 79 | 5 | 6 | 0.17 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.6 | 505 | 32 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.5 | 364 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.6 | 506 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 16.1 | 555 | 36 | 40 | 1.22 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 308 | 20 | 22 | 0.68 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.2 | 413 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 177.0 | 52 | 3 | 4 | 0.13 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.7 | 257 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.9 | 256 | 16 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.9 | 256 | 16 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 20.6 | 446 | 29 | 32 | 1.08 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.76 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.9 | 270 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 89.5 | 108 | 7 | 8 | 0.40 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 56.9 | 170 | 11 | 12 | 0.63 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.0 | 170 | 11 | 12 | 0.63 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 33.5 | 289 | 19 | 21 | 1.07 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 324 | 21 | 23 | 1.20 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.1 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.6 | 1,017 | 65 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.6 | 924 | 59 | 67 | 0.90 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 84.5 | 718 | 46 | 52 | 0.70 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.6 | 1,018 | 65 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 52.4 | 1,157 | 74 | 84 | 1.13 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.1 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.1 | 1,029 | 66 | 75 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 119.4 | 509 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.2 | 918 | 59 | 67 | 0.89 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.8 | 701 | 45 | 51 | 0.68 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 917 | 59 | 66 | 0.89 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.7 | 1,132 | 73 | 82 | 1.10 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.6 | 339 | 22 | 25 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.7 | 1,023 | 66 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 229.2 | 267 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.2 | 909 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.6 | 689 | 44 | 50 | 0.67 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.4 | 907 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.7 | 1,117 | 72 | 81 | 1.09 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.5 | 338 | 22 | 25 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.3 | 943 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 458.5 | 134 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.6 | 861 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 90.4 | 681 | 44 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.5 | 861 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 62.6 | 985 | 63 | 71 | 1.04 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.4 | 338 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 72.1 | 869 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 917.0 | 68 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.4 | 463 | 30 | 34 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.6 | 552 | 35 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.7 | 551 | 35 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 72.1 | 869 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 215.6 | 291 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.6 | 570 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 309.6 | 209 | 13 | 15 | 0.37 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 198.4 | 326 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 198.1 | 327 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 105.5 | 614 | 39 | 44 | 1.08 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 264.6 | 245 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,300 | 84 | 94 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 35.5 | 854 | 55 | 62 | 0.66 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 34.3 | 884 | 57 | 64 | 0.68 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 37.4 | 811 | 52 | 59 | 0.62 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 35.4 | 857 | 55 | 62 | 0.66 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 28.8 | 1,052 | 68 | 76 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.6 | 327 | 21 | 24 | 0.25 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,305 | 84 | 95 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 63.5 | 479 | 31 | 35 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 34.9 | 873 | 56 | 63 | 0.67 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 38.8 | 784 | 50 | 57 | 0.60 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 34.8 | 873 | 56 | 63 | 0.67 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 28.9 | 1,052 | 68 | 76 | 0.81 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.8 | 328 | 21 | 24 | 0.25 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,315 | 85 | 95 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 159.6 | 192 | 12 | 14 | 0.15 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 36.1 | 847 | 54 | 61 | 0.64 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 40.7 | 751 | 48 | 54 | 0.57 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 36.1 | 846 | 54 | 61 | 0.64 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.0 | 1,020 | 66 | 74 | 0.78 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.2 | 328 | 21 | 24 | 0.25 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.5 | 1,122 | 72 | 81 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 317.1 | 97 | 6 | 7 | 0.09 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 39.4 | 783 | 50 | 57 | 0.70 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.9 | 672 | 43 | 49 | 0.60 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 39.4 | 784 | 50 | 57 | 0.70 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.0 | 996 | 64 | 72 | 0.89 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.5 | 330 | 21 | 24 | 0.29 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.4 | 841 | 54 | 61 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 639.8 | 49 | 3 | 4 | 0.06 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 99.5 | 316 | 20 | 23 | 0.38 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 68.8 | 457 | 29 | 33 | 0.54 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 68.5 | 459 | 30 | 33 | 0.55 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 41.4 | 760 | 49 | 55 | 0.90 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 102.6 | 307 | 20 | 22 | 0.36 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 63.1 | 517 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 234.8 | 139 | 9 | 10 | 0.27 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 130.0 | 251 | 16 | 18 | 0.49 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 130.6 | 250 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 73.1 | 446 | 29 | 32 | 0.86 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 98.7 | 331 | 21 | 24 | 0.64 |

## w4a16_dequant

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj N=6144 K=4096 | rows | ok (0.0e+00) | 52.4 | 1,208 | 78 | 88 | 1.00 |
| qkv_proj N=6144 K=4096 | tiled | ok (0.0e+00) | 52.6 | 1,204 | 77 | 87 | 1.00 |
| o_proj N=4096 K=4096 | rows | ok (0.0e+00) | 35.5 | 1,188 | 76 | 86 | 1.00 |
| o_proj N=4096 K=4096 | tiled | ok (0.0e+00) | 34.4 | 1,228 | 79 | 89 | 1.03 |
| gate_up_proj N=28672 K=4096 | rows | ok (0.0e+00) | 238.7 | 1,238 | 80 | 90 | 1.00 |
| gate_up_proj N=28672 K=4096 | tiled | ok (0.0e+00) | 242.2 | 1,220 | 78 | 88 | 0.99 |
| down_proj N=4096 K=14336 | rows | ok (0.0e+00) | 120.2 | 1,229 | 79 | 89 | 1.00 |
| down_proj N=4096 K=14336 | tiled | ok (0.0e+00) | 120.8 | 1,223 | 79 | 89 | 1.00 |

