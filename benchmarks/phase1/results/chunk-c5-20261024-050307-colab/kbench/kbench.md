# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,387 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 18.7 | 694 | 45 | 50 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 21.9 | 594 | 38 | 43 | 0.86 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 20.2 | 643 | 41 | 46 | 0.93 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.4 | 609 | 39 | 44 | 0.88 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 17.8 | 730 | 47 | 53 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cuda_tiled | ok (1.6e-02) | 15.9 | 817 | 53 | 59 | 1.18 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.46 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.8 | 694 | 45 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 36.4 | 359 | 23 | 26 | 0.52 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 20.3 | 643 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 22.6 | 577 | 37 | 42 | 0.83 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 20.3 | 642 | 41 | 46 | 0.93 |
| qkv_proj M=4 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.9 | 819 | 53 | 59 | 1.18 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 20 | 23 | 0.46 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 18.8 | 698 | 45 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 59.1 | 222 | 14 | 16 | 0.32 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 20.5 | 642 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.3 | 542 | 35 | 39 | 0.78 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 20.5 | 641 | 41 | 46 | 0.92 |
| qkv_proj M=8 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 16.0 | 823 | 53 | 59 | 1.18 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.1 | 630 | 41 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.2 | 120 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 27.7 | 479 | 31 | 35 | 0.76 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 21.3 | 625 | 40 | 45 | 0.99 |
| qkv_proj M=16 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 15.8 | 842 | 54 | 61 | 1.34 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.2 | 501 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.3 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 30.9 | 441 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.2 | 339 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.2 | 339 | 22 | 24 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 19.5 | 698 | 45 | 50 | 1.39 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.64 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.8 | 319 | 21 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 72.4 | 197 | 13 | 14 | 0.62 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 63.6 | 225 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 34.8 | 411 | 26 | 30 | 1.29 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.04 |
| qkv_proj M=128 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 62.1 | 251 | 16 | 18 | 1.00 |
| qkv_proj M=128 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=128 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 49.6 | 315 | 20 | 23 | 1.25 |
| qkv_proj M=128 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 56.1 | 278 | 18 | 20 | 1.11 |
| qkv_proj M=256 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 99.5 | 183 | 12 | 13 | 1.00 |
| qkv_proj M=256 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| qkv_proj M=256 N=6144 K=4096 | cuda_tiled | ok (3.1e-02) | 74.1 | 246 | 16 | 18 | 1.34 |
| qkv_proj M=256 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 61.0 | 298 | 19 | 22 | 1.63 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.4 | 499 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.3 | 531 | 34 | 38 | 1.06 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 15.6 | 557 | 36 | 40 | 1.12 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.3 | 532 | 34 | 38 | 1.07 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.2 | 658 | 42 | 47 | 1.32 |
| o_proj M=1 N=4096 K=4096 | cuda_tiled | ok (1.6e-02) | 14.5 | 596 | 38 | 43 | 1.19 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.6 | 303 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.2 | 507 | 33 | 37 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.8 | 314 | 20 | 23 | 0.62 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 17.8 | 490 | 31 | 35 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 15.7 | 555 | 36 | 40 | 1.10 |
| o_proj M=4 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.5 | 602 | 39 | 43 | 1.19 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 303 | 20 | 22 | 0.60 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.2 | 510 | 33 | 37 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 49.3 | 178 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.3 | 540 | 35 | 39 | 1.06 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.1 | 437 | 28 | 32 | 0.86 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.3 | 540 | 35 | 39 | 1.06 |
| o_proj M=8 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.5 | 605 | 39 | 44 | 1.19 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 305 | 20 | 22 | 0.60 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.4 | 460 | 30 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 91.7 | 97 | 6 | 7 | 0.21 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.4 | 365 | 23 | 26 | 0.79 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.4 | 512 | 33 | 37 | 1.11 |
| o_proj M=16 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 14.8 | 600 | 39 | 43 | 1.30 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 21.7 | 423 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 175.8 | 52 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 259 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.6 | 258 | 17 | 19 | 0.61 |
| o_proj M=32 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 18.5 | 496 | 32 | 36 | 1.17 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.4 | 313 | 20 | 23 | 0.74 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.3 | 275 | 18 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 70.4 | 138 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.1 | 170 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 29.9 | 325 | 21 | 23 | 1.18 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 324 | 21 | 23 | 1.18 |
| o_proj M=128 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 52.6 | 204 | 13 | 15 | 1.00 |
| o_proj M=128 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=128 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 38.8 | 277 | 18 | 20 | 1.35 |
| o_proj M=128 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 259 | 17 | 19 | 1.27 |
| o_proj M=256 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 71.4 | 180 | 12 | 13 | 1.00 |
| o_proj M=256 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| o_proj M=256 N=4096 K=4096 | cuda_tiled | ok (3.1e-02) | 55.7 | 231 | 15 | 17 | 1.28 |
| o_proj M=256 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 54.2 | 237 | 15 | 17 | 1.32 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 58.9 | 1,029 | 66 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 59.4 | 1,020 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 65.4 | 927 | 60 | 67 | 0.90 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 85.9 | 706 | 45 | 51 | 0.69 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 59.4 | 1,021 | 66 | 74 | 0.99 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_tiled | ok (1.6e-02) | 51.8 | 1,170 | 75 | 84 | 1.14 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.2 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.7 | 1,019 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.5 | 513 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 66.3 | 918 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 86.2 | 705 | 45 | 51 | 0.69 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 66.3 | 918 | 59 | 66 | 0.90 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 52.7 | 1,154 | 74 | 83 | 1.13 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.9 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.0 | 1,018 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 226.8 | 269 | 17 | 19 | 0.26 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 67.2 | 910 | 58 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 88.1 | 694 | 45 | 50 | 0.68 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 67.1 | 910 | 59 | 66 | 0.89 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 53.4 | 1,143 | 74 | 82 | 1.12 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.3 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.1 | 946 | 61 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 453.5 | 136 | 9 | 10 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 71.6 | 861 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 90.8 | 679 | 44 | 49 | 0.72 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 71.5 | 861 | 55 | 62 | 0.91 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 54.6 | 1,129 | 73 | 81 | 1.19 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.9 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.7 | 874 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 907.2 | 69 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 135.6 | 462 | 30 | 33 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 113.3 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 113.4 | 553 | 36 | 40 | 0.63 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 60.5 | 1,035 | 67 | 75 | 1.18 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.7 | 292 | 19 | 21 | 0.33 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 111.7 | 580 | 37 | 42 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 308.3 | 210 | 14 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 197.2 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 197.4 | 328 | 21 | 24 | 0.57 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 88.0 | 736 | 47 | 53 | 1.27 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.0 | 244 | 16 | 18 | 0.42 |
| gate_up_proj M=128 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 170.4 | 405 | 26 | 29 | 1.00 |
| gate_up_proj M=128 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=128 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 157.5 | 438 | 28 | 32 | 1.08 |
| gate_up_proj M=128 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 270.3 | 255 | 16 | 18 | 0.63 |
| gate_up_proj M=256 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 325.3 | 238 | 15 | 17 | 1.00 |
| gate_up_proj M=256 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| gate_up_proj M=256 N=28672 K=4096 | cuda_tiled | ok (3.1e-02) | 296.3 | 261 | 17 | 19 | 1.10 |
| gate_up_proj M=256 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 320.0 | 242 | 16 | 17 | 1.02 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.4 | 907 | 58 | 65 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 47.4 | 639 | 41 | 46 | 0.70 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 41.2 | 736 | 47 | 53 | 0.81 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 45.5 | 666 | 43 | 48 | 0.73 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 38.2 | 793 | 51 | 57 | 0.87 |
| down_proj M=1 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.4 | 998 | 64 | 72 | 1.10 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.7 | 330 | 21 | 24 | 0.36 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.5 | 908 | 58 | 65 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 65.8 | 463 | 30 | 33 | 0.51 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 41.8 | 728 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.0 | 647 | 42 | 47 | 0.71 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 41.7 | 729 | 47 | 53 | 0.80 |
| down_proj M=4 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.6 | 993 | 64 | 72 | 1.09 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.3 | 330 | 21 | 24 | 0.36 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 33.6 | 909 | 58 | 66 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 176.4 | 173 | 11 | 12 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 42.8 | 714 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 48.5 | 631 | 41 | 45 | 0.69 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 42.8 | 714 | 46 | 51 | 0.79 |
| down_proj M=8 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 30.9 | 989 | 64 | 71 | 1.09 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.2 | 331 | 21 | 24 | 0.36 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 36.8 | 838 | 54 | 60 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 342.1 | 90 | 6 | 7 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.81 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 53.1 | 582 | 37 | 42 | 0.69 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 45.2 | 683 | 44 | 49 | 0.82 |
| down_proj M=16 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 31.9 | 968 | 62 | 70 | 1.16 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.8 | 333 | 21 | 24 | 0.40 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 48.1 | 654 | 42 | 47 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 664.0 | 47 | 3 | 3 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 104.5 | 301 | 19 | 22 | 0.46 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 72.6 | 433 | 28 | 31 | 0.66 |
| down_proj M=32 N=4096 K=14336 | cuda_tiled | ok (1.6e-02) | 36.5 | 863 | 55 | 62 | 1.32 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.4 | 313 | 20 | 23 | 0.48 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 81.4 | 401 | 26 | 29 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 237.4 | 137 | 9 | 10 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 131.1 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 131.1 | 249 | 16 | 18 | 0.62 |
| down_proj M=64 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 59.5 | 549 | 35 | 40 | 1.37 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.5 | 328 | 21 | 24 | 0.82 |
| down_proj M=128 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 100.7 | 348 | 22 | 25 | 1.00 |
| down_proj M=128 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=128 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 92.8 | 377 | 24 | 27 | 1.09 |
| down_proj M=128 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 111.7 | 313 | 20 | 23 | 0.90 |
| down_proj M=256 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 168.3 | 236 | 15 | 17 | 1.00 |
| down_proj M=256 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_mma | skipped: the mma kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_wide | skipped: the wide kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_auto | skipped: the auto kernel takes up to 64 rows | - | - | - | - | - |
| down_proj M=256 N=4096 K=14336 | cuda_tiled | ok (3.1e-02) | 152.3 | 261 | 17 | 19 | 1.10 |
| down_proj M=256 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 144.7 | 274 | 18 | 20 | 1.16 |

