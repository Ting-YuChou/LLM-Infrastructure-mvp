# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,379 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 19.1 | 680 | 44 | 49 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 22.4 | 581 | 37 | 42 | 0.85 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv_pf2 | ok (0.0e+00) | 23.2 | 559 | 36 | 41 | 0.82 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv_pf4 | ok (0.0e+00) | 21.0 | 619 | 40 | 45 | 0.91 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 21.0 | 619 | 40 | 45 | 0.91 |
| qkv_proj M=1 N=6144 K=4096 | cuda_wide | ok (1.6e-02) | 21.9 | 593 | 38 | 43 | 0.87 |
| qkv_proj M=1 N=6144 K=4096 | cuda_auto | ok (0.0e+00) | 18.1 | 717 | 46 | 52 | 1.05 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.1 | 316 | 20 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 684 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 45.7 | 286 | 18 | 21 | 0.42 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv_pf2 | ok (6.1e-05) | 34.4 | 380 | 24 | 28 | 0.56 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv_pf4 | ok (6.1e-05) | 36.5 | 358 | 23 | 26 | 0.52 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 21.0 | 620 | 40 | 45 | 0.91 |
| qkv_proj M=4 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 23.2 | 563 | 36 | 41 | 0.82 |
| qkv_proj M=4 N=6144 K=4096 | cuda_auto | ok (6.1e-05) | 21.1 | 620 | 40 | 45 | 0.91 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.0 | 319 | 20 | 23 | 0.47 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 689 | 44 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 71.7 | 183 | 12 | 13 | 0.27 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv_pf2 | ok (6.1e-05) | 61.2 | 215 | 14 | 16 | 0.31 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv_pf4 | ok (6.1e-05) | 67.1 | 196 | 13 | 14 | 0.28 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 21.3 | 618 | 40 | 45 | 0.90 |
| qkv_proj M=8 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 24.6 | 534 | 34 | 39 | 0.77 |
| qkv_proj M=8 N=6144 K=4096 | cuda_auto | ok (3.1e-05) | 21.2 | 619 | 40 | 45 | 0.90 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.2 | 319 | 20 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.2 | 627 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 111.2 | 120 | 8 | 9 | 0.19 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv_pf2 | ok (1.2e-04) | 119.0 | 112 | 7 | 8 | 0.18 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv_pf4 | ok (1.2e-04) | 128.6 | 103 | 7 | 8 | 0.17 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 22.1 | 603 | 39 | 44 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 28.2 | 472 | 30 | 34 | 0.75 |
| qkv_proj M=16 N=6144 K=4096 | cuda_auto | ok (7.8e-03) | 22.1 | 603 | 39 | 44 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.5 | 321 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.3 | 499 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 220.1 | 62 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv_pf2 | ok (7.8e-03) | 234.4 | 58 | 4 | 4 | 0.12 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv_pf4 | ok (7.8e-03) | 250.6 | 54 | 3 | 4 | 0.11 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 31.0 | 440 | 28 | 32 | 0.88 |
| qkv_proj M=32 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 40.1 | 340 | 22 | 25 | 0.68 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.65 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 44.9 | 318 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv_pf2 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv_pf4 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 93.0 | 154 | 10 | 11 | 0.48 |
| qkv_proj M=64 N=6144 K=4096 | cuda_wide | ok (3.1e-02) | 64.7 | 221 | 14 | 16 | 0.69 |
| qkv_proj M=64 N=6144 K=4096 | cuda_auto | ok (3.1e-02) | 64.5 | 221 | 14 | 16 | 0.70 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 334 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.5 | 496 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 16.8 | 517 | 33 | 37 | 1.04 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv_pf2 | ok (0.0e+00) | 17.3 | 501 | 32 | 36 | 1.01 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv_pf4 | ok (0.0e+00) | 17.8 | 487 | 31 | 35 | 0.98 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 16.6 | 521 | 33 | 38 | 1.05 |
| o_proj M=1 N=4096 K=4096 | cuda_wide | ok (1.6e-02) | 16.8 | 517 | 33 | 37 | 1.04 |
| o_proj M=1 N=4096 K=4096 | cuda_auto | ok (0.0e+00) | 13.6 | 639 | 41 | 46 | 1.29 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.7 | 302 | 19 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.7 | 492 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.5 | 317 | 20 | 23 | 0.64 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv_pf2 | ok (6.1e-05) | 27.9 | 312 | 20 | 23 | 0.63 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv_pf4 | ok (6.1e-05) | 29.6 | 295 | 19 | 21 | 0.60 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 16.8 | 518 | 33 | 38 | 1.05 |
| o_proj M=4 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 18.2 | 480 | 31 | 35 | 0.97 |
| o_proj M=4 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 16.8 | 518 | 33 | 38 | 1.05 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.8 | 303 | 19 | 22 | 0.62 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.6 | 498 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 58.2 | 151 | 10 | 11 | 0.30 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv_pf2 | ok (6.1e-05) | 41.1 | 213 | 14 | 15 | 0.43 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv_pf4 | ok (6.1e-05) | 50.6 | 173 | 11 | 13 | 0.35 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 17.1 | 513 | 33 | 37 | 1.03 |
| o_proj M=8 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 20.5 | 429 | 28 | 31 | 0.86 |
| o_proj M=8 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 17.1 | 513 | 33 | 37 | 1.03 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 303 | 20 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.6 | 454 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 111.9 | 80 | 5 | 6 | 0.18 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv_pf2 | ok (4.9e-04) | 79.4 | 112 | 7 | 8 | 0.25 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv_pf4 | ok (4.9e-04) | 96.7 | 92 | 6 | 7 | 0.20 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 18.2 | 491 | 32 | 36 | 1.08 |
| o_proj M=16 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 24.5 | 364 | 23 | 26 | 0.80 |
| o_proj M=16 N=4096 K=4096 | cuda_auto | ok (9.8e-04) | 18.2 | 490 | 32 | 36 | 1.08 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 307 | 20 | 22 | 0.68 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.0 | 417 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 172.1 | 53 | 3 | 4 | 0.13 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv_pf2 | ok (7.8e-03) | 155.9 | 59 | 4 | 4 | 0.14 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv_pf4 | ok (7.8e-03) | 187.9 | 49 | 3 | 4 | 0.12 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (9.8e-04) | 35.5 | 258 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 35.4 | 260 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 35.4 | 259 | 17 | 19 | 0.62 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.4 | 312 | 20 | 23 | 0.75 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.8 | 271 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_gemv_pf2 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_gemv_pf4 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 71.0 | 137 | 9 | 10 | 0.50 |
| o_proj M=64 N=4096 K=4096 | cuda_wide | ok (3.1e-02) | 57.4 | 169 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cuda_auto | ok (3.1e-02) | 57.3 | 169 | 11 | 12 | 0.62 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 323 | 21 | 23 | 1.19 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 58.6 | 1,034 | 66 | 75 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 60.0 | 1,010 | 65 | 73 | 0.98 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv_pf2 | ok (3.9e-03) | 64.2 | 944 | 61 | 68 | 0.91 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv_pf4 | ok (3.9e-03) | 70.7 | 857 | 55 | 62 | 0.83 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 68.5 | 885 | 57 | 64 | 0.86 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_wide | ok (1.6e-02) | 87.2 | 695 | 45 | 50 | 0.67 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_auto | ok (3.9e-03) | 60.0 | 1,010 | 65 | 73 | 0.98 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 177.6 | 341 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.3 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 118.6 | 513 | 33 | 37 | 0.50 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv_pf2 | ok (3.9e-03) | 123.3 | 493 | 32 | 36 | 0.48 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv_pf4 | ok (3.9e-03) | 134.5 | 452 | 29 | 33 | 0.44 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 69.6 | 874 | 56 | 63 | 0.85 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 89.7 | 678 | 44 | 49 | 0.66 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 69.6 | 874 | 56 | 63 | 0.85 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 179.4 | 339 | 22 | 25 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.6 | 1,026 | 66 | 74 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 223.2 | 274 | 18 | 20 | 0.27 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv_pf2 | ok (7.8e-03) | 238.4 | 256 | 16 | 19 | 0.25 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv_pf4 | ok (7.8e-03) | 274.6 | 222 | 14 | 16 | 0.22 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 70.5 | 867 | 56 | 63 | 0.85 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 89.0 | 687 | 44 | 50 | 0.67 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 70.5 | 867 | 56 | 63 | 0.84 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.8 | 338 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 64.8 | 951 | 61 | 69 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 446.3 | 138 | 9 | 10 | 0.15 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv_pf2 | ok (7.8e-03) | 476.9 | 129 | 8 | 9 | 0.14 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv_pf4 | ok (7.8e-03) | 549.1 | 112 | 7 | 8 | 0.12 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 74.2 | 830 | 53 | 60 | 0.87 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 90.7 | 679 | 44 | 49 | 0.71 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_auto | ok (7.8e-03) | 74.2 | 830 | 53 | 60 | 0.87 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.6 | 337 | 22 | 24 | 0.35 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.9 | 872 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 892.5 | 70 | 5 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv_pf2 | ok (7.8e-03) | 953.8 | 66 | 4 | 5 | 0.08 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv_pf4 | ok (7.8e-03) | 1098.2 | 57 | 4 | 4 | 0.07 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 137.1 | 457 | 29 | 33 | 0.52 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 117.4 | 534 | 34 | 39 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 117.3 | 534 | 34 | 39 | 0.61 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 214.0 | 293 | 19 | 21 | 0.34 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 112.9 | 574 | 37 | 42 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv_pf2 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv_pf4 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 310.8 | 208 | 13 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_wide | ok (3.1e-02) | 203.2 | 319 | 20 | 23 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_auto | ok (3.1e-02) | 203.1 | 319 | 21 | 23 | 0.56 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 265.3 | 244 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.2 | 1,304 | 84 | 95 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 36.5 | 832 | 53 | 60 | 0.64 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv_pf2 | ok (2.4e-07) | 37.2 | 816 | 52 | 59 | 0.63 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv_pf4 | ok (2.4e-07) | 36.1 | 839 | 54 | 61 | 0.64 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 37.5 | 809 | 52 | 59 | 0.62 |
| down_proj M=1 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 39.8 | 762 | 49 | 55 | 0.58 |
| down_proj M=1 N=4096 K=14336 | cuda_auto | ok (2.4e-07) | 35.5 | 854 | 55 | 62 | 0.65 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.2 | 332 | 21 | 24 | 0.25 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.3 | 1,307 | 84 | 95 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 63.0 | 483 | 31 | 35 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv_pf2 | ok (9.8e-04) | 62.8 | 485 | 31 | 35 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv_pf4 | ok (9.8e-04) | 88.2 | 345 | 22 | 25 | 0.26 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 38.0 | 801 | 52 | 58 | 0.61 |
| down_proj M=4 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 41.1 | 739 | 48 | 54 | 0.57 |
| down_proj M=4 N=4096 K=14336 | cuda_auto | ok (9.8e-04) | 38.0 | 800 | 51 | 58 | 0.61 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.8 | 332 | 21 | 24 | 0.25 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.7 | 1,292 | 83 | 94 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 156.3 | 196 | 13 | 14 | 0.15 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv_pf2 | ok (3.9e-03) | 126.4 | 242 | 16 | 18 | 0.19 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv_pf4 | ok (3.9e-03) | 155.5 | 197 | 13 | 14 | 0.15 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 39.0 | 784 | 50 | 57 | 0.61 |
| down_proj M=8 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 42.9 | 712 | 46 | 52 | 0.55 |
| down_proj M=8 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 39.0 | 784 | 50 | 57 | 0.61 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.0 | 332 | 21 | 24 | 0.26 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.5 | 1,124 | 72 | 81 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 306.9 | 101 | 6 | 7 | 0.09 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv_pf2 | ok (3.9e-03) | 254.6 | 121 | 8 | 9 | 0.11 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv_pf4 | ok (3.9e-03) | 312.0 | 99 | 6 | 7 | 0.09 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 41.9 | 736 | 47 | 53 | 0.66 |
| down_proj M=16 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 47.4 | 651 | 42 | 47 | 0.58 |
| down_proj M=16 N=4096 K=14336 | cuda_auto | ok (3.9e-03) | 41.9 | 737 | 47 | 53 | 0.66 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.0 | 335 | 22 | 24 | 0.30 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.1 | 848 | 55 | 62 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 598.5 | 53 | 3 | 4 | 0.06 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv_pf2 | ok (7.8e-03) | 503.3 | 63 | 4 | 5 | 0.07 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv_pf4 | ok (7.8e-03) | 626.8 | 50 | 3 | 4 | 0.06 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 98.7 | 319 | 20 | 23 | 0.38 |
| down_proj M=32 N=4096 K=14336 | cuda_wide | ok (1.6e-02) | 70.7 | 445 | 29 | 32 | 0.52 |
| down_proj M=32 N=4096 K=14336 | cuda_auto | ok (1.6e-02) | 70.6 | 445 | 29 | 32 | 0.52 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.3 | 313 | 20 | 23 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 62.7 | 520 | 33 | 38 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_gemv_pf2 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_gemv_pf4 | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 236.9 | 138 | 9 | 10 | 0.26 |
| down_proj M=64 N=4096 K=14336 | cuda_wide | ok (3.1e-02) | 130.7 | 250 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cuda_auto | ok (3.1e-02) | 130.8 | 250 | 16 | 18 | 0.48 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 98.8 | 330 | 21 | 24 | 0.63 |

