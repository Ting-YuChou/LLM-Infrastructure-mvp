# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,378 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.0 | 1 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.7 | 9 | 1 | 1 | 6.37 |
| M=16 N=4096 | torch | ok (0.0e+00) | 28.5 | 9 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.4 | 114 | 7 | 8 | 11.98 |
| M=64 N=4096 | torch | ok (0.0e+00) | 30.6 | 35 | 2 | 3 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 2.8 | 384 | 25 | 28 | 11.11 |
| M=256 N=4096 | torch | ok (0.0e+00) | 42.9 | 98 | 6 | 7 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 4.2 | 1,005 | 65 | 73 | 10.27 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 257.6 | 130 | 8 | 9 | 1.00 |
| M=2048 N=4096 | triton | ok (0.0e+00) | 24.1 | 1,392 | 90 | 101 | 10.68 |

## silu_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.19 |
| M=16 N=14336 | torch | ok (0.0e+00) | 5.0 | 275 | 18 | 20 | 1.00 |
| M=16 N=14336 | triton | ok (3.1e-02) | 2.7 | 512 | 33 | 37 | 1.87 |
| M=64 N=14336 | torch | ok (0.0e+00) | 7.8 | 705 | 45 | 51 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.8 | 1,139 | 73 | 83 | 1.62 |
| M=256 N=14336 | torch | ok (0.0e+00) | 22.2 | 990 | 64 | 72 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 13.5 | 1,628 | 105 | 118 | 1.64 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 219.2 | 804 | 52 | 58 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 132.0 | 1,335 | 86 | 97 | 1.66 |

## linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | cublas | ok (7.8e-03) | 41.0 | 1,228 | 79 | 89 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | triton | ok (7.8e-03) | 42.2 | 1,192 | 77 | 87 | 0.97 |
| qkv_proj M=1 N=6144 K=4096 | cuda | ok (7.8e-03) | 40.9 | 1,231 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,232 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | triton | ok (1.3e-02) | 42.9 | 1,173 | 75 | 85 | 0.95 |
| qkv_proj M=2 N=6144 K=4096 | cuda | ok (1.3e-02) | 40.7 | 1,238 | 80 | 90 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,232 | 79 | 89 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | triton | ok (1.3e-02) | 43.4 | 1,162 | 75 | 84 | 0.94 |
| qkv_proj M=4 N=6144 K=4096 | cuda | ok (1.3e-02) | 43.2 | 1,166 | 75 | 85 | 0.95 |
| qkv_proj M=8 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.1 | 1,229 | 79 | 89 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | triton | ok (1.3e-02) | 44.2 | 1,142 | 73 | 83 | 0.93 |
| qkv_proj M=8 N=6144 K=4096 | cuda | ok (1.3e-02) | 59.2 | 852 | 55 | 62 | 0.69 |
| qkv_proj M=16 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.4 | 1,223 | 79 | 89 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | triton | ok (1.3e-02) | 45.1 | 1,122 | 72 | 81 | 0.92 |
| qkv_proj M=16 N=6144 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cublas | ok (7.8e-03) | 28.6 | 1,173 | 75 | 85 | 1.00 |
| o_proj M=1 N=4096 K=4096 | triton | ok (7.8e-03) | 31.0 | 1,082 | 70 | 79 | 0.92 |
| o_proj M=1 N=4096 K=4096 | cuda | ok (7.8e-03) | 29.2 | 1,149 | 74 | 83 | 0.98 |
| o_proj M=2 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,172 | 75 | 85 | 1.00 |
| o_proj M=2 N=4096 K=4096 | triton | ok (1.3e-02) | 31.6 | 1,061 | 68 | 77 | 0.91 |
| o_proj M=2 N=4096 K=4096 | cuda | ok (1.3e-02) | 28.5 | 1,180 | 76 | 86 | 1.01 |
| o_proj M=4 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,171 | 75 | 85 | 1.00 |
| o_proj M=4 N=4096 K=4096 | triton | ok (1.3e-02) | 31.9 | 1,054 | 68 | 77 | 0.90 |
| o_proj M=4 N=4096 K=4096 | cuda | ok (1.3e-02) | 31.4 | 1,072 | 69 | 78 | 0.92 |
| o_proj M=8 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.9 | 1,167 | 75 | 85 | 1.00 |
| o_proj M=8 N=4096 K=4096 | triton | ok (1.3e-02) | 32.2 | 1,047 | 67 | 76 | 0.90 |
| o_proj M=8 N=4096 K=4096 | cuda | ok (1.3e-02) | 48.7 | 692 | 45 | 50 | 0.59 |
| o_proj M=16 N=4096 K=4096 | cublas | ok (1.3e-02) | 29.0 | 1,168 | 75 | 85 | 1.00 |
| o_proj M=16 N=4096 K=4096 | triton | ok (1.3e-02) | 32.6 | 1,039 | 67 | 75 | 0.89 |
| o_proj M=16 N=4096 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cublas | ok (1.7e-02) | 177.8 | 1,321 | 85 | 96 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | triton | ok (1.3e-02) | 170.9 | 1,375 | 88 | 100 | 1.04 |
| gate_up_proj M=1 N=28672 K=4096 | cuda | ok (1.3e-02) | 164.9 | 1,424 | 92 | 103 | 1.08 |
| gate_up_proj M=2 N=28672 K=4096 | cublas | ok (1.3e-02) | 179.7 | 1,308 | 84 | 95 | 1.00 |
| gate_up_proj M=2 N=28672 K=4096 | triton | ok (1.3e-02) | 171.4 | 1,371 | 88 | 100 | 1.05 |
| gate_up_proj M=2 N=28672 K=4096 | cuda | ok (1.3e-02) | 167.7 | 1,401 | 90 | 102 | 1.07 |
| gate_up_proj M=4 N=28672 K=4096 | cublas | ok (1.3e-02) | 180.0 | 1,307 | 84 | 95 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | triton | ok (1.3e-02) | 172.2 | 1,366 | 88 | 99 | 1.05 |
| gate_up_proj M=4 N=28672 K=4096 | cuda | ok (1.3e-02) | 170.6 | 1,378 | 89 | 100 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cublas | ok (1.5e-02) | 181.6 | 1,297 | 83 | 94 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | triton | ok (1.5e-02) | 173.0 | 1,361 | 88 | 99 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cuda | ok (1.5e-02) | 234.5 | 1,004 | 65 | 73 | 0.77 |
| gate_up_proj M=16 N=28672 K=4096 | cublas | ok (1.5e-02) | 183.3 | 1,287 | 83 | 93 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | triton | ok (1.5e-02) | 176.2 | 1,339 | 86 | 97 | 1.04 |
| gate_up_proj M=16 N=28672 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.1 | 1,275 | 82 | 93 | 1.00 |
| down_proj M=1 N=4096 K=14336 | triton | ok (7.8e-03) | 91.3 | 1,287 | 83 | 93 | 1.01 |
| down_proj M=1 N=4096 K=14336 | cuda | ok (7.8e-03) | 88.6 | 1,326 | 85 | 96 | 1.04 |
| down_proj M=2 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.5 | 1,271 | 82 | 92 | 1.00 |
| down_proj M=2 N=4096 K=14336 | triton | ok (7.8e-03) | 92.3 | 1,273 | 82 | 92 | 1.00 |
| down_proj M=2 N=4096 K=14336 | cuda | ok (7.8e-03) | 92.7 | 1,267 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cublas | ok (7.8e-03) | 91.0 | 1,293 | 83 | 94 | 1.00 |
| down_proj M=4 N=4096 K=14336 | triton | ok (7.8e-03) | 92.9 | 1,266 | 81 | 92 | 0.98 |
| down_proj M=4 N=4096 K=14336 | cuda | ok (7.8e-03) | 98.0 | 1,200 | 77 | 87 | 0.93 |
| down_proj M=8 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.2 | 1,264 | 81 | 92 | 1.00 |
| down_proj M=8 N=4096 K=14336 | triton | ok (7.8e-03) | 93.5 | 1,259 | 81 | 91 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda | ok (7.8e-03) | 162.2 | 726 | 47 | 53 | 0.57 |
| down_proj M=16 N=4096 K=14336 | cublas | ok (1.3e-02) | 92.0 | 1,283 | 83 | 93 | 1.00 |
| down_proj M=16 N=4096 K=14336 | triton | ok (1.3e-02) | 95.2 | 1,240 | 80 | 90 | 0.97 |
| down_proj M=16 N=4096 K=14336 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| lm_head M=1 N=128256 K=4096 | cublas | ok (1.3e-02) | 721.0 | 1,458 | 94 | 106 | 1.00 |
| lm_head M=1 N=128256 K=4096 | triton | ok (1.3e-02) | 709.3 | 1,482 | 95 | 108 | 1.02 |
| lm_head M=1 N=128256 K=4096 | cuda | ok (1.3e-02) | 711.2 | 1,478 | 95 | 107 | 1.01 |
| lm_head M=2 N=128256 K=4096 | cublas | ok (1.4e-02) | 723.1 | 1,454 | 93 | 106 | 1.00 |
| lm_head M=2 N=128256 K=4096 | triton | ok (1.4e-02) | 712.2 | 1,476 | 95 | 107 | 1.02 |
| lm_head M=2 N=128256 K=4096 | cuda | ok (1.4e-02) | 722.2 | 1,455 | 94 | 106 | 1.00 |
| lm_head M=4 N=128256 K=4096 | cublas | ok (1.4e-02) | 728.7 | 1,443 | 93 | 105 | 1.00 |
| lm_head M=4 N=128256 K=4096 | triton | ok (1.4e-02) | 717.2 | 1,466 | 94 | 106 | 1.02 |
| lm_head M=4 N=128256 K=4096 | cuda | ok (1.4e-02) | 734.9 | 1,431 | 92 | 104 | 0.99 |
| lm_head M=8 N=128256 K=4096 | cublas | ok (1.5e-02) | 733.2 | 1,436 | 92 | 104 | 1.00 |
| lm_head M=8 N=128256 K=4096 | triton | ok (1.5e-02) | 722.0 | 1,458 | 94 | 106 | 1.02 |
| lm_head M=8 N=128256 K=4096 | cuda | ok (1.5e-02) | 1001.8 | 1,051 | 68 | 76 | 0.73 |
| lm_head M=16 N=128256 K=4096 | cublas | ok (1.5e-02) | 743.1 | 1,420 | 91 | 103 | 1.00 |
| lm_head M=16 N=128256 K=4096 | triton | ok (1.5e-02) | 731.5 | 1,442 | 93 | 105 | 1.02 |
| lm_head M=16 N=128256 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |

## fused_add_rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.6 | 2 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.7 | 15 | 1 | 1 | 6.62 |
| M=16 N=4096 | torch | ok (0.0e+00) | 29.7 | 18 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.7 | 194 | 12 | 14 | 10.83 |
| M=64 N=4096 | torch | ok (0.0e+00) | 32.5 | 65 | 4 | 5 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 3.2 | 648 | 42 | 47 | 10.01 |
| M=256 N=4096 | torch | ok (0.0e+00) | 45.8 | 183 | 12 | 13 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 6.1 | 1,381 | 89 | 100 | 7.53 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 297.5 | 226 | 15 | 16 | 1.00 |
| M=2048 N=4096 | triton | ok (1.6e-02) | 52.1 | 1,289 | 83 | 94 | 5.71 |

## silu_and_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 45 | 3 | 3 | 2.17 |
| M=16 N=14336 | torch | ok (0.0e+00) | 9.0 | 152 | 10 | 11 | 1.00 |
| M=16 N=14336 | triton | ok (6.2e-02) | 2.7 | 517 | 33 | 38 | 3.40 |
| M=64 N=14336 | torch | ok (0.0e+00) | 14.8 | 371 | 24 | 27 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.9 | 1,132 | 73 | 82 | 3.05 |
| M=256 N=14336 | torch | ok (0.0e+00) | 38.5 | 571 | 37 | 41 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 15.1 | 1,458 | 94 | 106 | 2.55 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 292.1 | 603 | 39 | 44 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 131.9 | 1,335 | 86 | 97 | 2.21 |

## rope_cache_write

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| tokens=1 | torch | ok (0.0e+00) | 37.6 | 1 | 0 | 0 | 1.00 |
| tokens=1 | triton | ok (1.6e-02) | 9.4 | 3 | 0 | 0 | 4.01 |
| tokens=16 | torch | ok (0.0e+00) | 43.4 | 9 | 1 | 1 | 1.00 |
| tokens=16 | triton | ok (1.6e-02) | 7.2 | 56 | 4 | 4 | 6.04 |
| tokens=64 | torch | ok (0.0e+00) | 47.2 | 34 | 2 | 2 | 1.00 |
| tokens=64 | triton | ok (3.1e-02) | 7.8 | 207 | 13 | 15 | 6.07 |
| tokens=256 | torch | ok (0.0e+00) | 63.5 | 101 | 7 | 7 | 1.00 |
| tokens=256 | triton | ok (3.1e-02) | 9.9 | 647 | 42 | 47 | 6.39 |
| tokens=2048 | torch | ok (0.0e+00) | 281.1 | 183 | 12 | 13 | 1.00 |
| tokens=2048 | triton | ok (3.1e-02) | 53.1 | 968 | 62 | 70 | 5.29 |

## paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 6.9 | 75 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.5 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 10.3 | 51 | 3 | 4 | 0.67 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.7 | 267 | 17 | 19 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.7 | 49 | 3 | 4 | 0.19 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 12.4 | 340 | 22 | 25 | 1.27 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.2 | 556 | 36 | 40 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 227.5 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 22.5 | 746 | 48 | 54 | 1.34 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.1 | 363 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 126.2 | 66 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 17.7 | 475 | 31 | 34 | 1.31 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 75.3 | 891 | 57 | 65 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 934.3 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 61.1 | 1,098 | 71 | 80 | 1.23 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 222.9 | 1,205 | 77 | 87 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3577.5 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 206.1 | 1,302 | 84 | 95 | 1.08 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.6 | 753 | 48 | 55 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 482.7 | 70 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 35.2 | 954 | 61 | 69 | 1.27 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 220.3 | 1,219 | 78 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3586.3 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 199.1 | 1,348 | 87 | 98 | 1.11 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 813.2 | 1,320 | 85 | 96 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 768.9 | 1,396 | 90 | 101 | 1.06 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 72.6 | 925 | 59 | 67 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 924.9 | 73 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 61.8 | 1,087 | 70 | 79 | 1.18 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 405.3 | 1,325 | 85 | 96 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 385.0 | 1,395 | 90 | 101 | 1.05 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1530.5 | 1,403 | 90 | 102 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1481.5 | 1,450 | 93 | 105 | 1.03 |

## paged_prefill_attention

| shape | impl | check | us | TFLOPs | % peak | x base |
|---|---|---|---:|---:|---:|---:|
| seqs=1 qlen=512 | vllm_fa2 | ok (1.6e-02) | 35.3 | 61.0 | 20 | 1.00 |
| seqs=1 qlen=512 | torch | ok (0.0e+00) | 665.5 | 3.2 | 1 | 0.05 |
| seqs=1 qlen=512 | triton | ok (1.6e-02) | 42.8 | 50.2 | 16 | 0.82 |
| seqs=1 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 84.1 | 102.2 | 33 | 1.00 |
| seqs=1 qlen=1024 | torch | ok (0.0e+00) | 2202.2 | 3.9 | 1 | 0.04 |
| seqs=1 qlen=1024 | triton | ok (1.6e-02) | 105.3 | 81.7 | 26 | 0.80 |
| seqs=1 qlen=4096 | vllm_fa2 | ok (1.6e-02) | 846.7 | 162.4 | 52 | 1.00 |
| seqs=1 qlen=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=1 qlen=4096 | triton | ok (1.6e-02) | 1021.8 | 134.5 | 43 | 0.83 |
| seqs=4 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 260.8 | 131.9 | 42 | 1.00 |
| seqs=4 qlen=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=4 qlen=1024 | triton | ok (1.6e-02) | 319.8 | 107.5 | 34 | 0.82 |
| seqs=16 qlen=256 | vllm_fa2 | ok (1.6e-02) | 113.6 | 75.9 | 24 | 1.00 |
| seqs=16 qlen=256 | torch | ok (0.0e+00) | 3455.8 | 2.5 | 1 | 0.03 |
| seqs=16 qlen=256 | triton | ok (1.6e-02) | 148.1 | 58.2 | 19 | 0.77 |

