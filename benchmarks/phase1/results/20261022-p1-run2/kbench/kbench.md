# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,377 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.0 | 1 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.9 | 9 | 1 | 1 | 5.90 |
| M=16 N=4096 | torch | ok (0.0e+00) | 28.5 | 9 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.4 | 114 | 7 | 8 | 11.97 |
| M=64 N=4096 | torch | ok (0.0e+00) | 30.6 | 35 | 2 | 3 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 2.8 | 383 | 25 | 28 | 11.09 |
| M=256 N=4096 | torch | ok (0.0e+00) | 42.9 | 98 | 6 | 7 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 4.2 | 1,003 | 65 | 73 | 10.26 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 257.7 | 130 | 8 | 9 | 1.00 |
| M=2048 N=4096 | triton | ok (0.0e+00) | 24.1 | 1,392 | 90 | 101 | 10.69 |

## silu_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.19 |
| M=16 N=14336 | torch | ok (0.0e+00) | 5.0 | 274 | 18 | 20 | 1.00 |
| M=16 N=14336 | triton | ok (3.1e-02) | 2.7 | 512 | 33 | 37 | 1.87 |
| M=64 N=14336 | torch | ok (0.0e+00) | 7.8 | 704 | 45 | 51 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.8 | 1,138 | 73 | 83 | 1.62 |
| M=256 N=14336 | torch | ok (0.0e+00) | 22.2 | 991 | 64 | 72 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 13.5 | 1,631 | 105 | 118 | 1.65 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 219.2 | 804 | 52 | 58 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 132.0 | 1,335 | 86 | 97 | 1.66 |

## linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | cublas | ok (7.8e-03) | 41.0 | 1,229 | 79 | 89 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | triton | ok (7.8e-03) | 42.2 | 1,192 | 77 | 87 | 0.97 |
| qkv_proj M=1 N=6144 K=4096 | cuda | ok (7.8e-03) | 41.1 | 1,226 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,233 | 79 | 90 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | triton | ok (1.3e-02) | 42.9 | 1,173 | 75 | 85 | 0.95 |
| qkv_proj M=2 N=6144 K=4096 | cuda | ok (1.3e-02) | 41.5 | 1,213 | 78 | 88 | 0.98 |
| qkv_proj M=4 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,231 | 79 | 89 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | triton | ok (1.3e-02) | 43.4 | 1,161 | 75 | 84 | 0.94 |
| qkv_proj M=4 N=6144 K=4096 | cuda | ok (1.3e-02) | 42.6 | 1,183 | 76 | 86 | 0.96 |
| qkv_proj M=8 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.1 | 1,228 | 79 | 89 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | triton | ok (1.3e-02) | 44.2 | 1,143 | 73 | 83 | 0.93 |
| qkv_proj M=8 N=6144 K=4096 | cuda | ok (1.3e-02) | 49.4 | 1,022 | 66 | 74 | 0.83 |
| qkv_proj M=16 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.4 | 1,223 | 79 | 89 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | triton | ok (1.3e-02) | 45.2 | 1,122 | 72 | 81 | 0.92 |
| qkv_proj M=16 N=6144 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cublas | ok (7.8e-03) | 28.5 | 1,177 | 76 | 86 | 1.00 |
| o_proj M=1 N=4096 K=4096 | triton | ok (7.8e-03) | 31.0 | 1,083 | 70 | 79 | 0.92 |
| o_proj M=1 N=4096 K=4096 | cuda | ok (7.8e-03) | 31.1 | 1,079 | 69 | 78 | 0.92 |
| o_proj M=2 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.6 | 1,173 | 75 | 85 | 1.00 |
| o_proj M=2 N=4096 K=4096 | triton | ok (1.3e-02) | 31.7 | 1,061 | 68 | 77 | 0.90 |
| o_proj M=2 N=4096 K=4096 | cuda | ok (1.3e-02) | 32.0 | 1,050 | 68 | 76 | 0.90 |
| o_proj M=4 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,171 | 75 | 85 | 1.00 |
| o_proj M=4 N=4096 K=4096 | triton | ok (1.3e-02) | 31.9 | 1,053 | 68 | 76 | 0.90 |
| o_proj M=4 N=4096 K=4096 | cuda | ok (1.3e-02) | 31.7 | 1,061 | 68 | 77 | 0.91 |
| o_proj M=8 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.9 | 1,167 | 75 | 85 | 1.00 |
| o_proj M=8 N=4096 K=4096 | triton | ok (1.3e-02) | 32.2 | 1,046 | 67 | 76 | 0.90 |
| o_proj M=8 N=4096 K=4096 | cuda | ok (1.3e-02) | 39.6 | 850 | 55 | 62 | 0.73 |
| o_proj M=16 N=4096 K=4096 | cublas | ok (1.3e-02) | 29.1 | 1,163 | 75 | 84 | 1.00 |
| o_proj M=16 N=4096 K=4096 | triton | ok (1.3e-02) | 32.6 | 1,036 | 67 | 75 | 0.89 |
| o_proj M=16 N=4096 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cublas | ok (1.7e-02) | 177.9 | 1,320 | 85 | 96 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | triton | ok (1.3e-02) | 171.2 | 1,373 | 88 | 100 | 1.04 |
| gate_up_proj M=1 N=28672 K=4096 | cuda | ok (1.3e-02) | 167.7 | 1,401 | 90 | 102 | 1.06 |
| gate_up_proj M=2 N=28672 K=4096 | cublas | ok (1.3e-02) | 179.5 | 1,310 | 84 | 95 | 1.00 |
| gate_up_proj M=2 N=28672 K=4096 | triton | ok (1.3e-02) | 171.4 | 1,371 | 88 | 100 | 1.05 |
| gate_up_proj M=2 N=28672 K=4096 | cuda | ok (1.3e-02) | 167.7 | 1,402 | 90 | 102 | 1.07 |
| gate_up_proj M=4 N=28672 K=4096 | cublas | ok (1.3e-02) | 179.6 | 1,309 | 84 | 95 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | triton | ok (1.3e-02) | 172.5 | 1,364 | 88 | 99 | 1.04 |
| gate_up_proj M=4 N=28672 K=4096 | cuda | ok (1.3e-02) | 171.3 | 1,373 | 88 | 100 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cublas | ok (1.5e-02) | 181.6 | 1,296 | 83 | 94 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | triton | ok (1.5e-02) | 173.1 | 1,360 | 87 | 99 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cuda | ok (1.5e-02) | 189.4 | 1,243 | 80 | 90 | 0.96 |
| gate_up_proj M=16 N=28672 K=4096 | cublas | ok (1.5e-02) | 182.5 | 1,293 | 83 | 94 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | triton | ok (1.5e-02) | 176.2 | 1,339 | 86 | 97 | 1.04 |
| gate_up_proj M=16 N=28672 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.0 | 1,277 | 82 | 93 | 1.00 |
| down_proj M=1 N=4096 K=14336 | triton | ok (7.8e-03) | 91.2 | 1,288 | 83 | 94 | 1.01 |
| down_proj M=1 N=4096 K=14336 | cuda | ok (7.8e-03) | 99.4 | 1,181 | 76 | 86 | 0.93 |
| down_proj M=2 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.2 | 1,261 | 81 | 92 | 1.00 |
| down_proj M=2 N=4096 K=14336 | triton | ok (7.8e-03) | 92.3 | 1,273 | 82 | 92 | 1.01 |
| down_proj M=2 N=4096 K=14336 | cuda | ok (7.8e-03) | 105.2 | 1,117 | 72 | 81 | 0.89 |
| down_proj M=4 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.7 | 1,268 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | triton | ok (7.8e-03) | 92.8 | 1,267 | 81 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda | ok (7.8e-03) | 100.9 | 1,165 | 75 | 85 | 0.92 |
| down_proj M=8 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.2 | 1,263 | 81 | 92 | 1.00 |
| down_proj M=8 N=4096 K=14336 | triton | ok (7.8e-03) | 93.6 | 1,258 | 81 | 91 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda | ok (7.8e-03) | 124.5 | 945 | 61 | 69 | 0.75 |
| down_proj M=16 N=4096 K=14336 | cublas | ok (1.3e-02) | 93.9 | 1,257 | 81 | 91 | 1.00 |
| down_proj M=16 N=4096 K=14336 | triton | ok (1.3e-02) | 95.1 | 1,241 | 80 | 90 | 0.99 |
| down_proj M=16 N=4096 K=14336 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| lm_head M=1 N=128256 K=4096 | cublas | ok (1.3e-02) | 720.7 | 1,458 | 94 | 106 | 1.00 |
| lm_head M=1 N=128256 K=4096 | triton | ok (1.3e-02) | 709.3 | 1,482 | 95 | 108 | 1.02 |
| lm_head M=1 N=128256 K=4096 | cuda | ok (1.3e-02) | 719.4 | 1,461 | 94 | 106 | 1.00 |
| lm_head M=2 N=128256 K=4096 | cublas | ok (1.4e-02) | 723.0 | 1,454 | 93 | 106 | 1.00 |
| lm_head M=2 N=128256 K=4096 | triton | ok (1.4e-02) | 712.3 | 1,476 | 95 | 107 | 1.02 |
| lm_head M=2 N=128256 K=4096 | cuda | ok (1.4e-02) | 722.1 | 1,456 | 94 | 106 | 1.00 |
| lm_head M=4 N=128256 K=4096 | cublas | ok (1.4e-02) | 728.3 | 1,444 | 93 | 105 | 1.00 |
| lm_head M=4 N=128256 K=4096 | triton | ok (1.4e-02) | 717.1 | 1,467 | 94 | 107 | 1.02 |
| lm_head M=4 N=128256 K=4096 | cuda | ok (1.4e-02) | 731.8 | 1,437 | 92 | 104 | 1.00 |
| lm_head M=8 N=128256 K=4096 | cublas | ok (1.5e-02) | 733.0 | 1,436 | 92 | 104 | 1.00 |
| lm_head M=8 N=128256 K=4096 | triton | ok (1.5e-02) | 721.8 | 1,459 | 94 | 106 | 1.02 |
| lm_head M=8 N=128256 K=4096 | cuda | ok (1.5e-02) | 787.4 | 1,337 | 86 | 97 | 0.93 |
| lm_head M=16 N=128256 K=4096 | cublas | ok (1.5e-02) | 742.7 | 1,420 | 91 | 103 | 1.00 |
| lm_head M=16 N=128256 K=4096 | triton | ok (1.5e-02) | 731.7 | 1,442 | 93 | 105 | 1.01 |
| lm_head M=16 N=128256 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |

## fused_add_rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.6 | 2 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.7 | 15 | 1 | 1 | 6.61 |
| M=16 N=4096 | torch | ok (0.0e+00) | 29.7 | 18 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.7 | 194 | 12 | 14 | 10.81 |
| M=64 N=4096 | torch | ok (0.0e+00) | 32.3 | 65 | 4 | 5 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 3.3 | 647 | 42 | 47 | 9.94 |
| M=256 N=4096 | torch | ok (0.0e+00) | 46.0 | 182 | 12 | 13 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 6.1 | 1,377 | 89 | 100 | 7.55 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 298.1 | 225 | 14 | 16 | 1.00 |
| M=2048 N=4096 | triton | ok (1.6e-02) | 52.0 | 1,290 | 83 | 94 | 5.73 |

## silu_and_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 45 | 3 | 3 | 2.17 |
| M=16 N=14336 | torch | ok (0.0e+00) | 9.0 | 153 | 10 | 11 | 1.00 |
| M=16 N=14336 | triton | ok (6.2e-02) | 2.7 | 517 | 33 | 38 | 3.39 |
| M=64 N=14336 | torch | ok (0.0e+00) | 14.8 | 371 | 24 | 27 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.9 | 1,123 | 72 | 82 | 3.02 |
| M=256 N=14336 | torch | ok (0.0e+00) | 38.5 | 571 | 37 | 41 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 15.0 | 1,472 | 95 | 107 | 2.58 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 292.3 | 603 | 39 | 44 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 131.9 | 1,336 | 86 | 97 | 2.22 |

## rope_cache_write

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| tokens=1 | torch | ok (0.0e+00) | 37.9 | 1 | 0 | 0 | 1.00 |
| tokens=1 | triton | ok (1.6e-02) | 9.3 | 3 | 0 | 0 | 4.07 |
| tokens=16 | torch | ok (0.0e+00) | 43.4 | 9 | 1 | 1 | 1.00 |
| tokens=16 | triton | ok (1.6e-02) | 7.2 | 56 | 4 | 4 | 6.03 |
| tokens=64 | torch | ok (0.0e+00) | 47.3 | 34 | 2 | 2 | 1.00 |
| tokens=64 | triton | ok (3.1e-02) | 7.8 | 207 | 13 | 15 | 6.10 |
| tokens=256 | torch | ok (0.0e+00) | 63.6 | 101 | 6 | 7 | 1.00 |
| tokens=256 | triton | ok (3.1e-02) | 9.9 | 647 | 42 | 47 | 6.40 |
| tokens=2048 | torch | ok (0.0e+00) | 281.1 | 183 | 12 | 13 | 1.00 |
| tokens=2048 | triton | ok (3.1e-02) | 53.1 | 968 | 62 | 70 | 5.30 |

## paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 6.9 | 76 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.5 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 12.0 | 44 | 3 | 3 | 0.58 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.7 | 267 | 17 | 19 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.9 | 49 | 3 | 4 | 0.18 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 13.6 | 308 | 20 | 22 | 1.15 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.2 | 555 | 36 | 40 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 228.1 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 22.4 | 750 | 48 | 54 | 1.35 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.2 | 361 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 126.4 | 66 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 18.1 | 463 | 30 | 34 | 1.28 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 75.7 | 887 | 57 | 64 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 933.6 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 64.5 | 1,040 | 67 | 76 | 1.17 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 222.5 | 1,206 | 78 | 88 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3576.7 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 221.7 | 1,211 | 78 | 88 | 1.00 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.4 | 755 | 49 | 55 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 482.8 | 70 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 33.7 | 996 | 64 | 72 | 1.32 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 221.6 | 1,211 | 78 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3586.1 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 222.4 | 1,207 | 78 | 88 | 1.00 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 813.2 | 1,320 | 85 | 96 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 860.7 | 1,247 | 80 | 91 | 0.94 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 72.5 | 925 | 59 | 67 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 925.3 | 73 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 60.0 | 1,118 | 72 | 81 | 1.21 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 404.0 | 1,329 | 85 | 97 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 388.3 | 1,383 | 89 | 100 | 1.04 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1525.2 | 1,408 | 91 | 102 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1512.1 | 1,420 | 91 | 103 | 1.01 |

## paged_prefill_attention

| shape | impl | check | us | TFLOPs | % peak | x base |
|---|---|---|---:|---:|---:|---:|
| seqs=1 qlen=512 | vllm_fa2 | ok (1.6e-02) | 35.2 | 61.1 | 20 | 1.00 |
| seqs=1 qlen=512 | torch | ok (0.0e+00) | 665.5 | 3.2 | 1 | 0.05 |
| seqs=1 qlen=512 | triton | ok (1.6e-02) | 50.2 | 42.9 | 14 | 0.70 |
| seqs=1 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 84.1 | 102.3 | 33 | 1.00 |
| seqs=1 qlen=1024 | torch | ok (0.0e+00) | 2203.2 | 3.9 | 1 | 0.04 |
| seqs=1 qlen=1024 | triton | ok (1.6e-02) | 129.6 | 66.3 | 21 | 0.65 |
| seqs=1 qlen=4096 | vllm_fa2 | ok (1.6e-02) | 847.1 | 162.3 | 52 | 1.00 |
| seqs=1 qlen=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=1 qlen=4096 | triton | ok (1.6e-02) | 1336.5 | 102.9 | 33 | 0.63 |
| seqs=4 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 260.9 | 131.8 | 42 | 1.00 |
| seqs=4 qlen=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=4 qlen=1024 | triton | ok (1.6e-02) | 402.7 | 85.4 | 27 | 0.65 |
| seqs=16 qlen=256 | vllm_fa2 | ok (1.6e-02) | 113.7 | 75.9 | 24 | 1.00 |
| seqs=16 qlen=256 | torch | ok (0.0e+00) | 3456.0 | 2.5 | 1 | 0.03 |
| seqs=16 qlen=256 | triton | ok (1.6e-02) | 168.6 | 51.2 | 16 | 0.67 |

