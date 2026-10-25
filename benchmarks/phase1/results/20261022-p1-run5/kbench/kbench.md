# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,387 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.0 | 1 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.9 | 9 | 1 | 1 | 5.90 |
| M=16 N=4096 | torch | ok (0.0e+00) | 28.6 | 9 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.4 | 113 | 7 | 8 | 11.99 |
| M=64 N=4096 | torch | ok (0.0e+00) | 30.7 | 34 | 2 | 2 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 2.8 | 383 | 25 | 28 | 11.11 |
| M=256 N=4096 | torch | ok (0.0e+00) | 43.0 | 98 | 6 | 7 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 4.2 | 1,003 | 65 | 72 | 10.27 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 257.1 | 131 | 8 | 9 | 1.00 |
| M=2048 N=4096 | triton | ok (0.0e+00) | 24.1 | 1,390 | 89 | 100 | 10.65 |

## silu_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.19 |
| M=16 N=14336 | torch | ok (0.0e+00) | 5.0 | 274 | 18 | 20 | 1.00 |
| M=16 N=14336 | triton | ok (3.1e-02) | 2.7 | 512 | 33 | 37 | 1.87 |
| M=64 N=14336 | torch | ok (0.0e+00) | 7.8 | 704 | 45 | 51 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.8 | 1,137 | 73 | 82 | 1.62 |
| M=256 N=14336 | torch | ok (0.0e+00) | 22.2 | 993 | 64 | 72 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 13.5 | 1,628 | 105 | 117 | 1.64 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 219.3 | 803 | 52 | 58 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 132.1 | 1,334 | 86 | 96 | 1.66 |

## linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | cublas | ok (7.8e-03) | 40.9 | 1,230 | 79 | 89 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | triton | ok (7.8e-03) | 41.4 | 1,216 | 78 | 88 | 0.99 |
| qkv_proj M=1 N=6144 K=4096 | cuda | ok (7.8e-03) | 40.9 | 1,231 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,232 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | triton | ok (1.3e-02) | 41.6 | 1,211 | 78 | 87 | 0.98 |
| qkv_proj M=2 N=6144 K=4096 | cuda | ok (1.3e-02) | 40.7 | 1,238 | 80 | 89 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,232 | 79 | 89 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | triton | ok (1.3e-02) | 41.9 | 1,204 | 77 | 87 | 0.98 |
| qkv_proj M=4 N=6144 K=4096 | cuda | ok (1.3e-02) | 43.3 | 1,163 | 75 | 84 | 0.94 |
| qkv_proj M=8 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.1 | 1,229 | 79 | 89 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | triton | ok (1.3e-02) | 42.3 | 1,194 | 77 | 86 | 0.97 |
| qkv_proj M=8 N=6144 K=4096 | cuda | ok (1.3e-02) | 59.2 | 852 | 55 | 61 | 0.69 |
| qkv_proj M=16 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.4 | 1,222 | 79 | 88 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | triton | ok (1.3e-02) | 43.2 | 1,173 | 75 | 85 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cublas | ok (7.8e-03) | 28.5 | 1,176 | 76 | 85 | 1.00 |
| o_proj M=1 N=4096 K=4096 | triton | ok (7.8e-03) | 29.5 | 1,139 | 73 | 82 | 0.97 |
| o_proj M=1 N=4096 K=4096 | cuda | ok (7.8e-03) | 29.3 | 1,145 | 74 | 83 | 0.97 |
| o_proj M=2 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,171 | 75 | 84 | 1.00 |
| o_proj M=2 N=4096 K=4096 | triton | ok (1.3e-02) | 29.6 | 1,135 | 73 | 82 | 0.97 |
| o_proj M=2 N=4096 K=4096 | cuda | ok (1.3e-02) | 28.5 | 1,180 | 76 | 85 | 1.01 |
| o_proj M=4 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,171 | 75 | 84 | 1.00 |
| o_proj M=4 N=4096 K=4096 | triton | ok (1.3e-02) | 29.8 | 1,127 | 72 | 81 | 0.96 |
| o_proj M=4 N=4096 K=4096 | cuda | ok (1.3e-02) | 31.4 | 1,070 | 69 | 77 | 0.91 |
| o_proj M=8 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.9 | 1,165 | 75 | 84 | 1.00 |
| o_proj M=8 N=4096 K=4096 | triton | ok (1.3e-02) | 30.1 | 1,120 | 72 | 81 | 0.96 |
| o_proj M=8 N=4096 K=4096 | cuda | ok (1.3e-02) | 48.6 | 693 | 45 | 50 | 0.59 |
| o_proj M=16 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.9 | 1,169 | 75 | 84 | 1.00 |
| o_proj M=16 N=4096 K=4096 | triton | ok (1.3e-02) | 30.7 | 1,101 | 71 | 79 | 0.94 |
| o_proj M=16 N=4096 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cublas | ok (1.7e-02) | 178.1 | 1,319 | 85 | 95 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | triton | ok (1.3e-02) | 168.3 | 1,396 | 90 | 101 | 1.06 |
| gate_up_proj M=1 N=28672 K=4096 | cuda | ok (1.3e-02) | 165.2 | 1,422 | 91 | 103 | 1.08 |
| gate_up_proj M=2 N=28672 K=4096 | cublas | ok (1.3e-02) | 179.7 | 1,308 | 84 | 94 | 1.00 |
| gate_up_proj M=2 N=28672 K=4096 | triton | ok (1.3e-02) | 168.9 | 1,391 | 89 | 100 | 1.06 |
| gate_up_proj M=2 N=28672 K=4096 | cuda | ok (1.3e-02) | 167.8 | 1,401 | 90 | 101 | 1.07 |
| gate_up_proj M=4 N=28672 K=4096 | cublas | ok (1.3e-02) | 180.1 | 1,306 | 84 | 94 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | triton | ok (1.3e-02) | 169.7 | 1,385 | 89 | 100 | 1.06 |
| gate_up_proj M=4 N=28672 K=4096 | cuda | ok (1.3e-02) | 170.7 | 1,378 | 89 | 99 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cublas | ok (1.5e-02) | 181.3 | 1,298 | 83 | 94 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | triton | ok (1.5e-02) | 170.8 | 1,378 | 89 | 99 | 1.06 |
| gate_up_proj M=8 N=28672 K=4096 | cuda | ok (1.5e-02) | 234.4 | 1,004 | 65 | 72 | 0.77 |
| gate_up_proj M=16 N=28672 K=4096 | cublas | ok (1.5e-02) | 183.2 | 1,288 | 83 | 93 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | triton | ok (1.5e-02) | 172.8 | 1,365 | 88 | 98 | 1.06 |
| gate_up_proj M=16 N=28672 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.5 | 1,257 | 81 | 91 | 1.00 |
| down_proj M=1 N=4096 K=14336 | triton | ok (7.8e-03) | 89.4 | 1,315 | 85 | 95 | 1.05 |
| down_proj M=1 N=4096 K=14336 | cuda | ok (7.8e-03) | 88.5 | 1,327 | 85 | 96 | 1.06 |
| down_proj M=2 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.4 | 1,272 | 82 | 92 | 1.00 |
| down_proj M=2 N=4096 K=14336 | triton | ok (7.8e-03) | 88.8 | 1,323 | 85 | 95 | 1.04 |
| down_proj M=2 N=4096 K=14336 | cuda | ok (7.8e-03) | 92.1 | 1,276 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.1 | 1,277 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | triton | ok (7.8e-03) | 90.2 | 1,303 | 84 | 94 | 1.02 |
| down_proj M=4 N=4096 K=14336 | cuda | ok (7.8e-03) | 98.1 | 1,199 | 77 | 86 | 0.94 |
| down_proj M=8 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.9 | 1,268 | 82 | 91 | 1.00 |
| down_proj M=8 N=4096 K=14336 | triton | ok (7.8e-03) | 89.7 | 1,313 | 84 | 95 | 1.04 |
| down_proj M=8 N=4096 K=14336 | cuda | ok (7.8e-03) | 162.3 | 725 | 47 | 52 | 0.57 |
| down_proj M=16 N=4096 K=14336 | cublas | ok (1.3e-02) | 92.7 | 1,273 | 82 | 92 | 1.00 |
| down_proj M=16 N=4096 K=14336 | triton | ok (1.3e-02) | 91.7 | 1,287 | 83 | 93 | 1.01 |
| down_proj M=16 N=4096 K=14336 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| lm_head M=1 N=128256 K=4096 | cublas | ok (1.3e-02) | 721.1 | 1,458 | 94 | 105 | 1.00 |
| lm_head M=1 N=128256 K=4096 | triton | ok (1.3e-02) | 712.6 | 1,475 | 95 | 106 | 1.01 |
| lm_head M=1 N=128256 K=4096 | cuda | ok (1.3e-02) | 711.3 | 1,477 | 95 | 107 | 1.01 |
| lm_head M=2 N=128256 K=4096 | cublas | ok (1.4e-02) | 723.0 | 1,454 | 94 | 105 | 1.00 |
| lm_head M=2 N=128256 K=4096 | triton | ok (1.4e-02) | 715.8 | 1,469 | 94 | 106 | 1.01 |
| lm_head M=2 N=128256 K=4096 | cuda | ok (1.4e-02) | 722.3 | 1,455 | 94 | 105 | 1.00 |
| lm_head M=4 N=128256 K=4096 | cublas | ok (1.4e-02) | 728.6 | 1,444 | 93 | 104 | 1.00 |
| lm_head M=4 N=128256 K=4096 | triton | ok (1.4e-02) | 720.8 | 1,459 | 94 | 105 | 1.01 |
| lm_head M=4 N=128256 K=4096 | cuda | ok (1.4e-02) | 735.0 | 1,431 | 92 | 103 | 0.99 |
| lm_head M=8 N=128256 K=4096 | cublas | ok (1.5e-02) | 733.0 | 1,436 | 92 | 104 | 1.00 |
| lm_head M=8 N=128256 K=4096 | triton | ok (1.5e-02) | 725.3 | 1,452 | 93 | 105 | 1.01 |
| lm_head M=8 N=128256 K=4096 | cuda | ok (1.5e-02) | 1001.6 | 1,051 | 68 | 76 | 0.73 |
| lm_head M=16 N=128256 K=4096 | cublas | ok (1.5e-02) | 742.8 | 1,420 | 91 | 102 | 1.00 |
| lm_head M=16 N=128256 K=4096 | triton | ok (1.5e-02) | 735.8 | 1,434 | 92 | 103 | 1.01 |
| lm_head M=16 N=128256 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |

## fused_add_rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.6 | 2 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.7 | 15 | 1 | 1 | 6.63 |
| M=16 N=4096 | torch | ok (0.0e+00) | 29.7 | 18 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.7 | 194 | 12 | 14 | 10.80 |
| M=64 N=4096 | torch | ok (0.0e+00) | 32.3 | 65 | 4 | 5 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 3.2 | 648 | 42 | 47 | 9.94 |
| M=256 N=4096 | torch | ok (0.0e+00) | 45.6 | 184 | 12 | 13 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 6.1 | 1,380 | 89 | 100 | 7.50 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 298.8 | 225 | 14 | 16 | 1.00 |
| M=2048 N=4096 | triton | ok (1.6e-02) | 52.1 | 1,289 | 83 | 93 | 5.74 |

## silu_and_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.17 |
| M=16 N=14336 | torch | ok (0.0e+00) | 9.2 | 150 | 10 | 11 | 1.00 |
| M=16 N=14336 | triton | ok (6.2e-02) | 2.7 | 517 | 33 | 37 | 3.45 |
| M=64 N=14336 | torch | ok (0.0e+00) | 15.0 | 368 | 24 | 27 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.9 | 1,127 | 72 | 81 | 3.06 |
| M=256 N=14336 | torch | ok (0.0e+00) | 38.6 | 570 | 37 | 41 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 15.0 | 1,468 | 94 | 106 | 2.58 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 292.1 | 603 | 39 | 43 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 131.9 | 1,336 | 86 | 96 | 2.21 |

## rope_cache_write

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| tokens=1 | torch | ok (0.0e+00) | 37.6 | 1 | 0 | 0 | 1.00 |
| tokens=1 | triton | ok (1.6e-02) | 9.3 | 3 | 0 | 0 | 4.03 |
| tokens=16 | torch | ok (0.0e+00) | 43.5 | 9 | 1 | 1 | 1.00 |
| tokens=16 | triton | ok (1.6e-02) | 7.2 | 56 | 4 | 4 | 6.04 |
| tokens=64 | torch | ok (0.0e+00) | 47.4 | 34 | 2 | 2 | 1.00 |
| tokens=64 | triton | ok (3.1e-02) | 7.8 | 207 | 13 | 15 | 6.12 |
| tokens=256 | torch | ok (0.0e+00) | 63.7 | 101 | 6 | 7 | 1.00 |
| tokens=256 | triton | ok (3.1e-02) | 9.9 | 646 | 42 | 47 | 6.41 |
| tokens=2048 | torch | ok (0.0e+00) | 280.9 | 183 | 12 | 13 | 1.00 |
| tokens=2048 | triton | ok (3.1e-02) | 53.1 | 968 | 62 | 70 | 5.29 |

## paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 6.9 | 75 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.5 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 13.4 | 39 | 3 | 3 | 0.52 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.7 | 267 | 17 | 19 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.9 | 49 | 3 | 4 | 0.19 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 14.4 | 291 | 19 | 21 | 1.09 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.2 | 555 | 36 | 40 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 228.2 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 21.0 | 801 | 51 | 58 | 1.44 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.2 | 362 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 126.1 | 67 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 16.8 | 501 | 32 | 36 | 1.38 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 75.4 | 890 | 57 | 64 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 934.2 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 62.1 | 1,081 | 70 | 78 | 1.21 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 221.5 | 1,212 | 78 | 87 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3578.4 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 206.2 | 1,302 | 84 | 94 | 1.07 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.5 | 755 | 49 | 54 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 483.3 | 69 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 34.9 | 961 | 62 | 69 | 1.27 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 220.9 | 1,215 | 78 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3584.9 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 199.7 | 1,344 | 86 | 97 | 1.11 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 813.1 | 1,320 | 85 | 95 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 769.1 | 1,396 | 90 | 101 | 1.06 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 72.8 | 922 | 59 | 66 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 925.7 | 72 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 61.5 | 1,091 | 70 | 79 | 1.18 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 406.6 | 1,321 | 85 | 95 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 380.1 | 1,413 | 91 | 102 | 1.07 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1525.4 | 1,408 | 91 | 102 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1477.3 | 1,454 | 93 | 105 | 1.03 |

## paged_prefill_attention

| shape | impl | check | us | TFLOPs | % peak | x base |
|---|---|---|---:|---:|---:|---:|
| seqs=1 qlen=512 | vllm_fa2 | ok (1.6e-02) | 35.3 | 61.0 | 20 | 1.00 |
| seqs=1 qlen=512 | torch | ok (0.0e+00) | 665.7 | 3.2 | 1 | 0.05 |
| seqs=1 qlen=512 | triton | ok (1.6e-02) | 51.9 | 41.5 | 13 | 0.68 |
| seqs=1 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 84.2 | 102.2 | 33 | 1.00 |
| seqs=1 qlen=1024 | torch | ok (0.0e+00) | 2202.1 | 3.9 | 1 | 0.04 |
| seqs=1 qlen=1024 | triton | ok (1.6e-02) | 105.8 | 81.3 | 26 | 0.80 |
| seqs=1 qlen=4096 | vllm_fa2 | ok (1.6e-02) | 846.5 | 162.4 | 52 | 1.00 |
| seqs=1 qlen=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=1 qlen=4096 | triton | ok (1.6e-02) | 955.5 | 143.9 | 46 | 0.89 |
| seqs=4 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 260.8 | 131.9 | 42 | 1.00 |
| seqs=4 qlen=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=4 qlen=1024 | triton | ok (1.6e-02) | 296.6 | 116.0 | 37 | 0.88 |
| seqs=16 qlen=256 | vllm_fa2 | ok (1.6e-02) | 114.7 | 75.2 | 24 | 1.00 |
| seqs=16 qlen=256 | torch | ok (0.0e+00) | 3450.1 | 2.5 | 1 | 0.03 |
| seqs=16 qlen=256 | triton | ok (1.6e-02) | 132.3 | 65.2 | 21 | 0.87 |

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 19.2 | 677 | 44 | 49 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | ok (0.0e+00) | 23.0 | 564 | 36 | 41 | 0.83 |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | ok (4.8e-07) | 23.9 | 544 | 35 | 39 | 0.80 |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 41.0 | 317 | 20 | 23 | 0.47 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 683 | 44 | 49 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 32.0 | 408 | 26 | 29 | 0.60 |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | ok (6.1e-05) | 27.5 | 475 | 31 | 34 | 0.70 |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.1 | 318 | 20 | 23 | 0.47 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.0 | 690 | 44 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | ok (6.1e-05) | 46.1 | 285 | 18 | 21 | 0.41 |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 24.6 | 534 | 34 | 38 | 0.77 |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.2 | 319 | 21 | 23 | 0.46 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.5 | 619 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | ok (1.2e-04) | 87.9 | 151 | 10 | 11 | 0.24 |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | ok (3.1e-05) | 26.6 | 500 | 32 | 36 | 0.81 |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.6 | 320 | 21 | 23 | 0.52 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.7 | 492 | 32 | 35 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | ok (7.8e-03) | 171.0 | 80 | 5 | 6 | 0.16 |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 39.0 | 349 | 22 | 25 | 0.71 |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 323 | 21 | 23 | 0.66 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 45.1 | 317 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | ok (7.8e-03) | 83.1 | 172 | 11 | 12 | 0.54 |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.8 | 333 | 21 | 24 | 1.05 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.5 | 495 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | ok (0.0e+00) | 19.5 | 444 | 29 | 32 | 0.90 |
| o_proj M=1 N=4096 K=4096 | cuda_mma | ok (0.0e+00) | 21.8 | 398 | 26 | 29 | 0.80 |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 28.5 | 304 | 20 | 22 | 0.61 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.5 | 498 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 27.9 | 312 | 20 | 23 | 0.63 |
| o_proj M=4 N=4096 K=4096 | cuda_mma | ok (6.1e-05) | 18.4 | 473 | 30 | 34 | 0.95 |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.7 | 303 | 20 | 22 | 0.61 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.7 | 496 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | ok (6.1e-05) | 42.5 | 207 | 13 | 15 | 0.42 |
| o_proj M=8 N=4096 K=4096 | cuda_mma | ok (3.9e-03) | 18.6 | 472 | 30 | 34 | 0.95 |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 28.9 | 304 | 20 | 22 | 0.61 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.5 | 457 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | ok (4.9e-04) | 82.8 | 108 | 7 | 8 | 0.24 |
| o_proj M=16 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 19.5 | 457 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.0 | 308 | 20 | 22 | 0.67 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.1 | 416 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | ok (7.8e-03) | 126.5 | 73 | 5 | 5 | 0.17 |
| o_proj M=32 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 30.1 | 304 | 20 | 22 | 0.73 |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.3 | 313 | 20 | 23 | 0.75 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.8 | 271 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | ok (7.8e-03) | 63.3 | 153 | 10 | 11 | 0.57 |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.0 | 324 | 21 | 23 | 1.19 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.6 | 1,017 | 65 | 73 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 71.6 | 846 | 54 | 61 | 0.83 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | ok (3.9e-03) | 70.4 | 861 | 55 | 62 | 0.85 |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.3 | 340 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.3 | 1,008 | 65 | 73 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | ok (3.9e-03) | 98.3 | 619 | 40 | 45 | 0.61 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 72.0 | 845 | 54 | 61 | 0.84 |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.0 | 338 | 22 | 24 | 0.34 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.5 | 1,010 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 164.5 | 371 | 24 | 27 | 0.37 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 72.9 | 837 | 54 | 60 | 0.83 |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.0 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 66.1 | 932 | 60 | 67 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 328.9 | 187 | 12 | 14 | 0.20 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 78.1 | 789 | 51 | 57 | 0.85 |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 182.9 | 337 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 72.0 | 870 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | ok (7.8e-03) | 657.3 | 95 | 6 | 7 | 0.11 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 136.1 | 460 | 30 | 33 | 0.53 |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 213.9 | 293 | 19 | 21 | 0.34 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.9 | 569 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | ok (7.8e-03) | 312.5 | 207 | 13 | 15 | 0.36 |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 264.7 | 245 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.5 | 1,291 | 83 | 93 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | ok (2.4e-07) | 43.2 | 702 | 45 | 51 | 0.54 |
| down_proj M=1 N=4096 K=14336 | cuda_mma | ok (2.4e-07) | 36.5 | 830 | 53 | 60 | 0.64 |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.9 | 326 | 21 | 24 | 0.25 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.7 | 1,282 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | ok (9.8e-04) | 63.7 | 477 | 31 | 34 | 0.37 |
| down_proj M=4 N=4096 K=14336 | cuda_mma | ok (9.8e-04) | 37.3 | 815 | 52 | 59 | 0.64 |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.1 | 327 | 21 | 24 | 0.25 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.7 | 1,288 | 83 | 93 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 102.1 | 299 | 19 | 22 | 0.23 |
| down_proj M=8 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 37.9 | 806 | 52 | 58 | 0.63 |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.5 | 327 | 21 | 24 | 0.25 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.5 | 1,122 | 72 | 81 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | ok (3.9e-03) | 203.9 | 151 | 10 | 11 | 0.13 |
| down_proj M=16 N=4096 K=14336 | cuda_mma | ok (3.9e-03) | 40.6 | 760 | 49 | 55 | 0.68 |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 93.8 | 329 | 21 | 24 | 0.29 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.2 | 846 | 54 | 61 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | ok (7.8e-03) | 408.8 | 77 | 5 | 6 | 0.09 |
| down_proj M=32 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 75.2 | 418 | 27 | 30 | 0.49 |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 101.4 | 310 | 20 | 22 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 62.8 | 520 | 33 | 37 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | ok (7.8e-03) | 187.5 | 174 | 11 | 13 | 0.34 |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 99.3 | 329 | 21 | 24 | 0.63 |

