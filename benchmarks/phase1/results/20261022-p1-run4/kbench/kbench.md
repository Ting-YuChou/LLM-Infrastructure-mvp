# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,380 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.0 | 1 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.9 | 9 | 1 | 1 | 5.90 |
| M=16 N=4096 | torch | ok (0.0e+00) | 28.5 | 9 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.4 | 113 | 7 | 8 | 11.96 |
| M=64 N=4096 | torch | ok (0.0e+00) | 30.6 | 34 | 2 | 2 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 2.8 | 383 | 25 | 28 | 11.09 |
| M=256 N=4096 | torch | ok (0.0e+00) | 42.9 | 98 | 6 | 7 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 4.2 | 1,003 | 65 | 73 | 10.25 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 256.8 | 131 | 8 | 9 | 1.00 |
| M=2048 N=4096 | triton | ok (0.0e+00) | 24.2 | 1,388 | 89 | 101 | 10.62 |

## silu_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.19 |
| M=16 N=14336 | torch | ok (0.0e+00) | 5.0 | 274 | 18 | 20 | 1.00 |
| M=16 N=14336 | triton | ok (3.1e-02) | 2.7 | 512 | 33 | 37 | 1.87 |
| M=64 N=14336 | torch | ok (0.0e+00) | 7.8 | 705 | 45 | 51 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.8 | 1,138 | 73 | 82 | 1.61 |
| M=256 N=14336 | torch | ok (0.0e+00) | 22.1 | 994 | 64 | 72 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 13.5 | 1,627 | 105 | 118 | 1.64 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 219.1 | 804 | 52 | 58 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 131.9 | 1,335 | 86 | 97 | 1.66 |

## linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | cublas | ok (7.8e-03) | 41.0 | 1,229 | 79 | 89 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | triton | ok (7.8e-03) | 41.4 | 1,216 | 78 | 88 | 0.99 |
| qkv_proj M=1 N=6144 K=4096 | cuda | ok (7.8e-03) | 40.9 | 1,231 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | cublas | ok (1.3e-02) | 40.9 | 1,232 | 79 | 89 | 1.00 |
| qkv_proj M=2 N=6144 K=4096 | triton | ok (1.3e-02) | 41.6 | 1,210 | 78 | 88 | 0.98 |
| qkv_proj M=2 N=6144 K=4096 | cuda | ok (1.3e-02) | 40.7 | 1,238 | 80 | 90 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.0 | 1,230 | 79 | 89 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | triton | ok (1.3e-02) | 41.9 | 1,204 | 77 | 87 | 0.98 |
| qkv_proj M=4 N=6144 K=4096 | cuda | ok (1.3e-02) | 43.3 | 1,165 | 75 | 84 | 0.95 |
| qkv_proj M=8 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.1 | 1,228 | 79 | 89 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | triton | ok (1.3e-02) | 42.3 | 1,193 | 77 | 86 | 0.97 |
| qkv_proj M=8 N=6144 K=4096 | cuda | ok (1.3e-02) | 59.3 | 852 | 55 | 62 | 0.69 |
| qkv_proj M=16 N=6144 K=4096 | cublas | ok (1.3e-02) | 41.5 | 1,221 | 79 | 88 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | triton | ok (1.3e-02) | 43.2 | 1,173 | 75 | 85 | 0.96 |
| qkv_proj M=16 N=6144 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cublas | ok (7.8e-03) | 28.4 | 1,181 | 76 | 86 | 1.00 |
| o_proj M=1 N=4096 K=4096 | triton | ok (7.8e-03) | 29.4 | 1,142 | 73 | 83 | 0.97 |
| o_proj M=1 N=4096 K=4096 | cuda | ok (7.8e-03) | 29.2 | 1,149 | 74 | 83 | 0.97 |
| o_proj M=2 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.6 | 1,174 | 75 | 85 | 1.00 |
| o_proj M=2 N=4096 K=4096 | triton | ok (1.3e-02) | 29.6 | 1,133 | 73 | 82 | 0.97 |
| o_proj M=2 N=4096 K=4096 | cuda | ok (1.3e-02) | 28.5 | 1,180 | 76 | 86 | 1.01 |
| o_proj M=4 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.7 | 1,172 | 75 | 85 | 1.00 |
| o_proj M=4 N=4096 K=4096 | triton | ok (1.3e-02) | 29.9 | 1,126 | 72 | 82 | 0.96 |
| o_proj M=4 N=4096 K=4096 | cuda | ok (1.3e-02) | 31.3 | 1,073 | 69 | 78 | 0.92 |
| o_proj M=8 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.8 | 1,169 | 75 | 85 | 1.00 |
| o_proj M=8 N=4096 K=4096 | triton | ok (1.3e-02) | 30.2 | 1,117 | 72 | 81 | 0.96 |
| o_proj M=8 N=4096 K=4096 | cuda | ok (1.3e-02) | 48.7 | 692 | 45 | 50 | 0.59 |
| o_proj M=16 N=4096 K=4096 | cublas | ok (1.3e-02) | 28.9 | 1,171 | 75 | 85 | 1.00 |
| o_proj M=16 N=4096 K=4096 | triton | ok (1.3e-02) | 30.8 | 1,099 | 71 | 80 | 0.94 |
| o_proj M=16 N=4096 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cublas | ok (1.7e-02) | 177.7 | 1,322 | 85 | 96 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | triton | ok (1.3e-02) | 168.6 | 1,394 | 90 | 101 | 1.05 |
| gate_up_proj M=1 N=28672 K=4096 | cuda | ok (1.3e-02) | 165.0 | 1,424 | 92 | 103 | 1.08 |
| gate_up_proj M=2 N=28672 K=4096 | cublas | ok (1.3e-02) | 180.1 | 1,305 | 84 | 95 | 1.00 |
| gate_up_proj M=2 N=28672 K=4096 | triton | ok (1.3e-02) | 168.9 | 1,392 | 90 | 101 | 1.07 |
| gate_up_proj M=2 N=28672 K=4096 | cuda | ok (1.3e-02) | 168.0 | 1,399 | 90 | 101 | 1.07 |
| gate_up_proj M=4 N=28672 K=4096 | cublas | ok (1.3e-02) | 179.6 | 1,309 | 84 | 95 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | triton | ok (1.3e-02) | 169.9 | 1,384 | 89 | 100 | 1.06 |
| gate_up_proj M=4 N=28672 K=4096 | cuda | ok (1.3e-02) | 170.7 | 1,377 | 89 | 100 | 1.05 |
| gate_up_proj M=8 N=28672 K=4096 | cublas | ok (1.5e-02) | 182.0 | 1,293 | 83 | 94 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | triton | ok (1.5e-02) | 170.6 | 1,380 | 89 | 100 | 1.07 |
| gate_up_proj M=8 N=28672 K=4096 | cuda | ok (1.5e-02) | 234.5 | 1,004 | 65 | 73 | 0.78 |
| gate_up_proj M=16 N=28672 K=4096 | cublas | ok (1.5e-02) | 182.7 | 1,291 | 83 | 94 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | triton | ok (1.5e-02) | 173.0 | 1,364 | 88 | 99 | 1.06 |
| gate_up_proj M=16 N=28672 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.3 | 1,273 | 82 | 92 | 1.00 |
| down_proj M=1 N=4096 K=14336 | triton | ok (7.8e-03) | 88.5 | 1,328 | 85 | 96 | 1.04 |
| down_proj M=1 N=4096 K=14336 | cuda | ok (7.8e-03) | 88.8 | 1,323 | 85 | 96 | 1.04 |
| down_proj M=2 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.6 | 1,270 | 82 | 92 | 1.00 |
| down_proj M=2 N=4096 K=14336 | triton | ok (7.8e-03) | 88.7 | 1,325 | 85 | 96 | 1.04 |
| down_proj M=2 N=4096 K=14336 | cuda | ok (7.8e-03) | 92.5 | 1,270 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cublas | ok (7.8e-03) | 92.5 | 1,272 | 82 | 92 | 1.00 |
| down_proj M=4 N=4096 K=14336 | triton | ok (7.8e-03) | 89.4 | 1,315 | 85 | 95 | 1.03 |
| down_proj M=4 N=4096 K=14336 | cuda | ok (7.8e-03) | 98.7 | 1,191 | 77 | 86 | 0.94 |
| down_proj M=8 N=4096 K=14336 | cublas | ok (7.8e-03) | 93.0 | 1,266 | 81 | 92 | 1.00 |
| down_proj M=8 N=4096 K=14336 | triton | ok (7.8e-03) | 89.5 | 1,316 | 85 | 95 | 1.04 |
| down_proj M=8 N=4096 K=14336 | cuda | ok (7.8e-03) | 162.5 | 724 | 47 | 52 | 0.57 |
| down_proj M=16 N=4096 K=14336 | cublas | ok (1.3e-02) | 93.1 | 1,268 | 82 | 92 | 1.00 |
| down_proj M=16 N=4096 K=14336 | triton | ok (1.3e-02) | 90.6 | 1,303 | 84 | 94 | 1.03 |
| down_proj M=16 N=4096 K=14336 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |
| lm_head M=1 N=128256 K=4096 | cublas | ok (1.3e-02) | 720.5 | 1,459 | 94 | 106 | 1.00 |
| lm_head M=1 N=128256 K=4096 | triton | ok (1.3e-02) | 712.6 | 1,475 | 95 | 107 | 1.01 |
| lm_head M=1 N=128256 K=4096 | cuda | ok (1.3e-02) | 711.3 | 1,477 | 95 | 107 | 1.01 |
| lm_head M=2 N=128256 K=4096 | cublas | ok (1.4e-02) | 723.1 | 1,454 | 93 | 105 | 1.00 |
| lm_head M=2 N=128256 K=4096 | triton | ok (1.4e-02) | 715.5 | 1,469 | 94 | 106 | 1.01 |
| lm_head M=2 N=128256 K=4096 | cuda | ok (1.4e-02) | 722.2 | 1,456 | 94 | 105 | 1.00 |
| lm_head M=4 N=128256 K=4096 | cublas | ok (1.4e-02) | 728.2 | 1,444 | 93 | 105 | 1.00 |
| lm_head M=4 N=128256 K=4096 | triton | ok (1.4e-02) | 720.7 | 1,459 | 94 | 106 | 1.01 |
| lm_head M=4 N=128256 K=4096 | cuda | ok (1.4e-02) | 734.8 | 1,431 | 92 | 104 | 0.99 |
| lm_head M=8 N=128256 K=4096 | cublas | ok (1.5e-02) | 732.8 | 1,437 | 92 | 104 | 1.00 |
| lm_head M=8 N=128256 K=4096 | triton | ok (1.5e-02) | 725.3 | 1,451 | 93 | 105 | 1.01 |
| lm_head M=8 N=128256 K=4096 | cuda | ok (1.5e-02) | 1001.6 | 1,051 | 68 | 76 | 0.73 |
| lm_head M=16 N=128256 K=4096 | cublas | ok (1.5e-02) | 742.5 | 1,421 | 91 | 103 | 1.00 |
| lm_head M=16 N=128256 K=4096 | triton | ok (1.5e-02) | 735.7 | 1,434 | 92 | 104 | 1.01 |
| lm_head M=16 N=128256 K=4096 | cuda | skipped: the CUDA GEMV takes up to 8 rows | - | - | - | - | - |

## fused_add_rms_norm

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=4096 | torch | ok (0.0e+00) | 17.9 | 2 | 0 | 0 | 1.00 |
| M=1 N=4096 | triton | ok (0.0e+00) | 2.7 | 15 | 1 | 1 | 6.73 |
| M=16 N=4096 | torch | ok (0.0e+00) | 29.6 | 18 | 1 | 1 | 1.00 |
| M=16 N=4096 | triton | ok (0.0e+00) | 2.7 | 194 | 12 | 14 | 10.77 |
| M=64 N=4096 | torch | ok (0.0e+00) | 32.0 | 66 | 4 | 5 | 1.00 |
| M=64 N=4096 | triton | ok (0.0e+00) | 3.2 | 648 | 42 | 47 | 9.86 |
| M=256 N=4096 | torch | ok (0.0e+00) | 45.6 | 184 | 12 | 13 | 1.00 |
| M=256 N=4096 | triton | ok (0.0e+00) | 6.1 | 1,380 | 89 | 100 | 7.50 |
| M=2048 N=4096 | torch | ok (0.0e+00) | 298.5 | 225 | 14 | 16 | 1.00 |
| M=2048 N=4096 | triton | ok (1.6e-02) | 52.1 | 1,289 | 83 | 93 | 5.74 |

## silu_and_mul

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| M=1 N=14336 | torch | ok (0.0e+00) | 4.1 | 21 | 1 | 2 | 1.00 |
| M=1 N=14336 | triton | ok (3.1e-02) | 1.9 | 46 | 3 | 3 | 2.17 |
| M=16 N=14336 | torch | ok (0.0e+00) | 9.2 | 150 | 10 | 11 | 1.00 |
| M=16 N=14336 | triton | ok (6.2e-02) | 2.7 | 516 | 33 | 37 | 3.45 |
| M=64 N=14336 | torch | ok (0.0e+00) | 15.0 | 367 | 24 | 27 | 1.00 |
| M=64 N=14336 | triton | ok (6.2e-02) | 4.9 | 1,127 | 72 | 82 | 3.07 |
| M=256 N=14336 | torch | ok (0.0e+00) | 38.6 | 570 | 37 | 41 | 1.00 |
| M=256 N=14336 | triton | ok (6.2e-02) | 15.1 | 1,454 | 93 | 105 | 2.55 |
| M=2048 N=14336 | torch | ok (0.0e+00) | 292.2 | 603 | 39 | 44 | 1.00 |
| M=2048 N=14336 | triton | ok (6.2e-02) | 131.9 | 1,335 | 86 | 97 | 2.22 |

## rope_cache_write

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| tokens=1 | torch | ok (0.0e+00) | 37.7 | 1 | 0 | 0 | 1.00 |
| tokens=1 | triton | ok (1.6e-02) | 9.3 | 3 | 0 | 0 | 4.05 |
| tokens=16 | torch | ok (0.0e+00) | 43.5 | 9 | 1 | 1 | 1.00 |
| tokens=16 | triton | ok (1.6e-02) | 7.2 | 56 | 4 | 4 | 6.04 |
| tokens=64 | torch | ok (0.0e+00) | 47.4 | 34 | 2 | 2 | 1.00 |
| tokens=64 | triton | ok (3.1e-02) | 7.8 | 207 | 13 | 15 | 6.10 |
| tokens=256 | torch | ok (0.0e+00) | 63.8 | 101 | 6 | 7 | 1.00 |
| tokens=256 | triton | ok (3.1e-02) | 9.9 | 646 | 42 | 47 | 6.41 |
| tokens=2048 | torch | ok (0.0e+00) | 280.9 | 183 | 12 | 13 | 1.00 |
| tokens=2048 | triton | ok (3.1e-02) | 53.1 | 968 | 62 | 70 | 5.29 |

## paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 7.0 | 75 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.6 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 13.3 | 39 | 3 | 3 | 0.52 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.7 | 267 | 17 | 19 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.9 | 49 | 3 | 4 | 0.18 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 14.4 | 291 | 19 | 21 | 1.09 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.2 | 556 | 36 | 40 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 228.1 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 21.0 | 799 | 51 | 58 | 1.44 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.2 | 361 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 126.5 | 66 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 16.8 | 499 | 32 | 36 | 1.38 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 74.9 | 896 | 58 | 65 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 934.4 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 62.2 | 1,079 | 69 | 78 | 1.20 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 222.3 | 1,207 | 78 | 87 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3578.1 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 208.4 | 1,288 | 83 | 93 | 1.07 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.5 | 755 | 49 | 55 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 483.2 | 69 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 35.2 | 954 | 61 | 69 | 1.26 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 220.4 | 1,218 | 78 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3585.5 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 200.9 | 1,336 | 86 | 97 | 1.10 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 813.1 | 1,320 | 85 | 96 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 769.2 | 1,396 | 90 | 101 | 1.06 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 73.0 | 919 | 59 | 67 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 925.0 | 73 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 62.1 | 1,081 | 70 | 78 | 1.18 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 404.6 | 1,327 | 85 | 96 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 383.8 | 1,399 | 90 | 101 | 1.05 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1525.3 | 1,408 | 91 | 102 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1478.2 | 1,453 | 93 | 105 | 1.03 |

## paged_prefill_attention

| shape | impl | check | us | TFLOPs | % peak | x base |
|---|---|---|---:|---:|---:|---:|
| seqs=1 qlen=512 | vllm_fa2 | ok (1.6e-02) | 35.3 | 61.0 | 20 | 1.00 |
| seqs=1 qlen=512 | torch | ok (0.0e+00) | 665.6 | 3.2 | 1 | 0.05 |
| seqs=1 qlen=512 | triton | ok (1.6e-02) | 51.9 | 41.5 | 13 | 0.68 |
| seqs=1 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 84.2 | 102.1 | 33 | 1.00 |
| seqs=1 qlen=1024 | torch | ok (0.0e+00) | 2202.4 | 3.9 | 1 | 0.04 |
| seqs=1 qlen=1024 | triton | ok (1.6e-02) | 105.5 | 81.5 | 26 | 0.80 |
| seqs=1 qlen=4096 | vllm_fa2 | ok (1.6e-02) | 846.1 | 162.5 | 52 | 1.00 |
| seqs=1 qlen=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=1 qlen=4096 | triton | ok (1.6e-02) | 955.4 | 143.9 | 46 | 0.89 |
| seqs=4 qlen=1024 | vllm_fa2 | ok (1.6e-02) | 260.9 | 131.8 | 42 | 1.00 |
| seqs=4 qlen=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - |
| seqs=4 qlen=1024 | triton | ok (1.6e-02) | 296.5 | 116.0 | 37 | 0.88 |
| seqs=16 qlen=256 | vllm_fa2 | ok (1.6e-02) | 113.6 | 75.9 | 24 | 1.00 |
| seqs=16 qlen=256 | torch | ok (0.0e+00) | 3456.3 | 2.5 | 1 | 0.03 |
| seqs=16 qlen=256 | triton | ok (1.6e-02) | 132.4 | 65.1 | 21 | 0.86 |

## w4a16_linear

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| qkv_proj M=1 N=6144 K=4096 | vllm_marlin | ok (1.6e-02) | 19.2 | 678 | 44 | 49 | 1.00 |
| qkv_proj M=1 N=6144 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=1 N=6144 K=4096 | cublas_bf16 | ok (1.6e-02) | 42.1 | 309 | 20 | 22 | 0.45 |
| qkv_proj M=4 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.1 | 684 | 44 | 50 | 1.00 |
| qkv_proj M=4 N=6144 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=4 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.0 | 311 | 20 | 23 | 0.45 |
| qkv_proj M=8 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 19.0 | 691 | 44 | 50 | 1.00 |
| qkv_proj M=8 N=6144 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=8 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 42.2 | 312 | 20 | 23 | 0.45 |
| qkv_proj M=16 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 21.4 | 623 | 40 | 45 | 1.00 |
| qkv_proj M=16 N=6144 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=16 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 41.6 | 320 | 21 | 23 | 0.51 |
| qkv_proj M=32 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 27.6 | 494 | 32 | 36 | 1.00 |
| qkv_proj M=32 N=6144 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=32 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 43.4 | 314 | 20 | 23 | 0.64 |
| qkv_proj M=64 N=6144 K=4096 | vllm_marlin | ok (3.1e-02) | 45.3 | 315 | 20 | 23 | 1.00 |
| qkv_proj M=64 N=6144 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| qkv_proj M=64 N=6144 K=4096 | cublas_bf16 | ok (3.1e-02) | 44.6 | 320 | 21 | 23 | 1.02 |
| o_proj M=1 N=4096 K=4096 | vllm_marlin | ok (1.6e-02) | 17.6 | 492 | 32 | 36 | 1.00 |
| o_proj M=1 N=4096 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=1 N=4096 K=4096 | cublas_bf16 | ok (1.6e-02) | 30.0 | 289 | 19 | 21 | 0.59 |
| o_proj M=4 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.5 | 498 | 32 | 36 | 1.00 |
| o_proj M=4 N=4096 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=4 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.6 | 294 | 19 | 21 | 0.59 |
| o_proj M=8 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 17.6 | 499 | 32 | 36 | 1.00 |
| o_proj M=8 N=4096 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=8 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.4 | 289 | 19 | 21 | 0.58 |
| o_proj M=16 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 19.6 | 454 | 29 | 33 | 1.00 |
| o_proj M=16 N=4096 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=16 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.5 | 292 | 19 | 21 | 0.64 |
| o_proj M=32 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 22.0 | 417 | 27 | 30 | 1.00 |
| o_proj M=32 N=4096 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=32 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 30.9 | 297 | 19 | 22 | 0.71 |
| o_proj M=64 N=4096 K=4096 | vllm_marlin | ok (3.1e-02) | 35.8 | 271 | 17 | 20 | 1.00 |
| o_proj M=64 N=4096 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| o_proj M=64 N=4096 K=4096 | cublas_bf16 | ok (3.1e-02) | 29.9 | 324 | 21 | 23 | 1.20 |
| gate_up_proj M=1 N=28672 K=4096 | vllm_marlin | ok (1.6e-02) | 59.7 | 1,015 | 65 | 74 | 1.00 |
| gate_up_proj M=1 N=28672 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=1 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 178.6 | 339 | 22 | 25 | 0.33 |
| gate_up_proj M=4 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 59.8 | 1,017 | 65 | 74 | 1.00 |
| gate_up_proj M=4 N=28672 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=4 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 180.9 | 336 | 22 | 24 | 0.33 |
| gate_up_proj M=8 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 60.3 | 1,013 | 65 | 73 | 1.00 |
| gate_up_proj M=8 N=28672 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=8 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 181.1 | 337 | 22 | 24 | 0.33 |
| gate_up_proj M=16 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 65.7 | 937 | 60 | 68 | 1.00 |
| gate_up_proj M=16 N=28672 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=16 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 183.5 | 336 | 22 | 24 | 0.36 |
| gate_up_proj M=32 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 71.8 | 873 | 56 | 63 | 1.00 |
| gate_up_proj M=32 N=28672 K=4096 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=32 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 213.0 | 294 | 19 | 21 | 0.34 |
| gate_up_proj M=64 N=28672 K=4096 | vllm_marlin | ok (3.1e-02) | 113.5 | 570 | 37 | 41 | 1.00 |
| gate_up_proj M=64 N=28672 K=4096 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| gate_up_proj M=64 N=28672 K=4096 | cublas_bf16 | ok (3.1e-02) | 264.1 | 245 | 16 | 18 | 0.43 |
| down_proj M=1 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.9 | 1,267 | 81 | 92 | 1.00 |
| down_proj M=1 N=4096 K=14336 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=1 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.4 | 332 | 21 | 24 | 0.26 |
| down_proj M=4 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.7 | 1,285 | 83 | 93 | 1.00 |
| down_proj M=4 N=4096 K=14336 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=4 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.7 | 332 | 21 | 24 | 0.26 |
| down_proj M=8 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 23.7 | 1,292 | 83 | 94 | 1.00 |
| down_proj M=8 N=4096 K=14336 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=8 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 91.9 | 333 | 21 | 24 | 0.26 |
| down_proj M=16 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 27.7 | 1,116 | 72 | 81 | 1.00 |
| down_proj M=16 N=4096 K=14336 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=16 N=4096 K=14336 | cublas_bf16 | ok (1.6e-02) | 92.4 | 334 | 21 | 24 | 0.30 |
| down_proj M=32 N=4096 K=14336 | vllm_marlin | ok (1.6e-02) | 37.3 | 843 | 54 | 61 | 1.00 |
| down_proj M=32 N=4096 K=14336 | cuda_gemv | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=32 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 100.2 | 314 | 20 | 23 | 0.37 |
| down_proj M=64 N=4096 K=14336 | vllm_marlin | ok (3.1e-02) | 62.8 | 520 | 33 | 38 | 1.00 |
| down_proj M=64 N=4096 K=14336 | cuda_gemv | skipped: the gemv kernel takes up to 32 rows | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cuda_mma | error: RuntimeError: building kernels/csrc/w4a16.cu failed (see the log) | - | - | - | - | - |
| down_proj M=64 N=4096 K=14336 | cublas_bf16 | ok (3.1e-02) | 98.6 | 331 | 21 | 24 | 0.64 |

