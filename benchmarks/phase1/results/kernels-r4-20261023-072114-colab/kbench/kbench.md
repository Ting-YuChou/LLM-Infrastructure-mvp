# kbench on NVIDIA A100-SXM4-40GB

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,385 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

## paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 6.9 | 75 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.4 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 16.7 | 31 | 2 | 2 | 0.42 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.6 | 269 | 17 | 19 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.7 | 49 | 3 | 4 | 0.18 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 14.4 | 292 | 19 | 21 | 1.08 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.0 | 560 | 36 | 40 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 226.9 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 21.0 | 800 | 51 | 58 | 1.43 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.2 | 362 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 125.7 | 67 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 17.1 | 492 | 32 | 36 | 1.36 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 75.7 | 887 | 57 | 64 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 933.0 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 61.8 | 1,085 | 70 | 78 | 1.22 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 223.4 | 1,202 | 77 | 87 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3573.9 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 205.0 | 1,309 | 84 | 95 | 1.09 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.4 | 755 | 49 | 55 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 482.1 | 70 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 35.0 | 960 | 62 | 69 | 1.27 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 219.7 | 1,222 | 79 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3582.4 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 200.5 | 1,339 | 86 | 97 | 1.10 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 814.4 | 1,318 | 85 | 95 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 770.2 | 1,394 | 90 | 101 | 1.06 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 72.4 | 927 | 60 | 67 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 924.1 | 73 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 61.4 | 1,093 | 70 | 79 | 1.18 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 403.7 | 1,330 | 86 | 96 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 380.5 | 1,411 | 91 | 102 | 1.06 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1530.2 | 1,403 | 90 | 101 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1477.4 | 1,454 | 93 | 105 | 1.04 |

## paged_decode_attention_fp8

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=1024 | triton | ok (9.8e-04) | 12.0 | 180 | 12 | 13 | 1.00 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 19.8 | 437 | 28 | 32 | 1.00 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 53.9 | 642 | 41 | 46 | 1.00 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 156.1 | 886 | 57 | 64 | 1.00 |
| B=64 ctx=1024 | triton | ok (2.0e-03) | 147.3 | 940 | 60 | 68 | 1.00 |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 541.6 | 1,022 | 66 | 74 | 1.00 |
| B=128 ctx=1024 | triton | ok (2.0e-03) | 273.2 | 1,013 | 65 | 73 | 1.00 |
| B=128 ctx=4096 | triton | ok (4.9e-04) | 1033.8 | 1,071 | 69 | 77 | 1.00 |

