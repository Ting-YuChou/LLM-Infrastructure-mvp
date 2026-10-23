# Phase 1 results: run kernels-r2-20261023-065546-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | **1** | 243 |
| kbench | ok | 57 |
| e2e | **1** | 771 |
| quality | ok | 126 |
| environment | ok | 2 |

## Environment

```
run_id: kernels-r2-20261023-065546-colab
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=/usr/local/cuda-12.8; nvcc: Build cuda_12.8.r12.8/compiler.35583870_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Tests

```
9 failed, 535 passed, 1 warning in 240.81s (0:04:00)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,380 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### paged_decode_attention

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=128 | vllm_fa2 | ok (2.0e-03) | 6.9 | 76 | 5 | 5 | 1.00 |
| B=1 ctx=128 | torch | ok (0.0e+00) | 50.4 | 10 | 1 | 1 | 0.14 |
| B=1 ctx=128 | triton | ok (2.0e-03) | 15.6 | 34 | 2 | 2 | 0.44 |
| B=1 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 15.6 | 269 | 17 | 20 | 1.00 |
| B=1 ctx=1024 | torch | ok (0.0e+00) | 84.8 | 49 | 3 | 4 | 0.18 |
| B=1 ctx=1024 | triton | ok (9.8e-04) | 14.4 | 292 | 19 | 21 | 1.08 |
| B=1 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 30.0 | 559 | 36 | 41 | 1.00 |
| B=1 ctx=4096 | torch | ok (0.0e+00) | 226.9 | 74 | 5 | 5 | 0.13 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 20.9 | 801 | 52 | 58 | 1.43 |
| B=16 ctx=128 | vllm_fa2 | ok (3.9e-03) | 23.2 | 362 | 23 | 26 | 1.00 |
| B=16 ctx=128 | torch | ok (0.0e+00) | 125.9 | 67 | 4 | 5 | 0.18 |
| B=16 ctx=128 | triton | ok (3.9e-03) | 17.1 | 492 | 32 | 36 | 1.36 |
| B=16 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 75.6 | 888 | 57 | 64 | 1.00 |
| B=16 ctx=1024 | torch | ok (0.0e+00) | 933.1 | 72 | 5 | 5 | 0.08 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 62.3 | 1,078 | 69 | 78 | 1.21 |
| B=16 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 223.4 | 1,202 | 77 | 87 | 1.00 |
| B=16 ctx=4096 | torch | ok (0.0e+00) | 3573.5 | 75 | 5 | 5 | 0.06 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 204.6 | 1,312 | 84 | 95 | 1.09 |
| B=64 ctx=128 | vllm_fa2 | ok (3.9e-03) | 44.4 | 755 | 49 | 55 | 1.00 |
| B=64 ctx=128 | torch | ok (0.0e+00) | 481.7 | 70 | 4 | 5 | 0.09 |
| B=64 ctx=128 | triton | ok (3.9e-03) | 34.9 | 960 | 62 | 70 | 1.27 |
| B=64 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 221.7 | 1,211 | 78 | 88 | 1.00 |
| B=64 ctx=1024 | torch | ok (0.0e+00) | 3583.1 | 75 | 5 | 5 | 0.06 |
| B=64 ctx=1024 | triton | ok (9.8e-04) | 201.0 | 1,335 | 86 | 97 | 1.10 |
| B=64 ctx=4096 | vllm_fa2 | ok (4.9e-04) | 814.2 | 1,319 | 85 | 96 | 1.00 |
| B=64 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 770.2 | 1,394 | 90 | 101 | 1.06 |
| B=128 ctx=128 | vllm_fa2 | ok (3.9e-03) | 72.5 | 926 | 60 | 67 | 1.00 |
| B=128 ctx=128 | torch | ok (0.0e+00) | 923.7 | 73 | 5 | 5 | 0.08 |
| B=128 ctx=128 | triton | ok (3.9e-03) | 61.0 | 1,100 | 71 | 80 | 1.19 |
| B=128 ctx=1024 | vllm_fa2 | ok (9.8e-04) | 403.7 | 1,330 | 86 | 96 | 1.00 |
| B=128 ctx=1024 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=1024 | triton | ok (9.8e-04) | 379.9 | 1,413 | 91 | 102 | 1.06 |
| B=128 ctx=4096 | vllm_fa2 | ok (9.8e-04) | 1525.0 | 1,408 | 91 | 102 | 1.00 |
| B=128 ctx=4096 | torch | skipped: reference too slow to time at this size | - | - | - | - | - |
| B=128 ctx=4096 | triton | ok (9.8e-04) | 1484.9 | 1,446 | 93 | 105 | 1.03 |

### paged_decode_attention_fp8

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| B=1 ctx=1024 | triton | ok (9.8e-04) | 14.2 | 153 | 10 | 11 | 1.00 |
| B=1 ctx=4096 | triton | ok (4.9e-04) | 23.1 | 375 | 24 | 27 | 1.00 |
| B=16 ctx=1024 | triton | ok (9.8e-04) | 58.7 | 590 | 38 | 43 | 1.00 |
| B=16 ctx=4096 | triton | ok (4.9e-04) | 188.5 | 734 | 47 | 53 | 1.00 |
| B=64 ctx=1024 | triton | ok (2.0e-03) | 147.9 | 936 | 60 | 68 | 1.00 |
| B=64 ctx=4096 | triton | ok (4.9e-04) | 543.3 | 1,019 | 66 | 74 | 1.00 |
| B=128 ctx=1024 | triton | ok (2.0e-03) | 273.3 | 1,013 | 65 | 73 | 1.00 |
| B=128 ctx=4096 | triton | ok (4.9e-04) | 1032.1 | 1,073 | 69 | 78 | 1.00 |


## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| bf16_async_fp8kv | paged | 1 | 128 | 256 | 12.09 | 83 | - | - | - | - |
| bf16_async_fp8kv | paged | 16 | 1024 | 256 | 14.01 | 1,142 | - | - | - | - |
| bf16_async_fp8kv | paged | 64 | 1024 | 256 | 20.87 | 3,067 | - | - | - | - |
| bf16_async_fp8kv | paged | 16 | 4096 | 256 | 18.07 | 886 | - | - | - | - |
| w4a16_async_fp8kv | paged | 1 | 128 | 256 | 5.79 | 173 | - | - | - | - |
| w4a16_async_fp8kv | paged | 16 | 1024 | 256 | 8.46 | 1,892 | - | - | - | - |
| w4a16_async_fp8kv | paged | 64 | 1024 | 256 | 21.50 | 2,977 | - | - | - | - |
| w4a16_async_fp8kv | paged | 16 | 4096 | 256 | 12.56 | 1,273 | - | - | - | - |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_async_fp8kv_b1 | 5.84 | 5.80 | 0.04 | 330 | 2 | gemm 84%, attention 8%, norm 3%, rope 2% |
| nano_w4a16_async_fp8kv_b16 | 8.32 | 8.28 | 0.04 | 332 | 2 | gemm 72%, attention 22%, norm 2%, rope 2% |

## Teacher-forced quality (288627 tokens, chunks of 2048)

| config | attention | linear | weights | KV cache | weight GiB | perplexity | top-1 agreement |
|---|---|---|---|---|---:|---:|---:|
| bf16 | triton | triton | none | auto | 15.0 | 7.232 | - |
| bf16_fp8kv | triton | triton | none | fp8 | 15.0 | 7.248 | 97.55% vs bf16 |
| w4a16 | triton | triton | w4a16 | auto | 5.3 | 8.024 | 85.88% vs bf16 |
| w4a16_fp8kv | triton | triton | w4a16 | fp8 | 5.3 | 8.040 | 85.79% vs bf16 |

