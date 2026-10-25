# Phase 1 results: run kernels-r4-20261023-072114-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| kbench | ok | 58 |
| e2e | **1** | 1092 |
| environment | ok | 2 |

## Environment

```
run_id: kernels-r4-20261023-072114-colab
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=/usr/local/cuda-12.8; nvcc: Build cuda_12.8.r12.8/compiler.35583870_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,385 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### paged_decode_attention

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

### paged_decode_attention_fp8

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


## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| bf16_async | paged | 1 | 128 | 256 | 12.05 | 83 | bf16 | 12.60 | 79 | 0.96x |
| bf16_async | paged | 16 | 1024 | 256 | 14.13 | 1,132 | bf16 | 14.82 | 1,080 | 0.95x |
| bf16_async | paged | 64 | 1024 | 256 | 22.60 | 2,832 | bf16 | 23.04 | 2,777 | 0.98x |
| bf16_async | paged | 16 | 4096 | 256 | 18.73 | 854 | - | - | - | - |
| bf16_async_fp8kv | paged | 1 | 128 | 256 | 12.01 | 83 | bf16 | 13.05 | 77 | 0.92x |
| bf16_async_fp8kv | paged | 16 | 1024 | 256 | 13.81 | 1,158 | bf16 | 14.50 | 1,104 | 0.95x |
| bf16_async_fp8kv | paged | 64 | 1024 | 256 | 20.87 | 3,066 | bf16 | 21.30 | 3,004 | 0.98x |
| bf16_async_fp8kv | paged | 16 | 4096 | 256 | 17.16 | 933 | - | - | - | - |
| w4a16_async | paged | 1 | 128 | 256 | 5.68 | 176 | awq_marlin | 5.56 | 180 | 1.02x |
| w4a16_async | paged | 16 | 1024 | 256 | 8.48 | 1,886 | awq_marlin | 7.86 | 2,035 | 1.08x |
| w4a16_async | paged | 64 | 1024 | 256 | 23.27 | 2,750 | awq_marlin | 16.19 | 3,952 | 1.44x |
| w4a16_async | paged | 16 | 4096 | 256 | 13.01 | 1,230 | - | - | - | - |
| w4a16_async_fp8kv | paged | 1 | 128 | 256 | 5.65 | 177 | awq_marlin | 6.05 | 165 | 0.93x |
| w4a16_async_fp8kv | paged | 16 | 1024 | 256 | 8.25 | 1,940 | awq_marlin | 7.55 | 2,118 | 1.09x |
| w4a16_async_fp8kv | paged | 64 | 1024 | 256 | 21.52 | 2,974 | awq_marlin | 14.44 | 4,433 | 1.49x |
| w4a16_async_fp8kv | paged | 16 | 4096 | 256 | 11.60 | 1,380 | - | - | - | - |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_async_fp8kv_b1 | 5.72 | 5.68 | 0.04 | 330 | 2 | gemm 85%, attention 7%, norm 3%, rope 2% |
| nano_w4a16_async_fp8kv_b16 | 8.24 | 8.21 | 0.04 | 332 | 2 | gemm 73%, attention 21%, norm 2%, rope 2% |

