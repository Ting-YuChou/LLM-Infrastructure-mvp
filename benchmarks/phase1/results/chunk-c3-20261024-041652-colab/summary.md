# Phase 1 results: run chunk-c3-20261024-041652-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| e2e | ok | 1103 |
| environment | ok | 2 |

## Environment

```
run_id: chunk-c3-20261024-041652-colab
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=/usr/local/cuda-12.8; nvcc: Build cuda_12.8.r12.8/compiler.35583870_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| w4a16_tiled_full | paged | 64 | 1024 | 256 | 16.15 | 3,962 | awq_marlin | 16.18 | 3,956 | 1.00x |
| w4a16_tiled_full | paged | 128 | 1024 | 256 | 28.37 | 4,511 | awq_marlin | 27.47 | 4,660 | 1.03x |
| w4a16_tiled_full | paged | 128 | 512 | 256 | 22.40 | 5,715 | awq_marlin | 21.48 | 5,959 | 1.04x |
| w4a16_tiled_full | paged | 256 | 512 | 256 | 40.96 | 6,249 | awq_marlin | 39.09 | 6,549 | 1.05x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_full_b128 | 27.16 | 27.11 | 0.05 | 300 | 2 | gemm 50%, attention 47%, norm 1%, activation 1% |
| nano_w4a16_tiled_full_b256 | 38.72 | 38.67 | 0.06 | 300 | 2 | gemm 62%, attention 35%, activation 1%, norm 1% |
| nano_w4a16_tiled_full_b64 | 15.58 | 15.52 | 0.05 | 300 | 2 | gemm 53%, attention 44%, norm 1%, activation 1% |
| vllmw4_b128 | 26.26 | 26.19 | 0.07 | 431 | 15 | attention 49%, other 43%, gemm 3%, elementwise/copy 2% |
| vllmw4_b256 | 36.52 | 36.45 | 0.08 | 430 | 14 | other 55%, attention 37%, gemm 3%, elementwise/copy 3% |
| vllmw4_b64 | 15.81 | 15.72 | 0.08 | 430 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

