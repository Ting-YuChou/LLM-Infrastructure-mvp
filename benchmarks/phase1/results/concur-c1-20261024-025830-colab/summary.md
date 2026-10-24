# Phase 1 results: run concur-c1-20261024-025830-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_nsys | ok | 0 |
| download_model | ok | 68 |
| install_vllm | ok | 86 |
| cuda129 | ok | 40 |
| e2e | ok | 1204 |
| environment | ok | 2 |

## Environment

```
run_id: concur-c1-20261024-025830-colab
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
| w4a16_tiled | paged | 64 | 1024 | 256 | 16.02 | 3,996 | awq_marlin | 16.20 | 3,951 | 0.99x |
| w4a16_tiled | paged | 128 | 1024 | 256 | 30.57 | 4,187 | awq_marlin | 27.45 | 4,663 | 1.11x |
| w4a16_tiled | paged | 128 | 512 | 256 | 24.57 | 5,209 | awq_marlin | 21.42 | 5,974 | 1.15x |
| w4a16_tiled | paged | 256 | 512 | 256 | 50.44 | 5,075 | awq_marlin | 39.04 | 6,558 | 1.29x |
| w4a16_tiled_full | paged | 64 | 1024 | 256 | 16.04 | 3,989 | awq_marlin | 16.20 | 3,951 | 0.99x |
| w4a16_tiled_full | paged | 128 | 1024 | 256 | 30.20 | 4,239 | awq_marlin | 27.45 | 4,663 | 1.10x |
| w4a16_tiled_full | paged | 128 | 512 | 256 | 24.21 | 5,286 | awq_marlin | 21.42 | 5,974 | 1.13x |
| w4a16_tiled_full | paged | 256 | 512 | 256 | 49.99 | 5,121 | awq_marlin | 39.04 | 6,558 | 1.28x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_full_b128 | 28.97 | 28.92 | 0.05 | 428 | 2 | gemm 53%, attention 45%, norm 1%, activation 1% |
| nano_w4a16_tiled_full_b256 | 47.78 | 47.50 | 0.27 | 428 | 2 | gemm 42%, other 28%, attention 28%, norm 1% |
| nano_w4a16_tiled_full_b64 | 15.44 | 15.39 | 0.04 | 300 | 2 | gemm 52%, attention 45%, norm 1%, activation 1% |
| vllmw4_b128 | 26.36 | 26.29 | 0.08 | 429 | 15 | attention 49%, other 43%, gemm 3%, elementwise/copy 2% |
| vllmw4_b256 | 36.72 | 36.64 | 0.08 | 428 | 14 | other 55%, attention 37%, gemm 3%, elementwise/copy 3% |
| vllmw4_b64 | 15.83 | 15.74 | 0.08 | 429 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

