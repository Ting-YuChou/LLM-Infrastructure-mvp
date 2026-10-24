# Phase 1 results: run chunk-c7-20261024-052230-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | ok | 219 |
| golden | ok | 3 |
| e2e | ok | 1229 |
| environment | ok | 2 |

## Environment

```
run_id: chunk-c7-20261024-052230-colab
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
711 passed, 1 warning in 216.56s (0:03:36)
```

## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| w4a16_tiled_full | static | 1 | 128 | 256 | 41.27 | 24 | awq_marlin | 5.55 | 180 | 7.44x |
| w4a16_tiled_full | paged | 1 | 128 | 256 | 5.24 | 191 | awq_marlin | 5.55 | 180 | 0.94x |
| w4a16_tiled_full | paged | 16 | 1024 | 256 | 7.11 | 2,249 | awq_marlin | 7.85 | 2,039 | 0.91x |
| w4a16_tiled_full | paged | 64 | 1024 | 256 | 15.03 | 4,257 | awq_marlin | 16.18 | 3,956 | 0.93x |
| w4a16_tiled_full | paged | 128 | 1024 | 256 | 25.83 | 4,955 | awq_marlin | 27.48 | 4,658 | 0.94x |
| w4a16_tiled_full | paged | 128 | 512 | 256 | 19.86 | 6,447 | awq_marlin | 21.43 | 5,973 | 0.93x |
| w4a16_tiled_full | paged | 256 | 512 | 256 | 36.14 | 7,083 | awq_marlin | 39.05 | 6,555 | 0.93x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_tiled_full_b1 | 5.21 | 5.17 | 0.04 | 330 | 2 | gemm 83%, attention 8%, norm 4%, rope 2% |
| nano_w4a16_tiled_full_b128 | 24.60 | 24.55 | 0.04 | 299 | 2 | attention 52%, gemm 45%, norm 1%, activation 1% |
| nano_w4a16_tiled_full_b16 | 7.03 | 6.98 | 0.04 | 332 | 2 | gemm 65%, attention 29%, norm 3%, rope 1% |
| nano_w4a16_tiled_full_b256 | 33.84 | 33.81 | 0.04 | 299 | 2 | gemm 57%, attention 40%, activation 1%, norm 1% |
| nano_w4a16_tiled_full_b64 | 14.49 | 14.44 | 0.05 | 299 | 2 | gemm 49%, attention 47%, norm 1%, activation 1% |
| vllmw4_b1 | 5.66 | 5.59 | 0.07 | 366 | 15 | other 68%, gemm 13%, attention 11%, norm 4% |
| vllmw4_b128 | 26.28 | 26.20 | 0.08 | 429 | 15 | attention 49%, other 43%, gemm 3%, elementwise/copy 2% |
| vllmw4_b16 | 7.81 | 7.72 | 0.09 | 430 | 15 | other 52%, attention 30%, gemm 10%, elementwise/copy 4% |
| vllmw4_b256 | 36.61 | 36.53 | 0.07 | 428 | 14 | other 55%, attention 37%, gemm 3%, elementwise/copy 3% |
| vllmw4_b64 | 15.80 | 15.72 | 0.08 | 431 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

