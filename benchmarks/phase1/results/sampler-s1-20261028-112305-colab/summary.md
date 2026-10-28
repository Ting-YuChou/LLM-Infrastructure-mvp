# Phase 1 results: run sampler-s1-20261028-112305-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_nsys | ok | 0 |
| download_model | ok | 52 |
| install_vllm | ok | 263 |
| cuda129 | ok | 24 |
| tests | **1** | 517 |
| kbench | **1** | 43 |
| e2e | ok | 999 |
| environment | ok | 2 |

## Environment

```
run_id: sampler-s1-20261028-112305-colab
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
1 failed, 776 passed, 1 warning in 514.41s (0:08:34)
```

## Kernels (kbench)

Peak 1,555 GB/s HBM and 312 TFLOPs bf16; measured copy 1,379 GB/s. Time per call from CUDA-graph replay, inputs rotated past L2. x base: speed relative to the first impl.

### sampler

| shape | impl | check | us | GB/s | % peak | % copy | x base |
|---|---|---|---:|---:|---:|---:|---:|
| peaked greedy B=1 | vllm | ok (None) | 41.1 | 6 | 0 | 0 | 1.00 |
| peaked greedy B=1 | nano_torch | ok (None) | 52.0 | 5 | 0 | 0 | 0.79 |
| peaked greedy B=1 | nano_cuda | ok (None) | 15.6 | 16 | 1 | 1 | 2.63 |
| peaked greedy B=4 | vllm | ok (None) | 43.4 | 24 | 2 | 2 | 1.00 |
| peaked greedy B=4 | nano_torch | ok (None) | 53.2 | 19 | 1 | 1 | 0.82 |
| peaked greedy B=4 | nano_cuda | ok (None) | 16.1 | 64 | 4 | 5 | 2.69 |
| peaked greedy B=8 | vllm | ok (None) | 46.6 | 44 | 3 | 3 | 1.00 |
| peaked greedy B=8 | nano_torch | ok (None) | 53.6 | 38 | 2 | 3 | 0.87 |
| peaked greedy B=8 | nano_cuda | ok (None) | 17.4 | 118 | 8 | 9 | 2.67 |
| peaked greedy B=16 | vllm | ok (None) | 51.0 | 80 | 5 | 6 | 1.00 |
| peaked greedy B=16 | nano_torch | ok (None) | 54.3 | 76 | 5 | 5 | 0.94 |
| peaked greedy B=16 | nano_cuda | ok (None) | 20.4 | 201 | 13 | 15 | 2.50 |
| peaked greedy B=32 | vllm | ok (None) | 66.1 | 124 | 8 | 9 | 1.00 |
| peaked greedy B=32 | nano_torch | ok (None) | 54.8 | 150 | 10 | 11 | 1.21 |
| peaked greedy B=32 | nano_cuda | ok (None) | 23.2 | 353 | 23 | 26 | 2.85 |
| peaked greedy B=64 | vllm | ok (None) | 106.0 | 155 | 10 | 11 | 1.00 |
| peaked greedy B=64 | nano_torch | ok (None) | 55.3 | 297 | 19 | 22 | 1.91 |
| peaked greedy B=64 | nano_cuda | ok (None) | 34.9 | 471 | 30 | 34 | 3.04 |
| peaked greedy B=128 | vllm | ok (None) | 168.8 | 194 | 13 | 14 | 1.00 |
| peaked greedy B=128 | nano_torch | ok (None) | 68.2 | 482 | 31 | 35 | 2.48 |
| peaked greedy B=128 | nano_cuda | ok (None) | 50.2 | 654 | 42 | 47 | 3.36 |
| peaked greedy B=256 | vllm | ok (None) | 297.1 | 221 | 14 | 16 | 1.00 |
| peaked greedy B=256 | nano_torch | ok (None) | 92.9 | 707 | 45 | 51 | 3.20 |
| peaked greedy B=256 | nano_cuda | ok (None) | 78.9 | 832 | 54 | 60 | 3.77 |
| peaked t0.7p0.9 B=1 | vllm | ok (None) | 268.8 | 1 | 0 | 0 | 1.00 |
| peaked t0.7p0.9 B=1 | nano_torch | error: RuntimeError: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=1 | nano_cuda | error: RuntimeError: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=4 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=4 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=4 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=8 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=8 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=8 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=16 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=16 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=16 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=32 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=32 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=32 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=64 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=64 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=64 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=128 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=128 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=128 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=256 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=256 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9 B=256 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=1 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=1 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=1 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=4 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=4 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=4 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=8 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=8 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=8 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=16 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=16 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=16 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=32 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=32 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=32 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=64 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=64 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=64 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=128 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=128 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=128 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=256 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=256 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t1 B=256 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=1 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=1 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=1 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=4 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=4 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=4 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=8 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=8 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=8 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=16 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=16 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=16 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=32 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=32 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=32 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=64 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=64 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=64 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=128 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=128 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=128 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=256 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=256 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| peaked t0.7p0.9k50 B=256 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=1 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=1 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=1 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=4 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=4 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=4 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=8 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=8 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=8 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=16 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=16 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=16 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=32 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=32 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=32 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=64 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=64 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=64 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=128 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=128 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=128 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=256 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=256 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9 B=256 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=1 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=1 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=1 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=4 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=4 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=4 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=8 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=8 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=8 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=16 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=16 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=16 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=32 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=32 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=32 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=64 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=64 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=64 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=128 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=128 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=128 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=256 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=256 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t1 B=256 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=1 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=1 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=1 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=4 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=4 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=4 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=8 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=8 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=8 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=16 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=16 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=16 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=32 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=32 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=32 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=64 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=64 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=64 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=128 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=128 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=128 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=256 | vllm | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=256 | nano_torch | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |
| flat t0.7p0.9k50 B=256 | nano_cuda | error: RuntimeError: reference failed: Offset increment outside graph capture encountered unexpectedly. | - | - | - | - | - |


## End to end decode (nano_vllm vs vLLM CUDA graph)

| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step | vLLM tok/s | nano / vLLM time |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| bf16 | paged | 1 | 128 | 256 | 12.01 | 83 | bf16 | 12.62 | 79 | 0.95x |
| bf16 | paged | 16 | 1024 | 256 | 14.08 | 1,136 | bf16 | 14.82 | 1,079 | 0.95x |
| bf16 | paged | 64 | 1024 | 256 | 22.55 | 2,839 | bf16 | 23.02 | 2,780 | 0.98x |
| bf16_sampled | paged | 1 | 128 | 256 | 12.10 | 83 | bf16 | 12.87 | 78 | 0.94x |
| bf16_sampled | paged | 16 | 1024 | 256 | 14.44 | 1,108 | bf16 | 16.14 | 991 | 0.89x |
| bf16_sampled | paged | 64 | 1024 | 256 | 23.64 | 2,707 | bf16 | 24.76 | 2,585 | 0.96x |
| w4a16 | paged | 1 | 128 | 256 | 5.09 | 196 | awq_marlin | 5.54 | 181 | 0.92x |
| w4a16 | paged | 16 | 1024 | 256 | 7.06 | 2,266 | awq_marlin | 7.85 | 2,039 | 0.90x |
| w4a16 | paged | 64 | 1024 | 256 | 15.00 | 4,267 | awq_marlin | 16.21 | 3,947 | 0.93x |
| w4a16_sampled | paged | 1 | 128 | 256 | 5.16 | 194 | awq_marlin | 5.81 | 172 | 0.89x |
| w4a16_sampled | paged | 16 | 1024 | 256 | 7.41 | 2,160 | awq_marlin | 9.33 | 1,715 | 0.79x |
| w4a16_sampled | paged | 64 | 1024 | 256 | 16.22 | 3,946 | awq_marlin | 18.11 | 3,535 | 0.90x |

## Where a nano decode step goes (nsys)

| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---|
| nano_w4a16_sampled_b1 | 5.26 | 5.21 | 0.05 | 332 | 4 | gemm 82%, attention 8%, norm 4%, rope 2% |
| nano_w4a16_sampled_b16 | 7.33 | 7.28 | 0.05 | 334 | 4 | gemm 62%, attention 28%, sampling 5%, norm 3% |
| nano_w4a16_sampled_b64 | 15.99 | 15.93 | 0.06 | 302 | 4 | gemm 44%, attention 43%, sampling 10%, norm 1% |
| vllmw4_b1 | 5.66 | 5.58 | 0.08 | 366 | 15 | other 68%, gemm 13%, attention 11%, norm 4% |
| vllmw4_b16 | 7.79 | 7.71 | 0.08 | 429 | 15 | other 52%, attention 29%, gemm 10%, elementwise/copy 4% |
| vllmw4_b64 | 15.77 | 15.69 | 0.08 | 428 | 15 | other 45%, attention 44%, gemm 5%, elementwise/copy 3% |

