# Phase 1 results: run marlin-t1b-20261023-093008-colab

## Steps

| step | rc | seconds |
|---|---:|---:|
| tune | **1** | 3 |
| ncu | ok | 107 |
| environment | ok | 2 |

## Environment

```
run_id: marlin-t1b-20261023-093008-colab
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=/usr/local/cuda-12.8; nvcc: Build cuda_12.8.r12.8/compiler.35583870_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Kernels under ncu

### ncu kbench_w4a16_linear: 99 kernels, 7949 us, 468 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 4>` | gemm | 18 | 30 | 132.3 | 17 | 293 | 21 | 12/12 | 190 | 77 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 4, 2>` | gemm | 9 | 24 | 213.9 | 10 | 179 | 19 | 12/12 | 188 | 85 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 4, 64` | gemm | 9 | 9 | 81.4 | 28 | 506 | 29 | 12/12 | 238 | 76 | wait, math_pipe_throttle |
| `void <unnamed>::gemv_w4a16_kernel<__nv_bfloat16, 1, 2>` | gemm | 18 | 9 | 39.8 | 53 | 918 | 57 | 35/62 | 48 | 14 | long_scoreboard, not_selected |
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 1>` | gemm | 9 | 6 | 55.4 | 39 | 665 | 28 | 13/25 | 124 | 19 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 1, 4>` | gemm | 9 | 5 | 46.4 | 45 | 788 | 33 | 16/19 | 130 | 15 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_tiled_kernel<__nv_bfloat16, 1, 12` | gemm | 9 | 5 | 43.0 | 49 | 850 | 35 | 12/25 | 117 | 54 | wait, selected |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 9 | 4 | 35.7 | 56 | 973 | 27 | 12/12 | 128 | 15 | long_scoreboard, selected |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 6 | 4 | 49.8 | 26 | 449 | 29 | 6/6 | 255 | 73 | wait, selected |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 3 | 3 | 90.7 | 51 | 800 | 53 | 12/12 | 255 | 52 | wait, math_pipe_throttle |

