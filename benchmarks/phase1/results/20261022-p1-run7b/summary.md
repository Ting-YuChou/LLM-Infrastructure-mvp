# Phase 1 results: run 20261022-p1-run7b

## Steps

| step | rc | seconds |
|---|---:|---:|
| ncu | ok | 55 |
| environment | ok | 2 |

## Environment

```
run_id: 20261022-p1-run7b
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
RmProfilingAdminOnly: 1
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | triton 3.6.0
CUDA_HOME=/usr/local/cuda-12.8; nvcc: Build cuda_12.8.r12.8/compiler.35583870_0
NVIDIA Nsight Systems version 2025.3.1.0
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Kernels under ncu

### ncu kbench_w4a16_linear: 45 kernels, 4010 us, 375 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `void <unnamed>::gemv_w4a16_kernel<__nv_bfloat16, 8, 2>` | gemm | 6 | 26 | 176.5 | 12 | 185 | 30 | 23/25 | 104 | 45 | long_scoreboard, selected |
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 4>` | gemm | 6 | 20 | 134.2 | 17 | 266 | 21 | 12/12 | 190 | 75 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 4, 2>` | gemm | 3 | 18 | 243.7 | 9 | 142 | 16 | 12/12 | 188 | 84 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_mma_kernel<__nv_bfloat16, 1, 4>` | gemm | 9 | 11 | 49.0 | 43 | 674 | 30 | 15/19 | 130 | 41 | long_scoreboard, wait |
| `void <unnamed>::gemm_w4a16_wide_kernel<__nv_bfloat16, 1>` | gemm | 6 | 8 | 54.2 | 39 | 609 | 27 | 13/25 | 124 | 32 | long_scoreboard, wait |
| `void <unnamed>::gemv_w4a16_kernel<__nv_bfloat16, 1, 2>` | gemm | 6 | 6 | 40.3 | 52 | 805 | 54 | 26/62 | 48 | 12 | long_scoreboard, selected |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 6 | 5 | 34.0 | 62 | 961 | 29 | 12/12 | 138 | 20 | long_scoreboard, wait |
| `void marlin::Marlin<1125899906909960, 1125899907892224, 1125` | other | 3 | 5 | 67.4 | 34 | 521 | 36 | 6/6 | 255 | 73 | wait, selected |

