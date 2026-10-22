# Phase 0 results: run 20261022-colab-a100-40gb

## Environment

```
run_id: 20261022-colab-a100-40gb
name, driver_version, memory.total [MiB], clocks.max.sm [MHz], clocks.max.memory [MHz], power.limit [W]
NVIDIA A100-SXM4-40GB, 580.82.07, 40960 MiB, 1410 MHz, 1215 MHz, 400.00 W
	[4mGPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	0-11	0		N/A

Legend:

RmProfilingAdminOnly: 1
CPU(s):                                  12
Model name:                              Intel(R) Xeon(R) CPU @ 2.20GHz
               total        used        free      shared  buff/cache   available
Mem:              83           2          52           0          30          81
vllm 0.19.1 | torch 2.10.0+cu128 cuda 12.8 | transformers 5.20.2
sglang 0.5.22 | torch 2.14.1+cu130
flashinfer 0.7.1
NVIDIA Nsight Systems version 2025.3.1.0
repo: uploaded working tree
Version 2025.3.1.0 (build 36398880) (public-release)
```

## Batch=1 decode

| what | ms/token | tok/s | % of roofline |
|---|---:|---:|---:|
| roofline, datasheet 1,555 GB/s | 9.67 | 103 | 100% |
| roofline, measured copy 1,380 GB/s | 10.90 | 92 | 89% |
| cuBLAS linear layers only (M=1, no attention) | 12.49 | 80 | 77% |
| vllm serve, TPOT p50 (in 128 / out 256) | 12.97 | 77 | 75% |
| sglang serve, TPOT p50 (in 128 / out 256) | 12.61 | 79 | 77% |
| vllm_eager_b1 offline engine step | 22.68 | 44 | 43% |
| vllm_graph_b1 offline engine step | 12.62 | 79 | 77% |
| nano_vllm static (in 128) | 44.10 | 23 | 22% |
| nano_vllm paged (in 128) | 53.63 | 19 | 18% |
| nano_vllm paged (in 1024) | 61.94 | 16 | 16% |

## Serving sweep (random, in 1024 / out 256, ignore_eos)

| engine | conc | out tok/s | roofline % | total tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sglang | 1 | 78 | 76% | 389 | 52.7 | 55.1 | 12.68 | 12.71 | 12.69 | 12.91 |
| sglang | 2 | 149 | 73% | 745 | 96.1 | 222.7 | 13.02 | 13.52 | 13.01 | 13.59 |
| sglang | 4 | 295 | 74% | 1,472 | 79.0 | 84.9 | 13.30 | 13.36 | 13.30 | 13.96 |
| sglang | 8 | 565 | 74% | 2,823 | 84.8 | 100.8 | 13.85 | 13.91 | 13.86 | 14.45 |
| sglang | 16 | 927 | 65% | 4,633 | 198.8 | 1,046.6 | 15.46 | 18.58 | 14.86 | 15.48 |
| sglang | 32 | 1,409 | 56% | 7,039 | 171.6 | 2,076.6 | 18.51 | 25.79 | 18.27 | 19.37 |
| sglang | 64 | 1,316 | 33% | 6,576 | 1,993.9 | 4,175.5 | 40.02 | 49.60 | 34.57 | 38.74 |
| sglang | 128 | 1,658 | 41% | 8,283 | 4,398.2 | 17,554.6 | 48.43 | 61.94 | 35.64 | 42.82 |
| vllm | 1 | 76 | 74% | 381 | 26.4 | 29.2 | 13.07 | 13.11 | 13.07 | 13.32 |
| vllm | 2 | 146 | 72% | 729 | 40.6 | 49.5 | 13.60 | 13.65 | 13.60 | 14.25 |
| vllm | 4 | 286 | 72% | 1,427 | 44.7 | 58.7 | 13.86 | 13.90 | 13.86 | 14.69 |
| vllm | 8 | 542 | 71% | 2,706 | 75.3 | 91.9 | 14.51 | 14.81 | 14.26 | 15.94 |
| vllm | 16 | 888 | 62% | 4,437 | 144.7 | 988.1 | 16.36 | 19.36 | 15.84 | 64.41 |
| vllm | 32 | 1,332 | 53% | 6,656 | 234.0 | 1,683.7 | 21.44 | 26.68 | 19.74 | 136.38 |
| vllm | 64 | 1,840 | 46% | 9,195 | 606.3 | 3,827.4 | 27.25 | 40.10 | 24.74 | 151.77 |
| vllm | 128 | 1,725 | 42% | 8,616 | 737.4 | 9,818.7 | 71.52 | 72.69 | 34.11 | 186.21 |

sglang server ready after 152 s.

vllm server ready after 86 s.

## Where a decode step goes (nsys)

| capture | batch | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |
|---|---:|---:|---:|---:|---:|---:|---|
| nano_paged_b1 | 1 | 69.88 | 19.87 | 50.01 | 1604 | 1604 | gemm 72%, elementwise/copy 24%, norm 1%, other 1% |
| nano_paged_b16 | 16 | 807.21 | 128.20 | 679.02 | 17619 | 17619 | elementwise/copy 70%, gemm 25%, attention 4%, sampling 1% |
| nano_static_b1 | 1 | 59.37 | 19.08 | 40.30 | 1573 | 1573 | gemm 73%, elementwise/copy 22%, other 2%, norm 1% |
| vllm_eager_b1 | 1 | 28.74 | 12.87 | 15.87 | 369 | 369 | gemm 89%, attention 5%, norm 2%, activation 2% |
| vllm_graph_b1 | 1 | 12.85 | 12.57 | 0.28 | 365 | 15 | gemm 91%, attention 5%, norm 2%, elementwise/copy 1% |
| vllm_graph_b64 | 64 | 22.69 | 22.62 | 0.07 | 463 | 15 | gemm 65%, attention 31%, elementwise/copy 2%, norm 1% |

## Decode kernels under ncu (one decode step)

### ncu nano_static_b1: 300 kernels, 3460 us, 799 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn` | gemm | 12 | 34 | 97.3 | 79 | 1232 | 26 | 6/- | 158 | 21 | long_scoreboard, wait |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_` | gemm | 19 | 29 | 53.0 | 73 | 1175 | 24 | 6/- | 146 | 12 | long_scoreboard, wait |
| `std::enable_if<!T7, void>::type internal::kernel<int, int, _` | other | 20 | 6 | 10.0 | 36 | 621 | 13 | 11/19 | 164 | 15 | long_scoreboard, lg_throttle |
| `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl` | elementwise/copy | 24 | 4 | 5.4 | 0 | 5 | 0 | 6/100 | 16 | 78 | long_scoreboard, imc_miss |
| `void at::index_elementwise_kernel<128, 4, void at::gpu_index` | elementwise/copy | 12 | 3 | 7.7 | 0 | 6 | 0 | 6/75 | 36 | 78 | imc_miss, long_scoreboard |
| `void at::<unnamed>::CatArrayBatchedCopy<at::<unnamed>::Opaqu` | elementwise/copy | 12 | 2 | 6.4 | 0 | 2 | 4 | 82/100 | 23 | 81 | imc_miss, no_instruction |
| `void at::vectorized_elementwise_kernel<4, at::CUDAFunctor_ad` | elementwise/copy | 24 | 2 | 2.9 | 0 | 7 | 0 | 6/100 | 32 | 80 | imc_miss, long_scoreboard |
| `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl` | elementwise/copy | 12 | 2 | 5.7 | 3 | 50 | 23 | 51/100 | 16 | 84 | long_scoreboard, imc_miss |
| `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl` | elementwise/copy | 12 | 2 | 5.3 | 0 | 4 | 0 | 6/100 | 16 | 87 | long_scoreboard, imc_miss |
| `void at::unrolled_elementwise_kernel<at::direct_copy_kernel_` | elementwise/copy | 12 | 2 | 4.9 | 1 | 7 | 0 | 6/100 | 32 | 78 | imc_miss, no_instruction |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat1` | gemm | 12 | 2 | 4.8 | 1 | 19 | 10 | 39/100 | 32 | 65 | imc_miss, no_instruction |
| `void at::reduce_kernel<512, 1, at::ReduceOp<long, at::func_w` | other | 12 | 2 | 4.7 | 1 | 12 | 0 | 2/50 | 32 | 78 | imc_miss, no_instruction |
| `void at::vectorized_elementwise_kernel<4, at::bfloat16_copy_` | elementwise/copy | 18 | 1 | 2.8 | 0 | 4 | 0 | 6/100 | 28 | 87 | imc_miss, long_scoreboard |
| `_rms_norm_kernel` | norm | 13 | 1 | 3.8 | 0 | 6 | 0 | 6/50 | 61 | 84 | imc_miss, long_scoreboard |
| `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl` | elementwise/copy | 6 | 1 | 5.1 | 0 | 5 | 0 | 6/100 | 16 | 78 | long_scoreboard, imc_miss |

### ncu vllm_graph_b1: 367 kernels, 13387 us, 1147 GB/s DRAM over kernel time

| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_` | gemm | 33 | 47 | 190.7 | 88 | 1375 | 28 | 12/12 | 146 | 23 | long_scoreboard, wait |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_` | gemm | 64 | 31 | 65.0 | 75 | 1198 | 25 | 6/- | 146 | 15 | long_scoreboard, wait |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_re` | gemm | 32 | 11 | 44.0 | 77 | 1198 | 26 | 6/6 | 97 | 15 | long_scoreboard, wait |
| `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kerne` | attention | 32 | 3 | 13.7 | 1 | 21 | 0 | 6/56 | 52 | 44 | long_scoreboard, wait |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits` | attention | 32 | 2 | 9.9 | 4 | 61 | 3 | 7/12 | 252 | 52 | imc_miss, long_scoreboard |
| `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __n` | attention | 32 | 1 | 4.9 | 0 | 6 | 0 | 19/50 | 54 | 76 | imc_miss, long_scoreboard |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2` | norm | 32 | 1 | 4.6 | 1 | 8 | 0 | 24/100 | 30 | 69 | long_scoreboard, imc_miss |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | norm | 32 | 1 | 4.2 | 0 | 7 | 0 | 24/100 | 27 | 76 | long_scoreboard, imc_miss |
| `triton_poi_fused_3` | elementwise/copy | 31 | 1 | 3.8 | 0 | 6 | 0 | 6/62 | 47 | 77 | long_scoreboard, imc_miss |
| `triton_poi_fused_mul_silu_slice_1` | activation | 32 | 1 | 3.1 | 1 | 20 | 1 | 6/100 | 31 | 64 | imc_miss, long_scoreboard |
| `void at::reduce_kernel<512, 1, at::ReduceOp<float, at::ArgMa` | sampling | 1 | 0 | 55.6 | 1 | 10 | 0 | 25/100 | 32 | 43 | long_scoreboard, wait |
| `void at::unrolled_elementwise_kernel<at::direct_copy_kernel_` | elementwise/copy | 4 | 0 | 4.2 | 1 | 23 | 1 | 8/100 | 32 | 72 | imc_miss, no_instruction |
| `void at::index_elementwise_kernel<128, 4, void at::gpu_index` | elementwise/copy | 2 | 0 | 6.4 | 0 | 7 | 0 | 5/75 | 36 | 80 | imc_miss, long_scoreboard |
| `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0` | norm | 1 | 0 | 5.7 | 0 | 4 | 0 | 25/100 | 32 | 89 | long_scoreboard, imc_miss |
| `_compute_slot_mapping_kernel` | other | 1 | 0 | 4.9 | 0 | 3 | 0 | 6/50 | 64 | 94 | long_scoreboard, imc_miss |

## nano_vllm today

| path | batch | in | ms/step | tok/s | model ms/step | engine overhead ms |
|---|---:|---:|---:|---:|---:|---:|
| static | 1 | 128 | 44.10 | 23 | - | - |
| paged | 1 | 128 | 53.63 | 19 | 53.40 | 0.23 |
| paged | 1 | 1024 | 61.94 | 16 | 61.70 | 0.23 |
| paged | 4 | 1024 | 173.92 | 23 | 173.60 | 0.32 |
| paged | 16 | 1024 | 618.92 | 26 | 618.50 | 0.42 |

