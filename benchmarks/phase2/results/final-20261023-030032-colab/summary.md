# Phase 2 run final-20261023-030032-colab

| step | rc | seconds |
|---|---:|---:|
| tests | 0 | 30 |
| serve | 0 | 2373 |
| nsys | 0 | 223 |
| offline | 0 | 116 |
| summarize | 0 | 0 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 78 | 152 | 297 | 571 | 939 | 1411 | 1976 |
| sglang | 78 | 151 | 296 | 570 | 940 | 1424 | 1284 |
| nano_final | 82 | 162 | 316 | 596 | 975 | 1470 | 1997 |
| nano_final_b512 | 82 | 162 | 316 | 598 | 963 | 1453 | 1972 |
| nano_m2_coldjit | 79 | 154 | 297 | 552 | 852 | 1234 | 1618 |
| nano_w1c16 | - | - | - | - | 701 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 1081 |

**TTFT p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 26.3 | 40.7 | 57.0 | 75.0 | 142.2 | 225.7 | 407.9 |
| sglang | 53.0 | 94.7 | 78.2 | 82.1 | 152.1 | 169.9 | 2235.0 |
| nano_final | 32.6 | 57.3 | 50.0 | 73.7 | 118.4 | 220.6 | 463.3 |
| nano_final_b512 | 32.1 | 56.0 | 49.1 | 73.2 | 122.8 | 170.1 | 341.7 |
| nano_m2_coldjit | 32.2 | 46.9 | 57.3 | 67.0 | 137.7 | 213.3 | 672.0 |
| nano_w1c16 | - | - | - | - | 226.2 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 2553.4 |

**ITL p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 12.81 | 13.06 | 13.28 | 13.75 | 14.87 | 18.50 | 23.29 |
| sglang | 12.61 | 12.92 | 13.22 | 13.75 | 14.75 | 18.10 | 34.31 |
| nano_final | 12.08 | 12.23 | 12.47 | 13.19 | 14.15 | 17.36 | 22.65 |
| nano_final_b512 | 12.08 | 12.23 | 12.47 | 13.17 | 14.16 | 17.40 | 22.77 |
| nano_m2_coldjit | 12.56 | 12.78 | 13.24 | 14.27 | 15.95 | 20.61 | 28.79 |
| nano_w1c16 | - | - | - | - | 14.13 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 22.23 |

**ITL p99 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 13.07 | 13.57 | 13.92 | 14.37 | 69.42 | 133.41 | 147.06 |
| sglang | 12.86 | 13.46 | 13.82 | 14.25 | 15.33 | 19.43 | 37.84 |
| nano_final | 12.43 | 12.70 | 12.95 | 13.62 | 74.82 | 76.91 | 80.85 |
| nano_final_b512 | 12.28 | 12.45 | 12.93 | 13.61 | 38.39 | 40.07 | 44.13 |
| nano_m2_coldjit | 12.84 | 13.07 | 13.73 | 14.86 | 42.44 | 45.40 | 52.12 |
| nano_w1c16 | - | - | - | - | 75.22 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 79.76 |

## W2: shared prefixes

w2a: one 2048-token prefix + 128 tokens, output 128, c=16. w2b: 48 prefixes of 4096 tokens + 256, output 128, c=16. Hit rate: prompt tokens served from the prefix cache during the run.

| config | run | ok | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 | hit rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nano_final | w2a | yes | 7.84 | 1004 | 165.0 | 373.7 | 13.87 | 92.6% |
| nano_final_b512 | w2a | yes | 7.97 | 1020 | 116.7 | 408.4 | 13.86 | 92.6% |
| sglang | w2a | yes | 7.20 | 921 | 205.8 | 607.9 | 14.95 | - |
| vllm | w2a | yes | 7.32 | 936 | 269.6 | 382.4 | 14.92 | 92.6% |
| nano_final | w2b | yes | 3.34 | 428 | 401.7 | 3958.1 | 18.65 | 48.5% |
| nano_final_b512 | w2b | yes | 3.33 | 427 | 477.1 | 4713.2 | 32.68 | 48.4% |
| sglang | w2b | yes | 3.02 | 387 | 1709.7 | 3803.0 | 19.29 | - |
| vllm | w2b | yes | 3.28 | 420 | 579.6 | 3850.8 | 19.60 | 50.6% |

## W3: prefill interference

Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. ITL columns are the background's.

| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_final | 16.36 | 29.4 | 147.3 | 151.3 | 1820 | 355.7 | 685.6 |
| nano_final_b512 | 16.41 | 43.2 | 47.2 | 50.2 | 1821 | 423.1 | 851.7 |
| nano_w3 | 16.34 | 33.2 | 148.3 | 152.5 | 1819 | 355.8 | 679.5 |
| sglang | 17.05 | 34.1 | 319.2 | 592.8 | 1729 | 349.2 | 619.7 |
| vllm | 17.22 | 28.3 | 148.2 | 152.1 | 1742 | 358.6 | 685.1 |
| vllm_mbt512 | 17.22 | 44.7 | 50.9 | 53.0 | 1754 | 433.8 | 919.0 |

## Offline decode (nano_bench, prefill-first, CUDA graphs)

| async | batch | in/out | decode ms/step | p50 | tok/s |
|---|---:|---|---:|---:|---:|
| yes | 1 | 128/256 | 12.00 | 12.04 | 83 |
| yes | 16 | 1024/256 | 15.03 | 14.10 | 1065 |
| yes | 64 | 1024/256 | 23.38 | 22.42 | 2737 |
| no | 1 | 128/256 | 12.27 | 12.27 | 81 |
| no | 16 | 1024/256 | 14.55 | 14.53 | 1100 |
| no | 64 | 1024/256 | 23.55 | 23.57 | 2717 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_offline_async_b16 | 14.06 | 14.02 | 0.04 | 425 | collect 12911, replay 806, sample 184, schedule 44, postprocess 32 |
| nano_offline_sync_b16 | 14.95 | 14.03 | 0.92 | 428 | collect 13808, replay 792, sample 187, schedule 45, postprocess 31 |
| nano_w1c16 | 14.02 | 13.98 | 0.04 | 423 | collect 12903, forward 29042, replay 759, sample 183, detokenize 19, prepare 1401, postprocess 54 |
| nano_w1c64 | 21.95 | 21.93 | 0.03 | 329 | collect 20363, forward 29884, detokenize 19, prepare 2073, replay 977, sample 198, postprocess 164 |
| nano_w3 | 16.21 | 16.18 | 0.03 | 362 | collect 14817, detokenize 20, replay 789, forward 29322, sample 194, postprocess 134, schedule 146 |

## Server settings from the logs

- sglang_server: max_total_num_tokens=139721, chunked_prefill_size=4096
- vllm_mbt512_server: max_num_batched_tokens=512, GPU KV cache size: 163,264 tokens
- vllm_server: GPU KV cache size: 163,168 tokens

