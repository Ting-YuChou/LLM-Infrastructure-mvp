# Phase 2 run w4serve-20261029-211009-colab

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 276 |
| install_nsys | 0 | 0 |
| download_model | 0 | 44 |
| cuda129 | 0 | 24 |
| serve | 0 | 1104 |
| nsys | 0 | 101 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 167 | 1502 | 1949 |
| nano_w4 | 192 | 1629 | 2101 |
| nano_w4_c256 | 193 | 1596 | 1993 |
| nano_w1c16 | - | 898 | - |
| nano_w1c64 | - | - | 1085 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 21.1 | 140.2 | 568.9 |
| nano_w4 | 15.0 | 109.1 | 283.1 |
| nano_w4_c256 | 14.8 | 175.7 | 1877.3 |
| nano_w1c16 | - | 263.8 | - |
| nano_w1c64 | - | - | 3121.6 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 5.76 | 7.90 | 16.39 |
| nano_w4 | 5.16 | 7.11 | 15.09 |
| nano_w4_c256 | 5.14 | 7.10 | 27.32 |
| nano_w1c16 | - | 7.06 | - |
| nano_w1c64 | - | - | 14.66 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 6.19 | 76.10 | 183.64 |
| nano_w4 | 5.50 | 86.42 | 92.72 |
| nano_w4_c256 | 5.52 | 25.14 | 31.12 |
| nano_w1c16 | - | 87.31 | - |
| nano_w1c64 | - | - | 92.31 |

## W1u: random 1024/128, prompts no earlier run sent (uncached)

**output tok/s**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 154 | 887 |
| nano_w4 | 171 | 923 |
| nano_w4_c256 | 170 | 908 |

**TTFT p50 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 95.1 | 654.1 |
| nano_w4 | 89.1 | 347.5 |
| nano_w4_c256 | 95.5 | 175.1 |

**ITL p50 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 5.77 | 7.84 |
| nano_w4 | 5.16 | 7.03 |
| nano_w4_c256 | 5.15 | 9.44 |

**ITL p99 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 6.07 | 164.52 |
| nano_w4 | 5.81 | 87.38 |
| nano_w4_c256 | 5.32 | 25.51 |

## W1s: random 1024/256, temperature 0.7, top-p 0.9

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 156 | 796 | 1839 |
| nano_w4 | 181 | 1385 | 2102 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 94.2 | 612.7 | 597.3 |
| nano_w4 | 88.4 | 347.3 | 282.8 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 6.03 | 9.47 | 18.33 |
| nano_w4 | 5.18 | 7.16 | 15.21 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 6.30 | 164.11 | 185.61 |
| nano_w4 | 5.60 | 86.92 | 92.98 |

## W3: prefill interference

Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. ITL columns are the background's.

| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_w4 | 8.33 | 12.1 | 128.9 | 158.1 | 3538 | 356.6 | 656.8 |
| vllm_awq | 9.20 | 11.9 | 160.3 | 181.7 | 3209 | 416.2 | 782.6 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_w1c16 | 5.71 | 5.67 | 0.04 | 262 | collect 4474, forward 37080, replay 777, sample 255, detokenize 19, prepare 1789, postprocess 59 |
| nano_w1c64 | 9.52 | 9.48 | 0.04 | 201 | collect 7904, forward 35130, replay 1037, detokenize 19, prepare 1913, sample 310, postprocess 173 |

## Server settings from the logs

- vllm_awq_server: GPU KV cache size: 237,824 tokens

