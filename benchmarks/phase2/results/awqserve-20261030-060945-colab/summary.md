# Phase 2 run awqserve-20261030-060945-colab

| step | rc | seconds |
|---|---:|---:|
| tests | 0 | 300 |
| serve | 0 | 911 |
| nsys | 0 | 99 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 172 | 1502 | 1941 |
| nano_awq | 204 | 1679 | 2107 |
| nano_awq_c512 | 204 | 1660 | 2052 |
| nano_w1c16 | - | 916 | - |
| nano_w1c64 | - | - | 1084 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 21.0 | 134.7 | 574.6 |
| nano_awq | 14.4 | 108.6 | 286.8 |
| nano_awq_c512 | 14.7 | 119.0 | 206.6 |
| nano_w1c16 | - | 264.2 | - |
| nano_w1c64 | - | - | 3129.6 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 5.75 | 7.88 | 16.35 |
| nano_awq | 4.85 | 6.79 | 15.01 |
| nano_awq_c512 | 4.84 | 6.80 | 15.37 |
| nano_w1c16 | - | 6.77 | - |
| nano_w1c64 | - | - | 14.60 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 6.08 | 75.20 | 186.18 |
| nano_awq | 5.16 | 86.65 | 93.84 |
| nano_awq_c512 | 5.51 | 44.85 | 49.96 |
| nano_w1c16 | - | 87.95 | - |
| nano_w1c64 | - | - | 93.12 |

## W1u: random 1024/128, prompts no earlier run sent (uncached)

**output tok/s**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 154 | 881 |
| nano_awq | 180 | 933 |
| nano_awq_c512 | 180 | 925 |

**TTFT p50 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 95.4 | 645.1 |
| nano_awq | 89.1 | 350.4 |
| nano_awq_c512 | 92.1 | 200.4 |

**ITL p50 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 5.77 | 7.83 |
| nano_awq | 4.86 | 6.74 |
| nano_awq_c512 | 4.85 | 6.73 |

**ITL p99 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm_awq | 6.20 | 166.90 |
| nano_awq | 5.18 | 87.78 |
| nano_awq_c512 | 5.04 | 44.90 |

## W1s: random 1024/256, temperature 0.7, top-p 0.9

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 156 | 794 | 1830 |
| nano_awq | 191 | 1415 | 2095 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 94.6 | 670.8 | 586.8 |
| nano_awq | 88.3 | 351.0 | 288.2 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 6.02 | 9.44 | 18.27 |
| nano_awq | 4.88 | 6.86 | 15.15 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm_awq | 6.27 | 165.84 | 187.62 |
| nano_awq | 5.26 | 87.64 | 94.11 |

## W3: prefill interference

Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. ITL columns are the background's.

| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_awq | 8.08 | 12.2 | 130.5 | 159.2 | 3639 | 351.9 | 658.5 |
| nano_awq_c512 | 8.10 | 29.3 | 47.9 | 52.0 | 3590 | 433.2 | 804.7 |
| vllm_awq | 9.23 | 11.6 | 161.9 | 184.9 | 3193 | 398.6 | 799.9 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_w1c16 | 5.29 | 5.24 | 0.05 | 258 | collect 4105, forward 36503, replay 756, sample 246, detokenize 19, prepare 2007, postprocess 60 |
| nano_w1c64 | 9.20 | 9.16 | 0.04 | 208 | collect 7568, forward 34805, replay 1054, detokenize 20, prepare 1839, sample 319, postprocess 177 |

## Server settings from the logs

- vllm_awq_server: GPU KV cache size: 237,824 tokens

