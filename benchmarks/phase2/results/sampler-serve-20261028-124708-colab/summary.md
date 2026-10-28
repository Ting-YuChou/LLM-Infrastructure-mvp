# Phase 2 run sampler-serve-20261028-124708-colab

| step | rc | seconds |
|---|---:|---:|
| serve | 0 | 682 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 78 | 937 | 1780 |
| nano | 83 | 979 | 1840 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 27.1 | 127.3 | 490.2 |
| nano | 22.3 | 132.9 | 249.2 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 12.78 | 14.83 | 23.22 |
| nano | 12.03 | 14.08 | 22.62 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 13.35 | 61.49 | 148.67 |
| nano | 12.33 | 74.50 | 81.26 |

## W1s: random 1024/256, temperature 0.7, top-p 0.9

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 75 | 612 | 1692 |
| nano | 81 | 904 | 1833 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 80.3 | 498.1 | 513.2 |
| nano | 75.0 | 226.9 | 248.9 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 13.05 | 16.27 | 25.00 |
| nano | 12.05 | 14.16 | 22.73 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 13.57 | 128.95 | 150.39 |
| nano | 12.40 | 75.40 | 81.38 |

## Server settings from the logs

- vllm_server: GPU KV cache size: 159,072 tokens

