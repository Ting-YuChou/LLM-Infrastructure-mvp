# Phase 2 run kernels-serve-20261023-075342-colab

| step | rc | seconds |
|---|---:|---:|
| tests | 0 | 231 |
| serve | 0 | 1244 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 78 | 940 | 1777 |
| vllm_fp8 | 76 | 952 | 1915 |
| nano | 82 | 976 | 1839 |
| nano_fp8 | 83 | 985 | 1880 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 26.3 | 140.1 | 495.3 |
| vllm_fp8 | 27.4 | 144.0 | 469.7 |
| nano | 22.2 | 131.9 | 250.2 |
| nano_fp8 | 25.2 | 154.4 | 259.4 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 12.80 | 14.87 | 23.24 |
| vllm_fp8 | 13.12 | 14.57 | 21.38 |
| nano | 12.08 | 14.13 | 22.65 |
| nano_fp8 | 12.04 | 13.81 | 21.00 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 13.12 | 61.55 | 149.56 |
| vllm_fp8 | 13.57 | 59.66 | 138.31 |
| nano | 12.34 | 74.97 | 81.59 |
| nano_fp8 | 12.34 | 78.95 | 84.69 |

## W1s: random 1024/256, temperature 0.7, top-p 0.9

**output tok/s**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 75 | 609 | 1693 |
| nano | 78 | 855 | 1711 |

**TTFT p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 79.7 | 488.7 | 509.0 |
| nano | 75.2 | 229.9 | 256.1 |

**ITL p50 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 13.07 | 16.28 | 25.03 |
| nano | 12.50 | 15.17 | 25.31 |

**ITL p99 (ms)**

| config | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| vllm | 13.30 | 129.72 | 150.81 |
| nano | 12.86 | 76.36 | 83.92 |

## W2: shared prefixes

w2a: one 2048-token prefix + 128 tokens, output 128, c=16. w2b: 48 prefixes of 4096 tokens + 256, output 128, c=16. Hit rate: prompt tokens served from the prefix cache during the run.

| config | run | ok | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 | hit rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nano | w2b | yes | 3.33 | 426 | 404.7 | 3946.2 | 18.66 | 48.2% |
| nano_fp8 | w2b | yes | 3.99 | 511 | 159.5 | 4652.7 | 17.19 | 70.6% |
| vllm | w2b | yes | 3.22 | 412 | 544.1 | 3847.2 | 19.68 | 48.9% |
| vllm_fp8 | w2b | yes | 4.16 | 532 | 258.8 | 4096.3 | 18.11 | 70.6% |

## Server settings from the logs

- vllm_fp8_server: GPU KV cache size: 326,352 tokens
- vllm_server: GPU KV cache size: 159,072 tokens

