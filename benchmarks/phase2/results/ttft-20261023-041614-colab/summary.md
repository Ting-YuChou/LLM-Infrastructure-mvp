# Phase 2 run ttft-20261023-041614-colab

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 262 |
| install_sglang | 0 | 273 |
| install_nsys | 0 | 0 |
| download_model | 0 | 42 |
| cuda129 | 0 | 30 |
| tests | 1 | 189 |
| serve | 0 | 1708 |
| nsys | 0 | 129 |
| summarize | 0 | 0 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 77 | 152 | 297 | 571 | 939 | 1398 | 1891 |
| nano_pw | 82 | 162 | 316 | 597 | 975 | 1472 | 1985 |
| nano_pw_b512 | 82 | 162 | 316 | 598 | 963 | 1450 | 1974 |
| nano_nopw | 82 | 162 | 317 | 594 | 974 | 1463 | 1998 |
| nano_w1c1 | 79 | - | - | - | - | - | - |
| nano_w1c64 | - | - | - | - | - | - | 1083 |

**TTFT p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 26.6 | 40.7 | 57.1 | 73.2 | 140.7 | 235.6 | 667.6 |
| nano_pw | 22.4 | 39.9 | 44.8 | 53.2 | 131.6 | 211.2 | 407.1 |
| nano_pw_b512 | 22.4 | 39.9 | 45.1 | 51.8 | 131.9 | 170.2 | 519.2 |
| nano_nopw | 33.0 | 57.9 | 51.7 | 75.8 | 117.2 | 232.8 | 535.3 |
| nano_w1c1 | 76.6 | - | - | - | - | - | - |
| nano_w1c64 | - | - | - | - | - | - | 2524.3 |

**ITL p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 12.81 | 13.05 | 13.28 | 13.76 | 14.88 | 18.50 | 23.35 |
| nano_pw | 12.08 | 12.23 | 12.46 | 13.18 | 14.14 | 17.37 | 22.69 |
| nano_pw_b512 | 12.08 | 12.23 | 12.46 | 13.17 | 14.16 | 17.38 | 22.78 |
| nano_nopw | 12.08 | 12.23 | 12.46 | 13.18 | 14.14 | 17.36 | 22.69 |
| nano_w1c1 | 12.18 | - | - | - | - | - | - |
| nano_w1c64 | - | - | - | - | - | - | 22.25 |

**ITL p99 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 13.15 | 13.57 | 13.94 | 14.45 | 61.29 | 134.24 | 149.48 |
| nano_pw | 12.26 | 12.57 | 13.01 | 14.45 | 75.28 | 77.34 | 81.47 |
| nano_pw_b512 | 12.40 | 12.53 | 12.97 | 14.40 | 39.75 | 41.16 | 45.13 |
| nano_nopw | 12.30 | 12.46 | 12.91 | 13.52 | 75.27 | 77.51 | 81.30 |
| nano_w1c1 | 12.76 | - | - | - | - | - | - |
| nano_w1c64 | - | - | - | - | - | - | 80.41 |

## W1u: random 1024/128, prompts no earlier run sent (uncached)

**output tok/s**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm | 75 | 704 |
| nano_pw | 79 | 716 |
| nano_pw_b512 | 79 | 711 |
| nano_nopw | 79 | 716 |

**TTFT p50 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm | 81.8 | 395.4 |
| nano_pw | 76.1 | 227.3 |
| nano_pw_b512 | 82.0 | 158.0 |
| nano_nopw | 76.5 | 227.4 |

**ITL p50 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm | 12.82 | 14.82 |
| nano_pw | 12.09 | 14.07 |
| nano_pw_b512 | 12.09 | 14.11 |
| nano_nopw | 12.09 | 14.06 |

**ITL p99 (ms)**

| config | c=1 | c=16 |
|---|---:|---:|
| vllm | 13.21 | 131.35 |
| nano_pw | 12.54 | 76.01 |
| nano_pw_b512 | 12.38 | 39.96 |
| nano_nopw | 12.32 | 76.05 |

## W2: shared prefixes

w2a: one 2048-token prefix + 128 tokens, output 128, c=16. w2b: 48 prefixes of 4096 tokens + 256, output 128, c=16. Hit rate: prompt tokens served from the prefix cache during the run.

| config | run | ok | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 | hit rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nano_pw | w2a | yes | 7.85 | 1005 | 171.2 | 366.8 | 13.85 | 92.6% |
| nano_pw | w2b | yes | 3.33 | 426 | 410.9 | 3657.4 | 18.63 | 48.5% |

## W3: prefill interference

Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. ITL columns are the background's.

| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_pw | 16.36 | 28.9 | 148.4 | 152.4 | 1819 | 355.7 | 681.1 |
| nano_pw_b512 | 16.41 | 43.9 | 48.2 | 55.8 | 1819 | 427.4 | 865.0 |
| vllm | 17.22 | 29.6 | 149.5 | 153.8 | 1742 | 369.2 | 699.3 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_w1c1 | 12.11 | 12.07 | 0.04 | 421 | collect 11084, replay 768, sample 183, forward 22355, detokenize 35, publish 15, postprocess 11 |
| nano_w1c64 | 21.96 | 21.93 | 0.03 | 329 | collect 20153, forward 29262, detokenize 19, replay 1113, prepare 1800, sample 214, postprocess 184 |

## Server settings from the logs

- vllm_server: GPU KV cache size: 159,072 tokens

