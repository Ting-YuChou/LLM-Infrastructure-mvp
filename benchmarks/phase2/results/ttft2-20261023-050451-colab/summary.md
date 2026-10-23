# Phase 2 run ttft2-20261023-050451-colab

| step | rc | seconds |
|---|---:|---:|
| tests | 0 | 112 |
| nsys | 0 | 89 |
| summarize | 0 | 0 |

## W1: random 1024/256

**output tok/s**

| config | c=1 |
|---|---:|
| nano_w1c1c | 81 |

**TTFT p50 (ms)**

| config | c=1 |
|---|---:|
| nano_w1c1c | 22.6 |

**ITL p50 (ms)**

| config | c=1 |
|---|---:|
| nano_w1c1c | 12.18 |

**ITL p99 (ms)**

| config | c=1 |
|---|---:|
| nano_w1c1c | 12.80 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_w1c1c | 12.11 | 12.07 | 0.04 | 421 | collect 11061, replay 800, sample 183, forward 22438, detokenize 36, pieces 3966, publish 16 |

