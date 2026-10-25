# Phase 2 run m1to6-20261023-015904-colab

| step | rc | seconds |
|---|---:|---:|
| serve | 0 | 2542 |
| nsys | 0 | 229 |
| offline | 0 | 230 |
| summarize | 0 | 0 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_m2 | 79 | 154 | 298 | 551 | 719 | 1073 | 1551 |
| nano_m2_b2048 | 79 | 154 | 298 | 550 | 857 | 1236 | 1611 |
| nano_m3 | 80 | 158 | 309 | 580 | 905 | 1387 | 1883 |
| nano_m4 | 82 | 162 | 317 | 597 | 963 | 1451 | 1973 |
| nano_m4_inproc | 82 | 161 | 316 | 597 | 960 | 1444 | 1958 |
| nano_autokv | - | - | - | - | - | 1094 | - |
| nano_w1c16 | - | - | - | - | 669 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 940 |

**TTFT p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_m2 | 31.1 | 47.6 | 56.7 | 66.8 | 149.6 | 544.3 | 692.9 |
| nano_m2_b2048 | 31.2 | 47.1 | 56.8 | 67.4 | 158.6 | 191.2 | 704.8 |
| nano_m3 | 32.1 | 42.9 | 48.3 | 55.9 | 129.2 | 135.6 | 396.4 |
| nano_m4 | 31.7 | 57.9 | 49.7 | 75.2 | 122.2 | 172.3 | 329.9 |
| nano_m4_inproc | 31.2 | 58.3 | 57.9 | 67.3 | 135.7 | 173.5 | 665.8 |
| nano_autokv | - | - | - | - | - | 150.6 | - |
| nano_w1c16 | - | - | - | - | 168.1 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 3321.1 |

**ITL p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_m2 | 12.54 | 12.79 | 13.20 | 14.28 | 15.99 | 20.65 | 28.87 |
| nano_m2_b2048 | 12.58 | 12.81 | 13.22 | 14.31 | 15.99 | 20.61 | 28.62 |
| nano_m3 | 12.36 | 12.53 | 12.79 | 13.55 | 14.65 | 18.11 | 23.91 |
| nano_m4 | 12.08 | 12.23 | 12.47 | 13.18 | 14.16 | 17.38 | 22.78 |
| nano_m4_inproc | 12.08 | 12.23 | 12.47 | 13.18 | 14.15 | 17.40 | 22.80 |
| nano_autokv | - | - | - | - | - | 20.74 | - |
| nano_w1c16 | - | - | - | - | 14.16 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 31.66 |

**ITL p99 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_m2 | 12.75 | 13.24 | 13.68 | 14.98 | 42.46 | 45.72 | 53.25 |
| nano_m2_b2048 | 12.84 | 13.28 | 13.75 | 14.72 | 77.91 | 81.66 | 89.69 |
| nano_m3 | 12.59 | 13.05 | 13.27 | 14.30 | 41.47 | 43.27 | 47.39 |
| nano_m4 | 12.36 | 12.69 | 12.93 | 13.62 | 38.33 | 40.22 | 44.15 |
| nano_m4_inproc | 12.33 | 12.48 | 12.96 | 13.70 | 38.36 | 40.36 | 44.95 |
| nano_autokv | - | - | - | - | - | 45.70 | - |
| nano_w1c16 | - | - | - | - | 35.48 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 42.82 |

## W2: shared prefixes

w2a: one 2048-token prefix + 128 tokens, output 128, c=16. w2b: 48 prefixes of 4096 tokens + 256, output 128, c=16. Hit rate: prompt tokens served from the prefix cache during the run.

| config | run | ok | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 | hit rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nano_m1 | w2a | yes | 6.91 | 884 | 267.8 | 436.2 | 15.50 | 92.6% |
| nano_m2 | w2a | yes | 6.73 | 861 | 92.4 | 557.9 | 15.63 | 92.6% |
| nano_m4 | w2a | yes | 7.94 | 1016 | 117.1 | 438.8 | 13.85 | 92.6% |
| nano_m6 | w2a | yes | 7.89 | 1010 | 111.4 | 479.4 | 13.85 | 92.6% |
| vllm | w2a | yes | 7.42 | 949 | 200.1 | 412.9 | 14.95 | 92.6% |
| nano_m1 | w2b | yes | 2.71 | 346 | 1723.3 | 4784.6 | 20.21 | 49.6% |
| nano_m2 | w2b | yes | 3.10 | 398 | 471.0 | 5087.4 | 43.15 | 48.4% |
| nano_m4 | w2b | yes | 3.33 | 426 | 475.9 | 4406.5 | 32.65 | 48.2% |
| nano_m6 | w2b | yes | 3.34 | 428 | 474.6 | 4730.0 | 32.83 | 48.4% |
| vllm | w2b | yes | 3.27 | 419 | 552.2 | 3541.5 | 19.65 | 50.3% |

## W3: prefill interference

Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. ITL columns are the background's.

| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano_m2 | 20.19 | 49.8 | 54.1 | 55.0 | 1485 | 467.0 | 929.6 |
| nano_m2_b1024 | 19.92 | 76.6 | 91.4 | 92.6 | 1498 | 400.0 | 799.5 |
| nano_m2_b2048 | 20.04 | 35.8 | 153.6 | 155.7 | 1493 | 354.5 | 700.6 |
| nano_m2_b256 | 20.05 | 36.3 | 38.5 | 49.7 | 1480 | 678.4 | 1334.6 |
| nano_m3 | 17.27 | 45.4 | 50.4 | 60.0 | 1733 | 427.9 | 880.1 |
| nano_m4 | 16.43 | 43.1 | 47.5 | 50.4 | 1822 | 422.2 | 851.6 |
| nano_w3 | 16.42 | 43.8 | 48.3 | 55.2 | 1809 | 419.6 | 860.6 |

## Offline decode (nano_bench, prefill-first, CUDA graphs)

| async | batch | in/out | decode ms/step | p50 | tok/s |
|---|---:|---|---:|---:|---:|
| yes | 1 | 128/256 | 16.50 | 16.56 | 61 |
| yes | 16 | 1024/256 | 51.81 | 51.39 | 309 |
| yes | 64 | 1024/256 | 162.41 | 162.37 | 394 |
| no | 1 | 128/256 | 16.79 | 16.79 | 60 |
| no | 16 | 1024/256 | 51.85 | 51.83 | 309 |
| no | 64 | 1024/256 | 163.53 | 163.46 | 391 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_offline_async_b16 | 46.65 | 46.57 | 0.08 | 1320 | collect 44425, replay 1854, sample 191, schedule 45, postprocess 35 |
| nano_offline_sync_b16 | 48.61 | 46.57 | 2.03 | 1324 | collect 46394, replay 1847, sample 189, schedule 44, postprocess 33 |
| nano_w1c16 | 14.06 | 14.02 | 0.04 | 423 | forward 29694, collect 12828, replay 761, detokenize 19, prepare 972, sample 185, postprocess 55 |
| nano_w1c64 | 30.88 | 30.04 | 0.84 | 362 | forward 30004, collect 7318, detokenize 19, prepare 1005, replay 811, sample 191, postprocess 96 |
| nano_w3 | 16.29 | 16.26 | 0.03 | 362 | collect 14729, forward 29616, detokenize 20, replay 782, sample 190, postprocess 141, schedule 154 |

## Server settings from the logs

- vllm_server: GPU KV cache size: 163,168 tokens

