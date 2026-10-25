# Phase 2 run m0-20261023-010333-colab

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 53 |
| install_sglang | 0 | 51 |
| install_nsys | 0 | 0 |
| download_model | 0 | 44 |
| tests | 1 | 108 |
| serve | 0 | 2091 |
| nsys | 0 | 252 |
| summarize | 0 | 0 |

## W1: random 1024/256

**output tok/s**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 78 | 152 | 297 | 572 | 935 | 1403 | 1906 |
| nano | 80 | 154 | 274 | 551 | 870 | 1252 | 1590 |
| nano_gil | - | - | - | - | - | 1098 | - |
| sglang | 78 | 150 | 297 | 570 | 934 | 1423 | 1286 |
| nano_w1c16 | - | - | - | - | 616 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 944 |

**TTFT p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 26.1 | 40.8 | 51.0 | 64.8 | 137.4 | 224.7 | 771.0 |
| nano | 40.8 | 77.7 | 100.1 | 150.9 | 275.4 | 563.3 | 1537.7 |
| nano_gil | - | - | - | - | - | 2248.1 | - |
| sglang | 51.8 | 95.0 | 77.0 | 82.1 | 148.0 | 182.6 | 2303.9 |
| nano_w1c16 | - | - | - | - | 1159.7 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 4507.4 |

**ITL p50 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 12.83 | 13.07 | 13.30 | 13.77 | 14.89 | 18.51 | 23.36 |
| nano | 24.72 | 12.75 | 13.18 | 14.17 | 15.74 | 20.42 | 28.27 |
| nano_gil | - | - | - | - | - | 20.09 | - |
| sglang | 12.61 | 12.92 | 13.22 | 13.76 | 14.74 | 18.13 | 34.15 |
| nano_w1c16 | - | - | - | - | 16.20 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 27.81 |

**ITL p99 (ms)**

| config | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm | 13.05 | 13.56 | 13.86 | 14.43 | 63.31 | 133.33 | 147.86 |
| nano | 24.93 | 25.59 | 26.67 | 28.43 | 31.85 | 41.43 | 57.37 |
| nano_gil | - | - | - | - | - | 40.89 | - |
| sglang | 12.91 | 13.36 | 13.75 | 14.32 | 15.50 | 19.29 | 37.39 |
| nano_w1c16 | - | - | - | - | 33.98 | - | - |
| nano_w1c64 | - | - | - | - | - | - | 60.67 |

## W2: shared prefixes

w2a: one 2048-token prefix + 128 tokens, output 128, c=16. w2b: 48 prefixes of 4096 tokens + 256, output 128, c=16. Hit rate: prompt tokens served from the prefix cache during the run.

| config | run | ok | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 | hit rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| nano | w2a | yes | 5.46 | 699 | 393.8 | 1896.1 | 30.30 | - |
| nano_nocache | w2a | yes | 3.14 | 402 | 2950.6 | 3759.2 | 33.93 | - |
| sglang | w2a | yes | 7.20 | 922 | 201.1 | 607.6 | 14.96 | - |
| vllm | w2a | yes | 7.38 | 944 | 222.3 | 364.6 | 14.96 | - |
| nano | w2b | yes | 2.63 | 337 | 3349.0 | 5032.9 | 39.57 | - |
| sglang | w2b | yes | 2.98 | 382 | 1701.4 | 3956.9 | 19.32 | - |
| vllm | w2b | yes | 3.21 | 411 | 579.5 | 3565.6 | 19.66 | - |

## W3: prefill interference

Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. ITL columns are the background's.

| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano | 20.59 | 45.0 | 396.3 | 902.2 | 1514 | 461.5 | 975.2 |
| nano_nochunk | 20.36 | 44.5 | 322.6 | 668.7 | 1547 | 399.6 | 695.5 |
| nano_w3 | 21.10 | 47.6 | 386.9 | 742.8 | 1473 | 383.6 | 759.8 |
| sglang | 17.05 | 33.5 | 319.0 | 589.7 | 1730 | 350.9 | 629.5 |
| vllm | 17.27 | 28.4 | 147.9 | 152.8 | 1741 | 361.6 | 692.0 |
| vllm_mbt512 | 17.22 | 44.9 | 50.7 | 52.4 | 1753 | 434.5 | 926.1 |
| vllm_nochunk | 17.14 | 20.3 | 273.4 | 434.1 | 1744 | 335.8 | 659.7 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_w1c16 | 16.11 | 14.09 | 2.01 | 424 | sample 14060, forward 22527, replay 671, detokenize 19, prepare 571, postprocess 47, schedule 30 |
| nano_w1c64 | 33.21 | 28.75 | 4.47 | 312 | sample 21940, forward 22499, replay 897, detokenize 18, prepare 581, postprocess 206, schedule 93 |
| nano_w3 | 20.20 | 16.29 | 3.91 | 360 | sample 16230, replay 2899, detokenize 20, forward 22269, postprocess 141, publish 65, schedule 53 |

## Server settings from the logs

- sglang_server: max_total_num_tokens=139721, chunked_prefill_size=4096
- vllm_mbt512_server: max_num_batched_tokens=512, GPU KV cache size: 159,264 tokens
- vllm_nochunk_server: GPU KV cache size: 158,304 tokens
- vllm_server: GPU KV cache size: 159,072 tokens

