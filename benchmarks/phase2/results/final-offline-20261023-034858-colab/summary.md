# Phase 2 run final-offline-20261023-034858-colab

| step | rc | seconds |
|---|---:|---:|
| offline | 0 | 115 |

## Offline decode (nano_bench, prefill-first, CUDA graphs)

| async | batch | in/out | decode ms/step | p50 | tok/s |
|---|---:|---|---:|---:|---:|
| yes | 1 | 128/256 | 12.04 | 12.04 | 83 |
| yes | 16 | 1024/256 | 14.13 | 14.11 | 1133 |
| yes | 64 | 1024/256 | 22.57 | 22.43 | 2836 |
| no | 1 | 128/256 | 12.27 | 12.27 | 81 |
| no | 16 | 1024/256 | 14.54 | 14.52 | 1101 |
| no | 64 | 1024/256 | 23.54 | 23.56 | 2719 |

## nsys: nano server per engine step

| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |
|---|---:|---:|---:|---:|---|
| nano_offline_async_b16 | 14.05 | 14.02 | 0.04 | 426 | collect 12909, replay 803, sample 183, schedule 44, postprocess 32 |
| nano_offline_sync_b16 | 14.94 | 14.03 | 0.91 | 428 | collect 13818, replay 779, sample 181, schedule 44, postprocess 31 |

