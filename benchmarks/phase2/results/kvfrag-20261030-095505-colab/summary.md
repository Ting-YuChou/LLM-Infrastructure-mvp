# Phase 2 run kvfrag-20261030-095505-colab

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 178 |
| install_nsys | 0 | 0 |
| download_model | 0 | 44 |
| cuda129 | 0 | 23 |
| tests | 0 | 520 |
| serve | 0 | 148 |

## W1: random 1024/256

**output tok/s**

| config | c=1 |
|---|---:|
| nano_awq | 205 |
| nano_w4 | 204 |

**TTFT p50 (ms)**

| config | c=1 |
|---|---:|
| nano_awq | 14.4 |
| nano_w4 | 14.4 |

**ITL p50 (ms)**

| config | c=1 |
|---|---:|
| nano_awq | 4.84 |
| nano_w4 | 4.84 |

**ITL p99 (ms)**

| config | c=1 |
|---|---:|
| nano_awq | 5.17 |
| nano_w4 | 5.48 |

