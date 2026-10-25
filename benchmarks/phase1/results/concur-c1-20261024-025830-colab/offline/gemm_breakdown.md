# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b64 | 34.7 | 28.3 | 102.1 | 62.0 | 227.1 | 1984 |
| vllm_awq_b64 | 34.8 | 30.5 | 89.5 | 66.5 | 221.4 | 1024 |
| nano_b128 | 69.1 | 56.7 | 203.7 | 124.7 | 454.2 | 1984 |
| vllm_awq_b128 | 49.8 | 41.6 | 167.1 | 94.7 | 353.1 | 1024 |
| nano_b256 | 110.7 | 78.5 | 556.8 | 262.1 | 1008.0 | 1984 |
| vllm_awq_b256 | 80.7 | 59.0 | 326.6 | 166.1 | 632.4 | 1024 |
