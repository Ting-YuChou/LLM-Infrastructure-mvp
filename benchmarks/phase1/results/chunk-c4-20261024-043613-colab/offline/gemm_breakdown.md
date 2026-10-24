# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b64 | 34.0 | 29.2 | 84.5 | 54.7 | 202.4 | 1984 |
| vllm_awq_b64 | 34.8 | 30.3 | 89.4 | 65.9 | 220.5 | 1024 |
| nano_b128 | 47.9 | 38.0 | 153.4 | 88.5 | 327.8 | 1984 |
| vllm_awq_b128 | 49.4 | 41.7 | 166.5 | 94.5 | 352.2 | 1024 |
| nano_b256 | 75.4 | 58.2 | 288.1 | 151.7 | 573.4 | 1984 |
| vllm_awq_b256 | 80.7 | 58.8 | 325.8 | 165.5 | 630.8 | 1024 |
