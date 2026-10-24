# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b64 | 35.6 | 28.7 | 104.1 | 63.3 | 231.8 | 1984 |
| vllm_awq_b64 | 34.8 | 30.3 | 89.4 | 66.1 | 220.6 | 1024 |
| nano_b128 | 54.8 | 41.5 | 194.1 | 109.4 | 399.7 | 1984 |
| vllm_awq_b128 | 49.4 | 41.6 | 166.0 | 94.3 | 351.2 | 1024 |
| nano_b256 | 92.4 | 68.2 | 369.9 | 192.2 | 722.7 | 1984 |
| vllm_awq_b256 | 80.0 | 58.3 | 322.5 | 164.1 | 624.9 | 1024 |
