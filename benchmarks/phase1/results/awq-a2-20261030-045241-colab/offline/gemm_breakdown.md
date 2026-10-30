# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.4 | 14.8 | 52.3 | 30.3 | 113.8 | 1984 |
| vllm_awq_b1 | 16.4 | 14.6 | 55.6 | 31.1 | 117.8 | 1024 |
| nano_b64 | 33.2 | 28.4 | 91.5 | 58.8 | 211.9 | 1984 |
| vllm_awq_b64 | 34.9 | 30.3 | 89.5 | 65.8 | 220.5 | 1024 |
| nano_b128 | 48.5 | 37.9 | 164.9 | 94.8 | 346.0 | 1984 |
| vllm_awq_b128 | 49.2 | 41.2 | 165.4 | 93.8 | 349.5 | 1024 |
