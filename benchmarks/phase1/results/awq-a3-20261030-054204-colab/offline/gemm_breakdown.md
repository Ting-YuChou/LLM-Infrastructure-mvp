# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.5 | 14.8 | 52.3 | 30.3 | 113.9 | 1984 |
| vllm_awq_b1 | 16.4 | 14.6 | 55.7 | 31.0 | 117.7 | 1024 |
| nano_b16 | 16.2 | 15.1 | 55.9 | 32.3 | 119.5 | 1984 |
| vllm_awq_b16 | 17.8 | 16.2 | 58.8 | 33.3 | 126.1 | 1024 |
| nano_b64 | 33.0 | 28.5 | 87.5 | 56.3 | 205.3 | 1984 |
| vllm_awq_b64 | 34.9 | 30.2 | 89.4 | 65.7 | 220.2 | 1024 |
| nano_b128 | 47.0 | 37.3 | 159.3 | 88.9 | 332.4 | 1984 |
| vllm_awq_b128 | 49.2 | 41.2 | 165.3 | 93.8 | 349.5 | 1024 |
