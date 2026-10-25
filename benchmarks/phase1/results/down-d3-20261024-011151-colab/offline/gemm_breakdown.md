# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.6 | 14.8 | 52.3 | 30.4 | 114.2 | 1984 |
| vllm_awq_b1 | 16.4 | 14.6 | 55.6 | 31.0 | 117.6 | 1024 |
| nano_b16 | 16.4 | 15.2 | 55.0 | 31.7 | 118.3 | 1984 |
| vllm_awq_b16 | 17.8 | 16.3 | 58.7 | 33.3 | 126.0 | 1024 |
| nano_b64 | 34.6 | 28.3 | 102.0 | 62.0 | 227.0 | 1984 |
| vllm_awq_b64 | 34.8 | 30.4 | 89.6 | 66.0 | 220.8 | 1024 |
