# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.1 | 14.8 | 51.4 | 30.0 | 112.3 | 1984 |
| vllm_awq_b1 | 16.4 | 14.6 | 55.5 | 31.0 | 117.4 | 1024 |
| nano_b16 | 16.2 | 15.3 | 54.9 | 31.9 | 118.2 | 1984 |
| vllm_awq_b16 | 17.8 | 16.2 | 58.7 | 33.2 | 125.9 | 1024 |
| nano_b64 | 31.8 | 27.4 | 85.1 | 53.7 | 197.9 | 1984 |
| vllm_awq_b64 | 34.9 | 30.3 | 89.5 | 65.9 | 220.6 | 1024 |
