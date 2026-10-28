# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.1 | 14.8 | 51.5 | 30.0 | 112.4 | 1984 |
| vllm_awq_b1 | 16.4 | 14.6 | 55.6 | 31.1 | 117.6 | 1024 |
| nano_b16 | 16.1 | 15.1 | 54.8 | 31.8 | 117.8 | 1984 |
| vllm_awq_b16 | 17.8 | 16.2 | 58.8 | 33.2 | 126.0 | 1024 |
| nano_b64 | 31.8 | 27.4 | 85.1 | 53.7 | 197.9 | 1984 |
| vllm_awq_b64 | 34.8 | 30.4 | 89.5 | 65.8 | 220.4 | 1024 |
