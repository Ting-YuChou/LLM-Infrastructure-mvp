# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.5 | 14.8 | 52.4 | 30.4 | 114.0 | 1984 |
| vllm_awq_b1 | 16.5 | 14.7 | 55.6 | 31.0 | 117.7 | 1024 |
| nano_b16 | 16.8 | 15.4 | 56.1 | 32.4 | 120.8 | 1984 |
| vllm_awq_b16 | 18.0 | 16.4 | 58.8 | 33.5 | 126.6 | 1024 |
| nano_b64 | 33.4 | 28.7 | 92.6 | 60.1 | 214.8 | 1984 |
| vllm_awq_b64 | 34.9 | 30.5 | 89.8 | 66.2 | 221.3 | 1024 |
