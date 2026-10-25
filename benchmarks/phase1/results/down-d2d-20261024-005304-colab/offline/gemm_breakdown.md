# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.7 | 14.8 | 52.3 | 30.4 | 114.3 | 1984 |
| vllm_awq_b1 | 16.4 | 14.6 | 55.6 | 30.9 | 117.6 | 1024 |
| nano_b16 | 16.4 | 15.2 | 55.0 | 31.7 | 118.4 | 1984 |
| vllm_awq_b16 | 17.8 | 16.3 | 58.8 | 33.3 | 126.1 | 1024 |
| nano_b64 | 38.2 | 32.2 | 102.0 | 71.7 | 244.1 | 1984 |
| vllm_awq_b64 | 34.9 | 30.4 | 89.5 | 66.0 | 220.9 | 1024 |
