# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.7 | 14.8 | 52.3 | 30.4 | 114.2 | 1984 |
| vllm_awq_b1 | 16.4 | 14.7 | 55.6 | 31.0 | 117.7 | 1024 |
| nano_b16 | 16.4 | 15.2 | 55.0 | 31.7 | 118.3 | 1984 |
| vllm_awq_b16 | 17.8 | 16.3 | 58.8 | 33.3 | 126.2 | 1024 |
| nano_b64 | 41.2 | 34.8 | 100.3 | 81.4 | 257.7 | 1984 |
| vllm_awq_b64 | 34.8 | 30.3 | 89.5 | 65.9 | 220.4 | 1024 |
