# GEMM time per projection inside decode (nsys, median µs per call)

| run | qkv | o_proj | gate_up | down | per layer | calls per projection |
|---|---:|---:|---:|---:|---:|---:|
| nano_b1 | 16.5 | 15.0 | 52.2 | 30.6 | 114.3 | 1984 |
| vllm_awq_b1 | 16.4 | 14.7 | 55.6 | 31.0 | 117.7 | 1024 |
| nano_b16 | 16.5 | 15.7 | 55.2 | 32.1 | 119.6 | 1984 |
| vllm_awq_b16 | 17.8 | 16.3 | 58.8 | 33.3 | 126.2 | 1024 |
| nano_b64 | 38.2 | 32.2 | 102.1 | 71.7 | 244.3 | 1984 |
| vllm_awq_b64 | 34.8 | 30.3 | 89.5 | 65.9 | 220.5 | 1024 |
