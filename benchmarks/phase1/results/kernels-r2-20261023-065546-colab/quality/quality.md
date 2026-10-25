# Teacher-forced quality (288627 tokens, chunks of 2048)

| config | attention | linear | weights | KV cache | weight GiB | perplexity | top-1 agreement |
|---|---|---|---|---|---:|---:|---:|
| bf16 | triton | triton | none | auto | 15.0 | 7.232 | - |
| bf16_fp8kv | triton | triton | none | fp8 | 15.0 | 7.248 | 97.55% vs bf16 |
| w4a16 | triton | triton | w4a16 | auto | 5.3 | 8.024 | 85.88% vs bf16 |
| w4a16_fp8kv | triton | triton | w4a16 | fp8 | 5.3 | 8.040 | 85.79% vs bf16 |
