# Teacher-forced quality (288627 tokens, chunks of 2048)

| config | attention | linear | weights | weight GiB | perplexity | top-1 agreement |
|---|---|---|---|---:|---:|---:|
| bf16_torch | torch | torch | none | 15.0 | 7.233 | - |
| bf16_triton | triton | triton | none | 15.0 | 7.232 | 98.81% vs bf16_torch |
| w4a16_triton | triton | triton | w4a16 | 5.3 | 8.024 | 85.90% vs bf16_torch |
