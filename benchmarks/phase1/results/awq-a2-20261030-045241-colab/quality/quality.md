# Teacher-forced quality (288627 tokens, chunks of 2048)

| config | attention | linear | weights | KV cache | weight GiB | perplexity | top-1 agreement |
|---|---|---|---|---|---:|---:|---:|
| awq_triton | triton | triton | awq | auto | 5.4 | 7.563 | - |
| awq_int8head | triton | triton | awq+int8 lm_head | auto | 4.9 | 7.563 | 99.01% vs awq_triton |
| vllm_bf16 | vllm | vllm | none | auto | - | 7.232 | 90.09% vs awq_triton |
| vllm_awq | vllm | vllm | awq_marlin | auto | - | 7.560 | 98.39% vs awq_triton |
