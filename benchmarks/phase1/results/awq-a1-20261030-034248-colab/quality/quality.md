# Teacher-forced quality (288627 tokens, chunks of 2048)

| config | attention | linear | weights | KV cache | weight GiB | perplexity | top-1 agreement |
|---|---|---|---|---|---:|---:|---:|
| bf16_triton | triton | triton | none | auto | 15.0 | 7.232 | - |
| w4a16_triton | triton | triton | w4a16 | auto | 5.4 | 8.023 | 85.91% vs bf16_triton |
| awq_triton | triton | triton | awq | auto | 5.4 | 7.562 | 90.10% vs bf16_triton |
| awq_dequant | triton | triton | awq (dequant prefill) | auto | 5.4 | 7.563 | 90.11% vs bf16_triton |
| awq_int8head | triton | triton | awq+int8 lm_head | auto | 4.9 | 7.563 | 90.06% vs bf16_triton |
| vllm_bf16 | vllm | vllm | none | auto | - | 7.232 | 98.81% vs bf16_triton |
| vllm_awq | vllm | vllm | awq_marlin | auto | - | 7.560 | 90.13% vs bf16_triton |
