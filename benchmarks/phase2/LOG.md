# Phase 2 log

Chronological record of each Phase 2 run on the Colab A100 (40GB): what ran,
what failed and the fix. Results tables are in README.md; raw runs in
results/<run_id>/.

## M0: online harness (2026-10-22)

- nano's server could not be measured by `vllm bench serve` as it was:
  `ignore_eos` was dropped by the request model, there was no usage chunk (so
  `total_output_tokens` came from re-tokenized text), no `/v1/models`, and the
  admission semaphore allowed 16 requests. Added `ignore_eos`,
  `stream_options.include_usage`, `/v1/models`, and `/start_profile` /
  `/stop_profile` (cudaProfilerStart/Stop, so nsys captures only a workload).
- `vllm bench serve` 0.19.1 no longer sends temperature 0 by default (each
  server applies its own default; vLLM's comes from the model's
  generation_config). Phase 2 sends `--temperature 0` to every server
  (`BENCH_TEMPERATURE` in lib.sh); Phase 0's sweeps ran with the servers'
  defaults.
- Baseline settings, changed only through the environment: nano's KV cache
  holds as many tokens as vLLM's (its logged "GPU KV cache size" / 16 blocks),
  64 sequences, 256 admitted requests, max_model_len 8192, 4096 max new tokens,
  bf16. Engine behaviour is unchanged: prefill-first scheduling with 512-token
  chunks and the hash-chained prefix cache.
