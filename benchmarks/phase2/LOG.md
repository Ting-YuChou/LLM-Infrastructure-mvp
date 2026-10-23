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

### Run m0-20261023-010333-colab (M0 baseline: vLLM, SGLang, nano as of Phase 1)

- 8 GPU tests failed, all CUDA-extension builds (the CUDA GEMV and W4A16
  kernels): phase 2's installs had no `cuda129` step (CUDA 12 nvcc for
  cpp_extension) and no ninja. Moved `cuda129` into lib.sh and added both.
  The serving numbers do not use those kernels (linear_backend triton).
- `serve` and `nsys` passed. summarize.py looked for the P2_SERVE lines in
  logs/serve.log, but bench_serve prints them to the console: fixed, the
  summary was regenerated from console.log.
- The driver hit a syntax error after downloading the results: I edited
  colab_phase0.sh while it was running, and bash reads scripts as it goes.
  The VM was kept (COLAB_KEEP=1), so nothing was lost. Don't edit a script a
  running driver is executing.
- nano's ITL p50 at c=1 read 24.7 ms with a TPOT of 12.38 ms: nano's server
  sent a chunk only when it had text, so a token whose text was held back
  (an incomplete UTF-8 character) merged into the next chunk and doubled
  that interval. vLLM sends a chunk per token (empty text if held back).
  From M1 on nano does the same; for M0, compare decode speed by TPOT.
- The workload-snapshot code (prefix-cache hit rates) landed after this run
  started, so M0 has no hit rates.
- The GIL diagnostic (NANO_SWITCH_INTERVAL=0.0005, W1 c=32) did not help:
  1098 vs 1252 tok/s, ITL p50 20.1 vs 20.4 ms. A shorter switch interval
  costs more than it recovers, so it does not isolate GIL waits; nsys does
  (below).
- nsys of the nano server: GPU idle 2.0 ms of a 16.1 ms step at c=16 and
  4.5 of 33.2 ms at c=64 (offline: 0.66-0.87 ms). The graph-replay host
  phase (filling the staging buffers) takes 0.67 ms at c=16 and 2.9 ms in
  W3, where 32 streams' detokenization and SSE run on the event loop thread
  at the same time: the model thread waits for the GIL.

## M1-M6 validation

### Run m1to6-20261023-015102-colab (stopped after the tests)

- The GPU tests crashed pytest: `CUDA error: device-side assert triggered`
  after "FF.FF" (the async tests, then the decode-graph tests). Cause: the
  M4 rewrite computed the decode graphs' input ids (the gather of in-flight
  tokens from `sampled`) in `_static_batch`, *before* `torch.cuda.graph`,
  so each graph recorded a read of that one temporary tensor: replays fed
  the capture-time tokens, and once the allocator reused the memory, garbage
  ids reached the embedding lookup. CPU tests never capture graphs, so they
  could not see it. Fix: `_graph_step` builds the inputs and runs the
  forward pass, and the warm-up and the capture both call it, so the gather
  is part of the graph.
- The run had no `installs` step, so the new `cuda129` install never ran;
  the rerun includes it (every install is skipped when already done).
- Stopped the remote run by hand (pkill through `colab console`) before the
  serving numbers, which would have decoded from wrong tokens.
