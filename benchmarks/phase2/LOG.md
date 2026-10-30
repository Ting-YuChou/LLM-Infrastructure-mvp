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

### Run m1tests-20261023-015432-colab (installs, GPU tests, prefix_bench)

- `cuda129` installed CUDA 12.8 nvcc (39 s). 417 tests passed, none skipped:
  the GPU suite (decode graphs with the fixed capture, async scheduling
  against sync in graph and eager mode, a mixed batch against the
  reference, KV sizing) plus the nano unit tests, including the
  multiprocess and C++ radix-tree ones.
- prefix_bench on the VM's CPU (slower than the Mac): a 2k-token
  shared-prefix admission takes 18.5 ms with Phase 1's hash chain, 168 us
  with the Python radix tree and 16 us in C++; with eviction (W2b) 3.1 ms,
  314 us and 33 us at p50, 36.8 ms, 387 us and 50 us at p99. Same hit rate
  everywhere (93.6% / 47.2%).

### Run m1to6-20261023-015904-colab (serve, nsys, offline)

Every server and workload passed the `ok` check (all requests completed with
the full output length). nano got 9,942 KV blocks (M0's vLLM capacity).

- `colab exec` lost its connection after 2,950 s ("Connection was lost.")
  during the offline step; the run finished on the VM and
  `colab_phase0.sh fetch` downloaded it.
- summarize on the VM printed no serve tables: it looked for console.log in
  the run directory, but the launcher writes the console to
  `/content/p0/console.log` and copies it into the report tarball only
  afterwards. summarize.py now also reads `<root>/console.log`; summary.md
  was regenerated locally.
- The offline step ran nano_bench with its default torch attention and
  linear paths (16.5 / 51.8 / 162 ms per decode step at b=1/16/64, against
  12.15 / 14.63 / 23.51 ms with Triton in Phase 1), so its numbers and both
  offline nsys captures are void. The step now passes the server's backends
  (`--attention-backend triton --linear-backend triton`).
- **Mid-serving Triton compiles.** nano_m2's W1 (the first mixed-scheduling
  server on this VM) had TPOT p90 27.3 / 35.7 ms at c=16 / 32 against p50
  16.3 / 22.3, while nano_m2_b2048 a few minutes later had 19.4 / 27.6: a
  quarter of the requests (those in flight at one moment) lost ~3 s each,
  once per concurrency level. Triton compiles a kernel variant per
  specialization of its arguments: whether each pointer is 16-byte aligned,
  whether each integer is 1 or divisible by 16, and every constexpr value.
  Mixed batches changed all three with the batch shape: build_batch packed
  the int32 fields back to back (and the prefill rows' context lengths and
  block tables were slices at row num_decode_rows), the block-table stride
  was the batch's widest table, and the eager decode kernel's NUM_SPLITS
  comes from the decode row count (15 rows: a split count no decode graph
  was captured with). Each new combination compiled mid-step; later servers
  loaded the variants from Triton's on-disk cache, which is why only the
  first W1 shows it. W3 on the same server came after W1 and was clean.
  Fixes: every build_batch field starts on 16 bytes and the prefill rows
  get fields of their own (5c315c4); the block-table width is rounded up to
  a multiple of 16; capture_graphs first runs one eager batch per decode
  row count (`ModelRunner.warm_up_eager`, about 66 forward passes).
  tests/gpu/test_nano_vllm_warmup.py installs Triton's JIT cache-miss hook
  after the warm-up and fails on any kernel compiled during a mixed
  workload with every decode row count.
- Async scheduling in the frontend's process (nano_m4_inproc) gets the same
  W1 throughput as with the core in its own process (1,958 vs 1,973 tok/s at
  c=64): the GPU always has the next step queued, so the host's time,
  waiting for the GIL included, is hidden as long as it is shorter than a
  step. The separate process still halves TTFT at c=64 (330 vs 666 ms p50).
- nsys of the server with the M4 settings: GPU idle 0.04 ms of a 14.1 ms
  step at W1 c=16, 0.03 of 16.3 ms in W3, 0.84 of 30.9 ms at c=64 (M0: 2.0,
  3.9 and 4.5 ms). The c=64 median step is an eager mixed step (362 kernels,
  364 launches); its 0.84 ms idle is 2.7% of the step, the most M5 (CUDA
  graphs for mixed steps) could recover, so M5 is not done.
- vLLM 0.19.1 runs with async scheduling by default ("Asynchronous
  scheduling is enabled") and, on this 40 GB A100, max_num_batched_tokens
  2048 ("compile range (1, 2048)"). nano_m2_b2048's W3 ITL max (156 ms)
  matches vLLM's default (153 ms) as it should.
- `NUM_KVCACHE_BLOCKS=0` sized the cache to 9,958 blocks (159k tokens, vLLM
  has 159-163k) with 35,980 MiB in use at gpu_memory_utilization 0.9.

## Final round

### Run final-20261023-030032-colab (tests, serve, nsys, offline)

- 436 GPU and unit tests passed, none skipped, including the new
  `test_no_kernel_compiles_after_warm_up` (torch and Triton linears, sync
  and async): no kernel compiled after `capture_graphs` in a mixed workload
  covering 9+ decode row counts.
- `nano_m2_coldjit` repeated the first M2 server's W1 (one process,
  synchronous, 512-token chunks) with an empty `TRITON_CACHE_DIR`: c=16 /
  32 / 64 gave 852 / 1,234 / 1,618 tok/s with TPOT p90 19.5 / 27.2 ms,
  against 719 / 1,073 / 1,551 and 27.3 / 35.7 ms before the fix, and in line
  with the warm-cache 2048-chunk run (857 / 1,236 / 1,611). The stalls are
  gone. Startup takes 30 s instead of 20 s with a cold cache.
- vLLM's W1 came out 0-4% above its M0 run (1,976 vs 1,906 tok/s at c=64),
  a measure of run-to-run spread; the final tables compare servers from
  this run only. vLLM's KV capacity was 163,168 tokens this time; nano kept
  9,942 blocks (159k tokens), slightly less.
- SGLang collapsed at W1 c=64 again (1,284 tok/s, TTFT p50 2.2 s), as in M0:
  with 140k KV tokens and 4096-token prefill chunks it queues admissions.
  Left as measured.
- `colab exec` timed out ("Timeout waiting for reply") after the report was
  written; the driver's fetch still downloaded it.
- The offline step now ran the Triton kernels (b=1/16/64 sync: 12.27 /
  14.55 / 23.55 ms per decode step, Phase 1: 12.15 / 14.63 / 23.51), but
  with async scheduling the mean was above the median (15.03 vs 14.10 ms at
  b=16; b=64 23.38 vs 22.42): nano_bench gave each step the wall time of the
  core.step() call that launched it, and with async scheduling that call
  waits for the previous step, so the first decode step absorbed the last
  prefill chunk (~240 ms over 255 steps). nano_bench now times a step by
  the call that collects its tokens; rerun below.

### Run final-offline-20261023-034858-colab (offline only)

- With steps timed by the call that collects them, async scheduling's mean
  and median agree: 12.04 / 14.13 / 22.57 ms per decode step at b=1/16/64
  (median 12.04 / 14.11 / 22.43), synchronous 12.27 / 14.54 / 23.54.
- The kernel kept the earlier run's `--env` settings: `colab exec --env`
  sets variables in the Jupyter kernel, which lives as long as the VM, so
  the final round inherited `P2_NANO_KV_BLOCKS=9942` from the M1-M6 run
  (nano had 9,942 blocks, not vLLM's 10,198-block capacity of that run).
  The driver now also passes the names it set (`P0_ENV_NAMES`) and the
  launcher drops every other P0_/P1_/P2_ variable.
- Host checks: 3 gateway integration tests fail on this machine because a
  local Redis is running and keeps `/usage` counters across test runs
  (`total_requests` 41 where the test expects 1). Unrelated to Phase 2; the
  branch does not touch the gateway.

## TTFT follow-up: piecewise CUDA graphs

- W1's c=1 TTFT (nano 33 ms, vLLM 26) measured cached prompts: vllm bench's
  random prompt i is the same in every seed-0 run, and 50% of W1's prompt
  tokens hit the prefix cache in both servers' counters. nsys's NVTX
  `forward` phase put an eager step at ~29 ms of host time regardless of
  size, against ~12 ms of GPU work for a 16-token prefill. Fix: piecewise
  CUDA graphs (e4c8fe6), as vLLM's FULL_AND_PIECEWISE mode does.

### Run ttft-20261023-041614-colab (new VM: installs, tests, serve, nsys)

- GPU tests: 237 failed, 204 passed. The first failure was
  `test_piecewise_replay_matches_eager[triton-w4a16]`: "operation failed due
  to a previous error during capture" (cudaErrorStreamCaptureInvalidated);
  every later failure was "Offset increment outside graph capture
  encountered unexpectedly": the failed capture left the CUDA RNG in
  capture state for the rest of the process. On the kept VM: every W4A16
  kernel (mma, wide, dequantize + F.linear, 8-96 rows) captures fine on its
  own; the test passes alone and with the other graph tests; it fails only
  after test_nano_vllm_async.py; with `gc.collect()` before each test the
  same order passes. Cause: the async tests' engines (now with dozens of
  piecewise graphs each) became garbage in reference cycles, and the cyclic
  collector ran during the next capture, destroying those graphs and freeing
  their memory pool with cudaFree, which invalidates a capture. Fix: both
  captures run `gc.collect()` first and keep the collector disabled while
  capturing (vLLM freezes the GC there too). A server captures once at
  startup, so serving was not affected.
- Serve: all `ok`. W1 c=1 TTFT 22.4 ms with piecewise graphs, 33.0 without
  (same run), vLLM 26.6. New workload w1u (seeds 101 / 116, prompts no other
  run sends): uncached c=1 TTFT 76.1 / 76.5 / 81.8 ms (pieces / eager /
  vLLM). vLLM's own c=1 TTFT goes from 26.6 to 81.8 ms without the cache,
  which confirms what W1 measured.
- nsys of w1c1 showed no `pieces` phase: with no warmup before it, all ten
  prompts were uncached 1024-token prefills, past the largest piecewise size
  (512), so eager. Added w1c1c (the same prompts twice).

### Run ttft2-20261023-050451-colab (kept VM: tests, nsys)

- 441 GPU and unit tests passed, none skipped, including the piecewise W4A16
  test after the async tests.
- nsys of w1c1c: host time per step 22.4 ms for eager steps (the first
  pass's 1024-token prefills) and 4.0 ms for piecewise steps (the second
  pass's cached prefills), decode steps 12.1 ms with 0.04 ms idle.

## W4A16 online (2026-10-29)

### Run w4serve-20261029-211009-colab (new VM `w4serve`, stopped after the run)

- Question: does the W4A16 path past 256 rows put nano behind vLLM AWQ online? That path dequantizes each weight and calls cuBLAS. TTFT and the mixed steps both go through it.
- Harness: `run_phase2.sh` gained engine `vllmw4`. It is vLLM on `P2_VLLM_W4_MODEL` (default the AWQ checkpoint Phase 1 uses), downloaded on first use and served under the BF16 model's name. SGLang is now installed only when `P2_RUNS` has an sglang entry.
- Settings: `P2_STEPS="installs serve nsys summarize" P2_CONCURRENCY="1 16 64" P2_NSYS="w1c16 w1c64"`, with these `P2_RUNS`:
  - `vllmw4:vllm_awq:w1,w1s,w1u,w3`;
  - `nano:nano_w4:w1,w1s,w1u,w3`, with `WEIGHT_QUANT=w4a16,ENGINE_MULTIPROCESS=true,MAX_NUM_BATCHED_TOKENS=2048,NUM_KVCACHE_BLOCKS=0`;
  - `nano:nano_w4_c256:w1,w1u`, the same with 256-token chunks.
  - `P2_NSYS_ENV` used the same W4A16 settings in one process.
  - No GPU tests: the code under test is main at ac24d86, which `sampler-s2` tested.
- 26 minutes in all. vLLM's venv took 276 s, the server runs 1,104 s, nsys 101 s.
- Serve: every run was `ok`. nano is ahead on every W1, W1u, W1s and W3 metric except W1's ITL p99 at c=16 (86.4 against 76.1 ms). Examples:
  - W1 c=64: 2,101 against 1,949 tok/s; TTFT p50 283 against 569 ms;
  - W1u c=1 TTFT p50: 89.1 against 95.1 ms.
- nsys:
  - each step past 256 rows spends 14.0 ms in 128 dequantize launches and 63 ms in cuBLAS GEMMs;
  - w1c16 had 32 such steps and w1c64 had 64.
  - The dequantization moves 17.6 GB, about 81% of peak bandwidth.
- 256-token chunks:
  - no dequantization, and ITL p99 drops to 25–31 ms;
  - but at c=64, TTFT p50 is 1,877 ms and ITL p50 27.3 ms;
  - so 2,048 stays the default.
- vLLM's sampled c=16 throughput again collapsed: 796 against 1,502 tok/s greedy.
- The "engine core died" line at the end of each nano log comes from `stop_server` killing the process group. It is in `sampler-serve`'s and `kernels-serve`'s logs too.

## AWQ online (2026-10-30)

### Run awqserve-20261030-060945-colab (VM `awq`, kept from Phase 1's runs `awq-a2`/`awq-a3`, stopped after this run)

- Question: with the same AWQ checkpoint on both sides, does nano stay ahead of vLLM online, and do 512-token chunks (all steps within `gemm_w4a16_tiled` now that it takes 512 rows) fix W1's ITL p99 at c=16?
- Code: 4edfb73 (AWQ loading, zero points in the tiled kernel with the prefetch fix, `lm_head_quant` `auto` = `int8_dense`, `tiled_max_rows` 512).
- Settings: `P2_STEPS="tests serve nsys summarize" P2_CONCURRENCY="1 16 64" P2_NSYS="w1c16 w1c64"`, with these `P2_RUNS`:
  - `vllmw4:vllm_awq:w1,w1s,w1u,w3`;
  - `nano:nano_awq:w1,w1s,w1u,w3`, with `WEIGHT_QUANT=awq,ENGINE_MULTIPROCESS=true,MAX_NUM_BATCHED_TOKENS=2048,NUM_KVCACHE_BLOCKS=0`;
  - `nano:nano_awq_c512:w1,w1u,w3`, the same with 512-token chunks;
  - `P2_NSYS_ENV="WEIGHT_QUANT=awq,MAX_NUM_BATCHED_TOKENS=2048,NUM_KVCACHE_BLOCKS=0"`.
- 22 minutes: tests 300 s (962 passed, no skips), servers 911 s, nsys 99 s.
- Serve: every run was `ok` with 0 failed requests.
  - nano is ahead on every W1, W1u and W1s metric but W1's ITL p99 at c=16 (86.65 against 75.20 ms), as in `w4serve`. Examples: W1 c=1 204 against 172 tok/s, ITL p50 4.85 against 5.75 ms; W1s c=16 1,415 against 794 tok/s.
  - W3: nano's background ITL p50 8.08 against 9.23 ms, p99 12.2 against 11.6, p99.9 130.5 against 161.9; long-prompt TTFT p50 351.9 against 398.6 ms.
- 512-token chunks: W1 ITL p99 44.85 / 49.96 ms at c=16 / 64 (2,048: 86.65 / 93.84), W1 TTFT p50 at c=64 206.6 against 286.8 ms, W1u c=16 TTFT 200.4 against 350.4; but W1 c=64 throughput 2,052 against 2,107 tok/s, W3 ITL p99 29.3 ms and long-prompt TTFT p50 433.2 ms. The benchmark default stays 2,048; nano's own default is 512.
- W1u c=1 TTFT p50 89.1 ms, the same as `w4serve`: a 1,024-row step still dequantizes. Phase 1's G4 asked for −10 ms; missed.
- KV: 215,424 tokens against `w4serve`'s 227,824. Allocated before sizing 5.87 against 5.33 GiB (the int8 lm_head is 0.49 GiB), but 8.32 against 6.80 GiB in use on the GPU: about 1 GiB reserved and not allocated, likely fragmentation from unpacking AWQ on the GPU. (Run `kvfrag` below: it was the int8 lm_head's quantization.)

## KV after loading (2026-10-30)

### Run kvfrag-20261030-095505-colab (new VM `kvfrag`, stopped after this run)

- Question: does building each weight on a side stream give the KV cache back the memory left reserved after loading?
- Cause, found before the run with a CPU trace of the real `_load_weight` (8B shapes, meta tensors) replayed through a model of the CUDA caching allocator:
  - Converting weights at load makes large GPU temporaries: the int8 lm_head quantizes 128,256 x 4,096 in fp32 (2 GiB per temporary), and RTN and AWQ unpacking reach ~470 MiB per projection.
  - Best fit puts the weights loaded after them in their freed blocks, and `empty_cache` releases only segments with no live block.
  - Replay of `awqserve`'s load: 5.86 GiB allocated (5.87 measured) and 2.00 GiB reserved but unallocated. The projections leave 0.76 of that (the same for RTN and AWQ) and the int8 lm_head 1.24.
  - Measured in use minus allocated: 0.55 GiB for bf16, 1.47 for `w4serve`, 2.45 for `awqserve`, i.e. 0.55 plus the replay's figures.
  - `empty_cache` after each weight: 1.44 GiB in the replay.
- Code: ba685ff. `_build_on_stream` builds each weight (and a tied int8 head) on a side stream and clones it onto the current one; the KV sizing line prints PyTorch's reserved and allocated memory. Replay: 0.00 GiB reserved but unallocated.
- Settings: `P2_STEPS="installs tests serve summarize" P2_CONCURRENCY="1"`, with these `P2_RUNS`:
  - `nano:nano_awq:w1`, with `WEIGHT_QUANT=awq,ENGINE_MULTIPROCESS=true,MAX_NUM_BATCHED_TOKENS=2048,NUM_KVCACHE_BLOCKS=0`;
  - `nano:nano_w4:w1`, the same with `WEIGHT_QUANT=w4a16` (RTN, now with the `int8_dense` head).
- 15 minutes, 1.25 CU: installs 245 s, tests 520 s (963 passed, no skips), servers 148 s.
- KV, both servers: 230,736 tokens (`awqserve` 215,424, `w4serve` 227,824, vLLM 237,824).
  - 6.45 GiB in use on the GPU after sizing (`awqserve` 8.32);
  - PyTorch reserved 5.92 and allocated 5.88 GiB (5.87 before sizing).
- W1 c=1:
  - nano_awq: 205 tok/s, TTFT p50 14.4 ms, ITL p50 4.84 / p99 5.17 ms;
  - nano_w4: 204 tok/s, TTFT p50 14.4 ms, ITL p50 4.84 / p99 5.48 ms.
- Startup: 25 s for nano_awq (the VM's first server) and 20 s for nano_w4; `awqserve`'s nano_awq took 20 s.
