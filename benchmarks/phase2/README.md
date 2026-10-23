# Phase 2: the engine's systems side

Roadmap Phase 2 (memory management, CPU overhead, multithreading) for
`src/serving/nano_vllm/`, measured online against vLLM 0.19.1 and SGLang on one
Colab A100-SXM4-40GB with Llama 3.1 8B Instruct (BF16).

| milestone | change |
|---|---|
| M0 | nano's OpenAI server measurable by `vllm bench serve`; NVTX phase ranges; baseline |
| M1 | radix-tree prefix cache, KV cache sized from free memory, cache metrics |
| M2 | mixed prefill/decode batches (chunked prefill) |
| M3 | engine core in its own process; tokenization and detokenization in the frontend |
| M4 | async scheduling: step N+1 launched before step N's tokens reach the host |
| M5 | CUDA graphs for mixed steps (only if nsys shows them CPU-bound) |
| M6 | the radix tree in C++ (nanobind) |

## How to run

```bash
BENCH_PHASE=phase2 COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh          # new VM
BENCH_PHASE=phase2 COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh rerun    # same VM
```

`run_phase2.sh` documents the workloads (W1 throughput/latency sweep, W2
shared prefixes, W3 prefill interference), the steps and the `P2_*`
variables. Each run writes `summary.md` in its results directory.

## Results

### M0: baseline (run `m0-20261023-010333-colab`)

Every server got the same KV capacity (vLLM's: 159k tokens; nano 9,942
blocks of 16), max_model_len 8192, greedy sampling and `--ignore-eos`. nano
was Phase 1's engine: prefill-first scheduling with 512-token chunks, the
hash-chained prefix cache, one asyncio loop with the model step in a worker
thread.

**W1 (random 1024/256): output tok/s, and TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|
| vLLM 0.19.1 | 78 · 12.83 | 297 · 13.30 | 935 · 15.37 | 1,403 · 19.01 | 1,906 · 26.18 |
| SGLang | 78 · 12.60 | 297 · 13.22 | 934 · 15.35 | 1,423 · 18.36 | 1,286 · 41.18 |
| nano (Phase 1) | 80 · 12.38 | 274 · 13.11 | 870 · 15.55 | 1,252 · 19.93 | 1,590 · 27.47 |

Decode per token matches vLLM up to c=16. The gap at c=32-64 is prefill:
nano's TTFT p50 is 563 / 1,538 ms at c=32 / 64 against vLLM's 225 / 771,
because every prefill step stalls the decodes and vice versa.

**W2 (shared prefixes)**

| server | w2a req/s | w2a TTFT p50 | w2b req/s | w2b TTFT p50 |
|---|---:|---:|---:|---:|
| vLLM | 7.38 | 222 ms | 3.21 | 580 ms |
| SGLang | 7.20 | 201 ms | 2.98 | 1,701 ms |
| nano | 5.46 | 394 ms | 2.63 | 3,349 ms |
| nano, no prefix cache | 3.14 | 2,951 ms | - | - |

**W3 (4096-token prompts arriving on 32 running decodes): the decodes' ITL**

| server | p50 | p99 | p99.9 | max | long-prompt TTFT p50 |
|---|---:|---:|---:|---:|---:|
| vLLM (default chunking) | 17.3 | 28.4 | 148 | 153 | 362 |
| vLLM, 512-token chunks | 17.2 | 44.9 | 51 | 52 | 435 |
| vLLM, no chunked prefill | 17.1 | 20.3 | 273 | 434 | 336 |
| SGLang (4096-token chunks) | 17.1 | 33.5 | 319 | 590 | 351 |
| nano, prefill-first, 512 chunks | 20.6 | 45.0 | 396 | 902 | 462 |
| nano, prefill-first, no chunking | 20.4 | 44.5 | 323 | 669 | 400 |

With prefill-first scheduling chunking does not help: a 4096-token prompt
becomes eight consecutive prefill steps that all stall the decodes, plus the
chunking overhead. vLLM's 512-token chunks ride along with the decodes and
cap the stall at 52 ms.

**Where the host time goes (nsys of the nano server)**

| capture | step | GPU busy | GPU idle |
|---|---:|---:|---:|
| W1 c=16 | 16.1 ms | 14.1 ms | 2.0 ms |
| W1 c=64 | 33.2 ms | 28.7 ms | 4.5 ms |
| W3 | 20.2 ms | 16.3 ms | 3.9 ms |

Offline (Phase 1) the idle was 0.66-0.87 ms per step. Online it is 2-4.5
ms: filling the graph's staging buffers takes 0.67 ms at c=16 and 2.9 ms in
W3, while 32 streams' detokenization and SSE formatting run on the event
loop thread and hold the GIL the model thread needs. A shorter GIL switch
interval (0.5 ms) did not help (1,098 vs 1,252 tok/s at c=32), so the fix
is to move the model's thread out of that process (M3) and to hide the host
work behind the GPU (M4), not to tune the interpreter.

### M1-M6 (run `m1to6-20261023-015904-colab`)

One server per milestone, each with the previous milestones' changes, on the
same VM and KV capacity as M0 (9,942 blocks). vLLM runs its defaults (async
scheduling on, 2048-token chunks); its W2 rows are from this run, the W1 and
W3 rows from M0. `ok` held everywhere.

#### M1: radix-tree prefix cache

Block-granular radix tree in `block_manager.py` (refcounted paths, LRU leaf
eviction, edges extended in place on decode, duplicate blocks folded into
the tree). Measured with M0's prefill-first scheduler so only the cache
changes:

| W2 | vLLM | nano M0 (hash chain) | nano M1 (radix tree) |
|---|---:|---:|---:|
| w2a req/s · TTFT p50 | 7.42 · 200 ms | 5.46 · 394 ms | 6.91 · 268 ms |
| w2b req/s · TTFT p50 | 3.27 · 552 ms | 2.63 · 3,349 ms | 2.71 · 1,723 ms |
| hit rate w2a / w2b | 92.6% / 50.3% | - | 92.6% / 49.6% |

The hit rate cannot improve (a block-granular tree matches exactly the
prefixes the hash chain did) and matches vLLM's. What changes is the host
cost of admission, which prefill-first scheduling puts on the critical path:
`prefix_bench.py` replays W2 against the block managers alone at serving
scale (VM CPU, `m1tests-20261023-015432-colab`):

| per request, p50 | hash chain | radix (Python) | radix (C++, M6) |
|---|---:|---:|---:|
| admit, 2k shared prefix (w2a) | 18.5 ms | 168 µs | 16 µs |
| admit with eviction (w2b) | 3.1 ms | 314 µs | 33 µs |

`NUM_KVCACHE_BLOCKS=0` now sizes the cache from `gpu_memory_utilization`
after a profiled forward pass: 9,958 blocks (159k tokens; vLLM 159-163k),
35,980 MiB in use. `/stats` and `/metrics` export free/cached/used blocks,
query/hit tokens (preempted sequences' re-admissions counted apart),
evictions and internal fragmentation. The snapshots around each workload are
taken idle, where fragmentation is 0; under load it is below one block per
running sequence: at most 1.5% of the tokens in W1 (1024-1280-token
sequences), about 0.6% on average.

#### M2: mixed prefill/decode batches

Decode rows first (the decode kernel on `[:B]`), prefill chunks after them
(the varlen prefill kernel on `[B:]`), one token budget. W3, 32 decodes with
4096-token prompts arriving:

| config | ITL p50 | p99 | p99.9 | max | long-prompt TTFT p50 |
|---|---:|---:|---:|---:|---:|
| vLLM, 512-token chunks (M0) | 17.2 | 44.9 | 51 | 52 | 435 |
| vLLM default, 2048 (M0) | 17.3 | 28.4 | 148 | 153 | 362 |
| nano M0, prefill-first, 512 | 20.6 | 45.0 | 396 | **902** | 462 |
| nano M2, 256 | 20.1 | 36.3 | 38 | 50 | 678 |
| nano M2, 512 | 20.2 | 49.8 | 54 | **55** | 467 |
| nano M2, 1024 | 19.9 | 76.6 | 91 | 93 | 400 |
| nano M2, 2048 | 20.0 | 35.8 | 154 | 156 | 354 |

The worst decode stall drops from 902 to 55 ms at the same chunk size, and
the chunk size now trades the decodes' tail against the long prompts' TTFT
the way vLLM's does (nano at 2048 matches vLLM's default at 153 ms).

In W1, mixing cut TTFT p50 at c=64 from 1,538 to 693 ms (vLLM 771), but the
first M2 server's c=16 and c=32 sweeps lost ~3 s each to Triton compiling
kernel variants mid-serving (LOG.md); fixed afterwards by aligning the batch
metadata and compiling every eager variant before serving, and re-measured
in the final round.

#### M3: engine core in its own process

`EngineCore` (scheduler, block manager, model runner; no asyncio) runs a busy
loop in a spawned process; the frontend keeps tokenization, detokenization,
stop strings and SSE, connected by two pipes with one message per step.

| W1: tok/s · TPOT p50 (ms) | c=1 | c=4 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|
| vLLM (M0) | 78 · 12.83 | 297 · 13.30 | 935 · 15.37 | 1,403 · 19.01 | 1,906 · 26.18 |
| nano M0 | 80 · 12.38 | 274 · 13.11 | 870 · 15.55 | 1,252 · 19.93 | 1,590 · 27.47 |
| nano M2, 512 | 79 · 12.54 | 298 · 13.23 | 719\* · 16.29 | 1,073\* · 22.31 | 1,551 · 33.27 |
| nano M2, 2048 | 79 · 12.58 | 298 · 13.25 | 857 · 16.26 | 1,236 · 22.25 | 1,611 · 30.77 |
| nano M3 | 80 · 12.36 | 309 · 12.80 | 905 · 16.16 | 1,387 · 18.45 | 1,883 · 25.28 |
| nano M4 | 82 · 12.08 | 317 · 12.48 | 963 · 14.34 | 1,451 · 18.22 | 1,973 · 24.56 |
| nano M4, one process | 82 · 12.08 | 316 · 12.47 | 960 · 14.28 | 1,444 · 17.70 | 1,958 · 24.45 |

\* includes a mid-serving Triton compile (M2 above).

Out of the frontend's process the model loop no longer waits for the GIL
while 64 streams are detokenized: +21% throughput at c=64 over M2 and TPOT
down from 33.3 to 25.3 ms. W3's background ITL p50 falls from 20.2 to
17.3 ms (vLLM 17.2).

#### M4: async scheduling

Step N+1 is scheduled and launched before step N's tokens reach the host:
sampled tokens stay in a device buffer that the next step's inputs gather
from (inside the decode CUDA graph too), and come back through two pinned
buffers with an event each. Greedy output is token-identical to synchronous
scheduling (CPU and GPU tests), and a steady decode step makes no host sync
besides waiting for the previous step's event (`torch.cuda.set_sync_debug_mode("error")`).

M4 is at or above vLLM on every W1 point (table above), and on W2 and W3:

| | vLLM | nano M3 | nano M4 |
|---|---:|---:|---:|
| w2a req/s · TTFT p50 | 7.42 · 200 ms | - | 7.94 · 117 ms |
| w2b req/s · TTFT p50 | 3.27 · 552 ms | - | 3.33 · 476 ms |
| W3 background ITL p50 · max | 17.2 · 52 (512 chunks) | 17.3 · 60 | 16.4 · 50 |

nsys of the server (core process; M0 in parentheses):

| capture | step | GPU idle per step |
|---|---:|---:|
| W1 c=16 | 14.1 ms (16.1) | 0.04 ms (2.0) |
| W1 c=64 | 30.9 ms (33.2) | 0.84 ms (4.5) |
| W3 | 16.3 ms (20.2) | 0.03 ms (3.9) |

The plan expected async scheduling to hide only the engine's own host time,
not GIL waits, and so to need M3 first. Measured, async scheduling in the
frontend's process recovers the same throughput (1,958 vs 1,973 tok/s at
c=64): with the next step already queued, any host time shorter than a GPU
step is hidden, GIL waits included. The separate process still halves TTFT
at c=64 (330 vs 666 ms p50), since admissions no longer queue behind
detokenization.

#### M5: CUDA graphs for mixed steps (not done)

The condition was nsys showing mixed steps CPU-bound. With M3 and M4 the
c=64 median step, an eager mixed step (362 kernels), leaves the GPU idle
0.84 ms of 30.9 ms: 2.7% is the most mixed-step graphs could recover, for
about 45 more graphs, 5-10 s of startup and a prefill kernel that finds its
sequence on the device.

#### M6: the radix tree in C++

`csrc/radix_tree.cpp` (nanobind, built with the system compiler on first use,
no CMake) holds the tree, the block allocator and each sequence's tokens;
`prefix_cache_backend: cpp` selects it and the Python tree stays the
reference (the radix tests run against both, plus a randomized parity test).
Admission is 10x faster than the Python tree (16 vs 168 µs, table in M1), and
end to end the difference is noise, as expected once M4 hides host time:
w2a 7.89 vs 7.94 req/s, w2b 3.34 vs 3.33.

### Final round (runs `final-20261023-030032-colab`, `final-offline-20261023-034858-colab`)

Every server in one run on one VM: vLLM 0.19.1 with its defaults (async
scheduling, 2048-token chunks, 163k KV tokens), SGLang with its defaults
(4096-token chunks, 140k KV tokens), and nano with everything from Phase 2
(core process, async scheduling, mixed batches, radix prefix cache; 9,942
blocks = 159k tokens), at vLLM's chunk size and at 512.

**W1: output tok/s · TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=32 | c=64 | TTFT p50 c=1 / c=64 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM | 78 · 12.81 | 297 · 13.28 | 939 · 15.32 | 1,411 · 20.25 | 1,976 · 25.85 | 26 / 408 ms |
| SGLang | 78 · 12.60 | 296 · 13.23 | 940 · 14.82 | 1,424 · 18.28 | 1,284 · 41.26 | 53 / 2,235 ms |
| nano, 2048 | 82 · 12.09 | 316 · 12.48 | 975 · 14.39 | 1,470 · 19.17 | 1,997 · 25.77 | 33 / 463 ms |
| nano, 512 | 82 · 12.09 | 316 · 12.48 | 963 · 14.28 | 1,453 · 17.62 | 1,972 · 24.47 | 32 / 342 ms |
| nano M0 (Phase 1) | 80 · 12.38 | 274 · 13.11 | 870 · 15.55 | 1,252 · 19.93 | 1,590 · 27.47 | - / 1,538 ms |

**W2: req/s · TTFT p50 · prefix hit rate**

| server | w2a (one 2k prefix) | w2b (48 x 4k prefixes, evicting) |
|---|---:|---:|
| vLLM | 7.32 · 270 ms · 92.6% | 3.28 · 580 ms · 50.6% |
| SGLang | 7.20 · 206 ms | 3.02 · 1,710 ms |
| nano, 2048 | 7.84 · 165 ms · 92.6% | 3.34 · 402 ms · 48.5% |
| nano, 512 | 7.97 · 117 ms · 92.6% | 3.33 · 477 ms · 48.4% |

**W3: background decodes' ITL (ms) with 4096-token prompts arriving**

| server | p50 | p99 | p99.9 | max | background tok/s | long-prompt TTFT p50 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM (2048) | 17.22 | 28.3 | 148 | 152 | 1,742 | 359 |
| nano, 2048 | 16.36 | 29.4 | 147 | 151 | 1,820 | 356 |
| vLLM, 512 | 17.22 | 44.7 | 51 | 53 | 1,754 | 434 |
| nano, 512 | 16.41 | 43.2 | 47 | 50 | 1,821 | 423 |
| SGLang (4096) | 17.05 | 34.1 | 319 | 593 | 1,729 | 349 |

**Offline decode, ms per step (nano_bench, Triton kernels, CUDA graphs)**

| | b=1 | b=16 | b=64 |
|---|---:|---:|---:|
| Phase 1 | 12.15 | 14.63 | 23.51 |
| Phase 2, synchronous | 12.27 | 14.54 | 23.54 |
| Phase 2, async scheduling | 12.04 | 14.13 | 22.57 |

**nsys, nano server: step and GPU idle per step, M0 against final**

| capture | M0 | final |
|---|---:|---:|
| W1 c=16 | 16.1 ms, idle 2.0 | 14.0 ms, idle 0.04 |
| W1 c=64 | 33.2 ms, idle 4.5 | 22.0 ms, idle 0.03 |
| W3 | 20.2 ms, idle 3.9 | 16.2 ms, idle 0.03 |
| offline b=16 | sync 14.9 ms, idle 0.91 | async 14.1 ms, idle 0.04 |

(The final W1 c=64 capture ran with 2048-token chunks, so its median step
is a decode graph replay; with 512-token chunks it is an eager mixed step,
idle 0.84 of 30.9 ms, M4 above.)

**Where nano stands.** At equal chunk sizes nano matches or beats vLLM on
throughput, TPOT and median ITL at every W1 concurrency, on W2 throughput
and TTFT, and on W3's ITL tail within a few percent (p99 29.4 vs 28.3 ms at
2048-token chunks, lower at 512). Its w2b hit rate is 2 points lower (48.5
vs 50.6%), not investigated: vLLM evicts single blocks from the front of
its free queue, nano evicts whole leaf tails by LRU, which can keep a
different set of prefixes under w2b's shuffled reuse. Two gaps remain. TTFT at c=1 is 33 against 26 ms: a 1024-token prefill is slower
in nano's kernels (Phase 1 tuned decode, not prefill). And at c=64 with
2048-token chunks, TTFT p50 is 463 against 408 ms; with 512-token chunks
nano is at 342 ms. The mid-serving compile fix was verified with an empty
Triton cache (LOG.md): the first M2 server's c=16 / 32 W1 went from 719 /
1,073 to 852 / 1,234 tok/s.

**Against the plan's expectations**

| milestone | expected | measured |
|---|---|---|
| M1 | admission from ms to 0.1-0.3 ms; same hit rate as vLLM | 18.5 ms to 168 µs (16 µs in C++); same hit rate |
| M2 | p99 ITL from a full 4k prefill to one step; TTFT +10-20% at 256-token chunks; W1 c=64 +10-20% | max ITL 902 to 55 ms at 512 chunks; TTFT +45% at 256 (678 vs 467 ms); W1 flat until M3/M4 |
| M3 | ITL a few ms lower at c>=32; c=1 50-100 µs worse | c=64 ITL p50 28.9 to 23.9 ms, TPOT 33.3 to 25.3; c=1 slightly better (12.54 to 12.36) |
| M4 | offline about 11.5 / 13.8 / 22.9 ms; no per-step idle in nsys | 12.04 / 14.13 / 22.57 ms; idle 0.03-0.04 ms per step |
| M5 | only if mixed steps are CPU-bound | not CPU-bound (2.7% idle), not done |
| M6 | 10-50x in the microbenchmark; e2e in the noise | 10x; e2e in the noise |
