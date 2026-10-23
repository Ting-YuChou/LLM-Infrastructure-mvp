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
