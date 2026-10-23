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

Filled in per milestone.
