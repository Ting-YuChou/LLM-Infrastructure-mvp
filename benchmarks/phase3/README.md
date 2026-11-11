# Phase 3: speculative decoding

Roadmap Phase 3 for `src/serving/nano_vllm/`: lossless speculative decoding
with the public DFlash drafter (`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`,
the checkpoint vLLM and SGLang load), measured against both on one Colab
A100-SXM4-40GB with Llama 3.1 8B Instruct (BF16 and the AWQ INT4 checkpoint).

| milestone | change |
|---|---|
| M0 | baselines on real prompts (vLLM and SGLang with and without DFlash, nano as it is) and the drafter's acceptance study |
| M1 | the DFlash drafter in nano, reference block attention |
| M2 | synchronous greedy cycle on the existing kernels: output equal to no speculation |
| M3 | block attention kernel: one KV read for all of a sequence's verify rows |
| M4 | the cycle as CUDA graphs, 4-bit drafter |

## How to run

```bash
BENCH_PHASE=phase3 COLAB_SESSION=phase3 COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh          # new VM
BENCH_PHASE=phase3 COLAB_SESSION=phase3 COLAB_KEEP=1 bash benchmarks/phase0/colab_phase0.sh rerun    # same VM
```

`run_phase3.sh` documents the workloads, the steps and the `P3_*` variables.
Each run writes `summary.md` in its results directory.

## Results

None yet: M0 has not run.
