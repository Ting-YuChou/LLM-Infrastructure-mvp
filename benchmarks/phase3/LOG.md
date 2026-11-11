# Phase 3 log

Chronological record of each Phase 3 run on the Colab A100 (40GB): what ran,
what failed and the fix. Results tables are in README.md; raw runs in
results/<run_id>/.

## M0: harness (2026-11-10)

- Real prompts instead of random tokens: Alpaca, HumanEval, GSM8K (the sets
  the drafter's card reports) and CNN/DailyMail articles, rendered with the
  chat template and sent as text to /v1/completions; requests stop at EOS
  (`BENCH_IGNORE_EOS=0` in lib.sh, `serve_summary.py --variable-output`).
- The baselines need a vLLM with `"method": "dflash"`; 0.19.1 (phases 0-2)
  stays as the load generator and nano's venv. The drafter is pinned to
  revision d3af30de; the acceptance study runs its `dflash.py`.
- `accept_study.py --self-test` (stub models, CPU) passes locally.

### Run m0-smoke-20261111-065251-colab (4 prompts per dataset, DFlash servers only)

- Baselines installed as vLLM 0.33.0 (torch 2.14.0) and SGLang 0.5.23.
- The acceptance study ran end to end (293 s); the replay of the reference
  arm matched the live loop on every prompt.
- No server produced a number. vLLM and SGLang with DFlash died at start:
  flashinfer JIT-builds its sampling ops with `$CUDA_HOME/bin/nvcc`, and the
  run had sourced cuda129's CUDA 12.8 toolkit (nvcc, cudart and cccl only,
  for nano's kernels), which has no `curand.h`. The toolkit now goes to the
  nano server's environment only.
- `vllm bench serve --dataset-name custom` needs pandas (`vllm[bench]`), so
  nano's runs wrote no result. Added to the load generator's venv.
