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

### Run m0-smoke2-20261111-072152-colab (same smoke, sampled, after the two fixes)

- All four servers (vllm_dflash, vllmw4_dflash, sglang_dflash, nano_awq)
  started and completed every request; every server counts the same prompt
  tokens (42 / 163 / 99 / 1223).

### Run m0-20261111-074458-colab (full M0: 16 prompts per dataset, 8 servers)

- accept 1233 s, serve 4754 s (about 1 h 40 min, ~6 CU). All steps rc=0, all
  requests completed. The VM was kept (`COLAB_KEEP=1`) and was reclaimed by
  Colab afterwards: the next run starts from a new VM (venv installs again).
- Baselines at c=1, temperature 0 (output tok/s): vLLM AWQ + DFlash is the
  one to beat (380 / 589 / 434 / 245 on alpaca / humaneval / gsm8k / long);
  vLLM BF16 + DFlash 220 / 322 / 256 / 161; SGLang BF16 + DFlash
  196 / 295 / 222 / 157.
- vLLM's cycle (TPOT x tokens per cycle as it counts them): AWQ 8.2-8.7 ms
  against a 5.64 ms decode step, BF16 15.2-15.6 ms against 12.7 ms. Draft
  plus overhead is 2.5-3.0 ms per cycle on both.
- DFlash with a fixed block stops paying at high concurrency on the AWQ
  target: vllmw4_dflash is 0.99x (c=16) and 0.58x (c=64) of vllmw4 on
  alpaca. nano AWQ without speculation is already the fastest server at
  c=16 on alpaca (2567) and at c=64 on both (6012 / 5159).
- Acceptance: tokens per cycle (greedy, block 10) 3.56 / 5.21 / 4.14 / 3.05
  on the BF16 target, 3.47 / 5.12 / 4.08 / 2.83 on the AWQ target. The
  servers' own counters agree (vLLM 3.53 / 5.28 / 4.15 / 3.03). The replay
  matched the live loop on every prompt.
- Decisions the study was for:
  - 4-bit (RTN) drafter: at most 2.6% fewer tokens per cycle than BF16
    (long, BF16 target), usually under 2%. Adopted.
  - Coupled sampling at T=0.7: within 2.1% of greedy drafts (above on long
    and on gsm8k/AWQ), 2-9% above independent draws. Adopted.
  - Block 6 / 4 lose 2-18% / 10-35% of the tokens per cycle of block 10 at
    b=1; shorter verifies are for high batch only (M6).
  - The drafter's second choice is the target's token at the first rejected
    depth 33-51% of the time (input for the M10 sibling simulation).
- M0 gate (projected nano c=1 >= 1.2x the best DFlash baseline, 4-bit
  drafter): met on the AWQ target (1.33x-1.76x), NOT met on the BF16 target
  (1.11x-1.14x on alpaca / humaneval / gsm8k, 1.29x on long). On BF16 the
  12 ms verify (16 GB of weights per pass) is 84% of the cycle and is the
  same for every engine; only the 2.5-3 ms of draft and overhead can shrink.
- Not done in M0: kbench of the prefill kernel at verify shapes, nano's
  b=1 against b=16 stream divergence, vLLM adaptive verification on the
  A100, spec_sim.py.
