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
