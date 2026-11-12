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

## M2: the synchronous cycle (2026-11-11, CPU only)

- A step with decode rows is a cycle when the engine has a proposer
  (`speculative_model`, or one handed to EngineCore): each decode row is the
  last accepted token plus K drafts, verified in the same forward pass as
  any prefill rows (`ModelRunner._launch_spec`); the scheduler appends the
  accepted drafts and the target's next token, and counts the KV behind them
  (`Scheduler._complete_spec`).
- Every computed token's taps go to the drafter's KV pool (prompt chunks and
  verify rows), which shares block ids with the target's cache; KV slots for
  K positions past the last token are reserved, and a request needs room for
  them (`kv_capacity`, the RoPE table and the admission check are K larger).
- Fixed: a request that finished inside a step that emitted several of its
  tokens had the finish reason on every one, and the frontend dropped the
  rest after the first (`EngineCore._complete`).
- Async scheduling is off while speculating (M7), drafts are greedy (M5),
  and the decode CUDA graphs are not captured (M4 replaces them); the verify
  pass goes through the piecewise graphs.
- Tests (CPU, fp32): `test_nano_vllm_spec_engine.py` (scripted target and
  proposer: accept lengths 0..K, EOS inside a block, max_tokens mid-block,
  stop strings, aborts, preemption, the prefix cache on both backends) and
  `test_nano_vllm_spec_exact.py` (tiny Llama + random DFlash drafters: the
  output equals transformers' greedy decode for two drafters, the drafter's
  KV equals `write_context` of transformers' hidden states). Each was run
  against seven deliberate bugs; all are caught.
- Not run on the GPU yet (the VM is gone): M1's `parity.py`, M2's gates on
  the real model (tokens per cycle within 2% of M0, prefill at most 3%
  slower), and a warm-up of the prefill kernel at verify shapes. On the GPU
  a batch of more than 6 sequences has more than 64 verify rows, where the
  lm_head switches from int8 to dense: to settle before M6.

### Run m2-smoke-20261112-041533-colab (new VM: installs, parity, nano AWQ with and without the drafter)

- 20 minutes, everything ran the first time on the GPU.
- Parity (M1 gate, top-1 agreement >= 99% over 200 blocks): 98.89%, NOT
  met by 0.11 points. All 20 disagreements are where the reference's top-1
  leads its top-2 by at most 0.25, and the largest logit difference between
  the two implementations is 0.25: one bf16 step at these magnitudes. 90% of
  the blocks agree at every position. Read as rounding, not a porting error;
  the tokens per cycle below say the same.
- Tokens per cycle as nano counts them (AWQ target, c=1): 3.39 / 5.36 / 3.98
  / 3.04 on alpaca / humaneval / gsm8k / long, against vLLM AWQ + DFlash on
  the same prompts in M0: 3.38 / 5.44 / 3.90 / 3.04.
- TTFT p50 with the drafter against without (AWQ, c=1): 9.46 / 14.99 / 14.79
  / 104.25 ms against 9.51 / 14.69 / 14.41 / 102.56 ms: at most 2.6% slower.
- Completions with identical text, drafter against none at c=1: 12 of 32.
  The server without a drafter agrees with itself as often between one
  request at a time and 16 at a time (12 of 32), so text equality cannot
  tell a bug from rounding on the GPU; spec_check.py was written for that.

### Run m2-20261112-043725-colab (same VM: spec_check, GPU tests, AWQ at every concurrency, BF16 pair)

- tests/gpu and the nano unit tests: 1039 passed, none skipped (509 s).
- spec_check.py (every emitted token against the argmax of one
  teacher-forced pass; about 3,000 tokens per row):

  | target, mode, batch | not argmax | largest margin |
  |---|---:|---:|
  | awq plain b=1 / b=16 | 0.43% / 0.33% | 0.250 / 0.250 |
  | awq spec b=1 / b=16 | 0.30% / 0.40% | 0.125 / 0.250 |
  | bf16 plain b=1 / b=16 | 0.27% / 0.17% | 0.250 / 0.250 |
  | bf16 spec b=1 / b=16 | 0.14% / 0.21% | 0.125 / 0.125 |

  No margin above 0.5 anywhere: with the drafter the engine emits the
  target's own tokens exactly as often as without.
- BF16 target, tokens per cycle at c=1: 3.48 / 5.35 / 4.03 / 3.04 against
  vLLM BF16 + DFlash 3.53 / 5.28 / 4.15 / 3.03 (gsm8k 2.9% lower, the rest
  within 1.4%). TTFT with the drafter is not higher (long: 91.6 ms against
  94.1 ms).
- Speed, which M2 does not aim at (output tok/s at c=1, alpaca / humaneval /
  gsm8k / long): AWQ 207 / 316 / 238 / 144 against 207 / 205 / 205 / 183
  without the drafter and vLLM AWQ + DFlash's 380 / 589 / 434 / 245; BF16
  142 / 212 / 164 / 108 against 83 / 83 / 83 / 80 and vLLM BF16 + DFlash's
  220 / 322 / 256 / 161. A cycle takes about 16 ms on AWQ and 24 ms on BF16
  (TPOT x tokens per cycle): the verify pass plus about 11 ms of drafter,
  context writes and host work that M3 and M4 are for.
- With a fixed block the drafter costs throughput at high concurrency: AWQ
  alpaca 736 / 1722 / 2259 at c=4 / 16 / 64 against 753 / 2567 / 6012
  without (M0). M6's verify-length policy is for this.
- M2 gates: output is the target's own (CPU exact, GPU by spec_check): met.
  Tokens per cycle within 2% of the reference: 7 of 8 within 2.1% of vLLM's
  counters on the same prompts, gsm8k on BF16 2.9% lower. Prefill at most
  3% slower: met. Nothing changes without a drafter: the GPU tests pass and
  nano AWQ at c=1 measures 206 / 205 / 205 / 183 (M0: 207 / 206 / 205 / 178).
- Two runs, about 55 minutes of VM, ~3.5 CU. The VM is stopped.

## M3: the block attention kernel (2026-11-12)

- `kernels/block_attention.py`: the Triton kernel behind
  `paged_block_attention`. It is the decode kernel with more rows: a program
  owns one (sequence, kv head, split, row tile) and reads its slice of the
  context once for `block_r` of the `q_max * group` queries of that kv head;
  the mask is a per-row limit (causal for the target's verify rows, the
  whole context for the drafter's block). A second kernel merges the splits.
- `kernels/rope_cache.py`: `norm_rope_and_cache_write`, one kernel for the
  drafter's q_norm / k_norm, RoPE and KV write per layer, also for context
  tokens (K/V only), which went through the PyTorch reference before.
- Wiring: a step's verify rows are block rows in `BatchMetadata`
  (`num_block_rows`, `block_len`); `model._paged_attention` sends them to
  the block kernel and any prefill rows after them to the prefill kernel
  (with their own aligned fields, as after decode rows). The warm-up runs a
  speculative cycle per number of verify rows, so no kernel compiles while
  serving.
- Checked without a GPU: 320 nano unit tests; the Triton interpreter
  (`interp_check.py`, now with `INTERP_ONLY=<regex>`) on 96 block cases and
  16 norm + RoPE cases against the references. Five deliberate bugs in the
  kernel: four caught; the fifth (not clipping a row's limit to the split's
  end) changes nothing, because a tile passes the end of a split only where
  that end is the context length, so the clip was removed.

### Run m3-kernel-20261112-064729-colab (new VM: installs, block tests, kbench, tune)

- First version: one program per (sequence, kv head, split) with all 64
  rows (40 real), the decode path's reduce kernel, the decode kernel's
  launch tiers. Against the prefill kernel (the verify path of M2) it is
  3.3 to 5.5 times faster on every shape and faster than vLLM's
  flash-attention on all but one (B=1, 256 tokens), but against a decode
  step's attention at the same context: 1.08x / 1.38x / 2.13x for one
  sequence at 256 / 1024 / 4096 tokens, 1.67x / 1.49x / 1.50x for 64.
- The drafter's norm + RoPE + KV write: 5 to 17 us against 45 to 206 us for
  the PyTorch reference (1 to 640 tokens).
- 8 of 340 tests failed, none a kernel bug: `block_n=128, num_stages=3` (the
  decode kernel's tier from 64 sequences) needs more shared memory than a
  program has with 64 fp32 rows; and my bf16 and fp8 tolerances were tighter
  than what the existing fp8 tests allow (now compared the same way).

### Run m3-tune-20261112-070917-colab (same VM: rows per program, a reduce kernel per sequence)

- The reduce kernel had one program per (row, query head): 5,120 programs
  at 16 sequences. A version with one program per (sequence, query head)
  looping over the splits: 16 sequences at 256 tokens 56.5 -> 31.7 us, but
  one sequence at 1024 tokens 19.7 -> 34 us. Each iteration waits for its
  own load: about 0.5 us per split, 15 us at 32 splits.
- Rows per program (`block_r` 16 / 32 / 64, the rest in further row tiles
  that read the same context again): little difference at any batch size,
  so the second and third read of a tile come from the L2 cache.
- 241 block and warm-up tests passed.

### Run m3-tune2-20261112-072015-colab (same VM: reduce per (row, kv head), all splits at once)

- A reduce program per (row, kv head) that loads every split at once is as
  fast as or faster than the loop on every shape, so the loop is gone.
- Best static settings per batch size (us; decode is the decode kernel's
  time from the first run):

  | B, context | decode | block | block / decode | settings (block_r, block_n, warps, stages, splits) |
  |---|---:|---:|---:|---|
  | 1, 256 | 16.9 | 13.3 | 0.79 | 16, 64, 4, 2, half |
  | 1, 1024 | 14.4 | 14.6 | 1.01 | same |
  | 1, 4096 | 20.9 | 26.6 | 1.27 | same |
  | 4, 1024 | 21.0 | 26.8 | 1.27 | 16, 64, 4, 2, quarter |
  | 4, 4096 | 62.7 | 76.4 | 1.22 | same |
  | 16, 256 | 21.8 | 25.0 | 1.15 | same (one split) |
  | 16, 1024 | 61.4 | 75.6 | 1.23 | same |
  | 16, 4096 | 208.8 | 274.3 | 1.31 | same |
  | 64, 256 | 57.8 | 76.4 | 1.32 | 64, 64, 4, 2, one split |
  | 64, 1024 | 198.8 | 249.0 | 1.25 | same |
  | 64, 4096 | 764.6 | 930.4 | 1.22 | same |

  Below 64 sequences the kernel is bound by a program's latency, not by
  bandwidth: programs of 16 rows (the decode kernel's size) and fewer
  splits win. From 64 a program takes the whole block. These are
  `BLOCK_CONFIGS`.
- The VM was reclaimed by Colab four minutes into the next run (kbench,
  all tests, spec_check, serving); a new A100 was not available for a while
  ("Service Unavailable").

### Runs m3-20261112-085116-colab and m3-20261112-092848-colab (new VMs: all tests, spec_check, kbench gate, serving)

- The same code and steps twice, the second by mistake (below). The two
  agree within 2% everywhere but alpaca at c=16 (2055 / 1962 tok/s);
  numbers here are the first run's.
- tests/gpu and the nano unit tests: 1287 passed, none skipped (771 s).
- G2, verify attention for Q=10 against a decode step's attention at the
  same context (us, bf16):

  | B, context | decode | vLLM flash-attn | prefill (M2) | block | block / decode |
  |---|---:|---:|---:|---:|---:|
  | 1, 256 | 16.7 | 11.4 | 20.7 | 13.3 | 0.79 |
  | 1, 1024 | 14.4 | 29.6 | 65.3 | 14.6 | 1.02 |
  | 1, 4096 | 20.8 | 103.8 | 243.2 | 26.4 | 1.27 |
  | 16, 1024 | 61.5 | 115.0 | 229.6 | 75.6 | 1.23 |
  | 16, 4096 | 206.8 | 439.9 | 875.0 | 274.6 | 1.33 |
  | 64, 256 | 57.8 | 126.1 | 240.3 | 76.2 | 1.32 |
  | 64, 1024 | 198.7 | 477.4 | 835.2 | 248.5 | 1.25 |
  | 64, 4096 | 767.1 | 2499.6 | 3248.2 | 926.6 | 1.21 |

  Gate (at most 1.3x for one sequence, 1.5x for 64): met on all six
  shapes; the closest is one sequence at 4096 tokens, 1.27x. Shorter
  blocks cost less: Q=3 / Q=5 at 1024 tokens are 0.87x / 0.95x for one
  sequence and 1.11x / 1.06x for 64.
- spec_check.py: emitted tokens that are not the teacher-forced argmax
  0.17-0.50% with the drafter, 0.17-0.43% without; largest margin 0.25, none
  above 0.5. Tokens per cycle at c=1 (AWQ): 3.50 / 5.32 / 3.97 / 3.02 (M2:
  3.39 / 5.36 / 3.98 / 3.04).
- Output tok/s at c=1 (alpaca / humaneval / gsm8k / long): AWQ + DFlash
  246 / 359 / 276 / 182 (M2: 207 / 316 / 238 / 144; vLLM AWQ + DFlash:
  380 / 589 / 434 / 245); BF16 + DFlash 172 / 266 / 203 / 138 (M2: 142 /
  212 / 164 / 108; vLLM BF16 + DFlash: 220 / 322 / 256 / 161). A cycle is
  now about 14 ms on AWQ and 20 ms on BF16 (16 and 24 after M2): the verify
  pass is about 5 ms on AWQ, so some 9 ms are still the drafter in BF16,
  eager kernels and host round trips, which is M4.
- AWQ + DFlash at c=4 / 16 / 64: alpaca 963 / 2055 / 2739 (M2: 736 / 1722 /
  2259), humaneval 1362 / 2898 / 4007; vLLM AWQ + DFlash at c=16 / 64 is
  2084 / 2943 and 2950 / 4204. With the fixed block it is still below nano
  without a drafter at c=16 and up (2567 / 6012 on alpaca), as is
  spec_check at b=16 on AWQ (1901 against 1960 tok/s). M6.
- Without a drafter nothing moved: nano AWQ at c=1 206 / 204 / 204 / 177
  (second run 207 / 206 / 206 / 178; M0 207 / 206 / 205 / 178). TTFT at c=1
  with the drafter is at most 3% higher (17.1 against 16.6 ms on humaneval).
- Compute units: 16.0 CU between 08:28 and 20:54 UTC (435.80 -> 419.76),
  of which one run of 35 minutes was needed. The retry loop I left waiting
  for an A100 tested "did the run start" with `grep` on its log; in this
  shell `grep` is a wrapper function that found no match in a log with
  progress-bar bytes, so the loop started the whole run again twice. The
  third was cut off when the laptop went to sleep; its VM stayed up until
  Colab reclaimed it and its results are lost. No session is active now.
  For the next loop: decide on the driver's exit code, and stop the session
  when the run ends.

## M4: the cycle as one CUDA graph (2026-11-12)

- After M3 a cycle took about 14 ms on AWQ where the verify pass alone is
  about 5 ms: the drafter's ~60 kernels, the target's 33 piecewise graphs
  with an eager attention per layer, the context writes and the sampling
  were each launched from the host.
- `ModelRunner.capture_spec_graphs`: one graph per padded number of
  sequences (the decode graphs' sizes) holding the whole cycle: draft, the
  drafts into the verify rows, the target's pass with taps, the drafter's
  context writes, sampling and the accepted counts. A step of verify rows
  only fills two pinned buffers and the sampler's settings, replays, and
  copies tokens and counts down. Steps with prompt rows next to the verify
  rows, or more sequences than `cuda_graph_max_bs`, take the eager path as
  before. A proposer opts in with `device_only` (it must read nothing but
  the batch's tensors); on the CPU a "graph" is the same function run
  eagerly, so the unit tests cover the staging, the padding and the
  acceptance.
- `Sampler.sample` is now `stage` (host: pack and upload the rows'
  settings) plus `draw` (device only). In a graph `draw` runs all five
  kernels; a row leaves the ones it does not need at once, and padded rows
  are staged as greedy.
- `size_kv_cache` with a drafter measures one eager cycle of max_num_seqs
  sequences (logits for every verify and draft row) and keeps that much
  twice: the graphs' pool, and an eager step next to it.
- The 4-bit drafter (`speculative_weight_quant: w4a16`, RTN at load) was
  already there from M1; M4 measures it.
- `spec_check.py` times every engine step that has no prompt rows: a decode
  step, or a whole cycle as the host sees it. This replaces the
  `nano_bench.py --speculative` of the plan: it runs the real prompts and
  needs no second harness.
- Checked without a GPU: 326 nano unit tests (6 new in
  `test_nano_vllm_spec_graph.py`; five deliberate bugs in the staging and
  acceptance all caught).

### Run m4-smoke-20261112-211212-colab (new VM: the new GPU tests, AWQ spec_check)

- The ten tests of `tests/gpu/test_nano_vllm_spec_gpu.py` passed: a replay
  gives the logits of the eager cycle (2e-2, as the decode graphs' test),
  returns what its own logits and drafts say, launches without waiting for
  the GPU, and samples per row.
- The engine's own order failed: `capture_graphs` captures the piecewise
  graphs first, in inference mode, and the cycle's capture after them was
  not ("Inplace update to inference tensor outside InferenceMode" from
  `capture_begin`). PyTorch keeps the default generator's graph state in
  tensors made when the first live graph is captured and updates them in
  place at every later capture and replay. Three warm-up tests failed on
  it, two sampler tests after them on the state the failed capture left,
  and spec_check stopped after its plain rows. `capture_spec_graphs` and
  `_replay_spec` now run in inference mode like the piecewise ones.
- A plain decode step as the host sees it (AWQ, one sequence): 5.00 ms
  median.

### Run m4-smoke2-20261112-213151-colab (same VM: the same tests, AWQ spec_check with both drafters)

- The cycle as the host sees it, AWQ target, one sequence (median of about
  750 cycles; a plain decode step is 5.01 ms):

  | drafter | ms / cycle | tokens / cycle | tok/s | M3 |
  |---|---:|---:|---:|---:|
  | BF16 | 7.69 | 4.00 | 475 | about 14.5 ms, 275 tok/s |
  | 4-bit RTN | 6.70 | 3.94 | 531 | - |

  G3 (at most 8.0 ms): met with either. Per dataset the cycle is 7.65 to
  7.82 ms (BF16 drafter) and 6.66 to 6.81 ms (4-bit): the long prompts'
  context costs 0.15 ms. The 4-bit drafter saves 1.0 ms and costs 1.5% of
  the tokens per cycle here.
- Emitted tokens that are not the teacher-forced argmax: 0.33-0.50% with
  either drafter (plain: 0.33-0.43%), largest margin 0.25.
- 16 sequences at once: a cycle of 160 verify rows takes 26 ms, a plain
  decode step of 16 rows 6.3 ms; at 4.0 tokens per cycle that is 1991
  against 1963 tok/s. 160 rows are past what the 4-bit GEMM does at the
  price of one (16). M6's verify lengths are for this.
- 101 of 103 tests passed. The two sampler tests failed in this order of
  test files: the sampler's own graph test captures outside inference mode
  while the warm-up tests' engines (in reference cycles) still held graphs
  captured inside it. Now every capture and replay of the model runner is
  in inference mode, the decode graphs' included, the sampler test's too,
  and `tests/gpu/conftest.py` collects garbage after each GPU test.

### Run m4-20261112-214147-colab (same VM: all tests, spec_check on both targets, serving)

- tests/gpu and the nano unit tests: 1303 passed, none skipped (927 s).
- The cycle as the host sees it, one sequence (median ms; spec_check.py):

  | target | plain decode step | cycle, BF16 drafter | cycle, 4-bit drafter |
  |---|---:|---:|---:|
  | AWQ | 5.00 | 7.75 | 6.68 |
  | BF16 | 12.29 | 15.45 | 14.50 |

  G3 (AWQ, at most 8.0 ms): met with either drafter. On BF16 the cycle is
  what M0 estimated for vLLM's (15.5 ms): the verify pass reads the same
  16 GB in every engine, and both lm_head passes (the drafter's and the
  target's) are 1 GB each in BF16.
- Emitted tokens that are not the teacher-forced argmax: 0.17-0.50% in all
  eight speculative rows (plain: 0.17-0.43%), largest margin 0.25.
- Output tok/s at c=1, temperature 0 (alpaca / humaneval / gsm8k / long):

  | server | tok/s | M3 | best DFlash baseline (M0) | ratio |
  |---|---|---|---|---|
  | nano AWQ + DFlash | 428 / 641 / 484 / 289 | 246 / 359 / 276 / 182 | vLLM AWQ: 380 / 589 / 434 / 245 | 1.13 / 1.09 / 1.11 / 1.18 |
  | nano AWQ + 4-bit DFlash | 477 / 714 / 547 / 316 | - | same | 1.26 / 1.21 / 1.26 / 1.29 |
  | nano BF16 + DFlash | 216 / 333 / 254 / 170 | 172 / 266 / 203 / 138 | vLLM BF16: 220 / 322 / 256 / 161 | 0.98 / 1.03 / 0.99 / 1.06 |
  | nano BF16 + 4-bit DFlash | 228 / 357 / 272 / 180 | - | same | 1.03 / 1.11 / 1.06 / 1.12 |

  SGLang BF16 + DFlash is 196 / 295 / 222 / 157, below all four rows' BF16
  numbers. G4 (above vLLM's and SGLang's DFlash at c=1 on chat and code,
  both targets): met with the 4-bit drafter on all eight; with the BF16
  drafter met on AWQ, and on the BF16 target nano is level with vLLM
  (alpaca and gsm8k 1-2% below, humaneval and long 3-6% above). The
  baselines are M0's, from another VM of the same type; nano AWQ without a
  drafter, the control, measures 207 / 206 / 205 / 177 here and 207 / 206 /
  205 / 178 there.
- Against nano without a drafter at c=1: AWQ 2.07x / 3.12x / 2.36x / 1.63x
  (4-bit drafter: 2.31x / 3.47x / 2.66x / 1.78x).
- The 4-bit drafter costs 1.3-3.1% of the tokens per cycle on AWQ (3.39 /
  5.25 / 3.90 / 2.98 against 3.50 / 5.32 / 3.97 / 3.02) and under 1.5% on
  BF16, and saves 1.0 ms of the cycle.
- TTFT at c=1 with the drafter: at most 3% higher (10.1 against 9.8 ms on
  alpaca).
- Higher concurrency, AWQ + DFlash at c=4 / 16 / 64: alpaca 1214 / 2229 /
  2739 (M3: 963 / 2055 / 2739), humaneval 1765 / 2966 / 3946. At c=16 and
  c=64 that is about vLLM AWQ + DFlash's (2084 / 2943 and 2950 / 4204) and
  below nano without a drafter (2567 / 6012): a block of 10 for every
  sequence is 160 or 640 verify rows a step, and most steps at c=64 have a
  prompt row and take the eager path. M6.
- Not done: the drafter's lm_head pass could read an int8 copy on the BF16
  target too (its logits only pick proposals), about 0.4 ms of the 15.45.
- Three runs, 69 minutes of VM, 7.4 CU (419.76 -> 412.37): an A100 is about
  6.4 CU per hour now, not the 3.5 of earlier entries. The VM is stopped.
