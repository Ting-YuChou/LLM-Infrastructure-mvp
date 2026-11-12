# Phase 3 run m4-smoke2-20261112-213151-colab

```
load generator / nano: vllm 0.19.1
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 441: /content/p0/venvs/vllm_spec/bin/python: No such file or directory
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 442: /content/p0/venvs/sglang_spec/bin/python: No such file or directory
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| tests | 1 | 216 |
| spec_check | 0 | 257 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 | 5.01 | 5.02 |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1963 | 6.29 | 5.97 |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 475 | 7.69 | 7.74 |
| awq spec b=16 | 3034 | 12 | 0.40% | 0.250 | 0 | 0 | 4.00 | 1991 | 26.06 | 21.11 |
| awq spec, w4a16 drafter b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 3.94 | 531 | 6.70 | 6.72 |
| awq spec, w4a16 drafter b=16 | 3034 | 10 | 0.33% | 0.250 | 0 | 0 | 4.02 | 2008 | 26.16 | 20.93 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 5.00 | 5.01 | 5.00 | 5.06 |
| awq spec | 7.65 | 7.68 | 7.67 | 7.82 |
| awq spec, w4a16 drafter | 6.67 | 6.70 | 6.66 | 6.81 |

## M4 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.69 ms (met); spec, w4a16 drafter: 6.70 ms (met).

