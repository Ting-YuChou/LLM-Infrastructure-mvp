# Phase 3 run m0-smoke-20261111-065251-colab

```
load generator / nano: vllm 0.19.1
vllm baseline: 0.33.0 | torch 2.14.0
sglang baseline: 0.5.23 | torch 2.14.1
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 51 |
| install_study | 0 | 16 |
| install_vllm_spec | 0 | 45 |
| install_sglang_spec | 0 | 31 |
| download_model | 0 | 44 |
| download_w4_model | 0 | 16 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 40 |
| workloads | 0 | 45 |
| accept | 0 | 293 |
| serve | 1 | 978 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Acceptance study (accept_study.py)

Tokens per cycle (accepted drafts + the bonus token), `drafter-draw-block` per column.

**bf16 target, greedy**

| dataset | bf16-greedy-b10 | bf16-greedy-b6 | bf16-greedy-b4 | rtn4-greedy-b10 |
|---|---:|---:|---:|---:|
| alpaca | 3.98 | 3.65 | 3.09 | 3.86 |
| humaneval | 4.91 | 4.19 | 3.39 | 4.83 |
| gsm8k | 4.03 | 3.70 | 3.00 | 3.98 |
| long | 3.09 | 3.20 | 2.62 | 3.06 |

**bf16 target, sampled**

| dataset | bf16-greedy-b10 | bf16-coupled-b10 | bf16-independent-b10 | bf16-coupled-b6 | bf16-coupled-b4 | rtn4-coupled-b10 |
|---|---:|---:|---:|---:|---:|---:|
| alpaca | 4.29 | 4.26 | 4.05 | 3.76 | 3.09 | 4.03 |
| humaneval | 5.08 | 5.13 | 4.92 | 4.27 | 3.32 | 5.00 |
| gsm8k | 3.46 | 3.52 | 3.38 | 3.24 | 2.83 | 3.48 |
| long | 2.89 | 3.15 | 2.80 | 2.92 | 2.72 | 3.08 |

**awq target, greedy**

| dataset | bf16-greedy-b10 | bf16-greedy-b6 | bf16-greedy-b4 | rtn4-greedy-b10 |
|---|---:|---:|---:|---:|
| alpaca | 3.84 | 3.58 | 2.95 | 3.92 |
| humaneval | 5.18 | 4.29 | 3.33 | 5.05 |
| gsm8k | 3.60 | 3.29 | 2.80 | 3.48 |
| long | 2.88 | 2.82 | 2.66 | 2.82 |

**awq target, sampled**

| dataset | bf16-greedy-b10 | bf16-coupled-b10 | bf16-independent-b10 | bf16-coupled-b6 | bf16-coupled-b4 | rtn4-coupled-b10 |
|---|---:|---:|---:|---:|---:|---:|
| alpaca | 3.79 | 3.64 | 3.60 | 3.44 | 2.94 | 3.62 |
| humaneval | 4.45 | 4.33 | 4.23 | 3.74 | 3.14 | 4.31 |
| gsm8k | 3.84 | 3.99 | 3.69 | 3.60 | 3.01 | 3.94 |
| long | 2.57 | 2.79 | 2.44 | 2.62 | 2.40 | 2.57 |

Survival by depth (the share of cycles that accept at least that many drafts), bf16 drafter, greedy drafts, block 10; the last columns: how often the drafter's second choice was the target's token at the first rejected depth, and how many prompts' replay matched the live loop.

| stream | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 2nd | replay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 greedy alpaca | 0.82 | 0.67 | 0.48 | 0.35 | 0.25 | 0.17 | 0.13 | 0.09 | 0.06 | 0.48 | 4/4 |
| bf16 greedy humaneval | 0.90 | 0.77 | 0.64 | 0.51 | 0.40 | 0.29 | 0.20 | 0.14 | 0.11 | 0.53 | 4/4 |
| bf16 greedy gsm8k | 0.86 | 0.66 | 0.51 | 0.35 | 0.24 | 0.17 | 0.13 | 0.09 | 0.06 | 0.41 | 4/4 |
| bf16 greedy long | 0.73 | 0.53 | 0.36 | 0.24 | 0.16 | 0.08 | 0.02 | 0.00 | 0.00 | 0.34 | 2/2 |
| bf16 sampled alpaca | 0.85 | 0.68 | 0.57 | 0.41 | 0.28 | 0.24 | 0.17 | 0.10 | 0.06 | 0.38 | 4/4 |
| bf16 sampled humaneval | 0.85 | 0.72 | 0.63 | 0.54 | 0.45 | 0.35 | 0.26 | 0.19 | 0.13 | 0.52 | 4/4 |
| bf16 sampled gsm8k | 0.76 | 0.57 | 0.39 | 0.30 | 0.23 | 0.11 | 0.08 | 0.04 | 0.01 | 0.40 | 4/4 |
| bf16 sampled long | 0.74 | 0.55 | 0.30 | 0.17 | 0.10 | 0.03 | 0.01 | 0.01 | 0.01 | 0.34 | 2/2 |
| awq greedy alpaca | 0.83 | 0.62 | 0.44 | 0.33 | 0.26 | 0.18 | 0.11 | 0.07 | 0.06 | 0.42 | 4/4 |
| awq greedy humaneval | 0.89 | 0.79 | 0.68 | 0.55 | 0.46 | 0.33 | 0.25 | 0.17 | 0.10 | 0.48 | 4/4 |
| awq greedy gsm8k | 0.76 | 0.56 | 0.43 | 0.31 | 0.22 | 0.14 | 0.10 | 0.07 | 0.05 | 0.41 | 4/4 |
| awq greedy long | 0.75 | 0.52 | 0.28 | 0.19 | 0.10 | 0.04 | 0.01 | 0.01 | 0.00 | 0.41 | 2/2 |
| awq sampled alpaca | 0.81 | 0.58 | 0.47 | 0.33 | 0.25 | 0.17 | 0.11 | 0.07 | 0.04 | 0.45 | 4/4 |
| awq sampled humaneval | 0.84 | 0.66 | 0.54 | 0.43 | 0.34 | 0.21 | 0.19 | 0.15 | 0.11 | 0.47 | 4/4 |
| awq sampled gsm8k | 0.78 | 0.60 | 0.47 | 0.35 | 0.27 | 0.17 | 0.11 | 0.08 | 0.05 | 0.39 | 4/4 |
| awq sampled long | 0.68 | 0.39 | 0.22 | 0.10 | 0.08 | 0.07 | 0.04 | 0.02 | 0.00 | 0.41 | 2/2 |

