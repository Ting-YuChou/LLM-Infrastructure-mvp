# Phase 3 run m2-smoke-20261112-041533-colab

```
load generator / nano: vllm 0.19.1
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 393: /content/p0/venvs/vllm_spec/bin/python: No such file or directory
/content/p0repo/benchmarks/phase3/run_phase3.sh: line 394: /content/p0/venvs/sglang_spec/bin/python: No such file or directory
study: torch 2.9.0 | transformers 4.57.3
drafter: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat @ d3af30def9601abdd10810aba220d692f0e803f0
```

## Steps

| step | rc | seconds |
|---|---:|---:|
| install_venv | 0 | 341 |
| install_study | 0 | 60 |
| download_model | 0 | 42 |
| download_w4_model | 0 | 38 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 23 |
| workloads | 0 | 29 |
| parity | 1 | 29 |
| serve | 0 | 608 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq | 42 | 163 | 99 | 1223 |
| nano_awq_dflash | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq | 206 |
| nano_awq_dflash | 205 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 4.82 |
| nano_awq_dflash | 4.65 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 9.5 |
| nano_awq_dflash | 9.5 |

**output tok/s over the same server without a drafter**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 0.99x |

### humaneval, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq | 205 |
| nano_awq_dflash | 322 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 4.83 |
| nano_awq_dflash | 3.01 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 14.7 |
| nano_awq_dflash | 15.0 |

**output tok/s over the same server without a drafter**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 1.57x |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq | 205 |
| nano_awq_dflash | 244 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 4.83 |
| nano_awq_dflash | 4.16 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 14.4 |
| nano_awq_dflash | 14.8 |

**output tok/s over the same server without a drafter**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 1.19x |

### long, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq | 183 |
| nano_awq_dflash | 151 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 4.88 |
| nano_awq_dflash | 5.91 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq | 102.6 |
| nano_awq_dflash | 104.2 |

**output tok/s over the same server without a drafter**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 0.82x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1357 | 2.39 | 3.39 |
| nano_awq_dflash | humaneval_c1 | 1080 | 4.36 | 5.36 |
| nano_awq_dflash | gsm8k_c1 | 1151 | 2.98 | 3.98 |
| nano_awq_dflash | long_c1 | 1087 | 2.04 | 3.04 |

## Completions at temperature 0 (texts.py)

Completions with identical text: a server's 16-at-a-time pass against its own one-at-a-time pass (logits depend slightly on the batch size), and a drafter server's passes against the one-at-a-time pass of the same server without a drafter. Last column: output tok/s of the one-at-a-time pass.

| server | c16 = own c1 | c1 = base c1 | c16 = base c1 | c1 tok/s |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 9/32 | 12/32 | 10/32 | 215 |
| nano_awq | 12/32 | - | - | 201 |

## Drafter parity (parity.py)

nano's DFlashDrafter against the checkpoint's own code over 200 blocks (1800 draft positions): top-1 agreement 0.9889 (gate 0.99: NOT met), blocks identical at every position 0.900, largest logit difference 0.250, largest reference top-1 margin where they disagree 0.250. By draft position: 0.995, 0.990, 1.000, 1.000, 0.990, 0.990, 0.980, 0.980, 0.975. The reference accepted 3.65 tokens per cycle.

