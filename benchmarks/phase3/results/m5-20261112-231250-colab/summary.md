# Phase 3 run m5-20261112-231250-colab

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
| install_venv | 0 | 58 |
| install_study | 0 | 15 |
| download_model | 0 | 44 |
| download_w4_model | 0 | 17 |
| download_drafter | 0 | 7 |
| cuda129 | 0 | 43 |
| workloads | 0 | 42 |
| tests | 0 | 979 |
| spec_check | 0 | 251 |
| serve | 0 | 1010 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano_awq_dflashg | 42 | 163 | 99 | 1223 |
| nano_awq_dflash4 | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |
| nano_dflashg | 42 | 163 | 99 | 1223 |
| nano_dflash4 | 42 | 163 | 99 | 1223 |
| nano_awq | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 423 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 2.17 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 10.1 |

### humaneval, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 637 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 1.50 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 17.1 |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 483 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 2.02 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 15.2 |

### long, temperature 0

**output tok/s**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 288 |

**TPOT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 2.63 |

**TTFT p50 (ms)**

| server | c=1 |
|---|---:|
| nano_awq_dflash | 106.0 |

### alpaca, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 383 | 2097 |
| nano_awq_dflashg | 419 | 2141 |
| nano_awq_dflash4 | 448 | 2057 |
| nano_dflash | 210 | 1809 |
| nano_dflashg | 216 | 1770 |
| nano_dflash4 | 219 | 1735 |
| nano_awq | 206 | 2500 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.34 | 6.74 |
| nano_awq_dflashg | 2.24 | 6.56 |
| nano_awq_dflash4 | 2.16 | 6.90 |
| nano_dflash | 4.56 | 7.59 |
| nano_dflashg | 4.49 | 7.20 |
| nano_dflash4 | 4.38 | 7.65 |
| nano_awq | 4.81 | 5.38 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 9.4 | 52.2 |
| nano_awq_dflashg | 9.8 | 49.9 |
| nano_awq_dflash4 | 9.9 | 50.5 |
| nano_dflash | 18.4 | 56.4 |
| nano_dflashg | 18.6 | 56.3 |
| nano_dflash4 | 18.5 | 56.0 |
| nano_awq | 10.1 | 22.9 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.86x | 0.84x |
| nano_awq_dflashg | 2.03x | 0.86x |
| nano_awq_dflash4 | 2.17x | 0.82x |

### humaneval, temperature 0.7, top-p 0.9

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 602 | 2863 |
| nano_awq_dflashg | 616 | 2934 |
| nano_awq_dflash4 | 684 | 2875 |
| nano_dflash | 315 | 2591 |
| nano_dflashg | 321 | 2605 |
| nano_dflash4 | 332 | 2594 |
| nano_awq | 205 | 2518 |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.58 | 4.86 |
| nano_awq_dflashg | 1.53 | 4.86 |
| nano_awq_dflash4 | 1.37 | 4.89 |
| nano_dflash | 3.05 | 5.42 |
| nano_dflashg | 3.06 | 5.09 |
| nano_dflash4 | 2.88 | 5.41 |
| nano_awq | 4.83 | 5.76 |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 9.9 | 70.4 |
| nano_awq_dflashg | 17.3 | 75.2 |
| nano_awq_dflash4 | 17.2 | 72.7 |
| nano_dflash | 22.7 | 73.5 |
| nano_dflashg | 22.7 | 73.0 |
| nano_dflash4 | 23.0 | 73.6 |
| nano_awq | 17.0 | 154.1 |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.94x | 1.14x |
| nano_awq_dflashg | 3.01x | 1.17x |
| nano_awq_dflash4 | 3.34x | 1.14x |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1309 | 2.50 | 3.50 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflash | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflash | alpaca_c1s | 1447 | 2.16 | 3.16 |
| nano_awq_dflash | alpaca_c16s | 3285 | 2.52 | 3.52 |
| nano_awq_dflash | humaneval_c1s | 1190 | 3.95 | 4.95 |
| nano_awq_dflash | humaneval_c16s | 3043 | 4.36 | 5.36 |
| nano_awq_dflashg | alpaca_c1s | 1354 | 2.40 | 3.40 |
| nano_awq_dflashg | alpaca_c16s | 3277 | 2.53 | 3.53 |
| nano_awq_dflashg | humaneval_c1s | 1121 | 4.14 | 5.14 |
| nano_awq_dflashg | humaneval_c16s | 2954 | 4.49 | 5.49 |
| nano_awq_dflash4 | alpaca_c1s | 1436 | 2.20 | 3.20 |
| nano_awq_dflash4 | alpaca_c16s | 3382 | 2.44 | 3.44 |
| nano_awq_dflash4 | humaneval_c1s | 1141 | 4.04 | 5.04 |
| nano_awq_dflash4 | humaneval_c16s | 3065 | 4.34 | 5.34 |
| nano_dflash | alpaca_c1s | 1369 | 2.35 | 3.35 |
| nano_dflash | alpaca_c16s | 3272 | 2.54 | 3.54 |
| nano_dflash | humaneval_c1s | 1119 | 4.12 | 5.12 |
| nano_dflash | humaneval_c16s | 3062 | 4.25 | 5.25 |
| nano_dflashg | alpaca_c1s | 1325 | 2.47 | 3.47 |
| nano_dflashg | alpaca_c16s | 3313 | 2.55 | 3.55 |
| nano_dflashg | humaneval_c1s | 1101 | 4.21 | 5.21 |
| nano_dflashg | humaneval_c16s | 2987 | 4.36 | 5.36 |
| nano_dflash4 | alpaca_c1s | 1392 | 2.30 | 3.30 |
| nano_dflash4 | alpaca_c16s | 3286 | 2.46 | 3.46 |
| nano_dflash4 | humaneval_c1s | 1126 | 4.09 | 5.09 |
| nano_dflash4 | humaneval_c16s | 3080 | 4.25 | 5.25 |

## Coupled sampling (temperature 0.7, top-p 0.9)

Tokens per cycle and output tok/s with the drafts drawn with the target's noise (coupled) and with the drafter's argmax as drafts, same seed: the emitted tokens are the same requests' same streams up to the logits' batch noise.

| server, run | coupled tokens/cycle | argmax tokens/cycle | ratio | coupled tok/s | argmax tok/s |
|---|---:|---:|---:|---:|---:|
| nano_awq_dflash alpaca_c1s | 3.16 | 3.40 | 0.930x | 383 | 419 |
| nano_awq_dflash alpaca_c16s | 3.52 | 3.53 | 0.997x | 2097 | 2141 |
| nano_awq_dflash humaneval_c1s | 4.95 | 5.14 | 0.964x | 602 | 616 |
| nano_awq_dflash humaneval_c16s | 5.36 | 5.49 | 0.977x | 2863 | 2934 |
| nano_dflash alpaca_c1s | 3.35 | 3.47 | 0.968x | 210 | 216 |
| nano_dflash alpaca_c16s | 3.54 | 3.55 | 0.998x | 1809 | 1770 |
| nano_dflash humaneval_c1s | 5.12 | 5.21 | 0.983x | 315 | 321 |
| nano_dflash humaneval_c16s | 5.25 | 5.36 | 0.980x | 2591 | 2605 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 | 5.00 | 5.02 |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1966 | 6.30 | 5.96 |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 475 | 7.74 | 7.74 |
| awq spec b=16 | 3034 | 12 | 0.40% | 0.250 | 0 | 0 | 4.00 | 1972 | 26.26 | 21.31 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 4.99 | 5.00 | 5.00 | 5.07 |
| awq spec | 7.71 | 7.73 | 7.68 | 7.78 |

## M4 and M5 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.74 ms (met).

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 423 | vllmw4_dflash 380 | 1.11x | met |
| humaneval / nano_awq_dflash | 637 | vllmw4_dflash 589 | 1.08x | met |
| gsm8k / nano_awq_dflash | 483 | vllmw4_dflash 434 | 1.11x | met |
| long / nano_awq_dflash | 288 | vllmw4_dflash 245 | 1.18x | met |

M5: output tok/s at c=1, temperature 0.7, top-p 0.9, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | above |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 383 | vllmw4_dflash 371 | 1.03x | met |
| alpaca / nano_awq_dflash4 | 448 | vllmw4_dflash 371 | 1.21x | met |
| alpaca / nano_awq_dflashg | 419 | vllmw4_dflash 371 | 1.13x | met |
| alpaca / nano_dflash | 210 | vllm_dflash 208 | 1.01x | met |
| alpaca / nano_dflash4 | 219 | vllm_dflash 208 | 1.06x | met |
| alpaca / nano_dflashg | 216 | vllm_dflash 208 | 1.04x | met |
| humaneval / nano_awq_dflash | 602 | vllmw4_dflash 556 | 1.08x | met |
| humaneval / nano_awq_dflash4 | 684 | vllmw4_dflash 556 | 1.23x | met |
| humaneval / nano_awq_dflashg | 616 | vllmw4_dflash 556 | 1.11x | met |
| humaneval / nano_dflash | 315 | vllm_dflash 310 | 1.02x | met |
| humaneval / nano_dflash4 | 332 | vllm_dflash 310 | 1.07x | met |
| humaneval / nano_dflashg | 321 | vllm_dflash 310 | 1.04x | met |

