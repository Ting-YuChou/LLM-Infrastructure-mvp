# Phase 3 run m4-20261112-214147-colab

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
| tests | 0 | 930 |
| spec_check | 0 | 349 |
| serve | 0 | 1066 |
| summarize | 0 | 0 |

## Prompt tokens each server counts (one prompt per dataset)

| server | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 42 | 163 | 99 | 1223 |
| nano_awq_dflash4 | 42 | 163 | 99 | 1223 |
| nano_dflash | 42 | 163 | 99 | 1223 |
| nano_dflash4 | 42 | 163 | 99 | 1223 |
| nano_awq | 42 | 163 | 99 | 1223 |

Every server tokenizes the prompts the same.

## Serving

### alpaca, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 428 | 1214 | 2229 | 2739 |
| nano_awq_dflash4 | 477 | 1312 | 2260 | 2673 |
| nano_dflash | 216 | - | - | - |
| nano_dflash4 | 228 | - | - | - |
| nano_awq | 207 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.14 | 2.90 | 6.14 | 21.51 |
| nano_awq_dflash4 | 1.93 | 2.62 | 6.04 | 22.47 |
| nano_dflash | 4.50 | - | - | - |
| nano_dflash4 | 4.23 | - | - | - |
| nano_awq | 4.80 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 10.1 | 27.2 | 50.8 | 165.6 |
| nano_awq_dflash4 | 9.9 | 25.3 | 49.9 | 168.3 |
| nano_dflash | 18.4 | - | - | - |
| nano_dflash4 | 18.2 | - | - | - |
| nano_awq | 9.8 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 2.07x | - | - | - |
| nano_awq_dflash4 | 2.31x | - | - | - |

### humaneval, temperature 0

**output tok/s**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 641 | 1765 | 2966 | 3946 |
| nano_awq_dflash4 | 714 | 1901 | 2938 | 3870 |
| nano_dflash | 333 | - | - | - |
| nano_dflash4 | 357 | - | - | - |
| nano_awq | 206 | - | - | - |

**TPOT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 1.49 | 2.00 | 4.81 | 14.08 |
| nano_awq_dflash4 | 1.34 | 1.90 | 4.79 | 14.21 |
| nano_dflash | 2.85 | - | - | - |
| nano_dflash4 | 2.59 | - | - | - |
| nano_awq | 4.81 | - | - | - |

**TTFT p50 (ms)**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 17.0 | 34.8 | 75.6 | 209.7 |
| nano_awq_dflash4 | 17.1 | 33.2 | 70.3 | 200.0 |
| nano_dflash | 22.6 | - | - | - |
| nano_dflash4 | 22.4 | - | - | - |
| nano_awq | 16.6 | - | - | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=4 | c=16 | c=64 |
|---|---:|---:|---:|---:|
| nano_awq_dflash | 3.12x | - | - | - |
| nano_awq_dflash4 | 3.47x | - | - | - |

### gsm8k, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 484 | 2408 |
| nano_awq_dflash4 | 547 | 2375 |
| nano_dflash | 254 | - |
| nano_dflash4 | 272 | - |
| nano_awq | 205 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.02 | 5.75 |
| nano_awq_dflash4 | 1.75 | 5.81 |
| nano_dflash | 3.96 | - |
| nano_dflash4 | 3.69 | - |
| nano_awq | 4.81 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 15.0 | 59.8 |
| nano_awq_dflash4 | 15.1 | 60.4 |
| nano_dflash | 22.1 | - |
| nano_dflash4 | 21.8 | - |
| nano_awq | 15.7 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.36x | - |
| nano_awq_dflash4 | 2.66x | - |

### long, temperature 0

**output tok/s**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 289 | 795 |
| nano_awq_dflash4 | 316 | 798 |
| nano_dflash | 170 | - |
| nano_dflash4 | 180 | - |
| nano_awq | 177 | - |

**TPOT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 2.63 | 16.14 |
| nano_awq_dflash4 | 2.36 | 16.42 |
| nano_dflash | 5.23 | - |
| nano_dflash4 | 4.85 | - |
| nano_awq | 4.86 | - |

**TTFT p50 (ms)**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 106.8 | 256.1 |
| nano_awq_dflash4 | 106.4 | 261.0 |
| nano_dflash | 93.2 | - |
| nano_dflash4 | 94.0 | - |
| nano_awq | 104.8 | - |

**output tok/s over the same server without a drafter**

| server | c=1 | c=16 |
|---|---:|---:|
| nano_awq_dflash | 1.63x | - |
| nano_awq_dflash4 | 1.78x | - |

## Drafts as the servers count them

| server | run | drafts | accepted per draft | tokens per cycle |
|---|---:|---:|---:|---:|
| nano_awq_dflash | alpaca_c1 | 1309 | 2.50 | 3.50 |
| nano_awq_dflash | alpaca_c4 | 1345 | 2.59 | 3.59 |
| nano_awq_dflash | alpaca_c16 | 3113 | 2.70 | 3.70 |
| nano_awq_dflash | alpaca_c64 | 14678 | 2.50 | 3.50 |
| nano_awq_dflash | humaneval_c1 | 1093 | 4.32 | 5.32 |
| nano_awq_dflash | humaneval_c4 | 1019 | 4.54 | 5.54 |
| nano_awq_dflash | humaneval_c16 | 2916 | 4.57 | 5.57 |
| nano_awq_dflash | humaneval_c64 | 7312 | 4.60 | 5.60 |
| nano_awq_dflash | gsm8k_c1 | 1142 | 2.97 | 3.97 |
| nano_awq_dflash | gsm8k_c16 | 2529 | 3.16 | 4.16 |
| nano_awq_dflash | long_c1 | 1108 | 2.02 | 3.02 |
| nano_awq_dflash | long_c16 | 2898 | 2.08 | 3.08 |
| nano_awq_dflash4 | alpaca_c1 | 1353 | 2.39 | 3.39 |
| nano_awq_dflash4 | alpaca_c4 | 1363 | 2.59 | 3.59 |
| nano_awq_dflash4 | alpaca_c16 | 3088 | 2.69 | 3.69 |
| nano_awq_dflash4 | alpaca_c64 | 14648 | 2.49 | 3.49 |
| nano_awq_dflash4 | humaneval_c1 | 1108 | 4.25 | 5.25 |
| nano_awq_dflash4 | humaneval_c4 | 1025 | 4.53 | 5.53 |
| nano_awq_dflash4 | humaneval_c16 | 2931 | 4.54 | 5.54 |
| nano_awq_dflash4 | humaneval_c64 | 7355 | 4.56 | 5.56 |
| nano_awq_dflash4 | gsm8k_c1 | 1162 | 2.90 | 3.90 |
| nano_awq_dflash4 | gsm8k_c16 | 2534 | 3.13 | 4.13 |
| nano_awq_dflash4 | long_c1 | 1123 | 1.98 | 2.98 |
| nano_awq_dflash4 | long_c16 | 2958 | 2.03 | 3.03 |
| nano_dflash | alpaca_c1 | 1353 | 2.45 | 3.45 |
| nano_dflash | humaneval_c1 | 1022 | 4.42 | 5.42 |
| nano_dflash | gsm8k_c1 | 1050 | 3.08 | 4.08 |
| nano_dflash | long_c1 | 1143 | 2.02 | 3.02 |
| nano_dflash4 | alpaca_c1 | 1372 | 2.40 | 3.40 |
| nano_dflash4 | humaneval_c1 | 1023 | 4.41 | 5.41 |
| nano_dflash4 | gsm8k_c1 | 1053 | 3.07 | 4.07 |
| nano_dflash4 | long_c1 | 1143 | 2.02 | 3.02 |

## Emitted tokens against the target's argmax (spec_check.py)

Every generated token against the argmax of one teacher-forced pass of the target over the finished sequence; the margin is the top logit minus the emitted token's. ms/step: the wall time of an engine step with no prompt rows (a decode step, or a whole speculative cycle).

| target, mode, batch | tokens | not argmax | share | largest margin | margins > 0.5 | margins > 1 | tokens/cycle | tok/s | ms/step median | ms/step mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq plain b=1 | 3035 | 13 | 0.43% | 0.250 | 0 | 0 | - | 193 | 5.00 | 5.01 |
| awq plain b=16 | 3037 | 10 | 0.33% | 0.250 | 0 | 0 | - | 1966 | 6.27 | 5.96 |
| awq spec b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 4.00 | 473 | 7.75 | 7.77 |
| awq spec b=16 | 3034 | 12 | 0.40% | 0.250 | 0 | 0 | 4.00 | 1980 | 26.22 | 21.23 |
| awq spec, w4a16 drafter b=1 | 3003 | 15 | 0.50% | 0.250 | 0 | 0 | 3.94 | 532 | 6.68 | 6.70 |
| awq spec, w4a16 drafter b=16 | 3034 | 10 | 0.33% | 0.250 | 0 | 0 | 4.02 | 2014 | 26.08 | 20.87 |
| bf16 plain b=1 | 2940 | 8 | 0.27% | 0.250 | 0 | 0 | - | 81 | 12.29 | 12.29 |
| bf16 plain b=16 | 2974 | 5 | 0.17% | 0.250 | 0 | 0 | - | 864 | 13.67 | 13.38 |
| bf16 spec b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 247 | 15.45 | 15.45 |
| bf16 spec b=16 | 2917 | 5 | 0.17% | 0.250 | 0 | 0 | 4.06 | 1690 | 26.60 | 23.26 |
| bf16 spec, w4a16 drafter b=1 | 3009 | 5 | 0.17% | 0.250 | 0 | 0 | 3.98 | 263 | 14.50 | 14.49 |
| bf16 spec, w4a16 drafter b=16 | 2917 | 6 | 0.21% | 0.250 | 0 | 0 | 4.03 | 1699 | 26.63 | 22.82 |

Median ms/step at batch 1 per dataset:

| target, mode | alpaca | humaneval | gsm8k | long |
|---|---:|---:|---:|---:|
| awq plain | 4.99 | 4.99 | 4.99 | 5.06 |
| awq spec | 7.75 | 7.74 | 7.71 | 7.83 |
| awq spec, w4a16 drafter | 6.68 | 6.67 | 6.65 | 6.78 |
| bf16 plain | 12.22 | 12.27 | 12.31 | 12.34 |
| bf16 spec | 15.43 | 15.44 | 15.40 | 15.54 |
| bf16 spec, w4a16 drafter | 14.45 | 14.48 | 14.44 | 14.57 |

## M4 gates

G3 (AWQ target, batch 1, a cycle within 8.0 ms, median): spec: 7.75 ms (met); spec, w4a16 drafter: 6.68 ms (met).

G4: output tok/s at c=1, temperature 0, against the fastest DFlash server on the same weights (baselines from m0-20261111-074458-colab):

| dataset / server | nano tok/s | best baseline | ratio | G4 |
|---|---:|---:|---:|---:|
| alpaca / nano_awq_dflash | 428 | vllmw4_dflash 380 | 1.13x | met |
| alpaca / nano_awq_dflash4 | 477 | vllmw4_dflash 380 | 1.26x | met |
| alpaca / nano_dflash | 216 | vllm_dflash 220 | 0.98x | NOT met |
| alpaca / nano_dflash4 | 228 | vllm_dflash 220 | 1.03x | met |
| humaneval / nano_awq_dflash | 641 | vllmw4_dflash 589 | 1.09x | met |
| humaneval / nano_awq_dflash4 | 714 | vllmw4_dflash 589 | 1.21x | met |
| humaneval / nano_dflash | 333 | vllm_dflash 322 | 1.03x | met |
| humaneval / nano_dflash4 | 357 | vllm_dflash 322 | 1.11x | met |
| gsm8k / nano_awq_dflash | 484 | vllmw4_dflash 434 | 1.11x | met |
| gsm8k / nano_awq_dflash4 | 547 | vllmw4_dflash 434 | 1.26x | met |
| gsm8k / nano_dflash | 254 | vllm_dflash 256 | 0.99x | NOT met |
| gsm8k / nano_dflash4 | 272 | vllm_dflash 256 | 1.06x | met |
| long / nano_awq_dflash | 289 | vllmw4_dflash 245 | 1.18x | met |
| long / nano_awq_dflash4 | 316 | vllmw4_dflash 245 | 1.29x | met |
| long / nano_dflash | 170 | vllm_dflash 161 | 1.06x | met |
| long / nano_dflash4 | 180 | vllm_dflash 161 | 1.12x | met |

