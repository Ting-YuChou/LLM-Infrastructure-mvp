# Phase 3 run m4-smoke-20261112-211212-colab

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
| install_venv | 0 | 357 |
| install_study | 0 | 66 |
| download_model | 0 | 43 |
| download_w4_model | 0 | 16 |
| download_drafter | 0 | 6 |
| cuda129 | 0 | 24 |
| workloads | 0 | 28 |
| tests | 1 | 251 |
| spec_check | 1 | 239 |

