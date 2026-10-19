# CLAUDE.md

End-to-end LLM infrastructure MVP: SFT -> reward model -> PPO/RLHF -> MLflow registry -> vLLM serving behind a FastAPI gateway. The target model is **Llama 3 8B base** (`meta-llama/Meta-Llama-3-8B`), trained on rented multi-GPU machines (RunPod). Treat this as a prototyping stack, not a hardened platform.

## Layout

- `src/training/`: `sft_trainer.py`, `reward_trainer.py`, `rlhf_trainer.py` (PPO), `pipeline.py` (runs all three stages), plus two shared helpers:
  - `chat_format.py`: the single source of the Llama 3 chat template, pad token and stop tokens.
  - `hf_config.py`: maps the YAML configs to `TrainingArguments` (precision, eval strategy, reporting, early stopping).
- `src/serving/vllm_server.py`: OpenAI-compatible server built on vLLM `AsyncLLMEngine`. `src/serving/nano_vllm/` is a small educational engine.
- `src/api/gateway.py`: JWT auth, rate limiting and model routing in front of the backends.
- `src/registry/`, `src/monitoring/`, `src/optimization/`: MLflow client, Prometheus metrics, bitsandbytes quantization. bitsandbytes INT4 is **not** AWQ.
- `config/`: YAML for each stage (`sft_config.yaml`, `reward_config.yaml`, `rlhf_config.yaml`, `training_pipeline.yaml`), serving, and eval gates.
- `tests/unit/`, `tests/integration/`. `tests/unit/tiny_llama.py` builds a tiny real tokenizer and Llama for CPU tests.

## Commands

```bash
python scripts/train_pipeline.py --config config/training_pipeline.yaml --stages all
python scripts/train_sft.py --config config/sft_config.yaml --use-lora
python -m src.training.reward_trainer --config config/reward_config.yaml
python -m src.training.rlhf_trainer --config config/rlhf_config.yaml --prompts <prompts.json>
python -m src.serving.vllm_server --config config/serving_config.yaml
pytest tests/unit/ tests/integration/
bash scripts/run_host_checks.sh   # what the CI "Unit Tests" job runs
```

## Rules that are easy to get wrong

### Dependencies
- Pins move together: torch 2.10.0, transformers 4.57.6, datasets 2.16.0, vLLM 0.19.1, CUDA 12.9.1. If you change one, update `requirements.txt`, `docker/Dockerfile.base` and `docker/Dockerfile.serving` in the same change.
- An unpinned `pip install` pulls in transformers 5.x and datasets 5.x, and those break the trainers.
- transformers 4.57 renamed `evaluation_strategy` to `eval_strategy`. Build `TrainingArguments` through `hf_config.build_training_arguments_kwargs` instead of hand-writing kwargs.

### Precision
- Set precision with `model.precision` (`bf16` / `fp16` / `fp32`) in each stage's YAML.
- `training.fp16` and `training.bf16` are rejected on purpose.
- Use bf16 for Llama 3. A full fine-tune in fp16 fails in GradScaler.
- PPO's `ValueHead` stays fp32 and casts its input itself.

### Chat format
- Every stage renders prompts through `chat_format.py`, and the checkpoint saves the resulting tokenizer and `generation_config.json`. Do not add another template or role-prefix format anywhere.
- After the template is rendered, encode with `add_special_tokens=False`, or BOS appears twice. vLLM gets `{"prompt_token_ids": ...}` for the same reason.
- `<|eot_id|>` is the EOS/stop token. The pad token must be a reserved token such as `<|finetune_right_pad_id|>` and must never equal EOS.
- The chat special tokens in Llama 3 base have untrained embeddings. A LoRA config must include `embed_tokens` and `lm_head` in `modules_to_save`.

### PPO
- Prompts are left-padded for generation.
- `response_mask_from_stops` keeps the first stop token and masks everything after it.
- Old log-probs come from a temperature-scaled forward pass, not from `generate()` scores.
- `position_ids` are derived from the attention mask.

## Tests

- Most training tests stub `transformers`, `datasets` and `peft`. Stubs hid real crashes before, so any change to the trainers must also pass the tests that run the real libraries:
  - `test_training_real_smoke.py`
  - `test_training_chat_format.py`
  - `test_training_hf_config.py`
- Install stubs with `_stub_if_missing` so they only stand in when the real package is absent.
- `datasets` cannot be imported twice in one process (pyarrow registers extension types globally). `test_training_real_smoke.py` caches real modules in `_REAL_MODULES` for this reason.
- Tests run on CPU only. Nothing here has been validated on a GPU or on the real 8B weights, so say that when you report results.

## Working conventions

- Prefer extending existing modules and YAML keys over adding parallel code paths.
- Serving settings come from both YAML and environment variables. Keep `config/serving_config.yaml`, `docker-compose*.yml`, the k8s ConfigMap and the README in sync.
- AWQ serving needs a checkpoint that is already AWQ-quantized. Use llm-compressor (AutoAWQ is archived); `awq_marlin` is the fast kernel.
- Keep checkpoints and outputs under `models/` or `outputs/`, never in git.
- `agents.md` is a local, gitignored note. This file is the shared one.
