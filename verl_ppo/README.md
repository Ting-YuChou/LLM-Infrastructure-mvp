# PPO with verl (proof of concept)

[verl](https://github.com/volcengine/verl) runs PPO with Ray: vLLM generates
the rollouts, and the actor and critic train under FSDP. Our own PPO
(`src/training/rlhf_trainer.py`) generates with HF `generate()` and supports
only LoRA + DDP. This directory runs verl as an alternative PPO stage on the
SFT and reward checkpoints our pipeline already produces, so the two can be
compared on speed.

## Separate environment

verl v0.9.1 needs `transformers>=5.5.3`, `numpy>=2` and `pyarrow>=19`. Our
trainers pin transformers 4.57.6, numpy 1.26.4 and pyarrow 14 (see
`requirements.txt`). So verl runs in its own uv environment, locked by verl's
`uv.lock` at a fixed tag (torch 2.11 cu130, vLLM 0.24), and nothing here
changes our pins. The handoff between the two environments is plain Hugging
Face checkpoints.

```bash
VERL_DIR=/workspace/verl VERL_REF=v0.9.1 verl_ppo/setup_env.sh
python verl_ppo/prepare_prompts.py --input data/prompts/rl_prompts.json --output-dir data/verl
MODE=lora SFT_CKPT=models/sft/final RM_CKPT=models/reward/final DATA_DIR=data/verl \
  OUTPUT_DIR=outputs/verl_lora VERL_DIR=/workspace/verl verl_ppo/run_ppo.sh
```

The host needs a CUDA 13 driver (cu130 torch).

## How our checkpoints plug in

| verl role | Checkpoint | Notes |
|---|---|---|
| actor / reference / rollout | SFT `final/` | verl applies the Llama 3 chat template saved with its tokenizer, and stops on the `<|eot_id|>` eos. |
| reward model | reward `final/` | `LlamaForSequenceClassification`, served by vLLM's `classify` endpoint (raw logit). verl re-templates `prompt + response` with the RM tokenizer, which gives the same tokens the RM was trained on (`test_verl_reward_model_input_matches_reward_training_tokens`). |
| critic | reward `final/` | Loaded as `AutoModelForTokenClassification`; its `score` head starts the value head. |

`MODE=lora` uses the same adapter as `config/rlhf_config.yaml` (r=16,
alpha=32, all linear projections). `MODE=full` trains every parameter, with
FSDP param and optimizer offload. `run_ppo.sh` keeps all the overrides in one
place; `VERL_DRY_RUN=1` prints them without running.

## GPU comparison

`scripts/runpod_gpu_smoke.sh` runs both trainers on one pod with the same
prompts per step, prompt and response lengths, and step count:

```bash
SMOKE_PRESET=8b_ppo_bench SMOKE_STAGES="sft reward rlhf verl_setup verl_lora verl_full"
```

The resulting `summary.json` contains a `ppo_comparison` table: per-step
generation, scoring and update seconds, plus generated tokens per second, for
`rlhf`, `verl_lora` and `verl_full`. `verl_report.py` also reloads each
exported actor and generates once as a check.

### Result: Llama 3 8B, 4x A100 SXM 80GB (2026-10-21)

Fake data, 6 steps of 64 prompts x up to 512 response tokens; means over steps
2-6, checkpoint saves excluded. All three passed and their exported actors
reload and generate.

| | `rlhf` (ours) | `verl_lora` | `verl_full` |
|---|---|---|---|
| generation s/step | 32.6 | 8.9 | 8.4 |
| scoring s/step | 2.4 | 10.0 | 12.0 |
| update s/step | 7.3 | 55.9 (34 of it weight sync) | 28.0 |
| step s | 42.4 | 74.8 | 48.4 |
| generated tokens/s | 767 | 3,319 | 3,615 |
| mean response tokens | 391 | 460 | 476 |
| peak GPU memory | 58.5 GB | 62.3 GB | 54.5 GB |

vLLM generates 4-5x more tokens per second than our HF `generate()`. The
whole step is not faster because the work differs: our value function is a
head on the policy, while verl trains a separate full 8B critic (from the RM)
with CPU offload. In LoRA mode, syncing base weights to vLLM layer by layer
costs 34 s per step.

### verl v0.9.1 workarounds in these scripts

- `UV_PYTHON=3.12`: the lockfile only resolves for Python >= 3.12.
- `setup_env.sh` points flash-attn at the wheelhouse release
  `flash-attention-v2.8.3-py3.12-torch2.11.0`; the lock's URL now 404s.
- Run `.venv/bin/python`, not `uv run`: under `uv run`, Ray starts each worker
  through `uv run` too, and LoRA workers died with SIGBUS.
- LoRA uses `trainer.use_v1=False`: the V1 trainer leaks the reference pass's
  `no_lora_adapter` flag into the non-LoRA critic's value pass.
- The critic is offloaded in both modes and LoRA sets
  `rollout.layered_summon=True`; without them the LoRA weight sync ran out of
  GPU memory.
- verl's `timing_s/step` includes `save_checkpoint`; `verl_report.py`
  subtracts it.
