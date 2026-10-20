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
