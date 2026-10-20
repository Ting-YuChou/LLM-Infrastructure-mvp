#!/usr/bin/env bash
# Build verl's own environment: a checkout at VERL_REF plus `uv sync` against
# verl's uv.lock (torch 2.11 cu130, vLLM 0.24, transformers 5.x). It never
# touches this repo's pinned training environment.
#
#   VERL_DIR=/workspace/verl VERL_REF=v0.9.1 verl_ppo/setup_env.sh
#
# Afterwards run verl as: (cd "$VERL_DIR" && uv run --frozen --all-packages --extra vllm --extra fsdp ...)
set -euo pipefail

VERL_REF="${VERL_REF:-v0.9.1}"
VERL_DIR="${VERL_DIR:?VERL_DIR}"

command -v uv >/dev/null 2>&1 || python -m pip install --quiet uv
if [[ ! -d "${VERL_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${VERL_REF}" https://github.com/volcengine/verl.git "${VERL_DIR}"
fi
cd "${VERL_DIR}"
echo "verl $(git describe --tags --always) at ${VERL_DIR}"
# uv downloads the Python it needs (verl requires >=3.10,<3.13).
uv sync --frozen --extra fsdp --extra vllm
uv run --frozen --all-packages --extra vllm --extra fsdp python - <<'EOF'
import torch, transformers, vllm, ray, verl
print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
       "vllm": vllm.__version__, "ray": ray.__version__, "verl": verl.__version__,
       "gpus": torch.cuda.device_count()})
EOF
