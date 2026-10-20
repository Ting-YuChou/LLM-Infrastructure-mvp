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
# verl's uv.lock only resolves for Python >= 3.12; uv downloads it if the host lacks it.
export UV_PYTHON="${UV_PYTHON:-3.12}"

command -v uv >/dev/null 2>&1 || python -m pip install --quiet uv
if [[ ! -d "${VERL_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${VERL_REF}" https://github.com/volcengine/verl.git "${VERL_DIR}"
fi
cd "${VERL_DIR}"
echo "verl $(git describe --tags --always) at ${VERL_DIR}"
# v0.9.1's uv.lock fetches flash-attn from the wheelhouse release
# flash-attention-v2.8.3, which a later torch bump overwrote (now 404). The same
# torch 2.11 build lives on under a per-torch tag; the lock pins no hash.
FLASH_ATTN_RELEASE="${FLASH_ATTN_RELEASE:-flash-attention-v2.8.3-py3.12-torch2.11.0}"
sed -i.orig "s#releases/download/flash-attention-v2.8.3/#releases/download/${FLASH_ATTN_RELEASE}/#g" uv.lock
uv sync --frozen --extra fsdp --extra vllm
uv run --frozen --all-packages --extra vllm --extra fsdp python - <<'EOF'
import torch, transformers, vllm, ray, verl
print({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
       "vllm": vllm.__version__, "ray": ray.__version__, "verl": verl.__version__,
       "gpus": torch.cuda.device_count()})
EOF
