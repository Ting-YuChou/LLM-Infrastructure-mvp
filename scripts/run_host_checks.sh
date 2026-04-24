#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-host-check"
BOOTSTRAP_MARKER="${VENV_DIR}/.bootstrap-complete"
VENV_CREATED=0

pick_python() {
  if command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "python"
}

PYTHON_BIN="${PYTHON_BIN:-$(pick_python)}"
TARGET_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "Using Python: ${PYTHON_BIN} (${TARGET_PYTHON_VERSION})"

if [ -x "${VENV_DIR}/bin/python" ]; then
  CURRENT_VENV_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "${CURRENT_VENV_VERSION}" != "${TARGET_PYTHON_VERSION}" ]; then
    echo "Recreating ${VENV_DIR} because it was built with Python ${CURRENT_VENV_VERSION}"
    rm -rf "${VENV_DIR}"
  fi
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  VENV_CREATED=1
fi

if [ "${VENV_CREATED}" -eq 1 ] || [ ! -f "${BOOTSTRAP_MARKER}" ] || [ "${FORCE_BOOTSTRAP:-0}" = "1" ]; then
  echo "Bootstrapping host-check dependencies"
  "${VENV_DIR}/bin/pip" install --disable-pip-version-check -r "${ROOT_DIR}/requirements.host-check.txt"
  touch "${BOOTSTRAP_MARKER}"
fi

echo
echo "[1/4] gateway import smoke"
PYTHONPATH="${ROOT_DIR}/src" "${VENV_DIR}/bin/python" -c \
  "import api.gateway as g; print('gateway-import-ok')"

echo
echo "[2/4] load test import smoke"
PYTHONPATH="${ROOT_DIR}" "${VENV_DIR}/bin/python" -c \
  "import scripts.load_test as lt; import scripts.compose_smoke_test as st; print('load-and-smoke-import-ok')"

echo
echo "[3/4] gateway route smoke"
JWT_SECRET=host-check-secret AUTH_USERS=local:local PYTHONPATH="${ROOT_DIR}/src" "${VENV_DIR}/bin/python" -c \
  "from api.gateway import APIGateway; app = APIGateway().app; print(sorted(route.path for route in app.routes if route.path in {'/health', '/metrics', '/auth/token', '/v1/completions', '/v1/chat/completions'}))"

echo
echo "[4/4] targeted pytest"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="${ROOT_DIR}/src" \
  "${VENV_DIR}/bin/python" -m pytest -q \
  -p pytest_asyncio.plugin \
  "${ROOT_DIR}/tests/unit/test_vllm_server_config.py" \
  "${ROOT_DIR}/tests/unit/test_metrics.py" \
  "${ROOT_DIR}/tests/integration/test_api_gateway.py"
