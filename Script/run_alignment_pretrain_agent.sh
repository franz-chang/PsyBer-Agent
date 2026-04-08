#!/usr/bin/env sh
set -eu

if (set -o pipefail) >/dev/null 2>&1; then
  set -o pipefail
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/alignment_pretrain_agent.py" "$@"
