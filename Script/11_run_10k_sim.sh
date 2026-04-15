#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

# Silence macOS malloc stack logging warning in task logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

# Prefer dedicated runtime env first, then activated env, then system python3.
if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -x "/root/miniconda3/bin/python3" ]; then
  PYTHON_BIN="/root/miniconda3/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

API_POOL_FILE="${PROJECT_ROOT}/settings/api_pool.yaml"
if [ ! -f "${API_POOL_FILE}" ]; then
  API_POOL_FILE="${PROJECT_ROOT}/Settings/api_pool.yaml"
fi
API_POOL_ARGS=(--api-pool-file "${API_POOL_FILE}" --api-pool-required)

# Fail fast on missing core deps to avoid a full batch of failed tasks.
if ! "${PYTHON_BIN}" - <<'PY'
missing = []
for name in ("camel.toolkits", "openai", "playwright", "yaml"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    print("MISSING:" + ",".join(missing))
    raise SystemExit(1)
PY
then
  echo "[ERROR] Missing python dependencies for batch launcher." >&2
  echo "Install with: ${PYTHON_BIN} -m pip install camel-ai openai playwright pyyaml" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/launch_websim_batch.py" \
  --total-tasks 10000 \
  --max-concurrency 10 \
  --launch-rate 15 \
  --headless \
  --print-task-pid \
  "${API_POOL_ARGS[@]}" \
  --track-override 20 \
  --task-timeout-sec 1800 \
  --profile-selection-mode random \
  "$@"
