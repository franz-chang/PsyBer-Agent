#!/usr/bin/env bash
set -euo pipefail

cd /Users/chongzhang/PsyBer-Agent

# Prefer dedicated runtime env first, then activated env, then system python3.
if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

exec "${PYTHON_BIN}" launch_websim_batch.py \
  --total-tasks 10000 \
  --max-concurrency 10 \
  --launch-rate 15 \
  --headless \
  --print-task-pid \
  --track-override 20 \
  --task-timeout-sec 1800 \
  --profile-selection-mode random \
  "$@"
