#!/usr/bin/env bash
set -euo pipefail

cd /Users/chongzhang/PsyBer-Agent

# Prefer the currently activated conda env's python.
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

exec "${PYTHON_BIN}" /Users/chongzhang/PsyBer-Agent/prelearning_agent.py "$@"
