#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

# Prefer dedicated runtime env first, then activated env, then system python3.
if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

# Check Qwen API key source before launch (env or inline code value).
INLINE_KEY="$(
  "${PYTHON_BIN}" -c "import re; from pathlib import Path; text=Path('${PROJECT_ROOT}/web_sim_camel_agent.py').read_text(encoding='utf-8'); m=re.search(r'^\\s*QWEN_API_KEY\\s*=\\s*[\\\"\\'](.*)[\\\"\\']\\s*$', text, flags=re.MULTILINE); print((m.group(1).strip() if m else ''))"
)"

if [ -z "${QWEN_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -z "${INLINE_KEY}" ]; then
  echo "[ERROR] Missing Qwen API key." >&2
  echo "Set QWEN_API_KEY or DASHSCOPE_API_KEY, or fill QWEN_API_KEY in web_sim_camel_agent.py." >&2
  exit 1
fi

# Ensure Playwright Chromium exists before launching the agent.
if ! "${PYTHON_BIN}" -c "import os,sys; from playwright.sync_api import sync_playwright; p=sync_playwright().start(); path=p.chromium.executable_path; p.stop(); sys.exit(0 if os.path.exists(path) else 1)"; then
  echo "[INFO] Playwright Chromium not found. Installing..."
  "${PYTHON_BIN}" -m playwright install chromium
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/web_sim_camel_agent.py" "$@"
