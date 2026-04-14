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

# Stage-03 inference should default to trained policy network only.
# Set PSY_STRICT_FAST_INFERENCE=0 to allow Qwen fallback.
STRICT_FAST_INFERENCE_RAW="${PSY_STRICT_FAST_INFERENCE:-1}"
STRICT_FAST_INFERENCE_NORM="$(printf '%s' "${STRICT_FAST_INFERENCE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${STRICT_FAST_INFERENCE_NORM}" in
  0|false|no|off)
    STRICT_FAST_INFERENCE=0
    ;;
  *)
    STRICT_FAST_INFERENCE=1
    ;;
esac

user_disable_fast_inference=0
for arg in "$@"; do
  case "$arg" in
    --disable-fast-inference)
      user_disable_fast_inference=1
      ;;
  esac
done

EXTRA_ARGS=()
if [ "${STRICT_FAST_INFERENCE}" -eq 1 ]; then
  if [ "${user_disable_fast_inference}" -eq 1 ]; then
    echo "[ERROR] Strict fast inference is enabled, but '--disable-fast-inference' was passed." >&2
    echo "Unset PSY_STRICT_FAST_INFERENCE or remove '--disable-fast-inference'." >&2
    exit 1
  fi

  MISSING_FAST_POLICY="$(
    PROJECT_ROOT_FOR_POLICY_CHECK="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(
    os.environ.get("PROJECT_ROOT_FOR_POLICY_CHECK", "/Users/chongzhang/PsyBer-Agent")
)
task_file = project_root / "Settings" / "task.yaml"
policy_paths = {
    "ml1m": Path("/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/Embeddings/fast_inference_policy.pt"),
    "amazon_mi": Path("/Users/chongzhang/WebSim_Dataset/amazon_v2/Musical_Instruments/Embeddings/fast_inference_policy.pt"),
}
dataset_alias = {
    "ml1m": "ml1m",
    "movielens-1m": "ml1m",
    "movielens": "ml1m",
    "movie lens-1m": "ml1m",
    "movie lens": "ml1m",
    "amazon_mi": "amazon_mi",
    "amazon musical instruments": "amazon_mi",
    "amazon musical_instruments": "amazon_mi",
    "musical instruments": "amazon_mi",
    "musical_instruments": "amazon_mi",
}

task_raw = {}
if task_file.exists():
    try:
        task_raw = yaml.safe_load(task_file.read_text(encoding="utf-8")) or {}
    except Exception:
        task_raw = {}

datasets = task_raw.get("dataset", [])
if isinstance(datasets, str):
    datasets = [datasets]
elif not isinstance(datasets, list):
    datasets = []

resolved = []
for item in datasets:
    key = dataset_alias.get(str(item).strip().lower())
    if key:
        resolved.append(key)
if not resolved:
    # Keep consistent with web_sim_camel_agent.py fallback dataset.
    resolved = ["ml1m"]

checked = []
missing = []
for key in resolved:
    if key in checked:
        continue
    checked.append(key)
    path = policy_paths.get(key)
    if path is None:
        continue
    if not path.exists():
        missing.append(f"{key}: {path}")

print("\n".join(missing))
PY
  )"

  if [ -n "${MISSING_FAST_POLICY}" ]; then
    echo "[ERROR] Missing fast policy bundle(s) required by stage-03 inference:" >&2
    printf '%s\n' "${MISSING_FAST_POLICY}" >&2
    echo "Please run Script/02_run_alignment_pretrain_agent.sh first (or provide --fast-policy-path)." >&2
    exit 1
  fi

  EXTRA_ARGS+=(--require-fast-inference)
else
  # Qwen key is only required when fallback decision path is allowed.
  INLINE_KEY="$(
    "${PYTHON_BIN}" -c "import re; from pathlib import Path; text=Path('${PROJECT_ROOT}/web_sim_camel_agent.py').read_text(encoding='utf-8'); m=re.search(r'^\\s*QWEN_API_KEY\\s*=\\s*[\\\"\\'](.*)[\\\"\\']\\s*$', text, flags=re.MULTILINE); print((m.group(1).strip() if m else ''))"
  )"

  if [ -z "${QWEN_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -z "${INLINE_KEY}" ]; then
    echo "[ERROR] Missing Qwen API key." >&2
    echo "Set QWEN_API_KEY or DASHSCOPE_API_KEY, or fill QWEN_API_KEY in web_sim_camel_agent.py." >&2
    exit 1
  fi
fi

# Ensure Playwright Chromium exists before launching the agent.
if ! "${PYTHON_BIN}" -c "import os,sys; from playwright.sync_api import sync_playwright; p=sync_playwright().start(); path=p.chromium.executable_path; p.stop(); sys.exit(0 if os.path.exists(path) else 1)"; then
  echo "[INFO] Playwright Chromium not found. Installing..."
  "${PYTHON_BIN}" -m playwright install chromium
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/web_sim_camel_agent.py" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" "$@"
