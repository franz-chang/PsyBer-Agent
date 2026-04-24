#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

# Silence macOS malloc stack logging warning in logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

# Compatibility shim:
# Resolve WebSim root directory with backward compatibility.
WEB_SIM_ROOT=""
for cand in \
  "${WEBSIM_DIR:-}" \
  "/root/autodl-tmp/Web_simulation" \
  "/root/autodl-tmp/WebSim" \
  "/root/autodl-tmp/Web_sim" \
  "/Users/chongzhang/WebSim" \
  "/Users/chongzhang/Web_sim"; do
  if [ -n "${cand}" ] && [ -d "${cand}" ]; then
    WEB_SIM_ROOT="${cand}"
    break
  fi
done
if [ -z "${WEB_SIM_ROOT}" ]; then
  WEB_SIM_ROOT="/root/autodl-tmp/Web_simulation"
fi
export WEBSIM_DIR="${WEB_SIM_ROOT}"
EXPECTED_SERVICE_SCRIPT="${WEB_SIM_ROOT}/run_service.sh"
FALLBACK_SERVICE_SCRIPT="${WEB_SIM_ROOT}/scripts/run_service.sh"

if [ ! -f "${EXPECTED_SERVICE_SCRIPT}" ] && [ -f "${FALLBACK_SERVICE_SCRIPT}" ]; then
  cat > "${EXPECTED_SERVICE_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
exec "${SCRIPT_DIR}/scripts/run_service.sh" "$@"
EOF
  chmod +x "${EXPECTED_SERVICE_SCRIPT}"
fi

if [ ! -f "${EXPECTED_SERVICE_SCRIPT}" ]; then
  echo "[ERROR] Missing WebSim service launcher script: ${EXPECTED_SERVICE_SCRIPT}" >&2
  echo "Expected fallback was also missing: ${FALLBACK_SERVICE_SCRIPT}" >&2
  exit 1
fi

# Prefer dedicated runtime env first, then activated env, then system python3.
if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -x "/root/miniconda3/bin/python" ]; then
  PYTHON_BIN="/root/miniconda3/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[ERROR] Cannot find a usable python interpreter." >&2
  echo "Tried: /opt/anaconda3/envs/psyber/bin/python, /root/miniconda3/bin/python, CONDA_PREFIX/bin/python, python3, python." >&2
  exit 1
fi

# Some server images export OMP_NUM_THREADS=0, which triggers libgomp warnings.
if ! [[ "${OMP_NUM_THREADS:-1}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

# Prevent silent fallback to random-feed mode when WebSim has no recommender artifacts.
# Set PSY_REQUIRE_WEBSIM_MODELS=0 to bypass this guard intentionally.
REQUIRE_WEBSIM_MODELS_RAW="${PSY_REQUIRE_WEBSIM_MODELS:-1}"
REQUIRE_WEBSIM_MODELS_NORM="$(printf '%s' "${REQUIRE_WEBSIM_MODELS_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${REQUIRE_WEBSIM_MODELS_NORM}" in
  0|false|no|off)
    REQUIRE_WEBSIM_MODELS=0
    ;;
  *)
    REQUIRE_WEBSIM_MODELS=1
    ;;
esac

# Wait for dataset model options to be populated on slower servers.
# You can override this by setting WEBSIM_MODEL_OPTIONS_WAIT_SEC manually.
MODEL_OPTIONS_WAIT_RAW="${WEBSIM_MODEL_OPTIONS_WAIT_SEC:-8}"
if [[ "${MODEL_OPTIONS_WAIT_RAW}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  export WEBSIM_MODEL_OPTIONS_WAIT_SEC="${MODEL_OPTIONS_WAIT_RAW}"
else
  export WEBSIM_MODEL_OPTIONS_WAIT_SEC=8
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
user_fast_policy_path=""
expect_fast_policy_path_value=0
for arg in "$@"; do
  case "$arg" in
    --disable-fast-inference)
      user_disable_fast_inference=1
      ;;
    --fast-policy-path)
      expect_fast_policy_path_value=1
      ;;
    --fast-policy-path=*)
      user_fast_policy_path="${arg#*=}"
      expect_fast_policy_path_value=0
      ;;
    *)
      if [ "${expect_fast_policy_path_value}" -eq 1 ]; then
        user_fast_policy_path="${arg}"
        expect_fast_policy_path_value=0
      fi
      ;;
  esac
done

if [ "${expect_fast_policy_path_value}" -eq 1 ]; then
  echo "[ERROR] '--fast-policy-path' requires a file path value." >&2
  exit 1
fi

EXTRA_ARGS=()
if [ "${STRICT_FAST_INFERENCE}" -eq 1 ]; then
  if [ "${user_disable_fast_inference}" -eq 1 ]; then
    echo "[ERROR] Strict fast inference is enabled, but '--disable-fast-inference' was passed." >&2
    echo "Unset PSY_STRICT_FAST_INFERENCE or remove '--disable-fast-inference'." >&2
    exit 1
  fi

  if [ -n "${user_fast_policy_path}" ]; then
    if [ "${user_fast_policy_path#/}" != "${user_fast_policy_path}" ]; then
      RESOLVED_FAST_POLICY_PATH="${user_fast_policy_path}"
    else
      RESOLVED_FAST_POLICY_PATH="${PROJECT_ROOT}/${user_fast_policy_path}"
    fi

    if [ ! -f "${RESOLVED_FAST_POLICY_PATH}" ]; then
      echo "[ERROR] Provided fast policy file was not found: ${user_fast_policy_path}" >&2
      echo "Resolved path: ${RESOLVED_FAST_POLICY_PATH}" >&2
      exit 1
    fi
  else
    MISSING_FAST_POLICY="$(
      PROJECT_ROOT_FOR_POLICY_CHECK="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(
    os.environ.get("PROJECT_ROOT_FOR_POLICY_CHECK", "/Users/chongzhang/PsyBer-Agent")
)
settings_dir = project_root / "settings"
if not settings_dir.exists():
    settings_dir = project_root / "Settings"
task_file = settings_dir / "task.yaml"
policy_paths = {
    "ml1m": project_root / "outputs" / "fast_policies" / "ml1m" / "fast_inference_policy.pt",
    "amazon_mi": project_root / "outputs" / "fast_policies" / "amazon_mi" / "fast_inference_policy.pt",
    # Amazon_MM_2018 datasets currently reuse ml1m fast policy as fallback.
    "amazon_all_beauty": project_root / "outputs" / "fast_policies" / "ml1m" / "fast_inference_policy.pt",
    "amazon_magazine_subscriptions": project_root / "outputs" / "fast_policies" / "ml1m" / "fast_inference_policy.pt",
}
dataset_alias = {
    "ml1m": "ml1m",
    "movielens-1m": "ml1m",
    "movielens": "ml1m",
    "movie lens-1m": "ml1m",
    "movie lens": "ml1m",
    "amazon all beauty": "amazon_all_beauty",
    "amazon all_beauty": "amazon_all_beauty",
    "all beauty": "amazon_all_beauty",
    "amazon_all_beauty": "amazon_all_beauty",
    "amazon-all-beauty": "amazon_all_beauty",
    "amazon magazine subscriptions": "amazon_magazine_subscriptions",
    "amazon magazine_subscriptions": "amazon_magazine_subscriptions",
    "magazine subscriptions": "amazon_magazine_subscriptions",
    "amazon_magazine_subscriptions": "amazon_magazine_subscriptions",
    "amazon-magazine-subscriptions": "amazon_magazine_subscriptions",
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
    # Keep consistent with WebSim_agent.py fallback dataset.
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
      echo "Please run script/02_run_alignment_pretrain_agent.sh first (or provide --fast-policy-path)." >&2
      exit 1
    fi
  fi

  EXTRA_ARGS+=(--require-fast-inference)
fi

if [ "${REQUIRE_WEBSIM_MODELS}" -eq 1 ]; then
  MISSING_WEBSIM_MODELS="$(
    PROJECT_ROOT_FOR_MODEL_CHECK="${PROJECT_ROOT}" \
    WEBSIM_ROOT_FOR_MODEL_CHECK="${WEB_SIM_ROOT}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

project_root = Path(os.environ.get("PROJECT_ROOT_FOR_MODEL_CHECK", ""))
websim_root = Path(os.environ.get("WEBSIM_ROOT_FOR_MODEL_CHECK", ""))
artifacts_dir = websim_root / "artifacts"

settings_dir = project_root / "settings"
if not settings_dir.exists():
    settings_dir = project_root / "Settings"
task_file = settings_dir / "task.yaml"
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

dataset_alias = {
    "ml1m": "ml1m",
    "movielens-1m": "ml1m",
    "movielens": "ml1m",
    "movie lens-1m": "ml1m",
    "movie lens": "ml1m",
    "amazon all beauty": "amazon_all_beauty",
    "amazon all_beauty": "amazon_all_beauty",
    "all beauty": "amazon_all_beauty",
    "amazon_all_beauty": "amazon_all_beauty",
    "amazon-all-beauty": "amazon_all_beauty",
    "amazon magazine subscriptions": "amazon_magazine_subscriptions",
    "amazon magazine_subscriptions": "amazon_magazine_subscriptions",
    "magazine subscriptions": "amazon_magazine_subscriptions",
    "amazon_magazine_subscriptions": "amazon_magazine_subscriptions",
    "amazon-magazine-subscriptions": "amazon_magazine_subscriptions",
}

resolved = []
for item in datasets:
    key = dataset_alias.get(str(item).strip().lower())
    if key:
        resolved.append(key)
if not resolved:
    resolved = ["ml1m"]

artifact_candidates = {
    "ml1m": [
        "sasrec_ml1m.pt",
        "lightgcn_ml1m.pt",
        "multvae_ml1m.pt",
        "poprec_ml1m.pt",
        "bprmf_ml1m.pt",
        "gru4rec_ml1m.pt",
        "bert4rec_ml1m.pt",
    ],
    "amazon_all_beauty": [
        "sasrec_amazon_all_beauty.pt",
        "lightgcn_amazon_all_beauty.pt",
        "multvae_amazon_all_beauty.pt",
        "poprec_amazon_all_beauty.pt",
        "bprmf_amazon_all_beauty.pt",
        "gru4rec_amazon_all_beauty.pt",
        "bert4rec_amazon_all_beauty.pt",
    ],
    "amazon_magazine_subscriptions": [
        "sasrec_amazon_magazine_subscriptions.pt",
        "lightgcn_amazon_magazine_subscriptions.pt",
        "multvae_amazon_magazine_subscriptions.pt",
        "poprec_amazon_magazine_subscriptions.pt",
        "bprmf_amazon_magazine_subscriptions.pt",
        "gru4rec_amazon_magazine_subscriptions.pt",
        "bert4rec_amazon_magazine_subscriptions.pt",
    ],
}

missing = []
seen = set()
for key in resolved:
    if key in seen:
        continue
    seen.add(key)
    names = artifact_candidates.get(key, [])
    if not names:
        continue
    any_exists = any((artifacts_dir / name).exists() for name in names)
    if not any_exists:
        joined = ", ".join(str(artifacts_dir / name) for name in names)
        missing.append(f"{key}: {joined}")

print("\n".join(missing))
PY
  )"

  if [ -n "${MISSING_WEBSIM_MODELS}" ]; then
    echo "[ERROR] Missing WebSim recommender artifacts for selected dataset(s)." >&2
    echo "Running now would trigger random-feed mode and warning logs." >&2
    printf '%s\n' "${MISSING_WEBSIM_MODELS}" >&2
    echo "Please train/export WebSim model artifacts first, or set PSY_REQUIRE_WEBSIM_MODELS=0 to bypass." >&2
    exit 1
  fi
fi

# Qwen is the final decider (fast policy only provides candidate actions), so key is always required.
INLINE_KEY="$(
  "${PYTHON_BIN}" -c "import re; from pathlib import Path; text=Path('${PROJECT_ROOT}/src/WebSim_agent.py').read_text(encoding='utf-8'); m=re.search(r'^\\s*QWEN_API_KEY\\s*=\\s*[\\\"\\'](.*)[\\\"\\']\\s*$', text, flags=re.MULTILINE); print((m.group(1).strip() if m else ''))"
)"

if [ -z "${QWEN_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -z "${INLINE_KEY}" ]; then
  echo "[ERROR] Missing Qwen API key." >&2
  echo "Set QWEN_API_KEY or DASHSCOPE_API_KEY, or fill QWEN_API_KEY in src/WebSim_agent.py." >&2
  exit 1
fi

# Ensure Playwright Chromium exists before launching the agent.
if ! "${PYTHON_BIN}" -c "import os,sys; from playwright.sync_api import sync_playwright; p=sync_playwright().start(); path=p.chromium.executable_path; p.stop(); sys.exit(0 if os.path.exists(path) else 1)"; then
  echo "[INFO] Playwright Chromium not found. Installing..."
  "${PYTHON_BIN}" -m playwright install chromium
fi

# Headless is required on remote servers without X/Wayland display.
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  EXTRA_ARGS+=(--headless)
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/WebSim_agent.py" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" "$@"
