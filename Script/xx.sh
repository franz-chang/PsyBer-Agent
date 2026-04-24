#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

N_VALUE=""
REPEATS=3
TRACK_OVERRIDE=20
LAUNCH_RATE=20
TASK_TIMEOUT_SEC=1800
MAX_CONCURRENCY_CAP=""
DECISION_MODEL=""
QWEN_BASE_URL=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash script/xx.sh -n <concurrency> [options]

Options:
  -n, --n <int>                    Concurrency user count (required)
  -r, --repeats <int>              Repeat times (default: 3)
  -t, --track <int>                Track override (default: 20)
      --launch-rate <float>        Launch rate (default: 20)
      --task-timeout-sec <float>   Task timeout seconds (default: 1800)
      --max-concurrency-cap <int>  Cap actual process concurrency
      --decision-model <name>      Qwen model override
      --qwen-base-url <url>        Qwen base URL override
      --dry-run                    Dry run (no real task execution)
  -h, --help                       Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--n)
      N_VALUE="${2:-}"
      shift 2
      ;;
    -r|--repeats)
      REPEATS="${2:-}"
      shift 2
      ;;
    -t|--track)
      TRACK_OVERRIDE="${2:-}"
      shift 2
      ;;
    --launch-rate)
      LAUNCH_RATE="${2:-}"
      shift 2
      ;;
    --task-timeout-sec)
      TASK_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --max-concurrency-cap)
      MAX_CONCURRENCY_CAP="${2:-}"
      shift 2
      ;;
    --decision-model)
      DECISION_MODEL="${2:-}"
      shift 2
      ;;
    --qwen-base-url)
      QWEN_BASE_URL="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "${N_VALUE}" ]; then
  echo "[ERROR] Missing required argument: -n <concurrency>" >&2
  usage
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/scalability_cost/run_scalability_cost.py"
  --project-root "${PROJECT_ROOT}"
  -n "${N_VALUE}"
  --repeats "${REPEATS}"
  --track-override "${TRACK_OVERRIDE}"
  --launch-rate "${LAUNCH_RATE}"
  --task-timeout-sec "${TASK_TIMEOUT_SEC}"
  --headless
)

if [ -n "${MAX_CONCURRENCY_CAP}" ]; then
  CMD+=(--max-concurrency-cap "${MAX_CONCURRENCY_CAP}")
fi
if [ -n "${DECISION_MODEL}" ]; then
  CMD+=(--decision-model "${DECISION_MODEL}")
fi
if [ -n "${QWEN_BASE_URL}" ]; then
  CMD+=(--qwen-base-url "${QWEN_BASE_URL}")
fi
if [ "${DRY_RUN}" -eq 1 ]; then
  CMD+=(--dry-run)
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[Run] ${CMD[*]}"
exec "${CMD[@]}"

