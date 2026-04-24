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

CONDITION_IDS_DEFAULT="base_clean,decision_mask_10,decision_mask_20,decision_mask_30,decision_adjacent_swap,decision_top1_replacement,decision_distractor_insertion,decision_conflicting_signals"

CONDITION_IDS="${CONDITION_IDS:-${CONDITION_IDS_DEFAULT}}"
BASE_TRACK="${BASE_TRACK:-20}"
BASE_DATASET="${BASE_DATASET:-MovieLens-1M}"
BASE_MODEL="${BASE_MODEL:-LightGCN}"
DECISION_MODEL="${DECISION_MODEL:-qwen-plus}"
SEEDS="${SEEDS:-101,202,303}"
AB_CASES_FILE="${AB_CASES_FILE:-}"
QWEN_BASE_URL="${QWEN_BASE_URL:-}"
HEADLESS=1

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  script/14_run_decision_robustness.sh [options] [extra args...]

Options:
  -m, --decision-model <name>     Decision model name (default: qwen-plus)
  -a, --ab-cases-file <path>      AB cases json file (optional, for Sign/Lift/Type-I)
      --seeds <csv>               Seeds list (default: 101,202,303)
      --base-track <int>          Base track length (default: 20)
      --base-dataset <name>       Base dataset (default: MovieLens-1M)
      --base-model <name>         Base recommendation model (default: LightGCN)
      --condition-ids <csv>       Override condition ids
      --qwen-base-url <url>       Custom Qwen base url (optional)
      --no-headless               Disable headless mode
      --headless                  Enable headless mode (default)
  -h, --help                      Show this help

Examples:
  script/14_run_decision_robustness.sh -m qwen-plus
  script/14_run_decision_robustness.sh -m qwen-max --ab-cases-file /abs/path/ab_cases.json
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -m|--decision-model)
      DECISION_MODEL="${2:-}"
      shift 2
      ;;
    -a|--ab-cases-file)
      AB_CASES_FILE="${2:-}"
      shift 2
      ;;
    --seeds)
      SEEDS="${2:-}"
      shift 2
      ;;
    --base-track)
      BASE_TRACK="${2:-}"
      shift 2
      ;;
    --base-dataset)
      BASE_DATASET="${2:-}"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="${2:-}"
      shift 2
      ;;
    --condition-ids)
      CONDITION_IDS="${2:-}"
      shift 2
      ;;
    --qwen-base-url)
      QWEN_BASE_URL="${2:-}"
      shift 2
      ;;
    --no-headless)
      HEADLESS=0
      shift
      ;;
    --headless)
      HEADLESS=1
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

if [ -z "${DECISION_MODEL}" ]; then
  echo "[ERROR] decision model cannot be empty." >&2
  exit 1
fi

if [ -n "${AB_CASES_FILE}" ]; then
  if [ "${AB_CASES_FILE#/}" = "${AB_CASES_FILE}" ]; then
    AB_CASES_FILE="${PROJECT_ROOT}/${AB_CASES_FILE}"
  fi
  if [ ! -f "${AB_CASES_FILE}" ]; then
    echo "[ERROR] AB cases file not found: ${AB_CASES_FILE}" >&2
    exit 1
  fi
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robustness/run_robustness_suite.py"
  --condition-ids "${CONDITION_IDS}"
  --base-track "${BASE_TRACK}"
  --base-dataset "${BASE_DATASET}"
  --base-model "${BASE_MODEL}"
  --decision-model "${DECISION_MODEL}"
  --seeds "${SEEDS}"
)

if [ "${HEADLESS}" -eq 1 ]; then
  CMD+=(--headless)
fi

if [ -n "${AB_CASES_FILE}" ]; then
  CMD+=(--ab-cases-file "${AB_CASES_FILE}")
else
  echo "[Warn] No --ab-cases-file provided. Sign Accuracy/Lift MAE/Type-I may be N/A."
fi

if [ -n "${QWEN_BASE_URL}" ]; then
  CMD+=(--qwen-base-url "${QWEN_BASE_URL}")
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[Info] Project root: ${PROJECT_ROOT}"
echo "[Info] Python: ${PYTHON_BIN}"
echo "[Info] Decision model: ${DECISION_MODEL}"
echo "[Info] Seeds: ${SEEDS}"
echo "[Info] Condition IDs: ${CONDITION_IDS}"
if [ -n "${AB_CASES_FILE}" ]; then
  echo "[Info] AB cases: ${AB_CASES_FILE}"
fi
echo "[Run] ${CMD[*]}"

QWEN_MODEL="${DECISION_MODEL}" "${CMD[@]}"

