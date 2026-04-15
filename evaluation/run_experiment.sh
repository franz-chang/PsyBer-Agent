#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <all|E1|E2|E3|E4|E5|E6|E7|E1,E3,...> [extra args]"
  echo "Env overrides:"
  echo "  RUNS_ROOT BATCH_ROOT OUTPUT_DIR REFERENCE_FILE AB_CASES_FILE"
  echo "  RUN_METADATA_FILE MAX_E7_CASES PYTHON_BIN"
  exit 1
fi

EXPERIMENTS="$1"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Silence macOS malloc stack logging warning in logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
BATCH_ROOT="${BATCH_ROOT:-${PROJECT_ROOT}/batch_launches}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/reports}"
REFERENCE_FILE="${REFERENCE_FILE:-}"
AB_CASES_FILE="${AB_CASES_FILE:-}"
RUN_METADATA_FILE="${RUN_METADATA_FILE:-}"
MAX_E7_CASES="${MAX_E7_CASES:-5}"

mkdir -p "${OUTPUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_TAG="$(echo "${EXPERIMENTS}" | tr '[:lower:]' '[:upper:]' | tr ', ' '__' | tr -cd 'A-Z0-9_')"
if [[ -z "${EXP_TAG}" ]]; then
  EXP_TAG="ALL"
fi

EXP_LOWER="$(echo "${EXPERIMENTS}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXP_LOWER}" == "all" ]]; then
  OUTPUT_NAME="experiment_suite_${STAMP}.json"
else
  OUTPUT_NAME="experiment_${EXP_TAG}_${STAMP}.json"
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment_suite.py"
  --runs-root "${RUNS_ROOT}"
  --batch-root "${BATCH_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --output-name "${OUTPUT_NAME}"
  --experiments "${EXPERIMENTS}"
  --max-e7-cases "${MAX_E7_CASES}"
)

if [[ -n "${REFERENCE_FILE}" ]]; then
  CMD+=(--reference-file "${REFERENCE_FILE}")
fi
if [[ -n "${AB_CASES_FILE}" ]]; then
  CMD+=(--ab-cases-file "${AB_CASES_FILE}")
fi
if [[ -n "${RUN_METADATA_FILE}" ]]; then
  CMD+=(--run-metadata-file "${RUN_METADATA_FILE}")
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[Run] experiments=${EXPERIMENTS}"
echo "[Run] runs_root=${RUNS_ROOT}"
echo "[Run] batch_root=${BATCH_ROOT}"
echo "[Run] output=${OUTPUT_DIR}/${OUTPUT_NAME}"
"${CMD[@]}"
echo "[Done] ${OUTPUT_DIR}/${OUTPUT_NAME}"
