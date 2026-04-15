#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Silence macOS malloc stack logging warning in logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

# -------- defaults --------
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
BATCH_ROOT="${BATCH_ROOT:-${PROJECT_ROOT}/batch_launches}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/reports}"
REPORTS_DIR="${REPORTS_DIR:-${OUTPUT_DIR}}"
OUTPUT_MD="${OUTPUT_MD:-${REPORTS_DIR}/preliminary_experiment_results.md}"
MAX_E7_CASES="${MAX_E7_CASES:-5}"
MAX_RUNS="${MAX_RUNS:-20}"
EXPERIMENTS="${EXPERIMENTS:-all}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_OUTPUT_NAME="${SUITE_OUTPUT_NAME:-experiment_suite_${STAMP}.json}"
MEMORY_OUTPUT_NAME="${MEMORY_OUTPUT_NAME:-memory_eval_${STAMP}.json}"

STEP_DIR="${SCRIPT_DIR}/steps"
STEP_01="${STEP_DIR}/01_run_experiment_suite.sh"
STEP_02="${STEP_DIR}/02_run_memory_eval.sh"
STEP_03="${STEP_DIR}/03_generate_markdown_report.sh"

if [[ ! -x "${STEP_01}" || ! -x "${STEP_02}" || ! -x "${STEP_03}" ]]; then
  echo "[ERROR] Missing step scripts under: ${STEP_DIR}" >&2
  echo "Expected executable files:" >&2
  echo "  - ${STEP_01}" >&2
  echo "  - ${STEP_02}" >&2
  echo "  - ${STEP_03}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${REPORTS_DIR}"

echo "[Pipeline] Start full evaluation pipeline"
echo "[Pipeline] runs_root=${RUNS_ROOT}"
echo "[Pipeline] batch_root=${BATCH_ROOT}"
echo "[Pipeline] reports_dir=${REPORTS_DIR}"

# Step 1: E1-E7 suite
PYTHON_BIN="${PYTHON_BIN}" \
RUNS_ROOT="${RUNS_ROOT}" \
BATCH_ROOT="${BATCH_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
REFERENCE_FILE="${REFERENCE_FILE:-}" \
AB_CASES_FILE="${AB_CASES_FILE:-}" \
RUN_METADATA_FILE="${RUN_METADATA_FILE:-}" \
MAX_E7_CASES="${MAX_E7_CASES}" \
EXPERIMENTS="${EXPERIMENTS}" \
SUITE_OUTPUT_NAME="${SUITE_OUTPUT_NAME}" \
"${STEP_01}" "$@"

SUITE_REPORT="${OUTPUT_DIR}/${SUITE_OUTPUT_NAME}"

# Step 2: memory metrics
PYTHON_BIN="${PYTHON_BIN}" \
RUNS_ROOT="${RUNS_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
REFERENCE_FILE="${REFERENCE_FILE:-}" \
MEMORY_REFERENCE_FILE="${MEMORY_REFERENCE_FILE:-}" \
MEMORY_OUTPUT_NAME="${MEMORY_OUTPUT_NAME}" \
"${STEP_02}"

MEMORY_REPORT="${OUTPUT_DIR}/${MEMORY_OUTPUT_NAME}"

# Step 3: markdown report
PYTHON_BIN="${PYTHON_BIN}" \
RUNS_ROOT="${RUNS_ROOT}" \
REPORTS_DIR="${REPORTS_DIR}" \
SUITE_REPORT="${SUITE_REPORT}" \
MEMORY_REPORT="${MEMORY_REPORT}" \
OUTPUT_MD="${OUTPUT_MD}" \
MAX_RUNS="${MAX_RUNS}" \
"${STEP_03}"

echo "[Pipeline] Done"
echo "[Pipeline] suite_report=${SUITE_REPORT}"
echo "[Pipeline] memory_report=${MEMORY_REPORT}"
echo "[Pipeline] markdown_report=${OUTPUT_MD}"
