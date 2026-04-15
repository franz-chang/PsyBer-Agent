#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

# Silence macOS malloc stack logging warning in logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
BATCH_ROOT="${BATCH_ROOT:-${PROJECT_ROOT}/batch_launches}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/reports}"
MAX_E7_CASES="${MAX_E7_CASES:-5}"
EXPERIMENTS="${EXPERIMENTS:-all}"
SUITE_OUTPUT_NAME="${SUITE_OUTPUT_NAME:-}"

CMD=(
  "${PYTHON_BIN}" "${EVAL_DIR}/run_experiment_suite.py"
  --runs-root "${RUNS_ROOT}"
  --batch-root "${BATCH_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --experiments "${EXPERIMENTS}"
  --max-e7-cases "${MAX_E7_CASES}"
)

if [[ -n "${SUITE_OUTPUT_NAME}" ]]; then
  CMD+=(--output-name "${SUITE_OUTPUT_NAME}")
fi
if [[ -n "${REFERENCE_FILE:-}" ]]; then
  CMD+=(--reference-file "${REFERENCE_FILE}")
fi
if [[ -n "${AB_CASES_FILE:-}" ]]; then
  CMD+=(--ab-cases-file "${AB_CASES_FILE}")
fi
if [[ -n "${RUN_METADATA_FILE:-}" ]]; then
  CMD+=(--run-metadata-file "${RUN_METADATA_FILE}")
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[Step1] Run experiment suite (E1-E7)"
echo "[Step1] runs_root=${RUNS_ROOT}"
echo "[Step1] batch_root=${BATCH_ROOT}"
echo "[Step1] output_dir=${OUTPUT_DIR}"
if [[ -n "${SUITE_OUTPUT_NAME}" ]]; then
  echo "[Step1] output_name=${SUITE_OUTPUT_NAME}"
fi
"${CMD[@]}"

