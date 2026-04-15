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
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/reports}"
MEMORY_OUTPUT_NAME="${MEMORY_OUTPUT_NAME:-}"
MEMORY_REFERENCE_FILE="${MEMORY_REFERENCE_FILE:-${REFERENCE_FILE:-}}"

CMD=(
  "${PYTHON_BIN}" "${EVAL_DIR}/evaluate_memory_metrics.py"
  --runs-root "${RUNS_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MEMORY_OUTPUT_NAME}" ]]; then
  CMD+=(--output-name "${MEMORY_OUTPUT_NAME}")
fi
if [[ -n "${MEMORY_REFERENCE_FILE}" ]]; then
  CMD+=(--reference-file "${MEMORY_REFERENCE_FILE}")
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[Step2] Run memory metrics evaluation"
echo "[Step2] runs_root=${RUNS_ROOT}"
echo "[Step2] output_dir=${OUTPUT_DIR}"
if [[ -n "${MEMORY_OUTPUT_NAME}" ]]; then
  echo "[Step2] output_name=${MEMORY_OUTPUT_NAME}"
fi
"${CMD[@]}"

