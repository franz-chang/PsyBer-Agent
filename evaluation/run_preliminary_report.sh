#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
REPORTS_DIR="${REPORTS_DIR:-${SCRIPT_DIR}/reports}"
SUITE_REPORT="${SUITE_REPORT:-}"
MEMORY_REPORT="${MEMORY_REPORT:-}"
OUTPUT_MD="${OUTPUT_MD:-${REPORTS_DIR}/preliminary_experiment_results.md}"
MAX_RUNS="${MAX_RUNS:-20}"
if ! [[ "${MAX_RUNS}" =~ ^[0-9]+$ ]] || [[ "${MAX_RUNS}" -le 0 ]]; then
  MAX_RUNS=20
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_preliminary_report.py"
  --runs-root "${RUNS_ROOT}"
  --reports-dir "${REPORTS_DIR}"
  --output "${OUTPUT_MD}"
  --max-runs "${MAX_RUNS}"
)

if [[ -n "${SUITE_REPORT}" ]]; then
  CMD+=(--suite-report "${SUITE_REPORT}")
fi
if [[ -n "${MEMORY_REPORT}" ]]; then
  CMD+=(--memory-report "${MEMORY_REPORT}")
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[Run] output_md=${OUTPUT_MD}"
"${CMD[@]}"
