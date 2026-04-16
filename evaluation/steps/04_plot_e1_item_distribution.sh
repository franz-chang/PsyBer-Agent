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
DATASET_KEY="${DATASET_KEY:-ml1m}"
RATING_THRESHOLD="${RATING_THRESHOLD:-0}"
TOP_K="${TOP_K:-30}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
OFFLINE_RATINGS_FILE="${OFFLINE_RATINGS_FILE:-}"
OFFLINE_TITLES_FILE="${OFFLINE_TITLES_FILE:-}"

CMD=(
  "${PYTHON_BIN}" "${EVAL_DIR}/plot_e1_item_distribution.py"
  --runs-root "${RUNS_ROOT}"
  --dataset-key "${DATASET_KEY}"
  --rating-threshold "${RATING_THRESHOLD}"
  --top-k "${TOP_K}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${OUTPUT_PREFIX}" ]]; then
  CMD+=(--output-prefix "${OUTPUT_PREFIX}")
fi
if [[ -n "${OFFLINE_RATINGS_FILE}" ]]; then
  CMD+=(--offline-ratings-file "${OFFLINE_RATINGS_FILE}")
fi
if [[ -n "${OFFLINE_TITLES_FILE}" ]]; then
  CMD+=(--offline-titles-file "${OFFLINE_TITLES_FILE}")
fi

echo "[Run] Plot E1 item distribution"
echo "[Run] dataset_key=${DATASET_KEY}"
echo "[Run] runs_root=${RUNS_ROOT}"
echo "[Run] output_dir=${OUTPUT_DIR}"
"${CMD[@]}"
