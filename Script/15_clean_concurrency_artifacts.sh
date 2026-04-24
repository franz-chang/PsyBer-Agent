#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DRY_RUN=0
INCLUDE_LEGACY=0

usage() {
  cat <<'EOF'
Usage:
  bash Script/15_clean_concurrency_artifacts.sh [options]

Options:
      --project-root <path>   Project root path (default: auto-detected)
      --dry-run               Preview only; do not delete anything
      --include-legacy        Also clean compare_runs/legacy/batch_launches
  -h, --help                  Show this help

Default clean targets:
  1) <project_root>/batch_launches
  2) <project_root>/experiments/scalability_cost/reports
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-legacy)
      INCLUDE_LEGACY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "${PROJECT_ROOT}" ]; then
  echo "[ERROR] --project-root cannot be empty" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "[ERROR] project root does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi

TARGET_DIRS=(
  "${PROJECT_ROOT}/batch_launches"
  "${PROJECT_ROOT}/experiments/scalability_cost/reports"
)

if [ "${INCLUDE_LEGACY}" -eq 1 ]; then
  TARGET_DIRS+=("${PROJECT_ROOT}/compare_runs/legacy/batch_launches")
fi

count_entries() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    echo "0"
    return
  fi
  find "${dir}" -mindepth 1 -maxdepth 1 | wc -l | awk '{print $1}'
}

size_human() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    echo "missing"
    return
  fi
  du -sh "${dir}" | awk '{print $1}'
}

delete_contents() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    return
  fi
  find "${dir}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

echo "[Clean] project_root=${PROJECT_ROOT}"
echo "[Clean] dry_run=${DRY_RUN}, include_legacy=${INCLUDE_LEGACY}"
echo

total_before=0
total_after=0

for dir in "${TARGET_DIRS[@]}"; do
  before_count="$(count_entries "${dir}")"
  before_size="$(size_human "${dir}")"
  total_before=$((total_before + before_count))

  echo "[Target] ${dir}"
  echo "  before: entries=${before_count}, size=${before_size}"

  if [ ! -d "${dir}" ]; then
    echo "  action: skipped (missing)"
    echo
    continue
  fi

  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  action: dry-run (no deletion)"
    after_count="${before_count}"
    after_size="${before_size}"
  else
    delete_contents "${dir}"
    after_count="$(count_entries "${dir}")"
    after_size="$(size_human "${dir}")"
    echo "  action: cleaned"
  fi

  total_after=$((total_after + after_count))
  echo "  after:  entries=${after_count}, size=${after_size}"
  echo "  removed=$((before_count - after_count))"
  echo
done

echo "[Summary]"
echo "  total_before_entries=${total_before}"
echo "  total_after_entries=${total_after}"
echo "  total_removed_entries=$((total_before - total_after))"

