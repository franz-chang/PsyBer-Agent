#!/usr/bin/env sh
set -eu

# Enable pipefail when shell supports it (bash/zsh), ignore otherwise.
if (set -o pipefail) >/dev/null 2>&1; then
  set -o pipefail
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

# Prefer the currently activated conda env's python.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

DEFAULT_OUTPUT_DIR="/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/Embeddings"
AMAZON_MI_OUTPUT_DIR="/Users/chongzhang/WebSim_Dataset/amazon_v2/Musical_Instruments/Embeddings"

has_agent_num=0
has_seed=0
has_help=0
output_dir=""
expect_output_dir_value=0

# Single pass: discover existing options and output-dir value.
for arg in "$@"; do
  if [ "${expect_output_dir_value}" -eq 1 ]; then
    output_dir="$arg"
    expect_output_dir_value=0
    continue
  fi
  case "$arg" in
    --agent-num|--agent-num=*)
      has_agent_num=1
      ;;
    --seed|--seed=*)
      has_seed=1
      ;;
    -h|--help)
      has_help=1
      ;;
    --output-dir)
      expect_output_dir_value=1
      ;;
    --output-dir=*)
      output_dir="${arg#*=}"
      ;;
  esac
done

if [ "${has_help}" -eq 1 ]; then
  exec "${PYTHON_BIN}" "${PROJECT_ROOT}/prelearning_agent.py" "$@"
fi

if [ "${has_agent_num}" -eq 0 ]; then
  set -- --agent-num 500 "$@"
fi

if [ "${has_seed}" -eq 0 ]; then
  # Default to a non-fixed seed so each run can produce a fresh random profile pool.
  set -- --seed "$(date +%s)" "$@"
fi

if [ -z "${output_dir}" ]; then
  output_dir="${DEFAULT_OUTPUT_DIR}"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/prelearning_agent.py" "$@"

profiles_file="${output_dir}/agent_profiles.json"
if [ -f "${profiles_file}" ]; then
  mkdir -p "${AMAZON_MI_OUTPUT_DIR}"
  target_profiles_file="${AMAZON_MI_OUTPUT_DIR}/agent_profiles.json"
  if [ "${profiles_file}" != "${target_profiles_file}" ]; then
    cp -f "${profiles_file}" "${target_profiles_file}"
    echo "[Sync] Copied profiles -> ${target_profiles_file}"
  else
    echo "[Sync] Profiles already in target path: ${target_profiles_file}"
  fi
else
  echo "[Warn] profile file not found: ${profiles_file}" >&2
fi
