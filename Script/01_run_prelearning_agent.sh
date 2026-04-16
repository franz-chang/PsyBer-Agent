#!/usr/bin/env sh
set -eu

# Enable pipefail when shell supports it (bash/zsh), ignore otherwise.
if (set -o pipefail) >/dev/null 2>&1; then
  set -o pipefail
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

# Silence macOS malloc stack logging warning in logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

# Prefer the currently activated conda env's python.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

DEFAULT_OUTPUT_DIR="/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main/Embeddings"
AMAZON_MI_OUTPUT_DIR="/Users/chongzhang/WebSim_Dataset/amazon_v2/Musical_Instruments/Embeddings"
DEFAULT_DATASET_DIR="/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main"
AMAZON_ALL_BEAUTY_DATASET_DIR="/Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/All_Beauty"
AMAZON_ALL_BEAUTY_OUTPUT_DIR="${AMAZON_ALL_BEAUTY_DATASET_DIR}/Embeddings"
AMAZON_MAGAZINE_DATASET_DIR="/Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/Magazine_Subscriptions"
AMAZON_MAGAZINE_OUTPUT_DIR="${AMAZON_MAGAZINE_DATASET_DIR}/Embeddings"

has_agent_num=0
has_seed=0
has_help=0
has_dataset_dir=0
has_output_dir=0
output_dir=""
dataset_dir=""
expect_output_dir_value=0
expect_dataset_dir_value=0

# Single pass: discover existing options and output-dir value.
for arg in "$@"; do
  if [ "${expect_dataset_dir_value}" -eq 1 ]; then
    dataset_dir="$arg"
    has_dataset_dir=1
    expect_dataset_dir_value=0
    continue
  fi
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
    --dataset-dir)
      expect_dataset_dir_value=1
      ;;
    --dataset-dir=*)
      has_dataset_dir=1
      dataset_dir="${arg#*=}"
      ;;
    --output-dir)
      has_output_dir=1
      expect_output_dir_value=1
      ;;
    --output-dir=*)
      has_output_dir=1
      output_dir="${arg#*=}"
      ;;
  esac
done

if [ "${has_help}" -eq 1 ]; then
  exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/prelearning_agent.py" "$@"
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

run_prelearning() {
  run_dataset_dir="$1"
  run_output_dir="$2"
  run_label="$3"
  shift 3

  echo "[Run] ${run_label}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/prelearning_agent.py" \
    --dataset-dir "${run_dataset_dir}" \
    --output-dir "${run_output_dir}" \
    "$@"
}

sync_profiles_if_present() {
  src_profiles_file="$1"
  dst_output_dir="$2"

  if [ ! -f "${src_profiles_file}" ]; then
    echo "[Warn] profile file not found: ${src_profiles_file}" >&2
    return 0
  fi

  mkdir -p "${dst_output_dir}"
  dst_profiles_file="${dst_output_dir}/agent_profiles.json"
  if [ "${src_profiles_file}" != "${dst_profiles_file}" ]; then
    cp -f "${src_profiles_file}" "${dst_profiles_file}"
    echo "[Sync] Copied profiles -> ${dst_profiles_file}"
  else
    echo "[Sync] Profiles already in target path: ${dst_profiles_file}"
  fi
}

if [ "${has_dataset_dir}" -eq 1 ] || [ "${has_output_dir}" -eq 1 ]; then
  if [ -z "${dataset_dir}" ]; then
    dataset_dir="${DEFAULT_DATASET_DIR}"
  fi
  if [ "${has_dataset_dir}" -eq 0 ]; then
    set -- --dataset-dir "${dataset_dir}" "$@"
  fi
  if [ "${has_output_dir}" -eq 0 ]; then
    set -- --output-dir "${dataset_dir}/Embeddings" "$@"
  fi
  exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/prelearning_agent.py" "$@"
fi

run_prelearning "${DEFAULT_DATASET_DIR}" "${DEFAULT_OUTPUT_DIR}" "MM-ML-1M" "$@"
run_prelearning "${AMAZON_ALL_BEAUTY_DATASET_DIR}" "${AMAZON_ALL_BEAUTY_OUTPUT_DIR}" "Amazon_MM_2018/All_Beauty" "$@"
run_prelearning "${AMAZON_MAGAZINE_DATASET_DIR}" "${AMAZON_MAGAZINE_OUTPUT_DIR}" "Amazon_MM_2018/Magazine_Subscriptions" "$@"

profiles_file="${output_dir}/agent_profiles.json"
sync_profiles_if_present "${profiles_file}" "${AMAZON_MI_OUTPUT_DIR}"
