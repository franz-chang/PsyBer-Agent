#!/usr/bin/env sh
set -eu

if (set -o pipefail) >/dev/null 2>&1; then
  set -o pipefail
fi

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

has_help=0
has_alignment_mode=0
alignment_mode=""
expect_alignment_mode_value=0
has_imitation_coef=0
has_deterministic_action=0

for arg in "$@"; do
  if [ "${expect_alignment_mode_value}" -eq 1 ]; then
    alignment_mode="$arg"
    has_alignment_mode=1
    expect_alignment_mode_value=0
    continue
  fi

  case "$arg" in
    -h|--help)
      has_help=1
      ;;
    --alignment-mode)
      expect_alignment_mode_value=1
      ;;
    --alignment-mode=*)
      has_alignment_mode=1
      alignment_mode="${arg#*=}"
      ;;
    --imitation-coef|--imitation-coef=*)
      has_imitation_coef=1
      ;;
    --deterministic-action)
      has_deterministic_action=1
      ;;
  esac
done

if [ "${has_help}" -eq 1 ]; then
  exec "${PYTHON_BIN}" "${PROJECT_ROOT}/alignment_pretrain_agent.py" "$@"
fi

# Default to simplified process: UGW soft-label supervision.
if [ "${has_alignment_mode}" -eq 0 ]; then
  alignment_mode="softlabel_ugw"
  set -- --alignment-mode "${alignment_mode}" "$@"
fi

if [ "${alignment_mode}" != "legacy_rl" ]; then
  if [ "${has_imitation_coef}" -eq 0 ]; then
    set -- --imitation-coef 0.0 "$@"
  fi
  if [ "${has_deterministic_action}" -eq 0 ]; then
    set -- --deterministic-action "$@"
  fi
fi

echo "[Run] alignment-mode=${alignment_mode:-softlabel_ugw}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/alignment_pretrain_agent.py" "$@"
