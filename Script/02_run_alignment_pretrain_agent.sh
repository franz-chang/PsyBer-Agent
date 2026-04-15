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
has_epochs=0
has_lr=0
has_imitation_coef=0
has_imitation_decay=0
has_skip_recovery_boost=0
has_ugw_eps=0
has_expert_time_weight=0
has_agent_time_weight=0
has_agent_delta_weight=0
has_ugw_init_temperature=0
has_ugw_init_delta_weight=0
has_ugw_init_time_weight=0
has_ugw_init_reg_epochs=0
has_ugw_init_reg_batch_size=0
has_ugw_init_reg_lr=0

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
    --epochs|--epochs=*)
      has_epochs=1
      ;;
    --lr|--lr=*)
      has_lr=1
      ;;
    --imitation-coef|--imitation-coef=*)
      has_imitation_coef=1
      ;;
    --imitation-decay|--imitation-decay=*)
      has_imitation_decay=1
      ;;
    --skip-recovery-boost|--skip-recovery-boost=*)
      has_skip_recovery_boost=1
      ;;
    --ugw-eps|--ugw-eps=*)
      has_ugw_eps=1
      ;;
    --expert-time-weight|--expert-time-weight=*)
      has_expert_time_weight=1
      ;;
    --agent-time-weight|--agent-time-weight=*)
      has_agent_time_weight=1
      ;;
    --agent-delta-weight|--agent-delta-weight=*)
      has_agent_delta_weight=1
      ;;
    --ugw-init-temperature|--ugw-init-temperature=*)
      has_ugw_init_temperature=1
      ;;
    --ugw-init-delta-weight|--ugw-init-delta-weight=*)
      has_ugw_init_delta_weight=1
      ;;
    --ugw-init-time-weight|--ugw-init-time-weight=*)
      has_ugw_init_time_weight=1
      ;;
    --ugw-init-reg-epochs|--ugw-init-reg-epochs=*)
      has_ugw_init_reg_epochs=1
      ;;
    --ugw-init-reg-batch-size|--ugw-init-reg-batch-size=*)
      has_ugw_init_reg_batch_size=1
      ;;
    --ugw-init-reg-lr|--ugw-init-reg-lr=*)
      has_ugw_init_reg_lr=1
      ;;
  esac
done

if [ "${has_help}" -eq 1 ]; then
  exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/alignment_pretrain_agent.py" "$@"
fi

# Default to simplified process: regression-initialized UGW soft-label supervision.
if [ "${has_alignment_mode}" -eq 0 ]; then
  alignment_mode="softlabel_ugw"
  set -- --alignment-mode "${alignment_mode}" "$@"
fi

# Use a practical default training length for stage-02.
if [ "${has_epochs}" -eq 0 ]; then
  set -- --epochs 100 "$@"
  epochs_note="default(100)"
else
  epochs_note="user"
fi

# Stabilize soft-label training defaults (including UGW regression initializer).
if [ "${alignment_mode}" = "softlabel_ugw" ]; then
  if [ "${has_lr}" -eq 0 ]; then
    set -- --lr 5e-3 "$@"
  fi
  if [ "${has_imitation_coef}" -eq 0 ]; then
    set -- --imitation-coef 0.20 "$@"
  fi
  if [ "${has_imitation_decay}" -eq 0 ]; then
    set -- --imitation-decay 0.97 "$@"
  fi
  if [ "${has_skip_recovery_boost}" -eq 0 ]; then
    set -- --skip-recovery-boost 0.10 "$@"
  fi
  if [ "${has_ugw_eps}" -eq 0 ]; then
    set -- --ugw-eps 0.05 "$@"
  fi
  if [ "${has_expert_time_weight}" -eq 0 ]; then
    set -- --expert-time-weight 3.0 "$@"
  fi
  if [ "${has_agent_time_weight}" -eq 0 ]; then
    set -- --agent-time-weight 3.0 "$@"
  fi
  if [ "${has_agent_delta_weight}" -eq 0 ]; then
    set -- --agent-delta-weight 0.10 "$@"
  fi
  if [ "${has_ugw_init_temperature}" -eq 0 ]; then
    set -- --ugw-init-temperature 0.05 "$@"
  fi
  if [ "${has_ugw_init_delta_weight}" -eq 0 ]; then
    set -- --ugw-init-delta-weight 0.20 "$@"
  fi
  if [ "${has_ugw_init_time_weight}" -eq 0 ]; then
    set -- --ugw-init-time-weight 0.40 "$@"
  fi
  if [ "${has_ugw_init_reg_epochs}" -eq 0 ]; then
    set -- --ugw-init-reg-epochs 50 "$@"
  fi
  if [ "${has_ugw_init_reg_batch_size}" -eq 0 ]; then
    set -- --ugw-init-reg-batch-size 2048 "$@"
  fi
  if [ "${has_ugw_init_reg_lr}" -eq 0 ]; then
    set -- --ugw-init-reg-lr 5e-3 "$@"
  fi
fi

echo "[Run] alignment-mode=${alignment_mode:-softlabel_ugw}, epochs=${epochs_note}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/alignment_pretrain_agent.py" "$@"
