#!/usr/bin/env sh
set -eu

if (set -o pipefail) >/dev/null 2>&1; then
  set -o pipefail
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
AUTODL_ROOT="$(CDPATH= cd -- "${PROJECT_ROOT}/.." && pwd -P)"
DATASET_ROOT="${AUTODL_ROOT}/WebSim_Dataset"
SETTINGS_DIR="${PROJECT_ROOT}/settings"
if [ ! -d "${SETTINGS_DIR}" ]; then
  SETTINGS_DIR="${PROJECT_ROOT}/Settings"
fi
TASK_FILE="${SETTINGS_DIR}/task.yaml"

cd "${PROJECT_ROOT}"

# Silence macOS malloc stack logging warning in logs.
unset MallocStackLogging
unset MallocStackLoggingNoCompact

# Guard against invalid OpenMP thread count values from shell environments.
case "${OMP_NUM_THREADS:-}" in
  ""|*[!0-9]*|0)
    export OMP_NUM_THREADS=1
    ;;
esac

# Prefer dedicated runtime env first, then activated env, then system python3.
if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

resolve_dataset_key_from_task() {
  TASK_FILE_FOR_DATASET="${TASK_FILE}" "${PYTHON_BIN}" - <<'PY'
import os
import re
from pathlib import Path


task_file = Path(os.environ.get("TASK_FILE_FOR_DATASET", ""))

alias = {
    "ml1m": "ml1m",
    "movielens-1m": "ml1m",
    "movielens": "ml1m",
    "movie lens-1m": "ml1m",
    "movie lens": "ml1m",
    "movielens 1m": "ml1m",
    "movie lens 1m": "ml1m",
    "amazon all beauty": "amazon_all_beauty",
    "amazon all_beauty": "amazon_all_beauty",
    "amazon-all-beauty": "amazon_all_beauty",
    "all beauty": "amazon_all_beauty",
    "all_beauty": "amazon_all_beauty",
    "amazon_all_beauty": "amazon_all_beauty",
    "amazon magazine subscriptions": "amazon_magazine_subscriptions",
    "amazon magazine_subscriptions": "amazon_magazine_subscriptions",
    "amazon-magazine-subscriptions": "amazon_magazine_subscriptions",
    "magazine subscriptions": "amazon_magazine_subscriptions",
    "magazine_subscriptions": "amazon_magazine_subscriptions",
    "amazon_magazine_subscriptions": "amazon_magazine_subscriptions",
    "amazon_mi": "amazon_mi",
    "amazon musical instruments": "amazon_mi",
    "amazon musical_instruments": "amazon_mi",
    "musical instruments": "amazon_mi",
    "musical_instruments": "amazon_mi",
}


def normalize_items(items):
    out = []
    for item in items:
        key = alias.get(str(item).strip().lower())
        if key:
            out.append(key)
    return out


resolved = []
if task_file.exists():
    text = task_file.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text) or {}
        datasets = raw.get("dataset", [])
        if isinstance(datasets, str):
            datasets = [datasets]
        elif not isinstance(datasets, list):
            datasets = []
        resolved = normalize_items(datasets)
    except Exception:
        items = []
        in_dataset = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^dataset\s*:\s*$", stripped):
                in_dataset = True
                continue
            if in_dataset:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", stripped):
                    break
                if stripped.startswith("-"):
                    item = stripped[1:].split("#", 1)[0].strip().strip("\"'")
                    if item:
                        items.append(item)
        resolved = normalize_items(items)

print(resolved[0] if resolved else "ml1m")
PY
}

infer_policy_key_from_dataset_dir() {
  dataset_path_lc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${dataset_path_lc}" in
    *amazon_v2*|*musical*instrument*)
      echo "amazon_mi"
      ;;
    *all_beauty*|*all-beauty*|*allbeauty*)
      echo "amazon_all_beauty"
      ;;
    *magazine_subscriptions*|*magazine-subscriptions*|*magazine*subscriptions*)
      echo "amazon_magazine_subscriptions"
      ;;
    *)
      echo "ml1m"
      ;;
  esac
}

infer_default_min_seq_len_from_dataset_dir() {
  dataset_path_lc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${dataset_path_lc}" in
    *all_beauty*|*all-beauty*|*allbeauty*|*magazine_subscriptions*|*magazine-subscriptions*|*magazine*subscriptions*)
      # Amazon_MM_2018 subsets are highly sparse; 20 would filter out almost all users.
      echo "3"
      ;;
    *)
      echo ""
      ;;
  esac
}

has_help=0
has_alignment_mode=0
alignment_mode=""
expect_alignment_mode_value=0
has_epochs=0
has_min_seq_len=0
has_lr=0
has_imitation_coef=0
has_imitation_decay=0
has_include_skip_action=0
has_skip_action_weight=0
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
has_dataset_dir=0
dataset_dir_value=""
expect_dataset_dir_value=0
has_embeddings_file=0
expect_embeddings_file_value=0
has_profiles_file=0
expect_profiles_file_value=0
has_policy_out=0
policy_out_value=""
expect_policy_out_value=0
has_report_out=0
expect_report_out_value=0
has_traj_preview_out=0
expect_traj_preview_out_value=0

for arg in "$@"; do
  if [ "${expect_alignment_mode_value}" -eq 1 ]; then
    alignment_mode="$arg"
    has_alignment_mode=1
    expect_alignment_mode_value=0
    continue
  fi
  if [ "${expect_dataset_dir_value}" -eq 1 ]; then
    dataset_dir_value="$arg"
    has_dataset_dir=1
    expect_dataset_dir_value=0
    continue
  fi
  if [ "${expect_embeddings_file_value}" -eq 1 ]; then
    has_embeddings_file=1
    expect_embeddings_file_value=0
    continue
  fi
  if [ "${expect_profiles_file_value}" -eq 1 ]; then
    has_profiles_file=1
    expect_profiles_file_value=0
    continue
  fi
  if [ "${expect_policy_out_value}" -eq 1 ]; then
    policy_out_value="$arg"
    has_policy_out=1
    expect_policy_out_value=0
    continue
  fi
  if [ "${expect_report_out_value}" -eq 1 ]; then
    has_report_out=1
    expect_report_out_value=0
    continue
  fi
  if [ "${expect_traj_preview_out_value}" -eq 1 ]; then
    has_traj_preview_out=1
    expect_traj_preview_out_value=0
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
    --min-seq-len|--min-seq-len=*)
      has_min_seq_len=1
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
    --include-skip-action|--disable-skip-action)
      has_include_skip_action=1
      ;;
    --skip-action-weight|--skip-action-weight=*)
      has_skip_action_weight=1
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
    --dataset-dir)
      has_dataset_dir=1
      expect_dataset_dir_value=1
      ;;
    --dataset-dir=*)
      has_dataset_dir=1
      dataset_dir_value="${arg#*=}"
      ;;
    --embeddings-file)
      has_embeddings_file=1
      expect_embeddings_file_value=1
      ;;
    --embeddings-file=*)
      has_embeddings_file=1
      ;;
    --profiles-file)
      has_profiles_file=1
      expect_profiles_file_value=1
      ;;
    --profiles-file=*)
      has_profiles_file=1
      ;;
    --policy-out)
      has_policy_out=1
      expect_policy_out_value=1
      ;;
    --policy-out=*)
      has_policy_out=1
      policy_out_value="${arg#*=}"
      ;;
    --report-out)
      has_report_out=1
      expect_report_out_value=1
      ;;
    --report-out=*)
      has_report_out=1
      ;;
    --traj-preview-out)
      has_traj_preview_out=1
      expect_traj_preview_out_value=1
      ;;
    --traj-preview-out=*)
      has_traj_preview_out=1
      ;;
  esac
done

if [ "${expect_alignment_mode_value}" -eq 1 ]; then
  echo "[ERROR] '--alignment-mode' requires a value." >&2
  exit 1
fi
if [ "${expect_dataset_dir_value}" -eq 1 ]; then
  echo "[ERROR] '--dataset-dir' requires a directory path value." >&2
  exit 1
fi
if [ "${expect_embeddings_file_value}" -eq 1 ]; then
  echo "[ERROR] '--embeddings-file' requires a file path value." >&2
  exit 1
fi
if [ "${expect_profiles_file_value}" -eq 1 ]; then
  echo "[ERROR] '--profiles-file' requires a file path value." >&2
  exit 1
fi
if [ "${expect_policy_out_value}" -eq 1 ]; then
  echo "[ERROR] '--policy-out' requires a file path value." >&2
  exit 1
fi
if [ "${expect_report_out_value}" -eq 1 ]; then
  echo "[ERROR] '--report-out' requires a file path value." >&2
  exit 1
fi
if [ "${expect_traj_preview_out_value}" -eq 1 ]; then
  echo "[ERROR] '--traj-preview-out' requires a file path value." >&2
  exit 1
fi

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
  if [ "${has_include_skip_action}" -eq 0 ]; then
    set -- --disable-skip-action "$@"
  fi
  if [ "${has_skip_action_weight}" -eq 0 ]; then
    set -- --skip-action-weight 0.10 "$@"
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
    set -- --ugw-init-reg-epochs 500 "$@"
  fi
  if [ "${has_ugw_init_reg_batch_size}" -eq 0 ]; then
    set -- --ugw-init-reg-batch-size 4096 "$@"
  fi
  if [ "${has_ugw_init_reg_lr}" -eq 0 ]; then
    set -- --ugw-init-reg-lr 5e-3 "$@"
  fi
fi

task_dataset_key="$(resolve_dataset_key_from_task)"
case "${task_dataset_key}" in
  ml1m)
    task_dataset_dir="${DATASET_ROOT}/MM-ML-1M-main"
    task_policy_key="ml1m"
    ;;
  amazon_all_beauty)
    task_dataset_dir="${DATASET_ROOT}/Amazon_MM_2018/All_Beauty"
    task_policy_key="amazon_all_beauty"
    ;;
  amazon_magazine_subscriptions)
    task_dataset_dir="${DATASET_ROOT}/Amazon_MM_2018/Magazine_Subscriptions"
    task_policy_key="amazon_magazine_subscriptions"
    ;;
  amazon_mi)
    task_dataset_dir="${DATASET_ROOT}/amazon_v2/Musical_Instruments"
    task_policy_key="amazon_mi"
    ;;
  *)
    task_dataset_key="ml1m"
    task_dataset_dir="${DATASET_ROOT}/MM-ML-1M-main"
    task_policy_key="ml1m"
    ;;
esac

selected_dataset_dir="${task_dataset_dir}"
selected_policy_key="${task_policy_key}"
if [ "${has_dataset_dir}" -eq 1 ] && [ -n "${dataset_dir_value}" ]; then
  selected_dataset_dir="${dataset_dir_value}"
  selected_policy_key="$(infer_policy_key_from_dataset_dir "${selected_dataset_dir}")"
fi

# Prevent sparse datasets from being filtered out by parser default min-seq-len=20.
if [ "${has_min_seq_len}" -eq 0 ]; then
  auto_min_seq_len="$(infer_default_min_seq_len_from_dataset_dir "${selected_dataset_dir}")"
  if [ -n "${auto_min_seq_len}" ]; then
    set -- --min-seq-len "${auto_min_seq_len}" "$@"
    min_seq_note="auto(${auto_min_seq_len})"
  else
    min_seq_note="parser_default"
  fi
else
  min_seq_note="user"
fi

default_embeddings_file="${selected_dataset_dir}/Embeddings/item_embeddings.json"
default_profiles_file="${selected_dataset_dir}/Embeddings/agent_profiles.json"
default_report_out="${selected_dataset_dir}/Embeddings/alignment_pretrain_report.json"
default_traj_preview_out="${selected_dataset_dir}/Embeddings/expert_intent_preview.json"
default_policy_out="${PROJECT_ROOT}/outputs/fast_policies/${selected_policy_key}/fast_inference_policy.pt"

if [ "${has_dataset_dir}" -eq 0 ]; then
  set -- --dataset-dir "${selected_dataset_dir}" "$@"
fi
if [ "${has_embeddings_file}" -eq 0 ]; then
  set -- --embeddings-file "${default_embeddings_file}" "$@"
fi
if [ "${has_profiles_file}" -eq 0 ]; then
  set -- --profiles-file "${default_profiles_file}" "$@"
fi
if [ "${has_report_out}" -eq 0 ]; then
  set -- --report-out "${default_report_out}" "$@"
fi
if [ "${has_traj_preview_out}" -eq 0 ]; then
  set -- --traj-preview-out "${default_traj_preview_out}" "$@"
fi
if [ "${has_policy_out}" -eq 0 ]; then
  mkdir -p "$(dirname -- "${default_policy_out}")"
  set -- --policy-out "${default_policy_out}" "$@"
fi

echo "[Task] dataset=${task_dataset_key}, dataset_dir=${selected_dataset_dir}"
if [ "${has_policy_out}" -eq 0 ]; then
  echo "[Task] policy_out=${default_policy_out}"
else
  echo "[Task] policy_out=provided_by_user"
fi

echo "[Run] alignment-mode=${alignment_mode:-softlabel_ugw}, epochs=${epochs_note}, min-seq-len=${min_seq_note}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/alignment_pretrain_agent.py" "$@"
