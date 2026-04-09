#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TEMPLATE_DIR="${TEMPLATE_DIR:-$ROOT_DIR/sample_data/templates}"
PROMPT_DIR="${PROMPT_DIR:-$ROOT_DIR/sample_data/instructions}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/outputs/pretrain_timed_run}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/Pretrained-Llama}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-128}"
LR="${LR:-1e-4}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

mkdir -p "$SAVE_PATH"

echo "[pretrain_timed] TEMPLATE_DIR=$TEMPLATE_DIR"
echo "[pretrain_timed] PROMPT_DIR=$PROMPT_DIR"
echo "[pretrain_timed] SAVE_PATH=$SAVE_PATH"
echo "[pretrain_timed] MODEL_PATH=$MODEL_PATH"
echo "[pretrain_timed] EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE MAX_LENGTH=$MAX_LENGTH"

python pretrain_timed.py \
  --template-dir "$TEMPLATE_DIR" \
  --prompt-dir "$PROMPT_DIR" \
  --save-path "$SAVE_PATH" \
  --model-path "$MODEL_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --subset-ratio "$SUBSET_RATIO" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --random-init-if-load-fail

echo "[pretrain_timed] Done. timing json: $SAVE_PATH/timing.json"
