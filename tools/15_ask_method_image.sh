#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# 默认就是你指定的图和问题
IMAGE_INPUT="autodl-tmp/PsyBer-Agent/Method.jpg"
QUESTION="这张图讲了什么？"

API_POOL_FILE="${PROJECT_ROOT}/settings/api_pool.yaml"
MODEL="${MODEL:-}"
SEED="${SEED:-}"
TIMEOUT="${TIMEOUT:-60}"
SHOW_FULL_RESPONSE=0

usage() {
  cat <<'EOF'
Usage:
  tools/15_ask_method_image.sh [options]

Options:
  -i, --image <path>            图片路径 (默认: autodl-tmp/PsyBer-Agent/Method.jpg)
  -q, --question <text>         问题文本 (默认: 这张图讲了什么？)
  -m, --model <name>            指定模型名 (可选)
      --api-pool-file <path>    API 池配置文件 (默认: settings/api_pool.yaml)
      --seed <int>              固定随机抽取 API 的种子 (可选)
      --timeout <sec>           请求超时秒数 (默认: 60)
      --show-full-response      打印完整响应
  -h, --help                    显示帮助
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -i|--image)
      IMAGE_INPUT="${2:-}"
      shift 2
      ;;
    -q|--question)
      QUESTION="${2:-}"
      shift 2
      ;;
    -m|--model)
      MODEL="${2:-}"
      shift 2
      ;;
    --api-pool-file)
      API_POOL_FILE="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    --show-full-response)
      SHOW_FULL_RESPONSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "${QUESTION}" ]; then
  echo "[ERROR] question 不能为空" >&2
  exit 1
fi

if [ "${API_POOL_FILE#/}" = "${API_POOL_FILE}" ]; then
  API_POOL_FILE="${PROJECT_ROOT}/${API_POOL_FILE}"
fi

if [ ! -f "${API_POOL_FILE}" ]; then
  echo "[ERROR] API pool 文件不存在: ${API_POOL_FILE}" >&2
  exit 1
fi

resolve_image_path() {
  local raw_path="$1"
  if [ -f "${raw_path}" ]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi
  if [ "${raw_path#/}" = "${raw_path}" ] && [ -f "/${raw_path}" ]; then
    printf '%s\n' "/${raw_path}"
    return 0
  fi
  if [ "${raw_path#/}" = "${raw_path}" ] && [ -f "${PROJECT_ROOT}/${raw_path}" ]; then
    printf '%s\n' "${PROJECT_ROOT}/${raw_path}"
    return 0
  fi
  return 1
}

if ! IMAGE_PATH="$(resolve_image_path "${IMAGE_INPUT}")"; then
  echo "[ERROR] 图片不存在: ${IMAGE_INPUT}" >&2
  echo "[HINT] 尝试过: ${IMAGE_INPUT}, /${IMAGE_INPUT}, ${PROJECT_ROOT}/${IMAGE_INPUT}" >&2
  exit 1
fi

if [ -x "/opt/anaconda3/envs/psyber/bin/python" ]; then
  PYTHON_BIN="/opt/anaconda3/envs/psyber/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "${PROJECT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "tools/qwen_multimodal_smoketest.py"
  --image "${IMAGE_PATH}"
  --question "${QUESTION}"
  --api-pool-file "${API_POOL_FILE}"
  --timeout "${TIMEOUT}"
)

if [ -n "${MODEL}" ]; then
  CMD+=(--model "${MODEL}")
fi

if [ -n "${SEED}" ]; then
  CMD+=(--seed "${SEED}")
fi

if [ "${SHOW_FULL_RESPONSE}" -eq 1 ]; then
  CMD+=(--show-full-response)
fi

echo "[Info] Project root: ${PROJECT_ROOT}"
echo "[Info] Python: ${PYTHON_BIN}"
echo "[Info] Image: ${IMAGE_PATH}"
echo "[Info] Question: ${QUESTION}"
echo "[Run] ${CMD[*]}"

"${CMD[@]}"
