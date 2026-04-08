#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"
exec python3 "${PROJECT_ROOT}/Test.py" --model qwen3.5:35b --question "请介绍一下Transformer" --count 100 --timeout 300
