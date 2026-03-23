#!/usr/bin/env bash
set -euo pipefail

cd /Users/chongzhang/PsyBer-Agent
exec python3 Test.py --model qwen3.5:35b --question "请介绍一下Transformer" --count 100 --timeout 300
