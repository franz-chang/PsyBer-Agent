#!/usr/bin/env bash
set -euo pipefail

cd /Users/chongzhang/PsyBer-Agent
exec python3 /Users/chongzhang/PsyBer-Agent/prelearning_agent.py "$@"
