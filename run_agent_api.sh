#!/usr/bin/env bash
set -euo pipefail

cd /Users/chongzhang/PsyBer-Agent
exec /Users/chongzhang/camel/.venv/bin/python /Users/chongzhang/PsyBer-Agent/web_sim_camel_agent.py "$@"
