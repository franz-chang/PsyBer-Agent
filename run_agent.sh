#!/usr/bin/env bash
set -euo pipefail

# One-click launch for the "degraded" path:
# force OpenAI request failure so the agent falls back to random actions.
cd /Users/chongzhang/PsyBer-Agent
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://127.0.0.1:1"

# exec /Users/chongzhang/camel/.venv/bin/python /Users/chongzhang/PsyBer-Agent/web_sim_camel_agent.py --headless "$@"
exec /Users/chongzhang/camel/.venv/bin/python /Users/chongzhang/PsyBer-Agent/web_sim_camel_agent.py "$@"