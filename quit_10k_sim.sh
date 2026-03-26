#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PATTERNS=(
  "python.*launch_websim_batch.py"
  "python.*web_sim_camel_agent.py"
)

collect_pids() {
  local pattern="$1"
  pgrep -f "$pattern" 2>/dev/null || true
}

ALL_PIDS=""
for pattern in "${PATTERNS[@]}"; do
  MATCHED="$(collect_pids "$pattern")"
  if [ -n "$MATCHED" ]; then
    echo "[INFO] Pattern matched (${pattern}): ${MATCHED//$'\n'/ }"
    ALL_PIDS+=$'\n'"${MATCHED}"
  fi
done

UNIQ_PIDS="$(echo "$ALL_PIDS" | awk 'NF' | sort -u)"

if [ -z "$UNIQ_PIDS" ]; then
  echo "[INFO] No running 10k simulation process found."
  exit 0
fi

echo "[INFO] Sending SIGTERM to: ${UNIQ_PIDS//$'\n'/ }"
kill ${UNIQ_PIDS} 2>/dev/null || true

ALIVE=""
for _ in $(seq 1 16); do
  ALIVE=""
  for pid in $UNIQ_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      ALIVE+=" $pid"
    fi
  done
  if [ -z "${ALIVE}" ]; then
    echo "[OK] 10k simulation stopped."
    exit 0
  fi
  sleep 0.5
done

echo "[WARN] Force killing remaining PID(s):${ALIVE}"
kill -9 ${ALIVE} 2>/dev/null || true
echo "[OK] 10k simulation stopped (forced)."
