#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PATTERNS=(
  "launch_websim_batch.py"
  "WebSim_agent.py"
  "web_sim_camel_agent.py"
)

STOP_SERVICE_PORT="${STOP_SERVICE_PORT:-0}"

collect_pattern_pids() {
  for pattern in "${PATTERNS[@]}"; do
    pgrep -f "$pattern" 2>/dev/null || true
  done
}

collect_service_pids_by_port() {
  # WebSim default service port.
  local port="19002"

  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null \
      | awk -v p=":${port}" '
          index($4, p) > 0 {
            if (match($0, /pid=[0-9]+/)) {
              pid = substr($0, RSTART + 4, RLENGTH - 4)
              print pid
            }
          }
        ' \
      || true
  fi
}

collect_all_pids() {
  if [ "$STOP_SERVICE_PORT" = "1" ]; then
    {
      collect_pattern_pids
      collect_service_pids_by_port
    } | awk 'NF' | sort -u
  else
    collect_pattern_pids | awk 'NF' | sort -u
  fi
}

TARGET_PIDS=()
while IFS= read -r line; do
  if [ -n "$line" ] && [ "$line" != "$$" ] && [ "$line" != "$PPID" ]; then
    TARGET_PIDS+=("$line")
  fi
done < <(collect_all_pids)

if [ "${#TARGET_PIDS[@]}" -eq 0 ]; then
  echo "[INFO] No running 10k simulation process found."
  exit 0
fi

echo "[INFO] Target PID(s): ${TARGET_PIDS[*]}"

TARGET_PGIDS=()
SELF_PGID="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')"
while IFS= read -r line; do
  if [ -n "$line" ] && [ "$line" != "$SELF_PGID" ]; then
    TARGET_PGIDS+=("$line")
  fi
done < <(
  for pid in "${TARGET_PIDS[@]}"; do
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true
  done | awk 'NF' | sort -u
)

if [ "${#TARGET_PGIDS[@]}" -gt 0 ]; then
  echo "[INFO] Sending SIGTERM to process group(s): ${TARGET_PGIDS[*]}"
  for pgid in "${TARGET_PGIDS[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done
fi

echo "[INFO] Sending SIGTERM to PID(s): ${TARGET_PIDS[*]}"
for pid in "${TARGET_PIDS[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

for _ in $(seq 1 20); do
  ALIVE_PIDS=()
  while IFS= read -r line; do
    if [ -n "$line" ] && [ "$line" != "$$" ] && [ "$line" != "$PPID" ]; then
      ALIVE_PIDS+=("$line")
    fi
  done < <(collect_all_pids)
  if [ "${#ALIVE_PIDS[@]}" -eq 0 ]; then
    echo "[OK] 10k simulation stopped."
    exit 0
  fi
  sleep 0.5
done

echo "[WARN] Escalating to SIGKILL."

ALIVE_PIDS=()
while IFS= read -r line; do
  if [ -n "$line" ] && [ "$line" != "$$" ] && [ "$line" != "$PPID" ]; then
    ALIVE_PIDS+=("$line")
  fi
done < <(collect_all_pids)
if [ "${#ALIVE_PIDS[@]}" -eq 0 ]; then
  echo "[OK] 10k simulation stopped."
  exit 0
fi

ALIVE_PGIDS=()
while IFS= read -r line; do
  if [ -n "$line" ] && [ "$line" != "$SELF_PGID" ]; then
    ALIVE_PGIDS+=("$line")
  fi
done < <(
  for pid in "${ALIVE_PIDS[@]}"; do
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true
  done | awk 'NF' | sort -u
)

if [ "${#ALIVE_PGIDS[@]}" -gt 0 ]; then
  echo "[WARN] Force killing process group(s): ${ALIVE_PGIDS[*]}"
  for pgid in "${ALIVE_PGIDS[@]}"; do
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done
fi

echo "[WARN] Force killing PID(s): ${ALIVE_PIDS[*]}"
for pid in "${ALIVE_PIDS[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done

for _ in $(seq 1 10); do
  ALIVE_PIDS=()
  while IFS= read -r line; do
    if [ -n "$line" ] && [ "$line" != "$$" ] && [ "$line" != "$PPID" ]; then
      ALIVE_PIDS+=("$line")
    fi
  done < <(collect_all_pids)
  if [ "${#ALIVE_PIDS[@]}" -eq 0 ]; then
    echo "[OK] 10k simulation stopped (forced)."
    exit 0
  fi
  sleep 0.3
done

echo "[WARN] Some process(es) are still alive: ${ALIVE_PIDS[*]}"
for pid in "${ALIVE_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "  - PID $pid"
    if command -v ps >/dev/null 2>&1; then
      ps -o pid=,ppid=,pgid=,etime=,command= -p "$pid" 2>/dev/null || true
    fi
  fi
done

exit 1
