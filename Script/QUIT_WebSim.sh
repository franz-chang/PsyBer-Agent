#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Processes launched by run_10k_sim.sh / run_websim_agent.sh.
PATTERNS=(
  "launch_websim_batch.py"
  "WebSim_camel_agent.py"
  "web_sim_camel_agent.py"
  "run_10k_sim.sh"
  "run_websim_agent.sh"
)

# WebSim default service port.
SERVICE_PORT="${SERVICE_PORT:-19002}"

collect_pattern_pids() {
  for pattern in "${PATTERNS[@]}"; do
    pgrep -f "$pattern" 2>/dev/null || true
  done
}

collect_service_pids_by_port() {
  local port="$1"

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
  {
    collect_pattern_pids
    collect_service_pids_by_port "${SERVICE_PORT}"
  } | awk 'NF' | sort -u
}

build_target_pid_list() {
  local self_pid="$1"
  local parent_pid="$2"
  local line=""
  while IFS= read -r line; do
    if [ -n "$line" ] && [ "$line" != "$self_pid" ] && [ "$line" != "$parent_pid" ]; then
      printf '%s\n' "$line"
    fi
  done < <(collect_all_pids)
}

TARGET_PIDS=()
while IFS= read -r pid; do
  [ -n "$pid" ] && TARGET_PIDS+=("$pid")
done < <(build_target_pid_list "$$" "$PPID")

if [ "${#TARGET_PIDS[@]}" -eq 0 ]; then
  echo "[INFO] No process found for run_10k_sim/run_websim_agent or service port ${SERVICE_PORT}."
  exit 0
fi

echo "[INFO] Target PID(s): ${TARGET_PIDS[*]}"
if command -v ps >/dev/null 2>&1; then
  for pid in "${TARGET_PIDS[@]}"; do
    ps -o pid=,ppid=,pgid=,etime=,command= -p "$pid" 2>/dev/null || true
  done
fi

SELF_PGID="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)"
TARGET_PGIDS=()
while IFS= read -r pgid; do
  if [ -n "$pgid" ] && [ "$pgid" != "$SELF_PGID" ]; then
    TARGET_PGIDS+=("$pgid")
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
  ALIVE=()
  while IFS= read -r pid; do
    [ -n "$pid" ] && ALIVE+=("$pid")
  done < <(build_target_pid_list "$$" "$PPID")
  if [ "${#ALIVE[@]}" -eq 0 ]; then
    echo "[OK] All target processes and service stopped."
    exit 0
  fi
  sleep 0.5
done

echo "[WARN] Escalating to SIGKILL..."

ALIVE=()
while IFS= read -r pid; do
  [ -n "$pid" ] && ALIVE+=("$pid")
done < <(build_target_pid_list "$$" "$PPID")

if [ "${#ALIVE[@]}" -eq 0 ]; then
  echo "[OK] All target processes and service stopped."
  exit 0
fi

ALIVE_PGIDS=()
while IFS= read -r pgid; do
  if [ -n "$pgid" ] && [ "$pgid" != "$SELF_PGID" ]; then
    ALIVE_PGIDS+=("$pgid")
  fi
done < <(
  for pid in "${ALIVE[@]}"; do
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true
  done | awk 'NF' | sort -u
)

if [ "${#ALIVE_PGIDS[@]}" -gt 0 ]; then
  echo "[WARN] Force killing process group(s): ${ALIVE_PGIDS[*]}"
  for pgid in "${ALIVE_PGIDS[@]}"; do
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done
fi

echo "[WARN] Force killing PID(s): ${ALIVE[*]}"
for pid in "${ALIVE[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done

for _ in $(seq 1 10); do
  LEFT=()
  while IFS= read -r pid; do
    [ -n "$pid" ] && LEFT+=("$pid")
  done < <(build_target_pid_list "$$" "$PPID")
  if [ "${#LEFT[@]}" -eq 0 ]; then
    echo "[OK] All target processes stopped (forced)."
    exit 0
  fi
  sleep 0.3
done

echo "[WARN] Some processes are still alive: ${LEFT[*]}"
if command -v ps >/dev/null 2>&1; then
  for pid in "${LEFT[@]}"; do
    ps -o pid=,ppid=,pgid=,etime=,command= -p "$pid" 2>/dev/null || true
  done
fi

exit 1
