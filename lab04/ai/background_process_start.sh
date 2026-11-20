#!/usr/bin/env bash
# Trening w tle → nohup.out; PID w nohup.pid; na końcu tail -f logu.
set -euo pipefail

clear

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PID_FILE="${ROOT}/nohup.pid"
LOG_FILE="${ROOT}/nohup.out"

# shellcheck source=_config_/load_config.sh
source "${ROOT}/_config_/load_config.sh"

if docker ps -q --filter "name=${PROJECT_NAME}-dev-run" 2>/dev/null | grep -q .; then
  echo "Kontener treningu już działa (${PROJECT_NAME}-dev-run-*) — nowy start pominięty." >&2
  echo "Podgląd: ./background_process_tail.sh  |  stop: ./background_process_cancel.sh" >&2
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Trening już działa (PID ${old_pid}) — nowy start pominięty." >&2
    echo "Podgląd: ./background_process_tail.sh  |  stop: ./background_process_cancel.sh" >&2
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

# Własna sesja — inaczej Ctrl+C przy tail -f trafia w tę samą grupę procesów co docker.
setsid nohup ./shell.sh ./start.sh >"${LOG_FILE}" 2>&1 </dev/null &
echo "$!" >"${PID_FILE}"
disown -h "$!" 2>/dev/null || true
echo "Start w tle (PID $(cat "${PID_FILE}"))."

exec "${ROOT}/background_process_tail.sh"
