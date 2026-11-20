#!/usr/bin/env bash
# Zatrzymuje wrapper (nohup.pid) + kontener(y) docker compose run (dev) — tam jest trening na GPU.
set -euo pipefail

clear

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT}/nohup.pid"

# shellcheck source=_config_/load_config.sh
source "${ROOT}/_config_/load_config.sh"

DEV_RUN_FILTER="name=${PROJECT_NAME}-dev-run"

_stop_wrapper_pid() {
  local pid="$1"
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 1
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in 1 2 3; do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -9 "${pid}" 2>/dev/null || true
  echo "Zatrzymano wrapper (PID ${pid})"
  return 0
}

_stop_training_containers() {
  local -a ids=()
  mapfile -t ids < <(docker ps -q --filter "${DEV_RUN_FILTER}" 2>/dev/null || true)
  if [[ ${#ids[@]} -eq 0 ]]; then
    return 1
  fi
  docker kill "${ids[@]}" >/dev/null
  echo "Zatrzymano kontener(ów) treningu: ${#ids[@]} (${PROJECT_NAME}-dev-run-*)"
  return 0
}

stopped_wrapper=0
stopped_containers=0

if [[ -f "${PID_FILE}" ]]; then
  pid="$(<"${PID_FILE}")"
  rm -f "${PID_FILE}"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && _stop_wrapper_pid "${pid}"; then
    stopped_wrapper=1
  elif [[ "${pid}" =~ ^[0-9]+$ ]]; then
    echo "Wrapper PID ${pid} już nie działał"
  else
    echo "Niepoprawny PID w ${PID_FILE}: ${pid}" >&2
  fi
else
  echo "Brak ${PID_FILE}" >&2
fi

if _stop_training_containers; then
  stopped_containers=1
fi

if [[ "${stopped_wrapper}" -eq 0 && "${stopped_containers}" -eq 0 ]]; then
  echo "Nic do zatrzymania (brak PID i kontenerów ${PROJECT_NAME}-dev-run)." >&2
  exit 1
fi