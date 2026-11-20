#!/usr/bin/env bash
# Kontenery z `docker compose run dev` (np. lab04_seg-dev-run-08e7327…), nie obraz :dev ani lab04_seg-shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_config_/load_config.sh
source "${ROOT}/_config_/load_config.sh"

DEV_RUN_FILTER="name=${PROJECT_NAME}-dev-run"

_stop_ids() {
  local -a ids=()
  mapfile -t ids < <(docker ps -q --filter "${DEV_RUN_FILTER}" 2>/dev/null || true)
  if [[ ${#ids[@]} -eq 0 ]]; then
    return 1
  fi
  docker kill "${ids[@]}" >/dev/null
  echo "Zatrzymano kontener(ów) treningu: ${#ids[@]} (${PROJECT_NAME}-dev-run-*)" >&2
  return 0
}

_rm_stopped() {
  local -a ids=()
  mapfile -t ids < <(docker ps -aq --filter "${DEV_RUN_FILTER}" 2>/dev/null || true)
  if [[ ${#ids[@]} -eq 0 ]]; then
    return 1
  fi
  docker rm -f "${ids[@]}" >/dev/null
  echo "Usunięto zatrzymane kontenery: ${#ids[@]} (${PROJECT_NAME}-dev-run-*)" >&2
  return 0
}

did=0
if _stop_ids; then did=1; fi
if _rm_stopped; then did=1; fi
exit "$((1 - did))"