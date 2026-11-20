#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cleanup_containers() {
  "${ROOT}/scripts/stop_dev_run_containers.sh" 2>/dev/null || true
}

# Ctrl+C / set -e po pipefail — bez trap nie dojdziemy do ręcznego cleanup na końcu
trap cleanup_containers EXIT INT TERM

# Stary run / wiszący compose run — żeby porty 8889/6007 i GPU nie zostały zajęte
cleanup_containers

clear

# pipefail + set -e: po SIGINT unbuffer≠0 i bash by wyszedł przed jakimkolwiek kodem poniżej
set +e
unbuffer ./shell.sh ./start.sh |& tee >(ansi2txt > normal_process.log)
run_status=${PIPESTATUS[0]}
set -e

exit "${run_status}"