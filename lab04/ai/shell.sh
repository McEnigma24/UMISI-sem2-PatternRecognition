#!/usr/bin/env bash
#   ./shell.sh                          — interaktywny kontener dev
#   ./shell.sh uv run python src/main.py
#   ./shell.sh bash ./start.sh            — to samo co ręcznie w kontenerze (nohup: background_process.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 ]]; then
  exec "${ROOT}/scripts/dsh" "$@"
fi

clear
exec "${ROOT}/scripts/dsh"