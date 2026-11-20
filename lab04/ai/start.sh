#!/bin/bash
# Użycie:
#   ./start                    — nowy run (main.py)
#   ./start latest             — wznowienie (SEG_RESUME)
#   ./start saved_models/finetuned/.../run_dir

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -t 1 ]]; then
  clear
fi

export SEG_RESUME="${1:-}"
unset SEG_CHECKPOINT

if [[ -x build/venv/bin/python ]]; then
  exec build/venv/bin/python src/main.py
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run python src/main.py
fi
exec python src/main.py