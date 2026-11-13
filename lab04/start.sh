#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync
uv run python src/dataset.py --build-index
uv run python src/analyze_dataset.py
uv run python src/visualize.py
