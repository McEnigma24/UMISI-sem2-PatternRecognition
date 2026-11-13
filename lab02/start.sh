#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync
uv run python src/prepare_dataset.py
chmod +x split_dataset.sh
./split_dataset.sh
uv run python src/main.py
uv run python src/visualize.py
uv run python src/classify.py
uv run python src/plot_csv.py
