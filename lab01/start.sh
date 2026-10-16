#!/bin/bash

uv run python src/main.py
uv run python src/visualize.py
uv run python src/classify.py
uv run python src/plot_csv.py
