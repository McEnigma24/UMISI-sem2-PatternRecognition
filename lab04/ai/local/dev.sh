#!/bin/bash
# WSL / host poza Dockerem: uv → build/venv (jak docker-compose i entrypoint).
# Użycie: source local/dev.sh   albo   . local/dev.sh
_LAB03_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${_LAB03_ROOT}/build/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${_LAB03_ROOT}/build/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${_LAB03_ROOT}/build/python}"