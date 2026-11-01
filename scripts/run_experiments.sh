#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" scripts/run_experiments.py "$@"
