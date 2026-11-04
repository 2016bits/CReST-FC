#!/usr/bin/env bash
# FEVER SUPPORTS/REFUTES Qwen2.5-7B frozen layer probe, GPUs 0-5 by default.
# Example: bash scripts/run_fever7b_e1.sh fever7b_e1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-fever7b_e1_$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then shift; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON_BIN" -m crestfc.fever_probe all --run-id "$RUN_ID" "$@"
