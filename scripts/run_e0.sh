#!/usr/bin/env bash
# Run E0 only on HOVER, PolitiHop, symmetric-PolitiHop, FEVER and symmetric-FEVER.
# Usage: bash scripts/run_e0.sh [run-id] [limit]
# limit=0 evaluates each complete split; e.g. limit=20 is a quick check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e0_prompt_v2_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${2:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/data/yangjun/.cache/huggingface}"
cd "$ROOT"
exec "$PYTHON_BIN" scripts/run_priority_datasets.py --run-id "$RUN_ID" --limit "$LIMIT" \
  --model-id "$MODEL_ID" --cache-dir "$CACHE_DIR"
