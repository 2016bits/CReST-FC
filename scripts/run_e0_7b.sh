#!/usr/bin/env bash
# Frozen Qwen2.5-7B-Instruct E0: generate one verdict label per input.
# Example: LIMIT=5 CONDITIONS=full bash scripts/run_e0_7b.sh e0_7b_smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e0_qwen7b_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-0}"
GPU_TOKENS="${GPU_TOKENS:-1,2,3,4,5}"
CONDITIONS="${CONDITIONS:-full}"
DATASETS="${DATASETS:-hover,politihop,symmetric_politihop,fever,symmetric_fever}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
E1_RUN="${E1_RUN:-$ROOT/outputs/experiment_runs/fever7b_e1_main}"
FEVER_BINARY="${FEVER_BINARY:-0}"
RESUME="${RESUME:-0}"
options=(--run-id "$RUN_ID" --limit "$LIMIT"
         --model-id Qwen/Qwen2.5-7B-Instruct
         --cache-dir /data/yangjun/.cache/huggingface
         --inference-mode generate --max-new-tokens "$MAX_NEW_TOKENS"
         --gpu-tokens "$GPU_TOKENS" --conditions "$CONDITIONS"
         --datasets "$DATASETS" --pin-e1-run "$E1_RUN")
if [[ "$FEVER_BINARY" == 1 ]]; then options+=(--fever-binary); fi
if [[ "$RESUME" == 1 ]]; then options+=(--resume); fi
cd "$ROOT"
exec "$PYTHON_BIN" scripts/run_priority_datasets.py "${options[@]}"
