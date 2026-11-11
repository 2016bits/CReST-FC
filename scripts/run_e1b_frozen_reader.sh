#!/usr/bin/env bash
# Training-free E1-B. Example: ALLOW_UNREVIEWED=1 MAX_PAIRS=2 LAYERS=0,19 bash scripts/run_e1b_frozen_reader.sh e1b_pilot
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e1b_$(date +%Y%m%d_%H%M%S)}"
PHASE="${PHASE:-dev}"
PAIR_SOURCE="${PAIR_SOURCE:-$ROOT/outputs/experiment_runs/e2_vitaminc_dev_pilot/pairs.jsonl}"
GPU_TOKEN="${GPU_TOKEN:-4}"
MAX_PAIRS="${MAX_PAIRS:-0}"
LAYERS="${LAYERS:-all}"
STAGE="${STAGE:-all}"
args=("$STAGE" --run-id "$RUN_ID" --pair-source "$PAIR_SOURCE"
      --phase "$PHASE" --max-pairs "$MAX_PAIRS" --layers "$LAYERS")
if [[ "${ALLOW_UNREVIEWED:-0}" == 1 ]]; then args+=(--allow-unreviewed); fi
if [[ -n "${REVIEW_FILE:-}" ]]; then args+=(--review-file "$REVIEW_FILE"); fi
if [[ -n "${LOCKED_DEV_SELECTION:-}" ]]; then args+=(--locked-dev-selection "$LOCKED_DEV_SELECTION"); fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_TOKEN"
cd "$ROOT"
exec "$PYTHON_BIN" -m crestfc.frozen_reader "${args[@]}"
