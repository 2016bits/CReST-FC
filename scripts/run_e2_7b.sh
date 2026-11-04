#!/usr/bin/env bash
# Frozen Qwen2.5-7B E2 causal tracing. Example:
# ALLOW_UNREVIEWED=1 MAX_PAIRS=20 GPU_TOKENS=0,1,2,4 bash scripts/run_e2_7b.sh e2_vitaminc_dev_pilot
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e2_vitaminc_dev}"
PHASE="${PHASE:-dev}"
PAIR_SOURCE="${PAIR_SOURCE:-$ROOT/outputs/experiment_runs/counterfactual_e1_calibration/vitaminc_dev/pairs.jsonl}"
E1_RUN="${E1_RUN:-$ROOT/outputs/experiment_runs/fever7b_e1_main}"
GPU_TOKENS="${GPU_TOKENS:-0,1,2,4}"
MAX_PAIRS="${MAX_PAIRS:-20}"
REVIEW_FILE="${REVIEW_FILE:-}"
DEV_SELECTION="${DEV_SELECTION:-}"
ALLOW_UNREVIEWED="${ALLOW_UNREVIEWED:-0}"
BOOTSTRAP="${BOOTSTRAP:-500}"
MIN_PAIRS="${MIN_PAIRS:-20}"
LAYERS="${LAYERS:-}"
IFS=',' read -r -a GPUs <<< "$GPU_TOKENS"
(( ${#GPUs[@]} > 0 )) || { echo 'GPU_TOKENS is empty' >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

options=(--run-id "$RUN_ID" --phase "$PHASE" --pair-source "$PAIR_SOURCE"
         --e1-run "$E1_RUN" --shards "${#GPUs[@]}" --max-pairs "$MAX_PAIRS")
if [[ -n "$REVIEW_FILE" ]]; then options+=(--review-file "$REVIEW_FILE"); fi
if [[ -n "$DEV_SELECTION" ]]; then options+=(--locked-dev-selection "$DEV_SELECTION"); fi
if [[ "$ALLOW_UNREVIEWED" == 1 ]]; then options+=(--allow-unreviewed); fi
if [[ -n "$LAYERS" ]]; then
  read -r -a layer_array <<< "$LAYERS"
  options+=(--layers "${layer_array[@]}")
fi
"$PYTHON_BIN" -m crestfc.e2_7b prepare "${options[@]}" > /dev/null

LOG_DIR="$ROOT/outputs/experiment_runs/$RUN_ID/logs"
mkdir -p "$LOG_DIR"
pids=()
for ((i=0; i<${#GPUs[@]}; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUs[i]}" "$PYTHON_BIN" -m crestfc.e2_7b run-shard \
    --run-id "$RUN_ID" --shard "$i" > "$LOG_DIR/shard${i}_gpu${GPUs[i]}.log" 2>&1 &
  pids+=("$!")
  echo "shard $i on physical GPU ${GPUs[i]}: PID ${pids[i]}"
done
failed=0
for ((i=0; i<${#pids[@]}; i++)); do
  if wait "${pids[i]}"; then
    echo "shard $i complete"
  else
    echo "shard $i failed: $LOG_DIR/shard${i}_gpu${GPUs[i]}.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

"$PYTHON_BIN" -m crestfc.e2_7b analyze --run-id "$RUN_ID" \
  --bootstrap "$BOOTSTRAP" --min-pairs "$MIN_PAIRS" \
  > "$LOG_DIR/analysis.log"
echo "E2 report: $ROOT/outputs/experiment_runs/$RUN_ID/analysis/e2_report.json"
echo "E3 layer selection: $ROOT/outputs/experiment_runs/$RUN_ID/analysis/e3_layer_selection.json"
