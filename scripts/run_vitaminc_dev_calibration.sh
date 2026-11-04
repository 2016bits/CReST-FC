#!/usr/bin/env bash
# Extract a group-independent VitaminC dev subset with frozen Qwen/E1 probes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-counterfactual_e1_calibration}"
GPU_TOKENS="${GPU_TOKENS:-0,1,2,4}"
MAX_PAIRS="${MAX_PAIRS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IFS=',' read -r -a GPUs <<< "$GPU_TOKENS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
"$PYTHON_BIN" -m crestfc.counterfactual_probe check-probes >/dev/null
"$PYTHON_BIN" -m crestfc.counterfactual_probe prepare --dataset vitaminc_dev \
  --run-id "$RUN_ID" --shards "${#GPUs[@]}" --max-pairs "$MAX_PAIRS" >/dev/null
LOG_DIR="$ROOT/outputs/experiment_runs/$RUN_ID/logs"
mkdir -p "$LOG_DIR"
pids=()
for (( shard=0; shard<${#GPUs[@]}; shard++ )); do
  CUDA_VISIBLE_DEVICES="${GPUs[shard]}" "$PYTHON_BIN" -m crestfc.counterfactual_probe extract-shard \
    --dataset vitaminc_dev --run-id "$RUN_ID" --shard "$shard" --batch-size "$BATCH_SIZE" \
    > "$LOG_DIR/vitaminc_dev_shard${shard}.log" 2>&1 &
  pids+=("$!")
  echo "GPU ${GPUs[shard]}: VitaminC dev shard $shard PID ${pids[shard]}"
done
failed=0
for (( shard=0; shard<${#pids[@]}; shard++ )); do
  if wait "${pids[shard]}"; then
    echo "VitaminC dev shard $shard complete"
  else
    echo "VitaminC dev shard $shard failed; see $LOG_DIR/vitaminc_dev_shard${shard}.log" >&2
    failed=1
  fi
done
if (( failed )); then exit 1; fi
"$PYTHON_BIN" -m crestfc.counterfactual_probe evaluate --dataset vitaminc_dev --run-id "$RUN_ID" \
  > "$LOG_DIR/vitaminc_dev_evaluate.log"
