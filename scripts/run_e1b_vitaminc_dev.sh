#!/usr/bin/env bash
# Full VitaminC dev, all 28 E1-B read layers; same-evidence claim-flip pairs.
# Example: GPU_TOKENS=1,2,3,4,5 bash scripts/run_e1b_vitaminc_dev.sh e1b_vitaminc_dev_full
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e1b_vitaminc_dev_full}"
GPU_TOKENS="${GPU_TOKENS:-1,2,3,4,5}"
MAX_FAMILIES="${MAX_FAMILIES:-0}"
LAYER_BATCH="${LAYER_BATCH:-7}"
IFS=, read -ra GPU_LIST <<< "$GPU_TOKENS"
if (( ${#GPU_LIST[@]} < 1 )); then echo 'No GPU tokens supplied' >&2; exit 2; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
"$PYTHON_BIN" -m crestfc.vitaminc_claimflip prepare --run-id "$RUN_ID" \
  --shards "${#GPU_LIST[@]}" --max-families "$MAX_FAMILIES" --layer-batch "$LAYER_BATCH"
run_dir="$ROOT/outputs/experiment_runs/$RUN_ID"
mkdir -p "$run_dir/logs"
pids=()
for shard in "${!GPU_LIST[@]}"; do
  GPU_TOKEN="${GPU_LIST[$shard]}"
  CUDA_VISIBLE_DEVICES="$GPU_TOKEN" "$PYTHON_BIN" -m crestfc.vitaminc_claimflip scan-shard \
    --run-id "$RUN_ID" --shard "$shard" > "$run_dir/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
  echo "GPU $GPU_TOKEN: shard $shard PID ${pids[$shard]} log $run_dir/logs/shard${shard}.log"
done
failed=0
for shard in "${!pids[@]}"; do
  if ! wait "${pids[$shard]}"; then
    echo "Shard $shard failed; inspect $run_dir/logs/shard${shard}.log" >&2
    failed=1
  fi
done
if (( failed )); then exit 1; fi
"$PYTHON_BIN" -m crestfc.vitaminc_claimflip analyze --run-id "$RUN_ID"
