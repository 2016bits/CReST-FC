#!/usr/bin/env bash
# Supplement a completed E1-B VitaminC run with four frozen-reader controls.
# Example: GPU_TOKENS=0,1,2,3,4,5 bash scripts/run_e1b_vitaminc_controls.sh e1b_vitaminc_controls_full
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e1b_vitaminc_controls_full}"
BASE_RUN="${BASE_RUN:-e1b_vitaminc_claimflip_full}"
GPU_TOKENS="${GPU_TOKENS:-0,1,2,3,4,5}"
MAX_FAMILIES="${MAX_FAMILIES:-0}"
SEED="${SEED:-42}"
IFS=, read -ra GPU_LIST <<< "$GPU_TOKENS"
if (( ${#GPU_LIST[@]} < 1 )); then echo 'No GPU tokens supplied' >&2; exit 2; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
"$PYTHON_BIN" -m crestfc.vitaminc_controls prepare --run-id "$RUN_ID" \
  --base-run "$BASE_RUN" --shards "${#GPU_LIST[@]}" --max-families "$MAX_FAMILIES" --seed "$SEED"
run_dir="$ROOT/outputs/experiment_runs/$RUN_ID"
mkdir -p "$run_dir/logs"
pids=()
for shard in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m crestfc.vitaminc_controls scan-shard \
    --run-id "$RUN_ID" --shard "$shard" > "$run_dir/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
  echo "GPU $gpu: shard $shard PID ${pids[$shard]} log $run_dir/logs/shard${shard}.log"
done
failed=0
for shard in "${!pids[@]}"; do
  if ! wait "${pids[$shard]}"; then
    echo "Shard $shard failed; inspect $run_dir/logs/shard${shard}.log" >&2
    failed=1
  fi
done
if (( failed )); then exit 1; fi
"$PYTHON_BIN" -m crestfc.vitaminc_controls analyze --run-id "$RUN_ID"
