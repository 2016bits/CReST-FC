#!/usr/bin/env bash
# Frozen E1 probe transfer to symmetric-FEVER and VitaminC test pairs.
# Requires the ORIGINAL serialized E1 full-condition probes. No fitting occurs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-counterfactual_e1_transfer}"
E1_RUN="${E1_RUN:-$ROOT/outputs/experiment_runs/fever7b_e1_main}"
E1_PROBE_DIR="${E1_PROBE_DIR:-$E1_RUN/analysis/probes}"
GPU_TOKENS="${GPU_TOKENS:-0,1,2,3}"
MAX_PAIRS="${MAX_PAIRS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IFS=',' read -r -a GPUs <<< "$GPU_TOKENS"
if (( ${#GPUs[@]} < 2 )); then
  echo "GPU_TOKENS must contain at least two physical GPU IDs (one for symmetric-FEVER, others for VitaminC)" >&2
  exit 2
fi
VIT_SHARDS=$(( ${#GPUs[@]} - 1 ))
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

# Fail before GPU allocation if the original E1 Probe parameters are unavailable.
"$PYTHON_BIN" -m crestfc.counterfactual_probe check-probes \
  --e1-run "$E1_RUN" --probe-dir "$E1_PROBE_DIR" >/dev/null

"$PYTHON_BIN" -m crestfc.counterfactual_probe prepare \
  --dataset symmetric_fever --run-id "$RUN_ID" --e1-run "$E1_RUN" \
  --shards 1 --max-pairs "$MAX_PAIRS"
"$PYTHON_BIN" -m crestfc.counterfactual_probe prepare \
  --dataset vitaminc --run-id "$RUN_ID" --e1-run "$E1_RUN" \
  --shards "$VIT_SHARDS" --max-pairs "$MAX_PAIRS"

LOG_DIR="$ROOT/outputs/experiment_runs/$RUN_ID/logs"
mkdir -p "$LOG_DIR"
pids=()
names=()

CUDA_VISIBLE_DEVICES="${GPUs[0]}" "$PYTHON_BIN" -m crestfc.counterfactual_probe extract-shard \
  --dataset symmetric_fever --run-id "$RUN_ID" --shard 0 --batch-size "$BATCH_SIZE" \
  > "$LOG_DIR/symmetric_fever_shard0.log" 2>&1 &
pids+=("$!")
names+=("symmetric_fever_shard0")

for (( i=1; i<${#GPUs[@]}; i++ )); do
  shard=$((i - 1))
  CUDA_VISIBLE_DEVICES="${GPUs[i]}" "$PYTHON_BIN" -m crestfc.counterfactual_probe extract-shard \
    --dataset vitaminc --run-id "$RUN_ID" --shard "$shard" --batch-size "$BATCH_SIZE" \
    > "$LOG_DIR/vitaminc_shard${shard}.log" 2>&1 &
  pids+=("$!")
  names+=("vitaminc_shard${shard}")
done

failed=0
for (( i=0; i<${#pids[@]}; i++ )); do
  if wait "${pids[i]}"; then
    echo "${names[i]} complete"
  else
    echo "${names[i]} failed; see $LOG_DIR/${names[i]}.log" >&2
    failed=1
  fi
done
if (( failed )); then exit 1; fi

"$PYTHON_BIN" -m crestfc.counterfactual_probe evaluate \
  --dataset symmetric_fever --run-id "$RUN_ID" --e1-run "$E1_RUN" --probe-dir "$E1_PROBE_DIR"
"$PYTHON_BIN" -m crestfc.counterfactual_probe evaluate \
  --dataset vitaminc --run-id "$RUN_ID" --e1-run "$E1_RUN" --probe-dir "$E1_PROBE_DIR"
