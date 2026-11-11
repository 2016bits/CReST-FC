#!/usr/bin/env bash
# Frozen Qwen2.5-7B same-layer Evidence-Flip activation patching.
#
# Reviewed 300-pair run:
#   REVIEW_FILE=/abs/path/vitaminc_reviews.jsonl GPU_TOKENS=0,1,2,3 \
#     bash scripts/run_e2_evidence_flip.sh e2_evidence_flip_reviewed_300_4gpu
#
# Exploratory 8-pair smoke test (all four shards nonempty):
#   ALLOW_UNREVIEWED=1 MAX_PAIRS=8 GPU_TOKENS=0,1,2,3 LAYERS='0 19' \
#     bash scripts/run_e2_evidence_flip.sh e2_evidence_flip_smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-e2_evidence_flip_reviewed_300_4gpu}"
PAIR_SOURCE="${PAIR_SOURCE:-$ROOT/outputs/experiment_runs/counterfactual_e1_calibration/vitaminc_dev/pairs.jsonl}"
SNAPSHOT="${SNAPSHOT:-/data/yangjun/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75}"
REVIEW_FILE="${REVIEW_FILE:-}"
ALLOW_UNREVIEWED="${ALLOW_UNREVIEWED:-0}"
MAX_PAIRS="${MAX_PAIRS:-300}"
GPU_TOKENS="${GPU_TOKENS:-0,1,2,3}"
LAYERS="${LAYERS:-0 3 6 9 12 15 18 19 20 23 27}"
BOOTSTRAP="${BOOTSTRAP:-500}"
SEED="${SEED:-42}"

IFS=',' read -r -a GPUS <<< "$GPU_TOKENS"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_TOKENS is empty" >&2; exit 2; }
if [[ -z "$REVIEW_FILE" && "$ALLOW_UNREVIEWED" != 1 ]]; then
  echo "A reviewed run requires REVIEW_FILE. Set ALLOW_UNREVIEWED=1 only for an exploratory smoke test." >&2
  exit 2
fi
read -r -a LAYER_ARRAY <<< "$LAYERS"
(( ${#LAYER_ARRAY[@]} > 0 )) || { echo "LAYERS is empty" >&2; exit 2; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

prepare=(
  "$PYTHON_BIN" -m crestfc.evidence_flip_patch prepare
  --run-id "$RUN_ID"
  --pair-source "$PAIR_SOURCE"
  --snapshot "$SNAPSHOT"
  --max-pairs "$MAX_PAIRS"
  --shards "${#GPUS[@]}"
  --seed "$SEED"
  --layers "${LAYER_ARRAY[@]}"
)
if [[ -n "$REVIEW_FILE" ]]; then prepare+=(--review-file "$REVIEW_FILE"); fi
if [[ "$ALLOW_UNREVIEWED" == 1 ]]; then prepare+=(--allow-unreviewed); fi
"${prepare[@]}"

RUN_DIR="$ROOT/outputs/experiment_runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
pids=()
for ((shard=0; shard<${#GPUS[@]}; shard++)); do
  gpu="${GPUS[shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m crestfc.evidence_flip_patch run-shard \
    --run-id "$RUN_ID" --shard "$shard" \
    > "$LOG_DIR/shard${shard}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "shard $shard on physical GPU $gpu: PID ${pids[shard]}"
done

failed=0
for ((shard=0; shard<${#pids[@]}; shard++)); do
  if wait "${pids[shard]}"; then
    echo "shard $shard complete"
  else
    echo "shard $shard failed: $LOG_DIR/shard${shard}_gpu${GPUS[shard]}.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

"$PYTHON_BIN" -m crestfc.evidence_flip_patch analyze \
  --run-id "$RUN_ID" --bootstrap "$BOOTSTRAP" \
  > "$LOG_DIR/analysis.log"
echo "Report: $RUN_DIR/analysis/report.json"
echo "Metrics: $RUN_DIR/analysis/layer_position_summary.csv"
echo "Figure: $RUN_DIR/analysis/causal_layer_position_profile.png"

