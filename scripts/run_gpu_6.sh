#!/usr/bin/env bash
# Usage: bash scripts/run_gpu_6.sh pilot [run-prefix]
#        bash scripts/run_gpu_6.sh formal [run-prefix] [reviewed-data]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pilot}"
PREFIX="${2:-crestfc_$(date +%Y%m%d_%H%M%S)}"
REVIEWED="${3:-$ROOT/data/processed/pilot_reviewed}"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/data/yangjun/.cache/huggingface}"
LOG_DIR="$ROOT/outputs/experiment_runs/$PREFIX/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable missing: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print("PyTorch", torch.__version__, "visible GPUs", torch.cuda.device_count())'

common=(--model-id "$MODEL_ID" --cache-dir "$CACHE_DIR")
case "$MODE" in
  pilot)
    # Candidate construction is CPU work; E0 and E1 each use GPU 0.
    CUDA_VISIBLE_DEVICES=0 PYTHON_BIN="$PYTHON_BIN" "$ROOT/scripts/run_experiments.sh" prepare \
      --run-id "$PREFIX" "${common[@]}" 2>&1 | tee "$LOG_DIR/prepare.log"
    for stage in e0 e1; do
      CUDA_VISIBLE_DEVICES=0 PYTHON_BIN="$PYTHON_BIN" "$ROOT/scripts/run_experiments.sh" "$stage" \
        --run-id "$PREFIX" "${common[@]}" 2>&1 | tee "$LOG_DIR/$stage.log"
    done
    echo "Candidate review: $ROOT/data/audits/pending_vitaminc_pilot.jsonl"
    echo "After human review, import with:"
    echo "  PYTHONPATH=$ROOT/src $PYTHON_BIN -m crestfc.cli import-audit --config $ROOT/configs/audit_import.yaml"
    ;;
  formal)
    [[ -f "$REVIEWED/samples.jsonl" && -f "$REVIEWED/pairs.jsonl" ]] || {
      echo "Reviewed samples.jsonl and pairs.jsonl required in $REVIEWED" >&2
      exit 1
    }
    # One process per GPU. Distinct run IDs avoid output/config collisions.
    pids=()
    for gpu in 0 1 2 3 4 5; do
      run_id="${PREFIX}_gpu${gpu}_seed$((42 + gpu))"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHON_BIN
        "$ROOT/scripts/run_experiments.sh" all --run-id "$run_id" \
          --seed "$((42 + gpu))" --reviewed-data "$REVIEWED" \
          --unlimited "${common[@]}"
      ) >"$LOG_DIR/gpu${gpu}.log" 2>&1 &
      pids+=("$!")
      echo "GPU $gpu: $run_id (PID ${pids[-1]})"
    done
    failed=0
    for gpu in 0 1 2 3 4 5; do
      if ! wait "${pids[$gpu]}"; then
        echo "GPU $gpu failed; see $LOG_DIR/gpu${gpu}.log" >&2
        failed=1
      fi
    done
    exit "$failed"
    ;;
  *) echo "Usage: $0 {pilot|formal} [run-prefix] [reviewed-data]" >&2; exit 2 ;;
esac
