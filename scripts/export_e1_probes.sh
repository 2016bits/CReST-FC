#!/usr/bin/env bash
# Refit E1 full-condition linear probes from cached activations and serialize them.
# Does not reload or train Qwen2.5-7B. Every refit must reproduce E1 dev predictions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/yangjun/.conda/envs/crestfc/bin/python}"
RUN_ID="${1:-fever7b_e1_main}"
if (( $# > 0 )); then shift; fi
GPU_TOKEN="${GPU_TOKEN:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_TOKEN"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PYTHON_BIN" -m crestfc.fever_probe export-probes \
  --run-id "$RUN_ID" --probe-device cuda:0 "$@"
