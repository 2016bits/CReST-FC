# E1 frozen Probe transfer to symmetric-FEVER and VitaminC

This experiment keeps the original Qwen2.5-7B-Instruct and its 28 `full` condition Linear Probes frozen. It forms one SUPPORTS/REFUTES pair per exact Claim, with different Evidence and opposite source labels. VitaminC pairs must also come from the same revision family. Claims present in the original FEVER E1 train or dev data are excluded. The external splits are symmetric-FEVER `symmetric.json` and VitaminC `test_2.json`.

The prepared data currently contain 238 symmetric-FEVER pairs and 18,342 VitaminC pairs. Their labels come from the source datasets and have not been manually re-verified. Pair preparation is available without a GPU:

```bash
PYTHONPATH=src /data/yangjun/.conda/envs/crestfc/bin/python -m crestfc.counterfactual_probe prepare --dataset symmetric_fever --run-id counterfactual_e1_transfer --shards 1
PYTHONPATH=src /data/yangjun/.conda/envs/crestfc/bin/python -m crestfc.counterfactual_probe prepare --dataset vitaminc --run-id counterfactual_e1_transfer --shards 3
```

## Required frozen Probe artifacts

The original E1 run saved dev predictions and summary metrics in `analysis/checkpoints/layerXX.npz`, but **did not save fitted coefficients or StandardScaler statistics**. To repair this historical omission, the E1 export command refits each `full`, `claim_only`, and `evidence_only` Probe from the already cached FEVER E1 activations using the same scaler and optimizer configuration. It requires **exact agreement on all 6,666 original dev predictions** before saving each Probe. The Qwen model and its activations are unchanged. These are reconstructed Probe parameters, not a byte-for-byte recovery of the vanished optimizer state. New E1 analyses now save Probe parameters during initial fitting.

```bash
GPU_TOKEN=0 bash scripts/export_e1_probes.sh fever7b_e1_main
```

The export is resumable. It writes 84 files to `outputs/experiment_runs/fever7b_e1_main/analysis/probes/` and records hashes in `export_manifest.json` when all 28 layers and three conditions are complete. The transfer experiment uses the 28 `full` files. Export does not rerun LLM forward passes.

Place all 28 files in `E1_PROBE_DIR`, named `layer00_full.npz` through `layer27_full.npz`. Each NumPy archive must contain:

| Key | Shape or value |
| --- | --- |
| `mean` | `(3584,)`, E1 training StandardScaler mean |
| `scale` | `(3584,)`, E1 training StandardScaler scale |
| `coef` | `(3584,)`, original full-condition logistic coefficient; positive score means REFUTES |
| `intercept` | scalar, original full-condition logistic intercept |
| `metadata` | JSON string with fields below |

`metadata` contains `model_id: "Qwen/Qwen2.5-7B-Instruct"`, `layer` (zero-based integer), `condition: "full"`, `read_position: "final_anchor"`, `label_order: ["SUPPORTS", "REFUTES"]`, `e1_run_id: "fever7b_e1_main"`, and `e1_manifest_sha256` equal to the SHA-256 of that E1 run's `manifest.json`. Exported files also record the source checkpoint hash and exact dev-prediction reproduction status. External pair scoring never refits a Probe.

## Full run

```bash
GPU_TOKENS=0,1,2,3 \
bash scripts/run_counterfactual_e1_transfer.sh counterfactual_e1_transfer
```

GPU 0 handles symmetric-FEVER; GPUs 1–3 each handle one VitaminC shard. `GPU_TOKENS` contains physical device tokens and can be changed if a GPU is unavailable. `MAX_PAIRS=20` makes a small pilot with a **new** run ID. Repeating the same command skips completed extraction shards. The model snapshot is pinned to the original E1 manifest, and extraction uses the same `render_encoder(..., "full")` prompt and final anchor position.

The runner writes `pairs.jsonl`, source and pair audits in `manifest.json`, per-shard activation arrays, `analysis/pair_predictions.jsonl`, `analysis/layer_metrics.csv`, and `analysis/evaluation_manifest.json`. Each layer reports accuracy on the SUPPORTS and REFUTES sides, pair-correct rate with a 95% Wilson interval, prediction flip rate, correct score-direction rate, and the REFUTES logit change when Evidence changes. Layers 18–23 are marked as E1-selected; all 28 are reported. Positive signed logit means REFUTES. These paired results can rule out a classifier using only the unchanged Claim, but evidence-only artifacts remain possible; this is still an observational Probe transfer test, not activation-patching causal evidence.
