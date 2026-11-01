# CReST-FC implementation and research protocol

This repository implements an executable, staged **research prototype** inspired by the activation reading and gradient-bridge design of [CoSToM](https://github.com/CGCL-codes/CoSToM). It is a fact-checking adaptation, not a reproduction of CoSToM's ToM experiments. The design follows the main [README](../README.md). No pilot metric is a paper result.

## Environment and GPU execution

Use **Python 3.10.15**. The server has NVIDIA driver 535.54.03 and CUDA 12.2; the pinned PyTorch 2.3.0 CUDA 12.1 wheel in `requirements.txt` is compatible with this CUDA 12.x driver. The existing `/data/yangjun/.conda/envs/llama/bin/python` has been verified on an RTX A6000. The host's default Python uses a CUDA 13 wheel and cannot access this driver.

```bash
# Optional fresh Conda environment (installs all requirements)
scripts/install_gpu_env.sh

# Reuse the already-working environment, if desired
export PYTHON_BIN=/data/yangjun/.conda/envs/llama/bin/python
$PYTHON_BIN -m unittest discover -s tests -v
scripts/run_experiments.sh e0 --run-id pilot --limit 20
scripts/run_experiments.sh e1 --run-id pilot --limit 20
```

`run_experiments.sh` checks Python, CUDA and the local model snapshot before launching. Defaults: Qwen2.5-0.5B-Instruct, local Hugging Face cache, BF16, one visible GPU, seed 42 and pilot sample limits. Set `CUDA_VISIBLE_DEVICES=0` to select a GPU. Use `--model-id` for a different cached model and `--unlimited` only for a deliberate larger run. The script reads local model files and sets Hugging Face offline mode.

For the installed `crestfc` environment and GPUs 0–5, run `bash scripts/run_gpu_6.sh pilot study1` first. After reviewing `data/audits/pending_vitaminc_pilot.jsonl`, set the review file in `configs/audit_import.yaml`, import it with `PYTHONPATH=src /data/yangjun/.conda/envs/crestfc/bin/python -m crestfc.cli import-audit --config configs/audit_import.yaml`, then run `bash scripts/run_gpu_6.sh formal study1 data/processed/pilot_reviewed`. Formal mode launches six independent seeds (42–47), one per GPU, with distinct output directories and logs in `outputs/experiment_runs/study1/logs/`. It removes pilot sample limits. The E2 causal-selection and E3 latent-sensitivity gates still apply; a failed seed has a nonzero exit status and its own log. Override `MODEL_ID`, `CACHE_DIR`, or `PYTHON_BIN` in the shell environment if needed.

To evaluate HOVER, PolitiHop, symmetric PolitiHop, FEVER and symmetric FEVER without counterfactual review, run `python scripts/run_priority_datasets.py --run-id priority_full --limit 0` in the `crestfc` environment. The five E0 jobs use GPUs 0–4 and save per-dataset metrics and logs under `outputs/experiment_runs/priority_full/`. The default `--limit 20` is a smoke test using source order, so it is not a balanced or publication-ready estimate. This reads the EGR-FV converted PolitiHop and FEVER datasets; their supplied evidence is marked `retrieved`, not gold/oracle. Raw PolitiHop rating labels remain excluded. This command only runs E0; E2/E3 still need verified relation-change and bias-change pairs, and a clean train/dev split.

The revised E0 classification prompt is `v2_fact_checking_answer` and defines SUPPORTS, REFUTES and NEI explicitly. Run the five datasets with `bash scripts/run_e0.sh e0_prompt_v2` (GPU 0–4, full splits), or `bash scripts/run_e0.sh e0_prompt_v2_smoke 20` for a quick check. The run script scores all three labels; these converted datasets contain only SUPPORTS/REFUTES ground truth, so their three-label macro-F1 is not directly comparable with the earlier two-label E0 run. E1/E2 keep their separate `Relation anchor:` input.

```bash
# Build candidate pairs and the review sheet
scripts/run_experiments.sh prepare --run-id study1

# After human review, import decisions using configs/audit_import.yaml
$PYTHON_BIN -m crestfc.cli import-audit --config configs/audit_import.yaml

# Run all dev stages against reviewed pairs; stops if causal or latent gates fail
scripts/run_experiments.sh all --run-id study1 \
  --reviewed-data data/processed/pilot_reviewed

# Evaluate a held-out test split only after the protocol is fixed
scripts/run_experiments.sh e4 --run-id study1 \
  --reviewed-data data/processed/pilot_reviewed --split test
```

The run artifacts and generated YAML configurations are saved under `outputs/experiment_runs/<run-id>/`. `--reviewed-data` must contain both `samples.jsonl` and `pairs.jsonl`. Without it, `all` runs preparation, E0 and E1, then stops at the human review gate. The default pilot has only train/dev data; a test run requires an added held-out test split. For formal experiments, use reviewed data, several seeds and a cached main model; sample limits and placeholder model layers in configs are for smoke tests.

## Stage map

| README stage | Command | Output | Current scope |
|---|---|---|---|
| Data | `prepare`, `merge`, `validate-data`, `export-audit`, `import-audit` | JSONL samples and pairs, audit | VitaminC, FEVEROUS, HOVER, EXFEVER with conservative label mapping; FEVER indexed evidence with wiki index |
| E0 / direct LoRA | `baseline`, `train-baseline` | predictions, metrics, adapter | Oracle evidence and matched-budget single-model baseline |
| E1 | `extract-activations`, `train-probes`, `plot-probes` | per-layer activations, probe CSV and figure | Linear probe, anchor/claim/evidence positions |
| E2 | `causal-patch`, `select-layers` | bidirectional IE and control scores | Anchor patch; same-input and random-sample controls |
| E3a | `calibrate-decoder` | decoder LoRA, projector, dev metrics | Latent-only labels; optional guided label calibration |
| E3b | `train-steering` | encoder LoRA, gradient audit | `L_rel`, `L_inv`, `L_flip`; reviewed pair gate |
| E4 | `evaluate`, `report` | verdict predictions, pair metrics, null/shuffled ablations, summary | Latent-only or latent-guided verdict classification; optional structured generation with reviewed `gold_justification` |

Not implemented: automatic justification faithfulness evaluation, FEVER/AVeriTeC official retrieval scorers, AVeriTeC four-class ingestion, nonlinear probe, `L_persist`, full-sequence/edited-span patching, multi-seed significance tables, and a paper-ready 7B run. Structured JSON generation and evidence-ID parsing are implemented, but training requires a human-written `gold_justification` for every calibration sample, and justification faithfulness remains a human review task. These require more data, annotation or engineering; outputs must not be presented as completed experimental evidence.

PolitiHop's `half-true` class does not map to FEVER's NEI. EXFEVER's `NOT_SUPPORTED` likewise does not distinguish REFUTES from NEI. The adapter rejects those labels instead of silently relabeling them.

## Run the pilot

```bash
$PY -m crestfc.cli prepare --config configs/prepare_vitaminc.yaml
$PY -m crestfc.cli prepare --config configs/prepare_vitaminc_dev.yaml
$PY -m crestfc.cli merge --config configs/merge_pilot.yaml
$PY -m crestfc.cli validate-data --config configs/pilot.yaml
$PY -m crestfc.cli baseline --config configs/pilot.yaml
$PY -m crestfc.cli extract-activations --config configs/probe.yaml
```

For dev activations, copy `configs/probe.yaml` and set `split: dev`; then run `train-probes` with `activation_dir: outputs/e1_probes/activations`, `output_dir: outputs/e1_probes/pilot` and the desired label list/C grid. `configs/patch.yaml` expects **approved** dev pairs. The example `configs/steering.yaml` expects approved train pairs of both kinds and a calibrated Decoder with nonzero dev latent sensitivity. Formal steering must use `selected_layers_path` from the E2 dev selection output; `allow_manual_layers` is for smoke tests only. It intentionally fails until these gates are met. Layer IDs in the sample configs are placeholders, not selected research findings.

### Human review gate

After `prepare`/`merge`, export candidates:

```yaml
# save as configs/audit_export.yaml
seed: 42
data:
  samples: data/processed/pilot/samples.jsonl
  pairs: data/processed/pilot/pairs.jsonl
output_path: data/audits/pending.jsonl
```

```bash
$PY -m crestfc.cli export-audit --config configs/audit_export.yaml
```

A reviewer sets `approved`, `reviewer`, `reason`, `label_correct`, `evidence_sufficient`, and the relevant `bias_preserves_relation` or `edit_changes_relation`. Then use `import-audit` with `review_path` and a **new** `output_dir`; point subsequent experiment configs at the reviewed copy. Automatic candidates default to `pending` and cannot enter E2 or pair losses. The reviewer should inspect the full source context and minimum edit, not infer correctness from the dataset label alone.

For original FEVER, first create a SQLite index from the official `wiki-pages/*.jsonl`:

```yaml
wiki_dir: /path/to/fever/wiki-pages
output_db: data/interim/fever_wiki.sqlite
```

Run `index-wiki`, then set `source.dataset: FEVER`, `source.path` to an official FEVER JSONL file, and `source.wiki_db` to the SQLite file. Only complete annotated evidence sets enter oracle evidence experiments. NEI is excluded until insufficient retrieved evidence is constructed and audited.

## Scientific interpretation

- Only dev data selects layers, hyperparameters and checkpoints. Test data is held for a fixed final protocol.
- FEVER/VitaminC oracle-evidence accuracy is a classification result, **not** official FEVER Score.
- An E1 probe reading relation information is not proof the model uses it. E2 patching and its controls are required.
- `same_input` patch effects should be near zero. Large effects indicate token/prompt misalignment or a hook error.
- E3 calibration reports whether null latent changes dev predictions. If there is no response, do not interpret later Steering as latent-driven.
- `bias_consistency` can be high for a constant wrong predictor. `bias_pair_correct` and `relation_pair_correct` are the main paired metrics.
- Generated files under `outputs/` and `data/processed/` are ignored by Git. Every CLI run with an output directory saves its resolved config, dependency/hardware manifest and input hashes.

For a direct LoRA comparison, run `train-baseline` with `configs/baseline_lora.yaml`, then set `lora_checkpoint` in an E0 baseline config. Set `include_approved_pairs: true` only after pair review. `plot-probes` and `report` use `configs/plot_probes.yaml` and `configs/report.yaml`.
