# VitaminC frozen Probe score diagnostics

This diagnostic uses the existing frozen FEVER E1 Probes and 18,342 VitaminC test pairs. It also extracts 2,000 **independent VitaminC dev revision families** for scalar decision-threshold calibration. The dev pairs exclude claims present in E1 train/dev and claims present in VitaminC test; source-family overlap between dev and test is zero. No LLM or Probe weights are fitted. The test labels are used only for final evaluation and error analysis, never for threshold selection.

```bash
GPU_TOKENS=0,1,2,4 MAX_PAIRS=2000 bash scripts/run_vitaminc_dev_calibration.sh counterfactual_e1_calibration
PYTHONPATH=src /data/yangjun/.conda/envs/crestfc/bin/python -m crestfc.score_diagnostics --bootstrap 500
```

The outputs are in `outputs/experiment_runs/counterfactual_e1_transfer/score_diagnostics/`: `report.json`, `calibration.json`, `layer_score_distributions.csv`, `vitaminc_score_distributions_L18_L19_L20.png`, and `hard_cases_for_review.jsonl`.

## Paired score decomposition at the FEVER threshold of zero

The signed score is the REFUTES logit. For a valid pair, the REFUTES-side score should exceed the SUPPORTS-side score. Crossing zero in the correct direction means both individual predictions are correct.

| Layer | Correct score direction | Both predictions correct | Correct direction, no correct threshold crossing | Wrong or tied direction |
| ---: | ---: | ---: | ---: | ---: |
| 18 | 91.44% | 50.42% | 41.02% | 8.56% |
| 19 | 91.30% | 50.14% | 41.16% | 8.70% |
| 20 | 90.98% | 49.64% | 41.34% | 9.02% |

At layer 18, 27.17% of all pairs have the correct direction but **both scores are on the REFUTES side**; 13.85% have both on the SUPPORTS side. Thus threshold placement explains much of the gap between correct ordering and pair correctness. The wrong-direction subset cannot be fixed by moving a single intercept, but it is not proof of semantic incapacity: source labels, ambiguity, and hard reasoning cases require separate inspection.

## Distribution shift

At layer 18, median SUPPORTS/REFUTES scores are **−4.14/+4.48** on FEVER dev and **−1.80/+3.40** on VitaminC test. Label-conditional empirical KS distances are 0.322 (SUPPORTS) and 0.163 (REFUTES). Median Evidence length is 259 characters in FEVER dev and 134 in VitaminC test. These descriptive differences support investigating transfer/calibration; they do not establish what causes the shift.

## Dev-only intercept calibration

The 2,000 dev families are split deterministically into 1,000 threshold-fitting and 1,000 holdout pairs. For each layer, the threshold maximizes balanced side accuracy on the fitting half; ties favor the threshold nearest zero. The chosen value is frozen before reading test labels. A group bootstrap resamples the 9,761 VitaminC test source families (up to two pairs each), not individual pairs.

| Layer | Dev threshold | Dev holdout pair correct, zero → calibrated | Fixed test pair correct, zero → calibrated | Test change, source-family 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 18 | +1.119 | 45.90% → 47.50% | 50.42% → 50.81% | −0.22 to +0.98 points |
| 19 | −0.068 | 46.70% → 46.90% | 50.14% → 50.03% | −0.25 to +0.02 points |
| 20 | +0.951 | 48.10% → 46.70% | 49.64% → 50.64% | +0.45 to +1.49 points |

The dev holdout gives only a small layer-18 improvement, essentially no layer-19 gain, and a layer-20 decline. Therefore the test improvement at layer 20 is **not** a valid reason to select or endorse that threshold. The original layer-18 pair-correct rate has a family-bootstrap 95% interval of 49.56%–51.26%, slightly wider than its pairwise Wilson interval of 49.70%–51.14%.

## Source-label audit before claiming reasoning failures

The review queue contains the 30 most negative score shifts for each of layers 18–20. Some examples have source labels that appear inconsistent with their texts. For example, VitaminC test IDs `5ec55991c9e77c000842dbbe_1` and `_2` share the claim that Major Lazer's fourth album was titled *Lazerism*: the SUPPORTS-labeled evidence says *Music is the Weapon*, while the REFUTES-labeled evidence says *Lazerism*. Another pair, `5ed50882c9e77c000848f5ef_1` and `_2`, describes *preceded by* versus *succeeded by* for the same claim but has apparently reversed labels. These are audit candidates, not automatic relabelings. Human review should check the source context and intended temporal semantics before computing an error rate for genuinely complex relations.
