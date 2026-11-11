# E1-B 冻结 Reader 运行说明

新 E1 入口是 `src/crestfc/frozen_reader.py` 和 `scripts/run_e1b_frozen_reader.sh`。它使用与 E0/E1 相同的 Qwen2.5-7B-Instruct 权重，所有参数 `requires_grad=False`，只做前向、Hook 和两个完整标签的 teacher-forced 概率计算。不会拟合 Probe、Scaler、LoRA、投影器或阈值。原 `scripts/run_fever7b_e1.sh` 是历史监督 Linear Probe 复现入口；已有 E1.5/E2 产物仍依赖它。

输入 `PAIR_SOURCE` 为 JSONL，每行必须含 `pair_id`、`claim`、`supports` 和 `refutes`。两个成员须有完全相同的 Claim、不同 Evidence、同一 split，以及分别为 `SUPPORTS`/`REFUTES` 的标签。默认要求人工审核；审核 JSONL 的每行需有 `pair_id`、`approved: true`、`reviewer` 和 `claim_unchanged`、`evidence_changed`、`supports_label_correct`、`refutes_label_correct`、`evidence_sufficient` 五个 `true` 字段。`ALLOW_UNREVIEWED=1` 仅允许工程诊断，层选择会保持不合格。

## 先跑 GPU 小样本闭环

```bash
# 本地已有的 VitaminC dev 配对只有来源标签，未人工审核；此命令仅检查代码。
ALLOW_UNREVIEWED=1 MAX_PAIRS=2 LAYERS=0,19 GPU_TOKEN=4 \
  bash scripts/run_e1b_frozen_reader.sh e1b_smoke

# 完整 28 层、20 对工程诊断；不要用该结果作论文选层。
ALLOW_UNREVIEWED=1 MAX_PAIRS=20 GPU_TOKEN=4 \
  bash scripts/run_e1b_frozen_reader.sh e1b_dev_pilot
```

默认配对源为 `outputs/experiment_runs/e2_vitaminc_dev_pilot/pairs.jsonl`。指定其他源时设置 `PAIR_SOURCE=/absolute/path/pairs.jsonl`。可通过 `STAGE=prepare|scan|analyze|all` 单独执行各阶段；重新执行相同命令会复用已完成的单对结果。运行目录在 `outputs/experiment_runs/<run-id>/`，保存 `manifest.json`、`data_audit.json`、`selected_pairs.jsonl`、`token_protocol.json`、`pair_results/`、`predictions.jsonl`、`metrics_by_layer.csv`、`pairs.csv`、`bootstrap_ci.csv` 和 `layer_selection.json`。

## 审核后的开发集

```bash
PAIR_SOURCE=/absolute/path/reviewed_dev_pairs.jsonl \
REVIEW_FILE=/absolute/path/dev_reviews.jsonl \
GPU_TOKEN=4 bash scripts/run_e1b_frozen_reader.sh e1b_reviewed_dev
```

只有所有入选配对经审核、至少 200 对且扫描了全部 28 层，`layer_selection.json` 的 `eligible` 才可能为 `true`。选择规则要求 Full 的 Direction 超过 0.5，并且相对于 Claim-only、Random Donor、No Patch 的按来源家族聚类 PairCorrect 差值 95% 区间下界均大于 0。若无层通过，应保留空选择结果，不得在 test 上调参。

## 独立测试

```bash
PHASE=test PAIR_SOURCE=/absolute/path/reviewed_test_pairs.jsonl \
REVIEW_FILE=/absolute/path/test_reviews.jsonl \
LOCKED_DEV_SELECTION=outputs/experiment_runs/e1b_reviewed_dev/layer_selection.json \
GPU_TOKEN=4 bash scripts/run_e1b_frozen_reader.sh e1b_reviewed_test
```

测试命令拒绝未审核配对，并核对锁定选择的模型快照与扫描层。开发集和测试集还须由研究者按 Claim、来源家族和文档检查独立性；同源配对不能跨 split。

## 解释边界

主指标来自跨层接口 `Encoder Lℓ block output → Decoder L0 block output`。跨层分布可能不匹配，低分不能证明源层没有关系信息。Direct Text 使用同模型的完整标签概率评分；它与此前 E0 的自由生成结果是不同推断协议。历史监督 Probe 的 F1 与新 Reader 的 F1 也不能直接解释为训练效果差值。只有审核过的配对、完整对照和独立测试均完成后，才能报告正式 E1-B 结论。
