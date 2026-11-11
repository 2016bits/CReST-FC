# E2 Evidence Flip：原模型同层激活补丁

本实验直接回答：冻结 Qwen2.5-7B-Instruct 在正常二分类事实核查过程中，如果把某一层、某一语义位置的 block output 换成 donor 样本在同层同位置的激活，最终的 `SUPPORTS` / `REFUTES` 决策是否朝 donor 标签移动。

实现位于 `src/crestfc/evidence_flip_patch.py`，运行入口是 `scripts/run_e2_evidence_flip.sh`。模型全程 `eval()`、`requires_grad_(False)`，不训练 Probe、不拟合阈值。层号为 zero-based。

## 固定协议

- 数据：同 Claim、不同 Evidence、相反标签的 VitaminC dev 配对；正式结论只使用人工审核通过的 300–500 对。
- 层：默认 `L0 L3 L6 L9 L12 L15 L18 L19 L20 L23 L27`。
- 位置：`evidence_end`（最后一个 Evidence token）、`relation_anchor`（`Relation anchor:` 末 token）、`answer_pre`（chat assistant 开始生成标签前的最后 token）。
- 决策：对完整标签序列计算 teacher-forced log likelihood，定义 `margin = logP(SUPPORTS) - logP(REFUTES)`。
- 双向干预：每一对都执行 `SUPPORTS→REFUTES` 和 `REFUTES→SUPPORTS`。
- `cf_donor`：同 Claim、相反 Evidence 的主干预。
- `identity`：用 target 自己在同一前向几何中捕获的激活覆盖自己，检查 Hook 和数值误差。
- `random_donor`：不同 Claim/来源家族、长度近似、但与主 donor 标签相同的激活。
- `opposite_direction_donor`：不同 Claim/来源家族、与 target 标签相同的激活，即反方向 donor 对照。

对每次补丁，代码计算 donor 方向的有符号 Margin 改变量：

```text
SUPPORTS donor: patched_margin - clean_margin
REFUTES donor:   clean_margin - patched_margin
```

主输出包括均值与配对 bootstrap 95% CI、正确方向比例、预测翻转率、翻到 donor 标签的比例，以及两种补丁方向是否同时正确。`identity` 的效应应接近零。

## 先生成审核队列

```bash
PYTHONPATH=src /data/yangjun/.conda/envs/crestfc/bin/python \
  -m crestfc.evidence_flip_patch review-queue \
  --run-id e2_evidence_flip_review_queue_300 --max-pairs 300
```

编辑生成的 `outputs/experiment_runs/e2_evidence_flip_review_queue_300/review_queue.jsonl`。每条正式纳入的记录必须具有非空 `reviewer`，并将 `approved`、`claim_unchanged`、`evidence_changed`、`supports_label_correct`、`refutes_label_correct`、`evidence_sufficient` 全部设为 `true`。被拒绝的记录保持 `approved: false`。如果最终批准少于 300 对，可以用 `--max-pairs 500` 先生成更大的审核队列。

## 正式运行

```bash
REVIEW_FILE="$PWD/outputs/experiment_runs/e2_evidence_flip_review_queue_300/review_queue.jsonl" \
MAX_PAIRS=300 GPU_TOKENS=0,1,2,3 \
  bash scripts/run_e2_evidence_flip.sh e2_evidence_flip_reviewed_300_4gpu
```

同一命令可续跑，已经完整写入的 pair 结果会复用。若协议、层、数据、模型快照或代码改变，必须换新的 run ID。

## 工程 smoke test

未审核数据只能用于确认张量、位置和日志链路，不能用于研究结论：

```bash
ALLOW_UNREVIEWED=1 MAX_PAIRS=8 GPU_TOKENS=0,1,2,3 LAYERS='0 19' BOOTSTRAP=100 \
  bash scripts/run_e2_evidence_flip.sh e2_evidence_flip_smoke
```

输出目录 `outputs/experiment_runs/<run-id>/analysis/` 包含：

- `clean_baseline.jsonl`：原模型的 clean 决策；
- `causal_effects.jsonl`：每个 pair、方向、层、位置和 donor 对照的原始结果；
- `layer_position_summary.csv`：Margin 改变量、正确方向比例和翻转率；
- `pair_summary.csv`：两方向的逐对汇总；
- `report.json`：候选层-位置排序、审核资格和解释边界；
- `causal_layer_position_profile.png`：主 `cf_donor` 的层-位置曲线。

稳定、双向正确且 Identity 近零的 L19 效应，会使 L19 比单纯依赖 Linear Probe 的 93.76% 更适合作为 Steering 候选。但跨样本补丁仍可能直接传递 Evidence 内容，因此本实验只证明该层/位置对固定输出指标具有因果影响，不能单独证明存在独立、抽象的关系变量。

