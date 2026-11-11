# E1-B VitaminC dev 全量 Claim Flip

本实验使用 `data/VitaminC/converted_data/dev_2.json` 的全部 54,012 条 SUPPORTS/REFUTES 样本。冻结 Qwen2.5-7B-Instruct，读取 Encoder L0–L27 的完整序列 Block 输出，分别注入同模型 Decoder Block 0 的前置占位位置。每层用两个完整标签的 teacher-forced log-likelihood 差得到关系分数；没有训练、阈值拟合或测试集选层。

每层分别评价 `full_patch`（Claim + Gold Evidence）与 `claim_only_patch`（仅 Claim）。单样本 Macro-F1 以全部验证集样本为分母。Claim Flip 只在来源修订家族相同、`gold_evidence` 完全相同、Claim 不同且标签相反的样本中计算；每家族按预先固定的字面相似度规则选一对。当前源数据可构造 **11,305 对**，其余样本仍计入单样本 Macro-F1。`claimflip_direction` 是 SUPPORTS Claim 的 `score(S)-score(R)` 大于 REFUTES Claim 的比例，平局计 0.5；`claimflip_paircorrect` 要求两个 Claim 分别被正确判为 SUPPORTS 与 REFUTES。`claim_only_patch` 的同 Evidence 对照可检测 Claim 文本自身是否提供标签线索，但其结果不能自动证明这种线索无效或有害。

```bash
# 8 条样本、2 对的工程检查。
MAX_FAMILIES=2 GPU_TOKENS=4 LAYER_BATCH=7 \
  bash scripts/run_e1b_vitaminc_dev.sh e1b_vitaminc_claimflip_pilot

# 完整验证集；默认使用 5 张 GPU，每张处理互不重叠的来源家族。
GPU_TOKENS=1,2,3,4,5 LAYER_BATCH=7 \
  bash scripts/run_e1b_vitaminc_dev.sh e1b_vitaminc_claimflip_full
```

同一命令可续跑：每个 shard 的 JSONL 按样本追加，已完成样本会跳过；全部 shard 完成后才写 `layer_metrics.csv` 与 `layer_curves.png`。进度见 `outputs/experiment_runs/<run-id>/logs/shard*.log`。`manifest.json` 固定数据源哈希、模型快照、层批量大小和代码哈希。改变任一协议参数必须使用新 run ID。

`layer_metrics.csv` 每层每条件包含 `macro_f1`、`accuracy`、`balanced_accuracy`、类别召回率、输出分布、`claimflip_direction`、`claimflip_paircorrect`、配对分数平局率和平均分数差。`analysis_summary.json` 明确记录来源标签未经人工复核，因此这些结果是描述性验证集分析，不据此正式选层。跨家族重复证据与源标签错误仍需单独人工审核。

批量大小固定为 7，属于实验协议的一部分。小样本对照中，批量 7 与 28 的 Full 分数相同，但部分 Claim-only 分数因 BF16 数值路径不同而变化；不得混合不同层批量的运行结果。该 Reader 与原始 E0 文本生成决策的推断接口不同，层级分数只说明通过指定冻结读取接口可获得的信息。
