# E1-B VitaminC 补充对照

以已经完成的 `e1b_vitaminc_claimflip_full` 为基线，复用其 54,012 条样本的 `Full Patch` 与 `Claim-only` 原始双标签分数。新增运行使用同一 Qwen2.5-7B-Instruct 快照、同一 28 层 Block 输出、Decoder Block 0 写入、7 层一批、相同 `SUPPORTS`/`REFUTES` 完整续写的 log-likelihood；不训练模型、不拟合阈值。

| 条件 | 具体输入或操作 | 层数 |
|---|---|---:|
| `full_patch` | 正确 Claim + Evidence，来自基线 | 28 |
| `claim_only_patch` | Claim + `[not provided]` Evidence，来自基线 | 28 |
| `evidence_only_patch` | `[not provided]` Claim + 正确 Evidence | 28 |
| `random_evidence_patch` | 原 Claim + 其他来源家族的 Evidence；按长度匹配后固定随机选取，同一家族所有目标共用同一 donor | 28 |
| `no_patch_slots` | Full 输入长度个占位 Token + 固定诊断 Prompt，不注入 Hidden States | 无 |
| `direct_text_7b` | 7B 直接读取原 Claim + Evidence，使用二分类文本 Prompt；仍用相同两标签续写评分 | 无 |
| `direct_text_7b_generate` | 同一文本 Prompt，让 7B 贪心生成标签，严格解析完整输出 | 无 |

Random Evidence 的替换证据**没有核实后的新标签**。其 `macro_f1`、`claimflip_direction`、`claimflip_paircorrect` 只是相对于原始标签的诊断性数值，不应当当作新证据条件下的真实任务准确率。它适合与 Full 比较分数变化，检查预测是否依赖原始证据。No Patch 的占位数与逐条 Full Encoder 的 Token 数相等，因此也可显示位置和长度偏置。Direct Text 保持同一基础模型，既保存双标签续写评分，也保存最多 8 Token 的贪心生成原文及严格解析结果；无效生成计入 `INVALID`，不会被猜成某个标签。生成模式没有连续关系分数，因此 Direction 留空。

```bash
# 先做工程检查。MAX_FAMILIES=2 取前两个合格的 Claim Flip 家族。
MAX_FAMILIES=2 GPU_TOKENS=4 \
  bash scripts/run_e1b_vitaminc_controls.sh e1b_vitaminc_controls_generate_smoke

# 完整验证集；6 张卡并行，允许同一命令断点续跑。
GPU_TOKENS=0,1,2,3,4,5 \
  bash scripts/run_e1b_vitaminc_controls.sh e1b_vitaminc_controls_full
```

启动前必须有完整基线运行及全部 shard 完成标记。新增输出保存在 `outputs/experiment_runs/<run-id>/`：`random_evidence_donors.jsonl` 记录 donor 来源和长度，`shards/*.jsonl` 保存原始分数和生成原文，`layer_metrics.csv` 合并条件，`pair_metrics.csv` 记录逐对 margin，`layer_curves.png` 画层曲线，`analysis_summary.json` 记录解释边界。`layer=-1` 表示 No Patch 或 Direct Text 无读取层。完整实验的来源标签未人工审核，全部结论只针对 VitaminC dev 的描述性分析。
