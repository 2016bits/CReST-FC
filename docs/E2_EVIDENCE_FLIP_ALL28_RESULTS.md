# VitaminC dev 300 对：Qwen2.5-7B 全 28 层 Evidence Flip 因果扫描

运行目录：`outputs/experiment_runs/e2_evidence_flip_vitaminc_dev_300_all28_4gpu_20261110/`。

## 协议与完整性

- 冻结 `Qwen/Qwen2.5-7B-Instruct`，在正常二分类事实核查 prompt 中评分完整 `SUPPORTS` / `REFUTES` 标签序列。
- 使用 VitaminC dev 的 300 对同 Claim、不同 Evidence、相反来源标签样本；`n_reviewed_pairs=0`，本结果属于探索性开发集分析。
- 在 0–27 共 28 个 block output、`evidence_end` / `relation_anchor` / `answer_pre` 三个位置，进行双向补丁，以及 `cf_donor` / `identity` / `random_donor` / `opposite_direction_donor` 四种模式。
- 四个分片分别完成 80、65、77、78 对，共 300 对；每对 672 条干预，总计 201,600 条。Identity 的最大绝对效应为 0。
- 与先前 11 层实验使用相同的 300 对及 donor 映射；重叠的 79,200 条补丁记录和 clean 分数逐条一致。

原模型 clean 决策在 170/300 对上同时判对两个标签（56.7%）；269/300 对的 SUPPORTS 样本 Margin 高于同 Claim 的 REFUTES 样本（89.7%）。

Margin 定义为 `logP(SUPPORTS) − logP(REFUTES)`。下表的“同 Claim”与“随机”均指朝 donor 标签方向的平均 Margin 改变量；随机 donor 来自不同 Claim/来源家族，但与同 Claim donor 的标签相同。“差值”逐对计算后再平均；同时区间由 300 个独立来源家族的 pair bootstrap 得到，覆盖本次扫描的全部 28×3 个层位比较。

| 层与位置 | 同 Claim | 随机 | 差值 | 差值同时 95% 区间 | 同 Claim 正确方向 | 预测翻转 |
|---|---:|---:|---:|---:|---:|---:|
| L15 Evidence 末尾 | 3.757 | 3.354 | 0.404 | −0.717～1.524 | 77.8% | 10.8% |
| L16 Evidence 末尾 | 4.795 | 4.094 | 0.701 | −0.512～1.913 | 79.8% | 14.2% |
| L17 Evidence 末尾 | 4.770 | 4.165 | 0.604 | −0.528～1.737 | 77.7% | 14.7% |
| L18 Evidence 末尾 | 1.285 | 0.957 | 0.328 | −0.012～0.669 | 74.7% | 4.2% |
| L19 Evidence 末尾 | 0.308 | 0.235 | 0.073 | −0.035～0.180 | 71.5% | 0.0% |
| L19 Relation Anchor | 0.020 | 0.007 | 0.013 | −0.012～0.038 | 52.8% | 0.0% |
| L19 答案前 | 14.400 | 14.000 | 0.399 | −1.282～2.080 | 89.2% | 50.7% |
| L27 答案前 | 19.134 | 18.879 | 0.255 | −1.829～2.338 | 89.3% | 58.3% |

## 解读

1. **有输出因果效应。** Evidence 末尾的 L15–L17 补丁稳定移动模型 Margin，并在约 11%–15% 的方向性干预中改变二分类预测。L16/L17 的两种补丁方向均有正的原始 Margin 改变量；两者对 `REFUTES→SUPPORTS` 的效应明显更大。
2. **尚未定位到相对于随机 donor 稳定且特异的层位。** 同标签随机 donor 在 L15–L17 Evidence 末尾及 L18–L27 答案前产生近似的效应。28×3 个差值的同时 95% 区间没有一个完全大于 0。L18 Evidence 末尾的差值最接近该标准（下界 −0.012），但翻转率仅 4.2%。这不等同于“关系信息不存在”；它限制的是该单点补丁和对照下能够支持的结论。
3. **L19 的高 Probe F1 不是这里的最强局部因果证据。** E1 在另一 prompt 的最终 Relation Anchor 上获得 L19 Full Macro-F1 93.76%；本次 L19 Relation Anchor 几乎不影响决策，L19 Evidence 末尾没有预测翻转。L19 答案前的强效应与随机同标签 donor 相近，符合下游答案信息可被替换的解释。E1 和本实验的 prompt/位置不同，不能把它们当作对同一个状态的直接复现。

当前最值得进一步验证的区域是 **L15–L18 的 Evidence 末尾**，尤其是 L16/L17 的输出敏感性，以及 L18 相对随机 donor 的小幅增量。若目标是选择语义关系 Steering 层，需要在独立样本、人工抽检和更严格的内容匹配对照下复核，并区分中间推理因果作用与答案信息直接传递。

## 复现与文件

```bash
ALLOW_UNREVIEWED=1 MAX_PAIRS=300 GPU_TOKENS=0,1,2,3 \
LAYERS='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27' \
bash scripts/run_e2_evidence_flip.sh e2_evidence_flip_vitaminc_dev_300_all28_4gpu_20261110

PYTHONPATH=src /data/yangjun/.conda/envs/crestfc/bin/python \
scripts/analyze_e2_evidence_flip_specificity.py \
--run outputs/experiment_runs/e2_evidence_flip_vitaminc_dev_300_all28_4gpu_20261110
```

- `analysis/report.json`、`analysis/layer_position_summary.csv`：原始干预的逐层/逐位置汇总。
- `analysis/specificity_report.json`、`analysis/specificity_by_layer_position.csv`：同 Claim 对随机 donor 的逐对比较及同时区间。
- `analysis/causal_effects.jsonl`：201,600 条完整干预记录。
- `analysis/specificity_layer_profile.png`：三个位置的同 Claim、随机 donor、差值曲线。

