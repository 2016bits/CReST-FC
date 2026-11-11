# CReST-FC：面向去偏事实核查的因果关系表示增强

> **CReST-FC — Causal Relation Steering for Trustworthy Fact-Checking**  
> **文档性质：可执行研究计划 / Python 工程实现规格（v1.0，2026-10-29）**  
> **状态：研究规格 + 可运行原型；已有小模型烟测，尚无可用于论文结论的正式实验。超参数、预算、阈值均为建议起点。**

推荐论文标题：**Beyond Surface Bias: Causal Relation Steering for Intrinsic Evidence-Grounded Fact-Checking in Large Language Models**。

本 README 是研究规格，明确回答：**要做什么、为何这样做、输入输出格式是什么、用什么代码接口、怎样判断实现正确、结果怎样才能支持研究结论**。目前已实现可运行的 E0–E4 核心原型，具体命令、已验证范围与未完成项见 [实现与复现说明](docs/IMPLEMENTATION.md)。小模型烟测不构成论文实验结果。

---

## 目录

1. [研究问题与关键假设](#1-研究问题与关键假设)
2. [研究边界与实验原则](#2-研究边界与实验原则)
3. [整体架构与实验路线](#3-整体架构与实验路线)
4. [环境、模型及项目结构](#4-环境模型及项目结构)
5. [数据集、标签体系与反事实样本构建](#5-数据集标签体系与反事实样本构建)
6. [统一 Prompt、Tokenizer 和中间表示](#6-统一-prompttokenizer-和中间表示)
7. [实验 E0：数据质量与基础模型诊断](#7-实验-e0数据质量与基础模型诊断)
8. [实验 E1：层级关系定位（Locate）](#8-实验-e1层级关系定位locate)
9. [实验 E2：因果追踪与关系衰减（Trace）](#9-实验-e2因果追踪与关系衰减trace)
10. [实验 E3：关系引导与梯度桥（Steer）](#10-实验-e3关系引导与梯度桥steer)
11. [实验 E4：下游事实核查（Leverage）](#11-实验-e4下游事实核查leverage)
12. [基线、消融实验与公平比较](#12-基线消融实验与公平比较)
13. [评价指标与统计方法](#13-评价指标与统计方法)
14. [建议的 Python 模块接口及关键伪代码](#14-建议的-python-模块接口及关键伪代码)
15. [配置示例、命令行和阶段产物](#15-配置示例命令行和阶段产物)
16. [单元测试、集成测试与防泄漏检查](#16-单元测试集成测试与防泄漏检查)
17. [运行顺序、资源预算与停止条件](#17-运行顺序资源预算与停止条件)
18. [结果报告模板和结论判定](#18-结果报告模板和结论判定)
19. [风险、替代方案与研究贡献](#19-风险替代方案与研究贡献)
20. [参考资料](#20-参考资料)

---

## 1. 研究问题与关键假设

### 1.1 任务定义

给定声明 `C`（Claim）、证据集合 `E={e1,...,en}`（Evidence）和可控的非因果偏差因素 `B`（Bias），模型预测证据相对于声明的关系 `Y`：

\[
Y=R(C,E)\in\{\text{SUPPORTS},\text{REFUTES},\text{NEI}\}.
\]

- **SUPPORTS**：证据足以支持声明。
- **REFUTES**：证据足以反驳声明。
- **NEI**（Not Enough Information）：给定证据不足以做出支持或反驳判断；**不等于声明在客观世界中为假**。
- 对于 AVeriTeC，另有 `CONFLICTING_EVIDENCE`，不与 NEI 合并；见第 5 节。

我们希望模型利用 `C` 与 `E` 的关系做出判断，而不是走 `C -> Y`、`B -> Y` 或单纯关键词匹配的捷径。

### 1.2 四个研究问题

| ID | 研究问题 | 主要实验 | 所需证据 |
|---|---|---|---|
| RQ1 | Claim–Evidence 的关系信息在哪些层、哪些位置可被读取？ | E1 | 分层探针、反事实关系变化下的正确判别 |
| RQ2 | 关系信息在深层衰减，还是仍存在但不再用于输出决策？ | E1 + E2 | 分层可解码性、patching 因果效应、对照分析 |
| RQ3 | 定位敏感层后，Gradient Bridge + 定向训练能否强化关系表示并抑制偏差？ | E3 | 冻结 Decoder 探针、梯度检查、对比损失和层级迁移 |
| RQ4 | 内部关系增强能否提高标准事实核查准确率、反事实鲁棒性和证据忠实性？ | E4 | 准确率、Macro-F1、成对正确率、证据评价与消融 |

### 1.3 可证伪假设

- **H1（存在）**：在至少部分中间激活中，包含可超越 Claim-only 捷径的关系信息。
- **H2a（衰减）**：部分模型的后层关系信息可解码性和/或因果影响弱于前部峰值层。
- **H2b（脱节）**：即使深层仍可解码关系，最终决策也可能受偏差影响；不能把两种现象混为一谈。
- **H3（干预有效）**：因果定位的层优于匹配预算的随机层或固定层训练。
- **H4（真实去偏）**：在固定事实关系、仅变动无关偏差时，输出应稳定且正确；在改变关键证据、关系变化时，输出应相应改变且正确。

**重要**：CoSToM 在 ToM 任务观察到的层级现象，**不是**事实核查任务的预设结论。如果事实关系在中后层才形成、没有显著衰减，按结果调整方法而非强行报告“vanishing relation”。

## 2. 研究边界与实验原则

1. **先机制发现，再方法优化**：E1/E2 在冻结原模型下完成，不能按测试集表现挑层。
2. **Probe 可解码 ≠ 模型实际使用**：必须用干预或等价的因果证据补充 Probe。
3. **输出改变 ≠ 被证据正确引导**：衡量变化是否与真实关系一致，不只看 Flip Rate。
4. **稳定 ≠ 去偏**：一个恒定输出 SUPPORTS 的模型也可能有高一致性；必须同时看成对正确率。
5. **反事实必须保语义**：Bias-changing 样本不得改变事实依据；Relation-changing 样本应保留 Claim，最小改写关键 Evidence。
6. **不使用测试集训练 Probe、Decoder、LoRA 或选择超参数**。
7. **区分 oracle-evidence 与 retrieved-evidence**：给模型提供黄金证据，不等于完成检索式 FEVER/AVeriTeC 任务。
8. **没有训练数据到推理的“标签注入”**：Activation Patching 只用于离线机制诊断；正式推理不能从带有正确答案的样本借用 Activation。
9. **双模型代价透明**：若采用 Encoder + Decoder，报告两份模型实际显存、延迟和吞吐。
10. **原论文与本研究创新分开标注**：CoSToM 的分层读取、冻结 Decoder + Gradient Bridge 是参考；事实核查反事实约束、关系持续性、双重反事实测试是本研究方案。

## 3. 整体架构与实验路线

```text
Raw FEVER / VitaminC / FEVER Symmetric / AVeriTeC
              |
              v
[Data adapter + train/dev/test group split]
              |
              v
[Bias-invariant pairs + Evidence-changing pairs + quality audit]
              |
        +-----+--------------------------+
        |                                |
        v                                v
[E0 baseline]                 [E1 layer-wise probe]
        |                                |
        |                                v
        |                   [E2 bidirectional activation patching]
        |                                |
        |                                v
        |                   [candidate relation-sensitive layers]
        |                                |
        +----------------------+---------+
                               |
                               v
                    [E3a decoder calibration]
                    Frozen encoder; train decoder/interface
                               |
                               v
                    [E3b CReST-FC steering]
                Frozen decoder; train encoder selected LoRA
                L_rel + L_inv + L_flip (+ L_persist)
                               |
                               v
                    [E4 downstream verification]
              Verdict + Evidence IDs + Justification
                               |
                               v
              Main results + Counterfactual robustness
              + Causal restoration + Ablations
```

**最低可行版本（MVP）**：先完成 FEVER/VitaminC 的双类别 SUPPORTS/REFUTES、最后锚点的分层 Linear Probe、双向 Patching 和 `L_rel` 的梯度桥。之后再增加 NEI、多证据、`L_inv`、`L_flip`、跨层保持和真实核查。

---

## 4. 环境、模型及项目结构

### 4.1 建议的工程环境

- Python 3.10 或 3.11；PyTorch 2.x。
- `transformers`、`tokenizers`、`peft`、`accelerate`、`datasets`。
- `numpy`、`pandas`、`scikit-learn`、`scipy`、`pyarrow`、`safetensors`、`pyyaml`、`tqdm`、`matplotlib`、`seaborn`（绘图可选）、`pytest`。
- 尽量保存**实际安装的版本**而非固定使用未经本机验证的版本组合：`pip freeze > outputs/environment/requirements-lock.txt`。
- 模型下载和数据使用遵循来源许可；如使用需要授权的模型，先完成授权。

```bash
# 以下是规划中的本地开发准备命令
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch transformers peft accelerate datasets numpy pandas \
  scikit-learn scipy pyarrow safetensors pyyaml tqdm matplotlib pytest
```

### 4.2 主模型与扩展模型

| 阶段 | 模型 | 目标 |
|---|---|---|
| Smoke test | 小型开放权重 Causal LM（约 0.5B–1.5B） | 验证 Hook、标签评分、反向传播，不用于最终论文结论 |
| Main | `Qwen/Qwen2.5-7B-Instruct` | 分层定位、干预和主实验 |
| Cross-model | `meta-llama/Meta-Llama-3-8B-Instruct`（需遵循许可） | 验证跨架构规律 |

不同模型**分别**做层选择，不要把 Qwen 第 12 层直接视为 Llama 第 12 层。结果可采用层相对深度 `layer_index / n_layers` 作跨模型可视化。

### 4.3 预期代码目录

```text
CReST-FC/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── fever.yaml
│   ├── vitaminc.yaml
│   ├── probe.yaml
│   ├── patch.yaml
│   ├── decoder_calibration.yaml
│   ├── steering.yaml
│   └── evaluate.yaml
├── data/
│   ├── raw/                 # 原始数据（保留来源说明，不要直接提交大文件）
│   ├── interim/             # 规范化中间数据
│   ├── processed/           # 分组切分、对照样本
│   └── audits/              # 反事实审核结果
├── src/crestfc/
│   ├── cli.py
│   ├── config.py
│   ├── seed.py
│   ├── schemas.py
│   ├── data/
│   │   ├── fever.py
│   │   ├── vitaminc.py
│   │   ├── symmetric.py
│   │   ├── averitec.py
│   │   ├── counterfactual.py
│   │   ├── splits.py
│   │   └── validate.py
│   ├── models/
│   │   ├── loader.py
│   │   ├── prompts.py
│   │   ├── token_positions.py
│   │   ├── activations.py
│   │   ├── patching.py
│   │   ├── probe.py
│   │   ├── bridge.py
│   │   ├── lora.py
│   │   └── label_scorer.py
│   ├── training/
│   │   ├── calibrate_decoder.py
│   │   ├── steer_encoder.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── verdict.py
│   │   ├── counterfactual.py
│   │   ├── evidence.py
│   │   ├── causal.py
│   │   └── bootstrap.py
│   └── analysis/
│       ├── layer_profiles.py
│       ├── layer_selection.py
│       └── plots.py
├── scripts/                  # 轻量级入口或作业脚本
├── tests/
│   ├── test_data_schema.py
│   ├── test_group_splits.py
│   ├── test_token_positions.py
│   ├── test_activation_hooks.py
│   ├── test_patching.py
│   ├── test_bridge_gradients.py
│   ├── test_label_scorer.py
│   └── test_metrics.py
└── outputs/
    ├── environment/
    ├── e0_baselines/
    ├── e1_probes/
    ├── e2_causal/
    ├── e3_steering/
    ├── e4_evaluation/
    └── reports/
```

### 4.4 全流程版本与重现实验记录

每次运行保存 `config_resolved.yaml`、Git commit、模型版本/权重修订号、tokenizer 修订号、数据文件 SHA-256、Prompt 模板版本、随机种子、硬件信息、PyTorch/CUDA 版本。训练阶段记录最佳 checkpoint 的**验证集指标**和选择依据；测试集评估只在最终方案固定后执行。

---

## 5. 数据集、标签体系与反事实样本构建

### 5.1 数据集角色分工

| 数据集 | 推荐角色 | 特别注意 |
|---|---|---|
| **FEVER** | 主训练、原始分布测试、三分类 | 原始 `evidence` 是证据文档/句子索引，需连接 Wikipedia 句子库；NEI 通常没有可直接使用的黄金证据 |
| **FEVER Symmetric v0.2** | 偏差诊断、外部挑战测试 | 官方版本包含约 708 对 dev / 712 对 test；核心是 SUPPORTS/REFUTES，不能伪称完整三分类 |
| **VitaminC** | 事实关系反事实对、近似文本变化 | 原始配对需根据具体 revision 字段恢复关系；不可默认每条相邻样本都是同 Claim 的证据翻转 |
| **AVeriTeC** | 真实世界迁移评估 | 原生 4 类（包含 conflicting evidence），不能直接静默映射成 FEVER 的 3 类 |
| **CReST-Bench（自建）** | 统一 `bias_change` 和 `relation_change` 配对 | 为本研究构建，需人工审查最小变更与真实标签 |

建议先使用 **FEVER + VitaminC + 自建小规模人工审查集**跑通流程；FEVER Symmetric 优先作为外部挑战测试，AVeriTeC 作为后期单独协议的迁移评估。

**FEVER 的 NEI 实现提醒**：原始 FEVER 数据仅含无法定位充分黄金证据的信息。不能把 `evidence=null` 当作模型输入为“空证据”后就宣称其具有现实场景的 NEI 检测能力。应显式构建**不充分的检索证据**，验证其确实未支持/反驳 Claim，并记录证据获取策略；在 MVP 中可以先做 SUPPORTS/REFUTES，后续再加 NEI。

### 5.2 统一 JSONL Schema

每行一个模型输入样本，训练/评估按 `group_id` 做分组。建议 Pydantic 或 dataclass 校验。

```json
{
  "sample_id": "g000001_bias_A",
  "group_id": "g000001",
  "pair_id": "g000001_bias",
  "pair_kind": "bias_change",
  "pair_role": "A",
  "source_dataset": "crest_bench",
  "source_id": "orig_000001",
  "split": "train",
  "claim": "A公司成立于2010年。",
  "evidence": [
    {
      "evidence_id": "E1",
      "text": "A公司于2010年正式成立。",
      "doc_id": "D1",
      "sent_id": "0",
      "is_gold": true
    }
  ],
  "claim_metadata": { "claim_publisher": "发布者A" },
  "label": "SUPPORTS",
  "label_space": "fever3",
  "evidence_mode": "oracle",
  "bias_type": "claim_publisher",
  "edited_span": null,
  "counterfactual_verified": true,
  "verification_method": "human_review",
  "source_provenance": { "dataset_revision": "v1" }
}
```

配对样本的另一行：相同 `group_id`、`pair_id`、`claim`、`evidence`、`label`，但 `pair_role=B`、`claim_metadata.claim_publisher=发布者B`。关系改变样本则 `pair_kind=relation_change`，固定 `claim` 和无关元数据，仅改变 `evidence` 中**经过审核**的事实，并更新 `label`。

**字段约束**：

- `sample_id` 全局唯一；`group_id` 是共享原始事实/声明的分组 ID；同组所有扩展样本只能出现在同一个 train/dev/test split。
- `pair_id` 恰好绑定一个有效对照；需要多视图时用多个 `pair_id`，不要把所有变体当作彼此可对照。
- `pair_kind ∈ {none, bias_change, relation_change}`。
- `label ∈ {SUPPORTS, REFUTES, NEI}`，AVeriTeC 的 `CONFLICTING_EVIDENCE` 使用独立 `label_space=averitec4`。
- `evidence_mode ∈ {oracle, retrieved, insufficient}`；使用 `retrieved` 时必须记录检索器版本、检索时间/知识库修订与 rank。
- `edited_span` 记录关系改变的字符级关键片段，以供后续 Token-level Patching。
- `source_provenance` 保存源数据 ID/Revision，避免跨数据集重复与时间泄漏。
- 原始数据保持不可变；变体单独保存，不要覆盖原始 `label` 和证据。

### 5.3 反事实类型 A：Bias-changing（关系应保持不变）

示例：

```text
Claim: A公司成立于2010年。
Evidence: A公司于2010年正式成立。
Claim metadata A: 声明发布者为发布者A。
Claim metadata B: 声明发布者为发布者B。
Gold: SUPPORTS（两个版本一致）
```

设计约束：

1. `B` 表示**无关的声明发布者背景**，不是证据出处或证据权威性信息。
2. 不能写“权威机构已证实此声明”，因为那会增加可能改变证据充分性的事实信息。
3. 可覆盖：Claim 发布者名称占位符、无意义段落格式、无关元信息、同义表述；先验证它们没有改变 Claim/Evidence 的语义。
4. 如果实验考察来源可信度，需明确区分**Claim 的发布者**和**Evidence 的来源**；后者可能是正当信息，不能简单作为需要消除的偏差。
5. 对每对样本记录长度差异、关键 Token 差异；优先选择长度匹配版本（用于可控 Patching），否则使用相同语义锚点而非绝对位置。

目标：

\[
R(C,E,B_A)=R(C,E,B_B)=Y.
\]

### 5.4 反事实类型 B：Relation-changing（关系应随证据翻转）

```text
Claim: A公司成立于2010年。
Evidence +: A公司于2010年正式成立。 => SUPPORTS
Evidence -: A公司于2012年正式成立。 => REFUTES
Evidence 0: A公司总部位于北京。   => NEI（仅此证据不足）
```

要求：

1. 优先**固定 Claim，最小改写 Evidence 的数字、实体、日期或否定结构**；不要同时改 Claim 与 Evidence。
2. 重新标注变化后的 Claim–Evidence 关系，不能仅凭编辑动作自动判定标签。
3. 避免证据存在冲突、时间歧义或别名歧义；若无法确定，剔除或标记待人工复核。
4. 多证据场景下需确认修改后所有 Evidence 的联合结论，不能只检查一个句子。
5. 对 NEI 的转换要标明**证据不足**而不是错误事实；建议独立统计 S↔R、S↔NEI、R↔NEI。

目标：

\[
R(C,E^+)\ne R(C,E^-),\qquad Y^+\ne Y^-.
\]

**建议创建两张表**：`samples.jsonl`（逐样本）和 `pairs.jsonl`（`pair_id, sample_a, sample_b, kind, changed_fields, audit_status`），避免将测试数据的对照关系隐式绑定在文件顺序上。

### 5.5 数据划分、防泄漏与审核

推荐初始目标：3,000–5,000 个基础单元，每单元 2–4 个对照版本；其中 500–1,000 个高质量单元由人工复核。数字仅用于预算，质量优先。

- **先按原始单元/实体/文章/revision family 分组切分，再生成变体**；建议在可控自建数据中采用 train/dev/test = 70/15/15。
- FEVER 官方测试协议保持原样。不可重分官方 blind test 来参与训练。
- `group_id` 防止同 Claim 的对照对跨 split；`source_id`/Wikipedia 页面或 revision family 进一步帮助排除近重复。
- 跨 FEVER/VitaminC 的可能重合样本，采用标准化文本哈希及高相似度检测，并进行人工复核。
- 自建评估集在开发中冻结版本 `crest_bench_v1`；每次修改记录语义审核理由。
- 标注时至少记录 `label_correct`、`bias_preserves_relation`、`edit_changes_relation`、`evidence_sufficient` 四项审核结果；不合格样本不进入核心测试。
- 记录各标签、偏差类型、修改类型、证据长度、实体、来源的数量分布。训练集可平衡，测试集同时保留自然分布与平衡子集。

**NEI 与 AVeriTeC 处理原则**：

- `fever3` 主实验如含 NEI，需要经过证据充分性验证；未验证的自动 NEI 不计入核心结论。
- `averitec4` 另配四类标签评分器和四分类评估；若只做可比的三类子集，应在实验表中明确写 `AVeriTeC-3 (conflicting excluded)` 并报告排除数量。
- FEVER Symmetric 官方对照中有些样本会改 Claim，不能不经检查直接当作“固定 Claim、改 Evidence”的 Relation-changing 对。

---

## 6. 统一 Prompt、Tokenizer 和中间表示

### 6.1 Encoder 输入模板（严禁泄漏 gold label）

```text
You are reading a fact-checking example.
Claim: {claim}
Evidence:
[E1] {evidence_1}
[E2] {evidence_2}
Claim metadata: {claim_metadata_or_none}
Relation anchor:
```

- `Relation anchor:` 是稳定的**最后位置读出锚点**，模型在此结束 Encoder 前向传播；不要给它接上 `SUPPORTS` 等答案。
- 可以使用模型自带 chat template，但**必须固定** `template_version`、system message 与 tokenizer revision。Prompt 不能在 train 与 test 发生非预期变动。
- 主实验记录四类读取位置：`final_anchor`、`claim_end`、`evidence_end`、`edited_evidence_span_mean`；后两类用 offset mapping 找位置。
- 不要把 `padding` Token 误认为最后一个真实 Token；按 `attention_mask` 获取每个样本的最后有效位置。
- `evidence_end` 对多证据需定义为最后一段证据末尾，同时可增加逐证据位置和池化读出。

### 6.2 标签与概率评分

**不能假设** `SUPPORTS`、`REFUTES`、`NEI` 各对应一个 tokenizer Token。使用以下两种之一：

1. 直接按有限候选答案对 Decoder 做 **teacher-forced sequence log-likelihood**：

   \[
   s_y(x)=\sum_{t=1}^{|T(y)|}\log P(T_t(y)\mid x,T_{<t}(y)).
   \]

   保持模板和答案前缀一致；是否使用长度归一化应在 dev 上选定并在所有模型中固定。标签多 Token 时不能仅比较第一个 Token 的 Logit。
2. 使用明确训练过的 **3/4 分类 Head** 评估内部表示；这种 Head 可作为 Probe，但其分类结果不要假称为原始 LLM 生成性能。

对于生成结果，约束输出为预定义标签和 JSON 字段，并统一解析：`SUPPORTS/REFUTES/NEI`，解析失败记录 `INVALID`，不得擅自映射到某一类。

### 6.3 Activation 约定

将第 `ℓ` 个 Transformer Block 的**输出 residual stream**记为：

\[
H^{(\ell)}\in\mathbb R^{B\times T\times d},\quad
z_{\ell,p}=\operatorname{Pool}(H^{(\ell)},p).
\]

- `ℓ` 在代码中使用 **0-based block index**：范围 `0 ... n_layers-1`。
- Hugging Face 的 `output_hidden_states[0]` 通常是 embedding；`hidden_states[ℓ+1]` 对应第 `ℓ` 个 block 输出；以实际模型验证，**不得默认任何模型完全一致**。
- 主要 Hook 放在 `model.model.layers[ℓ]` 模块输出；注意有些返回 Tensor，有些返回 Tuple，需保留其他返回项。
- 保存 `model_id`、`block_index`、`read_position`、`token_index`、`prompt_version`、`dtype`、`normalization`；未训练时的 Activation 可以离线缓存。
- 优先使用 `bfloat16` 模型前向、`float32` 聚合/Probe 训练；跨层激活的向量范数不同，应报告归一化策略。

---

## 7. 实验 E0：数据质量与基础模型诊断

**目标**：确保后续差异确实与 Evidence 使用和偏差有关，而不是数据问题或标签解析问题。

### E0.1 输入条件

| 条件 | 输入 | 作用 |
|---|---|---|
| `claim_only` | 仅 Claim | 测试 Claim-only Label Shortcut |
| `evidence_only` | 仅 Evidence | 测试 Evidence-only 词汇捷径 |
| `full` | Claim + Evidence | 标准核查 |
| `full_bias` | Claim + Evidence + 无关 Claim 元信息 | 测试偏差干扰 |
| `evidence_shuffled` | Claim + 非匹配 Evidence | 检查对无关证据的敏感性（标签需重新确定） |
| `no_evidence` | Claim + 显式“未提供证据” | 仅作为诊断，不等同 FEVER 官方 NEI |

统一 Temperature=0（或严格的候选标签打分），记录 `accuracy`、`macro_f1`、`invalid_rate`、类别混淆矩阵、反事实配对指标。

### E0.2 基本 Sanity Checks

- 如果 Claim-only 表现接近 Full，**只能提出可能存在捷径**，不能仅据此断言模型不使用证据。
- 对关系变化对检查正确翻转率；对偏差变化对检查双样本都正确率。
- 按标签与编辑类型抽样检查错误：时间事实、实体替换、否定、数字、证据冲突、NEI。
- **阶段验收**：数据 schema 通过、配对关系与 split 无泄漏、标签评分能够正确处理多 Token、示例在人工阅读下成立。

产物：`outputs/e0_baselines/predictions.jsonl`、`metrics.json`、`confusion_matrix.csv`、`audit_report.md`。

---

## 8. 实验 E1：层级关系定位（Locate）

当前 E1 采用 [E1-B 冻结 Activation Reading 方案](E1-B-readme.md)：冻结 Qwen2.5-7B-Instruct 的全部参数，不拟合 Linear Probe、StandardScaler、LoRA、投影器或决策阈值。历史线性 Probe 的代码和结果仍保存在独立入口，供复现实验与 E1.5/E2 已有分析使用；其结果不得与新 E1-B 混写为同一种读出方法。

对同 Claim、不同 Evidence 且标签翻转的人工审核配对，依次输入 Full、Claim-only 和 Evidence-only 上下文。一次 Encoder 前向捕获 28 个 Transformer Block 的**全部有效 Token 输出**；对每个读取层，将其覆盖到同权重冻结 Decoder 的 Block 0 输出所对应的前置占位位置。Decoder 的文字输入只有固定诊断问题，不含原始 Claim、Evidence 或 Gold Label。

两个完整标签 `SUPPORTS` / `REFUTES` 使用 teacher-forced 序列 log-likelihood 求和；分数差作为未经训练或校准的关系 Margin。主对照包括同模型 Direct Text、相同长度的 No Patch、Claim-only、Evidence-only、长度邻近的独立 Random Donor，以及同 Claim 反事实 Donor。每层报告单样本 Macro-F1、Balanced Accuracy，以及固定 Claim 配对的 Direction、Tie Rate、PairCorrect 和按来源家族聚类的 Bootstrap 区间。

开发集只用于定位层。只有经人工审核、达到预定样本量且完成全部 28 层扫描时，才允许根据 Full 相对负对照的配对差值选择层；测试集必须读取锁定的开发集选择结果。无训练 Reader 的高分说明冻结模型经指定接口可读取关系信息，并不能单独证明该层处于原模型自然生成的因果决策路径；这一点仍由 E2 检验。运行方法见 [E1-B 实现说明](docs/E1B_FROZEN_READER.md)。

VitaminC 完整验证集的 Full/Claim-only 层扫描见 [Claim Flip 实验说明](docs/E1B_VITAMINC_CLAIMFLIP.md)；Evidence-only、Random Evidence、No Patch、Direct Text 及贪心生成补充对照见 [控制实验说明](docs/E1B_VITAMINC_CONTROLS.md)。

同 Claim、相反 Evidence 配对的人工审核可使用 [离线审核页面](outputs/experiment_runs/e2_vitaminc_review_queue_300/review.html)，操作和导出格式见 [审核页面说明](docs/VITAMINC_REVIEW_UI.md)。

---

## 9. 实验 E2：因果追踪与关系衰减（Trace）

### E2.1 Causal Patching 的基本设置

对同一个 Claim 构造关系改变的对照：

\[
x_A=(C,E_A),\quad Y_A=\mathrm{SUPPORTS};
\qquad x_B=(C,E_B),\quad Y_B=\mathrm{REFUTES}.
\]

- **Clean forward**：运行 `x_A`，保存候选层/位置 Activation `z_A`。
- **Counterfactual forward**：运行 `x_B`，得到正常输出。
- **Patch forward**：运行 `x_B` 时，在第 `ℓ` 层将**指定 Token/片段**的激活替换成 `z_A`，继续前向生成或打分。
- **Reverse**：运行 `x_A` 时注入 `z_B`，双向重复，避免单向效果偏置。

要注意：**关系改变的 `x_B` 本身是另一条有正确标签的输入**；Patching 故意把它朝 `Y_A` 拉动，只用于诊断 `z_A` 对决策的因果影响，绝不能把这种输出视为真实核查任务的正确结果。

### E2.2 定义指标（支持多 Token 标签）

对候选标签集合 \(\mathcal Y\)，用统一的序列分数 `s_y(x)` 定义：

\[
m_y(x)=s_y(x)-\log\sum_{y'\ne y}\exp s_{y'}(x).
\]

关系因果效应：

\[
IE^{rel}_{\ell,p}(A\to B)
=m_{Y_A}(x_B;\operatorname{patch}_{\ell,p}(z_A))
-m_{Y_A}(x_B).
\]

再计算 \(B\to A\)；报告均值、中位数、正效应比例、95% bootstrap CI，及修补后模型实际是否预测源侧标签。**不只报告被 patch 后的原始输出类别改变比例**。

可选标准化：

\[
NIE=\frac{IE^{rel}}{m_{Y_A}(x_A)-m_{Y_A}(x_B)+\epsilon},
\]

但分母可能接近 0 或为负；必须记录失效比例，主结果以未经标准化的 Logit-Margin Delta 为准。

### E2.3 Bias Patching

固定 `(C,E,Y)`，仅改变 Claim 元数据 `B_A/B_B`：

\[
x_{unbias}=(C,E,B_A),\quad x_{bias}=(C,E,B_B).
\]

对 `x_bias` 注入 `x_unbias` 的对应激活：

\[
IE^{biasfix}_{\ell,p}
=m_Y(x_{bias};\operatorname{patch}_{\ell,p}(z_{unbias}))
-m_Y(x_{bias}).
\]

- 按 `x_bias` 是否原本错误分别报告；在原本正确的样本上不应强行期望正效应。
- 若 `x_bias` 错且 `x_unbias` 对，关注 patch 能否恢复正确标签。
- `IE_biasfix` 与 `IE_rel` 分开报告，不能随意相减解释为“纯去偏因果效应”。

### E2.4 Patch 位置、粒度和位置对齐

| 粒度 | 推荐处理 | 风险 |
|---|---|---|
| `final_anchor` | MVP 首选，相同模板锚点的相应位置 | 若输入长度不同，位置编码/上下文长度可能混淆 |
| `claim_end` | 作为对照 | 是否能看到全部 Evidence 取决于序列顺序；若 Claim 在前，其层状态不能读取后面的 Evidence |
| `evidence_end` | 辅助诊断 | 长度不一致，需语义位置匹配 |
| `edited_evidence_span` | 更细粒度分析 | 数字/日期可能拆成多个 Token，不能按绝对位置复制 |
| `full_residual_seq` | **仅严格长度匹配样本**使用 | 容易把大量非关系信息一起复制，解释性较弱 |

**关键的自回归注意事项**：如果 Prompt 按 `Claim -> Evidence -> Metadata -> Relation anchor` 排列，`claim_end` 无法看到后续 Evidence。正式“关系表示”结论优先来自 `final_anchor` 或能看到两者内容的位置；如果要研究 Claim token 的关系激活，需用重排模板并作为独立实验。

实现时严格记录：源位置、目标位置、两者是否同序列索引、Token span 长度、模板差异。对于不对齐的编辑片段采用聚合向量写入预设 `anchor`，**不要把不等长 Token span 直接广播成等价 Patch**。

### E2.5 关键负对照

1. **Same-input patch**：从同一输入相同层恢复同一激活，输出应数值接近原始结果。
2. **Random sample patch**：随机不相关样本的同层同位置激活；控制任意激活替换造成的扰动。
3. **Random layer patch**：同预算的随机层。
4. **Wrong-relation patch**：来自并不对应目标标签的样本。
5. **Non-relation token patch**：例如无关元数据末尾位置。
6. **Bidirectional**：A→B 与 B→A 的符号与效果均需审查。
7. **Length-matched subset**：排除仅由 Token 长度/位置引起的假阳性。

### E2.6 关系衰减的判定

同时观察：

- `Linear Probe` 随层变化；
- `MLP Probe` 随层变化；
- `IE_rel` 的因果作用随层变化；
- 模型最终关系预测和 Bias Flip 行为。

按证据区分：

| 现象 | 可支持的结论 | 不应直接断言 |
|---|---|---|
| 线性 Probe 下降，MLP 不下降 | 表示编码形式改变，线性可读性下降 | 表示“消失” |
| 两种 Probe 都下降，IE 也下降 | 与“表示可访问性衰减”一致 | 严格信息论意义的完全消失 |
| Probe 仍高，但最终判断随 Bias 改变 | **Evidence–Decision Misalignment** | “模型不懂证据” |
| IE_rel 在少数层显著高于对照 | 候选因果敏感层 | 单层是唯一因果机制 |

### E2.7 层选择（避免测试集泄漏）

只在 dev 的 `patch_pairs` 上计算一个预先登记的选择分数，例如：

\[
S_{\ell}=\operatorname{mean}IE^{rel}_{\ell}
+\eta\operatorname{mean}IE^{biasfix}_{\ell,\,\text{biased-wrong}}
-\rho\operatorname{mean}|IE^{random}_{\ell}|.
\]

只在具备足够样本且各项可估时计算；`η,ρ` 在 dev 上确定。或采用更稳健的预注册两阶段规则：先要求 `IE_rel` 明显超过随机 Patch，再按 `IE_biasfix` 和 Layer Probe PairCorrect 排序。推荐 `top_k ∈ {1, 2, 4}`；同时保留同预算随机层和相同深度间隔层作为控制。

产物：`causal_effects.parquet`、`causal_heatmap.png`、`candidate_layers_final.json`（包含验证集选择规则）。

---

## 10. 实验 E3：关系引导与梯度桥（Steer）

### E3.1 两种实现层级：MVP 与完整版

**MVP（便于先验证）**：冻结主模型，使用 `nn.Linear` Probe Head 对候选层的 Hidden State 计算 `L_rel + L_inv + L_flip`，只训练候选层 LoRA。它能测试“内部关系导向监督是否有效”，但**不等同于 CoSToM 式 LLM Decoder Gradient Bridge**，论文中必须单独命名为 `CReST-lite`。

**正式 CReST-FC**：使用 **Context Encoder + Probe/Generation Decoder + Activation Interface**。Decoder 先校准，随后冻结；梯度从 Decoder 的标签损失，经注入的 Activation 回到 Encoder 所选 LoRA。下面以正式版本为准。

### E3.2 激活写入接口：避免随机注入和标签泄漏

建议采用 `latent_slot` 设计：

1. Encoder 输入完整的 Claim、Evidence 和 Claim 元数据，读取定位得到的 `z_ℓ`。
2. Decoder Prompt 在**问题之前**预留一个固定的 `LATENT_SLOT` 位置，以确保自回归注意力可以读取它。
3. 在 Decoder 固定的 `write_layer`，通过 Hook 将 `LATENT_SLOT` 对应 Hidden State 替换为 `P_ℓ(z_ℓ)`，这里 `P_ℓ: R^{d_enc} -> R^{d_dec}` 是经训练校准的投影器。
4. 之后继续正常 Decoder 前向计算。`P_ℓ`、写入位置、`write_layer` 对训练和评估保持一致。
5. Decoder 的**关系诊断 Prompt**仅包含 latent slot 和关系问题，**不包含原始 C/E 或 gold 标签**，确保关系读出必须来自 Activation；生成核查 Prompt 可以包含 C/E，以便输出证据 ID 和解释，但必须做 `null_latent` 消融检查是否真的使用潜在向量。

示意：

```text
Encoder: [Claim + Evidence + Claim metadata + Relation anchor]
                  |
                  v
           h_enc[layer=ell]
                  |
                  v
           pool -> P_ell(z)
                  |
                  v
Decoder: [LATENT_SLOT] [Question / Verification prompt]
                  |
            overwrite hidden state at fixed write_layer
                  |
                  v
            output label / label + evidence + justification
```

如果 Encoder 与 Decoder 不同维度，必须引入 `P_ℓ`；即使维度相同也建议校准接口。**不要把一份未知来源的中间激活直接塞到随机层，然后把读取失败解释为“模型没编码关系”。**

### E3.3 E3a：Decoder Calibration（先训练，再冻结）

训练数据使用 `train_decoder` 子集：建议从官方 training split 内按 `group_id` 划出，用于校准 Decoder/interface；另设不重合的 `train_steer` 子集用于下一阶段。更节省数据的高级方案是 K-fold cross-fitting，但首次实现不必引入。

- 固定预训练 Encoder 参数，并用 `torch.no_grad()` 获取 Encoder activation（**此阶段可以 detach**）。
- 可训练：Decoder 的轻量 LoRA + `P_ℓ`，以及必要的 latent slot embedding；保持其它权重冻结。
- 训练任务 A：`latent_only_relation`，仅从 Activation 输出关系标签，保证 Decoder 的读出能力。
- 可选任务 B：`latent_guided_verify`，使用 C/E 文本 + latent 输出 `verdict/evidence_ids/justification`，以减少下游 Prompt 分布偏移。
- 对任务 B 同时训练**latent 被屏蔽/随机替换**的对照或在校准验证集检查其影响，避免 Decoder 完全忽略 latent。
- E3a 结束后冻结 Decoder、`P_ℓ` 和接口。选取 checkpoint 依据 `dev_decoder` 的关系准确率和潜在向量敏感性，**不能**使用正式最终测试集。

校准后应当通过三项单元测试：同一 Prompt 的不同有效关系 latent 会改变关系分布；相同 latent 重复前向结果一致；随机/null latent 性能显著不同于正确 latent（如无差异则继续优化接口）。

### E3.4 E3b：Gradient Bridge Steering（核心训练）

仅训练 Context Encoder 指定层的 LoRA；Decoder 和 `P_ℓ` 全冻结。

\[
z_\ell=E_{\theta}^{(\ell)}(C,E,B),\quad
p_{\phi}(Y\mid z_\ell,q)=D_{\phi}(q;\operatorname{Inject}(P_\ell(z_\ell))).
\]

损失反传路径：

```text
loss -> decoder output logits
     -> frozen decoder operations (autograd stays ON)
     -> frozen projector P_ell operations (autograd stays ON)
     -> injected encoder activation z_ell
     -> encoder selected LoRA parameters
```

**极其重要的实现禁忌**：

- `decoder.requires_grad_(False)` 是对的，**但不得把 Decoder 前向放进 `torch.no_grad()` / `torch.inference_mode()`**；否则梯度不能回到注入输入。
- E3b 的 `z_ℓ` **不能 `.detach()`、`.cpu().numpy()` 或来自只读缓存**。
- 使用 `torch.where(mask, injected_state, original_state)` 或可导的张量拼接更新；避免破坏 autograd 的危险 in-place 操作。
- 在 `model.train()` 时要保持冻结 Decoder 不训练 dropout（即 `decoder.eval()`）；Encoder LoRA 可训练。
- 训练阶段 `use_cache=False`，防止 KV-cache 与重算混用造成意外行为。
- 检查 `encoder_lora.grad` 非零，`decoder.grad` 为 `None`，且改变 `z_ℓ` 会改变 Decoder Label Score。
- 如果读取层为 `ℓ`，**只更新位于该激活上游或本层的 LoRA**。例如从第 12 个 Block 输出读取 `z`，第 20 层的 LoRA 不可能通过这条路径收到梯度；需要另加深层读出和损失才能训练后层。

### E3.5 四个损失函数

记 `p_y(z)` 为冻结 Decoder 在关系诊断任务中，根据注入的 `z` 输出的候选标签概率。

#### (1) Relation Alignment：正确读取真实关系

\[
\mathcal L_{rel}(x)=-\log p_{Y}(z_\ell(x)).
\]

必须覆盖真实标签，推荐对每个反事实 Pair 的 A/B 两侧都计算分类损失，避免只优化一个方向。

#### (2) Bias Invariance：无关偏差变化时关系预测保持一致

对 `x_A=(C,E,B_A)`、`x_B=(C,E,B_B)`，且 `Y_A=Y_B`：

\[
\mathcal L_{inv}=\mathrm{JS}\big(p(\cdot|z_A),p(\cdot|z_B)\big).
\]

Jensen–Shannon 散度应对完整 K 类概率分布计算，并保证数值稳定（`clamp_min(1e-8)` / log-space）。`L_rel` 与此项联合使用，否则恒定错误预测会取得较小散度。

#### (3) Evidence Sensitivity：关键证据改变时关系预测正确改变

对 `Y_A != Y_B`，定义候选标签的 Logit Margin `m_y(z)`；使用双向 Margin Ranking：

\[
\begin{aligned}
\mathcal L_{flip}={}&
[\gamma-m_{Y_A}(z_A)+m_{Y_A}(z_B)]_+\\
&+[\gamma-m_{Y_B}(z_B)+m_{Y_B}(z_A)]_+.
\end{aligned}
\]

其中 `[a]_+=max(0,a)`，推荐 `γ=0.5` 作为**起始搜索值**，在 dev 上搜索 `{0.2, 0.5, 1.0}`。对于不同长度的文本标签，先把其序列分数按固定策略统一到关系 logits/probabilities。

#### (4) Cross-layer Persistence：关系在深层仍可读（可选增强）

对选定的后继层 `S`，用**各层已校准的** `P_ℓ`/Decoder 读取关系：

\[
\mathcal L_{persist}=\sum_{\ell\in S}w_\ell\,
  [-\log p_{Y,\ell}(z_\ell)].
\]

只有在 E1/E2 确实观察到深层关系失配、且各层注入接口经校准时才加入。**不要拿单一在浅层训练的 Decoder 强行解码所有深层，再把失败当作衰减证据。**

#### 总损失

\[
\boxed{
\mathcal L=\lambda_1\mathcal L_{rel}
+\lambda_2\mathcal L_{inv}
+\lambda_3\mathcal L_{flip}
+\lambda_4\mathcal L_{persist}
}
\]

初始搜索空间（不是实验结果）：

| 参数 | 初始值 | dev 搜索建议 |
|---|---:|---|
| `lambda_rel` | 1.0 | 固定 1.0 |
| `lambda_inv` | 0.2 | `{0, 0.1, 0.2, 0.5}` |
| `lambda_flip` | 0.5 | `{0, 0.2, 0.5, 1.0}` |
| `lambda_persist` | 0.0（基础版） | `{0, 0.1, 0.2}` |
| `margin_gamma` | 0.5 | `{0.2, 0.5, 1.0}` |
| `lora_rank` | 8 | `{8,16}` |
| `lora_alpha` | 16 | `{16,32}` |
| `lora_dropout` | 0.05 | `{0,0.05}` |
| `learning_rate` | 2e-4 | `{5e-5, 1e-4, 2e-4}` |
| `batch_size_per_device` | 1–2 pair | 根据显存配置，需记录有效全局 batch |
| `max_length` | 512 | `{256,512,1024}`，观察截断率 |

**采样策略**：一个训练 Batch 至少包括明确标记的 `bias_change` / `relation_change` pair。若某类 pair 缺失，其损失应置零并记录有效 Pair 数，不可用随机配对代替。建议从 Batch sampler 中显式保持两类 pair 的可用率。

### E3.6 LoRA 层范围与训练顺序

- MVP：`q_proj`、`v_proj`，只对 E2 选出的层添加 LoRA。
- 完整版再比较 `q_proj/k_proj/v_proj/o_proj`。
- 根据 PEFT 的 `layers_to_transform`、`layers_pattern` 或自定义模块筛选实现，并断言实际可训练权重名称与选层一致。
- 先训练 `L_rel`；再加 `L_inv`；再加 `L_flip`；如有必要再加 `L_persist`。这样最便于定位收益与失败。
- 默认验证指标：`Macro-F1` + `Bias PairCorrect` + `Relation PairCorrect`，避免只按训练损失选择 checkpoint。

产物：`decoder_calibration/`（冻结 Decoder/interface）、`encoder_lora/`、`train_log.jsonl`、`gradient_audit.json`、`validation_metrics.json`。

---

## 11. 实验 E4：下游事实核查（Leverage）

### E4.1 两个推理模式

**模式 A：Latent-only Relation Readout（机制诊断）**

```text
Encoder(C,E,B) -> z_ell -> Frozen Decoder(q_rel, latent slot) -> verdict
```

不向 Decoder 提供 Claim/Evidence 明文，直接检查内部关系是否足以生成正确标签。它是**机制指标**，不代表完整事实核查系统的实用性。

**模式 B：Latent-guided Fact-checking（主下游评估）**

```text
Encoder(C,E,B) -> z_ell
Decoder(C,E, injected z_ell, q_verify)
    -> {verdict, evidence_ids, justification}
```

- Decoder 可看到 C/E 文本以便引用证据，但**不能看到 gold label**。
- 保持和 E3a 一致的 latent slot / write layer；如果因生成 Prompt 改变导致 latent 实际不起作用，需要重新校准 Decoder，而不是临时用 gold 关系提示它。
- 主结果必须同时报告 `(i) latent-guided`、`(ii) null-latent`、`(iii) shuffled-latent`；若三者相近，不能声称下游收益由潜在表示引起。
- `evidence_ids` 必须来自输入列表中的 ID；不允许生成不存在的证据编号后仍算证据正确。

### E4.2 结构化输出

```json
{
  "verdict": "REFUTES",
  "evidence_ids": ["E2"],
  "justification": "E2 states that the company was founded in 2012, contradicting the year 2010 in the claim."
}
```

推荐分开评价：

- `verdict`：严格标签匹配、Macro-F1。
- `evidence_ids`：证据选择准确率或官方 evidence scorer；不能只凭解释文本看起来合理判断正确。
- `justification`：少量人工抽查“是否忠实于引用证据、是否存在证据中没有的事实”；可选独立自动指标作补充。

### E4.3 两种证据设置必须分别汇报

1. **Oracle evidence classification**：给定人工标注或确认的证据，评估关系推断与偏差鲁棒性；不能把此结果称为端到端 FEVER Score。
2. **Retrieved evidence pipeline**：锁定同一个检索器和检索知识库，先检索 Evidence 再调用核查模型；只在满足官方 scorer 输入要求时汇报 FEVER Score 或 AVeriTeC Score。检索器不可因下游方法不同而偷偷改变。

### E4.4 核心外部评估

- FEVER 官方开发/测试设置（按可用标签/官方 scorer 权限执行）。
- FEVER Symmetric v0.2 dev/test（专门的 SUP/REF 去偏挑战评估）。
- VitaminC 关系对（是否对最小事实变化做出**正确**翻转）。
- 自建 CReST-Bench 的两个配对子集（Bias Invariance 与 Evidence Sensitivity）。
- AVeriTeC（单独四分类或明确说明排除了冲突类的三分类子集）。

### E4.5 内部—外部因果关联分析

将 E3 前后比较：

1. 分层关系 Probe 的表现变化（**重新训练 Probe 和固定 Probe 两种分析都做**；固定 Probe 更便于比较原有读取坐标系，但可能低估表示旋转）。
2. E2 的 patching 效应是否发生变化。
3. 对于从错到对的样本，latent 修改是否可通过 `ablation` / `restoration` 改变最终决策。
4. 将已训练模型的**同一样本** latent 用原始模型 latent 替换（同一接口、同一目标输入），检查收益是否消失；严格控制 Decoder 和接口不变，避免模型兼容性带来的伪效果。
5. 记录“Probe 改善但 Verdict 未改善”的失败样本，检验内部改进是否真正进入最终决策。

若内部 Probe 变好、但下游 `null_latent` 与 `steered_latent` 无差异，结论应是“内部读出得到改善但未证实有效引导生成”，而非宣称完成了端到端去偏。

---

## 12. 基线、消融实验与公平比较

### 12.1 必备 Baselines

| ID | Baseline | 是否训练 | 核心对照 |
|---|---|---|---|
| B0 | Claim-only | 否 | 标签捷径程度 |
| B1 | Full zero-shot / constrained-label base LM | 否 | 原模型核查能力 |
| B2 | Explicit Relation Prompt / CoT | 否 | 外部提示的收益 |
| B3 | Standard SFT / LoRA | 是 | 同数据与参数预算的一般监督微调 |
| B4 | Counterfactual-pair LoRA | 是 | 只有反事实数据而没有内部定位/梯度桥 |
| B5 | Random-layer Latent Steering | 是 | 同预算随机层定位 |
| B6 | Probe-only / CReST-lite | 是 | 仅中间层分类器指导 |
| B7 | CoSToM-style `L_rel` Gradient Bridge | 是 | 参考方法的直接迁移 |
| B8 | **CReST-FC (`L_rel+L_inv+L_flip`)** | 是 | 核心方法 |
| B9 | CReST-FC + `L_persist`（条件启用） | 是 | 跨层保持是否额外有效 |

所有训练方法优先控制：训练集 ID、梯度更新数、有效 Batch Size、训练 Token 数、可训练参数量、Decoder 校准数据量、最终推理模型数量。无法完全匹配时给出清晰的计算/显存表。

### 12.2 核心消融

| 消融 | 研究的问题 |
|---|---|
| `w/o causal layer selection` | 换成随机层/均匀间隔层后是否下降？ |
| `w/o gradient bridge` | 换成直接 Probe 监督或标准 LoRA 后是否下降？ |
| `w/o L_inv` | 无关偏差稳定性是否下降？ |
| `w/o L_flip` | 模型是否逐渐对证据变化不敏感？ |
| `w/o L_persist` | 深层关系可读性和最终表现是否下降？ |
| `single-layer vs multi-layer` | 单点增强能否解决跨层传递问题？ |
| `correct latent vs null/shuffled latent` | 下游是否真的利用了内部 latent？ |
| `shuffled labels` | 改进是否来自学习有效关系，而非任意梯度噪声？ |
| `bias-type holdout` | 对训练中没有出现的偏差类型是否泛化？ |

**标签打乱消融**仅在对照训练运行中执行，确保不会把标签混入正式实验或误用于开发集。

---

## 13. 评价指标与统计方法

### 13.1 分类指标

- `Accuracy`：整体正确率。
- `Macro-F1`：三/四个标签的非加权平均 F1。
- `Balanced Accuracy`：各类别 Recall 的宏平均。
- `Per-class F1`：尤其关注 NEI 的错误类型。
- `Invalid Rate`：结构化输出无法解析的比例（不能从分母剔除）。
- 可选 `ECE/Brier`：使用可靠的归一化标签概率做校准分析。

### 13.2 偏差鲁棒性指标

对 `N_b` 个 Bias-changing Pair (`Y_A=Y_B=Y`)：

\[
\operatorname{BiasConsistency}
=\frac{1}{N_b}\sum_i\mathbf{1}[\hat Y_{iA}=\hat Y_{iB}],
\]

\[
\operatorname{BiasPairCorrect}
=\frac{1}{N_b}\sum_i\mathbf{1}[\hat Y_{iA}=Y_i\land\hat Y_{iB}=Y_i].
\]

另报 `BiasFlipRate = 1 - BiasConsistency` 和 `BiasAccGap = Acc(original) - Acc(biased)`（可为负数）。优先把 **BiasPairCorrect** 作为主稳健性指标。

### 13.3 证据敏感性指标

对 `N_r` 个 Relation-changing Pair (`Y_A≠Y_B`)：

\[
\operatorname{RelationPairCorrect}
=\frac{1}{N_r}\sum_i
\mathbf{1}[\hat Y_{iA}=Y_{iA}\land\hat Y_{iB}=Y_{iB}].
\]

可同时报告 `RawFlipRate = mean(Ŷ_A≠Ŷ_B)`，但 **RawFlipRate 高并不意味着正确**；主指标应为 `RelationPairCorrect`。

### 13.4 机制指标

| 指标 | 计算条件 | 解读 |
|---|---|---|
| `ProbeMacroF1(ℓ,p)` | 固定模型、分层训练 Probe | 关系线性/非线性可解码性 |
| `CausalIE(ℓ,p)` | Counterfactual Patching | 指定激活对下游标签 margin 的影响 |
| `PositiveIEFraction` | 同上 | 影响方向稳定性 |
| `NullLatentGap` | 正常与空 latent 的推理比较 | 下游是否依赖 latent |
| `ShuffledLatentGap` | 正常与错配 latent 比较 | 下游是否用到了样本特异信息 |
| `LayerPersistence` | 多层校准后的关系 Probe | 可读性沿网络深度的变化 |

若定义相对持久性可使用：

\[
P_\ell=\frac{F1_\ell-F1_{chance}}{\max_j F1_j-F1_{chance}},
\]

但 `F1_chance` 依赖类别分布与基线，不一定简单等于 `1/K`。为避免误解，**主文优先报告原始 Macro-F1 曲线**，此归一化仅作辅助展示。

### 13.5 证据正确性与官方指标

- FEVER Score：仅对满足其端到端证据检索/证据集格式要求的实验使用官方 scorer。
- Oracle Evidence Accuracy：单独命名，不能冒充 FEVER Score。
- Evidence ID Precision/Recall：只在有明确 gold evidence ID 且 ID 对应关系可信时计算。
- AVeriTeC 官方 Score：遵守四分类、证据质量及官方 scorer 的要求；若使用 gold evidence，则单列 oracle 条件。
- Justification：人工复核 Citation Correctness / Unsupported Claim Rate，自动 Judge 可作附录，不作为唯一证据。

### 13.6 统计检验

- 每个主要训练配置建议至少 **3 个随机种子**（例如 13、21、42），报告均值、标准差。
- 对测试指标用**按 `group_id` 聚类的 paired bootstrap**（推荐 10,000 次抽样，可按预算调整）给出 95% CI；同一 Claim 的多个变体不能按独立样本重抽样。
- 同一测试样本集的分类预测可做 McNemar 配对检验；多模型/多数据集比较注意多重检验问题。
- 先在 dev 固定主指标/主要对比，测试集只计算一次最终结果。

---

## 14. 建议的 Python 模块接口及关键伪代码

> 以下为**实现接口草图**，不是已提供的可运行源文件。实现时应根据具体 Hugging Face 模型类型编写适配器并逐项单测。

### 14.1 最小数据类型

```python
# schemas.py: sketch
from dataclasses import dataclass
from typing import Literal

Label3 = Literal["SUPPORTS", "REFUTES", "NEI"]
PairKind = Literal["none", "bias_change", "relation_change"]

@dataclass
class Evidence:
    evidence_id: str
    text: str
    doc_id: str | None = None
    sent_id: str | None = None

@dataclass
class Example:
    sample_id: str
    group_id: str
    claim: str
    evidence: list[Evidence]
    label: Label3
    pair_id: str | None = None
    pair_kind: PairKind = "none"
    claim_metadata: dict | None = None
```

**验证职责**：Pydantic/数据校验器负责检查字段、跨样本配对、split 一致性，`Example` 本身不需要承担所有全局校验。

### 14.2 模型和读出适配器

建议为不同 HF 架构实现：

```python
# activations.py: API contract

def get_transformer_blocks(model) -> list:
    """Return ordered decoder blocks; assert n == model.config.num_hidden_layers."""

def resolve_read_positions(tokenizer, rendered_prompt, attention_mask):
    """Return token indices/spans for anchor/evidence/edited region."""

def extract_layer_activations(model, encoded_batch, layers, positions):
    """Return activations with [batch, hidden] or [batch, span, hidden]."""

def patch_residual(model, encoded_batch, layer, target_positions, source_values):
    """Temporarily replace specified block-output residual states."""
```

要求每个 Hook 都是 Context Manager 或 `try/finally` 注册/删除，不能让前一条样本的 patch 泄漏到下一次前向。

### 14.3 Patching 伪代码（注意处理 Tuple 输出）

```python
# patching.py: pseudocode; for model-specific adaptation

def make_patch_hook(target_indices, source_values):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()             # evaluation patch, no in-place on original
        h_new[:, target_indices, :] = source_values
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new
    return hook

# Example usage:
# with temporary_forward_hook(blocks[layer], hook):
#     patched_scores = score_all_verdicts(model, target_input)
# Compare to normal target scores and clean source scores.
```

如果 `target_indices` 不是同一位置的向量化索引，需要按 batch 构造散点索引；上面只是语义说明。Hook 的输出 Tensor 形状、dtype、device 必须与原模块一致。

### 14.4 Gradient Bridge 伪代码（必须可微）

```python
# bridge.py / steer_encoder.py: core logic sketch

encoder.train()
decoder.eval()
for p in decoder.parameters():
    p.requires_grad_(False)
for p in frozen_projector.parameters():
    p.requires_grad_(False)

optimizer.zero_grad(set_to_none=True)

# Encoder forward MUST keep autograd graph.
z_a = read_encoder_activation(encoder, batch_a, layer=read_layer)
z_b = read_encoder_activation(encoder, batch_b, layer=read_layer)
assert z_a.requires_grad and z_b.requires_grad

# Projection and decoder forward MUST NOT be under torch.no_grad().
latent_a = frozen_projector(z_a)
latent_b = frozen_projector(z_b)

# decoder_from_latent: differentiably overwrites a fixed latent slot.
logits_a = decoder_from_latent(decoder, prompt_a, latent_a, write_layer)
logits_b = decoder_from_latent(decoder, prompt_b, latent_b, write_layer)

# relation_logits: map decoder outputs to comparable candidate-label scores.
p_a = relation_logits(logits_a)
p_b = relation_logits(logits_b)

loss_rel = relation_ce(p_a, gold_a) + relation_ce(p_b, gold_b)
loss_inv = js_loss(p_a, p_b) if pair_kind == "bias_change" else 0.0
loss_flip = symmetric_margin_loss(p_a, p_b, gold_a, gold_b) \
    if pair_kind == "relation_change" else 0.0
loss = loss_rel + lambda_inv * loss_inv + lambda_flip * loss_flip
loss.backward()

# Sanity checks on every smoke-test run:
assert any(p.grad is not None and p.grad.norm() > 0 for p in encoder_lora_params)
assert all(p.grad is None for p in decoder.parameters())
optimizer.step()
```

**实现细节警告**：这里 `relation_logits(logits)` 是一个接口占位符，实际语言模型应在 teacher forcing 下计算所有候选答案的**序列级条件得分**并组装为 `[B,K]`，不能把单步 Vocabulary Logits 直接当作 `K` 类 Logits。

### 14.5 Label Scorer 单元接口

```python
# label_scorer.py: required API

def score_candidate_sequences(
    model,
    decoder_input,
    candidates=("SUPPORTS", "REFUTES", "NEI"),
    normalize_by_length=False,
):
    """
    Return float tensor [B, K] of teacher-forced continuation log scores.
    Must remain differentiable w.r.t. injected activation during E3b.
    Padding, answer-prefix alignment, and multi-token labels are explicit.
    """
```

对 `K` 个候选答案逐条或批量计算 Logprob；答案 Token 只用于监督答案续写的位置，不得在 Encoder 输入、Decoder 输入问题部分或 Target evidence 中出现 gold 标签。

### 14.6 双重反事实 Pair Sampler

```python
# counterfactual.py: required API

def make_pair_batch(samples, pair_index, pair_kind, rng):
    """
    Returns (a, b, pair_kind) with:
      bias_change: same group, claim, evidence, label; bias metadata differs
      relation_change: same group and claim, edited evidence, differing gold labels
    Must not draw across train/dev/test.
    """
```

训练损失实现尽量使用**结构化 Pair 对象**，不要依赖同一个 Batch 的相邻行碰巧构成配对。

---

## 15. 配置示例、命令行和阶段产物

### 15.1 核心 YAML 示例

```yaml
# configs/steering.yaml (planned)
experiment:
  name: crestfc_qwen7b_main
  seed: 42
  output_root: outputs/e3_steering

model:
  encoder_id: Qwen/Qwen2.5-7B-Instruct
  decoder_id: Qwen/Qwen2.5-7B-Instruct
  encoder_dtype: bfloat16
  decoder_dtype: bfloat16
  max_length: 512
  use_cache: false

data:
  train_path: data/processed/train_steer.jsonl
  dev_path: data/processed/dev_steer.jsonl
  pair_index_path: data/processed/pairs.jsonl
  label_space: fever3
  evidence_mode: oracle
  prompt_version: v1_anchor_last

localization:
  selected_layers_path: outputs/e2_causal/candidate_layers_final.json
  read_position: final_anchor
  read_layers: [10, 14]  # placeholder; replace after E2 on dev

bridge:
  decoder_checkpoint: outputs/e3_steering/decoder_calibration/best
  write_layer: 0
  write_position: latent_slot
  interface: projected_residual_replacement
  latent_only_relation_query: true

lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, v_proj]
  # trainable layer IDs must be validated against model.named_parameters()

loss:
  lambda_rel: 1.0
  lambda_inv: 0.2
  lambda_flip: 0.5
  lambda_persist: 0.0
  margin_gamma: 0.5

train:
  lr: 0.0002
  epochs: 3
  per_device_pair_batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: bf16
  eval_every_steps: 100
  early_stopping_patience: 5
  select_metric: dev_macro_f1_and_paircorrect
```

`read_layers: [10,14]`、`write_layer: 0`、Loss 权重等只是**占位配置**，未进行有效层确认或 Decoder 校准之前不可直接用作最终实验结论。

### 15.2 核心 CLI（已实现的命令与配置见 `configs/`）

```bash
# 0. 数据适配、分组切分、反事实生成与审核
python -m crestfc.cli prepare --config configs/fever.yaml
python -m crestfc.cli validate-data --config configs/pilot.yaml

# 1. E0 原始模型性能
python -m crestfc.cli baseline --config configs/evaluate.yaml

# 2. E1 冻结主模型提取分层激活 + probes
python -m crestfc.cli extract-activations --config configs/probe.yaml
python -m crestfc.cli train-probes --config configs/probe.yaml
python -m crestfc.cli plot-probes --config configs/plot_probes.yaml

# 3. E2 因果修补与层选择
python -m crestfc.cli causal-patch --config configs/patch.yaml
python -m crestfc.cli select-layers --config configs/select_layers.yaml

# 4. E3a 校准 Decoder/投影器（Encoder 冻结）
python -m crestfc.cli calibrate-decoder \
  --config configs/decoder_calibration.yaml

# 5. E3b 梯度桥训练（Decoder 冻结、Encoder 目标层 LoRA）
python -m crestfc.cli train-steering --config configs/steering.yaml

# 6. E4 下游核查与反事实评价
python -m crestfc.cli evaluate --config configs/evaluate.yaml
python -m crestfc.cli evaluate-counterfactual --config configs/evaluate.yaml

# 7. 汇总论文表格与统计显著性
python -m crestfc.cli report --config configs/report.yaml
```

CLI 开发顺序以 `prepare -> baseline -> extract -> probe -> patch -> calibrate -> steer -> eval` 为准，不建议在没有 E1/E2 结果时直接运行全套训练。

### 15.3 实验记录 Schema

每次运行的 `metrics.json` 至少包含：

```json
{
  "run_id": "qwen7b_crestfc_seed42",
  "stage": "E4",
  "seed": 42,
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "selected_layers": [10, 14],
  "evidence_mode": "oracle",
  "dataset": "crest_bench_test",
  "label_space": "fever3",
  "n_examples": 0,
  "n_bias_pairs": 0,
  "n_relation_pairs": 0,
  "accuracy": null,
  "macro_f1": null,
  "bias_pair_correct": null,
  "relation_pair_correct": null,
  "invalid_rate": null,
  "latency_ms_per_example": null
}
```

上述 `0/null` 是输出文件的**结构示意**，不表示实际产生了零样本的实验结果。

---

## 16. 单元测试、集成测试与防泄漏检查

### 16.1 数据测试

- `test_label_mapping`：FEVER 三类和 AVeriTeC 四类映射准确，不能默默吞掉 `CONFLICTING_EVIDENCE`。
- `test_pair_bias_preserves_relation`：Bias Pair 的 Claim、Evidence、Gold Label 一致，只允许白名单偏差字段变化。
- `test_pair_relation_changes_label`：Relation Pair 的 Claim 不变，Evidence 发生有效修改且 Gold Label 不同。
- `test_no_group_leakage`：train/dev/test 的 `group_id` 不相交，源文章/revision family 根据设定做额外去重。
- `test_nei_evidence_policy`：NEI 为 insufficient evidence，不能把空证据当作随机替代、且不能误标成 REFUTES。
- `test_source_provenance`：每个样本能追溯源数据修订版。

### 16.2 Token 与 Activation 测试

- `test_final_anchor_index`：加入左/右 Padding 后锚点位置仍准确；不读取 Padding。
- `test_edited_span_alignment`：数字 `2010`、`2012` 的多 Token 形式能正确找到。
- `test_block_output_shape`：Hook 对 Tensor/Tuple 返回类型不改变 dtype、device、其它字段。
- `test_hook_cleanup`：Patching 结束后注册 Handle 被清理，同一输入多次执行一致。
- `test_same_input_patch_identity`：自 Patch 的分数在数值容忍范围内相同。
- `test_full_seq_length_guard`：不等长度时严禁执行全序列 Patch。
- `test_label_multi_token`：候选多 Token 时与手动 teacher forcing 分数一致。

### 16.3 Gradient Bridge 核心测试（必须先通过）

伪代码验收：

```python
loss.backward()
assert any(g is not None and g.abs().sum() > 0 for g in encoder_lora_grads)
assert all(g is None for g in decoder_param_grads)
assert all(g is None for g in frozen_projector_param_grads)
assert torch.isfinite(loss)
```

再补充：

- 只读层下游（网络上更深层）的 LoRA 不应错误声称收到梯度。
- `P_ℓ(z)` 对 `z` 的导数应有效，`z.detach()` 后必须触发预期的测试失败。
- Decoder Prompt 不得包含 gold verdict / gold explanation；如果是 teacher forcing，目标答案必须只出现在监督目标位置。
- 对固定 Batch 执行单步更新前后，核实目标 Encoder LoRA 参数确实变化，Decoder 参数完全不变。
- `L_inv` 对完全相同概率分布接近 0；`L_flip` 对正确且相互区分的两份关系得分应下降。

### 16.4 集成 Smoke Test

使用约 20–50 条人工确认的小样本、短序列和小模型：

1. 数据校验无错误；
2. 模型读取各层 Activation；
3. 两份输入可以进行双向 Patching；
4. Decoder 校准损失能在极小训练集上下降；
5. 冻结 Decoder 后 LoRA 有非零梯度；
6. 在固定的 toy training set 上可过拟合（但此结果不可用于正式论文结论）；
7. 输出 JSON 能解析且证据 ID 不越界；
8. 通过 `pytest` 后再上 7B 模型和多随机种子。

---

## 17. 运行顺序、资源预算与停止条件

### 17.1 里程碑与验收 Gate

| Milestone | 实现内容 | 完成标准 | 下一步 |
|---|---|---|---|
| M0 | 数据与配对校验、E0 | Pair 审核合格、无 split 泄漏，标签打分正确 | E1 |
| M1 | 全层 Linear/MLP Probe | 能产生可复现层级图，反事实 PairCorrect 合理 | E2 |
| M2 | 双向 Causal Patching | 目标层相对随机层具有稳定的因果效果（或确认无效） | E3a |
| M3 | Decoder Calibration + Gradient Bridge | 读取能力过关、Encoder LoRA 有梯度且 Decoder 冻结 | E3b |
| M4 | `L_rel`, `L_inv`, `L_flip` 实验 | dev 的准确率和至少一种成对稳健性有可解释变化 | E4 |
| M5 | 多模型/多数据集/消融/统计 | 测试集结果、置信区间、成本和失败分析齐全 | 写论文 |

**停止/转向规则**：

- M1 若 relation pair 上 Probe 与随机/Claim-only 无可辨差异：优先检查 Prompt、证据质量和 Probe 的 Token 位置，不急于 Steering。
- M2 若 Patching 仅在随机层也产生同等效果：重做控制实验，不能宣称找到关系特异因果层。
- M3 若 Decoder 的 `null_latent` 与真实 latent 结果相同：先修好注入接口和 Decoder Calibration。
- M4 若 `L_inv` 提高了 Bias Consistency 但降低 RelationPairCorrect：说明模型可能过度不变，调低权重或重做反事实构造。
- 如果没有深层关系衰减，移除 `L_persist` 并聚焦“关系—决策失配”。

### 17.2 资源预估（仅作为工程预算）

- 7B 参数模型以 BF16 保存**单份模型权重约 14 GB**（不含 KV/Activation/优化器/LoRA 等）；两份约 28 GB，再加训练激活、临时张量和显存碎片。
- E1 离线 Probe 不必同时加载两份 7B，单卡可分批读取；建议每次仅缓存目标位置。
- E2 Patching 主要是多次前向计算，GPU 时间与 `层数 × 位置 × 配对方向 × 样本数` 大致成正比；先在 200–500 对 dev 样本上筛层。
- E3 完整版的双模型 + Gradient Bridge 更占显存；7B 主实验建议优先采用多卡高显存环境（例如 2×80GB 级别）或经验证的分布式/CPU offload 方案。
- 1×24GB 环境建议先用 0.5B–1.5B 小模型跑完整正确性测试；不要假定两份 BF16 7B 训练可直接装下。
- 从 E1 开始记录 `peak_cuda_memory_allocated`、每千样本耗时、训练吞吐、推理延迟，作为公平比较的一部分。

### 17.3 缓存原则

- E1/E2 的离线激活可使用 `no_grad` 缓存；缓存 key 包含 `model_revision + tokenizer_revision + prompt_version + data_hash + layer + position`。
- E3b 的 Encoder Activation 参与反向传播，**不允许用失去计算图的离线缓存代替**。
- 多卡时避免把跨设备 Tensor 隐式复制到 CPU；先做单 Batch 梯度和设备一致性检查。
- 使用 gradient checkpointing 时确认 Hook 不会因重计算重复写入缓存或额外篡改状态。

---

## 18. 结果报告模板和结论判定

### 18.1 主结果表（所有空位等待实际运行填充）

| 方法 | FEVER Macro-F1 ↑ | Symmetric Acc ↑ | BiasPairCorrect ↑ | RelationPairCorrect ↑ | AVeriTeC Score ↑ | 推理耗时 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Base zero-shot | TBD | TBD | TBD | TBD | TBD | TBD |
| Standard LoRA | TBD | TBD | TBD | TBD | TBD | TBD |
| Counterfactual LoRA | TBD | TBD | TBD | TBD | TBD | TBD |
| Random-layer Bridge | TBD | TBD | TBD | TBD | TBD | TBD |
| CoSToM-style `L_rel` | TBD | TBD | TBD | TBD | TBD | TBD |
| **CReST-FC** | TBD | TBD | TBD | TBD | TBD | TBD |

**不要在 Oracle evidence 实验中填入 FEVER 官方 Score 或 AVeriTeC 官方 Score**；不适用项填 `N/A` 并在表注中说明。

### 18.2 论文所需最少图表

1. `Figure 1`：CReST-FC 数据流与 Gradient Bridge 模型结构图。
2. `Figure 2`：全层关系 Probe / Claim-only Probe / Bias Probe 曲线。
3. `Figure 3`：`Layer × Token Position` 的 CausalIE Heatmap，含随机 Patch 控制。
4. `Figure 4`：训练前后关系层级可读性和 downstream label margin。
5. `Figure 5`：BiasPairCorrect 与 RelationPairCorrect 的二维散点图（防止只追求稳定）。
6. `Table 1`：各模型和各数据集主结果。
7. `Table 2`：损失与层选择消融。
8. `Table 3`：训练参数、推理延迟/显存、推理模型数量。
9. `Appendix`：反事实示例、错误分类、随机种子、各层详细数值。

### 18.3 结论强度与对应证据

| 可以写进结论的表述 | 最低证据要求 |
|---|---|
| “发现部分中间层可读的 Claim–Evidence 关系信息” | E1 的关系配对表现优于 Claim-only / 随机标签 Probe |
| “特定层的表示对核查判断具有因果影响” | E2 双向 Patching 效果优于对照，有置信区间 |
| “关系线性可解码性随层衰减” | 固定、容量受控的分层 Linear Probe 趋势 |
| “关系信息本身消失” | **不建议仅根据目前设计下结论**；需要更强的信息论或结构性证据 |
| “内部关系与决策存在失配” | Probe 保留关系信息，实际输出却在 Bias 干预下失败，并有因果证据 |
| “内部增强改善了去偏事实核查” | E4 分类与两类 PairCorrect 同时合理改善、关键消融成立 |
| “增强 latent 对最终生成有因果贡献” | E4 normal/null/shuffled latent 和恢复/消融实验存在可解释差异 |

---

## 19. 风险、替代方案与研究贡献

### 19.1 主要风险

**R1：深层 Probe 下降可能只是表示旋转。**  
解决：Linear/MLP Probe、容量控制、Causal Patching、同一层读取接口的校准，避免称为严格信息丢失。

**R2：随机注入激活造成表面“因果效果”。**  
解决：same-input、random sample、random layer、wrong-label、token 长度匹配等负对照。

**R3：训练 Decoder 后，它学会了数据集捷径或记忆答案。**  
解决：诊断任务禁止直接提供 Claim/Evidence，校准/Steering 数据组分离，检查 null/shuffled latent，针对 held-out 关系对评估。

**R4：模型过度追求不变性，不再响应证据。**  
解决：联合 `L_inv` 和双向 `L_flip`，同时报告两种 PairCorrect。

**R5：NEI 和多证据不受控。**  
解决：先用高质量 SUPPORTS/REFUTES 单证据做机制实验，再单列充分性审核后的 NEI、多证据实验。

**R6：Decoder 直接看到 Claim/Evidence 后忽略注入 latent。**  
解决：强制 null/shuffled latent 消融；必要时通过 E3a 的任务 B 校准 latent 使用，或采用仅由 latent 输出标签、明文 C/E 只用于生成解释的分阶段输出架构。若采用分阶段方案，必须明说与“单次端到端生成”不同。

**R7：双模型成本太高，改进仅来自更多参数。**  
解决：参数/推理预算匹配对照，提供 CReST-lite 单模型基线，报告绝对模型副本数与成本。

**R8：事实核查时间与出处信息可能是真实证据。**  
解决：明确时间/出处在任务定义中的因果角色，不能将影响证据可信度的信息简单作为“非因果偏差”抹除。

### 19.2 建议的最精简论文叙事

1. 通过对照样本发现 Claim–Evidence 关系的**内部可读性**与**最终使用**可能不同。
2. 通过因果 Patching 找到影响最终核查的关系敏感层，分析表示衰减或决策失配。
3. 通过冻结 Decoder 的 Gradient Bridge，在敏感层训练 Encoder LoRA。
4. 使用“偏差变化保持正确 + 关系变化正确翻转”的双重反事实目标。
5. 在标准数据、外部挑战集、真实场景和因果消融上验证效果。

若最终证明的是“关系未衰减，但决策没用好”，仍然有重要价值，论文主张应改为 **Internal Evidence–Decision Misalignment**，不要为了模仿 CoSToM 强行主张层级坍缩。

---

## 20. 参考资料

以下链接优先使用论文与作者维护的官方页面（访问及许可按各项目规定处理）。

1. Li, Shi, Deng. **CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models**. ACL 2026. 论文：https://aclanthology.org/2026.acl-long.421/ ；源码：https://github.com/CGCL-codes/CoSToM
2. Thorne et al. **FEVER: a Large-scale Dataset for Fact Extraction and VERification**. NAACL 2018. 论文：https://aclanthology.org/N18-1074/ ；数据与格式：https://fever.ai/dataset/fever.html
3. Schuster et al. **Towards Debiasing Fact Verification Models**. EMNLP 2019. FEVER Symmetric 数据与说明：https://github.com/TalSchuster/FeverSymmetric
4. Schuster, Fisch, Barzilay. **Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence**. NAACL 2021. 数据与代码：https://github.com/TalSchuster/VitaminC
5. Schlichtkrull, Guo, Vlachos. **AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web**. NeurIPS Datasets & Benchmarks 2023. 数据说明：https://fever.ai/dataset/averitec.html

---

## 下一步的实际编码优先级（Checklist）

- [ ] 建好 `src/crestfc`、`configs` 和 `tests`，固定 `pyproject.toml`。
- [ ] 适配 FEVER 原始证据索引与 VitaminC 数据，输出统一 JSONL。
- [ ] 实现 `pairs.jsonl` 和自动/人工对照审核接口，先得到 200–500 个高质量 Pair。
- [ ] 实现三个 Prompt 条件与候选标签的 teacher-forced scoring。
- [ ] 实现 Qwen2.5 的 `get_transformer_blocks()` 和 Token span 定位。
- [ ] 跑通所有层的 `final_anchor` 读取和 Linear Probe。
- [ ] 实现双向关系变化 Patching + bias 变化 Patching + 负对照。
- [ ] 依据 dev 结果选层；固定选层规则和 Prompt 模板版本。
- [ ] 实现 Decoder 校准、冻结、latent slot 投影器与 Hook。
- [ ] 先验证 `L_rel` 梯度从 Decoder 回到 Encoder LoRA。
- [ ] 加入 `L_inv`、`L_flip` 和相应 Pair Sampler。
- [ ] 输出全套 E4 判定指标、消融表和置信区间。
- [ ] 根据 E1/E2 是否存在跨层衰减，决定是否实现 `L_persist`。
- [ ] 在第二种模型和 AVeriTeC 上做迁移、误差分析与资源对比。

**最终交付标准**：只有当 **内部关系读取/因果定位 + 偏差变化正确保持 + 证据变化正确响应 + 正式下游性能提升 + 对照消融** 形成一致证据链时，才宣称 CReST-FC 达到了“通过因果导向的内部关系 Steering 改善去偏事实核查”的研究目标。


## 工程实现状态（2026-11-02）

- 核心源码位于 `src/crestfc/`，配置位于 `configs/`，测试位于 `tests/`。
- 已完成数据适配、人工审核门槛、E0 分类、E1 线性 Probe、E2 最后锚点因果修补、E3 梯度桥与三个损失、E4 标签评价、可选结构化输出及 null/shuffled latent 对照。
- 当前未完成的正式论文实验及协议限制见 [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)。请勿把烟测指标或 Oracle evidence 分类结果写作正式 FEVER Score。

### GPU 环境与实验脚本

推荐 **Python 3.10.15**。本机驱动 535.54.03 / CUDA 12.2 可使用 `requirements.txt` 中固定的 PyTorch 2.3.0 CUDA 12.1 wheel；现有 `/data/yangjun/.conda/envs/llama/bin/python` 已通过 GPU 烟测。第三方库版本见 [requirements.txt](requirements.txt)，新建环境可运行 `scripts/install_gpu_env.sh`。

```bash
PYTHON_BIN=/data/yangjun/.conda/envs/llama/bin/python \
  scripts/run_experiments.sh e0 --run-id pilot --limit 20
PYTHON_BIN=/data/yangjun/.conda/envs/llama/bin/python \
  scripts/run_experiments.sh e1 --run-id pilot --limit 20
```

按阶段运行、人工审核门槛、`--reviewed-data` 和正式测试集的使用见 [实现与复现说明](docs/IMPLEMENTATION.md)。
