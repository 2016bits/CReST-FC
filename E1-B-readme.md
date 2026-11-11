# E1-B：基于冻结 LLM Activation Reading 的无训练关系定位实验

> 项目：**CReST-FC — Causal Relation Steering for Trustworthy Fact-Checking**  
> 文档类型：实验设计与工程实现规范（README，v1.0）  
> 日期：2026-11-04  
> 状态：**无训练 Reader 核心与运行入口已实现，并完成小样本工程检查；人工审核开发集、28 层正式扫描和独立测试尚未完成**  
> 对照研究：[CoSToM](https://github.com/CGCL-codes/CoSToM)；项目：[CReST-FC](https://github.com/2016bits/CReST-FC)

## 0. 一页摘要

**目标**：不使用 FEVER 训练集拟合线性分类器，也不微调 LLM；将 Claim–Evidence 输入冻结的 Qwen2.5-7B-Instruct（Context Encoder），提取每一 Transformer Block 的**完整序列隐藏状态**，将其注入另一次冻结的同权重 LLM 前向过程（Probe Decoder），让 Decoder 在**看不到原始 Claim/Evidence 文本**的条件下判断 `SUPPORTS` / `REFUTES`。通过逐层分类、反事实成对评价和严格负对照，定位模型内部可被冻结 LLM 读取的关系信息。

主实验配置：

| 项目 | E1-B 设定 |
|---|---|
| 任务 | 二分类：`SUPPORTS` / `REFUTES`；不包含 NEI |
| 模型 | Encoder 与 Decoder 均为 **Qwen/Qwen2.5-7B-Instruct**，同权重、`eval()`、冻结 |
| 是否训练 | **0 个可训练参数**；不进行梯度更新、监督 Probe 训练、LoRA、阈值拟合 |
| Encoder 读取位置 | Transformer **Block 输出**，zero-based `L0`–`L27` |
| 读取粒度 | **全部有效 Token** 的隐藏状态 `[1, T, 3584]`；Anchor-only 为后续消融 |
| Decoder 写入位置 | **Block 0 的输出**，覆盖前置占位 Token 的隐藏状态；初始设置固定 |
| Decoder 输入 | 占位 Token + 问题模板；**不输入 Claim/Evidence 原文或 Gold Label** |
| 主评分方式 | 两个完整标签的 teacher-forced log-likelihood 差值（未经标签监督校准） |
| 主数据 | 人工核查的**同 Claim、不同 Evidence、标签翻转**反事实配对 |
| 主指标 | Macro-F1、Balanced Accuracy、PairCorrect、Direction、Tie Rate |
| 重要基线 | 7B Direct Text、No Patch、Claim-only、Evidence-only、Random Donor、Counterfactual Donor |

**核心限制**：本实验检验的是 *frozen-decoder relation readability*（冻结 Decoder 可读取性），**不是**某层已经执行事实推理的充分证明，也不是原始事实核查决策路径使用该关系的因果证明；后者仍需 E2。

---

## 1. 背景与要回答的问题

### 1.1 已完成的相关工作（作为背景，不属于 E1-B 的结果）

- E0：Qwen2.5-0.5B-Instruct 直接分类；FEVER 完整输入 Macro-F1 为 **22.75%**。由于模型尺寸、标签空间与任务协议不同，**不可直接**与 7B 的 E1 相减解释。
- E1：冻结 Qwen2.5-7B-Instruct，取指定 Anchor 位置的不同层隐藏向量；分别训练每层 Logistic/Linear Probe。FEVER dev 中 L19：Full **93.76% Macro-F1**、Claim-only **85.37% Macro-F1**（不是 Accuracy）。
- E1.5：将训练好的 Probe 冻结，并用于同 Claim、改变 Evidence 的反事实配对；VitaminC 的较高方向一致率与相对较低整对判对率提示输出边界/跨域泛化问题。
- E2：已进行 Anchor Patching 小规模试验；不能据此断言 L18–L20 无因果作用。

原 E1 仅证明“**标签可通过大量监督样本从隐藏向量中线性解码**”。E1-B 换一个更贴近 CoSToM Locate 的问题：

> **无需在该任务上训练外部分类器，冻结 LLM 自身能否通过规定的激活接口，从第 ℓ 层的上下文激活中判断 Claim 与 Evidence 的关系？**

### 1.2 研究问题

- **RQ1：** 哪些读取层的隐藏状态能够支持冻结 Decoder 在二分类任务上正确输出？
- **RQ2：** 改变 Evidence 而保持 Claim 不变时，读取结果能否按正确方向变化，并同时跨过标签决策边界？
- **RQ3：** Full 激活相较 Claim-only、Evidence-only 和随机 Donor，是否提供可靠的额外信息？
- **RQ4：** E1-B 的最佳读取层是否与旧 E1 的 L18–L23 区域一致？如不一致，可能与读取接口、写入层和跨层分布有关。

### 1.3 与 CoSToM 的继承关系和差异

继承：**冻结 Context Encoder + 冻结 Probe Decoder；读取多层的完整上下文激活；向 Decoder 问题前的占位位置注入；由 Decoder 自己生成/评分答案。**

改动：CoSToM 研究 ToM/BDI；本项目研究 **Claim–Evidence 支持/反驳关系**。本项目增加反事实 Evidence Flip、无监督序列标签打分、跨域与数据充分性审核。

需要注意：CoSToM 的公开 `latent_qa()` 通过 `token_idx=None` 缓存序列激活，并在读取/写入模块间替换；`interpret_config.py` 给出 `layer_to_write=0` 的示例配置。本 README 是**参考并重新规范化实现**，不是逐行复制 CoSToM；其源码中的 Hook、Mask、Padding、生成细节均需独立验证。

主要参考：

1. CoSToM Locate 入口：https://github.com/CGCL-codes/CoSToM/blob/main/CoSToM_reading_tom.py
2. CoSToM 激活工具：https://github.com/CGCL-codes/CoSToM/blob/main/lit/utils/activation_utils.py
3. CoSToM 层配置：https://github.com/CGCL-codes/CoSToM/blob/main/interpret_config.py
4. CReST-FC 实现约定：https://github.com/2016bits/CReST-FC/blob/main/docs/IMPLEMENTATION.md

---

## 2. 方法总览及符号

令 Claim 为 `C`，Evidence 为 `E`，Gold Label 为 `y∈{S,R}`，其中 `S=SUPPORTS`，`R=REFUTES`。

```text
                        全部参数冻结

 Claim C + Evidence E
         │
         ▼
  Context Encoder (Qwen2.5-7B)
         │
         ├── Block L0: H_0 = [h_0,1 ... h_0,T]
         ├── Block L1: H_1
         ├── ...
         └── Block L27: H_27
                        │
                 选定读取层 ℓ
                        │  H_ℓ ∈ R^(1 × T × 3584)
                        ▼
             Activation Injection ──────┐
                                        │ 覆盖 Decoder 的前 T 个
 Decoder 输入：                          │ 占位位置在 Block 0 的输出
 [slot_1 ... slot_T] + diagnostic q      │
         │                              │
         ▼                              │
  Probe Decoder (同权重 Qwen2.5-7B) ◄───┘
         │
         ├── score("SUPPORTS")
         └── score("REFUTES")
                       │
            m_ℓ = score(S) - score(R)
                       │
              预测 + 配对评价
```

记 `Enc_θ` 和 `Dec_θ` 是**相同参数 θ 的两个逻辑角色**（可复用同一个物理模型实例，但必须顺序运行且严格管理 Hook）：

\[
H_\ell(C,E) = \mathrm{BlockOutput}_\ell(\mathrm{Enc}_\theta(x(C,E)))
\]

\[
\widehat{y}_\ell = \operatorname{argmax}_{y\in\{S,R\}}
\log P_{\mathrm{Dec}_\theta}\left(y\mid q,\operatorname{Patch}_{k=0}(H_\ell)\right)
\]

重点区分：

- `read_layer = ℓ` 是 **Encoder 读取 Block**，`0..27`；
- `write_layer = k` 是 **Decoder 被替换输出的 Block**，主实验固定为 `0`；
- 这是**跨层**写入（如从 `Encoder L19 → Decoder L0`），不是直接在同层模型上替换；
- `Block output` 与 `Block input`、`model.hidden_states` 的具体索引可能不同，必须由测试固定定义；
- 源激活为完整序列，**不是仅一个 `Relation anchor` 向量**；
- 不通过 `lm_head(h_ℓ)` 直接分类；Decoder 继续完成后续计算。

---

## 3. 数据协议：先保证任务本身可判定

### 3.1 首选数据

1. **调试集**：5–10 条可独立验证标签的样例；用于 Hook/Label Scorer 测试，不能作正式统计。
2. **开发集**：先选 50–100 对，再扩大到 200–500 对人工审核的 **同 Claim、不同 Evidence** 配对，优先利用已有 VitaminC 和 symmetric-FEVER 审核队列。
3. **最终测试集**：不同 claim、来源家族/文档分组隔离，**在固定配置后只评价一次**；同 Claim 的两个成员不得跨 split。
4. 可增加 FEVER 二分类普通样本集，观察 In-distribution 性能，但应当与配对指标分开报告。

**警告**：项目中 EGR-FV 转换后的 FEVER Evidence 标记为 `retrieved`，不能仅凭标签文件格式完整，就假定每条 Evidence 足以支持 Gold Label。新 E1-B 的严谨结论必须明确证据类型（retrieved/oracle）和人工审核比例。VitaminC 的修订对也不能自动视为“同 Claim 反事实”，必须验证对应字段和语义。

### 3.2 必要标签约定

- 本实验仅处理二类：`SUPPORTS` 和 `REFUTES`。
- `NOT_SUPPORTED` **不总是** `REFUTES`；不可悄悄混入 NEI/证据不足样本。
- 非二分类标签必须记录排除原因，不可无报告地映射。
- Evidence 不足、事实错误或 Claim 不相同的所谓“关系反转对”应排除或单独列为未通过审核。

### 3.3 输入样本 JSONL（建议统一格式）

```json
{"sample_id":"pair_001_s","pair_id":"pair_001","split":"dev","claim":"The company was founded in 2010.","evidence":"The company was established in 2010.","label":"SUPPORTS","dataset":"manual_demo","evidence_type":"curated","reviewed":true,"source_family":"demo_001"}
{"sample_id":"pair_001_r","pair_id":"pair_001","split":"dev","claim":"The company was founded in 2010.","evidence":"The company was established in 2012.","label":"REFUTES","dataset":"manual_demo","evidence_type":"curated","reviewed":true,"source_family":"demo_001"}
```

审查配对的固定条件：

- 同一个 `pair_id` **恰有两个**成员，Claim 在规范化后逐字相同；
- 两个 Evidence 不同，两个标签分别为 S/R；
- Evidence 自身足够作判断，不能依赖未输入文档；
- 两成员的 `source_family` 相同，所有同家族样本放在同一个 split；
- 样本 ID 唯一、空文本为零、Evidence 截断情况明确记录；
- 若同一 Claim 出现在多个配对/数据源中，按 Claim 或更严格家族粒度防泄漏。

### 3.4 输入条件与匹配规则

| 条件 | Encoder 读取文本 | 用途 |
|---|---|---|
| `full` | Claim + Evidence | 主实验 |
| `claim_only` | 只含 Claim | 检查 Claim 标签捷径；同 Claim 配对两端的激活相同（若模板完全一致） |
| `evidence_only` | 只含 Evidence | 检查 Evidence 文本本身的标签线索 |
| `no_patch` | 不使用 Encoder 激活 | Decoder 固定偏好 |
| `random_donor` | 注入另一条无关样本激活 | 信息是否与当前样本对应 |
| `cf_donor` | 注入同 Claim 相反 Evidence 的激活 | 测试标签/分数随证据切换的响应 |

**重要**：Full/Claim-only/Evidence-only 的 Token 数量本来可能不同。因此在不同条件之间比较 Macro-F1 之外，要增加**等长度随机/空占位**对照；配对内保持相同 Claim 且在可能范围内匹配 Evidence 长度，避免“序列长度本身泄露标签”。

---

## 4. 冻结模型和 Tokenizer 约定

建议使用项目服务器已缓存的 7B 权重，并记录**具体权重与 tokenizer revision**，而非仅记录模型名。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 单卡原型可改为确定性设备；需与 Hook/张量搬运兼容
    trust_remote_code=False,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

assert len(model.model.layers) == 28
assert model.config.hidden_size == 3584
```

建议在实验中采用 `torch.inference_mode()`、`use_cache=False`、确定性评分；记录 GPU/CUDA/torch/transformers、模型权重 revision、Prompt Hash、每条数据 SHA-256。

**物理加载策略**：初期可只加载**一份**模型权重——先作为 Encoder 做前向并缓存激活，再注册 Decoder Hook 做第二次前向。逻辑上依然是两次独立的冻结模型计算。正式多 GPU 版可用两份同权重实例以避免 Hook 状态交叉。切记不要把 Decoder 的 Hook 留到下一条 Encoder 前向。

---

## 5. Step-by-Step：从数据到逐层结果

### Step 1 — 构造 Encoder Prompt

建议固定 `encoder_prompt_version: e1b_ctx_v1`：

```text
Claim:
{claim}

Evidence:
{evidence}
```

- Encoder 只接收事实核查上下文；**不含 Gold Label**。
- 起步不加 `Relation anchor:`；可将“有/无 Anchor”作为单独消融。
- 如果启用 `apply_chat_template`，统一使用同一模板、相同 special token 设置；不要无意让 Encoder 预先看到 Decoder 问题。
- 三种输入条件（full/claim/evidence）采用完全可复现的格式，记录 `encoder_input_ids` 的长度和 Hash。

### Step 2 — 捕获每一层 Block 的完整输出

Qwen2.5-7B 有 28 个 Transformer Blocks；使用 `model.model.layers[0:28]` 上的 Forward Hook，缓存 `output[0]`（若为 tuple）或 `output`（若直接为 Tensor）。

```python
@torch.inference_mode()
def capture_block_outputs(model, encoded_input):
    captured = {}
    handles = []

    def make_hook(idx):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach().clone()  # [B, T, D]
        return hook

    try:
        for idx, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(make_hook(idx)))
        model(**encoded_input, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == len(model.model.layers)
    return captured
```

**这是核心示意代码，尚需在项目环境单元测试**。初期 `batch_size=1`、无 Padding，并以实际 Token 数 `T` 截取有效激活；以后批处理再支持 `attention_mask` 与每样本不同长度。不要缓存 Padding 激活并注入为有效内容。

若源激活在 CPU/磁盘缓存，注入前转为**写入 Block 所在设备和 dtype**；避免隐式把整个模型拷贝到另一个设备。保存层范数、有限值比例、Token 长度和激活统计，以便诊断异常。

### Step 3 — 构造 Decoder Prefix（激活占位 + 诊断问题）

推荐 `decoder_prompt_version: e1b_diag_v1`：

```text
You are given a claim and evidence through internal representations.
Determine whether the evidence SUPPORTS or REFUTES the claim.
Answer using exactly one label: SUPPORTS or REFUTES.
```

**Decoder 只能看到问题模板，不能看到 Claim/Evidence 原文。** 诊断问题对所有样本固定。

按每条 Encoder 输入的有效长度 `T`：

1. 选定一个普通、单 Token 的占位符 ID（例如由 tokenizer 检验 `"?"` 是否编码为恰好一个非 special token）；记 `slot_id`。
2. 构造 `slot_ids = [slot_id] * T`，**不要用字符串里 `T` 个问号代替**，因为 tokenizer 可能合并它们。
3. 用 Qwen Chat Template 对固定诊断问题编码，并以 `add_generation_prompt=True` 结束；得到 `question_ids`。
4. 构造 `decoder_prefix_ids = slot_ids + question_ids`；`attention_mask` 对所有有效位置设为 1。
5. 明确 `patch_start=0`、`patch_end=T`；问句部分不得被覆盖。

这样的 Decoder 在 Block 0 被 Hook 后，可以在 Block 1–27 的注意力计算中访问被注入的序列状态。若用其他 chat 模板，应将前缀中 BOS/chat special token 的位置与有无双重特殊 Token 问题写入日志。

### Step 4 — 在 Decoder 固定 Block 0 的**输出**写入激活

主实验：`read_layer = ℓ` 变化，`write_layer = 0` 固定；`patch_mode = block_output`，`patch_strength=1.0`（完全覆盖）。

```python
from contextlib import contextmanager
import torch

@contextmanager
def patch_block_output(model, write_layer, source_hidden, start_pos=0):
    # source_hidden: [B, T_source, D]
    source_hidden = source_hidden.detach()
    call_count = 0

    def hook(module, inputs, output):
        nonlocal call_count
        call_count += 1
        h = output[0] if isinstance(output, tuple) else output
        b, t, d = source_hidden.shape
        assert h.shape[0] == b and h.shape[2] == d
        assert 0 <= start_pos and start_pos + t <= h.shape[1]
        replacement = source_hidden.to(device=h.device, dtype=h.dtype)
        new_h = h.clone()
        new_h[:, start_pos:start_pos+t, :] = replacement
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = model.model.layers[write_layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        assert call_count > 0, "Decoder patch hook was not triggered"
```

**工程上必须额外做到**：

- Hook 仅在 Decoder 前向激活；无论异常与否都卸载；
- 捕获/写入的都是 **Block 输出**（同一边界定义）；
- 使用 clone，避免污染源激活或无关计算；
- 主实验评分使用整个 Prefix + Candidate 的**单次前向**，`use_cache=False`，使每次均有完整 `[prefix+candidate]` 序列，避免增量 decode 时 `T` 个占位不在当前前向；
- 如改为逐 Token `generate()`，必须额外处理 KV Cache、Prompt/Past 长度和 Hook 触发时序，不能直接复用这个简化 Hook；
- 出现设备不一致、shape 不一致或非有限值，**报错并记录样本**，不可静默跳过。

### Step 5 — 计算 `SUPPORTS` / `REFUTES` 的序列概率

**主指标不依赖生成文本解析。** 对每一个候选答案 `a` 使用 Teacher Forcing 计算：

\[
 s_\ell(a)=\sum_{j=1}^{|a|}\log P_\theta(a_j\mid
 \mathrm{prefix},a_{<j};\operatorname{Patch}(H_\ell)),
 \quad a\in\{\mathrm{SUPPORTS},\mathrm{REFUTES}\}.
\]

\[
 m_\ell=s_\ell(\mathrm{SUPPORTS})-s_\ell(\mathrm{REFUTES}).
\]

预测：`m_ℓ > 0 → SUPPORTS`；`m_ℓ < 0 → REFUTES`；**严格等于 0 记录为 tie**，不随意分配标签。

正确实现的伪代码（需将 mask、Batch 和 Hook 接口写成可测试模块）：

```python
@torch.inference_mode()
def candidate_logprob(model, prefix_ids, candidate_ids, device):
    # prefix_ids 和 candidate_ids 都是 int token id list
    input_ids = torch.tensor([prefix_ids + candidate_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits.float()
    p = len(prefix_ids)
    # 自回归位移：第 j 个 candidate 在位置 p+j 的 logit 来自 p+j-1
    logp = logits[0, p-1:p+len(candidate_ids)-1, :].log_softmax(dim=-1)
    targets = torch.tensor(candidate_ids, device=logp.device)
    return logp.gather(1, targets[:, None]).sum().item()
```

调用该函数时必须**在相应 Decoder Patch Context 内**；独立计算每个候选的完整分数，且两者使用相同的 Decoder Prefix。标签 ID 建议使用 tokenizer 的 `encode("SUPPORTS", add_special_tokens=False)` 和对应 `REFUTES`，确保它们确实是 Chat Template 完成后**答案位置**的续写；也可统一使用一个固定前导空格，但必须测试续写 Token 边界并写入配置。

注意：

- 不可直接比较两个 label 的**首 Token Logit**，除非验证两标签均为一个 Token 且语义上等价；
- 主指标选定**原始完整序列 log-probability**（概率归一化隐含长度偏差也要报告）；补充报告 token-length normalized margin 作敏感性分析；
- 固定 `temperature`/无采样；**不能**在测试集上拟合截距、阈值或自选 Prompt；
- 要验证 `score_no_patch` 与不带 Hook 的等长 Decoder 前向一致；
- 若在 batch 中直接进行 label scoring，请用正确的标签 Position Mask，不能将 padding 或 Prefix Token 纳入答案损失。

### Step 6 — 循环 28 个 Encoder 读取层

```python
# 概念性主循环：工具函数需在本项目中实现，非可直接执行的 CLI
for sample in dev_samples:
    for condition in ("full", "claim_only", "evidence_only"):
        encoder_ids = encode_context(sample, condition=condition)
        hidden_by_layer = capture_block_outputs(model, encoder_ids)

        for read_layer in range(28):
            source = valid_tokens(hidden_by_layer[read_layer], encoder_ids)
            prefix_ids = build_decoder_prefix(num_slots=source.shape[1])
            with patch_block_output(model, write_layer=0,
                                    source_hidden=source, start_pos=0):
                score_s = candidate_logprob_for_label(prefix_ids, "SUPPORTS")
                score_r = candidate_logprob_for_label(prefix_ids, "REFUTES")
            save_prediction(sample, condition, read_layer, score_s, score_r)
```

在实际主循环中还需要：

- 分别执行 No Patch、Random Donor、Counterfactual Donor；
- 对所有条件使用**同一批样本 ID**；
- 保证每次模型输入与输出的匹配不发生错位；
- 遇到中断时能按 `(sample_id, condition, read_layer, donor_id)` **幂等恢复**；
- 初期单样本运行；后续按 Token 长度分桶后做多 GPU 分片；
- 将原始每条 Margin 保存，不仅保存分类标签与汇总指标。

---

## 6. 对照设置（必须有，不能仅跑 Full Patch）

| 编号 | 条件 | Decoder 使用的激活 | 主要解释用途 |
|---|---|---|---|
| B0 | `direct_text_7b` | 无 Patch；Decoder 直接看到相同 Claim+Evidence | 同一个 7B 的常规事实核查能力 |
| B1 | `no_patch_slots` | 保留相同 T 个占位 Token，但不 Patch | 控制 Prompt、slot 数量、标签先验 |
| B2 | `full_patch` | 当前样本的 Claim+Evidence 激活 | 主要实验 |
| B3 | `claim_only_patch` | 当前 Claim 的激活 | 识别 Claim-only 捷径 |
| B4 | `evidence_only_patch` | 当前 Evidence 的激活 | 检验 Evidence 自身线索 |
| B5 | `random_donor_patch` | 长度匹配的其他 claim 的激活 | 证伪“任何激活均有效” |
| B6 | `cf_donor_patch` | 同 Claim 下相反 Evidence 的激活 | 证据关系反转响应；需记录源/目标关系 |
| B7 | `identity_patch` | 相同输入、相同层、同一位置的**自身激活** | 验证 Hook 正确性（工程测试，不直接等同主 E1-B） |

额外建议：

- **长度对照**：不同条件的 slot 数量会变；`no_patch_slots` 和 Random Donor 必须匹配相应长度，最好做长度分层统计。
- **标签平衡**：Random Donor 按标签/来源家族均衡抽样；记录实际 donor ID、donor label；不能从目标样本本身抽样。
- **源与目标分离**：对于 `cf_donor_patch`，Decoder 的文字问题不含 claim/evidence，因此所谓“target”只是评价样本的目标标签；请同时报告**按 donor gold** 和 **按 target gold** 的结果，避免混淆。若读出主要来自 donor，输出应当随 donor evidence 变化。
- **No Patch 的配对分数相同**：固定问题与相同 Slot 数时，不应人为把方向准确率解释为 0% 或 50%，要按下一节的 tie-aware 协议处理。

### 6.1 自身激活 Identity Patch 的正确做法

`Identity Patch` 必须与正常 E1-B 的跨模型阅读区分：

1. 对**完全相同的 Decoder 输入** `x`，缓存其 `Block k` 输出 `H_k(x)`；
2. 用同样的 `x` 再运行模型，在 `Block k` 的**同一 Token 位置**覆盖为 `H_k(x)`；
3. 未修改输出与 Identity Patch 输出的 logits/候选概率应在数值容差范围内一致；
4. 如果不一致，优先排查 Hook 边界、dtype、位置、dropout/eval、Cache。

这只是接口单元测试，**不证明**“Encoder L19 注入 Decoder L0”本身在分布上合理。

---

## 7. 反事实评价：核心不是单样本 Accuracy，而是 Evidence Flip

对同 Claim 的一对反事实样本：

\[
 x_i^{+}=(C_i,E_i^{+}),\quad y_i^{+}=\mathrm{SUPPORTS}
\]

\[
 x_i^{-}=(C_i,E_i^{-}),\quad y_i^{-}=\mathrm{REFUTES}.
\]

定义 `m^+ = score(S|x^+) - score(R|x^+)`、`m^- = score(S|x^-) - score(R|x^-)`。

### 7.1 单样本指标

- `Accuracy`、`Balanced Accuracy`、`Macro-F1`；分母必须排除“审核不合格”样本并报告计数。
- `per_class_precision`、`per_class_recall`、`confusion_matrix`；特别记录是否**总预测 SUPPORTS**。
- `tie_rate`、`invalid_rate`（用于自由生成标签时）、`mean_margin`、`label_score_distribution`。

### 7.2 配对指标

**方向准确率（Direction，主指标之一）**：

\[
 D_\ell=\frac{1}{N}\sum_i\big[\mathbf1(m_{\ell,i}^+>m_{\ell,i}^-)
 +0.5\mathbf1(m_{\ell,i}^+=m_{\ell,i}^-)\big].
\]

平局按 `0.5` 计算，**同时报告 tie rate**。对于输入完全相同的 Claim-only 两个配对成员，该规则通常得到 50% Direction；不应错误地得到 0%。实际数值运算可以用预先固定的很小 tolerance（例如绝对差 `<=1e-6`）定义 tie，不能根据测试表现调参。

**整对判对率（PairCorrect，主指标之一）**：

\[
 PC_\ell=\frac{1}{N}\sum_i
 \mathbf1(m_{\ell,i}^{+}>0\;\land\;m_{\ell,i}^{-}<0).
\]

若任一 Margin 为平局，不算整对判对。**PairCorrect 比 Direction 要求更严格**；高 Direction、低 PairCorrect 只说明分数方向有信息，不自动证明存在可简单修复的校准偏差。

补充指标：`mean_pair_gap = mean(m^+ - m^-)`、`median_pair_gap`、按家族的置信区间、长度分桶的 PairCorrect、完整 pair 的数目。

### 7.3 置信区间与显著性

- 单位优先按 `source_family`/Claim/文档群组进行 **cluster bootstrap**，而不是对 pair 逐条独立 Bootstrap；
- 固定 Bootstrap 次数（例如 2000）和随机种子；
- 比较 Full 与 Claim-only、No Patch、Random Donor 的**配对差异**并给出 95% CI；
- 开发集只用于发现问题、选层；最终汇总报告必须含独立 test。

### 7.4 层选择（仅使用 dev）

先执行 28 层完整扫描，层选择顺序建议：

1. 检查工程测试通过率与输入审核；
2. 优先选择在 `full` 的 **PairCorrect + Direction** 上表现良好、且相对于 No Patch/Claim-only 有可靠差异的层；
3. 若几层结果接近，保留候选**层区间**，不要只报告单点最优；
4. 把被选层和选择规则存入 `layer_selection.json`，锁定后再跑 test；
5. 不把旧 E1 的 L19 当作预设最佳层；可仅做事先声明的比较对象。

若所有层表现为接近常数标签偏好，判定 **Reader 尚未成功/结果不确定**，不能反推模型内部没有关系信息，更不能直接进入 Steering。

---

## 8. 工程文件与接口建议

项目已有 `src/crestfc/`、`scripts/`、`configs/`、`tests/`，且 E1 线性 Probe 为独立实现。建议以增量方式增加**新模块**，不要覆盖历史 E1 数据和权重：

```text
CReST-FC/
├── docs/
│   └── E1-B-readme.md                     # 本文件（建议最终放在 docs/）
├── configs/
│   └── e1b_frozen_reader.yaml             # 新增
├── src/crestfc/
│   ├── frozen_reader.py                   # 新增：编排 Encoder / Decoder
│   ├── frozen_reader_patching.py          # 新增：安全的注入 Context Manager
│   ├── frozen_reader_scoring.py           # 新增或复用：完整标签序列打分
│   └── frozen_reader_analysis.py          # 新增：配对评价、聚合、图表
├── scripts/
│   └── run_e1b_frozen_reader.py           # 新增：入口
├── tests/
│   ├── test_e1b_capture.py                # 新增
│   ├── test_e1b_slots_and_patch.py        # 新增
│   ├── test_e1b_label_scoring.py          # 新增
│   └── test_e1b_pair_metrics.py           # 新增
└── outputs/experiment_runs/
    └── e1b_frozen_reader_<run_id>/         # 运行时产生
```

这些**文件名与 CLI 是实施建议，并不意味着当前仓库已经存在**。在正式提交前应优先复用当前 `crestfc` 已实现的模型加载、打分和验证工具，避免创建两套定义不一致的指标。

### 8.1 主要函数接口（建议）

```python
load_frozen_model(model_id, revision=None, dtype="bfloat16")
validate_paired_dataset(samples, require_reviewed=True)
build_encoder_ids(sample, condition, tokenizer, prompt_version)
capture_block_outputs(model, encoder_inputs) -> dict[int, Tensor]
build_decoder_prefix(tokenizer, num_slots: int, question: str) -> list[int]
patch_block_output(model, write_layer: int, source_hidden: Tensor, start_pos: int)
score_binary_labels(model, prefix_ids, labels=("SUPPORTS", "REFUTES")) -> dict
run_layer_scan(samples, config) -> list[dict]
evaluate_pairs(predictions, group_by="source_family") -> dict
save_manifest_and_predictions(run_dir, config, results)
```

函数名固定后，为数据对照、源激活与预测中间结果制定共同键：`(sample_id, condition, read_layer, write_layer, donor_id, prompt_version)`。

### 8.2 YAML 配置草案

```yaml
experiment: e1b_frozen_reader
seed: 42
model:
  model_id: Qwen/Qwen2.5-7B-Instruct
  revision: null               # 正式实验建议填写具体权重提交版本
  dtype: bfloat16
  freeze: true
  use_cache: false
  trust_remote_code: false
  same_weights_for_encoder_decoder: true

data:
  path: data/processed/REPLACE_WITH_REVIEWED_DEV.jsonl
  split: dev
  labels: [SUPPORTS, REFUTES]
  require_reviewed: true
  require_paired: true
  max_pairs: 100               # pilot 用；formal 设置 null
  evidence_type: any          # 聚合时按 retrieved/curated/oracle 分开报告

prompts:
  encoder_version: e1b_ctx_v1
  decoder_version: e1b_diag_v1
  use_chat_template: true
  use_relation_anchor: false

activation:
  read_layers: all            # 具体实现时解析为 0..27
  read_boundary: block_output
  read_tokens: full_valid_sequence
  write_layer: 0
  write_boundary: block_output
  slot_start: 0
  slot_token_text: "?"       # 必须在启动时验证是单个正常 Token
  patch_strength: 1.0
  batch_size: 1
  attention_mask: full_valid_tokens

scoring:
  labels: [SUPPORTS, REFUTES]
  method: full_sequence_logprob
  primary_normalization: sum
  secondary_normalization: mean_per_token
  tie_tolerance: 1.0e-6
  no_calibration: true

controls:
  - direct_text_7b
  - no_patch_slots
  - full_patch
  - claim_only_patch
  - evidence_only_patch
  - random_donor_patch
  - cf_donor_patch
  random_donor_length_match: true

analysis:
  metrics: [macro_f1, balanced_accuracy, pair_correct, direction, tie_rate]
  bootstrap_group: source_family
  bootstrap_reps: 2000
  select_layers_on: dev

output:
  root: outputs/experiment_runs/e1b_frozen_reader_pilot
  save_per_sample_scores: true
  save_raw_activations: false  # 初期避免巨大磁盘开销
  resume: true
```

正式实现应在运行前进行 config 校验，例如 `read_layers` 必须解析到 `[0,27]`，主实验 `write_layer=0`、`patch_strength=1`，标签二分类且均为有效续写 Token 序列。

### 8.3 已实现的运行入口

```bash
# 来源标签未经审核，仅用于工程诊断；启动时自动执行 frozen/identity/Hook 检查。
ALLOW_UNREVIEWED=1 MAX_PAIRS=2 LAYERS=0,19 GPU_TOKEN=4 \
  bash scripts/run_e1b_frozen_reader.sh e1b_smoke

# 正式 dev/test 运行需要 REVIEW_FILE；test 还需要 LOCKED_DEV_SELECTION。
# 完整命令和输入格式见 docs/E1B_FROZEN_READER.md。
```

---

## 9. 产物格式与复现记录

运行目录建议如下：

```text
outputs/experiment_runs/e1b_frozen_reader_pilot/
├── config_resolved.yaml
├── manifest.json                       # commit/revision/hardware/data hashes
├── data_audit.json                      # 标签、split、证据审查情况
├── sanity_checks.json                   # Hook/身份补丁/打分测试
├── predictions.jsonl                    # 每条样本×读取层×对照的原始分数
├── pairs.csv                            # 每一 pair × 读取层的 m_plus/m_minus
├── metrics_by_layer.csv                 # 单样本指标、配对指标、控制条件
├── bootstrap_ci.csv                     # 家族 Bootstrap 的 95% CI
├── layer_selection.json                 # dev 选择结果及规则
├── figures/
│   ├── macro_f1_by_layer.png
│   ├── pair_correct_by_layer.png
│   ├── direction_by_layer.png
│   └── margin_distributions.png
└── logs/
    └── run.log
```

**`predictions.jsonl` 一行样例**：

```json
{"sample_id":"pair_001_s","pair_id":"pair_001","split":"dev","condition":"full_patch","read_layer":19,"write_layer":0,"write_boundary":"block_output","num_source_tokens":63,"num_slots":63,"donor_id":"pair_001_s","score_supports":-2.15,"score_refutes":-3.80,"margin":1.65,"predicted_label":"SUPPORTS","gold_label":"SUPPORTS","prompt_version":"e1b_diag_v1"}
```

此处数值为**示例格式**，不是实验结果。建议另记录候选 Token 数、各标签的长度归一化分数、占位/Prompt Token 数、每次 Hook 触发次数、时间和峰值显存。

---

## 10. 单元测试与验收门槛

### 10.1 必须通过的工程测试

- [ ] **Frozen 权重**：所有 `requires_grad=False`，无优化器/训练过程；相同输入多次评分重复一致（在设置的容差内）。
- [ ] **层索引**：读取 exactly 28 个 Transformer Block 输出，`L0` 明确代表第一 Block 的**输出**，非 Embedding。
- [ ] **有效 Token**：读出长度与 attention mask 一致；无 Padding 误注入；占位个数等于读出 Token 个数。
- [ ] **Hook 对齐**：写入的确为 Decoder Block 0 的**输出**，且只覆盖预期 `[0:T]`；注入位置的 `max(abs(diff))` 在 dtype 容差内为 0。
- [ ] **Hook 清理**：普通前向、异常前向后均无残留 Hook；Encoder 捕获不会受到上一轮 Decoder Hook 污染。
- [ ] **Identity Patch**：同输入同层同位置 Patch 后，相较无 Patch 的 logits 差异在设定容差内（例如 BF16 下合理的浮点误差）。
- [ ] **No Patch 等价性**：No-Patch 模式（相同 slot/prefix）与不注册 Hook 的前向输出一致。
- [ ] **Label Scorer**：手工用 logits 校验 multi-token teacher forcing 的 off-by-one 索引；不同 Token 长度均正确。
- [ ] **Decoder 无泄漏**：Decoder 原始 `input_ids` decode 后不包含 Claim、Evidence 或 Gold；未通过则失败。
- [ ] **输入长度**：随机长度与固定 Claim 配对的 Full/Claim/Evidence 三条件有效；截断必须显式标记。
- [ ] **配对统计**：方向、PairCorrect、tie-aware 的单元测试结果与手算完全相同。
- [ ] **重复运行**：相同配置、同一小样本得出相同序列分数；断点续跑不重复计数。

### 10.2 必须完成的科学对照

- [ ] 7B Direct Text 基线与 Frozen Reader 使用同一批数据、标签和模板版本记录。
- [ ] Full、Claim-only、Evidence-only、No Patch、Random Donor 在**相同样本集合**上评价。
- [ ] 固定 Claim 的 Evidence Flip 同时评价 **Direction、Tie Rate 和 PairCorrect**。
- [ ] 统计来源家族的 Cluster Bootstrap，报告样本、pair、家族计数。
- [ ] 研究结论不以 E0-0.5B 的 22.75% 对比 E1-B-7B 的 Macro-F1 作为模型内部能力差距。
- [ ] 不在 test 上改 Prompt、阈值、Patch 层、预测解析规则。

### 10.3 停止条件与故障定位

| 观察 | 优先排查 | 允许的结论 |
|---|---|---|
| 所有层输出同一标签 | 标签续写、Prompt、No Patch、Patch Hook 是否命中 | Reader 暂未体现有效关系读取；结论不确定 |
| Full 与 No Patch 完全一致 | source slot 对齐、激活实际是否覆盖、是否保存错结果 | 不能声称 Patch 有效 |
| 改变来源激活后分数有变化，但错误方向 | 输入标签审核、Donor 对齐、Cross-layer 分布 | 不可声称关系读取成功 |
| Direction 高但 PairCorrect 低 | 标签偏置、序列长度偏置、候选分数与 Prompt | 可报告排序敏感，不能直接归因“仅需截距修复” |
| 只有少数层异常高 | 同一批样本对照、Source 长度分布、数据泄漏 | 需要额外消融确认 |
| 主实验整体较差 | 尝试同层写入、不同写入层或 Span-only 作为**单独诊断实验** | 不能反证内部没有相关信息 |

---

## 11. 运行顺序与时间/资源控制

**阶段 A：仅验证接口（不看论文性能）**

1. 准备 5–10 条手工样本；
2. 用 1 条数据加载 Qwen2.5-7B，测试 `capture_block_outputs` 的 28 个 Shape；
3. 构造 Decoder 占位 Token，确认 `prefix_len = T + question_len`；
4. 做 Same-Input Identity Patch / No Patch；
5. 手工验证 `SUPPORTS/REFUTES` 的 Teacher Forcing 分数；
6. 排除 Hook 清理、Pad/position、logits 时序、device 错误。

**阶段 B：小样本扫描**

1. 先以 20 对反事实样本扫描 28 层的 Full 与 No Patch；
2. 检查完整原始 Margin 曲线，不只看预测标签；
3. 增加 Claim-only、Evidence-only、Random/CF Donor；
4. 若所有层始终为常数输出，先诊断，不要直接扩大到数千对。

**阶段 C：开发集统计**

1. 扩大到至少 200–500 对人工审核 pair；
2. 输出层曲线、家族置信区间和负对照；
3. **只根据 dev** 预先定义且执行层选择；
4. 冻结 Prompt、注入层、输出阈值与打分代码。

**阶段 D：独立测试**

1. 只使用锁定的配置，评价未参与开发集选择的 test；
2. 在测试集同时给出 Full 与所有关键控制条件差值；
3. 与原 E1-L19 Probe 可解码性、7B Direct Text 和 E2 后续因果实验**分层解释**。

资源提示：读取全层 Encoder 激活可**每条样本一次 Encoder 前向**缓存 28 层，再复用进行多次 Decoder 前向。避免把 FEVER 全训练集的全序列激活长期保存在磁盘。早期 `batch_size=1` 方便排查；真实批处理时按 `T` 分桶、谨慎处理前置 Padding 与 RoPE/position_ids。独立运行应保证模型实例和 Hook 不跨进程共享。

---

## 12. 结果表与论文解释模板

### 12.1 必备结果表（数值运行后填写）

| Layer | Full Macro-F1 | Full PairCorrect | Full Direction | Claim-only PairCorrect | Random Donor PairCorrect | No Patch F1 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| L0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... | ... |
| L18 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| L19 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| L27 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**论文中合理的论述**：

> 在不更新任何参数的情况下，将冻结 Qwen2.5-7B-Instruct 第 ℓ 层的上下文序列激活传递给具有相同权重的冻结 Decoder，可以在特定层取得高于 No Patch/Claim-only/Random Donor 的关系预测性能；在固定 Claim 的反事实配对上，关系 Margin 对 Evidence 变化呈现对应方向的响应。这表明隐藏状态经由所设计的冻结读取接口含有可用于关系判断的信息。

**不能直接写的论述**：

- “该层已经完成全部事实核查推理”——Decoder 可能在后续层重新计算关系。
- “原模型自然生成阶段一定使用该关系信息”——Frozen Reader 的读取接口不是原始生成路径。
- “L19 是因果关键层”——E1-B 即使最高，也必须通过 E2 干预验证。
- “E0 22.75% → E1-B 90% 是生成阶段偏差造成”——0.5B 与 7B、二类与三选项、监督与无监督等需要公平基线。

### 12.2 与 E2/E3 的衔接

- 如果 E1-B 的若干层明显优于负对照，在这些层及其前后设计 **E2 原模型内部 Patching**，检验关系信息对正常决策的因果作用。
- 如果 E1-B 读出好、E2 原模型 Patching 效果差，需检查**介入位置、跨层信息通道和自然路径**，不能直接断言“关系表示到决策错位”。
- E3 Steering 的选择应综合 E1-B 可读取性、E1 监督可解码性、E2 因果效应与稳定性；不要仅根据单个最高 Macro-F1 层选择。

---

## 13. 开发任务清单（按优先级）

### P0：数据和协议

- [ ] 固定 `e1b_ctx_v1`、`e1b_diag_v1`；冻结标签映射及续写评分方式。
- [ ] 审查配对是否同 Claim、相反标签且 Evidence 充分；建立 dev/test 家族隔离。
- [ ] 明确 Qwen 28 层 Hook 边界、占位 Token、Chat Template 和标签 Token IDs。

### P1：最小核心

- [ ] 实现全层 Encoder 捕获（全部有效 Token）。
- [ ] 实现 Block 0 输出替换 Hook 和异常清理。
- [ ] 实现两个完整标签的 Teacher-Forced Log-Likelihood。
- [ ] 完成 Identity、No Patch、Hook 位置、Label Scorer 单元测试。

### P2：实验主循环

- [ ] 实现 28 层读取、全套条件、pair ID/Donor 对齐与增量保存。
- [ ] 实现 Macro-F1、PairCorrect、Direction、Tie Rate。
- [ ] 完成 20 对/100 对规模诊断与错误案例核查。

### P3：正式分析

- [ ] 扩大 dev，来源家族 Bootstrap，固定候选层与协议。
- [ ] 仅在冻结配置之后评估 test。
- [ ] 对比 7B Direct Text 和原 E1/E1.5/E2，形成可读取性–因果利用–下游表现的证据链。

---

## 14. 实现时最容易踩的坑（请保留在 Code Review Checklist）

1. **不是训练 Probe**：不跑 `fit()`、`StandardScaler.fit()` 或任何 LoRA；不借助 dev/test label 学习决策边界。
2. **不是直接输出 `lm_head(h_ℓ)`**：必须经过冻结 Decoder 的后续计算。
3. **不是只取 `final_anchor`**：主协议是整段有效 Token；Anchor-only 单独报告。
4. **不是同层 Patch**：主协议 `read_layer` 变化、`write_layer=0` 固定；跨层分布不匹配是重要限制。
5. **不是 Decoder 还能看原文**：必须检查解码后的 Decoder `input_ids`，防止 Claim/Evidence 泄漏。
6. **不是一个字节等于一个 Token**：Slot 数按 Encoder Token 长度，不按字符串长度。
7. **不是只看生成标签**：完整标签序列 teacher forcing 为主，生成文本为辅。
8. **不是忽略自回归位移**：首个答案 Token 的 Logit 是 `prefix_len-1` 位置。
9. **不是默认 Pair Direction 能在无信息时得到 50%**：相同 Margin 会产生 ties；要定义并报告 tie-aware 指标。
10. **不是忽略 Evidence 的审核类型**：retrieved evidence 不能自动视为足以判定 gold。
11. **不是靠一两个层的高分选层**：dev-only 选择、家族 CI、原始 margin 和负对照同时检查。
12. **不是成功 Locate 就等于成功去偏**：原始模型 E2 因果验证和 E3/E4 才涉及最后的因果使用与去偏性能。

---

## 15. 完成定义（Definition of Done）

只有当以下全部完成，才将 E1-B 标为 **完成**：

1. 冻结 Qwen2.5-7B，28 层全序列激活读取 + 固定 Decoder L0 写入，跨样本无 Hook 泄漏；
2. 完整标签概率评分通过单元测试，预测与 Margin 可复现；
3. 数据审核和 split/家族防泄漏可追溯；
4. Full/Claim/Evidence/No Patch/Random Donor/Counterfactual Donor 对照齐全；
5. 同 Claim 反事实配对的 Direction、Tie Rate 和 PairCorrect 均报告；
6. 采用相同 7B 的 Direct Text 公平基线；
7. dev 选层、test 独立评估，按家族 Bootstrap 置信区间；
8. `manifest`、原始预测分数、逐层表、图表和实验配置完整归档；
9. 报告严格使用 **“冻结 Reader 的关系可读取性”**，不将其等同于因果定位或已经完成去偏。

**建议首先实现最小闭环**：10 条数据 → 28 层 Encoder 全序列读取 → L0 注入 → 两个标签 Teacher Forcing → Identity/No Patch 测试。闭环正确后再拓展到配对对照与正式评估。
