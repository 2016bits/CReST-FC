# VitaminC Evidence Flip 人工审核页面

打开 [`outputs/experiment_runs/e2_vitaminc_review_queue_300/review.html`](../outputs/experiment_runs/e2_vitaminc_review_queue_300/review.html)。页面是独立 HTML，可直接在浏览器中离线打开；无需上传数据、启动服务或安装前端依赖。

如果 IDE 运行在远程服务器、浏览器无法直接打开服务器文件，可在仓库根目录执行以下命令，并通过 IDE 转发 8765 端口，在本机浏览器访问 `http://localhost:8765/outputs/experiment_runs/e2_vitaminc_review_queue_300/review.html`：

```bash
/data/yangjun/.conda/envs/crestfc/bin/python -m http.server 8765 --bind 127.0.0.1
```

页面并排显示同一 Claim 的 SUPPORTS/REFUTES 来源证据，黄色高亮两者的文本差异。相似度和高亮仅帮助查找变化，不能替代事实判断。审核员逐项选择“通过 / 不通过 / 待定”：Claim 不变、Evidence 实质变化、左证据支持、右证据反驳、两侧证据充分。五项都通过、填写审核员姓名并主动点击“批准”后，该条记录才会导出 `approved: true`。至少一项不通过可驳回；难以判断的样本标记待讨论。备注可写驳回原因。

更改任一审核条件会把该条记录恢复为“未完成”，需要重新提交结论；“撤销结论”可保留已填写的条件与备注。左侧支持搜索和按状态筛选，J/K 可切换样本。

页面尝试将进度自动保存在当前浏览器的本地存储，但浏览器隐私设置可能限制本地文件的存储。请定期点击“导出 JSONL”，保存下载的 `vitaminc_dev_300_reviews.jsonl`；重开页面时可用“导入进度”恢复。导出含 300 条记录，未审核或待讨论的记录保持 `approved: false`。浏览器导出文件可直接作为 `REVIEW_FILE` 传给现有 E1-B/E2 脚本；两者只接纳实际批准且五项都为 `true` 的配对。`review_sheet.csv` 只供打印或在表格软件中查看，实验入口读取的是 JSONL。

如需重新生成页面：

```bash
/data/yangjun/.conda/envs/crestfc/bin/python scripts/build_evidence_review.py
```

生成器核对 `pairs.jsonl` 与 `review_queue.jsonl` 的 Pair ID、Claim、Evidence、来源标签及家族是否一致，再嵌入页面。网页只用 `textContent` 显示样本文本，避免将证据当成 HTML 执行。下载文件包含配对源 SHA-256；导入时检查来源及原文一致性。

审核完成后，可使用下载的 JSONL 启动冻结 E1-B 开发集扫描，例如：

```bash
PAIR_SOURCE=outputs/experiment_runs/e2_vitaminc_review_queue_300/pairs.jsonl \
REVIEW_FILE=/absolute/path/vitaminc_dev_300_reviews.jsonl \
GPU_TOKEN=4 bash scripts/run_e1b_frozen_reader.sh e1b_vitaminc_evidenceflip_reviewed
```

正式层选择还要求至少 200 对已批准样本及完整 28 层扫描。审核清单来源于开发集；测试集须另行准备并保持家族隔离。
