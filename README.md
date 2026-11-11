# 原始数据路径清单

此分支记录 CReST 使用的数据集在当前机器上的位置。数据文件未提交到 Git；以下绝对路径仅在相应文件系统挂载后可用。目录中也可能含有转换、检索或路由结果，使用原始数据时应进入标明的原始数据子目录。

| 数据集 | 项目中的链接 | 本机数据位置 | 原始数据位置或说明 |
| --- | --- | --- | --- |
| EXFEVER | `data/EXFEVER` | `/mnt/data/yangjun/data/EXFEVER` | `data/EXFEVER/raw/` |
| FEVER | `data/FEVER` | `/mnt/data/yangjun/fact/EGR-FV/data/FEVER` | `data/FEVER/raw/` |
| FEVEROUS | `data/FEVEROUS` | `/mnt/data/yangjun/data/FEVEROUS` | `data/FEVEROUS/claims/`；该目录是本地提供的声明数据，是否为官方原始发布文件需另行核实 |
| HOVER | `data/HOVER` | `/mnt/data/yangjun/data/HOVER` | `data/HOVER/data/raw/`；语料另见 `data/HOVER/corpus/` |
| PolitiHop | `data/PolitiHop` | `/mnt/data/yangjun/data/PolitiHop` | `data/PolitiHop/data/`；含 train、valid、test TSV |
| PolitiHopEGR | `data/PolitiHopEGR` | `/mnt/data/yangjun/fact/EGR-FV/data/PolitiHop` | `data/PolitiHopEGR/raw/`；与 PolitiHop 本地目录不同 |
| VitaminC | `data/VitaminC` | `/mnt/data/yangjun/data/VitaminC` | 当前目录没有 `raw/` 子目录；`converted_data/` 和 `analyze/` 是处理结果，原始文件来源需另行核实 |

本清单只记录当前机器上的路径，不保证其他机器可以直接访问，也不证明这些文件的上游发布来源。
