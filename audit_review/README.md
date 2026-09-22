# 审查报告索引

本目录收录对本项目物理理论与计算正确性的审查报告，以及可复现的验证脚本。

## 报告

| 文件 | 基线 | 发现 | 说明 |
|---|---|---|---|
| [`scientific_review.md`](scientific_review.md) | `5c41c06` | F1–F10 | 首轮审查。确认 3 项高影响问题：参考能量按粒子数误判、SEM 忽略链自相关、默认截断梯度被表述为「严格无偏」 |
| [`independent_physics_audit.md`](independent_physics_audit.md) | `97c7ca3` | D1–D12 | 独立复核，**未参考前一份报告**。判定来自源码逐行核对、解析可解模型对照、对称性变换实验与端到端复评四类证据 |

两份报告独立完成，编号体系不同（`F` 与 `D`），可按问题交叉对照。

## 当前修复状态

**D1–D4、D6–D12 已修复；D5 部分修复且原结论已更正**（完整核序置换不变性需改为对核做对称聚合，属架构改写，未纳入范围）。

全部 12 项发现均已有结论，测试套件全绿（173 项，1 项按环境能力显式跳过）。修复详情、实测数值与仍然成立的限制见 `independent_physics_audit.md` 各节。

**特别提示**：该报告 §3.5 记录了一个被实验否决的方案——用「按距离排序核轴」实现核序置换不变性会使波函数在垂直平分面上不连续，导致能量跌破变分下界。该方案已回退，并有回归测试锁定连续性。

## 复现脚本

| 文件 | 用途 |
|---|---|
| `verify_physics.py` | 物理量验证：势能/动能与解析式对照、Hutchinson 单探针方差测量、距离软化偏差量化 |
| `run_training_checks.py` | 多配置短训练复评，产出 `baseline_results.json` 与 `training_results.json` |
| `build_report_docx.py` | 由 `scientific_review.md` 生成 DOCX（**产物不入库**） |

## 数据

`results.json`、`baseline_results.json`、`performance_results.json`、`training_results.json` — 上述脚本的实测输出，作为报告结论的原始记录。

## 不入库的生成物

`rendered/`（PDF 与逐页 PNG）与 `*.docx` 均由脚本生成，见 `.gitignore`。需重新产出时运行 `build_report_docx.py` 及渲染流程。
