# Kaggle 双 T4：H₂ 多种子训练与冻结评估最终结果

日期：2026-10-22。六个实验均完成 20000 轮训练和独立冻结评估，Kaggle 任务状态为 `COMPLETE`。本次配置下，FermiNet 三个种子的平均参考偏差为 **1.796 mHa**，Psiformer 为 **3.916 mHa**；仅 FermiNet seed 11 通过项目化学精度判据。尚未证明跨种子稳定达到化学精度，也不能据此推广到其他分子或网络规模。

## 实验身份与配置

- 任务：[ferminet-t4x2-multiseed](https://www.kaggle.com/code/hwbwhzbsh/ferminet-t4x2-multiseed)，version 2。
- 固定源码：[d2618dfb6780f1da49f0bb4ab247d29ec15e41f8](https://github.com/spirlness/ferminet-pytorch/commit/d2618dfb6780f1da49f0bb4ab247d29ec15e41f8)。这是实验源码，后续文档提交不改变该身份。
- 运行前已通过真实双 T4 的两个主干各 1 epoch 验收，见[回执](physics_evidence/kaggle_one_epoch_v2/gate_receipt.json)。
- 运行环境：两张 Tesla T4，Python 3.13.15、PyTorch 2.11.0+cu128、CUDA runtime 12.8；一张卡顺序执行 FermiNet 三个种子，另一张卡执行 Psiformer 三个种子，无 DDP。
- 主进程启动于香港时间 **16:47:56**，最后一个实验完成于 **21:57:42**，两者相隔约 **5 小时 10 分钟**，不含启动前环境准备和最后的 notebook 导出。

实际配置与源码文件哈希以归档的 [campaign.json](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/campaign.json) 和各实验 `manifest.json` 为准；可复用配置见 [configs/kaggle_t4x2.json](configs/kaggle_t4x2.json)。

| 项目 | 实际设置 |
|---|---|
| 物理体系 | 固定核、非相对论 H₂，核间距 1.4 bohr，单重态，能量含核间排斥 |
| 参考能量 | −1.174475714220 Ha；匹配 H₂ BO 单重态与上述几何，来源由结果文件记录为 [Phys. Rev. A 82, 032509](https://doi.org/10.1103/PhysRevA.82.032509) |
| 主干与种子 | FermiNet、Psiformer，各 11、22、33，共六次独立优化 |
| 模型规模 | 8 行列式、3 层、单/双电子维度 64/16，Gaussian 包络、Kato Jastrow、单重态投影 |
| 训练 | FP32，512 walkers，每轮 10 步 MCMC，预热 500 步；精确反向动能，不使用 AMP、TF32 或 Hutchinson |
| 优化 | Adam，主干初始学习率 1e-3；前 1000 轮冻结 Jastrow，随后解冻并使用 1e-4 学习率；分阶段余弦调度，梯度裁剪 1，tanh 能量裁剪倍数 5 |
| 停止与 checkpoint | 六组均因达到 20000 轮结束，未触发 8400 秒训练预算；预先固定选用最后 checkpoint，未按训练或评估能量挑选 |
| 冻结评估 | 存储权重精确提升到 FP64，新采样器、新随机种子，预热 1000 步；生产采样冻结提议自适应，不裁剪或丢弃局域能量 |
| 评估样本 | 每组 2048 条时间记录 × 512 walkers = 1,048,576 个样本，每条前推进 20 步；局域能量分批 128 walkers，保留时间/链结构 |
| 评估随机种子 | FermiNet：100011/100022/100033；Psiformer：200011/200022/200033 |

## 每个种子的冻结评估精度

表中 SEM 为考虑自相关与分块诊断后的 Monte Carlo 标准误差；1 mHa = 0.001 Ha。“95% 误差判据量”为 `abs(E − E_ref) + 1.96 × SEM`，须不超过 **1.5936 mHa**，同时通过采样诊断和数值精度审计，才标记化学精度达标。它不是严格的全局误差上界。

| 模型 | 种子 | 冻结能量（Ha） | SEM（mHa） | 参考偏差（mHa） | 95% 误差判据量（mHa） | 化学精度 |
|---|---:|---:|---:|---:|---:|---|
| FermiNet | 11 | −1.17310604 | 0.055406 | 1.369670 | 1.478266 | 达到 |
| FermiNet | 22 | −1.17214550 | 0.066345 | 2.330213 | 2.460249 | 未达到 |
| FermiNet | 33 | −1.17278876 | 0.069309 | 1.686956 | 1.822802 | 未达到 |
| Psiformer | 11 | −1.17034287 | 0.106608 | 4.132845 | 4.341797 | 未达到 |
| Psiformer | 22 | −1.17155318 | 0.092280 | 2.922536 | 3.103405 | 未达到 |
| Psiformer | 33 | −1.16978325 | 0.118661 | 4.692462 | 4.925039 | 未达到 |

每组均有 **0 个非有限评估能量**；采样诊断全部通过，能量 split-Rhat 范围为 **1.0000047–1.0000572**，ESS 约 **94.1 万–101.8 万**，六组均检测到分块平台。每组训练前后在 32 个指定构型上完成 FP32/FP64 对数波函数、符号和局域能量审计；结束审计全部通过，局域能量 RMS 差为 **3.47e-7–6.20e-7 Ha**。这些构型上的数值审计不代表全空间误差界；能量诊断通过也不证明所有波函数观测量均已混合。

## 跨种子结论与证据边界

| 指标 | FermiNet | Psiformer |
|---|---:|---:|
| 完成实验数 | 3/3 | 3/3 |
| 化学精度通过数 | **1/3** | **0/3** |
| 三个 checkpoint 的能量均值（Ha） | −1.17268010 | −1.17055977 |
| 平均参考偏差（mHa） | **1.795613** | **3.915948** |
| checkpoint 能量的种子间 SEM（mHa） | 0.282557 | 0.522316 |
| 种子均值的描述性 95% 区间（Ha） | [−1.17389585, −1.17146436] | [−1.17280711, −1.16831242] |

区间沿用实际摘要的算法：Student t、自由度 2，使用种子间 SEM 与单组评估 SEM 合成值中的较大者。三个 checkpoint 对应三个不同波函数；能量均值只是优化重复性的描述，未合并三组轨迹，也不代表一个新波函数的精度。仅三个种子不足以证明架构的普遍优劣。

在本次固定配置中，FermiNet 在三个对应种子上均得到更低的冻结能量，但只有一个种子通过化学精度判据。各组参考偏差显著大于自身采样 SEM，说明单纯增加冻结采样不能消除当前点估计的偏差；改善优化或表示能力是后续可检验的方向，本实验没有区分两者的贡献。训练中的 tanh 裁剪改变梯度目标，也应作为后续对照变量。训练批次最低能量没有用于这里的精度结论。

## 原始证据与复核

- [最终摘要](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/summary.json)：六组状态、完整评估诊断、跨种子区间和通过判定。
- 各实验原始结果：[FermiNet 11](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/ferminet_seed11/result.json)、[22](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/ferminet_seed22/result.json)、[33](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/ferminet_seed33/result.json)；[Psiformer 11](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/psiformer_seed11/result.json)、[22](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/psiformer_seed22/result.json)、[33](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/psiformer_seed33/result.json)。
- [完整控制台日志](physics_evidence/kaggle_t4x2_long_v2/console.log)：含六个 `case_completed` 事件；[环境](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/bootstrap.json)、[实际配置与源码哈希](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/campaign.json)。
- [GPU 0 运行时测试](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/runtime_tests_gpu0.log)、[GPU 1 运行时测试](physics_evidence/kaggle_t4x2_long_v2/ferminet_results/runtime_tests_gpu1.log)：各 3 项通过，无跳过；这不是重新运行完整 267 项本地测试套件。
- [归档文件清单及 SHA-256](physics_evidence/kaggle_t4x2_long_v2/archive_manifest.json)、[归档范围](physics_evidence/kaggle_t4x2_long_v2/README.md)。原始 JSON 按下载字节保留；控制台日志由 Kaggle 返回的日志条目解码得到。

本仓库直接保存结果、清单及日志；大型 checkpoint 和原始评估轨迹 `.pt` 已另行下载到本地完整归档，位置和 SHA-256 见[完整产物归档说明](physics_evidence/kaggle_t4x2_long_v2/FULL_ARTIFACT_ARCHIVE.md)。归档包含全部 61 个远端产物及固定源码，不随 Git 克隆分发。后续核验已对六份最终 checkpoint 比对结果文件公布的哈希，并使用固定统计源码从六份轨迹复算能量、SEM 和采样诊断，均与原报告匹配，见[产物核验记录](physics_evidence/kaggle_t4x2_long_v2/full_artifact_validation.json)。此次没有重新训练；使用同一算法复算不构成独立物理验证。单次评估区间为近似统计区间，不包含优化误差，也不保证采样遍历性。
