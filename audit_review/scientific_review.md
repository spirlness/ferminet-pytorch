# FermiNet PyTorch 物理理论与计算实现审查报告

审查日期：2026-09-21  
审查基线：`5c41c06138af5cd3dfddb850ff4a238aac4e93a2`  
标准：科研可信度审查；短训练只作诊断，不视为论文级复现

## 结论

项目实现了一个可以工作的简化 FermiNet/VMC 原型。库仑势的符号和计数、对数波函数局域动能公式、同自旋行列式反对称性、对称高斯提议的 Metropolis 接受率、模型更新后的采样缓存刷新，以及 `exact`/实际 JVP Laplacian 的核心计算均通过了源码核对和独立实验。109 项现有测试全部通过；H2 六组短训练在训练后独立复评得到 -1.1648 至 -1.1740 Ha，表明基本链路有效。

不过，当前版本不应被称为“工业级”或论文级 FermiNet 复现。审查确认 3 项高影响问题、5 项中影响问题和 2 项低影响问题。最严重的是：体系参考能量仅凭核数和电子数推断，可能给错误分子画出错误“精确能量”线；SEM 将相关的 MCMC 样本当作独立样本；默认截断梯度仍被文档称为“严格无偏”。此外，旋转等变、精确 cusp、JVP 为 O(1)、单探针 Hutchinson 可保持高精度等表述均超出实现证据。

## 发现总表

| 编号 | 等级 | 判定 | 影响 |
|---|---|---|---|
| F1 | 高 | 已确认错误 | 参考能量和体系名称可能被错误匹配，收敛图产生虚假科学对照。 |
| F2 | 高 | 已确认错误 | `std/sqrt(B)` 忽略链自相关，误差带通常偏窄。 |
| F3 | 高 | 已确认错误的科学表述 | 人口期望梯度公式正确，但同批次中心化有有限样本因子，默认非线性截断改变目标。 |
| F4 | 中 | 已确认错误的科学表述 | 网络平移不变，但不具备一般旋转不变/等变保证。 |
| F5 | 中 | 合理数值近似被过度解释 | `sqrt(r^2+eps)` 软化 Coulomb 奇点并破坏精确 cusp；图中核标线不能证明 cusp。 |
| F6 | 中 | 已确认复杂度表述错误 | JVP 数值精确，但对 3N 个基向量做 `vmap`，不是 Forward Laplacian，也不是总工作 O(1)。 |
| F7 | 中 | 已确认语义错误 | `final_energy` 是最后一次参数更新前的批次估计，返回模型已更新。 |
| F8 | 中 | 已确认语义错误 | 1D/2D 图固定其余电子，是条件切片，不是单电子边缘密度。 |
| F9 | 低 | 已确认边界缺陷 | 行列式权重恰为零时梯度也为零，该分支无法恢复。 |
| F10 | 低 | 已确认过度表述 | 100 电子重合测试得到极小但非零振幅；“精确归零”只在精确算术成立。 |

## 详细发现与修复建议

### F1 参考能量按粒子数量误判

证据：`visualize.py:188-221` 和 `visualize.py:226-244` 只检查 `num_nuclei` 与 `total_electrons`；`train_vmc.py:288-291` 使用同一逻辑。反例使用核电荷 `[2,1]`、核距 4 Bohr、两电子体系，函数仍返回标签 H2 和 -1.17447 Ha。该值只适用于特定电荷、电子数和几何的 H2。

建议：体系必须携带稳定标识；参考能量键至少包含核电荷序列、自旋、核坐标或几何哈希以及参考方法。未知体系返回 `None`，禁止按数量猜测。

### F2 MCMC 的 SEM 被当作独立样本误差

证据：`loss.py:145-158` 直接计算 `std/sqrt(batch_size)`，可视化随后把它标成一倍标准误差带。AR(1) 独立反例中，相关系数 0.9 时朴素 SEM 为 0.00225，500 点分块估计为 0.00978，低估 4.35 倍，与理论膨胀因子 4.36 一致。NetKet 的参考实现同时报告自相关时间、有效样本数和链间诊断。

建议：保留 walker × 时间结构，训练后固定参数采集多步序列；用分块分析或积分自相关时间计算有效样本数和 SEM，并报告链间 R-hat。训练过程中的即时数值改名为 `iid_sem_proxy`。

### F3 “严格无偏梯度”不符合默认实现

证据：`loss.py:136-140` 用同一批样本估计均值并形成协方差，IID 情况下期望为真实协方差的 `(B-1)/B`。20,000 次合成实验在 B=2、4、8、32 时分别得到 0.510、0.748、0.874、0.969，与理论 0.5、0.75、0.875、0.96875 一致。默认 `tanh` 截断又把局域能量作非线性变换；离群点反例中梯度分量最大改变 30.41。官方 FermiNet 也只将截断能量用于梯度，并明确把它描述为稳健处理，而不是原能量梯度的严格无偏估计。

建议：将文档改为“基于 VMC 得分函数恒等式的代理梯度；无截断时渐近一致”。需要有限样本无偏协方差时乘 `B/(B-1)`，并单独说明这仍未处理 MCMC 相关性。默认截断应明确标记为稳健但有偏的训练目标；科学能量始终用未截断样本复评。

### F4 旋转等变声明不成立

证据：`features.py:70-109` 同时输入欧氏距离和笛卡尔位移，后者进入普通全连接层；没有 SO(3) 等变约束。将 H2 体系和电子构型共同平移后，`log|Psi|` 最大变化为 `4.4e-16`；绕分子轴旋转电子构型 60 度后变化为 `0.812`。这直接反驳 `README.md:44,98` 和 `BEGINNER_GUIDE.md:250` 的一般旋转等变表述。

建议：当前架构只声明平移不变与同自旋置换等变。若需要旋转不变，可使用仅标量不变量、局部旋转坐标、数据增强或 E(3)/SO(3) 等变层；之后增加共同旋转回归测试。

### F5 软化距离与精确 cusp 不兼容

证据：势能在 `hamiltonian.py:139,149` 使用 `sqrt(r^2+1e-10)`，特征和包络在 `features.py:77,107`、`envelope.py:134` 使用 `sqrt(r^2+1e-8)`。电子距核 `1e-6` Bohr 时实现势能为 -99,503.7 Ha，真实 Coulomb 值为 -1,000,000 Ha；核上势能被限制为 -100,000 Ha。设置衰减率为 1 的包络在核上的笛卡尔梯度严格为零，而未软化径向指数的右导数为 -1。Kato 条件要求球平均波函数具有由核电荷决定的非零径向导数。

建议：把 `eps` 解释成数值软化尺度，并量化其能量偏差；不要把带软化的核标线图称为 cusp 验证。科研模式应使用不改变非重合点 Hamiltonian 的稳定计算，并用球平均导数和局域能量奇点抵消做独立 cusp 测试；也可显式加入满足 cusp 的因子。

### F6 JVP 精确但不具备所宣称的复杂度

证据：`hamiltonian.py:269-286` 建立大小为 `3N` 的单位阵，对每个基方向求 JVP，再用 `vmap` 并行。它消除了 Python 循环，但仍计算全部对角方向；这不同于 Forward Laplacian 在网络前向过程中传播 Laplacian。解析二次模型上 `exact` 与 JVP 均为零误差，且没有回退警告，所以正确性成立，复杂度表述不成立。

建议：文档将其称为“批量化的逐基方向精确 Hessian 对角计算”，报告时间和峰值内存随 N 的实测缩放。只有实现论文所述算子级 Forward Laplacian 后，才使用该名称和相应复杂度结论。

### F7 返回能量与返回模型不对应

证据：`train_vmc.py:245-257` 先记录当前批次能量，再更新参数并刷新缓存；`train_vmc.py:301` 把更新前的最后批次均值作为 `final_energy`。六组 H2 中，例如 seed 0/tanh 报告 -1.18835 Ha，而更新后独立复评为 -1.17206 Ha。

建议：训练结束后冻结最终参数，重新预热/采样并用精确 Laplacian 评估；返回 `training_last_batch_energy` 与 `final_evaluation` 两个独立字段。

### F8 条件切片被称作电子密度

证据：`visualize.py:911-949` 和 `visualize.py:1053-1089` 固定 N-1 个电子，只扫描第一个电子。输出是配置空间的条件切片，既未对其余电子积分，也未按粒子数归一；`visualize.py:1097` 和文档却使用电子密度措辞。

建议：改名为“固定构型下的波函数条件切片”。一体电子密度应直接从平衡 MCMC 构型做边缘直方图/核密度估计，并附归一化、误差和采样说明。

### F9 零行列式系数成为死参数

证据：`model.py:278-290` 通过 `sign(w)` 与 `log(abs(w))` 组合。实验把第二个权重设为零后，梯度为 `[1.0, 0.0]`，零分支不能恢复。

建议：直接对带符号 logsumexp 写可微自定义算子，或采用不会在零点切断梯度的参数化；增加“零或近零系数可恢复”测试。

### F10 大体系“精确归零”是容差判定

证据：100 原子双精度快速基准中，同自旋电子重合后的 `log|Psi|` 为 -617.3 和 -608.3，符号仍为 -1，而非 `-inf/0`。`benchmark_100_atoms.py:271` 实际将 `log<-100` 视为通过，README 却写“精确归零”。

建议：文档改为“在浮点容差内趋零”。测试同时检查轨道行相同、符号行为和绝对/相对容差，避免将下溢阈值写成数学恒等。

## 已验证正确或合理的部分

- `MolecularSystem` 的电子-核、电子-电子、核-核项符号及 `i<j` 计数正确；软化项之外符合固定核、非相对论、Hartree 原子单位 Hamiltonian。
- `T_L=-1/2[∇²log|Psi|+|∇log|Psi||²]` 对实波函数且远离节点时正确。氢 1s 在 0.1、0.5、1、2 Bohr 的最大局域能量误差为 `5.0e-8` Ha，误差来自势能软化。
- 解析二次波函数上经典反向 Laplacian 和实际执行的 JVP 都达到 float64 机器精度，未触发回退。
- Hutchinson 迹估计在条件期望意义上无偏。500 次单探针平均误差 0.0164 Ha，小于重复均值标准误 0.0239 Ha；但单次标准差 0.535 Ha。
- 对称高斯提议的接受率 `min(1, exp(2Δlog|Psi|))` 正确；逐电子更新后缓存同步正确，参数更新后也会重算缓存。
- 行列式结构在精确算术下保证同自旋交换反对称；现有小体系测试和 100 电子容差测试均通过。
- H2 短训练六组均成功下降；训练后精确复评均值与 -1.17447 Ha 相差约 0.4 至 9.7 mHa，但块标准误仍为 2.8 至 5.6 mHa，不能据此声明化学精度。

## 性能与大体系结论

在 RTX 3060 Laptop GPU、PyTorch 2.14/CUDA 13.0 上，H10、64 walkers 的 Eager 延迟为 3.345 ms；`torch.compile(default)` 为 1.158 ms、2.89 倍加速，`reduce-overhead` 为 1.300 ms、2.57 倍加速。数值最大绝对差为 `1.12e-10`，但当前环境未复现文档的 3.6–5.8 倍范围，且首次默认编译耗时 24.12 秒。

100 原子双精度快速测试可在 6 GB 显存设备运行：前向 4.91 ms，势能 0.97 ms，一阶坐标梯度 630 ms，单探针 Hutchinson 局域能量 1,245 ms。两个 walker 的局域能量分别为 601 和 3,157 Ha；精确 300 维 Laplacian 被 `--quick` 跳过。因此该实验只验证可执行性和部分结构性质，没有证明 100 原子能量准确或训练可收敛。

## 现有测试覆盖缺口

109 项测试全部通过（42.768 秒，无跳过），但缺少以下科研回归：体系身份与几何匹配、相关样本有效样本数、旋转变换、Kato cusp 球平均导数、JVP 不允许静默回退、复杂度缩放、截断前后梯度目标、训练后独立复评、条件切片与边缘密度的区分、零行列式系数恢复。现有 Hutchinson 测试只检查多次均值或有限性，无法约束默认单探针的方差。

## 建议修复顺序

1. 先修 F1、F2、F3，并同步更正文档。它们直接影响能量图、置信区间和“无偏”结论。
2. 修正 F4、F5、F6、F8 的科学措辞和回归测试；架构升级可分后续里程碑完成。
3. 增加训练后独立复评，修复零权重梯度，再收紧大体系节点测试。
4. 将项目定位改为“教学与研究原型”，直到具备固定配置、检查点、独立复评、相关性诊断和论文表格复现实验。

## 可复现实验

```powershell
F:\conda\envs\torch\python.exe -m unittest discover -s . -p "test_*.py" -v
F:\conda\envs\torch\python.exe audit_review\verify_physics.py
F:\conda\envs\torch\python.exe audit_review\run_training_checks.py
F:\conda\envs\torch\python.exe benchmark_compile.py
F:\conda\envs\torch\python.exe benchmark_100_atoms.py --device cuda --double --quick
```

机器可读结果位于 `baseline_results.json`、`results.json`、`training_results.json` 和 `performance_results.json`。训练脚本使用 H2 种子 0、1、42，各运行 tanh/无截断 100 轮；LiH 使用 seed 42、单探针 Hutchinson 运行 20 轮。LiH 复评为 -5.54 Ha，远未接近 -8.0705 Ha，只证明短链路可运行。

## 参考资料与覆盖矩阵

下列 30 个去重来源覆盖原始论文、独立论文、官方源码、数值方法和统计文档。源码版本记录于审查日。

| # | 来源 | 类型/地区 | 本审查用途 |
|---|---|---|---|
| 1 | [Pfau et al. 2020 原始 FermiNet](https://arxiv.org/abs/1909.02487) | 论文/英美 | Ansatz、VMC、基准边界 |
| 2 | [Spencer et al. 2020 Better Faster FermiNet](https://arxiv.org/abs/2011.07125) | 论文/英美 | 简化架构与性能基线 |
| 3 | [Hermann et al. 2020 PauliNet](https://arxiv.org/abs/1909.08423) | 论文/欧洲 | cusp 与物理先验对照 |
| 4 | [von Glehn et al. 2023 Psiformer](https://arxiv.org/abs/2211.13672) | 论文/英国 | 前沿表达能力边界 |
| 5 | [Li et al. 2024 Forward Laplacian](https://arxiv.org/abs/2307.08214) | 论文/中国 | 精确 Laplacian 与复杂度 |
| 6 | [Schätzle et al. 2023 DeepQMC](https://arxiv.org/abs/2307.14123) | 论文/欧洲 | 独立软件流程对照 |
| 7 | [Goldshlager et al. 2024 SPRING](https://arxiv.org/abs/2401.10190) | 论文/美国 | 优化器能力边界 |
| 8 | [Pfau et al. 2024 激发态 VMC](https://arxiv.org/abs/2308.16848) | 论文/英美 | 明确本项目仅支持基态 |
| 9 | [Cassella et al. 2023 周期体系](https://arxiv.org/abs/2202.05183) | 论文/英美 | 平移对称与周期能力边界 |
| 10 | [Qian et al. 2022 力估计](https://arxiv.org/abs/2207.07810) | 论文/中国 | 力与几何扩展边界 |
| 11 | [Gold-standard solutions](https://arxiv.org/abs/2205.09438) | 论文/欧洲 | FermiNet/PauliNet 混合设计 |
| 12 | [Deep learning QMC for solids](https://arxiv.org/abs/2407.00707) | 综述/中国 | 固体和可观测量边界 |
| 13 | [Kato 1957](https://doi.org/10.1002/cpa.3160100201) | 基础论文/日本美国 | Coulomb cusp 条件 |
| 14 | [Flyvbjerg Petersen 1989](https://doi.org/10.1063/1.457480) | 基础论文/欧洲 | 相关数据误差与分块法 |
| 15 | [Hutchinson 1989](https://cir.nii.ac.jp/crid/1360025802809468672) | 基础论文/统计 | 随机迹估计 |
| 16 | [Avron Toledo 2011](https://doi.org/10.1145/1944345.1944349) | 基础论文/以色列美国 | 迹估计误差与样本数 |
| 17 | [Umrigar Filippi 2005](https://arxiv.org/abs/cond-mat/0412634) | 论文/美欧 | 能量/方差优化 |
| 18 | [Toulouse Umrigar 2007](https://arxiv.org/abs/physics/0701039) | 论文/美欧 | VMC 能量最小化 |
| 19 | [DeepMind FermiNet 仓库](https://github.com/google-deepmind/ferminet/tree/c4312c315dda1c5728994ba89629744f71c6eb66) | 官方源码/工业研究 | 总体参考实现 |
| 20 | [官方 loss.py](https://github.com/google-deepmind/ferminet/blob/c4312c315dda1c5728994ba89629744f71c6eb66/ferminet/loss.py) | 官方源码 | 截断与自定义梯度 |
| 21 | [官方 hamiltonian.py](https://github.com/google-deepmind/ferminet/blob/c4312c315dda1c5728994ba89629744f71c6eb66/ferminet/hamiltonian.py) | 官方源码 | 局域能量与 Laplacian |
| 22 | [官方 mcmc.py](https://github.com/google-deepmind/ferminet/blob/c4312c315dda1c5728994ba89629744f71c6eb66/ferminet/mcmc.py) | 官方源码 | MCMC 对照 |
| 23 | [官方 networks.py](https://github.com/google-deepmind/ferminet/blob/c4312c315dda1c5728994ba89629744f71c6eb66/ferminet/networks.py) | 官方源码 | 网络与包络对照 |
| 24 | [DeepQMC 仓库](https://github.com/deepqmc/deepqmc/tree/894feb48e2b1b487e68b8e10d5aa10d1c3dd4f79) | 官方源码/学术 | 独立架构与后处理 |
| 25 | [LapNet 仓库](https://github.com/bytedance/LapNet/tree/65101311fd1da759cd112b5f8f15e71d0520ed83) | 官方源码/工业研究 | Forward Laplacian 实现 |
| 26 | [Microsoft folx](https://github.com/microsoft/folx/tree/eec79b6871502b075b7adaf70c96abddbcfe4b95) | 官方源码/工业研究 | 算子级 Forward Laplacian |
| 27 | [PyTorch 2.14 jvp 文档](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html) | 官方文档 | JVP 语义与限制 |
| 28 | [NetKet statistics](https://netket.readthedocs.io/en/stable/api/_generated/stats/netket.stats.statistics.html) | 官方文档/学术开源 | 自相关时间、SEM、R-hat |
| 29 | [NetKet MCMC convergence](https://netket.readthedocs.io/en/stable/user-guides/mc_convergence.html) | 官方文档/学术开源 | 链收敛诊断 |
| 30 | [DeepQMC releases](https://github.com/deepqmc/deepqmc/releases) | 官方发布记录 | 当前能力边界与版本演进 |

## 冲突证据账本

| 主张 | 支持证据 | 冲突证据 | 当前判断 |
|---|---|---|---|
| 网络旋转等变 | README/初学者指南文字 | 源码含笛卡尔向量进入普通 MLP；旋转反例变化 0.812 | 主张错误，平移不变成立 |
| 默认梯度严格无偏 | 人口 VMC 恒等式；官方实现也使用得分函数 | 同批次中心化 `(B-1)/B`；tanh 截断改变梯度 | 恒等式正确，默认有限样本估计非严格无偏 |
| JVP O(1) | 无 Python 坐标循环 | 单位阵含 3N 方向并全部 JVP | 数值精确，复杂度主张错误 |
| Hutchinson 可保高精度百倍提速 | 条件期望无偏；减少 HVP 次数 | 单探针标准差 0.535 Ha；100 原子两个结果相差数千 Ha | 速度与误差必须共同报告 |
| 图展示 cusp | 图在核位置画标线 | 包络核上导数为零；未检验球平均 Kato 条件 | 图不能作为 cusp 证据 |
| 编译加速 3.6–5.8 倍 | README 历史测量 | 当前同类 GPU 实测 2.57–2.89 倍 | 环境相关，需附版本和原始结果 |

## 去重与限制

arXiv 与期刊版本按同一研究计一次；同一仓库的四个源码文件在来源数量统计中作为一个组织的不同一手证据，避免伪造多样性。博客和聚合站只用于发现线索，未支撑核心结论。当前没有执行论文超参数规模、HF 预训练、KFAC/SPRING 优化或长时间 LiH 训练，也没有验证激发态、周期体系、力、赝势和多 GPU；这些是能力边界，不判为缺陷。
