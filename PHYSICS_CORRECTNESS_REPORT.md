> 历史记录：本文保留物理修复时的接口、路径和测试结果。当前 0.3.0 架构与迁移见[重构说明](ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md)，最新重构验收见[第二轮报告](REFACTOR_PHASE2_REPORT.md)。历史训练结果不代表统一模型已经达到相同收敛水平。

# 物理正确性修复与证据报告

日期：2026-10-22。审查基线：`a1dc6c3`；本报告对应其后的工作区修复。

本项目的定位是研究型固定核、非相对论电子波函数框架。确定性物理验证、统计诊断和训练收敛是三个不同的验收层次，任何一层通过都不能替代其他层。历史 Level 5 报告已标为历史记录，不再作为当前认证。

## 1. 修复与验收映射

| 问题 | 修复后的行为 | 独立验证 |
|---|---|---|
| 参数更新后采样缓存过期 | 优化更新和最佳状态恢复后刷新 `log|Psi|`；正式采样冻结步长 | 连续两轮更新后缓存与重新前向一致；旧恢复状态回归 |
| Jastrow 与指数包络重复核尖峰 | 两种模型共用物理配置，核尖峰 Jastrow 默认 Gaussian；已知冲突配置报错 | 氢原子常数轨道投影的近核斜率与局域能量极限 |
| Psiformer 引入额外电子对尖峰 | 几何偏置使用 `d^2/(1+d)` 径向特征 | 完整 singlet 波函数的 `1/2` 斜率、方向性聚并能量 |
| 三重态仍使用 `1/2` | 根据目标双电子自旋配置 `1/4`，拒绝不相容显式 Jastrow | 三种有效模型组合的 triplet 聚并极限 |
| 零行列式分支导数丢失 | 共享 signed determinant sum，保留精确余子式一、二阶及更高阶导数 | `Psi=1+x+x^2` 在零点的两个对数导数均为 1；独立 Leibniz 多项式参照 |
| 大矩阵使用阻尼近似 | 删除尺寸大于 6 时的阻尼回退，按所请求导数阶数计算批量子式 | 7、8、10 阶解析秩亏矩阵的一阶导数，7、10 阶秩亏二矩阵的混合二阶导数 |
| JVP 的 mixed derivatives 错误、缓存丢切向量 | forward-over-reverse，dual 坐标重建几何；不静默回退后端 | `slogdet(1+x^2)`、真实模型的能量与参数导数对比 |
| 平滑缓存改变库仑势或采样目标 | 库仑势使用原始物理距离；模型拒用不匹配的特征缓存；变分态平滑参数必须匹配模型 | 势能不随 feature eps 改变；无梯度/不匹配缓存重建；独立坐标求导 |
| 参考能量没有绑定体系 | 内置参考同时匹配核电荷、电子数、目标自旋、键长 | 拉伸 H2、H2+ 和其他自旋态不误用平衡参考 |

修复前新增的 8 项审计回归全部失败。修复后这些反例通过，并补充了统计、架构交叉与大矩阵验证；测试没有通过调整错误期望来掩盖算法失败。

两个原测试把同自旋精确重合节点的局域能量有限作为要求，现已修正：波函数节点保持 `sign=0, log_abs=-inf`；真实库仑势在电子重合点发散，`H Psi / Psi` 在节点本身未定义。离开节点后再检查局域能量高阶导数。实际值与导数仍然由独立参照检查。压力测试保留耗时记录，删除依赖机器负载的 50/100 ms 数学正确性门槛；首轮完整 CUDA 测试因 163 ms 超过旧 100 ms 门槛失败，原始日志保留为 `cuda_complete_no_skip_first.log`，数值断言未放宽。

## 2. 科学含义与接口变化

- 核尖峰与异自旋非零 s-wave 尖峰应按球平均分量解释；同自旋或双电子三重态的 `1/4` 属于去掉线性 p-wave 零点后的径向系数，不是完整对数波函数的径向导数。
- 特征距离中的数值平滑仍有分辨率尺度，不能据有限距离实验声称严格数学极限已被证明。完整模型的聚并测试在 `1e-3` 至 `1e-5` bohr 等有限区间进行。
- 公共模型、损失和评估入口保留；checkpoint 参数名称和形状保留。启用 Jastrow 后的错误默认包络已修正，旧 checkpoint 的函数行为不能由参数键名兼容推断。应使用明确的模型配置重新构建并评估；不相容的指数包络与核尖峰 Jastrow 组合需要迁移配置。
- 势能 `eps` 默认改为 0，表示真实库仑势；非零值是用户显式请求的软核平方长度。势能只计算真实电子对，避免对未使用的自相互作用 `sqrt(0)` 求导。
- `VariationalState.eps` 默认跟随模型；显式不一致时报错。几何缓存记录其平滑规则，坐标图或规则不匹配时重建。
- 局域能量与参数损失使用独立求导图。能量截断和有限批次协方差估计不能声称严格无偏。
- 保留 `compute_autocorrelation_time()` 的三元组返回值；使用多滞后 Geyer 初始正/单调序列。冻结评估新增置信区间、split-Rhat、分块曲线、统计状态与原因，并拒绝非有限能量而非丢弃它们。
- 分块平台要求连续三个二次幂块长的 SEM 最大/最小比不超过 1.25，每级至少 16 块，最小块长至少 `2*tau`。统计诊断还要求至少 128 个时间记录、ESS 至少 1000、能量 split-Rhat 不超过 1.01。
- `chemical_accuracy_met=True` 仅在有匹配参考、诊断通过且 `|mean-reference|+1.96*SEM <= 0.0015936 Ha` 时成立。状态额外区分未达到、证据不足和无参考。
- 区间是针对固定 checkpoint 的近似正态 95% 区间；split-Rhat 不是 rank-normalized 指标，且只检查能量这一观测量，不证明遍历了全部空间模式。优化误差不在单 checkpoint 的 Monte Carlo SEM 内。
- TorchDynamo 当前不支持追踪带自定义 JVP 方法的 Function。共享行列式在编译时使用相同数学值和精确 VJP 的版本，在 eager 动能 JVP 中使用带切向规则的版本。编译采样前向通过不等于整个 Hessian 计算已由 Inductor 验证。

H2 的参考来自 [Pachucki 2010，Table IV](https://arxiv.org/pdf/1007.0322)：`R=1.4 bohr` 时 BO 能量约 `-1.174475714220 Ha`。历史常数 `-1.17447593` 对应接近 `R=1.4011 bohr` 的平衡几何，不能无条件用于所有 H2。尖峰解释参见 [Drummond 等人的原始论文，III 节](https://arxiv.org/pdf/0801.0378)。

## 3. 本次实际验证

实际硬件为 NVIDIA GeForce RTX 3060 Laptop GPU；CUDA 环境为 Conda Python 3.13.15、PyTorch 2.14.0、CUDA runtime 13.0。CPU 独立环境为项目 `.venv`。

| 验收层次 | 实际结果 | 证据文件 |
|---|---|---|
| CPU 完整发现 | 270 项，243 通过、27 CUDA 项按环境跳过，125.26 秒 | `physics_evidence/cpu_complete_final.log` |
| CUDA 可用环境完整发现，任何 skip 均视为失败 | **270 项全部通过，0 失败、0 错误、0 跳过，94.73 秒** | [结构化摘要](physics_evidence/cuda_complete_no_skip.json)、[完整日志](physics_evidence/cuda_complete_no_skip.log) |
| 专门 GPU 数值与导数检查 | 18 项实际在 CUDA 执行，通过；包含 CPU/GPU 对照及参数更新 | 上述完整日志中的 `test_cuda` |
| GPU 编译检查 | float32/64 Inductor 等价性、混合奇异矩阵、编译 MCMC、编译训练、图断裂检查通过 | 上述完整日志中的 `test_compile`、`test_adversarial_dynamo`、`test_graph_breaks_verification` |
| 独立 float64 解析核验 | 氢原子 72 个指定方向/半径点的能量误差与方差均为 0；三种模型 REVERSE/JVP 最大差：CPU `4.27e-14 Ha`，CUDA 0 | [analytic_results.json](physics_evidence/analytic_results.json) |
| 实验快照与最终代码回放 | 九组 checkpoint 的哈希核验通过；每组八个固定构型（含近聚并）下符号一致，对数振幅与局域能量最大差均为 0 | [final_code_equivalence.json](physics_evidence/20261022T045238Z/final_code_equivalence.json) |
| 分发包 | sdist/wheel 构建及隔离导入通过 | 完整套件中的 `test_packaging` |

完整套件含 CPU 测试及真正执行 CUDA 的测试；“270 项通过”不表示每一项都在 GPU 上执行。27 项原本需要 CUDA 的检查在此次 CUDA 环境中均已执行，没有禁用 skip 装饰器伪造覆盖率。独立解析结果只针对所列测试点和模型实例，不是全构型空间的数值误差上界。

Inductor 最初受到继承的 `CC=gcc` 和 MSYS2 头文件搜索路径影响，出现编译器错误。验收脚本只在子进程中选用 Triton 自带 TCC，并清除 `CPATH/C_INCLUDE_PATH/CPLUS_INCLUDE_PATH/INCLUDE/LIB/LIBPATH/LIBRARY_PATH`；同时启用 UTF-8、确认 CUDA 可用。未修改系统编译器或全局环境。历史失败与修复后的编译日志都保留，最终完整套件使用真实 Inductor。

严格重跑入口的历史独立脚本未纳入当前包；当前等价入口为 [scripts/run_cuda_tests.py](scripts/run_cuda_tests.py)：如果 CUDA 不可用则报错，任何 `skipped` 都导致非零退出，并记录当前源码 SHA256。完整测试不再依赖固定毫秒门槛，物理值、误差容限、反对称性、自旋与一/二阶导数断言均保留。

## 4. 预算内训练证据

固定顺序：经典 H2 的种子 11/22/33、可组合 H2 的种子 11/22/33、Psiformer H2、He、拉伸 H2。主实验不按能量选择 checkpoint，均使用最后一次更新后的参数。没有按结果挑选种子或延长某个表现较好的运行。

共同配置：float64、256 walkers、4 行列式、两层 32/8 隐藏维度、Gaussian 包络、核/电子对 Jastrow、singlet 投影、Adam 初始学习率 `1e-3`、按 600 epochs 调度的余弦学习率。主实验每次最多训练 600 epochs 或 100 秒，冻结评估最多 512 个时间记录或 70 秒，重新预热 50 步，每次记录前推进 10 步 MCMC。附加三个流程实验分别限制训练 60 秒和评估 40 秒。总 runner 时间上限 1500 秒。

这些是受时间限制的收敛实验，实际更新次数不同；不用于架构速度或精度排名。同机还执行了回归检查，wall time 不是独占硬件的性能基准。训练及冻结评估 runner 共耗时 1350.72 秒（22.51 分钟），在所选训练预算内；额外的全套 CUDA 回归是后续“测试不跳过”请求的验收工作，编译重试日志亦保留。

| 模型 / 体系 / 种子 | 更新次数 | 评估时间记录数 | 能量均值 / Ha | 相关性修正 SEM / Ha | 统计诊断 |
|---|---:|---:|---:|---:|---|
| Classic H2 / 11 | 363 | 297 | -0.99119034 | 0.00235891 | 通过 |
| Classic H2 / 22 | 368 | 290 | -1.01761393 | 0.00248467 | 通过 |
| Classic H2 / 33 | 345 | 304 | -1.03316211 | 0.00225963 | 通过 |
| Composable H2 / 11 | 307 | 240 | -0.95074294 | 0.00350091 | 证据不足：无分块平台 |
| Composable H2 / 22 | 309 | 223 | -1.02249169 | 0.00300645 | 证据不足：无分块平台 |
| Composable H2 / 33 | 345 | 171 | -1.00121511 | 0.00394438 | 证据不足：无分块平台 |
| Psiformer H2 / 11 | 46 | 15 | -0.71004305 | 0.01990079 | 证据不足：记录少、Rhat、无平台 |
| Composable He / 11 | 50 | 53 | -1.79372717 | 0.02816603 | 证据不足：记录少、Rhat、无平台 |
| Composable H2 R=3 / 11 | 195 | 135 | -0.27661891 | 0.02642963 | 证据不足：Rhat、无平台 |

每条记录含 256 walkers；九次评估均无非有限能量，未删除离群能量。所有 H2 平衡几何实验均未证明化学精度，前三次明确判为未达到，其余四次判为统计证据不足。He 和拉伸 H2 没有自动匹配参考，状态为 `no_reference`。诊断未通过时，表中 SEM 仅作为当前估计，不能当作已验证的可靠误差范围。

跨独立训练种子的汇总使用三个 checkpoint 能量均值，以及自由度 2 的 Student t 区间；SEM 取种子间与种子内估计的较大值。Classic 的均值为 `-1.01398879 Ha`，95% 区间 `[-1.06670074, -0.96127684]`；Composable 的均值为 `-0.99148324 Ha`，区间 `[-1.08302643, -0.89994006]`，后者的单次评估诊断均未通过，汇总仍是探索性描述。没有将不同已优化模型的样本合并成一条“高精度”轨迹。

完整机器可读结果：[results.json](physics_evidence/20261022T045238Z/results.json)。同目录保存九组训练历史、最终 checkpoint、未经能量截断的局域能量轨迹和源码快照。这些结果证明修复后的训练、冻结评估和失败状态报告流程可执行，尚未证明充分优化收敛、化学精度或某架构优于另一架构。

## 5. 复现、迁移与限制

```powershell
uv run python -m unittest discover -s tests -t . -v
F:\conda\envs\torch\python.exe -m unittest tests.test_cuda -v
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests
F:\conda\envs\torch\python.exe -m ferminet_pytorch.physics_validation --device cuda --budget-seconds 1500
uv run python -m ferminet_pytorch.physics_validation --verify-output physics_evidence/20261022T045238Z
```

验证入口创建新的输出目录，不覆盖历史 JSON。每次运行保存环境、源码 SHA256、提交与 diff 摘要、完整配置、训练历史、checkpoint SHA256，以及按 `(time, walker)` 保存的未截断局域能量。生产坐标本身不保存；固定 checkpoint、种子、记录长度与源代码可用于重新采样验证。时间预算导致重跑的 epoch 数不保证完全一致。

本次实验运行过程中保留了与启动时源码哈希一致的 `source_snapshot/`。后续修订增加编译兼容、缓存校验及势能高阶梯度保障；最终版本与实验快照的 checkpoint 能量等价性已另行检查，九个 checkpoint 均通过。实验 JSON 不修改或补写成比实际更好的结果。

本轮不提供大体系性能、普遍化学精度或长期优化收敛保证。精确余子式高阶求导成本随矩阵尺寸和请求阶数增长，未经本次实验验证的大体系需要单独测量。Hutchinson 减少固定探针数量下的 Hessian-vector product 次数，但新增方差和实际吞吐必须逐体系评估。
