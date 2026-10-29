# 第二轮：统一交互层、波函数与训练核心

日期：2026-10-22。版本：0.3.0。基于第一轮后的工作区进行破坏性重构，不维护旧模型、Modern 模块或训练调用接口。

当前模块职责、训练状态归属及逐项接口迁移见[重构架构与迁移说明](ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md)。本文记录重构完成时的实际验收；文档更新不代表重新执行了 CUDA 测试或长期训练。

## 结果与规模

| 范围 | 重构前 | 重构后 |
|---|---:|---:|
| 交互层、模型、训练及体系构造 | 6 个文件、1891 行 | 5 个文件、863 行 |
| 同范围有效 Python 语句（排除文档字符串） | 741 | 346 |

核心减少 **1028 行（54.4%）**，有效语句减少约 **53.3%**。行数包含注释与文档，并受格式化影响，故同时报告语句数。比较包含新拆出的 `systems.py`，没有通过把代码移出统计范围制造减少。原始源码保存在 `physics_evidence/phase2_baseline/`；这些审计快照不属于当前运行库。

- `layers.py` 是唯一两流交互层。自旋分区在初始化时确定，池化权重排除对角线，空通道分母安全；同一个线性权重按特征块应用，避免构造广播拼接特征。没有原地对角线扣减及其大数抵消。
- `model.py` 中只有 `FermiNet`，通过 `interaction_type` 选择 FermiNet 或 Psiformer 主干。两者共用几何、轨道、核包络、行列式、Jastrow、自旋投影和一致的输出接口。
- `VMCDriver.step` 是唯一优化更新核心。CLI、直接 Driver、冻结评估前训练和实验脚本均使用它。`VMCLoss.compute_step_loss` 负责共同的坐标/参数图隔离边界。
- 采样器独占 walkers 与对数波函数缓存。删除 `VariationalState`、`HamiltonianComposite`、额外的 `MetropolisSampler` 包装；不再同步两份 walker 状态或缓存。
- 更新前最佳状态、参数更新后缓存刷新、恢复权重后缓存刷新、梯度裁剪、调度、分阶段训练和 checkpoint 保存集中维护。
- `TrainingConfig` 替代长训练关键字列表；`train_vmc.py` 仅负责 CLI，`systems.py` 提供分子几何构造。实验脚本不再包含独立的优化器更新循环。
- 删除 `layers_modern.py`、`composable_model.py`、行列式门面、`train_two_stage_jastrow` 和旧 MCMC/自旋/损失别名，没有新增兼容包装层。

静态池化矩阵仍占用空间，attention 和包络仍有中间张量；没有声称零内存分配或固定吞吐提升。三种动能后端及自定义行列式 forward-AD/编译分支承担不同职责，继续保留。

## 新接口

```python
from ferminet_pytorch import FermiNet, TrainingConfig, train_vmc
from ferminet_pytorch.systems import build_h2_system

system = build_h2_system()
model = FermiNet(system.nuclear_coords, system.nuclear_charges, 1, 1,
                interaction_type="psiformer", use_cusp_jastrow=True,
                spin_adapted=True)
config = TrainingConfig(num_epochs=20, num_walkers=128, device="cpu",
                        two_stage_jastrow=True, jastrow_stage1_epochs=10)
result = train_vmc(system=system, model=model, config=config)
```

`forward_log_psi(r)` 返回 `(log_abs, sign)`；`model(r)` 返回波函数标量。直接使用 `VMCDriver(model, system, sampler, config=...)` 与高层训练执行相同更新；调用方为直接 Driver 准备好模型、体系与设备，采样器默认精度跟随模型。

旧的每次 forward 传自旋数、Composable 的 tuple forward、旧模块导入和多关键字训练调用均需迁移。配置中的模型选项仅在创建默认模型时生效，不会替换显式提供的模型。两阶段策略仅在模型存在 Jastrow 参数时启用。

统一后的直接 Driver 默认梯度裁剪阈值为 1，与训练入口一致；需要不裁剪时设置 `TrainingConfig(grad_clip=0)`。checkpoint 同时保存更新后模型、优化器状态及最佳模型快照；优化器属于更新后状态，最佳能量对应更新前测量的参数。保存 checkpoint 不等于提供完整随机链自动断点续训接口。

## 实际验证

最终源码验证结果：

| 验证 | 结果 |
|---|---|
| CUDA 可用环境完整发现 | **251 通过、0 失败、0 错误、0 跳过，83.90 秒** |
| GPU | RTX 3060 Laptop GPU；PyTorch 2.14.0；CUDA runtime 13.0 |
| GPU 执行范围 | 数值、梯度、自旋、包络、奇异行列式、Inductor、MCMC 和编译训练；新增 Psiformer 两阶段 checkpoint/缓存检查 |
| 独立数学回归 | 保留氢原子解析能量、尖峰聚并、反对称性、自旋、精确奇异导数、JVP 与 Hutchinson 检查 |
| 统一训练入口 | 相同初态、采样随机数和配置下，高层入口与直接 Driver 的训练历史、参数及 walkers 逐位一致 |
| 两阶段流程 | 第一阶段 Jastrow 参数不变、无梯度；第二阶段恢复梯度并更新；每步缓存与新模型前向一致 |
| 分发 | sdist/wheel 构建、隔离环境逐模块导入通过 |
| CLI | 安装后的 `ferminet-train` 完成 CPU Psiformer 两阶段训练；示例运行通过 |
| 静态检查 | Ruff F401/F811/F821 通过，当前脚本和示例全部可导入，`git diff --check` 通过 |

完整套件包含 CPU 测试和实际执行 CUDA 的检查，不表示所有 251 项都在 GPU 上执行。第一轮的 251 项加上 2 项统一 Driver 检查和 1 项 CUDA Driver 检查，再移除 3 项仅验证已删除别名关系的测试，最终仍为 251 项。物理矩阵由旧的三个类组合变成两个真实主干，不再对同一类做重复架构检查。

[最终完整日志](physics_evidence/phase2_cuda_complete/tests.log) · [结构化结果与源码哈希](physics_evidence/phase2_cuda_complete/summary.json)。验收后核对运行库、测试和脚本哈希，未发生代码漂移。

过程中保留了 CPU 初轮迁移失败日志及清理别名时的测试编辑错误记录；修复后重新执行完整套件。未放宽数值容限，也未禁用 CUDA skip 装饰器制造覆盖。

## 重构前后的独立数值对照

离线审计显式映射原经典 FermiNet、原可组合 FermiNet、原 Psiformer 的参数到统一模型，分别检查自旋数 `(1,0)`、`(1,1)`、`(2,2)`，共九组 float64 CPU 对照。映射仅存在于审计脚本，不进入运行库。两种模型处于相同权重时比较，未声称同种子重新初始化后的参数逐位一致。

| 量 | 九组中最大绝对差 |
|---|---:|
| 对数波函数 | 7.47e-14 |
| 符号 | 0 |
| 坐标梯度 | 2.95e-12 |
| 局域能量 | 5.92e-12 Ha |
| 局域能量对选定轨道权重的梯度 | 5.19e-11 |

表格向上取整；所有指定构型满足 `atol=rtol=1e-8`。这些差异是本次固定构型、模型和权重的实测值，不是全空间误差上界或所有参数梯度的普遍证明。

[完整对照数据与哈希](physics_evidence/phase2_equivalence.json) · [结构统计](physics_evidence/phase2_structure.json) · [审计脚本](scripts/verify_phase2_refactor.py)。

本轮没有重新进行长时间收敛实验，旧的短时训练记录不用于宣称统一模型已达到化学精度。默认初始化、统一梯度裁剪和调用接口有明确变化，历史训练轨迹不能假定逐位重现；旧 checkpoint 需要显式离线转换。

## 复现

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -t . -v
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests
uv run python -m scripts.verify_phase2_refactor
uv run ferminet-train --epochs 2 --walkers 8 --burn_in 1 --device cpu --interaction_type psiformer --two_stage_jastrow --jastrow_stage1_epochs 1
uv build
```

离线对照需要随本次工作区保留的 `phase2_baseline` 源码快照；历史物理证据目录不被加入运行分发包。
