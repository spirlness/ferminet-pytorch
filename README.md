# FermiNet PyTorch

用于固定核、非相对论多电子薛定谔方程的 PyTorch 研究框架。支持 FermiNet 和 Psiformer 主干，以及显式 Kato Jastrow、自旋投影和 VMC 优化。

当前版本为 **0.3.0**。本轮采用破坏性包结构重构，删除顶层模块导入、重复包络、Modern/Composable 模块和旧适配器，不提供兼容门面。物理正确性测试与训练收敛证据分别验收；已完成 H₂ 的双 T4 多种子 50000 轮训练和独立冻结评估，尚未证明跨种子稳定达到化学精度。

## 已完成的 H₂ 精度结果

H₂ 核间距 1.4 bohr、单重态，两种主干各运行种子 11/22/33，每组累计训练 50000 轮并用独立 FP64 采样评估 1,048,576 个样本：

| 主干 | 三个种子的参考偏差（mHa） | 平均偏差（mHa） | 化学精度通过数 |
|---|---|---:|---:|
| FermiNet | 0.809 / 1.657 / 1.253 | **1.240** | **2/3** |
| Psiformer | 3.115 / 1.988 / 3.586 | **2.896** | **0/3** |

六组采样诊断及前后数值精度审计均通过。化学精度要求 `abs(E − E_ref) + 1.96 × SEM ≤ 1.5936 mHa`；参考能量为 −1.174475714220 Ha。均值描述三个不同 checkpoint，不是合并样本后的单个波函数结果。结论限于这组 H₂ 配置，最新原始结果见[50000 轮续训摘要](physics_evidence/continuation_final_20261025/summary.json)及[检查点核验](physics_evidence/continuation_final_20261025/verification.json)；[原 20000 轮报告](KAGGLE_T4_FINAL_RESULTS.md)保留为历史基线。裁剪 × 包络的 12 组新实验见[受控实验说明](FACTORIAL_EXPERIMENT_GUIDE.md)与[执行记录](FACTORIAL_EXPERIMENT_STATUS.md)，正式 12 组已完成并核验，A/B/C/D 化学精度通过数分别为 2/3、3/3、3/3、3/3；三个预设配对效应的调整区间均跨零，改善证据不足，见[最终对照报告](physics_evidence/factorial_final_20261025/REPORT.md)及[可复核摘要](physics_evidence/factorial_final_20261025/verified_summary.json)。

## 安装与运行

```powershell
uv sync --locked
uv run python -m examples.example_h2
uv run python -m ferminet_pytorch.train_vmc --epochs 5 --walkers 64
```

安装后也可使用 `ferminet-train` 和 `ferminet-validate` 命令。库内模块统一从 `ferminet_pytorch` 导入：

```python
import torch
from ferminet_pytorch import FermiNet
from ferminet_pytorch.systems import build_h2_system
from ferminet_pytorch.hamiltonian import compute_local_energy

system = build_h2_system(dtype=torch.float64)
model = FermiNet(
    system.nuclear_coords, system.nuclear_charges, 1, 1,
    num_determinants=4, hidden_dim_one=32, hidden_dim_two=8,
    use_cusp_jastrow=True, spin_adapted=True,
).double()
r = torch.randn(8, 2, 3, dtype=torch.float64, requires_grad=True)
log_abs, sign = model.forward_log_psi(r)
energy, kinetic, potential = compute_local_energy(model, system, r)
```

## 训练 API

```python
from ferminet_pytorch import TrainingConfig, train_vmc

result = train_vmc(config=TrainingConfig(
    num_epochs=20, num_walkers=128, device="cpu",
    interaction_type="psiformer", use_cusp_jastrow=True, spin_adapted=True,
    two_stage_jastrow=True, jastrow_stage1_epochs=10,
))
```

显式模型可用 `train_vmc(system=system, model=model, config=config)` 训练。更低层的 `VMCDriver(model, system, sampler, ...)` 使用同一更新核心。旧的 `train_vmc(num_epochs=..., ...)` 和 `train_two_stage_jastrow` 接口不再提供。

## 目录

```text
ferminet_pytorch/     # 唯一的运行库，包内使用相对导入
    envelope.py      # 唯一的 NuclearEnvelope 实现
    model.py         # 唯一的 FermiNet，支持两种交互主干
    geometry.py, features.py, layers.py, psiformer_attention.py
    jastrow.py, spin.py, determinants.py
    hamiltonian.py, kinetic_modern.py
    mcmc.py, loss.py, vmc_driver.py, train_vmc.py, systems.py
    evaluation.py, vmc_statistics.py, physics_validation.py
tests/               # 正式实现的物理、数值、编译和分发测试
scripts/             # 基准、诊断与严格 CUDA 测试入口
examples/            # 可运行示例
physics_evidence/    # 原始日志、训练结果和不可改写的历史源码快照
```

交互层、模型组装和训练核心均已统一。`interaction_type="ferminet"` 与 `"psiformer"` 选择不同主干；两者共用物理因子与 VMC 更新逻辑。

## 核包络接口

包络只接收已经计算的电子—核距离，不负责坐标、核位置或平滑参数：

```python
from ferminet_pytorch.envelope import NuclearEnvelope

envelope = NuclearEnvelope(4, 2, 1, envelope_type="gaussian")
distances = torch.rand(8, 2, 1)  # (batch, electrons, nuclei)
values = envelope(distances)     # (batch, determinants, electrons, orbitals)
```

指数与 Gaussian 使用同一个实现。`NuclearEnvelopeModern`、`legacy_adapters` 以及旧的 `envelope(r, nuclear_coords, dist=...)` 调用已删除。

## 验证

```powershell
# CPU 环境：CUDA 项按环境跳过，输出实际数量
uv run python -m unittest discover -s tests -t . -v

# CUDA 数值、梯度与包络检查
F:\conda\envs\torch\python.exe -m unittest tests.test_cuda -v

# 全套严格 CUDA 验收：必须存在 GPU，任何 skipped 项均失败
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests

# 构建 sdist 和 wheel
uv build
```

严格 CUDA 入口仅为子进程隔离 Windows 编译器环境，并将日志、运行环境、结果及源码哈希写入新的 `physics_evidence/cuda_*/` 目录。完整套件包含 CPU 检查和真正执行 CUDA 的检查，不表示每项均在 GPU 上执行。

## 物理与统计约束

- 电子坐标统一为 `(batch, electrons, 3)`，核坐标为 `(nuclei, 3)`。
- 局域能量使用独立坐标求导图，脱离计算图后进入参数代理损失；优化和恢复模型后刷新采样缓存。
- 特征距离的平滑规则不改变真实库仑势；正式 MCMC 采样冻结步长自适应。
- 核及异自旋非零 s-wave 尖峰按球平均分量解释；同自旋的 `1/4` 是移除 p-wave 零点后的径向系数。精确节点处不定义局域能量。
- 冻结评估保留 `(time, walker)` 结构，报告自相关、ESS、split-Rhat、分块平台和不确定度；参考能量必须匹配体系和几何。
- Gaussian 默认包络用于避免与显式核 Jastrow 重复引入核尖峰。完整波函数的有限距离聚并检查不等于严格极限证明。

## 实验和文档

```powershell
uv run python -m ferminet_pytorch.physics_validation --device cpu --budget-seconds 120
F:\conda\envs\torch\python.exe -m ferminet_pytorch.physics_validation --device cuda --budget-seconds 1500
uv run python -m scripts.benchmark_m4_vmc_convergence --help
```

- [当前重构架构与迁移清单（0.3.0）](ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md)
- [第二轮统一核心与验收](REFACTOR_PHASE2_REPORT.md)
- [入门指南](BEGINNER_GUIDE.md)
- [开发指南](DEVELOPER_GUIDE.md)
- [Kaggle 双 T4 多种子训练与冻结评估](KAGGLE_T4_TRAINING_GUIDE.md)
- [Kaggle H₂ 六组最终精度与原始证据](KAGGLE_T4_FINAL_RESULTS.md)
- [延长训练、末期学习率和自动双卡测试流程](CONTINUATION_TRAINING_GUIDE.md)
- [贡献规范](AGENTS.md)

第二轮已记录的严格 CUDA 完整验收为 **251 通过、0 失败、0 错误、0 跳过**；[原始结果与源码哈希](physics_evidence/phase2_cuda_complete/summary.json)可核查。完整套件包含 CPU 与 CUDA 检查，长期收敛证据另行验收。

历史记录：

- [第一轮包结构与包络重构](REFACTOR_PHASE1_REPORT.md)
- [物理正确性修复与历史训练证据](PHYSICS_CORRECTNESS_REPORT.md)
- [旧架构方案原文](ARCHITECTURE_REFACTORING_PROPOSAL_HISTORY.md)
- [旧生产就绪报告](FERMINET_REFACTORING_EVALUATION_AND_PRODUCTION_READINESS_REPORT.md)
- [初期代码审查](CODE_REVIEW.md)

历史记录保留当时的接口与结果，不能作为当前性能或化学精度认证。第二轮没有新增长期收敛实验。

## 许可证

MIT。
