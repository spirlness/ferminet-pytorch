# FermiNet-PyTorch: 高性能、可组合的量子化学深度学习波函数框架

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-246%20Passing-brightgreen.svg)](#-自动化测试验证)
[![Readiness](https://img.shields.io/badge/Production%20Readiness-Level%205-gold.svg)](FERMINET_REFACTORING_EVALUATION_AND_PRODUCTION_READINESS_REPORT.md)

**FermiNet-PyTorch** 是一个基于 PyTorch 构建的现代化、高性能第一性原理量子化学深度神经网络波函数框架。项目融合了从教学级极简原型到工业生产级（Level 5）解耦架构的全景演进，支持通过变分量子蒙特卡洛 (VMC) 高精度求解多电子薛定谔方程 $\hat{H}\Psi = E\Psi$。

核心特性囊括：**自旋纯态投影算符 ($\hat{S}^2 \Psi = 0$)**、**显式 Kato 尖峰 Jastrow 因子**、**全局共享几何上下文引擎**、**Psiformer 自注意力机制**、**Hutchinson 随机迹估计动能求解器** 以及 **四层正交解耦的 VMC 驱动引擎**。

> 📖 **经典论文参考**：  
> **Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks**  
> *David Pfau, James S. Spencer, Alexander G. de G. Matthews, W. M. C. Foulkes*  
> *Physical Review Research* 2, 033429 (2020) · [DOI: 10.1103/PhysRevResearch.2.033429](https://doi.org/10.1103/PhysRevResearch.2.033429)

---

## 🗺️ 文档导航地图 (Documentation Map)

为了方便不同背景的学习者与科研开发者快速检索，本项目提供分层清晰的文档索引：

```text
                                  【README.md】
                             项目主门户与双轨快速上手
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
【BEGINNER_GUIDE.md】       【DEVELOPER_GUIDE.md】        【评估与审查报告】
新手入门与物理直觉宝典        系统架构与二次开发指南        生产就绪度与性能评测
• 量子力学极简直觉比喻       • 四层正交解耦架构设计        • 21项冗余消除落地矩阵
• 泡利原理与波函数可视化     • 四大可扩展点实战 (GNN/势能)  • 双环境 246 用例审计
• 5分钟快速上手实操          • 动能求解器高阶微分进阶      • 显存削峰与 Benchmark
• 物理指标白话对照表         • Single-Forward 显存契约     • Level 5 准予交付评级
```

| 文档名称 | 核心定位与适用人群 | 重点内容章节 |
| :--- | :--- | :--- |
| 🔰 [**新手入门宝典 (BEGINNER_GUIDE.md)**](file:///c:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/BEGINNER_GUIDE.md) | **初学者 / 跨学科入门**：零物理黑话，以程序员最容易理解的视角剖析 VMC 与深度波函数。 | 薛定谔方程通俗解释、Slater 行列式原理、MCMC 随机游走、输出指标解读、高频 FAQ。 |
| 🛠️ [**开发者与扩展指南 (DEVELOPER_GUIDE.md)**](file:///c:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/DEVELOPER_GUIDE.md) | **算法工程师 / 科研二次开发**：深入模块设计模式，指导如何编写新层、新势能和新优化流程。 | 四层正交架构拓扑、四大扩展实操案例、现代动能算符模式比较、向后兼容适配器。 |
| 📊 [**生产就绪度评估报告 (EVAL_REPORT.md)**](file:///c:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/FERMINET_REFACTORING_EVALUATION_AND_PRODUCTION_READINESS_REPORT.md) | **架构师 / 技术审计员**：全维度定量评估报告，记录项目生产级重构演进轨迹与 Benchmark。 | 21 项冗余缺陷消除实证、CPU/GPU 双环境基线压测、PyTorch 导数内存深度剖析。 |
| 🤖 [**智能体与规范准则 (AGENTS.md)**](file:///c:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/AGENTS.md) | **AI Coding Agent / 核心贡献者**：代码风格、测试规范与跨平台运行约束。 | PEP 8 命名规范、单前向契约、CUDA 开发环境路径、提交与 PR 规约。 |

---

## 🌟 核心特性与架构亮点

### 1. 严格物理不变量保证
- **泡利不相容原理**：通过神经网络构建单电子分子轨道并馈入 Slater 行列式，同自旋电子对坐标重合时波函数严格为零；
- **自旋纯态投影 (`spin.py`, `SpinProjector`)**：双电子单重态施加空间置换严格对称化 $\frac{1}{2}(I + P_{12})$，多电子体系利用自旋谱投影多项式排除高自旋杂质，消除自旋污染并严格保证 $\hat{S}^2 \Psi = 0$；
- **显式 Kato 尖峰边界条件 (`jastrow.py`, `KatoCuspJastrow`)**：采用 Padé 阻尼解析因子，在微观粒子碰撞点 ($r_{iI} \to 0$ 或 $r_{ij} \to 0$) 严格抵消库仑势能奇点，释放神经网络对奇异尖峰的拟合压力；
- **无尖峰平滑特征 (`features.py`)**：支持平滑特征变换 $d^2 / (1+d)$，杜绝特征层在原点的高阶拉普拉斯算子发散。

### 2. 现代可组合架构与显存优化 (Gen2 升级)
- **单批次共享几何引擎 (`geometry.py`)**：全局单次抽取不可变相对位移与平滑欧氏距离，彻底消除 4 次重复几何计算；
- **可组合解耦波函数拟设 (`composable_model.py`)**：
  - 经典两流等变层与 Psiformer 自注意力等变层自由切换；
  - 0 Graph Break 纯张量代数余子式伴随导数装配 (`BranchlessCofactorDeterminant`)；
- **多后端物理动能算子 (`kinetic_modern.py`)**：
  - `REVERSE`：经典精确反向模式（默认基准，逐 bit 吻合回归基准）；
  - `HUTCHINSON`：Rademacher 随机迹估计，将二阶导反向复杂度由 $\mathcal{O}(3N)$ 降至常数 $\mathcal{O}(1)$，大体系显存与吞吐大幅跃升；
  - `FORWARD_JVP`：基于前向切线推进。
- **现代化四层正交解耦引擎 (`vmc_driver.py`)**：
  - `VariationalState`（变分态与几何上下文）+ `HamiltonianComposite`（复合物理哈密顿量）+ `MetropolisSampler`（几何采样器）+ `VMCDriver`（训练驱动主循环）。

---

## ⚡ 双轨快速上手 (Dual-Track Quickstart)

本项目提供两条使用路径：**经典极简 API**（适合教学演示与快速脚本验证）与 **现代化解耦 API**（适合生产级训练与算法扩展）。

### 方式一：经典极简 API (5 行代码直接上手)

适合快速进行前向评估与脚本实验：

```python
import torch
from ferminet_pytorch import build_h2_system, FermiNet, compute_local_energy

# 1. 构建氢分子 (H2) 物理体系
system = build_h2_system()

# 2. 实例化经典 FermiNet (启用 Kato 尖峰 Jastrow 与自旋投影)
model = FermiNet(
    nuclear_coords=system.nuclear_coords,
    nuclear_charges=system.nuclear_charges,
    n_up=system.n_up,
    n_down=system.n_down,
    use_cusp_jastrow=True,
    spin_adapted=True,
)

# 3. 构造 4 个电子构型采样点并计算局域能量 E_L(r)
r = torch.randn(4, system.total_electrons, 3, requires_grad=True)
e_local, t_local, v_local = compute_local_energy(model=model, system=system, r=r)

print(f"H2 局域能量均值: {e_local.mean().item():.5f} Hartree")
```

也可直接运行预设教学脚本：
```bash
# 运行 H2 快速前向推理与能量拆解
uv run python example_h2.py

# 运行经典端到端 50 轮 VMC 优化
uv run python train_vmc.py --epochs 50 --walkers 512 --cusp_jastrow --spin_adapted
```

---

### 方式二：现代化解耦 API (可组合生产级框架)

通过四层正交引擎 `VMCDriver` 灵活调度物理算符、变分态与采样器：

```python
import torch
import ferminet_pytorch as fp

# 1. 物理体系与可组合模型
system = fp.build_h2_system()
model = fp.ComposableFermiNet(
    nuclear_coords=system.nuclear_coords,
    nuclear_charges=system.nuclear_charges,
    n_up=system.n_up,
    n_down=system.n_down,
    use_cusp_jastrow=True,
    spin_adapted=True,
)

# 2. 装配变分态与物理哈密顿量算符
state = fp.VariationalState(model=model, system=system, num_walkers=512)
hamiltonian = fp.HamiltonianComposite(system=system)

# 3. 装配采样器与优化器
sampler = fp.MetropolisSampler(variational_state=state, step_size=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = fp.build_cosine_scheduler(optimizer, max_epochs=50)

# 4. 组装现代 VMC 驱动引擎
driver = fp.VMCDriver(
    variational_state=state,
    hamiltonian=hamiltonian,
    sampler=sampler,
    optimizer=optimizer,
    scheduler=scheduler,
    clip_factor=5.0,
    clip_method="tanh",
    decorrelation_steps=10,
)

# 5. 执行端到端变分迭代
for epoch in range(1, 51):
    metrics = driver.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | 能量: {metrics['energy_mean']:.5f} Ha | 方差: {metrics['energy_std']**2:.4f} | 接受率: {metrics['acceptance_rate']*100:.1f}%")
```

---

## 🏗️ 端到端数据流与架构拓扑

```text
               ┌─────────────────────────────────────────────────────────────┐
               │                电子三维空间坐标 r: (B, N, 3)                │
               │               原子核坐标与电荷 R, Z: (M, 3), (M,)           │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
                      【全局共享几何上下文 GeometricContext geometry.py】
                             (单次计算相对位移 r_en, r_ee 与平滑距离)
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               │ 单电子流特征 h1: (B, N, 5M)   │ 双电子流特征 h2: (B, N, N, 4) │
               └──────────────────────┬───────┴──────────────┬───────────────┘
                                      │                      │
                        【等变主干交互层 FermiInteractionLayerModern】
                        (可选 PsiformerAttentionBlock 自注意力机制)
                                      │                      │
                                      ▼                      ▼
               ┌─────────────────────────────────────────────────────────────┐
               │    高阶单电子隐藏表征: h1_final (B, N, hidden_dim_one)       │
               └──────────────────────┬──────────────────────┬───────────────┘
                                      │                      │
                         (自旋向上电子切片)     (自旋向下电子切片)
                                      │                      │
                       ┌────────────────┴────┐            ┌────┴────────────────┐
                       │ 分子轨道投影器 (Up) │            │ 分子轨道投影器(Down)│
                       └────────┬────────────┘            └────────┬────────────┘
                                │                                  │
                                ▼                                  ▼
                   ┌─────────────────────────┐        ┌─────────────────────────┐
                   │ 分子轨道矩阵 D_up       │        │ 分子轨道矩阵 D_down     │
                   │ Shape: (B, K, n_up, n_up)│        │ Shape: (B, K, n_dn, n_dn)│
                   └────────────┬────────────┘        └────────────┬────────────┘
                                │                                  │
                                └────────────────┬─────────────────┘
                                                 │
                               【无分支伴随余子式行列式求导与加权求和】
                                                 │
                                                 ▼
                               ┌──────────────────────────────────┐
                               │  基础波函数: log|Psi_base|       │
                               └────────────────┬─────────────────┘
                                                 │
                                 【Kato 尖峰 Jastrow 因子 jastrow.py】
                                 ln J(r) = U_en(r) + U_ee(r) (解析对消)
                                                 │
                                                 ▼
                                 【自旋空间群投影 SpinProjector spin.py】
                                 Psi_proj = sum_k w_k * Psi(P_k r)
                                                 │
                                                 ▼
                               ┌──────────────────────────────────┐
                               │  最终严格物理波函数:             │
                               │  log_abs_psi: (B,)  | sign: (B,) │
                               └──────────────────────────────────┘
```

---

## 📊 命令行 CLI 参数完整字典

通过 `train_vmc.py` 执行端到端训练时，支持以下全量参数配置：

| 参数名 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--system` | `str` | `h2` | 目标物理体系，内置支持 `h2` ($H_2$) 或 `lih` ($LiH$) |
| `--epochs` | `int` | `50` | VMC 变分优化总迭代轮数 |
| `--walkers` | `int` | `512` | MCMC 并行构型采样数量 (Batch Size) |
| `--mcmc_steps` | `int` | `10` | 每次参数更新前执行的 MCMC 随机游走去相关步数 |
| `--burn_in` | `int` | `50` | 训练前初始热身平衡步数 |
| `--lr` | `float` | `1e-3` | 初始学习率 (配合 Cosine 退火调度) |
| `--step_size` | `float` | `0.5` | MCMC 初始移动步长 (自动调谐至 50% 接受率) |
| `--move_mode` | `str` | `all` | MCMC 移动模式：`all` (全电子一体化移动) 或 `one_by_one` (逐电子移动) |
| `--clip_factor` | `float` | `5.0` | 局域能量平滑截断半宽倍数 $\Delta = c \times \text{MAD}$ |
| `--clip_method` | `str` | `tanh` | 截断方案：`tanh` (Umrigar 平滑截断)、`hard` 或 `none` |
| `--grad_clip` | `float` | `1.0` | 优化器全局梯度范数裁剪阈值 |
| `--cusp_jastrow` | `flag` | `False` | **启用显式 Kato 尖峰 Jastrow 因子**，抵消粒子碰撞库仑发散 |
| `--spin_adapted` | `flag` | `False` | **启用自旋纯态投影**，消除自旋污染并保证 $\hat{S}^2=0$ |
| `--envelope_type` | `str` | `None` | 核包络类型：`gaussian` (平滑高斯) 或 `exponential` (经典指数) |
| `--smooth_cuspless` | `flag` | `False` | 强制几何特征使用平滑无尖峰距离 $d^2 / (1+d)$ |
| `--device` | `str` | `None` | 指定运行设备 (`cuda` 或 `cpu`) |
| `--seed` | `int` | `42` | 全局随机数种子 |

---

## 🧪 自动化测试验证

本项目拥有严苛的自动化测试套件（**25 个独立测试模块，共 246+ 项物理与工程测试**），涵盖物理不变量、导数极限定理、奇异行列式稳定性、向后兼容性以及 GPU CUDA 精度验证：

```bash
# 1. 运行 CPU 环境完整测试套件 (自动跳过 CUDA 测试)
uv run python -m unittest discover -s . -p "test_*.py" -v

# 2. 运行现代解耦组件专用回归测试
uv run python -m unittest test_refactor_review_regressions.py test_envelope_modern.py test_layers_modern.py test_kinetic_modern.py -v

# 3. 运行经典物理不变量与尖峰极限测试
uv run python -m unittest test_spin_and_cusp.py -v
```

### CUDA / GPU 开发环境与全量回归 (RTX 3060+ / CUDA 12.x)

如需启用完整 GPU 加速与 18 项专门的 CUDA 数值回归测试，可使用预配置的 Conda `torch` 环境：

```bash
# 执行 GPU 端到端变分训练
F:\conda\envs\torch\python.exe train_vmc.py --device cuda --epochs 5 --walkers 128

# 执行 18 项专门的 CUDA 物理与梯度回归测试
F:\conda\envs\torch\python.exe -m unittest test_cuda.py -v

# 执行 GPU 硬件加速下的完整测试套件 (246 项全部通过，0 skipped)
F:\conda\envs\torch\python.exe -m unittest discover -s . -p "test_*.py" -v
```

---

## ❓ 常见问题 FAQ

- **Q: 什么是 Hartree 原子单位制？**  
  量子化学中令 $\hbar = m_e = e = 1$。$1 \text{ Hartree (Ha)} \approx 27.2114 \text{ eV} \approx 627.51 \text{ kcal/mol}$；长度单位 $1 \text{ Bohr} \approx 0.529177 \text{ \AA}$。

- **Q: 为什么对数波函数 `log|Psi|` 经常是负数十甚至负几百？**  
  多电子联合概率密度是微观连续微元的乘积，在多维空间中真实振幅极其微小（如 $10^{-50}$）。使用 `torch.linalg.slogdet` 在对数空间下存储与运算，能彻底杜绝浮点下溢。

- **Q: 为什么要同时使用 Kato Cusp Jastrow 与自旋投影？**  
  Slater 行列式只能保证同自旋反对称，无法天然消除反平行自旋对碰撞时的库仑奇点发散（需要 Jastrow 因子），也无法自动消除高自旋态混合（需要自旋投影算符）。二者结合能从物理底层保证波函数处处解析且自旋严格守恒。

- **Q: 训练中如何判断模型是否收敛？**  
  依据量子力学零方差原理（Zero-Variance Principle）：当波函数收敛至精确本征态时，空间各处的局域能量均为常数，$E_L(\mathbf{r}) \equiv E_0$。因此，**局域能量均值下降的同时，能量方差 $\text{Var}(E_L)$ 趋向于 0** 是变分收敛的最关键判据。

---

## 📜 许可证

本项目遵循 MIT 开源许可证。欢迎引用与二次开发！
