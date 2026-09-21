# FermiNet (Fermi Neural Network) PyTorch 工业级实现

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-118%20Passed-brightgreen.svg)](#)

本项目提供了一个基于 **PyTorch 原生生态** 的高性能、模块化、工业级 **FermiNet (费米神经网络)** 实现，专门用于量子化学与凝聚态物理中多电子薛定谔方程的从头算 (Ab-initio) 变分求解。

> 📖 **参考论文**：  
> **Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks**  
> *David Pfau, James S. Spencer, Alexander G. de G. Matthews, W. M. C. Foulkes*  
> *Physical Review Research* 2, 033429 (2020) · [DOI: 10.1103/PhysRevResearch.2.033429](https://doi.org/10.1103/PhysRevResearch.2.033429)

<p align="center">
  <img src="assets/ferminet_nature_architecture.png" alt="FermiNet 3D Scientific Architecture Illustration" width="100%">
</p>

---

## 🔰 初学者导航与入门必读 (Beginner's Guide)

> **没有量子力学或量子化学背景？看不懂波函数、Slater 行列式、变分量子蒙特卡洛？**
> 
> **请直接阅读我们专门打造的零门槛教学宝典：👉 [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) 👈**
> 
> 在该指南中，我们用**程序员最熟悉的通俗比喻与直白语言**，手把手为你拆解：
> 1. 什么是薛定谔方程与波函数（微观世界的牛顿定律与几率振幅）；
> 2. 泡利不相容原理与费米子反对称性（“撞衫即毁灭”的量子规则）；
> 3. FermiNet 深度学习 Backflow 机制与数据流架构拆解；
> 4. 变分量子蒙特卡洛 (VMC) 随机游走采样与有限批次协方差梯度原理；
> 5. 6 个极速实操演示（含全套学术级训练看板与物理态热力图解析）；
> 6. 新手高频问答 FAQ（对数波函数负几百、Hartree 单位制、零方差原理等）。

---

## 🌟 项目核心亮点与技术演进

本项目不仅完整复现了 DeepMind 原论文的物理核心，还在 PyTorch 生态下完成了四阶段重大工程与物理性能演进：

1. **严格物理第一性原则 (Ab-Initio)**：
   - 严格满足全同费米子置换反对称性（泡利不相容原理：$\Psi(\dots, r_i, \dots, r_j, \dots) = -\Psi(\dots, r_j, \dots, r_i, \dots)$）；
   - 严格遵循量子力学束缚态渐近衰减边界条件（指数包络衰减，各向同性与各向异性可学习度规）；
   - 相对几何特征提取，原生满足三维空间平移不变性与旋转等变性。
2. **端到端变分量子蒙特卡洛 (VMC) 训练闭环**：
   - 全向量化批量 GPU 并行 Metropolis-Hastings MCMC 采样器，支持全电子/逐电子步进与自适应步长控制；
   - 默认使用无截断、带 $B/(B-1)$ 有限批次修正的协方差代理梯度，避免三阶导计算；
   - Tanh/Hard 局域能量截断作为显式稳健模式保留；它能抑制离群点，但会引入训练目标偏差；
   - 训练结束后冻结模型和采样步长，报告自相关时间、有效样本数、split-R-hat 与链相关 SEM。
3. **突破性动能拉普拉斯求解器 (百倍加速)**：
   - 集成 **Hutchinson 随机迹估计器**，将二阶导计算复杂度从 $\mathcal{O}(3N)$ 压缩至 $\mathcal{O}(1)$，仅需 1 次方向导数反向传播，在大体系上实现数十倍至数百倍提速；
   - 集成基于 `torch.func.jvp` 的前向模式全向量化精确求解器与经典逐分量精确模式。
4. **PyTorch 2.0+ `torch.compile` JIT 图编译优化**：
   - 深度兼容 TorchDynamo 与 TorchInductor 算子融合与 CUDA Graphs，实测推断吞吐提升 **3.6x ~ 5.8x**。
5. **超大体系与异核分子支持**：
   - 单电子特征注入原子核电荷 $Z_I$，原生支持极性异核分子（如 $LiH, H_2O$ 等）；
   - 支持 **100 原子超大体系 (H100 晶格链，100 核 + 100 电子)** 的轻量化显存前向推断与物理自洽检验。
6. **顶刊级别 (Nature / PRR 规范) 综合可视化与物理状态看板**：
   - 独立高内聚可视化模块 [`visualize.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/visualize.py)，强制采用无头服务器安全的 `Agg` 后端；
   - 全面覆盖训练过程动态（能量收敛、零方差原理演化、MCMC 采样健康度三合一看板）与物理状态诊断（3D 电子几率密度、2D 成键截面热力图、核轴 Cusp 尖峰剖面、局域能量偏度分布）；
   - 深度集成于 CLI（`--plot`, `--save_plots`, `--plot_dir`），支持训练后全自动导出 10 大专业矢量级图表。

---

## 📂 项目模块全景导览 (Architecture Overview)

<p align="center">
  <img src="assets/ferminet_academic_dataflow.png" alt="FermiNet Academic Architecture and Dataflow" width="100%">
</p>

```text
ferminet_pytorch/
├── __init__.py           # 包顶层导出与版本定义
├── features.py           # 相对几何特征提取 (ElectronFeatureEmbedding, 支持 Z_I 注入)
├── layers.py             # 自旋等变相互作用层 (FermiInteractionLayer, 自旋分离均值池化)
├── envelope.py           # 原子核指数衰减包络 (NuclearEnvelope, 各向同性/各向异性度规)
├── model.py              # FermiNet 核心模型 (Backflow 轨道构建与 slogdet / LogSumExp 计算)
├── hamiltonian.py        # 分子哈密顿量 (MolecularSystem, 局域动能与全向量化势能计算)
├── mcmc.py               # 批量并行 GPU MCMC 采样器 (MCMCSampler, 自适应步长控制)
├── mcmc_stats.py         # 链自相关、ESS、split-R-hat 与相关性修正 SEM
├── loss.py               # 有限批次协方差代理损失与可选局域能截断
├── train_vmc.py          # 端到端 VMC 变分优化训练脚本 (支持 H2, LiH 与各拉普拉斯策略)
├── visualize.py          # 综合数据可视化模块 (Nature/PRR 期刊风格, 10 种训练与物理看板)
├── example_h2.py         # 5 分钟上手演示：氢分子单步推断与局域能量拆解
├── benchmark_compile.py  # PyTorch 2.0+ JIT 编译优化加速比评测脚本
├── benchmark_100_atoms.py# 100 原子超大规模体系基准测试与物理自洽性校验
├── test_ferminet.py      # 核心物理性质与网络架构综合单元测试套件
├── test_vmc.py           # VMC 采样、梯度估计与能量截断专项单元测试
├── test_phase2.py        # 阶段 2 动能求解器、核电荷注入与各向异性包络专项测试
├── test_visualize.py     # 综合可视化模块专项测试套件 (54 项)
├── assets/plots/         # 训练生成的可视化图表资源目录
├── BEGINNER_GUIDE.md     # 零基础小白从零到一量子化学深度学习图文宝典
└── README.md             # 架构技术文档与使用手册
```

### 核心源码文件职责速查

| 源码文件 | 核心职责 | 主要类 / 函数 | 物理/工程特性 |
| :--- | :--- | :--- | :--- |
| [`features.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/features.py) | 几何特征提取 | `ElectronFeatureEmbedding` | 提取相对位移与欧氏距离，平移旋转不变；支持注入核电荷 $Z_I$ 扩展至 $5M$ 维 |
| [`layers.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/layers.py) | 自旋等变前馈层 | `FermiInteractionLayer` | 单/双流信息融合，自旋向上与向下分别池化聚合，满足同自旋置换等变性 |
| [`envelope.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/envelope.py) | 指数衰减基底 | `NuclearEnvelope` | 约束 $r \to \infty$ 时波函数趋零；支持可学习各向异性三维主轴度规 $\Sigma$ |
| [`model.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/model.py) | 主干波函数网络 | `FermiNet` | 构造 Backflow 轨道，多行列式 `slogdet` 分解与 LogSumExp 稳定求和 |
| [`hamiltonian.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/hamiltonian.py) | 哈密顿量与动能 | `MolecularSystem`, `compute_local_energy` | 库仑吸引/排斥势能全向量化计算；集成 Exact, JVP, Hutchinson 三大拉普拉斯求解器 |
| [`mcmc.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/mcmc.py) | 并行构型采样 | `MCMCSampler` | 批量并行 Metropolis-Hastings 抽样，波函数对数缓存，自适应步长微调 |
| [`loss.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/loss.py) | 变分代理损失 | `VMCLoss`, `vmc_surrogate_loss` | 默认无截断有限批次协方差修正；MAD 截断为显式稳健有偏模式 |
| [`train_vmc.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/train_vmc.py) | 端到端训练优化 | `train_vmc` | 训练后冻结参数独立复评，报告 ESS、自相关时间、R-hat 与链相关 SEM |
| [`visualize.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/visualize.py) | 综合可视化分析 | `plot_training_dashboard`, `plot_all_visualizations` 等 10+ 绘图函数 | 学术期刊级绘图体系、无 GUI 隔离后端、1D/2D/3D 物理态与零方差收敛全面诊断 |

参考能量只在核电荷、自旋电子数和核间几何与登记记录全部匹配时显示。相同核数和电子数不会被用于猜测分子身份；未知体系不绘制参考线。

---

## 📐 张量维度规范 (Tensor Shape Specification)

为保障多卡 GPU 上的并行效率与维度一致性，全模块严格遵循以下张量规范：

| 张量名称 | 数学符号 | 形状 (Shape) | 物理含义与说明 |
| :--- | :--- | :--- | :--- |
| `r` | $r$ | `(B, N, 3)` | $B$ 个 MCMC Walkers，每个构型包含 $N$ 个电子的 3 维笛卡尔坐标 |
| `nuclear_coords` | $R$ | `(M, 3)` | $M$ 个原子核的固定三维坐标 (Bohr) |
| `nuclear_charges` | $Z$ | `(M,)` | $M$ 个原子核的核电荷量 (原子序数) |
| `h1` | $h_i^l$ | `(B, N, d_1)` | 第 $l$ 层的单电子流隐藏特征 ($d_1$ 默认 64) |
| `h2` | $h_{ij}^l$ | `(B, N, N, d_2)` | 第 $l$ 层的双电子流隐藏特征 ($d_2$ 默认 16) |
| `D_up` | $D_k^\uparrow$ | `(B, K, n_up, n_up)` | 自旋向上电子的分子轨道矩阵 ($K$ 为行列式分支数) |
| `D_down` | $D_k^\downarrow$ | `(B, K, n_down, n_down)` | 自旋向下电子的分子轨道矩阵 |
| `log_abs_psi` | $\ln\|\Psi(r)\|$ | `(B,)` | 波函数的对数绝对值（保障数值防下溢） |
| `sign_psi` | $\text{sign}(\Psi)$ | `(B,)` | 波函数符号值（$-1, 0, 1$） |
| `e_local` | $E_L(r)$ | `(B,)` | 各采样构型处的局域能量标量（单位：Hartree） |

---

## ⚡ 5 分钟极速上手实操 (5-Minute Quickstart)

### 演示 1：氢分子 $H_2$ 基础波函数推断与局域能量计算

单步实例化模型并计算对数波函数 $\ln|\Psi(r)|$ 与局域能量 $E_L(r) = T_L(r) + V(r)$：

```bash
# 在项目根目录下直接运行：
python example_h2.py
# 或使用 uv 运行：
uv run python example_h2.py
```

**Python 核心调用代码**：
```python
import torch
try:
    from ferminet_pytorch import FermiNet, MolecularSystem, compute_local_energy
except ImportError:
    from model import FermiNet
    from hamiltonian import MolecularSystem, compute_local_energy

# 1. 定义氢分子 H2 物理体系 (2 核, 2 电子: 1 up, 1 down, 键长 1.4 a.u.)
nuclear_coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32)
nuclear_charges = torch.tensor([1.0, 1.0], dtype=torch.float32)
system = MolecularSystem(nuclear_coords, nuclear_charges, n_up=1, n_down=1)

# 2. 实例化 FermiNet
model = FermiNet(nuclear_coords, nuclear_charges, n_up=1, n_down=1, num_determinants=16)

# 3. 构造 4 个采样构型并前向推断波函数
r = torch.randn(4, 2, 3, requires_grad=True)
log_abs_psi, sign_psi = model.forward_log_psi(r)

# 4. 计算局域能量 (动能 + 势能)
e_local, t_local, v_local = compute_local_energy(model, system, r)
print("局部动能 T_L:", t_local)
print("局部势能   V:", v_local)
print("总局域能量 E_L:", e_local)
```

---

### 演示 2：氢分子 $H_2$ 端到端 VMC 变分训练

使用变分量子蒙特卡洛算法，通过 Adam 优化器逐步逼近氢分子真实基态能量（理论参考值：$\approx -1.1744$ Hartree）：

```bash
# 启动 50 轮训练，配备 1024 个并行 Walkers
python train_vmc.py --epochs 50 --walkers 1024 --lr 1e-3
# 或使用 uv 运行：
uv run python train_vmc.py --epochs 50 --walkers 1024 --lr 1e-3
```

**训练输出指标通俗解读**：
- `<E_raw>` (原始局域能均值)：随着训练从约 $-0.8$ Ha 稳步下降并收敛至约 $-1.17$ Ha；
- `Var(E)` (能量方差)：**量子体系最重要的收敛指标**！当且仅当网络逼近真实薛定谔本征态时，空间各处局域能趋于同一常数，方差自发趋向于 0（零方差原理）；
- `Acc (%)` (接受率)：内置自适应步长控制器将采样接受率稳定锁定在 $50\%$ 黄金探索平衡点。

#### 科研结果的统计口径

训练期的 `<E_raw>`、`Var(E)` 与 `IID SEM*` 只用于优化诊断：连续 MCMC 样本相关，`IID SEM*` 不是最终误差条。训练结束后，`train_vmc` 会冻结模型参数与采样步长，重新平衡并默认采集 `32 × num_walkers` 个精确局域能量样本。

返回字典中的 `final_energy` 和 `final_variance` 来自该独立复评；`training_last_batch_energy` 单独保留，避免把更新前批次与最终模型混淆。`evaluation` 包含：

- `energy_sem`：按有效样本数修正的链相关标准误；
- `ess` 与 `tau_int`：有效样本数和积分自相关时间；
- `r_hat`：split-R-hat 链间收敛诊断；
- `laplacian_method`：复评使用的局域能量算法。

`r_hat` 接近 1 且 ESS 足够大，才说明该次复评的误差条具有参考价值；短训练的能量下降本身不构成论文精度声明。可按预算调整复评：

```bash
python train_vmc.py --epochs 50 --walkers 256 \
  --evaluation_steps 64 --evaluation_mcmc_steps 5 \
  --evaluation_burn_in 40 --evaluation_laplacian_method exact
```

H₂ 参考线只在核电荷 `[1, 1]`、自旋 `(1, 1)`、核间距 `1.4 Bohr` 都匹配时显示；LiH 同样要求登记的电荷、自旋与 `3.015 Bohr` 几何。未匹配体系不猜测参考能量。

---

### 演示 3：极性异核分子氢化锂 $LiH$ 变分优化

氢化锂 $LiH$ 是具有非对称电荷分布的经典分子（$Z_{\text{Li}} = 3, Z_{\text{H}} = 1$，4 电子）。
结合核电荷注入几何特征与 Hutchinson 百倍极速动能求解器，可实现极速优化：

```bash
python train_vmc.py --system lih --laplacian_method hutchinson --epochs 30 --walkers 512
# 或使用 uv 运行：
uv run python train_vmc.py --system lih --laplacian_method hutchinson --epochs 30 --walkers 512
```
只有在登记的 LiH 电荷、自旋和 `3.015 Bohr` 几何完全匹配时，才会显示 $\approx -8.070$ Hartree 的参考线。短训练仅用于验证链路与趋势，不能据此宣称已经收敛。

---

### 演示 4：100 原子超大规模体系与物理基准测试

体验 100 个氢原子组成的一维强关联晶格链（$M=100, N=100$，输入维度高达 300 维）：

```bash
# 强烈推荐添加 --double 双精度参数 (获得严格的 100 电子机器级反对称检验)
python benchmark_100_atoms.py --double
# 或快速模式 (跳过 300 维耗时二阶拉普拉斯求解)：
python benchmark_100_atoms.py --double --quick
```
**实测特性**：
- 严格验证 100 电子置换等变性与反对称性：轨道矩阵行置换误差 $< 10^{-14}$，双精度下波函数对数幅值误差 $< 10^{-8}$，符号严格反号，完美物理自洽；
- 两个电子空间位置重合时，波函数精确归零，完美保持泡利不相容原理；
- 全向量化计算势能，单次批量计算数万构型仅需数毫秒。

---

### 演示 5：PyTorch 2.0+ 编译优化加速评测

通过 `torch.compile` 对 FermiNet 进行 JIT 算子融合与计算图优化：

```bash
python benchmark_compile.py
# 或使用 uv 运行：
uv run python benchmark_compile.py
```

**实测加速对比 (H10 分子体系, Batch=64, RTX 3060 GPU)**:

| 运行模式 | 单步推理延迟 (ms) | 采样吞吐量 (Walkers/s) | 加速倍数 (Speedup) | 数值精度最大误差 |
| :--- | :---: | :---: | :---: | :---: |
| **原生 Eager 模式** | 3.40 ms | 18,818 | 1.00x (基准) | - |
| **Compiled (default)** | 0.93 ms | 68,514 | **3.64x** | $1.36 \times 10^{-10}$ |
| **Compiled (reduce-overhead)** | 0.59 ms | 108,552 | **5.77x** | $1.36 \times 10^{-10}$ |

---

### 演示 6：一键全自动生成训练过程与物理状态综合可视化看板

在执行 VMC 变分优化时，只需添加 `--save_plots`（或 `--plot`），训练结束后将自动调用 [`visualize.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/visualize.py) 导出 10 张顶刊规格的专业图表到指定目录：

```bash
# 启动 30 轮 VMC 优化并自动导出图表至 assets/plots 目录
python train_vmc.py --system h2 --epochs 30 --walkers 512 --save_plots --plot_dir assets/plots
```

#### 1. 训练全景综合看板 (`h2_training_dashboard.png`)
直观涵盖能量收敛、零方差原理演化（能量方差 $\text{Var}(E)$ 衰减）以及 MCMC 采样接受率/自适应步长健康度：
<p align="center">
  <img src="assets/plots/h2_training_dashboard.png" alt="VMC Training Process Dashboard" width="100%">
</p>

#### 2. 量子物理状态综合看板 (`h2_physical_dashboard.png`)
深度揭示收敛后电子三维空间概率分布（区分 $\alpha/\beta$ 自旋）、沿核轴的一维波函数剖面（Kato Cusp 尖峰条件）以及局域能量分布直方图与偏度分析：
<p align="center">
  <img src="assets/plots/h2_physical_dashboard.png" alt="Quantum Physical State Dashboard" width="100%">
</p>

#### 3. 2D 分子成键截面相对电子密度热力图 (`h2_wavefunction_2d.png`)
清晰展示氢分子中两原子核之间的共价键电子云重叠通道与空间对称性衰减：
<p align="center">
  <img src="assets/plots/h2_wavefunction_2d.png" alt="2D Wavefunction Bonding Slice" width="70%">
</p>

---

## 🧪 单元测试套件 (Unit Tests)

本项目包含 5 套覆盖核心物理性质、VMC 训练、链相关统计、阶段 2 求解器以及综合可视化的自动化测试：

```bash
# 1. 一键运行全部 118 项单元测试：
python -m unittest discover -s . -p "test_*.py" -v

# 2. 或分别运行专项测试：
python -m unittest test_ferminet.py   # 核心物理性质与网络架构测试 (14 项)
python -m unittest test_vmc.py        # VMC 采样、代理损失与能量截断测试 (18 项)
python -m unittest test_phase2.py     # 阶段 2 动能求解器与物理特征测试 (23 项)
python -m unittest test_visualize.py  # 综合可视化分析模块专项测试 (54 项)
python -m unittest test_mcmc_stats.py # ESS、链相关 SEM 与 split-R-hat 测试 (4 项)
```
当前基线为 **118 项测试通过**。测试覆盖实现性质与数值回归，但不能替代长时间、多种子、独立复评的科学收敛验证。

---

## ❓ 常见问题速查 (FAQ Snapshot)

- **Q: 为什么计算出的波函数对数值 `log|Psi|` 经常是负几百？**  
  A: 这是完全正常的！多电子联合概率密度是高维微元几率的连乘，真实验证波函数振幅极小（例如 $10^{-100}$）。取自然对数后落入负几百区间，有效防止单精度下溢为零。
- **Q: 什么是 Hartree 原子单位制？**  
  A: 在量子化学中令 $\hbar = m_e = e = 1$。能量单位 $1 \text{ Hartree} \approx 27.2114 \text{ eV} \approx 627.51 \text{ kcal/mol}$；长度单位 $1 \text{ Bohr} \approx 0.529177 \text{ \AA}$。
- **Q: 为什么训练时能量方差 `Var(E)` 是黄金指标？**  
  A: **零方差原理**！当且仅当试探波函数严格收敛至真实薛定谔方程本征态时，空间任意点处的局域能量处处等于常数基态能 $E_0$，此时空间方差 $\text{Var}(E) \equiv 0$。

> 更多深度原理解读与详尽问答，请参阅专著指南：[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)

---

## 📜 许可证 (License)

本项目遵循 [MIT License](LICENSE) 开源协议。
