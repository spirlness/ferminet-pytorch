# FermiNet (Fermi Neural Network) PyTorch 极简教学项目

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#)

这是一个面向初学者的 **FermiNet (费米神经网络)** 极简 PyTorch 教学项目。旨在通过最清晰直观的代码与最小化的依赖，帮助学习者掌握深度学习求解量子化学多电子薛定谔方程的核心思想与实现方法。近期更引入了**自旋纯态投影 (Spin Projection)** 与 **显式 Kato 尖峰 Jastrow 因子 (Kato Cusp Jastrow)**，在网络架构底层严格消除自旋污染与库仑奇点发散。

> 📖 **参考论文**：  
> **Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks**  
> *David Pfau, James S. Spencer, Alexander G. de G. Matthews, W. M. C. Foulkes*  
> *Physical Review Research* 2, 033429 (2020) · [DOI: 10.1103/PhysRevResearch.2.033429](https://doi.org/10.1103/PhysRevResearch.2.033429)

---

## 🎯 核心物理与深度学习原理

FermiNet 使用深度神经网络直接参数化多电子试探波函数 $\Psi_\theta(\mathbf{r}_1, \dots, \mathbf{r}_N)$，并利用变分量子蒙特卡洛 (VMC) 最小化能量期望值，逼近基态薛定谔方程 $\hat{H}\Psi = E\Psi$ 的解。本项目完整实现了以下七大物理与计算支柱：

1. **同自旋费米子反对称性 (泡利不相容原理)**：
   电子是全同费米子，交换任意两个同自旋电子坐标时，波函数必须严格反号：
   $$\Psi(\dots, \mathbf{r}_i, \dots, \mathbf{r}_j, \dots) = -\Psi(\dots, \mathbf{r}_j, \dots, \mathbf{r}_i, \dots)$$
   FermiNet 通过将神经网络构建的单电子连续轨道送入 **Slater 行列式** 实现这一物理约束。

2. **自旋等变特征交互 (Backflow 机制)**：
   电子不仅受原子核吸引，电子之间还会互相排斥与瞬时避让。网络维护单电子流 $h_i$ 与双电子流 $h_{ij}$，通过对同自旋电子进行均值池化，在保证等变性的前提下充分捕捉电子多体动态关联。

3. **束缚态核包络衰减 (Nuclear Envelope)**：
   为保证电子在远离分子原子核时波函数以物理上正确的速率指数衰减至零，引入各向同性指数核包络乘子：
   $$\Phi_{k,i,j} = \phi_{k,i,j} \cdot \sum_m \pi_{k,j,m} e^{-\alpha_{k,j,m} |\mathbf{r}_i - \mathbf{R}_m|}$$
   当配合显式核尖峰 Jastrow 因子时，还支持平滑高斯包络（$\exp(-\alpha |\mathbf{r}_i - \mathbf{R}_m|^2)$），使网络轨道在原点附近平滑解析。

4. **【核心前沿】自旋纯态投影与自旋污染消除 (`spin.py`, `SpinProjector`)**：
   标准单个 Slater 行列式或无约束 Backflow 轨道并不是总自旋角动量算符 $\hat{S}^2$ 的严格本征态，往往存在非物理的“自旋污染”（例如单重态基态中混杂三重态等高自旋杂质）。
   本项目通过空间置换投影算符在波函数输出前施加自旋对称化：
   - **双电子单重态 (Singlet, $S=0$)**：施加空间置换严格对称化 $\hat{P}_{S=0} = \frac{1}{2}(I + P_{12})$，使 $\Psi(\mathbf{r}_1, \mathbf{r}_2) = \Psi(\mathbf{r}_2, \mathbf{r}_1)$，严格消除自旋污染并使 $\hat{S}^2 \Psi = 0$；
   - **双电子三重态 (Triplet, $S=1, M_S=0$)**：施加空间置换反对称化 $\hat{P}_{S=1} = \frac{1}{2}(I - P_{12})$；
   - **多电子体系**：利用 $\hat S^2$ 的谱投影多项式 $\hat P_S=\prod_{s\ne S}(\hat S^2-s(s+1)I)/(S(S+1)-s(s+1))$ 选择单重态或三重态。例如 LiH 单重态投影会排除五重态分量，并验证 $\hat S^2\Psi=0$；
   - 基础波函数须在各自旋组内反对称；不相容的电子数与目标自旋会报错。完全极化体系已具有 $S=N/2$，`spin_adapted=True` 默认使用恒等变换；
   - 使用带符号对数求和 (**Signed Log-Sum-Exp**)；多电子投影的置换数至多为 $\binom{N}{n_\uparrow}$，LiH 单重态使用 6 项，较双电子投影开销更高。

5. **【核心前沿】Kato 尖峰边界条件与解析 Jastrow 因子 (`jastrow.py`, `KatoCuspJastrow`)**：
   在库仑哈密顿量中，微观粒子重合碰撞时库仑势能发散（$r \to 0$ 时 $V \propto -Z/r$ 或 $1/r$）。为保证局域能量 $E_L = T_L + V$ 在重合点保持有限常数，波函数对数导数必须满足精确的 Kato 尖峰条件 (Kato Cusp Conditions)：
   - **电子-核碰撞** ($r_{iI} \to 0$)：$\left. \frac{\partial \ln|\Psi|}{\partial r_{iI}} \right|_{r_{iI}=0} = -Z_I$；
   - **反平行自旋电子对** (异自旋 $r_{ij} \to 0$)：$\left. \frac{\partial \ln|\Psi|}{\partial r_{ij}} \right|_{r_{ij}=0} = \frac{1}{2}$；
   - **平行自旋电子对** (同自旋 $r_{ij} \to 0$)：$\left. \frac{\partial \ln|\Psi|}{\partial r_{ij}} \right|_{r_{ij}=0} = \frac{1}{4}$。
   这些系数约束相应角动量通道的径向因子；同自旋对需要先除去线性节点。双电子三重态的空间部分反对称，包括 $M_S=0$ 的上下自旋对，也使用 $1/4$。`FermiNet` 会根据内置投影模式自动配置 Jastrow；独立构造时使用 `two_electron_spin=1`。显式配置与投影不相容时会报错。
   本项目采用 Padé 阻尼形式的显式解析 Jastrow 因子：
   $$J(\mathbf{r}) = \exp(U_{en}(\mathbf{r}) + U_{ee}(\mathbf{r}))$$
   $$U_{en}(\mathbf{r}) = -\sum_{i=1}^N \sum_{I=1}^M \frac{Z_I r_{iI}}{1 + b_{en, I} r_{iI}}, \quad U_{ee}(\mathbf{r}) = \sum_{1 \le i < j \le N} \frac{\gamma_{ij} r_{ij}}{1 + b_{ee, ij} r_{ij}}$$
   在微观重合点**完全解析抵消库仑势能奇点**，将神经网络从拟合剧烈奇异峰的繁重负担中解放出来。更可结合平滑无尖峰特征 `smooth_cuspless=True`（以 $d^2 / (1+d)$ 代替线性距离 $d$），消除特征在原点的高阶拉普拉斯发散。

6. **局域能量 (Local Energy) 与尖峰奇异性诊断 (`hamiltonian.py`)**：
   在采样构型 $\mathbf{r}$ 处，局域能量定义为：
   $$E_L(\mathbf{r}) = \frac{\hat{H}\Psi(\mathbf{r})}{\Psi(\mathbf{r})} = -\frac{1}{2} \left(\nabla^2 \ln|\Psi| + |\nabla \ln|\Psi||^2\right) + V(\mathbf{r})$$
   动能项通过 PyTorch 自动微分精确计算二阶拉普拉斯。提供 `diagnose_cusp_divergence` 工具，可径向切片扫描电子逼近原子核时的能量发散行为，直观验证尖峰对消效果。

7. **变分代理梯度与 MCMC 采样 (`loss.py`, `mcmc.py`)**：
   通过 Metropolis-Hastings (MH) 算法按 Born 概率密度 $p(\mathbf{r}) \propto |\Psi(\mathbf{r})|^2$ 采样电子构型，变分能量梯度表达为协方差形式：
   $$\nabla_\theta \langle E \rangle = 2 \, \mathbb{E}_{\mathbf{r} \sim |\Psi|^2} \left[ (E_L(\mathbf{r}) - \langle E_L \rangle) \nabla_\theta \ln|\Psi_\theta(\mathbf{r})| \right]$$
   构造代理损失 $\mathcal{L}(\theta) = 2 \cdot \text{mean}((E_{\text{clip}} - \bar{E}_{\text{clip}}) \ln|\Psi_\theta|)$，避免三阶导数计算，并结合基于 MAD 的 Umrigar Tanh 稳健截断防止优化器梯度崩溃。
   部分非有限局域能量会被排除，并发出警告；`stats` 和训练 `history` 记录 `num_valid`、`num_invalid`。所有能量无效或对数波函数非有限时，抛出 `FloatingPointError`，在优化器更新前停止训练。

---

## 💡 最小化教学架构设计 (Minimalist Refactoring)

为杜绝传统科学计算代码中常见的**过度工程化与冗余抽象**，本项目经过彻底重构，保持极致的纯净与易读性：

1. **消除无意义的参数梯度追踪**：
   在变分量子蒙特卡洛中，局域能量 $E_L(\mathbf{r}) = T_L(\mathbf{r}) + V(\mathbf{r})$ 是空间采样的**观测标量**。动能拉普拉斯算子是对空间坐标 $\mathbf{r}$ 求二阶导，而非对网络权重 $\theta$ 求导。反向传播遵循协方差定理进行梯度解耦 (`detach()`)。本项目彻底移除了对参数的手动状态暂存和高阶导冗余图，动能求解干净利落。

2. **聚焦经典标准算法**：
   - 能量截断统一采用量子化学界标准、基于中位数绝对偏差 (MAD) 的 **Umrigar Tanh 平滑截断**，剔除复杂的条件判断与冗余警告；
   - MCMC 采用完全向量化的**全电子一体化高斯随机游走**与目标接受率动态步长自适应调节；
   - 预计算并缓存核-核库仑排斥常数 $V_{nn}$，支持无缝设备迁移 (`.to()`)。

3. **直观清晰的训练主循环**：
   消除黑盒多层封装，在 [`train_vmc.py`](file:///c:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/train_vmc.py) 中任何人花 1 分钟就能看懂 VMC 的执行脉络：
   $$\text{MCMC 采样} \longrightarrow \text{计算 } E_L(\mathbf{r}) \longrightarrow \text{计算代理损失 } \mathcal{L}(\theta) \longrightarrow \text{梯度反向传播更新}$$

---

## 📂 项目结构（只保留最核心模块）

```text
ferminet_pytorch/
├── features.py     # 电子-核与电子-电子相对几何特征嵌入
├── layers.py       # 自旋分流均值池化的等变相互作用层
├── envelope.py     # 原子核指数/高斯衰减核包络模块
├── jastrow.py      # 显式满足电子-核与电子-电子边界的 Kato Cusp Jastrow 因子
├── spin.py         # 自旋投影算符 (双电子单重态空间对称化 1/2(I + P12))
├── model.py        # FermiNet 深度波函数模型 (Slater 行列式求和，融合 Jastrow 与自旋适配)
├── hamiltonian.py  # 分子体系库仑势能、自动微分局域动能与尖峰奇异性诊断
├── mcmc.py         # 批量 Metropolis-Hastings 采样器与自适应步长
├── loss.py         # 变分代理损失与基于 MAD 的稳健能量截断
├── train_vmc.py    # VMC 端到端优化训练入口与教程脚本
├── example_h2.py   # 氢分子 (H2) 快速前向推理与局域能量拆解演示
├── test_ferminet.py# 核心波函数、特征与物理不变量测试
├── test_vmc.py     # MCMC 采样、代理损失与微型训练测试
├── test_spin_and_cusp.py # 自旋纯态、Kato 尖峰导数极限与库仑奇点抵消测试
└── pyproject.toml  # 项目依赖与打包配置 (仅依赖 PyTorch 与 NumPy)
```

---

## 📐 张量维度规范 (Tensor Shape Specification)

全模块严格遵循以下张量规范：

| 张量名称 | 数学符号 | 形状 (Shape) | 物理含义与说明 |
| :--- | :--- | :--- | :--- |
| `r` | $\mathbf{r}$ | `(B, N, 3)` | $B$ 个 MCMC Walkers，每个构型包含 $N$ 个电子的 3 维笛卡尔坐标 |
| `nuclear_coords` | $\mathbf{R}$ | `(M, 3)` | $M$ 个原子核的固定三维坐标 (Bohr) |
| `nuclear_charges` | $Z$ | `(M,)` | $M$ 个原子核的核电荷量 (原子序数) |
| `h1` | $h_i^l$ | `(B, N, d_1)` | 第 $l$ 层的单电子流隐藏特征 ($d_1$ 默认 64) |
| `h2` | $h_{ij}^l$ | `(B, N, N, d_2)` | 第 $l$ 层的双电子流隐藏特征 ($d_2$ 默认 16) |
| `d_up` | $D_k^\uparrow$ | `(B, K, n_up, n_up)` | 自旋向上电子的分子轨道矩阵 ($K$ 为行列式分支数) |
| `d_down` | $D_k^\downarrow$ | `(B, K, n_down, n_down)` | 自旋向下电子的分子轨道矩阵 |
| `log_j` | $\ln J(\mathbf{r})$ | `(B,)` | Kato 尖峰 Jastrow 因子对数值 ($U_{en} + U_{ee}$) |
| `log_abs_psi` | $\ln|\Psi(\mathbf{r})|$ | `(B,)` | 最终波函数的对数绝对值（保障数值防下溢） |
| `sign_psi` | $\text{sign}(\Psi)$ | `(B,)` | 最终波函数符号值（$-1, 0, 1$） |
| `e_local` | $E_L(\mathbf{r})$ | `(B,)` | 各采样构型处的局域能量标量（单位：Hartree） |

---

## 🏗️ 完整波函数端到端架构数据流

```text
               ┌─────────────────────────────────────────────────────────────┐
               │                电子三维空间坐标 r: (B, N, 3)                │
               │               原子核坐标与电荷 R, Z: (M, 3), (M,)           │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
                      【几何特征嵌入 ElectronFeatureEmbedding features.py】
                           (可选平滑无尖峰特征: d^2 / (1 + d))
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               │ 单电子流特征 h1: (B, N, 5M)   │ 双电子流特征 h2: (B, N, N, 4) │
               │   [r_{iI}, d_{iI}, Z_I]       │   [r_{ij}, d_{ij}]            │
               └──────────────────────┬───────┴──────────────┬───────────────┘
                                      │                      │
                        【自旋分流交互层 FermiInteractionLayer layers.py】
                        (L 层循环：同自旋/异自旋分离均值池化 + 残差连接)
                                      │                      │
                                      ▼                      ▼
               ┌─────────────────────────────────────────────────────────────┐
               │    高阶单电子隐藏表征: h1_final (B, N, hidden_dim_one)       │
               └──────────────────────┬──────────────────────┬───────────────┘
                                      │                      │
                         (切片前 n_up 个电子)   (切片后 n_down 个电子)
                                      │                      │
                      ┌────────────────┴────┐            ┌────┴────────────────┐
                      │ 自旋向上分支 (Up)   │            │ 自旋向下分支 (Down) │
                      └────────┬────────────┘            └────────┬────────────┘
                               │                                  │
                  【线性投影 Backflow 变换】            【线性投影 Backflow 变换】
                               │                                  │
                      backflow_up: (B, K, n_up, n_up)    backflow_down: (B, K, n_down, n_down)
                               │                                  │
                  【核包络 NuclearEnvelope】            【核包络 NuclearEnvelope】
                  (指数型 或 平滑高斯型)                (指数型 或 平滑高斯型)
                               │ (逐元素相乘 Hadamard)           │ (逐元素相乘 Hadamard)
                               ▼                                  ▼
                  ┌─────────────────────────┐        ┌─────────────────────────┐
                  │ 分子轨道矩阵 D_up       │        │ 分子轨道矩阵 D_down     │
                  │ Shape: (B, K, n_up, n_up)│        │ Shape: (B, K, n_dn, n_dn)│
                  └────────────┬────────────┘        └────────────┬────────────┘
                               │                                  │
                               └────────────────┬─────────────────┘
                                                │
                                  【反对称行列式分解 model.py】
                                                │
                                   torch.linalg.slogdet(D)
                                                │
                                  (sign_up, log|det_up|) + (sign_down, log|det_down|)
                                                │
                                     多行列式加权求和 (LogSumExp)
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │  基础 Slater 波函数:             │
                               │  log|Psi_base|, sign_base        │
                               └────────────────┬─────────────────┘
                                                │
                                 【Kato 尖峰 Jastrow 因子 jastrow.py】
                                 ln J(r) = U_en(r) + U_ee(r) (Padé 形式)
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │  包含尖峰对消波函数:             │
                               │  log|Psi_unproj| = log_base + lnJ│
                               └────────────────┬─────────────────┘
                                                │
                                 【自旋投影算符 SpinProjector spin.py】
                                 Psi_proj = sum_k w_k * Psi(P_k r)
                                 (Signed Log-Sum-Exp 稳定对数加权)
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │  最终物理波函数输出:             │
                               │  - log_abs_psi: log|Psi(r)| (B,) │
                               │  - sign_psi: sign(Psi(r))   (B,) │
                               │  - psi: sign * exp(log|Psi|) (B,)│
                               └──────────────────────────────────┘
```

---

## ⚡ 极速上手实操

### 1. 运行基础波函数推断与局域能量拆解演示

计算氢分子 ($H_2$) 在未训练初态下的波函数与局域动能、库仑势能拆解：

```bash
uv run python example_h2.py
```

### 2. 运行端到端 VMC 变分训练

#### (A) 经典基线训练 (默认轻量 FermiNet)
```bash
# 运行 50 轮 VMC 优化，采样 512 个并行构型
uv run python train_vmc.py --epochs 50 --walkers 512
```

#### (B) 开启自旋纯态投影与显式 Kato Cusp Jastrow 因子 (物理完整增强版)
```bash
# 严苛消除单重态自旋污染与库仑奇点发散
uv run python train_vmc.py --epochs 50 --walkers 512 --cusp_jastrow --spin_adapted
```

#### (C) 4 电子极性异核分子氢化锂 ($LiH$) 训练
```bash
uv run python train_vmc.py --system lih --epochs 100 --walkers 512 --cusp_jastrow --spin_adapted
```

训练过程将实时输出每个 Epoch 的物理能量指标、截断能量、能量方差、MCMC 接受率与步长：
```text
Epoch  |  <E_raw> (Ha)  | <E_clip> (Ha)  |   Var(E)   | Acc (%)  |  Step  
------------------------------------------------------------------------
  1    |    -0.35043    |    -0.43569    |   0.8531   |   54.7   | 0.5402 
 25    |    -1.02511    |    -1.01844    |   0.1412   |   51.2   | 0.4851
 50    |    -1.14820    |    -1.14302    |   0.0683   |   50.4   | 0.4721
```

### 3. Kato 尖峰库仑奇异性诊断示例

利用内置的 `diagnose_cusp_divergence` 诊断工具，可验证电子碰撞原子核时 ($r_{iI} \to 0$)，Jastrow 因子如何使局域能量保持有限：

```python
import torch
from ferminet_pytorch import build_h2_system, FermiNet, diagnose_cusp_divergence

system = build_h2_system()
# 构建启用了 Kato 尖峰 Jastrow 因子的 FermiNet
model = FermiNet(
    nuclear_coords=system.nuclear_coords,
    nuclear_charges=system.nuclear_charges,
    n_up=system.n_up,
    n_down=system.n_down,
    use_cusp_jastrow=True,
)

r_base = torch.randn(1, system.total_electrons, 3)
distances = torch.logspace(-4, 0, steps=20)  # 扫描 1e-4 到 1.0 Bohr
diag = diagnose_cusp_divergence(
    model=model,
    system=system,
    r_base=r_base,
    electron_idx=0,
    nucleus_idx=0,
    distances=distances,
)

print("近核逼近局域能量 E_L(r):", diag["e_local"])
```

### 4. 完整命令行 CLI 参数对照表

| 参数名 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--system` | `str` | `h2` | 目标物理体系，可选 `h2` ($H_2$ 分子) 或 `lih` ($LiH$ 分子) |
| `--epochs` | `int` | `50` | VMC 变分优化迭代轮数 |
| `--walkers` | `int` | `512` | MCMC 并行构型采样数量 |
| `--mcmc_steps` | `int` | `10` | 每轮参数更新前执行的 MCMC 随机游走步数 |
| `--burn_in` | `int` | `50` | 初始平衡预热步数 |
| `--lr` | `float` | `1e-3` | Adam 优化器学习率 |
| `--step_size` | `float` | `0.5` | MCMC 初始移动步长 |
| `--move_mode` | `str` | `all` | MCMC 移动模式：`all` (全电子一体化移动) 或 `one_by_one` (逐电子循环) |
| `--clip_factor` | `float` | `5.0` | 局域能量 MAD 截断半宽倍数 $\Delta = c \times \text{MAD}$ |
| `--clip_method` | `str` | `tanh` | 局域能量平滑截断方法：`tanh`、`hard` 或 `none` |
| `--grad_clip` | `float` | `1.0` | 梯度范数裁剪阈值 |
| `--cusp_jastrow` | `flag` | `False` | **启用显式 Kato 尖峰 Jastrow 因子**，消除粒子碰撞奇点 |
| `--spin_adapted` | `flag` | `False` | **启用自旋纯态投影** (如双电子单重态空间对称化 $\frac{1}{2}(I + P_{12})$) |
| `--envelope_type` | `str` | `None` | 核包络类型：`gaussian` (平滑高斯) 或 `exponential` (经典指数) |
| `--smooth_cuspless` | `flag` | `False` | 强制几何特征使用无尖峰平滑距离 $d^2 / (1+d)$ |
| `--device` | `str` | `None` | 指定运行设备 (`cuda` 或 `cpu`) |
| `--seed` | `int` | `42` | 随机数种子 |

---

## 🧪 自动化测试验证

运行完整自动化单元测试套件：

```bash
uv run python -m unittest discover -s . -p "test_*.py" -v
```

测试集涵盖核心物理、训练闭环及发布交付回归：
1. **`test_ferminet.py` (基础网络与物理对称性)**：
   - 几何特征维度与无 NaN 边界；
   - 相互作用层自旋置换等变性；
   - 指数包络衰减与形状规范；
   - 泡利不相容原理（两同自旋电子重合时 $\Psi = 0$ 且无 NaN 梯度）；
   - 自动微分局域动能与库仑势能计算；
   - 核-核排斥常数 $V_{nn}$ 预计算与无缝设备迁移 (`.to()`)。
2. **`test_vmc.py` (采样与优化闭环)**：
   - 批量 Metropolis-Hastings 接受率与 50% 目标自适应步长调节；
   - 基于 MAD 的 Umrigar Tanh 与 Hard 能量截断平滑性及极值鲁棒性；
   - 协方差变分代理损失与无偏梯度反向传播；
   - 单电子极化体系 (如氢原子 $H$) 与双电子体系微型端到端训练闭环。
3. **`test_spin_and_cusp.py` (自旋纯态与 Kato 尖峰物理极限)**：
   - **自旋单重态空间交换对称性**：严格验证 $\Psi(\mathbf{r}_1, \mathbf{r}_2) = \Psi(\mathbf{r}_2, \mathbf{r}_1)$，误差 $< 10^{-10}$；
   - **自旋算符本征态**：验证总自旋算符期望值 $\hat{S}^2 \Psi_{\text{singlet}} = (1 - P_{12}) \Psi_{\text{singlet}} = 0$；
   - **自旋三重态空间反对称性**：验证 $\Psi(\mathbf{r}_1, \mathbf{r}_2) = -\Psi(\mathbf{r}_2, \mathbf{r}_1)$ 且重合点严格为 0；
   - **闭壳层空间翻转对称性**：验证 $LiH$ ($n_\uparrow=2, n_\downarrow=2$) 空间自旋翻转对称；完整 $S^2$ 本征态条件另由回归测试验证；
   - **Kato 尖峰导数解析极限**：
     - 电子-核碰撞极限：$\left. \frac{d\ln|\Psi|}{dr_{iI}} \right|_{r_{iI} \to 0} = -Z_I$ (针对 $Z=1, Z=3$)；
     - 异自旋电子对碰撞极限：$\left. \frac{d\ln|\Psi|}{dr_{ij}} \right|_{r_{ij} \to 0} = 0.50$；
     - 同自旋电子对碰撞极限：$\left. \frac{d\ln|\Psi|}{dr_{ij}} \right|_{r_{ij} \to 0} = 0.25$；
   - **库仑奇点发散完全抵消**：数值验证电子贴近原子核与贴近另一电子时，动能发散与势能发散精确抵消，局域能量 $E_L(r)$ 保持有限常数。
4. **新增回归测试**：`test_determinants.py` 验证奇异行列式的一、二阶导数及极小波函数；`test_packaging.py` 构建并从源码目录外导入发布包；`test_p2_regressions.py` 验证多电子 $S^2$ 本征态、投影幂等性、三重态碰撞极限、自定义置换和无效能量的更新阻断。
5. **CUDA 回归测试**：`test_cuda.py` 将上述数值回归显式放在 GPU 上执行，并比较 CPU/CUDA 的局域能量、坐标梯度，验证 H2/LiH 的参数更新和采样缓存均位于 CUDA。使用安装了 CUDA 版 PyTorch 的解释器运行：

   ```bash
   python -m unittest discover -s . -p "test_cuda.py" -v
   ```

   CPU 版 PyTorch 或无可用 GPU 时，这些测试明确显示 `skipped`，不能算作 CUDA 验证通过。`uv run` 使用项目环境中的 PyTorch；若该环境是 CPU 版，需要选择 CUDA 版解释器。

---

## ❓ 常见问题 FAQ

- **Q: 为什么对数波函数 `log|Psi|` 经常是负几或负几十？**  
  多电子联合概率密度是各坐标微元几率的连乘，真实振幅极小（如 $10^{-50}$）。使用 `torch.linalg.slogdet` 并在对数空间计算，能彻底防止浮点下溢。

- **Q: 什么是 Hartree 原子单位制？**  
  量子化学常用原子单位制（令 $\hbar = m_e = e = 1$）。$1 \text{ Hartree} \approx 27.2114 \text{ eV} \approx 627.51 \text{ kcal/mol}$；长度单位 $1 \text{ Bohr} \approx 0.529177 \text{ \AA}$。

- **Q: 为什么训练中要监控局域能量方差？**  
  根据量子力学零方差原理，当试探波函数严格达到哈密顿量的本征态时，局域能量在空间各点都等于本征能量常数，方差趋近于 0。因此能量均值下降且方差减小是变分收敛的关键标志。

---

## 📜 许可证

本项目遵循 MIT 开源许可证。
