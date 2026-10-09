# FermiNet (Fermi Neural Network) PyTorch 极简教学项目

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#)

这是一个面向初学者的 **FermiNet (费米神经网络)** 极简 PyTorch 教学项目。旨在通过最清晰直观的代码与最小化的依赖，帮助学习者掌握深度学习求解量子化学多电子薛定谔方程的核心思想与实现方法。

> 📖 **参考论文**：  
> **Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks**  
> *David Pfau, James S. Spencer, Alexander G. de G. Matthews, W. M. C. Foulkes*  
> *Physical Review Research* 2, 033429 (2020) · [DOI: 10.1103/PhysRevResearch.2.033429](https://doi.org/10.1103/PhysRevResearch.2.033429)

---

## 🎯 核心物理与深度学习原理

FermiNet 使用深度神经网络直接参数化多电子试探波函数 $\Psi_\theta(\mathbf{r}_1, \dots, \mathbf{r}_N)$，并利用变分量子蒙特卡洛 (VMC) 最小化能量期望值，逼近基态薛定谔方程 $H\Psi = E\Psi$ 的解。

1. **同自旋费米子反对称性 (泡利不相容原理)**：
   电子是费米子，交换任意两个同自旋电子坐标时，波函数必须严格反号：
   $$\Psi(\dots, \mathbf{r}_i, \dots, \mathbf{r}_j, \dots) = -\Psi(\dots, \mathbf{r}_j, \dots, \mathbf{r}_i, \dots)$$
   FermiNet 通过将神经网络构建的单电子轨道送入 **Slater 行列式** 实现这一物理约束。

2. **自旋等变特征交互 (Backflow 机制)**：
   电子不仅受原子核吸引，电子之间还会互相排斥与关联。网络维护单电子流 $h_i$ 与双电子流 $h_{ij}$，通过对同自旋电子进行均值池化，在保证等变性的前提下充分捕捉电子多体关联。

3. **指数包络 (Nuclear Envelope)**：
   为保证电子在远离分子原子核时波函数以物理上正确的速率指数衰减至零，引入各向同性指数核包络乘子：
   $$\Phi_{k,i,j} = \phi_{k,i,j} \cdot \sum_m \pi_{k,j,m} e^{-\alpha_{k,j,m} |\mathbf{r}_i - \mathbf{R}_m|}$$

4. **局域能量 (Local Energy)**：
   在采样构型 $\mathbf{r}$ 处，局域能量定义为：
   $$E_L(\mathbf{r}) = \frac{H\Psi(\mathbf{r})}{\Psi(\mathbf{r})} = -\frac{1}{2} \left(\nabla^2 \ln|\Psi| + |\nabla \ln|\Psi||^2\right) + V(\mathbf{r})$$
   动能项通过 PyTorch 自动微分精确计算二阶拉普拉斯。

5. **变分代理梯度与 MCMC 采样**：
   通过 Metropolis-Hastings (MH) 算法按 Born 概率密度 $p(\mathbf{r}) \propto |\Psi(\mathbf{r})|^2$ 采样电子构型，变分能量梯度表达为协方差形式：
   $$\nabla_\theta \langle E \rangle = 2 \, \mathbb{E}_{\mathbf{r} \sim |\Psi|^2} \left[ (E_L(\mathbf{r}) - \langle E_L \rangle) \nabla_\theta \ln|\Psi_\theta(\mathbf{r})| \right]$$
   构造代理损失 $L(\theta) = 2 \cdot \text{mean}((E_{\text{clip}} - \bar{E}_{\text{clip}}) \ln|\Psi_\theta|)$，避免三阶导数计算。

---

## 📂 项目结构（只保留最核心模块）

```text
ferminet_pytorch/
├── features.py     # 电子-核与电子-电子相对几何特征嵌入
├── layers.py       # 自旋分流均值池化的等变相互作用层
├── envelope.py     # 原子核指数衰减核包络模块
├── model.py        # FermiNet 深度波函数模型 (Slater 行列式求和)
├── hamiltonian.py  # 分子体系库仑势能与自动微分局域动能计算
├── mcmc.py         # 批量 Metropolis-Hastings 采样器与自适应步长
├── loss.py         # 变分代理损失与基于 MAD 的稳健能量截断
├── train_vmc.py    # VMC 端到端优化训练入口与教程脚本
├── example_h2.py   # 氢分子 (H2) 快速前向推理与局域能量拆解演示
├── test_ferminet.py# 核心波函数、特征与物理不变量测试
├── test_vmc.py     # MCMC 采样、代理损失与微型训练测试
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
| `log_abs_psi` | $\ln|\Psi(\mathbf{r})|$ | `(B,)` | 波函数的对数绝对值（保障数值防下溢） |
| `sign_psi` | $\text{sign}(\Psi)$ | `(B,)` | 波函数符号值（$-1, 0, 1$） |
| `e_local` | $E_L(\mathbf{r})$ | `(B,)` | 各采样构型处的局域能量标量（单位：Hartree） |

---

## ⚡ 极速上手实操

### 1. 运行波函数推断与局域能量拆解演示

计算氢分子 (H2) 在未训练初态下的波函数与局域动能、库仑势能拆解：

```bash
uv run python example_h2.py
```

### 2. 运行端到端 VMC 变分训练

使用变分量子蒙特卡洛算法训练优化网络参数，最小化局域能量期望值：

```bash
# 运行 50 轮 VMC 优化，采样 512 个并行构型
uv run python train_vmc.py --epochs 50 --walkers 512
```

训练过程将打印每个 Epoch 的能量均值、截断能量、方差、MCMC 接受率与步长：
```text
Epoch  |  <E_raw> (Ha)  | <E_clip> (Ha)  |   Var(E)   | Acc (%)  |  Step  
------------------------------------------------------------------------
  1    |    -0.35043    |    -0.43569    |   0.8531   |   54.7   | 0.5402 
 25    |    -1.02511    |    -1.01844    |   0.1412   |   51.2   | 0.4851
 50    |    -1.14820    |    -1.14302    |   0.0683   |   50.4   | 0.4721
```

也可切换为 4 电子分子氢化锂 (LiH)：
```bash
uv run python train_vmc.py --system lih --epochs 100 --walkers 512
```

---

## 🧪 自动化测试验证

运行完整单元测试套件：

```bash
uv run python -m unittest discover -s . -p "test_*.py" -v
```

测试集涵盖：
- `test_ferminet.py`：几何特征嵌入维度、等变层置换等变性、指数包络衰减、泡利不相容原理与同自旋反对称性、自动微分局域动能与势能计算。
- `test_vmc.py`：MCMC 构型采样、Metropolis-Hastings 接受率、自适应步长动态调整、局域能量平滑截断、变分代理损失梯度反向传播与微型端到端训练闭环。

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
