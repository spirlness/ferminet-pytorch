# FermiNet 从零到一：量子化学深度学习新手入门与实战宝典 (Zero to One Guide)

> **导读**：
> 如果你是一名计算机科学、人工智能或软件工程背景的同学，面对“量子力学”、“薛定谔方程”、“Slater 行列式”、“变分量子蒙特卡洛”这些看似高深晦涩的物理学名词，常常会望而却步。
> 
> **请放心！** 编写这份文档的目的，就是为了**彻底打破物理与人工智能之间的专业壁垒**。我们将抛弃枯燥难懂的物理教科书黑话，用程序员最容易理解的直白语言、生动的直觉比喻、清晰的 ASCII 架构图和可运行的代码，带你从零看懂 FermiNet（费米神经网络）是如何用深度学习破解量子化学世纪难题的！

---

## 目录 (Table of Contents)

1. [💡 写在前面：为什么用神经网络算量子力学？](#1-写在前面为什么用神经网络算量子力学)
2. [⚛️ 第一章：量子力学与量子化学极简极速前置](#2-第一章量子力学与量子化学极简极速前置)
   - 2.1 什么是薛定谔方程？微观世界的“牛顿第二定律”
   - 2.2 为什么电子多了解不出解析解？可怕的“指数墙”
   - 2.3 什么是电子波函数 $\Psi(r)$ 与玻恩几率解释 $|\Psi|^2$？
   - 2.4 费米子置换反对称性与泡利不相容原理：“撞衫即毁灭”
   - 2.5 传统化学方法是怎么做的？Hartree-Fock 与 Slater 行列式的局限
   - 2.6 FermiNet 深度学习 Backflow 机制做了什么突破？
3. [🧠 第二章：FermiNet 神经网络核心架构超清晰拆解](#3-第二章ferminet-神经网络核心架构超清晰拆解)
   - 3.1 [整体架构全景数据流 (学术期刊规范架构图)](#31-整体架构全景数据流-学术期刊规范架构图)
   - 3.2 几何特征提取 (`features.py`)：相对位移、距离与核电荷注入
   - 3.3 自旋分流交互层 (`layers.py`)：单/双电子流与自旋分离池化
   - 3.4 原子核指数衰减包络 (`envelope.py`)：物理渐近条件与各向异性度规
   - 3.5 反对称波函数构建 (`model.py`)：Backflow 轨道矩阵构造
   - 3.6 数值稳定性基石：`torch.linalg.slogdet` 与 LogSumExp 技巧
4. [🎲 第三章：变分量子蒙特卡洛 (VMC) 极简教程](#4-第三章变分量子蒙特卡洛-vmc-极简教程)
   - 4.1 为什么高维积分必须用 MCMC 随机游走采样？
   - 4.2 Metropolis-Hastings 接受/拒绝机制
   - 4.3 什么是局域能量 $E_L(r)$？动能与势能的精妙平衡
   - 4.4 变分原理：为什么能量越低越逼近真理？
   - 4.5 核心物理公式纠偏：有限批次协方差代理损失与求导真相
   - 4.6 局域能量平滑截断 (Umrigar Tanh Clipping)：防止优化器“炸飞”
5. [🗺️ 第四章：阶段 0、1、2 核心演进与模块全景导览](#5-第四章阶段-012-核心演进与模块全景导览)
   - 5.1 源码文件与核心职责对照表
   - 5.2 三大高性能动能拉普拉斯求解器对比 (Exact / JVP / Hutchinson)
   - 5.3 PyTorch 2.0+ `torch.compile` JIT 图编译加速秘籍
6. [🚀 第五章：5 分钟上手极速实操 (含输出物理指标通俗解读)](#6-第五章5-分钟上手极速实操-含输出物理指标通俗解读)
   - 6.1 演示 1：氢分子 $H_2$ 基础波函数推断与局域能量 (`example_h2.py`)
   - 6.2 演示 2：氢分子 $H_2$ 端到端 VMC 变分训练 (`train_vmc.py`)
   - 6.3 演示 3：极性异核分子氢化锂 $LiH$ 变分优化
   - 6.4 演示 4：100 原子超大规模体系基准测试 (`benchmark_100_atoms.py`)
   - 6.5 演示 5：PyTorch 编译加速基准评测 (`benchmark_compile.py`)
   - 6.6 演示 6：一键生成训练与物理状态诊断看板 (`visualize.py`)
   - 6.7 训练输出物理指标白话对照表 (解读 `<E_raw>`, `Var(E)`, `Acc (%)` 等)
7. [❓ 第六章：新手高频问答 FAQ](#7-第六章新手高频问答-faq)
   - Q1: 为什么波函数的对数值 `log|Psi|` 经常是负几百？
   - Q2: 什么是 Hartree 原子单位制？怎么换算成熟悉的单位？
   - Q3: 为什么 100 电子大体系强烈推荐使用 double (`float64`) 精度？
   - Q4: 为什么 MCMC 接受率最好在 50% 附近？
   - Q5: 为什么训练时能量方差 `Var(E)` 会越来越小？(神奇的零方差原理)

---

## 1. 💡 写在前面：为什么用神经网络算量子力学？

在经典物理学中，要预测行星轨道或小球弹跳，我们用牛顿力学公式 $F = ma$。输入初始位置和速度，就能精准计算未来轨迹。

但在微观分子世界里，电子小到必须遵循量子力学规律。量子力学之父之一狄拉克 (Paul Dirac) 曾在 1929 年感叹：
> *"解释绝大部分物理学和全部化学所必需的底层物理定律已完全为人所知，唯一的困难是这些定律对应的数学方程太过复杂，以至于根本无法求解。"*

在过去近百年里，化学家们发明了各种经验公式、近似算法（如密度泛函理论 DFT、耦合簇理论 CCSD(T) 等），但往往在“计算速度快”和“化学精度准”之间左右为难。

2020 年，DeepMind 团队发表了 **FermiNet (Fermi Neural Network)**。该方法用神经网络参数化多电子试探波函数，并结合变分量子蒙特卡洛进行优化。
它以电子—核几何和库仑相互作用为输入先验；不同实现的精度取决于 Hamiltonian、网络、采样、优化和误差统计。本仓库是研究型 PyTorch 实现，不自动继承论文结果。

本仓库提供模块化代码，用于复现实验链路并探索这一类架构；它不等同于官方实现或论文级复现。

---

## 2. ⚛️ 第一章：量子力学与量子化学极简极速前置

### 2.1 什么是薛定谔方程？微观世界的“牛顿第二定律”

对于一个包含 $M$ 个原子核、$N$ 个电子的分子体系（在玻恩-奥本海默近似下，原子核质量远大于电子，假定核静止），电子运动由**定态非相对论薛定谔方程 (Schrödinger Equation)** 支配：

$$\hat{H} \Psi = E \Psi$$

别被希腊字母吓到了，我们用程序员的面向对象思维来理解它：
- $\Psi$（Psi，波函数）：可以看作**存储系统状态的黑盒对象**，输入所有电子的空间坐标 $r$，输出一个物理几率振幅标量。
- $\hat{H}$（哈密顿算符，Hamiltonian）：可以看作一个**处理函数/操作符**，负责计算系统包含的总能量操作（动能 + 势能）。
- $E$（能量本征值）：是一个具体的数值（标量），代表体系在特定状态下的总能量。

这个方程本质上就是一个**高阶微分方程的特征值问题**：寻找一个波函数 $\Psi$，让算符 $\hat{H}$ 作用在它身上后，结果刚好等于一个常数 $E$ 乘以它自己！
化学家最关心的往往是**基态 (Ground State)**，也就是能量最低的稳定态 $E_0$。

### 2.2 为什么电子多了解不出解析解？可怕的“指数墙”

- **氢原子 (1 个质子 + 1 个电子)**：初中/高中物理学过，有完美漂亮的解析解（球谐函数、轨道）。
- **氦原子 (2 个电子) 及以上**：**彻底解不出解析解！**

为什么？因为电子带负电，电子与电子之间存在库仑相互排斥项 $\frac{1}{\|r_i - r_j\|}$！
这种多体相互作用让变量彻底耦合在了一起，无法做数学上的“变量分离”。

在数值计算上，如果有 $N$ 个电子，每个电子有 3 维笛卡尔坐标 $(x, y, z)$，整个波函数定义在 **$3N$ 维空间**中：
- 比如只有 10 个电子的简单分子，输入空间就是 $3 \times 10 = 30$ 维！
- 如果像传统网格离散化方法那样，每个维度只采样 10 个网格点，总共就需要 $10^{30}$ 个采样点！
- 全球所有超级计算机加起来也存不下一个分子的网格——这就是臭名昭著的**维度灾难 (Curse of Dimensionality)**。

### 2.3 什么是电子波函数 $\Psi(r)$ 与玻恩几率解释 $|\Psi|^2$？

在宏观世界，我们可以指着屏幕说“小球就在坐标 $(x, y, z)$ 处”。
但在量子力学中，海森堡不确定性原理告诉我们：**你无法同时精确获知电子的位置与动量。**

物理学家玻恩 (Max Born) 提出了伟大的几率解释：
- 波函数 $\Psi(r_1, r_2, \dots, r_N)$ 本身是一个空间幅值（甚至可以带正负号）；
- 它的**模平方 $|\Psi(r_1, r_2, \dots, r_N)|^2$** 代表所有电子同时出现在这些位置的**联合概率密度**！

$$\text{电子出现在空间微元 } dr \text{ 的概率} = \frac{|\Psi(r)|^2}{\int |\Psi(r')|^2 dr'} dr$$

波函数模方越大的地方，电子“喜欢”待在那里的概率就越高；波函数为 0 的地方（称为“节点面”，Nodal Surface），电子绝对不会出现！

### 2.4 费米子置换反对称性与泡利不相容原理：“撞衫即毁灭”

电子是一种**全同费米子 (Fermions)**。微观世界没有编号，所有的电子长得一模一样。
量子力学给费米子规定了一条铁律：**波函数必须关于同自旋电子置换具有严格的反对称性 (Antisymmetry)**。

换句话说，如果你交换任意两个同自旋电子 $i$ 和 $j$ 的坐标，波函数的绝对值完全不变，但**符号必须反号**：

$$\Psi(\dots, r_i, \dots, r_j, \dots) = - \Psi(\dots, r_j, \dots, r_i, \dots)$$

这个看似简单的负号，引出了化学中最重要的基石——**泡利不相容原理 (Pauli Exclusion Principle)**：
如果两个同自旋电子试图挤在完全相同的位置，即令 $r_i = r_j$：
$$\Psi(\dots, r_i, \dots, r_i, \dots) = - \Psi(\dots, r_i, \dots, r_i, \dots) \implies \Psi = -\Psi \implies \Psi = 0!$$
波函数直接等于 0，意味着两个同自旋电子处于相同位置的概率为 $|\Psi|^2 = 0$——它们绝对不可能占据完全相同的空间位置！这也是为什么物质不会坍缩、原子具有丰富化学性质的终极物理原因。

<p align="center">
  <img src="assets/fermion_antisymmetry_nodal_surface.png" alt="Pauli Exclusion Principle and Nodal Surface" width="100%">
  <br>
  <em>图 2-1 | 泡利不相容原理与费米子波函数反对称性学术图解。(a) 同自旋电子空间坐标置换引发波函数正负相位反转 $\Psi(\mathbf{r}_1, \mathbf{r}_2) = -\Psi(\mathbf{r}_2, \mathbf{r}_1)$；(b) 两个同自旋电子相互靠近时中间自发形成的节点面（Nodal Surface, $\Psi = 0$），严格阻止电子重合。</em>
</p>

### 2.5 传统化学方法是怎么做的？Hartree-Fock 与 Slater 行列式的局限

数学上，什么样的结构天然满足“交换两行/两列就变号，有两行相同就等于 0”？
答案是：**线性代数中的矩阵行列式 (Determinant)！**

20 世纪物理学家约翰·斯莱特 (John C. Slater) 提出了著名的 **Slater 行列式**：
构造一个 $N \times N$ 矩阵，矩阵第 $i$ 行第 $j$ 列填入“第 $i$ 个电子在第 $j$ 个固定单电子轨道 $\phi_j(r_i)$ 上的值”：

$$D = \begin{bmatrix}
\phi_1(r_1) & \phi_2(r_1) & \dots & \phi_N(r_1) \\
\phi_1(r_2) & \phi_2(r_2) & \dots & \phi_N(r_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(r_N) & \phi_2(r_N) & \dots & \phi_N(r_N)
\end{bmatrix}, \quad \Psi = \det(D)$$

根据行列式的性质：
1. 交换两行（相当于交换两个电子的位置），行列式自动变号 $\det \to -\det$；
2. 如果两个电子坐标相同 $r_1 = r_2$，两行完全相同，行列式恒等于 0。

**传统量子化学的致命短板：**
在经典的 Hartree-Fock (HF) 方法中，轨道 $\phi_j(r_i)$ **仅仅依赖单电子坐标 $r_i$**！
这相当于假定每个电子只感受到其他所有电子的“平均电场”（平均场近似，Mean-field approximation），**完全忽略了两个电子瞬时相互躲避的“动态关联 (Electron Correlation)”**。
为了弥补这个缺陷，传统化学需要把成千上万、甚至数百万个不同的 Slater 行列式线性组合叠加起来（如多组态 CISD, CASSCF 等），计算量呈现指数级爆炸！

### 2.6 FermiNet 深度学习 Backflow 机制做了什么突破？

既然传统轨道 $\phi_j(r_i)$ 只看单电子太简陋，能不能让轨道**同时感知到周围其他所有电子的位置**？
这就是物理学中经典的 **Backflow（回流）思想**：
$$\phi_j(r_i; \{r_{\setminus i}\})$$
每个轨道既是电子 $i$ 的函数，又动态依赖所有其他电子 $\{r_{\setminus i}\}$ 的瞬时排布！

但人工设计复杂的 Backflow 函数极其困难。DeepMind 提出的 **FermiNet** 做出了革命性突破：
**用多层前馈神经网络直接充当 Backflow 映射！**
- 神经网络输入所有电子和原子核的相对空间几何；
- 经过自旋分流的图式交互更新，让电子之间充分“对话”与“避让”；
- 输出动态的广义轨道，再送入 Slater 行列式！

这样可以用较少的行列式表达复杂的多电子节点结构；是否达到化学精度必须由统一基准、参考体系和多种子误差统计验证，不能由架构本身保证。

---

## 3. 🧠 第二章：FermiNet 神经网络核心架构超清晰拆解

<p align="center">
  <img src="assets/ferminet_nature_architecture.png" alt="FermiNet 3D Scientific Architecture Illustration" width="100%">
</p>

### 3.1 整体架构全景数据流 (学术期刊规范架构图)

<p align="center">
  <img src="assets/ferminet_academic_dataflow.png" alt="FermiNet Academic Dataflow Architecture" width="100%">
</p>

下面是对应的文本级数据流水线与张量传递概览，请在阅读后续章节时随时对照：

```text
               ┌─────────────────────────────────────────────────────────────┐
               │                电子三维空间坐标 r: (B, N, 3)                │
               │               原子核坐标与电荷 R, Z: (M, 3), (M,)           │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
                              【特征提取层 features.py】
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               │ 单电子流特征 h1: (B, N, 2M)   │ 双电子流特征 h2: (B, N, N, 1)│
               │   [||r_i - R_I||, Z_I]        │   [||r_i - r_j||]             │
               └──────────────────────┬───────┴──────────────┬───────────────┘
                                      │                      │
                       【自旋分流交互层 FermiInteractionLayer layers.py】
                       (重复 L 层，电子之间自旋分离均值池化交互)
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
                 【指数包络 NuclearEnvelope】          【指数包络 NuclearEnvelope】
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
                                    乘积行列式求和 (LogSumExp)
                                               ▼
                              ┌──────────────────────────────────┐
                              │  对数波函数值 log|Psi| : (B,)    │
                              │  波函数符号 sign(Psi)  : (B,)    │
                              │  标量波函数 Psi        : (B,)    │
                              └──────────────────────────────────┘
```

### 3.2 几何特征提取 (`features.py`)：相对位移、距离与核电荷注入

在 `ElectronFeatureEmbedding` 中，为什么我们**绝不能直接把电子的绝对坐标 $(x, y, z)$ 喂进全连接层**？

#### 物理不变性原理 (Physical Invariance)
如果我们将整个分子在空间中整体平移，分子的内部物理性质（能量、分子键长、相互作用）不应改变。**旋转同样应保持物理等价**——哈密顿量本身是旋转不变的，因此基态波函数也必须满足这一点。

做法有两种：**只用旋转不变量**，或者引入专门的 SO(3)/E(3) 等变结构。本项目的公共默认保留相对笛卡尔位移，以表达一般空间态；它保持平移不变，但不保证旋转不变。需要纯距离历史对照时，显式传入 `rotation_invariant=True`，此时特征中**不含任何笛卡尔位移向量**，整体旋转下前向计算逐位相同。

对本仓库的体系（H2、LiH、H100 链）全部是**线性分子**，到各原子核的距离已经完整确定了电子在分子标架下的位置（轴向坐标 + 垂直距离），所以这个特征集**不损失表达力**——H2 收敛对照：不变模式 $-1.17397$ Ha vs 笛卡尔模式 $-1.17416$ Ha，差异落在统计误差之内。

如果直接把绝对坐标丢给 MLP，神经网络必须死记硬背每个坐标位置，泛化能力极差。

因此，FermiNet 提取两类**相对几何特征**。显式纯距离（旋转不变）模式下：
1. **单电子-核几何特征 ($h_i^0$)**：
   - 相对欧氏距离：$d_{iI} = \sqrt{\operatorname{clamp}(\|r_i - R_I\|^2,\ \min=\epsilon)}$（1 维，旋转不变）
   - 原子核电荷：$Z_I$（1 维标度，标识当前原子核的元素种类）
   - 每个原子核提供 **2 维**特征，体系总共有 $M$ 个核，总维度为 $2 \times M$。
2. **双电子-电子几何特征 ($h_{ij}^0$)**：
   - 相对欧氏距离：$d_{ij} = \sqrt{\operatorname{clamp}(\|r_i - r_j\|^2,\ \min=\epsilon)}$（1 维）
   - 即 **1 维**特征。

> **默认模式（`rotation_invariant=False`）**：额外拼接笛卡尔位移向量——单电子流每核 5 维 $[r_{iI}, d_{iI}, Z_I]$（不注入电荷时 4 维），双电子流 4 维 $[r_{ij}, d_{ij}]$。方向信息更丰富，但**不具备旋转不变性**。它不自动保证特定角动量、宇称或纯总自旋态，也不构成基态收敛证明。

> **为什么用 `clamp` 而不是 $\sqrt{r^2 + \epsilon}$？**
> 两者都避免 $r = 0$ 处梯度为 `NaN`，但代价不同。加性 $\epsilon$ 在**每一个**距离上都引入相对偏差 $\epsilon / (2r^2)$，处处不为零；而 `clamp(r², min=ε)` 在**所有** $r^2 \ge \epsilon$ 处是恒等变换，因此距离与未软化时**逐位相同**，偏置精确为零。只有 $r^2 < \epsilon$ 时距离才被限制为 $\sqrt{\epsilon}$，以保证重合点的数值稳定。默认 $\epsilon = 10^{-24}$，软化半径仅 $10^{-12}$ Bohr。
>
> 这直接关系到 Kato 尖点条件：旧的加性 $\epsilon = 10^{-8}$（软化半径 $10^{-4}$ Bohr）使核处径向对数斜率趋于 0 而非 $-Z$，破坏了尖点；改用 clamp 后，$r \gg 10^{-12}$ 处的斜率恢复为精确值。
>
> **注意**：软化核只保证**势能**在非重合点精确。波函数 ansatz 本身并不强制 Kato 条件——包络只有在学到的衰减率 $\alpha$ 恰等于 $Z$ 时才给出正确的 $-Z$ 斜率。

> 💡 **核心代码亮点**：
> 全部采用 PyTorch 高阶广播机制（Broadcasting），如 `r.unsqueeze(2) - nuclei.unsqueeze(0).unsqueeze(0)`，零 Python 显式循环，原生 GPU 全并行向量化！

> **核序与同核置换（一个如实声明的限制）**：上表的单核特征是按**原子核顺序**拼接后送入全连接层的。若交换两个同种原子核（例如 $H_2$ 的两个质子），特征块的位置会互换，而全连接层对核块并不对称，因此网络输出随之改变——**完整波函数不具备核置换不变性**。
>
> 一个自然的想法是「按距离排序后再拼接」：把同核块按电子到核的距离排好序，顺序就只取决于几何而与标签无关。**这个方案是错误的，已被否决**。原因是排序在电子穿越两同核的**垂直平分面**时会翻转核块顺序，而两个块在该面上并不相等（位移向量 $r_{iI}$ 方向不同），于是波函数在该余维 1 曲面上出现**不连续跳变**（实测 $\ln|\Psi|$ 跳变约 0.044）。$\nabla^2 \ln|\Psi|$ 在不连续处失去意义，局域动能被污染——实测训练能量从 $-1.174$ Ha 退化到 $-1.162$ Ha，并一度跌破变分下界（低于精确值，物理上不可能）。
>
> 要做到既**不变**又**连续**，必须放弃「有序拼接」而改为对核做**对称聚合**（例如把每个核的块过同一个网络后求和）。那是一次架构改写，不在本阶段范围内。因此当前只保证**包络部分**不变（见 3.4 节）。回归测试见 `test_physics_invariants.py` 的 `test_wavefunction_is_continuous_across_nuclear_bisector`。

### 3.3 自旋分流交互层 (`layers.py`)：单/双电子流与自旋分离池化

分子中有两种自旋状态的电子：自旋向上 ($\uparrow$) 和自旋向下 ($\downarrow$)。
量子力学规定：**打乱两个同自旋电子的输入顺序，网络的输出特征也应该仅仅对应打乱顺序，特征内容不能发生不可控突变（置换等变性，Permutation Equivariance）**。

在 `FermiInteractionLayer` 中，每一层电子特征更新时：
对于任意第 $i$ 个电子，它不仅看自己的特征 $h_i$，还要聚合周围所有电子的信息：
1. **所有自旋向上电子的平均特征**：$\bar{h}^\uparrow = \frac{1}{n_\uparrow} \sum_{j \in \uparrow} h_j$
2. **所有自旋向下电子的平均特征**：$\bar{h}^\downarrow = \frac{1}{n_\downarrow} \sum_{j \in \downarrow} h_j$
3. **电子 $i$ 与所有自旋向上电子的双电子交互平均**：$\bar{h}_i^{\uparrow \uparrow} = \frac{1}{n_\uparrow} \sum_{j \in \uparrow} h_{ij}$
4. **电子 $i$ 与所有自旋向下电子的双电子交互平均**：$\bar{h}_i^{\uparrow \downarrow} = \frac{1}{n_\downarrow} \sum_{j \in \downarrow} h_{ij}$

把这 5 项特征拼在一起，送入带 $\tanh$ 激活函数和残差连接 (Residual Skip Connection) 的全连接层更新：
$$a_i = \left[ h_i, \; \bar{h}^\uparrow, \; \bar{h}^\downarrow, \; \bar{h}_i^{\uparrow \uparrow}, \; \bar{h}_i^{\uparrow \downarrow} \right]$$
$$h_i^{(l+1)} = \tanh\left( W^{(l)} a_i + b^{(l)} \right) + \text{Residual}(h_i^{(l)})$$

**为什么平均池化 (Mean Pooling) 能保证置换等变？**
因为求和/求均值是无序的（加法满足交换律）。打乱前 $n_\uparrow$ 个电子的次序，均值 $\bar{h}^\uparrow$ 纹丝不动！这使得神经网络天然服从量子物理的内在全同对称性。

> **末层的双电子流更新网络已被移除**：每层都会同时更新单电子流 $h_i$ 与双电子流 $h_{ij}$，但只有**单电子流**最终流入轨道构造（Backward 到 `linear_orb_up/down`）。因此**最后一层**的双电子流输出不被任何下游消费——它的线性层会变成永远收不到梯度的空转参数。本项目在构造末层时直接跳过该更新网络（`produce_two_electron_output=False`），消除这部分浪费。前面的层不受影响，双电子流照常逐层传递。

### 3.4 原子核指数衰减包络 (`envelope.py`)：物理渐近条件与各向异性度规

神经网络虽然拟合能力强，但如果放任它自由输出，当某个电子飞到距离原子核极其遥远的地方时（$r \to \infty$），未加约束的神经网络可能会输出荒谬的非零值甚至发散值。

但在物理真实的束缚态中，电子越远离带正电的原子核，被束缚的概率必须**指数级衰减至 0**！
为了把这个物理先验硬编码进模型，`NuclearEnvelope` 构建了指数包络函数：

#### 1. 各向同性包络 (Isotropic)
$$\text{Envelope}(r_i) = \sum_{m=1}^M \pi_{m} \exp\left(-\alpha_{m} \, d_{im}\right)$$
其中 $d_{im} = \sqrt{\operatorname{clamp}(\|r_i - R_m\|^2,\ \min=\epsilon)}$ 是上文定义的软化距离；衰减速率 $\alpha > 0$ 通过 `F.softplus` 映射保证严格为正，确保衰减绝不发散。

> **同核置换不变性**：$\pi_m$ 与 $\alpha_m$ 按**电荷组**共享——同种原子核共用同一组参数。若只共享 $\pi$ 或只共享 $\alpha$，把置换的哑指标 $m$ 重标号即可看出求和式仍会改变；两者必须同时共享，包络才在同核置换下严格不变。

#### 2. 各向异性包络 (Anisotropic，阶段 2 增强特性)
真实分子中的共价键往往沿着特定化学键轴向延伸，电子云并不是简单的球对称（例如 $p$ 轨道像哑铃形）。
各向异性模式为每个原子核学习一个 3 维主轴缩放度规 $\sigma = (\sigma_x, \sigma_y, \sigma_z)$：
$$\text{Envelope}(r_i) = \sum_{m=1}^M \pi_{m} \exp\left(-\sqrt{\operatorname{clamp}\left(\sum_{d \in \{x, y, z\}} \sigma_{m, d}^2 (r_{i, d} - R_{m, d})^2,\ \min=\epsilon\right)}\right)$$
这大幅增强了网络表征长条形、哑铃形等复杂分子轨道的物理精度！

### 3.5 反对称波函数构建 (`model.py`)：Backflow 轨道矩阵构造

经过 $L$ 层等变交互后，我们得到了单电子高级表征 $h_i^L$。
在 `model.py` 中：
1. **Backflow 线性映射**：将自旋向上的前 $n_\uparrow$ 个电子的特征，投影为一个方阵尺寸：
   $$\text{Shape: } (B, K, n_\uparrow, n_\uparrow)$$
   这里 $K$ 是行列式分支数（`num_determinants`），矩阵的第 $i$ 行代表第 $i$ 个电子，第 $j$ 列代表第 $j$ 个分子轨道。
2. **包络乘积**：将 Backflow 输出与指数包络逐元素相乘，得到最终的分子轨道方阵：
   $$D_k^\uparrow(i, j) = \text{Backflow}_{k, i, j}(r) \times \text{Envelope}_{k, i, j}(r_i)$$
   同理，构建自旋向下的分子轨道方阵 $D_k^\downarrow$，Shape 为 $(B, K, n_\downarrow, n_\downarrow)$。
3. **块对角行列式乘积**：
   $$\det(D_k) = \det(D_k^\uparrow) \times \det(D_k^\downarrow)$$
   根据分块对角矩阵性质，这样构成的整体波函数：
   - 交换两个自旋向上电子，$\det(D^\uparrow)$ 严格反号，$\det(D^\downarrow)$ 不变，整体反号；
   - 交换两个自旋向下电子，$\det(D^\uparrow)$ 不变，$\det(D^\downarrow)$ 严格反号，整体反号；
行列式结构在精确算术下保证同自旋交换反对称；实际 float32/float64 计算仍受舍入误差和条件数影响，应以数值容差验证。

### 3.6 数值稳定性基石：`torch.linalg.slogdet` 与 LogSumExp 技巧

这是深度学习与量子化学结合时最容易踩的“深坑”！

假设一个分子有 50 个电子，每个电子轨道值假设平均约为 0.1。
一个 $50 \times 50$ 矩阵的行列式大小粗略量级为：
$$\det(D) \sim (0.1)^{50} = 10^{-50}!$$
在计算机中，单精度浮点数 `float32` 能表示的最小正数约为 $1.17 \times 10^{-38}$。
如果直接调用 `torch.det(D)`，**结果会直接发生下溢 (Underflow)，变成纯 0.0！**
一旦变成 0，后续求对数全部变成 $-\infty$，反向传播梯度瞬间产生巨大的 `NaN`，模型彻底报废！

#### 解决方案：全面转入对数空间
FermiNet 全程在对数空间中计算：
1. 使用 PyTorch 的 `torch.linalg.slogdet(D)`，它把行列式分解为两部分：
   $$\det(D) = \text{sign} \times \exp(\text{log\_abs\_det})$$
   即使行列式小到 $10^{-300}$，它的对数也只是普通的数值 $-690.7$，在浮点数安全范围内！
2. 多个行列式的加权和使用 **LogSumExp 技巧**：
   $$\Psi = \sum_{k=1}^K w_k \det(D_k)$$
   提取出最大的对数值 $M = \max_k (\ln|w_k| + \ln|\det(D_k)|)$，将指数项除以 $e^M$，使得最大的项始终为 1.0，彻底根除溢出与下溢风险！

---

## 4. 🎲 第三章：变分量子蒙特卡洛 (VMC) 极简教程

有了能够计算任意电子构型波函数 $\Psi(r)$ 的 FermiNet，我们该如何训练它？

### 4.1 为什么高维积分必须用 MCMC 随机游走采样？

根据量子力学，分子的能量期望值是哈密顿算符在全空间的积分：
$$\langle E \rangle = \frac{\int \Psi^*(r) \hat{H} \Psi(r) dr}{\int |\Psi(r)|^2 dr}$$

我们把它重写为概率期望的形式：
$$\langle E \rangle = \int \left[ \frac{\hat{H} \Psi(r)}{\Psi(r)} \right] \frac{|\Psi(r)|^2}{\int |\Psi(r')|^2 dr'} dr = \mathbb{E}_{r \sim |\Psi|^2} [E_L(r)]$$

这说明：**能量期望值，就是所有服从玻恩概率密度 $p(r) \propto |\Psi(r)|^2$ 采样的“局域能量 $E_L(r)$”的统计均值！**

但是，前面说过输入空间是 $3N$ 维，根本无法画网格积分。
解决超高维积分唯一的物理利器就是**马尔可夫链蒙特卡洛 (MCMC)**！
我们在空间中释放成百上千个采样粒子（称为 **Walkers**），让它们在空间中随机游走。经过足够多次的游走步进，这批 Walkers 的空间分布就会精准契合概率密度 $|\Psi(r)|^2$！

### 4.2 Metropolis-Hastings 接受/拒绝机制

在 `MCMCSampler` 中，每个 Walker 怎么知道该往哪走？
1. **提议新坐标**：当前在 $r_{\text{old}}$，加上一个微小的高斯随机扰动 $\Delta r \sim \mathcal{N}(0, \sigma^2)$，得到新位置：
   $$r_{\text{new}} = r_{\text{old}} + \Delta r$$
2. **计算概率比**：看新位置的波函数模方是不是比老位置更大：
   $$A = \min\left(1, \; \frac{|\Psi(r_{\text{new}})|^2}{|\Psi(r_{\text{old}})|^2}\right) = \min\left(1, \; \exp\left(2 \left( \ln|\Psi(r_{\text{new}})| - \ln|\Psi(r_{\text{old}})| \right)\right)\right)$$
3. **掷骰子决定**：
   - 如果新位置波函数更大 ($A \ge 1$)：说明新位置电子更喜欢，**100% 接受移动**！
   - 如果新位置波函数变小 ($A < 1$)：生成一个 $[0, 1]$ 均匀随机数 $u$。若 $u < A$，依然接受；否则**拒绝移动，停在原地**。

通过这一机制，采样点自然会汇聚在波函数幅值高的化学键和原子核周围。

### 4.3 什么是局域能量 $E_L(r)$？动能与势能的精妙平衡

局域能量 (Local Energy) 定义为：
$$E_L(r) \equiv \frac{\hat{H} \Psi(r)}{\Psi(r)}$$

在原子单位制下（$\hbar = m_e = e = 1$），总能量算符 $\hat{H} = \hat{T} + \hat{V}$：
1. **动能算符 (Kinetic Energy $\hat{T}$)**：
   $$\hat{T} = -\frac{1}{2} \sum_{i=1}^N \nabla_{r_i}^2$$
   作用在对数波函数上，局域动能写为精妙的表达式：
   $$T_L(r) = -\frac{1}{2} \left[ \nabla_r^2 \ln|\Psi(r)| + \|\nabla_r \ln|\Psi(r)\|^2 \right]$$
   这里 $\nabla^2$ 是二阶导数（拉普拉斯算子），$\|\nabla\|^2$ 是一阶梯度的模平方。
2. **势能项 (Potential Energy $V(r)$)**：由库仑相互作用构成（完全不需要求导，纯几何距离算术）：
   $$V(r) = \underbrace{-\sum_{i=1}^N \sum_{I=1}^M \frac{Z_I}{\|r_i - R_I\|}}_{\text{电子-核吸引力 (负数)}} + \underbrace{\sum_{1 \le i < j \le N} \frac{1}{\|r_i - r_j\|}}_{\text{电子-电子排斥力 (正数)}} + \underbrace{\sum_{1 \le I < J \le M} \frac{Z_I Z_J}{\|R_I - R_J\|}}_{\text{核-核排斥力常数 (正数)}}$$

总局域能量就是 $E_L(r) = T_L(r) + V(r)$。

### 4.4 变分原理：为什么能量越低越逼近真理？

量子力学中的**变分原理 (Variational Principle)** 适用于属于目标 Hamiltonian 定义域、平方可积且具有有限能量的试探波函数。对满足这些条件并处于同一粒子数/自旋扇区的 $\Psi_\theta$，能量期望值满足：

$$\langle E \rangle_{\theta} = \frac{\langle \Psi_\theta | \hat{H} | \Psi_\theta \rangle}{\langle \Psi_\theta | \Psi_\theta \rangle} \ge E_0$$

本项目在核附近使用平滑距离以提高数值稳定性，因此实际计算的 Hamiltonian 与未软化的点核 Coulomb Hamiltonian 存在近核差异；参考能量比较必须遵守登记体系和相同模型假设。

当且仅当 $\Psi_\theta$ 与真实基态波函数完全一致时，等号才成立！
**这意味着什么？**
这意味着我们可以把**能量期望值 $\langle E \rangle$ 直接当做神经网络的优化目标（Loss 函数）！**
降低能量是有用的优化目标，但有限步数、采样误差、局部极小值和表达能力都会影响结果；能量下降不自动保证已收敛到真实基态。

### 4.5 核心物理公式纠偏：有限批次协方差代理损失与求导真相

> ⚠️ **新手极易犯的严重物理错误**：
> 既然优化目标是 $\langle E \rangle = \mathbb{E}[E_L(r)]$，那能不能写成 `loss = torch.mean(e_local)` 然后直接调用 `loss.backward()`？
> 
> **不建议直接这样做。**

为什么？
因为局域能量 $E_L(r)$ 本身就包含了波函数对电子坐标 $r$ 的二阶导数 $\nabla_r^2 \Psi$。如果你直接对 $E_L(r)$ 关于网络参数 $\theta$ 执行反向传播求导：
$$\frac{\partial}{\partial \theta} \left( \nabla_r^2 \Psi_\theta \right)$$
PyTorch 将被迫构建极度昂贵的三阶导数计算图（Third-order gradients），显存直接爆仓！
这种路径会把局域能量中的坐标二阶导数继续连接到参数导数，产生昂贵的高阶导数，并且不能直接替代考虑采样分布导数的 VMC 得分函数估计器；“必然有偏且无法收敛”不是对所有实现和采样方案都成立的普遍定理。

#### 正确的变分梯度定理 (Variational Gradient Theorem)
对能量期望值求关于参数 $\theta$ 的严格数学导数：
$$\nabla_\theta \langle E \rangle = 2 \, \mathbb{E}_{r \sim |\Psi|^2} \left[ \left( E_L(r) - \langle E_L \rangle \right) \nabla_\theta \ln|\Psi_\theta(r)| \right]$$

观察这个优雅的物理公式：
1. **$E_L(r) - \langle E_L \rangle$ 这一项不需要任何参数梯度！** 它只是一个标量系数；
2. **需要求参数梯度的只有 $\ln|\Psi_\theta(r)|$ 这一项！**

在 `loss.py` 中，默认使用无截断并带有限样本修正的协方差代理损失。若一个固定时刻的批次可视为 $B$ 个独立、平衡态 Walker，则乘上 $B/(B-1)$ 可消除同批次均值中心化的有限样本因子；实际 MCMC 链的时间相关和 burn-in 偏差仍需单独处理：
```python
# 1. 彻底分离局域能量的梯度图 (stop_gradient)
delta_e = (B / (B - 1)) * (e_local - torch.mean(e_local)).detach()

# 2. 构造代理损失函数
loss = 2.0 * torch.mean(delta_e * log_abs_psi)

# 3. 在 IID Walker 与平衡分布条件下得到有限批次协方差修正
loss.backward()
```
这省去了三阶导的高昂开销。有限批次无偏性只是在 IID Walker、平衡分布和未截断局域能量条件下的统计结论；实际 MCMC 训练仍可能存在链相关和有限 burn-in 偏差。

### 4.6 局域能量平滑截断 (Umrigar Tanh Clipping)：防止优化器“炸飞”

在 MCMC 采样中，有两个区域是非常极端的：
1. **波函数节点面附近**：$\Psi(r) \to 0$。分母极小，导致 $E_L(r) = \frac{\hat{H}\Psi}{\Psi} \to \pm \infty$！
2. **电子撞核奇异点**：当电子极其贴近原子核时，库仑势能 $-\frac{Z}{\|r - R\|} \to -\infty$。

虽然在数学上这些奇异点出现的概率极低，但一旦某个 Walker 碰巧踩到奇异点，产生一个比如 $E_L = -10^6$ 的超级离群值，乘在 $\nabla_\theta \ln|\Psi|$ 上，Adam 优化器的梯度就会瞬间被破坏，导致模型参数发散崩溃！

为了解决这个问题，FermiNet 引入了物理学大师 C. J. Umrigar 提出的 **基于中位数绝对偏差 (MAD) 的双曲正切截断**：
1. 计算当前批次能量的中位数 $E_{\text{median}}$ 和中位数绝对偏差 $\text{MAD} = \text{median}(|E_L - E_{\text{median}}|)$；
2. 设定截断半宽 $\Delta = 5 \times \text{MAD}$；
3. 用平滑的双曲正切函数处理离群点：
   $$E_{\text{clipped}} = E_{\text{median}} + \Delta \cdot \tanh\left( \frac{E_L - E_{\text{median}}}{\Delta} \right)$$

这种做法能限制离群能量并提高稳定性，但它改变了梯度目标，因此现在必须通过 `--clip_method tanh` 或 `hard` 显式启用，不能称为严格无偏估计。

---

## 5. 🗺️ 第四章：阶段 0、1、2 核心演进与模块全景导览

本项目代码经过多轮迭代，形成了便于实验和复现的研究型模块架构：

### 5.1 源码文件与核心职责对照表

| 源码文件 | 核心职责 | 主要类 / 函数 | 输入与输出 |
| :--- | :--- | :--- | :--- |
| [`features.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/features.py) | 相对几何特征嵌入 | `ElectronFeatureEmbedding` | 输入电子坐标 `(B, N, 3)` 与核坐标/电荷；默认输出包含方向信息的笛卡尔特征 `(B, N, 5M)` / `(B, N, N, 4)`；`rotation_invariant=True` 时为严格旋转不变的 `(B, N, 2M)` / `(B, N, N, 1)` |
| [`layers.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/layers.py) | 自旋分流前馈交互 | `FermiInteractionLayer` | 输入单/双电子流；按自旋分别平均池化交互，输出更新后的表征 |
| [`envelope.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/envelope.py) | 束缚态指数衰减包络 | `NuclearEnvelope` | 保证波函数在远离核时指数衰减；支持各向同性标量与各向异性三维主轴度规 |
| [`model.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/model.py) | FermiNet 主干网络 | `FermiNet` | 组装 Backflow 轨道，调用 `slogdet` 分解计算多行列式对数波函数 `log|Psi|` 与符号 `sign` |
| [`hamiltonian.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/hamiltonian.py) | 分子哈密顿量与动能 | `MolecularSystem`, `compute_local_energy` | 全向量化计算势能 $V(r)$ 与动能 $T_L(r)$，集成三大拉普拉斯求解器 |
| [`mcmc.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/mcmc.py) | 批量 GPU MCMC 采样器 | `MCMCSampler` | 并行 Metropolis-Hastings 拒绝抽样，自适应调节步长趋向默认目标；实际接受率依赖体系和提议分布 |
| [`loss.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/loss.py) | 协方差代理损失 | `VMCLoss`, `vmc_surrogate_loss`, `clip_local_energy` | 默认有限批次修正；Tanh/Hard 截断为显式稳健有偏模式 |
| [`mcmc_stats.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/mcmc_stats.py) | MCMC 统计 | `estimate_mcmc_statistics` | 由 walker 时间序列估计自相关时间、ESS、split-R-hat 与链相关 SEM |
| [`train_vmc.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/train_vmc.py) | 端到端训练脚本 | `train_vmc`, `main` | 串联 MCMC 采样、局域能量求解与 Adam 参数优化的完整训练闭环 |
| [`example_h2.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/example_h2.py) | 极简推断与能量演示 | `run_h2_demo` | 5 分钟上手脚本，单步前向推断 $H_2$ 波函数与局域能量拆解 |
| [`benchmark_compile.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/benchmark_compile.py) | JIT 图编译基准评测 | `benchmark_compilation` | 测试 PyTorch 2.0 `torch.compile` 对 FermiNet 的算子融合；具体收益依赖硬件和配置 |
| [`benchmark_100_atoms.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/benchmark_100_atoms.py) | 100 原子超大体系基准 | `run_benchmark` | 评测 100 核 100 电子链在超大规模下的显存占用、吞吐与物理反对称性 |
| [`visualize.py`](file:///C:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/visualize.py) | 学术风格综合可视化库 | `plot_training_dashboard`, `plot_physical_results_dashboard` | 提供训练、采样和条件切片诊断，无 GUI 隔离后端 |

### 5.2 三大高性能动能拉普拉斯求解器对比

动能拉普拉斯算子 $\nabla_r^2 \ln|\Psi| = \sum_{i=1}^{3N} \frac{\partial^2}{\partial r_i^2} \ln|\Psi|$ 是量子神经网络中最耗时的计算瓶颈。本代码库提供了三种求解策略：

```text
               ┌─────────────────────────────────────────────────────────────────┐
               │              动能拉普拉斯算子 Tr(Hessian) 计算策略               │
               └────────────────┬────────────────┬────────────────┬──────────────┘
                                │                │                │
             ┌──────────────────┴──┐    ┌────────┴────────┐    ┌──┴──────────────────┐
             │ 经典逐分量精确模式   │    │ JVP 前向模式    │    │ Hutchinson 随机迹估计│
             │   ("exact")         │    │ ("exact_jvp")   │    │   ("hutchinson")    │
             ├─────────────────────┤    ├─────────────────┤    ├─────────────────────┤
             │ 时间复杂度: 随 3N 增长│    │ 批量计算 3N 方向 │    │ 探针数与 N 无关     │
             │ 反向传播次数: 3N 次 │    │ 采用 vmap 向量化│    │ 少量方向导数       │
             │ 数值精度最高        │    │ 依赖 PyTorch JVP│    │ 条件无偏，方差可变  │
             └─────────────────────┘    └─────────────────┘    └─────────────────────┘
```

1. **`exact` (经典高精度反向微分模式)**：
   - 原理：针对 $3N$ 个空间坐标分量，循环调用 $3N$ 次 `torch.autograd.grad`，逐一提取二阶对角元素。
   - 优点：在给定 dtype 和条件数下数值精度最高，适合小体系对照；仍受浮点误差影响。
   - 缺点：体系变大后（例如 100 个电子，300 维），需要执行 300 次反向传播，耗时随电子数线性增长。
2. **`exact_jvp` (JVP 前向模式向量化精确求解)**：
   - 原理：基于 `torch.func.jvp` (Jacobian-Vector Product) 与 `torch.func.vmap`，在批次和坐标维度上全部展开为硬件级并行。
   - 优点：全向量化，无显式 Python 循环。
3. **`hutchinson` (Hutchinson 随机迹估计器，成本-方差折中)**：
   - 原理：根据随机线性代数定理，对于任意对称矩阵 $H$ 和满足 $\mathbb{E}[v v^T] = I$ 的随机向量 $v$（如 Rademacher 分布，分量以 0.5 概率取 $+1$ 或 $-1$）：
     $$\text{Tr}(H) = \mathbb{E}_v \left[ v^T H v \right]$$
   - 绝妙之处：计算 $v^T H v = v^T \nabla_r (v^T \nabla_r \ln|\Psi|)$ **只需要 1 次方向导数反向传播！**
   - 成本与误差：默认少量随机方向可减少方向导数调用；迹估计在探针分布条件下无偏，但单次方差可能很大，整体成本仍随体系、网络和 batch 变化。是否加速必须在统一硬件、精度和误差目标下实测。
   - **⚠️ 复杂度澄清**：「探针数与 $N$ 无关」只说明**单步所需的反向传播次数**不随体系规模增长；**总计算成本仍随 $N$ 增长**（每次反向传播本身要过一遍整个网络），且探针循环是 Python 级的，探针数越多开销线性增加。因此这不是 $O(1)$ 算法。
   - **⚠️ 精度澄清**：单探针估计的重复间标准差可达信号本身的 1.5 倍以上（实测：固定构型下精确动能 $T = -1.0183$，单探针估计在 32 次重复间标准差约 $1.76$）。因此它**只适合作为训练期的方差缩减手段，绝不可用于对外报告能量**。本项目的复评路径已强制使用精确求解器。

### 5.3 PyTorch 2.0+ `torch.compile` JIT 图编译加速秘籍

FermiNet 的前向推理包含大量的张量索引、小算子切片与矩阵运算。在 Python 原生 Eager 模式下，每次运算都会触发 CPU-GPU 之间的调度开销。

通过 PyTorch 2.0 引入的 TorchDynamo 与 TorchInductor，FermiNet 可以一键编译为融合后的定制 GPU 内核：
```python
# 一键编译优化
compiled_model = model.compile_model(backend="inductor", mode="reduce-overhead")
```
在特定硬件和固定图配置下可能观察到：
- 算子融合减少了显存频繁读写；
- CUDA Graphs 捕获固定图执行，消除了 CPU 发射延迟；
- 算子融合和图捕获可能降低稳态调度开销；加速比、首次编译成本和数值差异必须按硬件、版本、batch 与 dtype 单独报告，不能预设固定倍数或“机器精度”。

---

## 6. 🚀 第五章：5 分钟上手极速实操 (含输出物理指标通俗解读)

### 6.1 演示 1：氢分子 $H_2$ 基础波函数推断与局域能量 (`example_h2.py`)

这是最适合新手的第 1 步：仅需单次前向传播，亲眼观察波函数与动能、势能的计算。

```bash
# 在项目根目录下直接运行：
python example_h2.py
# 或使用 uv 运行：
uv run python example_h2.py
```

**核心终端输出节选与解读**：
```text
======================================================================
           FermiNet (PyTorch) 氢分子 (H2) 基础模型演示
======================================================================
[1] 分子体系设定:
    - 原子核数: 2 (电荷 Z: [1.0, 1.0])
    - 电子配置: N_up = 1, N_down = 1 (总电子数 = 2)

[2] FermiNet 模型已构建:
    - 行列式分支数 K: 8
    - 可学习参数总量: 11464

[3] 前向计算波函数:
    Walker 1:
      log|Psi(r)| = -2.71630, sign = 1
      真实波函数 Psi(r) = 6.61189e-02

[4] 基于自动微分计算局部能量 E_L(r) = T_L(r) + V(r):
    Walker 1:
      局部动能   T_L(r) = 0.58432 Hartree
      局部势能     V(r) = -1.62145 Hartree
      总局部能量 E_L(r) = -1.03713 Hartree
```
- **解说**：`log|Psi(r)| = -2.71` 对应实际波函数振幅约为 $0.066$。局域动能为正（$+0.58$ Ha），吸引势能为负（$-1.62$ Ha），总能量约为 $-1.04$ Ha，初步逼近真实能量。

### 6.2 演示 2：氢分子 $H_2$ 端到端 VMC 变分训练 (`train_vmc.py`)

体验完整的变分量子蒙特卡洛优化！模型从随机初始化的参数开始，通过 Adam 自动优化，基态能量不断下降。

```bash
# 启动 50 轮训练，使用 1024 个采样 Walkers
python train_vmc.py --epochs 50 --walkers 1024 --lr 1e-3 --rotation_invariant
# 或使用 uv 运行：
uv run python train_vmc.py --epochs 50 --walkers 1024 --lr 1e-3 --rotation_invariant
```

**训练实时输出展示**：
```text
===========================================================================
         FermiNet 变分量子蒙特卡洛 (VMC) 训练优化启动
===========================================================================
[*] 预热阶段: 正在执行 100 步 MCMC 预热...
    预热完成 (耗时: 1.25s, 最终平均接受率: 51.2%, 步长: 0.5023)
---------------------------------------------------------------------------
Epoch  |  <E_raw> (Ha)  | <E_clip> (Ha)  |   Var(E)   | IID SEM*  | Acc (%)  |  Step
---------------------------------------------------------------------------
  1    |    -0.81235    |    -0.81050    |  0.45120   |  0.02099  |   50.8   | 0.5023 
 10    |    -1.05412    |    -1.05380    |  0.21040   |  0.01433  |   51.5   | 0.5041 
 30    |    -1.14205    |    -1.14180    |  0.09820   |  0.00979  |   49.8   | 0.4985 
 50    |    -1.16850    |    -1.16820    |  0.04510   |  0.00663  |   50.2   | 0.5012 
===========================================================================
[*] 示例输出（仅展示字段格式，不代表固定收敛结果）:
[*] 独立复评能量: -1.16850 Hartree (参考线只适用于登记的 H2 体系)
[*] 最终能量方差: 0.04510 | 相关 SEM: 0.00663
[*] ESS: 8200.0 | tau_int: 1.998 | R-hat: 1.0040
===========================================================================
```

这里的“最终”不再是最后一次参数更新前的批次均值：训练结束后，程序会冻结模型和采样步长，重新预热，再沿每条 walker 链采集时间序列。返回的 `final_energy` 与 `final_variance` 来自该复评；训练最后一批则以 `training_last_batch_energy` 保存。`IID SEM*` 只用于观察训练过程，正式报告应同时给出 `evaluation.energy_sem`、`evaluation.ess`、`evaluation.tau_int` 与 `evaluation.r_hat`。

推荐把 `R-hat` 接近 1、ESS 足够大视为“误差条可解释”的前提，而不是把一次短训练的能量下降直接视为收敛。复评默认使用精确 Laplacian；可用下列参数提高统计稳健性：

```bash
python train_vmc.py --epochs 50 --walkers 256 \
  --evaluation_steps 64 --evaluation_mcmc_steps 5 \
  --evaluation_burn_in 40 --evaluation_laplacian_method exact
```

### 6.3 演示 3：极性异核分子氢化锂 $LiH$ 变分优化

氢化锂 $LiH$ 是一个典型的极性共价键分子：锂核电荷 $Z=3$，氢核电荷 $Z=1$，总电子数 $N=4$。
这里验证了阶段 2 的核电荷注入与各向异性包络能力；Hutchinson 是否降低端到端训练时间，应结合具体硬件和误差目标单独测量：

```bash
python train_vmc.py --system lih --laplacian_method hutchinson --epochs 30 --walkers 512 --rotation_invariant
# 或使用 uv 运行：
uv run python train_vmc.py --system lih --laplacian_method hutchinson --epochs 30 --walkers 512 --rotation_invariant
```
只有核电荷、自旋电子数与核间距 `3.015 Bohr` 同时匹配登记记录时，才会显示 $\approx -8.070$ Hartree 参考线；短训练只能检查链路和趋势。

### 6.4 演示 4：100 原子超大规模体系基准测试 (`benchmark_100_atoms.py`)

体验 100 个氢原子（100 个质子 + 100 个电子，输入维度高达 300 维）的一维强关联晶格链基准：

```bash
# 建议添加 --double 双精度参数（进行 100 电子数值容差测试）
python benchmark_100_atoms.py --double
# 或快速体验 (跳过 300 维耗时二阶拉普拉斯求解)：
python benchmark_100_atoms.py --double --quick
```
**关键评测指标亮点**：
- **数值性质检查**：在指定 dtype 和阈值下检查轨道置换与符号反号；同自旋电子重叠在精确算术下应为零，浮点实现只报告小幅值或对数幅值阈值，不宣称精确归零。
- **资源与吞吐需实测**：模型参数量和势能批处理耗时取决于 dtype、batch、硬件及是否包含完整局域能量；本示例只验证可执行性和数值容差。

### 6.5 演示 5：PyTorch 编译加速基准评测 (`benchmark_compile.py`)

运行原生 Eager 模式与 PyTorch 2.0+ `torch.compile` 的加速对比：

```bash
python benchmark_compile.py
# 或使用 uv 运行：
uv run python benchmark_compile.py
```
- **评测方法**：脚本比较 Eager 与 `torch.compile` 的稳态耗时；应同时记录硬件、版本、首次编译成本、dtype、batch 和误差，不能把单次结果推广为固定加速比。

### 6.6 演示 6：一键生成训练与物理状态诊断看板 (`visualize.py`)

在前面的训练中，我们在终端只看到了刷屏的数字，初学者往往很难建立空间几何与物理收敛的直观感觉。
为此，我们提供了**一键可视化导出功能**！在运行 `train_vmc.py` 时只需加上 `--save_plots` 参数：

```bash
# 运行 30 轮 VMC 优化并自动生成诊断可视化图表
python train_vmc.py --system h2 --epochs 30 --walkers 512 --save_plots --plot_dir assets/plots
```

脚本将自动在 `assets/plots/` 目录下生成包含**训练全景三合一看板**、**物理状态三合一看板**、**2D 成键热力图**在内的全套高清图表：

#### 🖼️ 看板 1：训练过程动力学全景看板 (`h2_training_dashboard.png`)

<p align="center">
  <img src="assets/plots/h2_training_dashboard.png" alt="VMC Training Process Dashboard" width="100%">
</p>

小白如何读懂这三张图？
1. **左图：能量收敛曲线（小球滚入量子基态）**
   - **蓝色实线（原始能量）** 与 **绿色实线（截断能量）**：示例运行可能从较高能量下降；曲线会随随机种子、采样和数值设置变化。红色参考线只适用于登记的 H₂ 电荷、自旋和几何；不应把短曲线视为收敛证明；
   - 训练曲线不再绘制误导性的 IID 误差带；可信误差来自训练后冻结参数采集的时间序列，并单独报告 ESS 与链相关 SEM。
2. **中图：能量方差趋势（零方差辅助诊断）**
   - **紫色实线（能量方差）** 与 **橙色点划线（IID 批次 SEM 代理量）**只用于训练诊断；最终误差必须查看冻结参数复评产生的相关 SEM、ESS 和 R-hat。
3. **右图：MCMC 采样器健康度与自适应步长（算法的“恒温器”）**
   - **青色折线（接受率）**：围绕着灰色水平虚线（50% 目标线）小幅振荡，说明采样器既没有盲目乱跳（接受率太低），也没有原地踏步（接受率太高）；
   - **棕色点划线（提议移动步长）**：控制器根据接受率调整步长；稳定值取决于体系、初始化和目标接受率，不应预期固定为某个 Bohr 数。

---

#### 🖼️ 看板 2：量子物理状态全景看板 (`h2_physical_dashboard.png`)

<p align="center">
  <img src="assets/plots/h2_physical_dashboard.png" alt="Quantum Physical State Dashboard" width="100%">
</p>

这些图展示采样构型或固定构型下的可视化诊断，不等同于经过边缘化和归一化的一体电子密度：
1. **左图：3D 采样构型/条件切片散点**
   - 中间两个黑色大球是两颗氢原子核（$z = \pm 0.7\,\text{Bohr}$）；
   - 浅蓝色散点是自旋向上电子（$\alpha$），浅粉色散点是自旋向下电子（$\beta$）；
   - 电子并非固定在轨道上打转，而是像一团有弹性的“棉花糖云雾”，在两颗核周围形成高密度的空间概率弥散，这正是量子力学玻恩几率诠释 $|\Psi(r)|^2$ 的真实写照！
2. **中图：沿核间轴的一维条件切片**
   - 蓝色实线是固定其余电子构型后沿轴线扫描得到的条件量；它不是对其余电子积分得到的一体电子密度，也不能单独证明成键机制。
3. **右图：局域能量直方图与统计偏度（防御奇异点）**
   - 展示了各个空间构型处算出来的局域能量 $E_L(r)$ 的分布直方图；
   - 浅蓝色区域代表 MAD 平滑截断窗口，它有效地将靠近节点面或靠近原子核可能出现的极端离群值柔和截断，左上角信息卡片记录了均值、标准差、偏度与峰度。

---

#### 🖼️ 看板 3：2D 固定构型下的波函数条件切片热力图 (`h2_wavefunction_2d.png`)

<p align="center">
  <img src="assets/plots/h2_wavefunction_2d.png" alt="2D Wavefunction Bonding Slice" width="70%">
</p>

这是固定构型下的波函数条件切片示意图，不是经过边缘化的一体电子密度图：
- 两个带白边的红圈标出了原子核的位置 $(0, \pm 0.7)$；
- 亮黄色和翠绿色代表高电子概率区域，可以清晰看到两核之间有一条宽阔的**高几率绿色通道**，这就是大名鼎鼎的 **$\sigma_g$ 分子成键轨道**！外围深紫色逐渐衰减至零，体现了指数衰减包络的物理约束。

---

### 6.7 训练输出物理指标白话对照表

初学者在看训练日志时，常常不知道各项指标代表什么含义，请参考下表：

| 输出字段 | 物理名称 | 通俗白话解释 | 理想健康状态 |
| :--- | :--- | :--- | :--- |
| `Epoch` | 迭代轮数 | 优化器完成了一次参数更新循环 | 正常递增 |
| `<E_raw>` | 原始局域能量均值 | 当前批次所有 Walker 局域能量的原始算术平均值（单位：Hartree） | **随着训练逐渐下降**，逐步逼近理论基态能量 |
| `<E_clip>` | 截断后局域能量均值 | 经过 Umrigar tanh 过滤掉奇异离群点后的能量均值 | 与 `<E_raw>` 极其接近，若两者差巨大说明节点奇异点较多 |
| `Var(E)` | 能量方差 | 局域能量在空间中的波动方差 $\text{Var}(E_L)$ | 重要辅助诊断；有限样本下会波动，低方差不能单独证明基态收敛 |
| `IID SEM*` | 单批次 IID 代理量 | 暂时忽略链自相关的训练诊断值，不是可信置信区间 | 仅观察趋势；最终结论使用复评的相关 SEM、ESS 与 R-hat |
| `evaluation.energy_sem` | 链相关均值标准误 | 使用 walker 时间序列与有效样本数修正的最终误差条 | 与 ESS、`tau_int`、`R-hat` 一起报告 |
| `ESS` | 有效样本数 | 把自相关链折算为等价独立样本数 | 越大越好，但必须结合采样预算解释 |
| `R-hat` | 链间收敛诊断 | 比较拆分 walker 链的方差与链内方差 | 接近 1 才适合解释 SEM |
| `Acc (%)` | MCMC 接受率 | 采样器提议移动并被接受的百分比 | 默认约 50% 的工程目标；合理范围取决于维度和提议分布 |
| `Step` | MCMC 空间移动步长 | 高斯提议分布的标准差 $\sigma$ | 自适应微调，大分子通常在 0.1 ~ 0.5 之间 |

---

## 7. ❓ 第六章：新手高频问答 FAQ

### Q1: 为什么波函数的对数值 `log|Psi|` 经常是负几百？

> **答**：
> 别慌，这是**完全正常且正确的物理现象**！
> 在波函数中，总几率在全空间的积分是 1。每个电子通常分布在几个立方 Bohr 的空间微元内，单电子几率密度大约在 $0.01 \sim 0.1$ 量级。
> 当体系有 50 或 100 个电子时，高维联合几率密度是这些数值的连乘：
> $$|\Psi|^2 \sim (0.01)^{50} \sim 10^{-100} \implies \ln|\Psi| \approx -115$$
> 如果不用对数表示，直接计算真实浮点数早就下溢成 0 了。因此看到 `log|Psi| = -200` 不仅不是 Bug，反而证明数值防溢出机制正在坚实地保护着你的计算！

### Q2: 什么是 Hartree 原子单位制？怎么换算成熟悉的单位？

> **答**：
> 在物理化学公式中，如果每次都写普朗克常数 $\hbar = 1.054 \times 10^{-34} \text{ J}\cdot\text{s}$、电子质量 $m_e = 9.109 \times 10^{-31} \text{ kg}$、基本电荷 $e = 1.602 \times 10^{-19} \text{ C}$，计算机会因为微小的浮点常数引发严重的精度截断误差。
> 
> 理论物理学家定义了 **原子单位制 (Atomic Units, a.u.)**，直接令：
> $$\hbar = m_e = e = \frac{1}{4\pi\varepsilon_0} = 1$$
> 在原子单位制下：
> - **长度单位**：1 Bohr ($a_0$) $\approx 0.529177$ 埃 ($\text{Å}$) $\approx 5.29 \times 10^{-11}$ 米；
> - **能量单位**：1 Hartree (Ha) $\approx 27.2114 \text{ eV} \approx 627.51 \text{ kcal/mol} \approx 2625.5 \text{ kJ/mol}$。
> 
> 1 Hartree 相当于氢原子基态电离能的两倍（氢原子基态能量严格等于 $-0.5$ Ha）。

### Q3: 为什么 100 电子大体系强烈推荐使用 double (`float64`) 精度？

> **答**：
> 1. **行列式条件数与奇异性**：随着电子数增加，Slater 矩阵维度扩大为 $50 \times 50$ 甚至更大。当两个同自旋电子距离稍近时，两行向量变得极其接近，矩阵接近奇异（病态条件数）。单精度 (`float32`) 只有约 7 位有效数字，求逆和行列式分解容易丢失精度甚至报 `NaN`；而双精度 (`float64`) 拥有 16 位有效数字，能提供强大的数值缓冲。
> 2. **自动微分求二阶导的累积截断误差**：动能项需要对坐标求二次偏导数，数值求导过程中的差分和乘积会放大舍入误差。在大体系中，使用 `dtype=torch.float64` 可以让能量评估具有严格的物理置信度。

### Q4: 为什么我的 MCMC 接受率最好在 50% 附近？

> **答**：
> 这是一个经典的采样探索 (Exploration) 与利用 (Exploitation) 的权衡：
> - **如果接受率极高 (如 95%)**：说明你提议的移动步长 `step_size` 太小了！电子每次只移动极其微小的距离，虽然容易被接受，但粒子被困在原地，成千上万步都探索不到新的空间构型，样本自相关性极强，统计严重失真；
> - **如果接受率极低 (如 5%)**：说明提议步长太大了！电子一步直接跳到了概率几乎为零的荒漠或者直接撞到了原子核内部，导致绝大多数移动全被拒绝，采样器原地踏步；
> - **约 50% 的工程目标**：接受率过高通常意味着步长过小，过低通常意味着步长过大；理论最优值随维度、提议分布和移动策略变化。项目控制器会尝试靠近默认目标，但这不是采样收敛的充分条件。

### Q5: 为什么训练时能量方差 `Var(E)` 会越来越小？(神奇的零方差原理)

> **答**：
> 这是量子蒙特卡洛中的**零方差原理 (Zero Variance Principle)**，但它是一个理想极限性质：
> 
> 观察薛定谔方程：
> $$\hat{H} \Psi_0(r) = E_0 \Psi_0(r)$$
> 如果你的神经网络拟合得**无比完美**，已经成为目标 Hamiltonian 的真实基态波函数 $\Psi_0$，那么在节点之外、局域能量定义良好的位置（$\Psi_0(r)\ne 0$）：
> $$E_L(r) = \frac{\hat{H} \Psi_0(r)}{\Psi_0(r)} = \frac{E_0 \Psi_0(r)}{\Psi_0(r)} \equiv E_0 \quad (\text{恒等于常数！})$$
> 
> 因此在避开节点和数值奇异点的有效采样区域，局域能量理论上等于常数 $E_0$；节点处的比值本身不定义。
> 一组完全等于常数的数列，它的方差是多少？
> $$\text{Var}(E_L) = 0!$$
> 
> **结论**：
> 在普通深度学习分类或回归任务中，Loss 往往很难降到严格的 0。但在量子变分蒙特卡洛中：
> 能量方差可以作为波函数质量的辅助诊断。训练过程中方差变小是有用信号，但有限采样、截断、软化势和优化噪声都可能影响它；它不能单独严格证明网络正在逼近真实基态。

---

## 结语：踏上你的量子深度学习探索之旅

恭喜你！读完这份宝典，你已经掌握了计算量子化学与深度学习交叉领域最核心的物理直觉与工程奥秘。
现在，不妨打开终端，运行 `uv run python -m ferminet_pytorch.train_vmc`，亲眼见证神经网络捕捉微观电子规律的奇迹吧！
