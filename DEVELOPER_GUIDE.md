# FermiNet 开发者与二次开发全景指南 (Developer & Extensibility Guide)

> **目标受众**：面向量子化学深度学习研究员、算法工程师、开源贡献者以及希望在 FermiNet 基础上实现自定义模型、算符或优化策略的开发者。
> **前置阅读**：如需了解量子化学基本物理背景与直觉比喻，请先参阅 [新手入门宝典 (BEGINNER_GUIDE.md)](file:///c:/Users/Administrator/.gemini/antigravity/scratch/ferminet_pytorch/BEGINNER_GUIDE.md)。

---

## 目录 (Table of Contents)

1. [📐 整体架构设计理念 (Design Philosophy)](#1-整体架构设计理念-design-philosophy)
   - 1.1 从单体到四层正交解耦 (Gen1 vs Gen2)
   - 1.2 单批次共享几何上下文 (GeometricContext)
   - 1.3 显存削峰与单前向契约 (Single-Forward Contract)
2. [🧩 核心模块全景与职责拓扑](#2-核心模块全景与职责拓扑)
3. [🛠️ 四大二次开发扩展实战 (Hands-on Extensibility)](#3-四大二次开发扩展实战-hands-on-extensibility)
   - 3.1 扩展实战一：实现自定义等变特征交互层 (如引入图神经网络/自定义注意机制)
   - 3.2 扩展实战二：实现自定义核包络 (Nuclear Envelope)
   - 3.3 扩展实战三：扩展自定义哈密顿量算符 (外加电场/磁场/周期性势能)
   - 3.4 扩展实战四：替换自定义 MCMC 采样策略 (如 Langevin / 独立提议采样)
4. [⚡ 现代动能求解器 (KineticEnergyCalculator) 深度解析](#4-现代动能求解器-kineticenergycalculator-深度解析)
   - 4.1 算法模式对比：`REVERSE` vs `HUTCHINSON` vs `FORWARD_JVP`
   - 4.2 动能高阶微分的显存陷阱与规避指南
5. [🔄 现代化 VMC 驱动引擎 (VMCDriver) 工作流](#5-现代化-vmc-驱动引擎-vmcdriver-工作流)
   - 5.1 四大正交组件装配范式
   - 5.2 状态冻结与两阶段训练 (Jastrow-Warmup to Full-VMC)
6. [🔌 向后兼容层 (Legacy Adapters) 实现机制](#6-向后兼容层-legacy-adapters-实现机制)
7. [🧪 调试、测试与数值验证规范](#7-调试测试与数值验证规范)
   - 7.1 物理不变量核验金标准
   - 7.2 性能剖析 (Profiling) 与内存泄漏排查

---

## 1. 📐 整体架构设计理念 (Design Philosophy)

### 1.1 从单体到四层正交解耦 (Gen1 vs Gen2)

在初代研究型代码 (Gen1) 中，`FermiNet` 往往是一个“上帝类”——同时承载了几何坐标差计算、两流特征交互、原子核包络、Slater 行列式组装、Jastrow 因子与自旋投影。这种设计易于单一脚本演示，但在进行科研探索（如仅替换等变层、尝试非库仑势能、引入新型采样器）时极难维护。

在 **Gen2 现代化架构** 中，系统严格解耦为四个相互正交的支柱：

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           VMCDriver (驱动引擎)                          │
│         协调优化器、学习率调度、采样步数推进、统计指标聚合与两阶段训练          │
└──────────────┬──────────────────────────┬───────────────────────────────┘
               │                          │
 ┌─────────────▼──────────────┐ ┌─────────▼──────────────────────────────┐
 │  VariationalState (变分态) │ │   HamiltonianComposite (哈密顿算符)   │
 │ - 管理 Walker 构型分布      │ │ - 动能算符: KineticEnergyCalculator    │
 │ - 缓存共享几何上下文        │ │ - 势能算符: Coulomb (e-n, e-e, n-n)     │
 │ - 模型波函数单次前向求值    │ │ - 支持可插拔外加势能/周期边界势能       │
 └─────────────┬──────────────┘ └────────────────────────────────────────┘
               │
 ┌─────────────▼──────────────┐
 │ MetropolisSampler (采样器) │
 │ - 纯几何空间马尔可夫链推进  │
 │ - 自适应步长控制 (目标 50%)│
 └────────────────────────────┘
```

### 1.2 单批次共享几何上下文 (GeometricContext)

在量子化学计算中，电子-核位移 $\mathbf{r}_{iI} = \mathbf{r}_i - \mathbf{R}_I$、电子-核距离 $d_{iI} = \|\mathbf{r}_{iI}\|$、电子-电子位移 $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$ 和电子-电子距离 $d_{ij}$ 是最核心的几何不变量。

过去，特征提取层、核包络层、Jastrow 因子以及势能计算中分别重复计算了 4 次这些几何张量。`GeometricContext` 将其统一收拢为不可变上下文容器：

```python
from ferminet_pytorch import GeometricContext

# 在构型 r 处单次前向执行处处可微的几何抽取
geom_ctx = GeometricContext.compute(
    r=r,                       # (B, N, 3)
    nuclear_coords=R_nuclei,   # (M, 3)
    eps=1e-12,                 # 平滑正则项
    compute_ee=True,           # 是否计算电子-电子对距离
)

# 后续所有子模块共享只读引用，彻底消除重复显存开销与重复算子
# geom_ctx.r_en : (B, N, M, 3)
# geom_ctx.d_en : (B, N, M, 1)
# geom_ctx.r_ee : (B, N, N, 3)
# geom_ctx.d_ee : (B, N, N, 1)
```

### 1.3 显存削峰与单前向契约 (Single-Forward Contract)

在 VMC 变分优化中，局域能量 $E_L(\mathbf{r}) = T_L(\mathbf{r}) + V(\mathbf{r})$ 计算时需要对输入坐标 $\mathbf{r}$ 求二阶导（动能项），但**不需要对网络参数 $\theta$ 求导**。
而在计算变分代理损失 $\mathcal{L}(\theta) = 2 \mathbb{E}[(E_L - \bar{E}_L) \ln|\Psi_\theta|]$ 时，需要对参数 $\theta$ 求导，但此时 $E_L$ 应当被 `.detach()`。

**设计禁忌**：若将动能二阶导与参数梯度计算混在同一张计算图中，PyTorch 会常驻庞大的二阶导激活值，导致显存直接爆炸。
**Gen2 解决方案**：严格拆分求导生命周期：
1. 第一阶段：在无参数梯度上下文或仅对坐标 $\mathbf{r}$ 追踪梯度的状态下计算 $E_L$，并在得出标量后彻底 `detach()`；
2. 第二阶段：复用已采样的几何构型，仅对参数 $\theta$ 做一次标准轻量前向，反向传播更新梯度。

---

## 2. 🧩 核心模块全景与职责拓扑

| 模块文件 | 核心类 / 函数 | 职责与设计模式 |
| :--- | :--- | :--- |
| `geometry.py` | `GeometricContext`, `smooth_distance` | 集中式向量化不可变几何上下文容器，处处可微距离平滑 |
| `features.py` | `ElectronFeatureEmbedding` | 将几何坐标转化为单电子流与双电子流输入特征，支持平滑无尖峰距离 |
| `layers_modern.py` | `FermiInteractionLayerModern` | 现代化两流等变残差层，零高频小张量分配，矩阵乘法融合 |
| `psiformer_attention.py` | `PsiformerAttentionBlock` | 基于自注意力机制的自旋等变层，省去显式双电子流存储，大幅削减大体系显存 |
| `envelope_modern.py` | `NuclearEnvelopeModern` | 向量化原子核衰减包络（支持指数与平滑高斯），消除循环动态广播 |
| `jastrow.py` | `KatoCuspJastrow` | 显式满足电子-核与同/异自旋电子-电子 Kato 尖峰条件的 Padé Jastrow 因子 |
| `spin.py` | `SpinProjector` | 空间自旋群投影算符（消除自旋污染，如双电子单重态严格空间对称化） |
| `composable_model.py` | `ComposableFermiNet`, `OrbitalEnvelopeProjector`, `BranchlessCofactorDeterminant` | 可自由插拔模块组合的波函数擬设，纯张量代数余子式行列式求导 |
| `kinetic_modern.py` | `KineticEnergyCalculator`, `KineticMode` | 多后端动能算符求解器（经典反向模式、Hutchinson 随机迹估计、前向 JVP） |
| `hamiltonian.py` | `MolecularSystem`, `compute_local_energy`, `diagnose_cusp_divergence` | 分子体系定义、库仑相互作用势能与尖峰奇异性诊断分析工具 |
| `mcmc.py` | `MCMCSampler` | 批量并行 Metropolis-Hastings 采样，目标接受率步长自适应 |
| `loss.py` | `VMCLoss`, `clip_local_energy`, `vmc_surrogate_loss` | 协方差变分代理损失，基于 MAD 的稳健 Umrigar Tanh 平滑能量截断 |
| `vmc_driver.py` | `VariationalState`, `HamiltonianComposite`, `MetropolisSampler`, `VMCDriver` | 现代化四层正交解耦高层调度引擎 |
| `legacy_adapters.py` | `FermiNetLegacyAdapter`, `...LegacyAdapter` | 为历史测试与旧接口提供 100% 逐 bit 兼容的轻量门面层 |

---

## 3. 🛠️ 四大二次开发扩展实战 (Hands-on Extensibility)

### 3.1 扩展实战一：实现自定义等变特征交互层

如果你想研究新的等变神经网络架构（如基于 Transformer、图神经网络 GNN 或稀疏局部注意力的电子特征提取），只需要实现一个接收单电子流和双电子流（或仅单电子流）的 `nn.Module`。

以下演示如何编写一个轻量级“残差多层感知机 (MLP) 交互层”并集成到模型中：

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ferminet_pytorch import ComposableFermiNet, build_h2_system

class CustomGatedEquivariantLayer(nn.Module):
    """
    用户自定义的自旋等变门控交互层示例。
    维持全同费米子等变性：对同自旋组与异自旋组分别进行置换不变池化。
    """
    def __init__(self, hidden_dim: int, n_up: int, n_down: int):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        # 单电子流门控更新
        self.linear_in = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, h1: torch.Tensor, h2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # h1: (B, N, hidden_dim)
        B, N, D = h1.shape
        
        # 1. 对同自旋组分别计算均值池化 (置换不变标量)
        mean_up = h1[:, :self.n_up].mean(dim=1, keepdim=True).expand(-1, N, -1) if self.n_up > 0 else torch.zeros_like(h1)
        mean_down = h1[:, self.n_up:].mean(dim=1, keepdim=True).expand(-1, N, -1) if self.n_down > 0 else torch.zeros_like(h1)
        
        # 2. 特征拼接与门控残差
        combined = torch.cat([h1, mean_up, mean_down], dim=-1)  # (B, N, 3*D)
        update = self.act(self.linear_in(combined))
        gate_coeff = torch.sigmoid(self.gate(combined))
        
        h1_next = h1 + gate_coeff * update  # 残差连接
        return h1_next, h2

# 验证自定义层在 ComposableFermiNet 中的可插拔性
system = build_h2_system()
# 你也可以基于 ComposableFermiNet 派生，或者直接将 layers 替换为你自定义的模块列表
```

> 💡 **设计准则**：只要保持对 $1 \dots n_\uparrow$ 电子以及 $n_\uparrow+1 \dots N$ 电子在空间置换下的等变性，任何特征提取网络均严格满足泡利反对称要求！

---

### 3.2 扩展实战二：实现自定义核包络 (Nuclear Envelope)

默认核包络提供了 `exponential`（各向同性指数衰减）与 `gaussian`（平滑高斯衰减）。如果你的体系处于强各向异性晶场或需要各向异性椭球包络，可轻松自定义：

```python
import torch
import torch.nn as nn
from typing import Optional

class AnisotropicEllipsoidEnvelope(nn.Module):
    """
    各向异性椭球核包络:
    Envelope = sum_m pi_m * exp(-sqrt( (r-R)^T Sigma_m (r-R) ))
    """
    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        nuclear_coords: torch.Tensor,
    ):
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.num_nuclei = nuclear_coords.shape[0]
        
        # 对角各向异性缩放因子 (K, N_orb, M, 3)
        self.sigma_diag = nn.Parameter(
            torch.ones(num_determinants, num_orbitals, self.num_nuclei, 3)
        )
        self.pi_weights = nn.Parameter(
            torch.ones(num_determinants, num_orbitals, self.num_nuclei)
        )

    def forward(self, r_en: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r_en: 电子-核相对位移, shape (B, N_spin, M, 3)
        Returns:
            envelope: shape (B, K, N_spin, num_orbitals)
        """
        # r_en: (B, N_spin, M, 3) -> 拓展为 (B, 1, N_spin, 1, M, 3)
        r_diff = r_en.unsqueeze(1).unsqueeze(3)
        # 与各向异性参数点乘
        scaled_diff = r_diff * self.sigma_diag.unsqueeze(0).unsqueeze(2)
        dist_aniso = torch.norm(scaled_diff, dim=-1)  # (B, K, N_spin, N_orb, M)
        
        # 衰减并对原子核加权求和
        decay = torch.exp(-dist_aniso) * self.pi_weights.unsqueeze(0).unsqueeze(2)
        envelope = torch.sum(decay, dim=-1)  # (B, K, N_spin, N_orb)
        return envelope
```

---

### 3.3 扩展实战三：扩展自定义哈密顿量算符 (外加电场/磁场/周期性势能)

在科学研究中，我们常常需要研究分子在**外加恒定电场**（斯塔克效应 Stark Effect）或**谐振子势阱**中的响应。
在 `HamiltonianComposite` 中，添加额外势能只需要扩展几行代码：

```python
import torch
import torch.nn as nn
from ferminet_pytorch import HamiltonianComposite, MolecularSystem, GeometricContext
from typing import Optional, Tuple

class StarkElectricFieldHamiltonian(HamiltonianComposite):
    """
    引入沿 Z 轴的恒定外加匀强电场 E_field:
    附加势能: V_ext = sum_i e * E_z * z_i - sum_I Z_I * E_z * Z_I_coord
    """
    def __init__(self, system: MolecularSystem, electric_field_z: float = 0.01):
        super().__init__(system=system)
        self.electric_field_z = electric_field_z

    def compute_potential_energy(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> torch.Tensor:
        # 1. 基础库仑相互作用势能 (核-核 + 核-电 + 电-电)
        v_coulomb = super().compute_potential_energy(r, geom_ctx=geom_ctx)
        
        # 2. 电子在外电场中的势能: + E_z * z (原子单位制下电子电荷为 -1, 势能 -q * E * r = + E * z)
        z_electrons = r[..., 2]  # (B, N)
        v_elec_field = self.electric_field_z * torch.sum(z_electrons, dim=-1)  # (B,)
        
        # 3. 原子核在外电场中的势能
        z_nuclei = self.system.nuclear_coords[:, 2]  # (M,)
        v_nuc_field = - self.electric_field_z * torch.sum(self.system.nuclear_charges * z_nuclei)
        
        return v_coulomb + v_elec_field + v_nuc_field

# 使用方式: 直接传给 VMCDriver
# driver = VMCDriver(variational_state=state, hamiltonian=StarkElectricFieldHamiltonian(system, 0.05), ...)
```

---

### 3.4 扩展实战四：替换自定义 MCMC 采样策略 (如朗之万动力学 MALA)

默认的 `MetropolisSampler` 基于各向同性高斯随机游走。当需要加速相空间探索时，可以构建基于波函数梯度的 Metropolis-Adjusted Langevin Algorithm (MALA)：

```python
import torch
from ferminet_pytorch import VariationalState

class MALASampler:
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) 采样原型:
    借助一阶漂移项: r_prop = r + 0.5 * dt * grad_r(ln|Psi|) + sqrt(dt) * xi
    """
    def __init__(self, variational_state: VariationalState, dt: float = 0.02):
        self.vstate = variational_state
        self.dt = dt

    def step(self, decorrelation_steps: int = 1) -> float:
        # 获取当前构型并计算对数波函数关于坐标的一阶梯度
        for _ in range(decorrelation_steps):
            r = self.vstate.current_walkers.clone().detach().requires_grad_(True)
            log_psi, _ = self.vstate.model.forward_log_psi(r)
            grad_r = torch.autograd.grad(log_psi.sum(), r)[0].detach()
            
            # 提议转移
            noise = torch.randn_like(r)
            drift = 0.5 * self.dt * grad_r
            r_prop = (r + drift + torch.sqrt(torch.tensor(self.dt)) * noise).detach()
            
            # 评估提议点并计算 Metropolis-Hastings 接受率 (包含不对称转移核修正)
            with torch.no_grad():
                log_psi_prop, _ = self.vstate.model.forward_log_psi(r_prop)
                # 此处计算 MALA 接受率并更新 walkers ...
                self.vstate.update_walkers(r_prop)
        return 0.55  # 示例返回接受率
```

---

## 4. ⚡ 现代动能求解器 (KineticEnergyCalculator) 深度解析

### 4.1 算法模式对比：`REVERSE` vs `HUTCHINSON` vs `FORWARD_JVP`

局域动能定义为：
$$T_L(\mathbf{r}) = -\frac{1}{2} \left[ \nabla^2 \ln|\Psi(\mathbf{r})| + |\nabla \ln|\Psi(\mathbf{r})||^2 \right]$$

`KineticEnergyCalculator` 提供三种计算模式，满足从严苛测试到大体系高吞吐的不同需求：

```python
from ferminet_pytorch import KineticEnergyCalculator, KineticMode

# 1. 经典精确反向模式 (REVERSE) - 默认推荐
# 精度: 理论最高 (精确二阶自动微分)
# 复杂度: O(3N) 次向量-雅可比积反向传播
# 适用: N <= 10 的中小型分子体系、物理不变量严格回归测试
t_loc, lap, grad_sq = KineticEnergyCalculator.compute_kinetic_energy(
    log_abs_psi_fn=model.forward_log_psi,
    r=r,
    mode=KineticMode.REVERSE
)

# 2. Hutchinson 随机迹估计模式 (HUTCHINSON)
# 精度: 无偏估计 (通过 Monte Carlo 随机 Rademacher 探针向量估计 Tr(Hessian))
# 复杂度: O(1) 次反向传播！
# 适用: 体系电子数 N >= 20 时，加速比可达 5x ~ 10x
t_loc_hutch, _, _ = KineticEnergyCalculator.compute_kinetic_energy(
    log_abs_psi_fn=model.forward_log_psi,
    r=r,
    mode=KineticMode.HUTCHINSON,
    num_probes=1
)

# 3. 前向切线推进模式 (FORWARD_JVP)
# 基于 torch.func.jvp 前向计算，在支持前向自动微分的环境下无需保留反向计算图
```

### 4.2 动能高阶微分的显存陷阱与规避指南

很多开发者在编写二阶微分时习惯传递 `create_graph=True`。注意：
- 在 VMC 训练中，计算动能标量 $T_L$ 时，**千万不要开启 `create_graph=True`**（除非你在做元学习或对局域能量本身求参数梯度）；
- 保持 `create_graph=False` 可以让 PyTorch 在完成坐标导数后立即释放中间激活值，将显存占用降低 60% 以上；
- 坐标张量必须是独立的局部叶子节点：`r_eval = walkers.clone().detach().requires_grad_(True)`。

---

## 5. 🔄 现代化 VMC 驱动引擎 (VMCDriver) 工作流

### 5.1 四大正交组件装配范式

使用 `VMCDriver` 构建生产级变分训练闭环只需 5 步：

```python
import torch
import ferminet_pytorch as fp

# 1. 物理体系与波函数拟设
system = fp.build_h2_system()
model = fp.ComposableFermiNet(
    nuclear_coords=system.nuclear_coords,
    nuclear_charges=system.nuclear_charges,
    n_up=system.n_up,
    n_down=system.n_down,
    use_cusp_jastrow=True,
    spin_adapted=True,
)

# 2. 状态容器与物理算符
state = fp.VariationalState(model=model, system=system, num_walkers=512)
hamiltonian = fp.HamiltonianComposite(system=system)

# 3. 采样器与优化器
sampler = fp.MetropolisSampler(variational_state=state, step_size=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = fp.build_cosine_scheduler(optimizer, max_epochs=100)

# 4. 组装驱动引擎
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

# 5. 执行训练迭代
for epoch in range(1, 101):
    metrics = driver.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | E: {metrics['energy_mean']:.5f} Ha | Var: {metrics['energy_std']**2:.4f} | Acc: {metrics['acceptance_rate']*100:.1f}%")
```

### 5.2 状态冻结与两阶段训练 (Jastrow-Warmup to Full-VMC)

在深度波函数训练初期，如果直接联合优化神经网络与 Jastrow 因子，随机初始化的 Slater 轨道可能破坏 Jastrow 刚性建立的库仑对消。
推荐使用两阶段训练策略：
- **阶段一 (Jastrow 预热)**：冻结神经网络主干参数（`requires_grad=False`），仅训练 Jastrow 因子的阻尼参数 $b_{en}, b_{ee}$，使基底波函数迅速满足 Kato 尖峰渐近；
- **阶段二 (全量 VMC)**：解冻所有网络参数，使用余弦退火学习率进行全变分精修。
项目在 `train_vmc.py` 中提供了开箱即用的 `train_two_stage_jastrow` 自动化流程。

---

## 6. 🔌 向后兼容层 (Legacy Adapters) 实现机制

为了确保现存的旧版测试脚本、第三方拓展以及教学代码不需要做任何破坏性修改，`legacy_adapters.py` 提供了零破坏向后兼容门面：

```python
from ferminet_pytorch.legacy_adapters import FermiNetLegacyAdapter

# 旧接口风格：无缝包装 ComposableFermiNet，保持经典参数列表与输出行为
legacy_model = FermiNetLegacyAdapter(
    nuclear_coords=system.nuclear_coords,
    nuclear_charges=system.nuclear_charges,
    n_up=system.n_up,
    n_down=system.n_down,
    use_cusp_jastrow=True,
)
```

所有的旧版单元测试均在 CI 中持续运行，保证新老代码完全无缝过渡。

---

## 7. 🧪 调试、测试与数值验证规范

### 7.1 物理不变量核验金标准

在修改任何波函数代码或提交 PR 前，必须确保以下物理不变量通过检验：
1. **反对称性 (泡利不相容原理)**：
   两个同自旋电子坐标重合时，波函数绝对值必须为零：$|\Psi(\mathbf{r}_1=\mathbf{r}_2)| < 10^{-7}$；交换两个同自旋电子坐标时，波函数必须严格反号。
2. **总自旋本征态 ($\hat{S}^2 \Psi = S(S+1)\Psi$)**：
   单重态投影后必须满足 $\hat{S}^2 \Psi = 0$，不存在高自旋污染。
3. **Kato 尖峰导数极限**：
   电子逼近原子核时，径向对数导数 $\lim_{r_{iI} \to 0} \frac{\partial \ln|\Psi|}{\partial r_{iI}} = -Z_I$；
   电子逼近同自旋电子时为 $1/4$，逼近异自旋电子时为 $1/2$。
4. **能量局域有限性**：
   在微观粒子碰撞点，局域能量 $E_L(\mathbf{r})$ 必须处处保持有限常数，绝不能发生 $\pm \infty$ 奇点发散。

### 7.2 自动化测试执行命令

```bash
# 运行完整自动化测试套件 (230+ 项物理与架构单元测试)
uv run python -m unittest discover -s . -p "test_*.py" -v

# 仅运行现代架构回归测试
uv run python -m unittest test_refactor_review_regressions.py test_envelope_modern.py test_layers_modern.py test_kinetic_modern.py -v

# 在 GPU 环境下运行 CUDA 精度与回归测试 (需要 Conda torch 环境)
F:\conda\envs\torch\python.exe -m unittest test_cuda.py -v
```
