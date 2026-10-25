"""
量子化学分子体系定义与局域能量计算。

- MolecularSystem: 分子体系的几何与电子构型
- compute_local_energy: 基于自动微分的局域能量计算
    E_L(r) = T_L(r) + V(r)
    T_L(r) = -0.5 * (∇²ln|Ψ| + |∇ln|Ψ||²)
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .geometry import GeometricContext, smooth_distance
from .kinetic_modern import KineticEnergyCalculator, KineticMode


class _FreezeModelParameters:
    """临时隔离模型参数的梯度追踪，阻止二阶计算图遍历模型权重。"""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.prev_states: List[bool] = []

    def __enter__(self) -> "_FreezeModelParameters":
        self.prev_states = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for p, state in zip(self.model.parameters(), self.prev_states):
            p.requires_grad = state


@dataclass
class MolecularSystem:
    """
    量子化学分子体系定义。

    Attributes:
        nuclear_coords: 原子核坐标, Shape: (M, 3)
        nuclear_charges: 原子核电荷, Shape: (M,)
        n_up: 自旋向上电子数
        n_down: 自旋向下电子数
    """

    nuclear_coords: torch.Tensor
    nuclear_charges: torch.Tensor
    n_up: int
    n_down: int
    v_nn: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        assert self.nuclear_coords.ndim == 2 and self.nuclear_coords.shape[1] == 3
        assert self.nuclear_charges.ndim == 1
        assert self.n_up >= 0 and self.n_down >= 0
        assert self.n_up + self.n_down > 0

        # 预计算核-核库仑排斥常数 V_nn = sum_{I < J} Z_I * Z_J / |R_I - R_J|
        num_nuclei = self.nuclear_coords.shape[0]
        if num_nuclei > 1:
            row_idx, col_idx = torch.triu_indices(
                num_nuclei, num_nuclei, offset=1, device=self.nuclear_coords.device
            )
            diff = self.nuclear_coords[row_idx] - self.nuclear_coords[col_idx]
            dist = torch.linalg.vector_norm(diff, dim=-1)
            charge_prod = self.nuclear_charges[row_idx] * self.nuclear_charges[col_idx]
            self.v_nn = torch.sum(charge_prod / dist)
        else:
            self.v_nn = torch.tensor(
                0.0, device=self.nuclear_coords.device, dtype=self.nuclear_coords.dtype
            )

    @property
    def total_electrons(self) -> int:
        return self.n_up + self.n_down

    @property
    def num_nuclei(self) -> int:
        return self.nuclear_coords.shape[0]

    def to(self, *args, **kwargs) -> "MolecularSystem":
        """移动张量到指定设备/类型。"""
        self.nuclear_coords = self.nuclear_coords.to(*args, **kwargs)
        self.nuclear_charges = self.nuclear_charges.to(*args, **kwargs)
        self.v_nn = self.v_nn.to(*args, **kwargs)
        return self

    def compute_potential_energy(
        self,
        r: torch.Tensor,
        eps: float = 0.0,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> torch.Tensor:
        """
        计算总库仑势能: V(r) = V_en + V_ee + V_nn。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            eps: 显式软核平方长度，默认 0 表示真实库仑距离
            geom_ctx: 可选的 GeometricContext 共享几何上下文

        Returns:
            v_total: 总势能, Shape: (B,)
        """
        batch_size, num_electrons, _ = r.shape
        device, dtype = r.device, r.dtype

        nuclear_charges = self.nuclear_charges.to(device=device, dtype=dtype)

        # Physical distances are independent of feature smoothing and caches.
        if eps < 0:
            raise ValueError("Potential distance regularization must be nonnegative")
        nuclear_coords = self.nuclear_coords.to(device=device, dtype=dtype)
        diff_en = r.unsqueeze(2) - nuclear_coords.unsqueeze(0).unsqueeze(0)
        dist_en = smooth_distance(diff_en, eps=eps, keepdim=False)

        charges = nuclear_charges.unsqueeze(0).unsqueeze(0)
        v_en = -torch.sum(charges / dist_en, dim=(-1, -2))

        # 电子-电子排斥: V_ee = sum_{i<j} 1/|r_i - r_j|
        if num_electrons > 1:
            triu_i, triu_j = torch.triu_indices(
                num_electrons, num_electrons, offset=1, device=device
            )
            # Never differentiate sqrt(0) for unused self-pairs: masking after
            # sqrt leaves 0/0 in the coordinate backward pass.
            diff_pairs = r[:, triu_i, :] - r[:, triu_j, :]
            dist_pairs = smooth_distance(diff_pairs, eps=eps, keepdim=False)
            v_ee = torch.sum(1.0 / dist_pairs, dim=-1)
        else:
            v_ee = torch.zeros(batch_size, device=device, dtype=dtype)

        # 核-核排斥: V_nn (预计算常数)
        v_nn = self.v_nn.to(device=device, dtype=dtype).expand(batch_size)

        return v_en + v_ee + v_nn


def compute_local_energy(
    model: nn.Module,
    system: MolecularSystem,
    r: torch.Tensor,
    create_graph: bool = False,
    geom_ctx: Optional[GeometricContext] = None,
    kinetic_mode: KineticMode = KineticMode.REVERSE,
    num_probes: int = 1,
    return_wavefunction: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    计算局域能量: E_L(r) = T_L(r) + V(r)。

    支持经典反向精确微分、Hutchinson 随机估计与前向切线推进模式。

    Args:
        model: FermiNet 模型
        system: 分子体系
        r: 电子坐标, Shape: (B, N, 3), 必须 requires_grad=True
        create_graph: 是否保留高阶计算图
        geom_ctx: 可选的 GeometricContext 共享几何上下文
        kinetic_mode: 动能估计算法模式 (默认 KineticMode.REVERSE)
        num_probes: Hutchinson 估计探针数
        return_wavefunction: 额外返回本次坐标求导前向的 detached log|Psi| 和 sign；
            仅用于观察记录，不能用作参数代理损失的前向图。

    Returns:
        e_local: 局域总能量, Shape: (B,)
        t_local: 局域动能, Shape: (B,)
        v_local: 局域势能, Shape: (B,)
        若 return_wavefunction=True，还返回 detached log_abs_psi 和 sign，均为 (B,)。
    """
    assert r.requires_grad, "输入 r 必须开启梯度追踪"

    freeze_ctx = _FreezeModelParameters(model) if not create_graph else nullcontext()
    observed_values = []
    with freeze_ctx:
        # 仅当 geom_ctx 确实源自当前开启求导的坐标 r 时才可安全复用，防止无梯度外部上下文导致动能导数丢失
        safe_geom_ctx = (
            geom_ctx
            if (
                geom_ctx is not None and geom_ctx.r is r and geom_ctx.d_en.requires_grad
            )
            else None
        )

        def _call_model_forward(
            coords: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # A dual coordinate needs geometry derived from that dual, not its primal.
            if safe_geom_ctx is not None and coords is r:
                try:
                    values = model.forward_log_psi(coords, geom_ctx=safe_geom_ctx)
                except TypeError as error:
                    if "unexpected keyword argument 'geom_ctx'" not in str(error):
                        raise
                    values = model.forward_log_psi(coords)
            else:
                values = model.forward_log_psi(coords)
            if return_wavefunction:
                observed_values[:] = [value.detach() for value in values]
            return values

        t_local, _, _ = KineticEnergyCalculator.compute_kinetic_energy(
            _call_model_forward,
            r,
            mode=kinetic_mode,
            num_probes=num_probes,
            create_graph=create_graph,
        )

        # 势能使用独立物理距离，geom_ctx 不改变哈密顿量
        if create_graph:
            v_local = system.compute_potential_energy(r, geom_ctx=geom_ctx)
        else:
            with torch.no_grad():
                v_local = system.compute_potential_energy(r, geom_ctx=geom_ctx)

        e_local = t_local + v_local
        if return_wavefunction:
            return e_local, t_local, v_local, *observed_values
        return e_local, t_local, v_local


def compute_kinetic_energy(
    model: nn.Module,
    r: torch.Tensor,
    create_graph: bool = False,
    mode: KineticMode = KineticMode.REVERSE,
    num_probes: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算局域动能及分量:
        T_L(r) = -0.5 * (∇²ln|Ψ| + |∇ln|Ψ||²)

    Args:
        model: 波函数模型
        r: 电子坐标, Shape: (B, N, 3), 必须 requires_grad=True
        create_graph: 是否保留二阶计算图
        mode: 动能估计算法模式 (默认 REVERSE)
        num_probes: Hutchinson 估计探针数

    Returns:
        t_local: 局域动能, Shape: (B,)
        laplacian: 对数波函数拉普拉斯 ∇²ln|Ψ|, Shape: (B,)
        grad_squared: 一阶梯度平方模 |∇ln|Ψ||², Shape: (B,)
    """
    assert r.requires_grad, "输入 r 必须开启梯度追踪"
    freeze_ctx = _FreezeModelParameters(model) if not create_graph else nullcontext()
    with freeze_ctx:
        return KineticEnergyCalculator.compute_kinetic_energy(
            model.forward_log_psi,
            r,
            mode=mode,
            num_probes=num_probes,
            create_graph=create_graph,
        )


def diagnose_cusp_divergence(
    model: nn.Module,
    system: MolecularSystem,
    r_base: torch.Tensor,
    electron_idx: int,
    nucleus_idx: int,
    distances: torch.Tensor,
    direction: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    诊断电子向原子核逼近时 (r_{iI} -> 0) 的局域能量及其奇点发散行为。

    用于验证 Kato 尖峰 Jastrow 因子是否有效抵消库仑势能发散。

    Args:
        model: FermiNet 模型 (可带有或不带有 Kato Cusp Jastrow 因子)
        system: 分子体系
        r_base: 基础构型参考点, Shape: (1, N, 3)
        electron_idx: 移动的电子序号
        nucleus_idx: 逼近的目标原子核序号
        distances: 逼近距离序列向量, Shape: (S,)
        direction: 逼近空间方向单位向量 (3,), 默认 [0, 0, 1]

    Returns:
        包含 distances, e_local, t_local, v_local 的张量字典
    """
    device = r_base.device
    dtype = r_base.dtype
    r_nuc = system.nuclear_coords[nucleus_idx].to(device=device, dtype=dtype)

    if direction is None:
        direction = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        direction = direction.to(device=device, dtype=dtype)
        direction = direction / torch.linalg.vector_norm(direction)

    num_samples = distances.shape[0]
    e_list = []
    t_list = []
    v_list = []

    for s in range(num_samples):
        dist = distances[s].item()
        r_sample = r_base.clone().detach()
        r_sample[0, electron_idx, :] = r_nuc + dist * direction
        r_sample.requires_grad_(True)

        e, t, v = compute_local_energy(model, system, r_sample, create_graph=False)
        e_list.append(e.item())
        t_list.append(t.item())
        v_list.append(v.item())

    return {
        "distances": distances.detach().clone(),
        "e_local": torch.tensor(e_list, device=device, dtype=dtype),
        "t_local": torch.tensor(t_list, device=device, dtype=dtype),
        "v_local": torch.tensor(v_list, device=device, dtype=dtype),
    }
