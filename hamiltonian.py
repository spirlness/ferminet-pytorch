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

try:
    from .features import smooth_distance
except (ImportError, ValueError):
    from features import smooth_distance


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

    def compute_potential_energy(self, r: torch.Tensor, eps: float = 1e-24) -> torch.Tensor:
        """
        计算总库仑势能: V(r) = V_en + V_ee + V_nn。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            eps: 最小平方距离

        Returns:
            v_total: 总势能, Shape: (B,)
        """
        batch_size, num_electrons, _ = r.shape
        device, dtype = r.device, r.dtype

        nuclear_coords = self.nuclear_coords.to(device=device, dtype=dtype)
        nuclear_charges = self.nuclear_charges.to(device=device, dtype=dtype)

        # 电子-核吸引: V_en = -sum_i sum_I Z_I / |r_i - R_I|
        diff_en = r.unsqueeze(2) - nuclear_coords.unsqueeze(0).unsqueeze(0)
        dist_en = smooth_distance(diff_en, eps=eps, keepdim=False)
        charges = nuclear_charges.unsqueeze(0).unsqueeze(0)
        v_en = -torch.sum(charges / dist_en, dim=(-1, -2))

        # 电子-电子排斥: V_ee = sum_{i<j} 1/|r_i - r_j|
        if num_electrons > 1:
            diff_ee = r.unsqueeze(2) - r.unsqueeze(1)
            dist_ee = smooth_distance(diff_ee, eps=eps, keepdim=False)
            mask_ee = torch.triu(
                torch.ones(num_electrons, num_electrons, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0)
            v_ee = torch.sum(
                torch.where(mask_ee, 1.0 / dist_ee, torch.zeros_like(dist_ee)),
                dim=(-1, -2),
            )
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算局域能量: E_L(r) = T_L(r) + V(r)。

    使用经典反向模式逐维精确微分求拉普拉斯:
        T_L = -0.5 * (∇²ln|Ψ| + |∇ln|Ψ||²)

    Args:
        model: FermiNet 模型
        system: 分子体系
        r: 电子坐标, Shape: (B, N, 3), 必须 requires_grad=True
        create_graph: 是否保留高阶计算图

    Returns:
        e_local: 局域总能量, Shape: (B,)
        t_local: 局域动能, Shape: (B,)
        v_local: 局域势能, Shape: (B,)
    """
    batch_size, num_electrons, _ = r.shape
    assert r.requires_grad, "输入 r 必须开启梯度追踪"

    freeze_ctx = _FreezeModelParameters(model) if not create_graph else nullcontext()
    with freeze_ctx:
        # 1. log|Psi(r)|
        log_abs_psi, _ = model.forward_log_psi(r)

        # 2. 一阶梯度 ∇ln|Ψ|
        ones = torch.ones_like(log_abs_psi)
        grad_log_psi = torch.autograd.grad(
            outputs=log_abs_psi,
            inputs=r,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N, 3)

        grad_squared = torch.sum(grad_log_psi ** 2, dim=(-1, -2))  # (B,)

        # 3. 拉普拉斯 ∇²ln|Ψ| (逐维精确二阶导)
        grad_flat = grad_log_psi.reshape(batch_size, -1)
        dim_total = grad_flat.shape[1]
        laplacian = torch.zeros(batch_size, device=r.device, dtype=r.dtype)

        for d in range(dim_total):
            g_d = grad_flat[:, d]
            sec_retain = True if create_graph else (d < dim_total - 1)
            sec_grad = torch.autograd.grad(
                outputs=g_d,
                inputs=r,
                grad_outputs=ones,
                create_graph=create_graph,
                retain_graph=sec_retain,
            )[0]
            sec_grad_flat = sec_grad.reshape(batch_size, -1)
            laplacian = laplacian + sec_grad_flat[:, d]
            del sec_grad, sec_grad_flat
        del grad_flat

        # 4. 动能: T_L = -0.5 * (∇²ln|Ψ| + |∇ln|Ψ|²)
        t_local = -0.5 * (laplacian + grad_squared)

        # 5. 势能: V(r)
        if create_graph:
            v_local = system.compute_potential_energy(r)
        else:
            with torch.no_grad():
                v_local = system.compute_potential_energy(r)

        e_local = t_local + v_local
        return e_local, t_local, v_local


def compute_kinetic_energy(
    model: nn.Module,
    r: torch.Tensor,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算局域动能及分量:
        T_L(r) = -0.5 * (∇²ln|Ψ| + |∇ln|Ψ||²)

    Args:
        model: 波函数模型
        r: 电子坐标, Shape: (B, N, 3), 必须 requires_grad=True
        create_graph: 是否保留二阶计算图

    Returns:
        t_local: 局域动能, Shape: (B,)
        laplacian: 对数波函数拉普拉斯 ∇²ln|Ψ|, Shape: (B,)
        grad_squared: 一阶梯度平方模 |∇ln|Ψ||², Shape: (B,)
    """
    batch_size = r.shape[0]
    assert r.requires_grad, "输入 r 必须开启梯度追踪"

    freeze_ctx = _FreezeModelParameters(model) if not create_graph else nullcontext()
    with freeze_ctx:
        log_abs_psi, _ = model.forward_log_psi(r)
        ones = torch.ones_like(log_abs_psi)

        grad_log_psi = torch.autograd.grad(
            outputs=log_abs_psi,
            inputs=r,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N, 3)

        grad_squared = torch.sum(grad_log_psi ** 2, dim=(-1, -2))  # (B,)

        grad_flat = grad_log_psi.reshape(batch_size, -1)
        dim_total = grad_flat.shape[1]
        laplacian = torch.zeros(batch_size, device=r.device, dtype=r.dtype)

        for d in range(dim_total):
            g_d = grad_flat[:, d]
            sec_retain = True if create_graph else (d < dim_total - 1)
            sec_grad = torch.autograd.grad(
                outputs=g_d,
                inputs=r,
                grad_outputs=ones,
                create_graph=create_graph,
                retain_graph=sec_retain,
            )[0]
            sec_grad_flat = sec_grad.reshape(batch_size, -1)
            laplacian = laplacian + sec_grad_flat[:, d]
            del sec_grad, sec_grad_flat
        del grad_flat

        t_local = -0.5 * (laplacian + grad_squared)
        return t_local, laplacian, grad_squared


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
