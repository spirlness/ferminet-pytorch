"""
量子化学分子体系定义与局域能量计算。

- MolecularSystem: 分子体系的几何与电子构型
- compute_local_energy: 基于自动微分的局域能量计算
    E_L(r) = T_L(r) + V(r)
    T_L(r) = -0.5 * (∇²ln|Ψ| + |∇ln|Ψ||²)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
import torch.nn as nn


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
    _v_nn_const: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.nuclear_coords.ndim == 2 and self.nuclear_coords.shape[1] == 3
        assert self.nuclear_charges.ndim == 1
        assert self.n_up >= 0 and self.n_down >= 0
        assert self.n_up + self.n_down > 0
        self._compute_v_nn_const()

    def _compute_v_nn_const(self) -> torch.Tensor:
        """预计算核-核库仑排斥常数 V_nn。"""
        num_nuclei = self.nuclear_coords.shape[0]
        if num_nuclei > 1:
            row_idx, col_idx = torch.triu_indices(
                num_nuclei, num_nuclei, offset=1, device=self.nuclear_coords.device
            )
            diff = self.nuclear_coords[row_idx] - self.nuclear_coords[col_idx]
            dist = torch.linalg.vector_norm(diff, dim=-1)
            charge_prod = self.nuclear_charges[row_idx] * self.nuclear_charges[col_idx]
            v_nn_const = torch.sum(charge_prod / dist)
        else:
            v_nn_const = torch.tensor(
                0.0, device=self.nuclear_coords.device, dtype=self.nuclear_coords.dtype
            )
        self._v_nn_const = v_nn_const
        return v_nn_const

    @property
    def v_nn(self) -> torch.Tensor:
        if self._v_nn_const is None:
            return self._compute_v_nn_const()
        return self._v_nn_const

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
        if self._v_nn_const is not None:
            self._v_nn_const = self._v_nn_const.to(*args, **kwargs)
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
        dist_en = torch.sqrt(torch.clamp(torch.sum(diff_en ** 2, dim=-1), min=eps))
        charges = nuclear_charges.unsqueeze(0).unsqueeze(0)
        v_en = -torch.sum(charges / dist_en, dim=(-1, -2))

        # 电子-电子排斥: V_ee = sum_{i<j} 1/|r_i - r_j|
        if num_electrons > 1:
            diff_ee = r.unsqueeze(2) - r.unsqueeze(1)
            dist_ee = torch.sqrt(torch.clamp(torch.sum(diff_ee ** 2, dim=-1), min=eps))
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

    # 无需为参数保留图时，临时关闭参数梯度
    prev_param_states = []
    if not create_graph:
        for p in model.parameters():
            prev_param_states.append(p.requires_grad)
            p.requires_grad = False

    try:
        # 1. log|Psi(r)|
        log_abs_psi, _ = model.forward_log_psi(r)

        # 2. 一阶梯度
        ones = torch.ones_like(log_abs_psi)
        grad_log_psi = torch.autograd.grad(
            outputs=log_abs_psi,
            inputs=r,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N, 3)

        grad_squared = torch.sum(grad_log_psi ** 2, dim=(-1, -2))  # (B,)

        # 3. 拉普拉斯 (逐维精确二阶导)
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

        # 4. 动能
        t_local = -0.5 * (laplacian + grad_squared)

        # 5. 势能
        if create_graph:
            v_local = system.compute_potential_energy(r)
        else:
            with torch.no_grad():
                v_local = system.compute_potential_energy(r)

        e_local = t_local + v_local

    finally:
        if not create_graph:
            for p, state in zip(model.parameters(), prev_param_states):
                p.requires_grad = state

    return e_local, t_local, v_local
