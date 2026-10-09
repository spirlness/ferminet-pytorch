"""
FermiNet 原子核指数包络衰减模块。

为每个行列式分支构建指数衰减基底，确保波函数在远离原子核时指数衰减:
    Envelope_{k,i,j}(r_i) = sum_m pi_{k,j,m} * exp(-alpha_{k,j,m} * |r_i - R_m|)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NuclearEnvelope(nn.Module):
    """
    各向同性指数衰减核包络。

    参数:
        pi (weights): 可学习线性组合权重, Shape: (K, N_orb, M)
        alpha (decay_rates): 可学习衰减速率 (通过 softplus 保正), Shape: (K, N_orb, M)
    """

    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        num_nuclei: int,
        eps: float = 1e-24,
        init_decay_rate: float = 1.0,
    ) -> None:
        """
        Args:
            num_determinants: 行列式数量 K
            num_orbitals: 该自旋通道的轨道数
            num_nuclei: 原子核数目 M
            eps: 最小平方距离
            init_decay_rate: 初始衰减速率
        """
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.num_nuclei = num_nuclei
        self.eps = eps

        self.weights = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, num_nuclei)
        )
        self.decay_rates_raw = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, num_nuclei)
        )

        self._reset_parameters(init_decay_rate)

    def _reset_parameters(self, init_decay_rate: float) -> None:
        nn.init.normal_(self.weights, mean=1.0, std=0.1)
        inv_val = torch.log(torch.exp(torch.tensor(init_decay_rate)) - 1.0).item()
        nn.init.constant_(self.decay_rates_raw, inv_val)

    def forward(
        self,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算指数包络衰减值。

        Args:
            r: 电子坐标, Shape: (B, N_spin, 3)
            nuclear_coords: 核坐标, Shape: (M, 3)

        Returns:
            envelope: Shape: (B, K, N_spin, N_orb)
        """
        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        diff = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)  # (B, N_spin, M, 3)

        # 欧氏距离
        dist = torch.sqrt(torch.clamp(
            torch.sum(diff ** 2, dim=-1), min=self.eps
        ))  # (B, N_spin, M)

        # softplus 保证衰减率为正
        alpha = F.softplus(self.decay_rates_raw)  # (K, N_orb, M)

        dist_expanded = dist.unsqueeze(1).unsqueeze(3)      # (B, 1, N_spin, 1, M)
        alpha_expanded = alpha.unsqueeze(0).unsqueeze(2)     # (1, K, 1, N_orb, M)
        weights_expanded = self.weights.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)

        exponent = torch.exp(-alpha_expanded * dist_expanded)  # (B, K, N_spin, N_orb, M)
        envelope = torch.sum(weights_expanded * exponent, dim=-1)  # (B, K, N_spin, N_orb)

        return envelope
