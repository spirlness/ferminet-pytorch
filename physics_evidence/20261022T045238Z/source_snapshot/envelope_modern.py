"""
ferminet_pytorch.envelope_modern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于共享几何上下文复用与权重 100% 向后兼容的高性能原子核衰减包络模块。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class NuclearEnvelopeModern(nn.Module):
    """
    原子核指数包络模块 (上下文共享与物理兼容版)。

    优化机理与技术澄清:
    1. 消除几何重复计算: 直接接收 GeometricContext 预计算的平滑标量距离 dist_en，
       彻底消除原实现在每个包络分支内对坐标差分 (B, N, M, 3) 和开方运算的重复执行；
    2. 广播与收缩机制澄清: 由于非线性指数项 exp(-alpha * d) 的存在，中间张量在
       局部广播空间 (B, N, K, N_orb, M) 展开是不可避免的。优化重点在于免除冗余输入
       几何分配，并在指数求值后高效沿原子核维度收缩输出 (B, K, N, N_orb)；
    3. 100% Checkpoint 兼容: 保持参数命名 decay_rates_raw 与 weights，保持 F.softplus
       物理保正，恢复核电荷 nuclear_charges 物理对齐初始化，确保旧权重无缝加载。
    """
    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        nuclear_coords: torch.Tensor,
        init_decay_rate: float = 1.0,
        envelope_type: str = "exponential",
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.num_nuclei = nuclear_coords.shape[0]
        self.envelope_type = envelope_type.lower()

        # 保持与原有 Checkpoint 100% 兼容的参数定义 (彻底修复 BUG-03)
        self.weights = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, self.num_nuclei)
        )
        self.decay_rates_raw = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, self.num_nuclei)
        )

        # 物理对齐参数重置
        self._reset_parameters(init_decay_rate, nuclear_charges=nuclear_charges)

    def _reset_parameters(
        self,
        init_decay_rate: float,
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        """初始化权重与衰减率，支持核电荷物理对齐。"""
        nn.init.normal_(self.weights, mean=1.0, std=0.1)
        if nuclear_charges is not None:
            charges = nuclear_charges.to(
                device=self.decay_rates_raw.device, dtype=self.decay_rates_raw.dtype
            )
            charges = torch.clamp(charges, min=1e-4)
            inv_vals = torch.where(charges > 20.0, charges, torch.log(torch.expm1(charges)))
            with torch.no_grad():
                self.decay_rates_raw.data.copy_(
                    inv_vals.reshape(1, 1, self.num_nuclei).expand_as(self.decay_rates_raw)
                )
        else:
            inv_val = torch.log(torch.exp(torch.tensor(init_decay_rate)) - 1.0).item()
            nn.init.constant_(self.decay_rates_raw, inv_val)

    def forward(
        self,
        dist_en: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dist_en: 电子-核平滑标量距离 (来自 GeometricContext),
                     shape: (B, N, M) 或 (B, N, M, 1)

        Returns:
            envelope: shape (B, K, N, num_orbitals), dtype 与 dist_en 一致
        """
        if dist_en.dim() == 4:
            dist_en = dist_en.squeeze(-1)  # (B, N, M)

        # 1. softplus 保证衰减速率为正，保持历史模型绝对数值等价
        alpha = F.softplus(self.decay_rates_raw)  # (K, N_orb, M)

        B, N, M = dist_en.shape
        K, O, _ = alpha.shape

        # 2. 局部广播求指数衰减
        dist_expanded = dist_en.unsqueeze(2).unsqueeze(3)      # (B, N, 1, 1, M)
        alpha_expanded = alpha.view(1, 1, K, O, M)            # (1, 1, K, O, M)
        weights_expanded = self.weights.view(1, 1, K, O, M)   # (1, 1, K, O, M)

        if self.envelope_type == "gaussian":
            decay = torch.exp(-alpha_expanded * (dist_expanded ** 2))
        else:
            decay = torch.exp(-alpha_expanded * dist_expanded)

        # 3. 沿核维度收缩输出 (B, N, K, O)
        envelope = torch.sum(weights_expanded * decay, dim=-1)

        # 4. 转置对齐标准输出形状: (B, K, N, N_orb)
        return envelope.permute(0, 2, 1, 3).contiguous()
