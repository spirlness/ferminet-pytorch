"""
ferminet_pytorch.geometry
~~~~~~~~~~~~~~~~~~~~~~~~~
全局单批次不可变几何上下文容器，提供集中式、处处可微的几何不变量抽取。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


def smooth_distance(
    r_diff: torch.Tensor,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    计算处处可微的平滑欧氏距离 sqrt(sum(r^2) + eps)。

    Args:
        r_diff: 坐标差向量, 最后一维为 3 (x, y, z)
        eps: 平滑化正则项 (默认 1e-12)
        keepdim: 是否保留缩减维度 (默认 False)

    Returns:
        平滑距离张量
    """
    sum_sq = torch.sum(r_diff ** 2, dim=-1, keepdim=keepdim)
    return torch.sqrt(sum_sq + eps)


@dataclass(frozen=True)
class GeometricContext:
    """
    单批次电子构型的全局共享几何上下文 (不可变张量容器)。

    Attributes:
        r: 电子笛卡尔坐标, shape (B, N, 3), dtype torch.float32/float64
        r_en: 电子-原子核相对位移向量, shape (B, N, M, 3)
        d_en: 平滑电子-原子核欧氏距离, shape (B, N, M, 1)
        r_ee: 电子-电子相对位移向量, shape (B, N, N, 3), N=1 时为 None
        d_ee: 平滑电子-电子欧氏距离, shape (B, N, N, 1), N=1 时为 None
    """
    r: torch.Tensor
    r_en: torch.Tensor
    d_en: torch.Tensor
    r_ee: Optional[torch.Tensor] = None
    d_ee: Optional[torch.Tensor] = None

    @classmethod
    def compute(
        cls,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
        eps: float = 1e-12,
        compute_ee: bool = True,
    ) -> "GeometricContext":
        """
        单次前向执行处处可微的向量化几何抽取。

        Args:
            r: 电子坐标张量, shape (B, N, 3), dtype torch.float32 或 float64
            nuclear_coords: 原子核固定坐标, shape (M, 3)
            eps: 平滑范数微小量，防止 r -> 0 时导数出现 0/0=NaN
            compute_ee: 是否构建电子对张量

        Returns:
            不可变全局几何上下文 GeometricContext 实例
        """
        batch_size, num_electrons, _ = r.shape
        num_nuclei, _ = nuclear_coords.shape

        # 1. 电子-原子核几何 (B, N, M, 3) 与 (B, N, M, 1)
        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        r_expanded = r.unsqueeze(2)                              # (B, N, 1, 3)
        nuc_expanded = coords.unsqueeze(0).unsqueeze(0)          # (1, 1, M, 3)
        r_en = r_expanded - nuc_expanded                         # (B, N, M, 3)
        d_en = smooth_distance(r_en, eps=eps, keepdim=True)      # (B, N, M, 1)

        # 2. 电子-电子几何 (B, N, N, 3) 与 (B, N, N, 1)
        r_ee: Optional[torch.Tensor] = None
        d_ee: Optional[torch.Tensor] = None
        if compute_ee and num_electrons > 1:
            r_i = r.unsqueeze(2)                                 # (B, N, 1, 3)
            r_j = r.unsqueeze(1)                                 # (B, 1, N, 3)
            r_ee = r_i - r_j                                     # (B, N, N, 3)
            d_ee = smooth_distance(r_ee, eps=eps, keepdim=True)  # (B, N, N, 1)

        return cls(r=r, r_en=r_en, d_en=d_en, r_ee=r_ee, d_ee=d_ee)
