"""
Fermi 输入特征嵌入模块。

提取两类物理几何特征:
1. 单电子-原子核相对位移与距离特征
2. 电子-电子相对位移与距离特征
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

def smooth_distance(
    r_diff: torch.Tensor,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    计算处处可微的平滑欧氏距离 sqrt(sum(r^2) + eps)。

    严正声明：仅用于几何特征与库仑势能平滑，严禁用于 KatoCuspJastrow 物理尖点！

    Args:
        r_diff: 坐标差向量, 最后一维为 3 (x, y, z)
        eps: 平滑化正则项 (默认 1e-12)
        keepdim: 是否保留缩减维度 (默认 False)

    Returns:
        平滑距离张量
    """
    sum_sq = torch.sum(r_diff ** 2, dim=-1, keepdim=keepdim)
    return torch.sqrt(sum_sq + eps)


class ElectronFeatureEmbedding(nn.Module):
    """
    电子几何特征提取。

    单核特征: [r_{iI}(3), d_{iI}, Z_I] (5维), 所有核拼接
    双电子特征: [r_{ij}(3), d_{ij}] (4维)
    """

    def __init__(self, eps: float = 1e-24, smooth_cuspless: bool = True) -> None:
        """
        Args:
            eps: 最小平方距离, 防止 sqrt(0) 的梯度为 NaN
            smooth_cuspless: 是否使用平滑无尖峰距离特征 d^2 / (1 + d),
                             消除距离在近核/短程处的 1/r 拉普拉斯奇异性
        """
        super().__init__()
        self.eps = eps
        self.smooth_cuspless = smooth_cuspless

    @staticmethod
    def feature_dims(
        num_nuclei: int,
        use_nuclear_charges: bool = True,
    ) -> Tuple[int, int]:
        """返回 (单电子流输入维度, 双电子流输入维度)。"""
        one = (5 if use_nuclear_charges else 4) * num_nuclei
        two = 4
        return one, two

    def forward(
        self,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取单电子与双电子几何特征。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            nuclear_coords: 原子核坐标, Shape: (M, 3)
            nuclear_charges: 原子核电荷, Shape: (M,), 可选

        Returns:
            h1_input: 单电子流特征, Shape: (B, N, 5*M) 或 (B, N, 4*M)
            h2_input: 双电子流特征, Shape: (B, N, N, 4)
        """
        batch_size, num_electrons, _ = r.shape
        num_nuclei, _ = nuclear_coords.shape

        # 1. 电子-核特征
        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        r_expanded = r.unsqueeze(2)                                 # (B, N, 1, 3)
        nuclei_expanded = coords.unsqueeze(0).unsqueeze(0)          # (1, 1, M, 3)

        r_en = r_expanded - nuclei_expanded                         # (B, N, M, 3)
        d_en = smooth_distance(r_en, eps=self.eps, keepdim=True)    # (B, N, M, 1)

        d_en_feature = (d_en ** 2) / (1.0 + d_en) if self.smooth_cuspless else d_en

        if nuclear_charges is not None:
            charges = nuclear_charges.to(device=r.device, dtype=r.dtype)
            charges_expanded = charges.reshape(1, 1, num_nuclei, 1).expand(
                batch_size, num_electrons, num_nuclei, 1
            )
            parts = [r_en, d_en_feature, charges_expanded]
        else:
            parts = [r_en, d_en_feature]

        h1_raw = torch.cat(parts, dim=-1)                           # (B, N, M, K)
        h1_input = h1_raw.reshape(batch_size, num_electrons, h1_raw.shape[-1] * num_nuclei)

        # 2. 电子-电子特征
        r_i = r.unsqueeze(2)   # (B, N, 1, 3)
        r_j = r.unsqueeze(1)   # (B, 1, N, 3)
        r_ee = r_i - r_j       # (B, N, N, 3)
        d_ee = smooth_distance(r_ee, eps=self.eps, keepdim=True)    # (B, N, N, 1)

        d_ee_feature = (d_ee ** 2) / (1.0 + d_ee) if self.smooth_cuspless else d_ee
        h2_input = torch.cat([r_ee, d_ee_feature], dim=-1)  # (B, N, N, 4)

        return h1_input, h2_input
