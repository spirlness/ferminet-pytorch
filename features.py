from typing import Optional, Tuple
import torch
import torch.nn as nn


class ElectronFeatureEmbedding(nn.Module):
    """
    FermiNet 输入特征嵌入模块。
    
    提取两类物理几何特征：
    1. 单电子-原子核相对位移及欧几里得距离特征 (One-electron stream inputs)，
       支持可选注入原子核电荷标度 Z_I：
       - 当注入核电荷时：单核特征为 5 维 [r_{iI}, d_{iI}, Z_I]，总维度为 5 * M；
       - 当不注入核电荷时：单核特征为 4 维 [r_{iI}, d_{iI}]，总维度为 4 * M。
    2. 电子-电子相对位移及欧几里得距离特征 (Two-electron stream inputs)。
    
    全量计算采用广播张量操作，严格无显式 Python 循环。
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """
        初始化特征嵌入模块。

        Args:
            eps (float): 防止计算欧氏距离平方根时出现零除或零处梯度不可导的数值微小量。
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向提取单电子与双电子相对几何特征。

        Args:
            r (torch.Tensor): 电子空间三维坐标。
                - Shape: `(B, N, 3)`，其中 B 为批量样本大小 (MCMC 采样 walkers 数量)，
                  N 为体系总电子数目 (N = N_up + N_down)，3 为 (x, y, z) 空间坐标。
                - Dtype: `torch.float32` 或 `torch.float64`。
            nuclear_coords (torch.Tensor): 原子核空间三维坐标。
                - Shape: `(M, 3)`，其中 M 为体系原子核总数。
                - Dtype: 与 r 一致。
            nuclear_charges (Optional[torch.Tensor]): 可选的原子核电荷量 (原子序数 Z_I)。
                - Shape: `(M,)` 或 `(M, 1)`。
                - Dtype: 与 r 一致。若提供，则单电子特征拼接 Z_I 扩展为 5*M 维；
                  若为 None，则保持向后兼容的 4*M 维。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - h1_input (torch.Tensor): 单电子流初始特征。
                    - Shape: 若 nuclear_charges is not None 为 `(B, N, 5 * M)`，
                             否则为 `(B, N, 4 * M)`。
                    - Dtype: 与 r 一致。
                - h2_input (torch.Tensor): 双电子流初始特征。
                    - Shape: `(B, N, N, 4)`，每对电子之间的相对位移向量 (3维) 与相对距离标量 (1维) 的拼接。
                    - Dtype: 与 r 一致。
        """
        batch_size, num_electrons, _ = r.shape
        num_nuclei, _ = nuclear_coords.shape

        # ---------------------------------------------------------------------
        # 1. 单电子-核相对位移与欧几里得距离计算
        # r 形状扩展为 (B, N, 1, 3)；nuclear_coords 扩展为 (1, 1, M, 3)
        # ---------------------------------------------------------------------
        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        r_expanded = r.unsqueeze(2)  # (B, N, 1, 3)
        nuclei_expanded = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, M, 3)

        # 相对位移向量: r_{iI} = r_i - R_I, shape: (B, N, M, 3)
        r_en = r_expanded - nuclei_expanded

        # 相对距离标量: d_{iI} = sqrt(|r_{iI}|^2 + eps), shape: (B, N, M, 1)
        d_en = torch.sqrt(torch.sum(r_en ** 2, dim=-1, keepdim=True) + self.eps)

        # 拼接单核特征: 若提供 nuclear_charges 则拼接 [r_{iI}, d_{iI}, Z_I] (5 维)
        if nuclear_charges is not None:
            charges = nuclear_charges.to(device=r.device, dtype=r.dtype)
            assert charges.numel() == num_nuclei, (
                f"nuclear_charges 元素数 ({charges.numel()}) 必须与原子核数 ({num_nuclei}) 一致"
            )
            charges = charges.reshape(num_nuclei, 1)  # (M, 1)
            charges_expanded = charges.unsqueeze(0).unsqueeze(0).expand(
                batch_size, num_electrons, num_nuclei, 1
            )  # (B, N, M, 1)
            h1_raw = torch.cat([r_en, d_en, charges_expanded], dim=-1)  # (B, N, M, 5)
            h1_input = h1_raw.reshape(batch_size, num_electrons, 5 * num_nuclei)
        else:
            # 兼容模式: [r_{iI}, d_{iI}], shape: (B, N, M, 4)
            h1_raw = torch.cat([r_en, d_en], dim=-1)
            h1_input = h1_raw.reshape(batch_size, num_electrons, 4 * num_nuclei)

        # ---------------------------------------------------------------------
        # 2. 电子-电子相对位移与欧几里得距离计算
        # r_i 扩展为 (B, N, 1, 3)；r_j 扩展为 (B, 1, N, 3)
        # ---------------------------------------------------------------------
        r_i = r.unsqueeze(2)  # (B, N, 1, 3)
        r_j = r.unsqueeze(1)  # (B, 1, N, 3)

        # 相对位移向量: r_{ij} = r_i - r_j, shape: (B, N, N, 3)
        r_ee = r_i - r_j

        # 相对距离标量: d_{ij} = sqrt(|r_{ij}|^2 + eps), shape: (B, N, N, 1)
        d_ee = torch.sqrt(torch.sum(r_ee ** 2, dim=-1, keepdim=True) + self.eps)

        # 拼接位移与距离特征: [r_{ij}, d_{ij}], shape: (B, N, N, 4)
        h2_input = torch.cat([r_ee, d_ee], dim=-1)

        return h1_input, h2_input
