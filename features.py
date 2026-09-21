from typing import Optional, Tuple
import torch
import torch.nn as nn


class ElectronFeatureEmbedding(nn.Module):
    """
    Fermi 输入特征嵌入模块。

    提取两类物理几何特征：
    1. 单电子-原子核相对位移及欧几里得距离特征 (One-electron stream inputs)，
       支持可选注入原子核电荷标度 Z_I：
       - 当注入核电荷时：单核特征为 5 维 [r_{iI}, d_{iI}, Z_I]，总维度为 5 * M；
       - 当不注入核电荷时：单核特征为 4 维 [r_{iI}, d_{iI}]，总维度为 4 * M。
    2. 电子-电子相对位移及欧几里得距离特征 (Two-electron stream inputs)。

    全量计算采用广播张量操作，严格无显式 Python 循环。

    已知限制 (如实声明，勿作更强声明)：
    ------------------------------------------------
    1. 不提供旋转不变性。单电子特征包含笛卡尔位移向量 r_{iI} 与 r_{ij} (各 3 维)，
       双电子特征包含 r_{ij}，它们经普通密集层 (nn.Linear) 混合，无 SO(3) 等变约束；
       整体旋转输入会改变输出特征。旋转等变是一项独立的架构工程，本阶段不实现
       (见 test_physics_invariants.py 的 D4 已知限制测试)。
    2. **不具备核置换不变性**。单电子特征按原子核顺序拼接为 5*M 维向量后送入
       普通密集层，而密集层对核块并不对称，因此交换两个同种原子核会改变网络输出。
       注：这一项**无法**通过"按距离排序后再拼接"来修复——排序在电子穿越两同核的
       垂直平分面时会翻转核块顺序，而两块在该面上并不相等 (r_{iI} 方向不同)，
       于是波函数在该余维 1 曲面上出现**不连续跳变** (实测 log|Psi| 跳变约 0.044)，
       使 ∇²ln|Psi| 失去意义并污染局域动能。要做到既不变又连续，必须改为对核做
       **对称聚合** (如按核求和/求均值)，那是一次架构改写，不在本阶段范围内。
       因此当前实现只保证：**包络部分** (NuclearEnvelope 按电荷组共享参数) 在同核
       置换下不变，而完整波函数不是。核置换不变性的回归测试见
       test_physics_invariants.py 的已知限制测试。
    """

    def __init__(self, eps: float = 1e-24) -> None:
        """
        初始化特征嵌入模块。

        Args:
            eps (float): 最小**平方**距离 (单位 Bohr^2)。距离计算为
                sqrt(clamp(r^2, min=eps))，即软化半径为 sqrt(eps) Bohr
                (默认 1e-24 对应 1e-12 Bohr)。
                语义说明：eps 不是加性软化项——clamp 在所有 r^2 >= eps 处是恒等变换，
                距离与未软化时逐位相同；仅在 r^2 < eps 的重合邻域内距离被限制为
                sqrt(eps)，有限且梯度为零。不能直接删除 eps：裸 sqrt(r^2) 在
                r = 0 处梯度为 inf * 0 = NaN (见 test_geometry_invariants.py)。
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

        核块按**原始核顺序**拼接，不做规范化重排 (见类 docstring 已知限制第 2 条)。

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

        # 相对距离标量: d_{iI} = sqrt(clamp(|r_{iI}|^2, min=eps)), shape: (B, N, M, 1)
        # D6: clamp 在所有 r^2 >= eps 处恒等 (与未软化逐位相同, 零偏置)；
        # r^2 < eps 时距离限制为 sqrt(eps), 有限且梯度为零。
        d_en = torch.sqrt(torch.clamp(torch.sum(r_en ** 2, dim=-1, keepdim=True), min=self.eps))

        if nuclear_charges is not None:
            charges = nuclear_charges.to(device=r.device, dtype=r.dtype)
            assert charges.numel() == num_nuclei, (
                f"nuclear_charges 元素数 ({charges.numel()}) 必须与原子核数 ({num_nuclei}) 一致"
            )
            charges_flat = charges.reshape(num_nuclei)  # (M,)
            charges_expanded = charges_flat.reshape(1, 1, num_nuclei, 1).expand(
                batch_size, num_electrons, num_nuclei, 1
            )  # (B, N, M, 1)

            # 拼接单核特征: [r_{iI}, d_{iI}, Z_I] (5 维), 按**原始核顺序**排列。
            # 核轴不做规范化重排——原因见类 docstring 的"已知限制"第 2 条。
            h1_raw = torch.cat([r_en, d_en, charges_expanded], dim=-1)  # (B, N, M, 5)
            h1_input = h1_raw.reshape(batch_size, num_electrons, 5 * num_nuclei)
        else:
            # 兼容模式: [r_{iI}, d_{iI}], shape: (B, N, M, 4), 不做核轴重排
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

        # 相对距离标量: d_{ij} = sqrt(clamp(|r_{ij}|^2, min=eps)), shape: (B, N, N, 1)
        # D6: 同 d_en 的 clamp 核语义。
        d_ee = torch.sqrt(torch.clamp(torch.sum(r_ee ** 2, dim=-1, keepdim=True), min=self.eps))

        # 拼接位移与距离特征: [r_{ij}, d_{ij}], shape: (B, N, N, 4)
        h2_input = torch.cat([r_ee, d_ee], dim=-1)

        return h1_input, h2_input
