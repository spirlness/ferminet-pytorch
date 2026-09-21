from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class NuclearEnvelope(nn.Module):
    """
    FermiNet 原子核指数包络衰减模块 (Nuclear Exponential Envelope)。

    在量子力学束缚态中，电子远离原子核时 (r -> inf)，体系波函数必须指数衰减至 0。
    本模块为每个行列式分支构建指数衰减基底：
    1. 各向同性模式 (anisotropic=False, 默认兼容模式):
        Envelope_{k, i, j}(r_i) = sum_{m=1}^M pi_{k, j, m} * exp(-alpha_{k, j, m} * |r_i - R_m|)
    2. 各向异性模式 (anisotropic=True, 提升非球对称成键轨道能力):
        学习三维主轴度规 Sigma_{k, j, m}^d:
        Envelope_{k, i, j}(r_i) = sum_{m=1}^M pi_{k, j, m} * exp(-sqrt(sum_{d=1}^3 (sigma_{k, j, m, d} * (r_{i, d} - R_{m, d}))^2 + eps))
    其中：
        - k: 行列式索引 (1 <= k <= num_determinants)
        - i: 电子索引 (1 <= i <= num_electrons)
        - j: 轨道索引 (1 <= j <= num_orbitals)
        - m: 原子核索引 (1 <= m <= num_nuclei)
        - pi_{k, j, m}: 可学习包络线性组合权重
        - alpha / sigma > 0: 可学习衰减速率/主轴度规参数 (通过 softplus 保证严格为正，避免发散)
    """

    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        num_nuclei: int,
        eps: float = 1e-8,
        init_decay_rate: float = 1.0,
        anisotropic: bool = False,
    ) -> None:
        """
        初始化原子核指数包络模块。

        Args:
            num_determinants (int): 行列式数量 (K)。
            num_orbitals (int): 该自旋通道对应的分子轨道数 (通常等于该自旋电子数)。
            num_nuclei (int): 原子核数目 (M)。
            eps (float): 防止距离计算除零的微小量。
            init_decay_rate (float): 衰减速率初始值 (默认为 1.0)。
            anisotropic (bool): 是否启用各向异性可学习三维主轴度规 (默认为 False 各向同性)。
        """
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.num_nuclei = num_nuclei
        self.eps = eps
        self.anisotropic = anisotropic

        # 可学习组合系数 pi: shape (num_determinants, num_orbitals, num_nuclei)
        self.weights = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, num_nuclei)
        )
        if self.anisotropic:
            # 各向异性: 三维主轴可学习度规参数，shape (K, N_orb, M, 3)
            self.decay_rates_raw = nn.Parameter(
                torch.empty(num_determinants, num_orbitals, num_nuclei, 3)
            )
        else:
            # 各向同性: 标量衰减参数，shape (K, N_orb, M)
            self.decay_rates_raw = nn.Parameter(
                torch.empty(num_determinants, num_orbitals, num_nuclei)
            )

        self._reset_parameters(init_decay_rate)

    def _reset_parameters(self, init_decay_rate: float) -> None:
        """参数初始化."""
        assert init_decay_rate > 0.0, f"初始衰减速率必须严格大于 0，当前为: {init_decay_rate}"
        nn.init.normal_(self.weights, mean=1.0, std=0.1)
        # softplus_inv(x) = log(exp(x) - 1)
        inv_val = torch.log(torch.exp(torch.tensor(init_decay_rate)) - 1.0).item()
        nn.init.constant_(self.decay_rates_raw, inv_val)

    def forward(
        self,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向计算所有电子在所有轨道和行列式下的指数包络衰减值。

        Args:
            r (torch.Tensor): 指定自旋通道的电子空间三维坐标。
                - Shape: `(B, N_spin, 3)`，其中 B 为 Batch 大小，N_spin 为该自旋电子数。
                - Dtype: `torch.float32` 或 `torch.float64`。
            nuclear_coords (torch.Tensor): 原子核空间坐标。
                - Shape: `(M, 3)`。
                - Dtype: 与 r 一致。

        Returns:
            torch.Tensor: 指数衰减包络张量。
                - Shape: `(B, num_determinants, N_spin, num_orbitals)`。
                - Dtype: 与 r 一致。
        """
        batch_size, n_spin, _ = r.shape
        num_nuclei, _ = nuclear_coords.shape
        assert num_nuclei == self.num_nuclei, (
            f"原子核数目不匹配: 输入 {num_nuclei} != 模块定义 {self.num_nuclei}"
        )

        # ---------------------------------------------------------------------
        # 1. 计算电子与所有核之间的相对位移
        # r: (B, N_spin, 1, 3), nuclear_coords: (1, 1, M, 3)
        # diff: (B, N_spin, M, 3)
        # ---------------------------------------------------------------------
        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        diff = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)

        if self.anisotropic:
            # -----------------------------------------------------------------
            # 2A. 各向异性模式: 三维主轴度规 sigma: (K, N_orb, M, 3)
            # -----------------------------------------------------------------
            sigma = F.softplus(self.decay_rates_raw)  # (K, N_orb, M, 3)
            diff_expanded = diff.unsqueeze(1).unsqueeze(3)  # (B, 1, N_spin, 1, M, 3)
            sigma_expanded = sigma.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M, 3)

            scaled_diff = sigma_expanded * diff_expanded  # (B, K, N_spin, N_orb, M, 3)
            dist_aniso = torch.sqrt(
                torch.sum(scaled_diff ** 2, dim=-1) + self.eps
            )  # (B, K, N_spin, N_orb, M)

            exponent = torch.exp(-dist_aniso)  # (B, K, N_spin, N_orb, M)
            weights_expanded = self.weights.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)
            envelope = torch.sum(weights_expanded * exponent, dim=-1)  # (B, K, N_spin, N_orb)
        else:
            # -----------------------------------------------------------------
            # 2B. 各向同性兼容模式: 欧氏标量距离与各向同性衰减率 alpha: (K, N_orb, M)
            # -----------------------------------------------------------------
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + self.eps)  # (B, N_spin, M)
            alpha = F.softplus(self.decay_rates_raw)  # (K, N_orb, M)

            dist_expanded = dist.unsqueeze(1).unsqueeze(3)  # (B, 1, N_spin, 1, M)
            alpha_expanded = alpha.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)
            weights_expanded = self.weights.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)

            exponent = torch.exp(-alpha_expanded * dist_expanded)  # (B, K, N_spin, N_orb, M)
            envelope = torch.sum(weights_expanded * exponent, dim=-1)  # (B, K, N_spin, N_orb)

        return envelope
