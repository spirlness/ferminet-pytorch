"""
Kato 尖峰解析 Jastrow 因子模块。

严格满足量子力学中的 Kato 尖峰边界条件 (Kato Cusp Conditions):
1. 电子-核碰撞 (r_{iI} -> 0):
   d ln|Psi| / d r_{iI} -> -Z_I
2. 电子-电子碰撞 (r_{ij} -> 0):
   - 反平行自旋对 (异自旋): d ln|Psi| / d r_{ij} -> 1/2
   - 平行自旋对 (同自旋):   d ln|Psi| / d r_{ij} -> 1/4

采用经典的 Padé 阻尼形式:
   U_en(r) = - sum_{i, I} (Z_I * r_{iI}) / (1 + b_{en, I} * r_{iI})
   U_ee(r) =   sum_{i < j} (gamma_{ij} * r_{ij}) / (1 + b_{ee, ij} * r_{ij})

在短程严格消除库仑奇异性发散，将神经网络从拟合剧烈尖峰中解放出来，
退化为仅需平滑拟合多体关联系数。
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class KatoCuspJastrow(nn.Module):
    """
    解析 Kato 尖峰 Jastrow 因子。

    计算:
        U(r) = U_en(r) + U_ee(r)
        J(r) = exp(U(r))

    属性:
        nuclear_charges: 原子核电荷 (M,)
        n_up: 自旋向上电子数
        n_down: 自旋向下电子数
        gamma_ee: 电子-电子尖峰系数矩阵 (N, N), 上三角非零
    """

    def __init__(
        self,
        nuclear_charges: torch.Tensor,
        n_up: int,
        n_down: int,
        init_b_ee_parallel: float = 1.0,
        init_b_ee_antiparallel: float = 1.0,
        init_b_en: float = 1.0,
        trainable: bool = True,
        cusp_en: bool = True,
        cusp_ee: bool = True,
        eps: float = 1e-24,
        two_electron_spin: Optional[int] = None,
    ) -> None:
        """
        初始化 Kato Cusp Jastrow 因子。

        Args:
            nuclear_charges: 原子核电荷向量, Shape: (M,)
            n_up: 自旋向上电子数
            n_down: 自旋向下电子数
            init_b_ee_parallel: 同自旋电子对阻尼因子初始值
            init_b_ee_antiparallel: 异自旋电子对阻尼因子初始值
            init_b_en: 电子-核阻尼因子初始值
            trainable: 阻尼参数是否作为可学习参数
            cusp_en: 是否启用电子-核尖峰
            cusp_ee: 是否启用电子-电子尖峰
            eps: 最小平方距离, 防止 sqrt(0) 的梯度为 NaN
            two_electron_spin: 双电子总自旋 S (0 或 1)。S=1 的空间反对称对
                使用 1/4 尖峰系数，包括 M_S=0 的上下自旋电子对。
        """
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        if two_electron_spin is not None:
            if self.total_electrons != 2 or two_electron_spin not in (0, 1):
                raise ValueError("two_electron_spin requires two electrons and S=0 or S=1")
            if two_electron_spin == 0 and n_up != n_down:
                raise ValueError("A two-electron singlet requires n_up=n_down=1")
        self.two_electron_spin = two_electron_spin
        self.num_nuclei = nuclear_charges.shape[0]
        self.cusp_en = cusp_en
        self.cusp_ee = cusp_ee
        self.eps = eps
        self.trainable = trainable

        # 注册非参数缓冲
        self.register_buffer("nuclear_charges", nuclear_charges.clone().detach())

        # 预计算电子-电子尖峰斜率矩阵 gamma_ee, Shape: (N, N)
        # 上三角元素 i < j:
        #   - 同自旋 (up-up 或 down-down): gamma = 1/4 = 0.25
        #   - 异自旋 (up-down):            gamma = 1/2 = 0.50
        n_tot = self.total_electrons
        gamma_mat = torch.zeros(n_tot, n_tot, dtype=nuclear_charges.dtype)
        is_parallel_mat = torch.zeros(n_tot, n_tot, dtype=torch.bool)

        for i in range(n_tot):
            for j in range(i + 1, n_tot):
                same_spin = (i < n_up and j < n_up) or (i >= n_up and j >= n_up)
                if same_spin or two_electron_spin == 1:
                    gamma_mat[i, j] = 0.25
                    is_parallel_mat[i, j] = True
                else:
                    gamma_mat[i, j] = 0.50
                    is_parallel_mat[i, j] = False

        self.register_buffer("gamma_ee", gamma_mat)
        self.register_buffer("is_parallel_ee", is_parallel_mat)

        # 阻尼参数参数化 (通过 softplus 保证 > 0)
        # softplus^{-1}(y) = ln(exp(y) - 1)
        raw_b_par = torch.log(torch.exp(torch.tensor(float(init_b_ee_parallel))) - 1.0)
        raw_b_anti = torch.log(torch.exp(torch.tensor(float(init_b_ee_antiparallel))) - 1.0)
        raw_b_en = torch.log(torch.exp(torch.tensor(float(init_b_en))) - 1.0)

        # 针对每个核分配一个阻尼参数
        raw_b_en_vec = raw_b_en.repeat(self.num_nuclei)

        if trainable:
            self.raw_b_ee_par = nn.Parameter(raw_b_par)
            self.raw_b_ee_anti = nn.Parameter(raw_b_anti)
            self.raw_b_en = nn.Parameter(raw_b_en_vec)
        else:
            self.register_buffer("raw_b_ee_par", raw_b_par)
            self.register_buffer("raw_b_ee_anti", raw_b_anti)
            self.register_buffer("raw_b_en", raw_b_en_vec)

    @property
    def b_ee_parallel(self) -> torch.Tensor:
        """同自旋电子对阻尼因子 b > 0。"""
        return F.softplus(self.raw_b_ee_par)

    @property
    def b_ee_antiparallel(self) -> torch.Tensor:
        """异自旋电子对阻尼因子 b > 0。"""
        return F.softplus(self.raw_b_ee_anti)

    @property
    def b_en(self) -> torch.Tensor:
        """各原子核电子-核阻尼因子向量 b > 0, Shape: (M,)。"""
        return F.softplus(self.raw_b_en)

    def compute_en(self, r: torch.Tensor, nuclear_coords: torch.Tensor) -> torch.Tensor:
        """
        计算电子-核 Kato 尖峰对数因子 U_en(r)。

        公式:
            U_en(r) = - sum_{i=1}^N sum_{I=1}^M (Z_I * r_{iI}) / (1 + b_{en, I} * r_{iI})

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            nuclear_coords: 核坐标, Shape: (M, 3)

        Returns:
            u_en: 电子-核对数贡献, Shape: (B,)
        """
        if not self.cusp_en or self.num_nuclei == 0 or self.total_electrons == 0:
            return torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)

        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        charges = self.nuclear_charges.to(device=r.device, dtype=r.dtype)  # (M,)
        b_en = self.b_en.to(device=r.device, dtype=r.dtype)                # (M,)

        # 计算 r_{iI} 相对位移与距离
        # r: (B, N, 1, 3), coords: (1, 1, M, 3) -> diff: (B, N, M, 3)
        diff_en = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dist_en = torch.sqrt(torch.clamp(
            torch.sum(diff_en ** 2, dim=-1), min=self.eps
        ))  # (B, N, M)

        # 广播系数: charges -> (1, 1, M), b_en -> (1, 1, M)
        charges_b = charges.view(1, 1, -1)
        b_en_b = b_en.view(1, 1, -1)

        # Padé 形式: - (Z_I * r_{iI}) / (1 + b_I * r_{iI})
        u_en_elements = - (charges_b * dist_en) / (1.0 + b_en_b * dist_en)  # (B, N, M)
        u_en = torch.sum(u_en_elements, dim=(-1, -2))                       # (B,)

        return u_en

    def compute_ee(self, r: torch.Tensor) -> torch.Tensor:
        """
        计算电子-电子 Kato 尖峰对数因子 U_ee(r)。

        公式:
            U_ee(r) = sum_{i < j} (gamma_{ij} * r_{ij}) / (1 + b_{ee, ij} * r_{ij})

        Args:
            r: 电子坐标, Shape: (B, N, 3)

        Returns:
            u_ee: 电子-电子对数贡献, Shape: (B,)
        """
        if not self.cusp_ee or self.total_electrons < 2:
            return torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)

        device, dtype = r.device, r.dtype
        gamma_mat = self.gamma_ee.to(device=device, dtype=dtype)              # (N, N)
        is_parallel = self.is_parallel_ee.to(device=device)                  # (N, N)
        b_par = self.b_ee_parallel.to(device=device, dtype=dtype)            # scalar
        b_anti = self.b_ee_antiparallel.to(device=device, dtype=dtype)        # scalar

        # 构造有效 b 矩阵 (N, N)
        b_mat = torch.where(is_parallel, b_par, b_anti)

        # 计算 r_{ij} 距离
        # r: (B, N, 1, 3) - (B, 1, N, 3) -> diff: (B, N, N, 3)
        diff_ee = r.unsqueeze(2) - r.unsqueeze(1)
        dist_ee = torch.sqrt(torch.clamp(
            torch.sum(diff_ee ** 2, dim=-1), min=self.eps
        ))  # (B, N, N)

        # 只对严格上三角 i < j 进行加和 (避免重复计算与自身对)
        mask = torch.triu(
            torch.ones(self.total_electrons, self.total_electrons, device=device, dtype=torch.bool),
            diagonal=1,
        )  # (N, N)

        gamma_active = gamma_mat.unsqueeze(0)  # (1, N, N)
        b_active = b_mat.unsqueeze(0)          # (1, N, N)

        # Padé 形式: (gamma * r_{ij}) / (1 + b * r_{ij})
        u_ee_elements = (gamma_active * dist_ee) / (1.0 + b_active * dist_ee)  # (B, N, N)
        u_ee = torch.sum(
            torch.where(mask.unsqueeze(0), u_ee_elements, torch.zeros_like(u_ee_elements)),
            dim=(-1, -2),
        )  # (B,)

        return u_ee

    def forward(self, r: torch.Tensor, nuclear_coords: torch.Tensor) -> torch.Tensor:
        """
        计算总 Jastrow 对数因子: ln J(r) = U_en(r) + U_ee(r)。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            nuclear_coords: 核坐标, Shape: (M, 3)

        Returns:
            log_j: ln J(r), Shape: (B,)
        """
        batch_size, num_electrons, _ = r.shape
        assert num_electrons == self.total_electrons, (
            f"电子数不匹配: 期望 {self.total_electrons}, 实际 {num_electrons}"
        )

        u_en = self.compute_en(r, nuclear_coords)
        u_ee = self.compute_ee(r)
        return u_en + u_ee
