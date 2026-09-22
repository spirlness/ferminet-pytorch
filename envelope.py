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
        Envelope_{k, i, j}(r_i) = sum_{m=1}^M pi_{k, j, g(m)} * exp(-alpha_{k, j, g(m)} * |r_i - R_m|)
    2. 各向异性模式 (anisotropic=True, 提升非球对称成键轨道能力):
        学习三维主轴度规 Sigma_{k, j, g(m)}^d:
        Envelope_{k, i, j}(r_i) = sum_{m=1}^M pi_{k, j, g(m)} * exp(-sqrt(sum_{d=1}^3 (sigma_{k, j, g(m), d} * (r_{i, d} - R_{m, d}))^2))
    其中：
        - k: 行列式索引 (1 <= k <= num_determinants)
        - i: 电子索引 (1 <= i <= num_electrons)
        - j: 轨道索引 (1 <= j <= num_orbitals)
        - m: 原子核索引 (1 <= m <= num_nuclei)
        - g(m): 核 m 的电荷组索引 (见下 "D5 修复")
        - pi_{k, j, g}: 可学习包络线性组合权重 (按电荷组参数化)
        - alpha / sigma > 0: 可学习衰减速率/主轴度规参数 (通过 softplus 保证严格为正，避免发散)

    D5 修复 (第一级)——同电荷核包络参数共享:
    ------------------------------------------------
    `weights` 与 `decay_rates_raw` 按**电荷组** g (而非核索引 m) 参数化，最后一维大小为
    G = 电荷等价类数目；`forward` 内用 buffer `nucleus_group_index` 将参数聚集回核轴
    (K, N_orb, M)。
    数学依据 (pi 与 alpha 必须**同时**共享)：对核置换 sigma 做哑指标重标号可得
        f'_{kj}(r_i) = sum_{m'} pi_{kj, sigma^{-1}(m')} * exp(-alpha_{kj, sigma^{-1}(m')} * d_{i, m'})。
    要使 f' == f 对所有电子位置 r_i 成立，必须 pi 与 alpha 同时在电荷等价类上为常数：
    只共享 pi 时组内不同的 alpha 使指数项错配；只共享 alpha 时组内不同的 pi 使线性组合
    系数错配，二者均会改变包络值 (对任意 r_i 逐项比对即得)。
    注意：仅靠本模块的参数共享**不足以**使 log|Psi| 完全核置换不变——特征侧仍按核序
    拼接送入密集层，需配合 features.py 的核轴规范化 (D5 第二级)。
    残留限制：若调用方未提供 nuclear_charges，本模块退回 G = M (不共享) 的旧行为，
    核置换不变性不成立。

    距离计算的数值稳定核 (D6):
    ------------------------------------------------
    距离统一取 sqrt(clamp(r^2, min=eps))。clamp 在**所有** r^2 >= eps 处是恒等变换，
    距离与未软化时逐位相同，偏置精确为零；而旧实现 sqrt(r^2 + eps) 处处存在相对偏差
    eps/(2 r^2)，会使核处径向对数斜率趋于 0 而非 -Z，破坏 Kato 尖点条件。
    在 r^2 < eps 的重合邻域内距离被限制为 sqrt(eps)，有限且梯度为零。
    不能直接删除 eps：裸 sqrt(r^2) 在 r = 0 处梯度为 inf * 0 = NaN (见
    test_geometry_invariants.py 的记录性测试)。
    """

    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        num_nuclei: int,
        eps: float = 1e-24,
        init_decay_rate: float = 1.0,
        anisotropic: bool = False,
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        """
        初始化原子核指数包络模块。

        Args:
            num_determinants (int): 行列式数量 (K)。
            num_orbitals (int): 该自旋通道对应的分子轨道数 (通常等于该自旋电子数)。
            num_nuclei (int): 原子核数目 (M)。
            eps (float): 最小**平方**距离 (单位 Bohr^2)。距离计算为
                sqrt(clamp(r^2, min=eps))，即软化半径为 sqrt(eps) Bohr
                (默认 1e-24 对应 1e-12 Bohr，远小于任何化学键长)。
            init_decay_rate (float): 衰减速率初始值 (默认为 1.0)。
            anisotropic (bool): 是否启用各向异性可学习三维主轴度规 (默认为 False 各向同性)。
            nuclear_charges (Optional[torch.Tensor]): 可选的原子核电荷量 (原子序数 Z_I)。
                - Shape: `(M,)` (也兼容 `(M, 1)` / `(1, M)`，按展平后处理)。
                - 提供时按电荷等价类 (torch.unique 升序) 建组，同电荷核共享包络参数
                  (D5 第一级)；为 None 时 G = M，行为与旧版逐位一致。
        """
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.num_nuclei = num_nuclei
        self.eps = eps
        self.anisotropic = anisotropic

        if nuclear_charges is not None:
            charges = nuclear_charges.detach().reshape(-1)
            assert charges.numel() == num_nuclei, (
                f"nuclear_charges 元素数 ({charges.numel()}) 必须与原子核数 ({num_nuclei}) 一致"
            )
            # 电荷等价类分组: uniq 按升序，inverse[m] in [0, G) 为核 m 所属组索引
            uniq, inverse = torch.unique(charges, return_inverse=True)
            self.num_groups = int(uniq.numel())
            group_index = inverse.to(torch.long)
        else:
            # 向后兼容: 未提供电荷时 G = M、索引为 arange(M)，
            # forward 中的聚集退化为恒等拷贝，参数形状与数值与旧实现逐位一致。
            self.num_groups = num_nuclei
            group_index = torch.arange(num_nuclei, dtype=torch.long)

        # 分组决定参数如何映射回核轴，属于模型物理状态，必须进 checkpoint。
        # FermiNet.load_state_dict 对缺少该 buffer 的旧 checkpoint 做受控迁移。
        self.register_buffer("nucleus_group_index", group_index)
        # tied: 是否存在实际的参数共享 (同电荷核多于一个)
        self.tied = self.num_groups < num_nuclei

        num_groups = self.num_groups
        # 可学习组合系数 pi: 按电荷组参数化, shape (num_determinants, num_orbitals, G)
        self.weights = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, num_groups)
        )
        if self.anisotropic:
            # 各向异性: 三维主轴可学习度规参数，shape (K, N_orb, G, 3)
            self.decay_rates_raw = nn.Parameter(
                torch.empty(num_determinants, num_orbitals, num_groups, 3)
            )
        else:
            # 各向同性: 标量衰减参数，shape (K, N_orb, G)
            self.decay_rates_raw = nn.Parameter(
                torch.empty(num_determinants, num_orbitals, num_groups)
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

        前置条件: `nuclear_coords` 的核顺序必须与构造时传入的 `nuclear_charges`
        一致 (同序)，因为 buffer `nucleus_group_index` 按构造时的电荷顺序建立。

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

        # ---------------------------------------------------------------------
        # 2. D5 第一级: 将按电荷组 (G) 参数化的包络权重聚集回核轴 (M)。
        # index_select 的反向是 scatter-add: 同组各核对共享参数的梯度自动求和，
        # 正是对称和的正确梯度。nuclear_charges=None 时索引为 arange(M)，
        # 本聚集为恒等拷贝 (形状与数值逐位不变)。
        # ---------------------------------------------------------------------
        weights = self.weights.index_select(2, self.nucleus_group_index)  # (K, N_orb, M)

        if self.anisotropic:
            # -----------------------------------------------------------------
            # 3A. 各向异性模式: 三维主轴度规 sigma: (K, N_orb, M, 3)
            # -----------------------------------------------------------------
            sigma = F.softplus(
                self.decay_rates_raw.index_select(2, self.nucleus_group_index)
            )  # (K, N_orb, M, 3)
            diff_expanded = diff.unsqueeze(1).unsqueeze(3)  # (B, 1, N_spin, 1, M, 3)
            sigma_expanded = sigma.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M, 3)

            scaled_diff = sigma_expanded * diff_expanded  # (B, K, N_spin, N_orb, M, 3)
            # D6: clamp 核, 见类 docstring; eps 为最小平方距离 (Bohr^2)
            dist_aniso = torch.sqrt(
                torch.clamp(torch.sum(scaled_diff ** 2, dim=-1), min=self.eps)
            )  # (B, K, N_spin, N_orb, M)

            exponent = torch.exp(-dist_aniso)  # (B, K, N_spin, N_orb, M)
            weights_expanded = weights.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)
            envelope = torch.sum(weights_expanded * exponent, dim=-1)  # (B, K, N_spin, N_orb)
        else:
            # -----------------------------------------------------------------
            # 3B. 各向同性兼容模式: 欧氏标量距离与各向同性衰减率 alpha: (K, N_orb, M)
            # -----------------------------------------------------------------
            # D6: clamp 核, 见类 docstring; eps 为最小平方距离 (Bohr^2)
            dist = torch.sqrt(torch.clamp(torch.sum(diff ** 2, dim=-1), min=self.eps))  # (B, N_spin, M)
            alpha = F.softplus(
                self.decay_rates_raw.index_select(2, self.nucleus_group_index)
            )  # (K, N_orb, M)

            dist_expanded = dist.unsqueeze(1).unsqueeze(3)  # (B, 1, N_spin, 1, M)
            alpha_expanded = alpha.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)
            weights_expanded = weights.unsqueeze(0).unsqueeze(2)  # (1, K, 1, N_orb, M)

            exponent = torch.exp(-alpha_expanded * dist_expanded)  # (B, K, N_spin, N_orb, M)
            envelope = torch.sum(weights_expanded * exponent, dim=-1)  # (B, K, N_spin, N_orb)

        return envelope
