"""
FermiNet (Fermi Neural Network) 深度波函数模型。

参考论文:
    Pfau et al., "Ab-Initio Solution of the Many-Electron Schrödinger Equation
    with Deep Neural Networks", Phys. Rev. Research 2, 033429 (2020).

核心架构：
1. 几何特征提取 → 2. 自旋等变前馈层 → 3. 分子轨道构建 → 4. Slater 行列式 → 5. 多行列式求和
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from .features import ElectronFeatureEmbedding
    from .layers import FermiInteractionLayer
    from .envelope import NuclearEnvelope
    from .jastrow import KatoCuspJastrow
    from .spin import SpinProjector
except (ImportError, ValueError):
    from features import ElectronFeatureEmbedding
    from layers import FermiInteractionLayer
    from envelope import NuclearEnvelope
    from jastrow import KatoCuspJastrow
    from spin import SpinProjector


class FermiNet(nn.Module):
    """
    FermiNet 深度波函数模型。

    将电子坐标 r ∈ (B, N, 3) 映射为波函数标量 Psi(r) ∈ (B,)，
    满足同自旋电子置换的反对称性 (通过 Slater 行列式实现)。
    """

    def __init__(
        self,
        nuclear_coords: torch.Tensor,
        nuclear_charges: torch.Tensor,
        n_up: int,
        n_down: int,
        num_determinants: int = 16,
        hidden_dim_one: int = 64,
        hidden_dim_two: int = 16,
        num_layers: int = 2,
        eps: float = 1e-24,
        use_cusp_jastrow: bool = False,
        jastrow: Optional[nn.Module] = None,
        jastrow_kwargs: Optional[dict] = None,
        spin_adapted: bool = False,
        spin_projector: Optional[nn.Module] = None,
        spin_projector_kwargs: Optional[dict] = None,
        envelope_type: Optional[str] = None,
        smooth_cuspless: Optional[bool] = None,
    ) -> None:
        """
        初始化 FermiNet 模型。

        Args:
            nuclear_coords: 原子核坐标, Shape: (M, 3)
            nuclear_charges: 原子核电荷 (原子序数), Shape: (M,)
            n_up: 自旋向上电子数
            n_down: 自旋向下电子数
            num_determinants: 行列式数量 K
            hidden_dim_one: 单电子流隐藏层维度
            hidden_dim_two: 双电子流隐藏层维度
            num_layers: 交互层数
            eps: 数值稳定性参数 (最小平方距离)
            use_cusp_jastrow: 是否启用显式 Kato 尖峰 Jastrow 因子
            jastrow: 自定义 Jastrow 模块 (若提供则优先使用)
            jastrow_kwargs: KatoCuspJastrow 初始化的额外参数字典
            spin_adapted: 是否启用自旋纯态投影算符 (如针对双电子单重态加入空间对称化 1/2(I + P12))
            spin_projector: 自定义自旋投影算符模块 (若提供则优先使用)
            spin_projector_kwargs: SpinProjector 初始化的额外参数字典
        """
        super().__init__()
        assert n_up >= 0 and n_down >= 0, "电子数必须非负"
        assert n_up + n_down > 0, "体系必须包含至少一个电子"

        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.num_determinants = num_determinants
        self.hidden_dim_one = hidden_dim_one
        self.hidden_dim_two = hidden_dim_two
        self.num_layers = num_layers
        self.eps = eps

        # 注册非训练参数缓冲
        self.register_buffer("nuclear_coords", nuclear_coords.clone().detach())
        self.register_buffer("nuclear_charges", nuclear_charges.clone().detach())

        num_nuclei = nuclear_coords.shape[0]
        self.num_nuclei = num_nuclei

        # 解析 Jastrow 状态与协同平滑参数
        has_jastrow = (jastrow is not None) or use_cusp_jastrow
        eff_env_type = envelope_type if envelope_type is not None else ("gaussian" if has_jastrow else "exponential")
        eff_smooth = smooth_cuspless if smooth_cuspless is not None else has_jastrow

        self.envelope_type = eff_env_type
        self.smooth_cuspless = eff_smooth

        # 1. 几何特征提取模块
        self.embedding = ElectronFeatureEmbedding(eps=eps, smooth_cuspless=eff_smooth)
        in_dim_one, in_dim_two = ElectronFeatureEmbedding.feature_dims(
            num_nuclei=num_nuclei,
            use_nuclear_charges=True,
        )

        # 2. 自旋等变交互层
        layers: List[FermiInteractionLayer] = []
        for l in range(num_layers):
            dim_in_1 = in_dim_one if l == 0 else hidden_dim_one
            dim_in_2 = in_dim_two if l == 0 else hidden_dim_two
            layers.append(
                FermiInteractionLayer(
                    in_dim_one=dim_in_1,
                    in_dim_two=dim_in_2,
                    out_dim_one=hidden_dim_one,
                    out_dim_two=hidden_dim_two,
                    produce_two_electron_output=(l != num_layers - 1),
                )
            )
        self.layers = nn.ModuleList(layers)

        # 3. 轨道投影层与核包络
        if self.n_up > 0:
            self.linear_orb_up = nn.Linear(hidden_dim_one, num_determinants * n_up)
            self.envelope_up = NuclearEnvelope(
                num_determinants=num_determinants,
                num_orbitals=n_up,
                num_nuclei=num_nuclei,
                eps=eps,
                envelope_type=eff_env_type,
            )
        else:
            self.linear_orb_up = None
            self.envelope_up = None

        if self.n_down > 0:
            self.linear_orb_down = nn.Linear(hidden_dim_one, num_determinants * n_down)
            self.envelope_down = NuclearEnvelope(
                num_determinants=num_determinants,
                num_orbitals=n_down,
                num_nuclei=num_nuclei,
                eps=eps,
                envelope_type=eff_env_type,
            )
        else:
            self.linear_orb_down = None
            self.envelope_down = None

        # 4. 多行列式线性组合系数
        self.det_weights = nn.Parameter(
            torch.ones(num_determinants) / (num_determinants ** 0.5)
        )

        # 5. Kato 尖峰 Jastrow 因子
        if jastrow is not None:
            self.jastrow = jastrow
        elif use_cusp_jastrow:
            jk = jastrow_kwargs or {}
            self.jastrow = KatoCuspJastrow(
                nuclear_charges=self.nuclear_charges,
                n_up=n_up,
                n_down=n_down,
                eps=eps,
                **jk,
            )
        else:
            self.jastrow = None

        # 6. 自旋投影算符 (自旋纯态适配)
        if spin_projector is not None:
            self.spin_projector = spin_projector
        elif spin_adapted:
            spk = spin_projector_kwargs or {}
            self.spin_projector = SpinProjector(
                n_up=n_up,
                n_down=n_down,
                **spk,
            )
        else:
            self.spin_projector = None

    def forward_orbitals(
        self, r: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        计算分子轨道矩阵。

        Args:
            r: 电子坐标, Shape: (B, N, 3)

        Returns:
            d_up: 自旋向上轨道矩阵, Shape: (B, K, n_up, n_up) 或 None
            d_down: 自旋向下轨道矩阵, Shape: (B, K, n_down, n_down) 或 None
        """
        batch_size, num_electrons, _ = r.shape
        assert num_electrons == self.total_electrons

        # 1. 提取几何特征
        h1, h2 = self.embedding(r, self.nuclear_coords, self.nuclear_charges)

        # 2. 等变交互传播
        for layer in self.layers:
            h1, h2 = layer(h1, h2, self.n_up, self.n_down)

        # 3. 构造自旋向上轨道
        d_up: Optional[torch.Tensor] = None
        if self.n_up > 0 and self.linear_orb_up is not None and self.envelope_up is not None:
            h1_up = h1[:, : self.n_up, :]
            r_up = r[:, : self.n_up, :]
            backflow_up = self.linear_orb_up(h1_up)
            backflow_up = backflow_up.view(batch_size, self.n_up, self.num_determinants, self.n_up)
            backflow_up = backflow_up.permute(0, 2, 1, 3)
            env_up = self.envelope_up(r_up, self.nuclear_coords)
            d_up = backflow_up * env_up

        # 4. 构造自旋向下轨道
        d_down: Optional[torch.Tensor] = None
        if self.n_down > 0 and self.linear_orb_down is not None and self.envelope_down is not None:
            h1_down = h1[:, self.n_up :, :]
            r_down = r[:, self.n_up :, :]
            backflow_down = self.linear_orb_down(h1_down)
            backflow_down = backflow_down.view(
                batch_size, self.n_down, self.num_determinants, self.n_down
            )
            backflow_down = backflow_down.permute(0, 2, 1, 3)
            env_down = self.envelope_down(r_down, self.nuclear_coords)
            d_down = backflow_down * env_down

        return d_up, d_down

    def _forward_log_psi_base(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算基础 FermiNet (未进行自旋投影和 Jastrow 修正) 的对数波函数。

        Args:
            r: 电子坐标, Shape: (B, N, 3)

        Returns:
            log_abs_psi: log|Psi(r)|, Shape: (B,)
            sign_psi: sign(Psi(r)), Shape: (B,)
        """
        batch_size = r.shape[0]
        d_up, d_down = self.forward_orbitals(r)

        # 计算行列式 (使用 slogdet 保证数值稳定性)
        if d_up is not None:
            sign_up, logabsdet_up = torch.linalg.slogdet(d_up)
        else:
            sign_up = torch.ones(batch_size, self.num_determinants, device=r.device, dtype=r.dtype)
            logabsdet_up = torch.zeros_like(sign_up)

        if d_down is not None:
            sign_down, logabsdet_down = torch.linalg.slogdet(d_down)
        else:
            sign_down = torch.ones(batch_size, self.num_determinants, device=r.device, dtype=r.dtype)
            logabsdet_down = torch.zeros_like(sign_down)

        # 合并上下自旋行列式
        sign_k = sign_up * sign_down          # (B, K)
        log_abs_k = logabsdet_up + logabsdet_down  # (B, K)

        # 多行列式加权求和: Psi = sum_k w_k * det_k
        w = self.det_weights.unsqueeze(0)  # (1, K)
        # LogSumExp 技巧保证数值稳定
        max_log = torch.amax(log_abs_k.detach(), dim=1, keepdim=True)  # (B, 1)
        all_zero = torch.isneginf(max_log)
        safe_max_log = torch.where(all_zero, torch.zeros_like(max_log), max_log)

        scaled_det = w * sign_k * torch.exp(log_abs_k - safe_max_log)
        scaled_det = torch.where(all_zero.expand_as(scaled_det), torch.zeros_like(scaled_det), scaled_det)
        summed = torch.sum(scaled_det, dim=1)  # (B,)

        abs_summed = torch.abs(summed)
        zero_mask = (abs_summed == 0) | all_zero.squeeze(1)

        sign_psi = torch.sign(summed)
        sign_psi = torch.where(zero_mask, torch.zeros_like(sign_psi), sign_psi)

        safe_abs_summed = torch.where(zero_mask, torch.ones_like(abs_summed), abs_summed)
        log_abs_psi = torch.where(
            zero_mask,
            torch.full_like(summed, float("-inf")),
            safe_max_log.squeeze(1) + torch.log(safe_abs_summed),
        )

        return log_abs_psi, sign_psi

    def _forward_log_psi_unprojected(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算未投影的全对数波函数 (基础 Slater 行列式融合解析 Kato Jastrow 因子)。
        """
        log_abs_psi, sign_psi = self._forward_log_psi_base(r)

        if self.jastrow is not None:
            log_j = self.jastrow(r, self.nuclear_coords)
            zero_mask = (sign_psi == 0) | torch.isneginf(log_abs_psi)
            log_abs_psi = torch.where(
                zero_mask,
                torch.full_like(log_abs_psi, float("-inf")),
                log_abs_psi + log_j,
            )

        return log_abs_psi, sign_psi

    def forward_log_psi(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算总对数波函数: (log|Psi(r)|, sign(Psi(r)))。
        首先通过神经网络与解析 Kato 尖峰 Jastrow 因子消除库仑奇点发散，
        并在波函数输出前施加自旋投影算符 (如针对双电子单重态加入空间对称化 1/2(I + P_12))，
        在架构底层彻底消除自旋污染。

        Args:
            r: 电子坐标, Shape: (B, N, 3)

        Returns:
            log_abs_psi: log|Psi(r)|, Shape: (B,)
            sign_psi: sign(Psi(r)), Shape: (B,)
        """
        if self.spin_projector is not None:
            return self.spin_projector.project_log_psi(
                self._forward_log_psi_unprojected, r
            )
        return self._forward_log_psi_unprojected(r)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        计算波函数标量值: Psi(r) = sign * exp(log|Psi|)。

        Args:
            r: 电子坐标, Shape: (B, N, 3)

        Returns:
            psi: 波函数值, Shape: (B,)
        """
        log_abs_psi, sign_psi = self.forward_log_psi(r)
        zero_mask = (sign_psi == 0) | torch.isneginf(log_abs_psi)
        safe_log = torch.where(zero_mask, torch.zeros_like(log_abs_psi), log_abs_psi)
        psi = sign_psi * torch.exp(safe_log)
        return torch.where(zero_mask, torch.zeros_like(psi), psi)

