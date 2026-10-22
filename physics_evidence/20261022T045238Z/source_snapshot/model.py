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
    from .geometry import GeometricContext
    from .composable_model import ComposableFermiNet
except (ImportError, ValueError):
    from features import ElectronFeatureEmbedding
    from layers import FermiInteractionLayer
    from envelope import NuclearEnvelope
    from jastrow import KatoCuspJastrow
    from spin import SpinProjector
    from geometry import GeometricContext
    from composable_model import ComposableFermiNet


try:
    from .determinants import _CofactorDeterminant, _determinant_terms, signed_determinant_sum
    from .physical_config import cusp_defaults, resolve_spin, pair_spin_for, validate_pair_cusp
except ImportError:
    from determinants import _CofactorDeterminant, _determinant_terms, signed_determinant_sum
    from physical_config import cusp_defaults, resolve_spin, pair_spin_for, validate_pair_cusp


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
        interaction_type: str = "ferminet",
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
            interaction_type: 相互作用层类型 ("ferminet" 经典两流模式或 "psiformer" 自注意力模式)
        """
        super().__init__()
        assert n_up >= 0 and n_down >= 0, "电子数必须非负"
        assert n_up + n_down > 0, "体系必须包含至少一个电子"
        self.interaction_type = interaction_type.lower()


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
        eff_env_type, eff_smooth = cusp_defaults(
            envelope_type, smooth_cuspless, has_jastrow, jastrow)

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
                nuclear_charges=self.nuclear_charges,
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
                nuclear_charges=self.nuclear_charges,
            )
        else:
            self.linear_orb_down = None
            self.envelope_down = None

        # 4. 多行列式线性组合系数
        self.det_weights = nn.Parameter(
            torch.ones(num_determinants) / (num_determinants ** 0.5)
        )

        self.spin_projector = resolve_spin(
            n_up, n_down, spin_adapted, spin_projector, spin_projector_kwargs)

        # 6. Kato 尖峰 Jastrow 因子
        pair_spin = pair_spin_for(n_up, n_down, self.spin_projector)
        if jastrow is not None:
            self.jastrow = jastrow
        elif use_cusp_jastrow:
            jk = dict(jastrow_kwargs or {})
            if pair_spin is not None:
                jk.setdefault("two_electron_spin", pair_spin)
            self.jastrow = KatoCuspJastrow(
                nuclear_charges=self.nuclear_charges,
                n_up=n_up,
                n_down=n_down,
                eps=eps,
                **jk,
            )
        else:
            self.jastrow = None
        validate_pair_cusp(self.jastrow, pair_spin)

    def forward_orbitals(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        计算分子轨道矩阵。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            geom_ctx: 可选的 GeometricContext 共享几何上下文

        Returns:
            d_up: 自旋向上轨道矩阵, Shape: (B, K, n_up, n_up) 或 None
            d_down: 自旋向下轨道矩阵, Shape: (B, K, n_down, n_down) 或 None
        """
        batch_size, num_electrons, _ = r.shape
        assert num_electrons == self.total_electrons

        # 1. 提取几何特征 (复用 geom_ctx)
        h1, h2 = self.embedding(r, self.nuclear_coords, self.nuclear_charges, geom_ctx=geom_ctx)

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
            dist_up = geom_ctx.d_en[:, :self.n_up, :, :] if geom_ctx is not None else None
            env_up = self.envelope_up(r_up, self.nuclear_coords, dist=dist_up)
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
            dist_down = geom_ctx.d_en[:, self.n_up:, :, :] if geom_ctx is not None else None
            env_down = self.envelope_down(r_down, self.nuclear_coords, dist=dist_down)
            d_down = backflow_down * env_down

        return d_up, d_down

    def _forward_log_psi_base(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算基础 FermiNet (未进行自旋投影和 Jastrow 修正) 的对数波函数。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            geom_ctx: 可选的 GeometricContext 共享几何上下文

        Returns:
            log_abs_psi: log|Psi(r)|, Shape: (B,)
            sign_psi: sign(Psi(r)), Shape: (B,)
        """
        batch_size = r.shape[0]
        d_up, d_down = self.forward_orbitals(r, geom_ctx=geom_ctx)

        return signed_determinant_sum(d_up, d_down, self.det_weights)

    def _forward_log_psi_unprojected(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算未投影的全对数波函数 (基础 Slater 行列式融合解析 Kato Jastrow 因子)。
        """
        log_abs_psi, sign_psi = self._forward_log_psi_base(r, geom_ctx=geom_ctx)

        if self.jastrow is not None:
            log_j = self.jastrow(r, self.nuclear_coords, geom_ctx=geom_ctx)
            zero_mask = (sign_psi == 0) | torch.isneginf(log_abs_psi)
            log_abs_psi = torch.where(
                zero_mask,
                torch.full_like(log_abs_psi, float("-inf")),
                log_abs_psi + log_j,
            )

        return log_abs_psi, sign_psi

    def forward_log_psi(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算总对数波函数: (log|Psi(r)|, sign(Psi(r)))。
        首先通过神经网络与解析 Kato 尖峰 Jastrow 因子消除库仑奇点发散，
        并在波函数输出前施加自旋投影算符 (如针对双电子单重态加入空间对称化 1/2(I + P_12))，
        在架构底层彻底消除自旋污染。

        Args:
            r: 电子坐标, Shape: (B, N, 3)
            geom_ctx: 可选的 GeometricContext 共享几何上下文

        Returns:
            log_abs_psi: log|Psi(r)|, Shape: (B,)
            sign_psi: sign(Psi(r)), Shape: (B,)
        """
        if self.spin_projector is not None:
            # 自旋投影会根据置换展开电子坐标 (B*K, N, 3)，外部针对单批次 (B, N, 3) 构建的 geom_ctx 批次与坐标均不匹配
            return self.spin_projector.project_log_psi(
                lambda coords: self._forward_log_psi_unprojected(
                    coords,
                    geom_ctx=geom_ctx if (
                        geom_ctx is not None
                        and geom_ctx.r.shape == coords.shape
                        and geom_ctx.r.data_ptr() == coords.data_ptr()
                    ) else None,
                ),
                r,
            )
        return self._forward_log_psi_unprojected(r, geom_ctx=geom_ctx)


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

    def has_trainable_jastrow(self) -> bool:
        """检查模型是否包含可学习的 Jastrow 因子参数。"""
        if self.jastrow is None:
            return False
        return any(isinstance(p, nn.Parameter) for p in self.jastrow.parameters())

    def freeze_jastrow(self) -> None:
        """冻结 Jastrow 因子参数，使其不参与反向传播梯度更新 (Stage 1: 预训练神经网络轨道)。"""
        if self.jastrow is not None:
            for p in self.jastrow.parameters():
                p.requires_grad = False

    def unfreeze_jastrow(self) -> None:
        """解冻 Jastrow 因子参数，使其恢复反向传播梯度更新 (Stage 2: Jastrow 联合微调)。"""
        if self.jastrow is not None:
            for p in self.jastrow.parameters():
                p.requires_grad = True

    def get_parameter_groups(
        self, lr: float, jastrow_lr: Optional[float] = None
    ) -> List[dict]:
        """
        获取支持分阶段或独立学习率训练的 PyTorch 参数组。

        Args:
            lr: 神经网络轨道与主干参数的学习率。
            jastrow_lr: Jastrow 因子的独立微调学习率。若为 None 则跟随 lr。

        Returns:
            符合 PyTorch 优化器格式的参数组字典列表。
        """
        jastrow_params = list(self.jastrow.parameters()) if self.jastrow is not None else []
        jastrow_ids = {id(p) for p in jastrow_params}
        backbone_params = [p for p in self.parameters() if id(p) not in jastrow_ids]

        groups: List[dict] = [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": lr, "name": "backbone"}
        ]
        trainable_j = [p for p in jastrow_params if p.requires_grad]
        if trainable_j:
            eff_j_lr = jastrow_lr if jastrow_lr is not None else lr
            groups.append({"params": trainable_j, "lr": eff_j_lr, "name": "jastrow"})
        return groups


__all__ = [
    "FermiNet",
    "ComposableFermiNet",
]


