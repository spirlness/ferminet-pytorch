"""
ferminet_pytorch.composable_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
完全符合单一职责原则 (SRP) 与可插拔架构的现代化解耦 FermiNet 波函数擬设。
支持传统两流相互作用层与 Psiformer 自注意力等变层无缝切换。
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn

from .geometry import GeometricContext, smooth_distance
from .features import ElectronFeatureEmbedding
from .layers_modern import FermiInteractionLayerModern
from .envelope import NuclearEnvelope
from .psiformer_attention import PsiformerAttentionBlock
from .jastrow import KatoCuspJastrow
from .spin import SpinProjector


class OrbitalEnvelopeProjector(nn.Module):
    """
    分子轨道与核包络投影组合器。
    将等变隐藏特征 h 投影到分子轨道，并与空间指数衰减包络逐元素加权。
    """
    def __init__(
        self,
        hidden_dim: int,
        num_determinants: int,
        num_orbitals: int,
        nuclear_coords: torch.Tensor,
        envelope_type: str = "exponential",
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.linear = nn.Linear(hidden_dim, num_determinants * num_orbitals)
        self.envelope = NuclearEnvelope(
            num_determinants=num_determinants,
            num_orbitals=num_orbitals,
            num_nuclei=nuclear_coords.shape[0],
            envelope_type=envelope_type,
            nuclear_charges=nuclear_charges,
        )

    def forward(
        self,
        h_spin: torch.Tensor,
        dist_spin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_spin: 对应自旋电子隐藏特征, shape (B, N_spin, hidden_dim)
            dist_spin: 电子-核平滑欧氏距离, shape (B, N_spin, M) 或 (B, N_spin, M, 1)

        Returns:
            d_spin: 轨道矩阵, shape (B, K, N_spin, N_spin)
        """
        B, N_spin, _ = h_spin.shape
        backflow = self.linear(h_spin)  # (B, N_spin, K * N_orb)
        backflow = backflow.view(B, N_spin, self.num_determinants, self.num_orbitals)
        backflow = backflow.permute(0, 2, 1, 3).contiguous()  # (B, K, N_spin, N_orb)
        env = self.envelope(dist_spin)  # (B, K, N_spin, N_orb)
        return backflow * env


from .determinants import signed_determinant_sum
from .physical_config import cusp_defaults, resolve_spin, pair_spin_for, validate_pair_cusp


class BranchlessCofactorDeterminant:
    """Compatibility facade for the shared exact determinant implementation."""
    @staticmethod
    def compute(
        d_up: Optional[torch.Tensor], d_down: Optional[torch.Tensor],
        det_weights: torch.Tensor, n_up: int, n_down: int, num_determinants: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return signed_determinant_sum(d_up, d_down, det_weights)


class ComposableFermiNet(nn.Module):
    """
    可组合、低显存、高性能的现代化 FermiNet 深度波函数模型。
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
        interaction_type: str = "ferminet",
        envelope_type: Optional[str] = None,
        smooth_cuspless: Optional[bool] = None,
        use_cusp_jastrow: bool = False,
        jastrow: Optional[nn.Module] = None,
        spin_adapted: bool = False,
        spin_projector: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if n_up < 0 or n_down < 0 or n_up + n_down == 0:
            raise ValueError('Electron counts must be nonnegative and nonempty')
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.num_determinants = num_determinants
        self.hidden_dim_one = hidden_dim_one
        self.hidden_dim_two = hidden_dim_two
        self.num_layers = num_layers
        self.eps = eps
        self.interaction_type = interaction_type.lower()

        self.register_buffer("nuclear_coords", nuclear_coords.clone().detach())
        self.register_buffer("nuclear_charges", nuclear_charges.clone().detach())
        num_nuclei = nuclear_coords.shape[0]
        self.num_nuclei = num_nuclei

        eff_env, eff_smooth = cusp_defaults(
            envelope_type, smooth_cuspless, jastrow is not None or use_cusp_jastrow, jastrow)
        self.envelope_type = eff_env
        self.smooth_cuspless = eff_smooth

        # 1. 几何特征提取层
        self.embedding = ElectronFeatureEmbedding(eps=eps, smooth_cuspless=eff_smooth)
        in_dim_one, in_dim_two = ElectronFeatureEmbedding.feature_dims(
            num_nuclei=num_nuclei, use_nuclear_charges=True
        )

        # 2. 等变主干交互层选型
        if self.interaction_type == "psiformer":
            # 采用 PsiformerAttentionBlock 消除显式双电子流
            self.input_proj = nn.Linear(in_dim_one, hidden_dim_one)
            attn_layers = []
            for _ in range(num_layers):
                attn_layers.append(
                    PsiformerAttentionBlock(
                        embed_dim=hidden_dim_one,
                        num_heads=max(1, hidden_dim_one // 16),
                        n_up=n_up,
                        n_down=n_down,
                    )
                )
            self.layers = nn.ModuleList(attn_layers)
        else:
            # 传统自旋等变层
            fermi_layers = []
            for l in range(num_layers):
                d_in_1 = in_dim_one if l == 0 else hidden_dim_one
                d_in_2 = in_dim_two if l == 0 else hidden_dim_two
                fermi_layers.append(
                    FermiInteractionLayerModern(
                        d_in_single=d_in_1,
                        d_in_double=d_in_2,
                        d_out_single=hidden_dim_one,
                        d_out_double=hidden_dim_two,
                        n_up=n_up,
                        n_down=n_down,
                    )
                )
            self.layers = nn.ModuleList(fermi_layers)

        # 3. 轨道投影与核包络
        self.projector_up = (
            OrbitalEnvelopeProjector(
                hidden_dim=hidden_dim_one,
                num_determinants=num_determinants,
                num_orbitals=n_up,
                nuclear_coords=nuclear_coords,
                envelope_type=eff_env,
                nuclear_charges=nuclear_charges,
            )
            if n_up > 0 else None
        )
        self.projector_down = (
            OrbitalEnvelopeProjector(
                hidden_dim=hidden_dim_one,
                num_determinants=num_determinants,
                num_orbitals=n_down,
                nuclear_coords=nuclear_coords,
                envelope_type=eff_env,
                nuclear_charges=nuclear_charges,
            )
            if n_down > 0 else None
        )

        # 4. 多行列式混合系数
        self.det_weights = nn.Parameter(torch.ones(num_determinants) / float(num_determinants))

        self.spin_projector = resolve_spin(n_up, n_down, spin_adapted, spin_projector)
        pair_spin = pair_spin_for(n_up, n_down, self.spin_projector)
        if jastrow is not None:
            self.jastrow = jastrow
        elif use_cusp_jastrow:
            self.jastrow = KatoCuspJastrow(
                nuclear_charges, n_up, n_down, eps=eps, two_electron_spin=pair_spin)
        else:
            self.jastrow = None
        validate_pair_cusp(self.jastrow, pair_spin)

    def forward_orbitals(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """计算分子轨道矩阵。"""
        if geom_ctx is None or not geom_ctx.valid_for(r, self.eps):
            geom_ctx = GeometricContext.compute(
                r=r, nuclear_coords=self.nuclear_coords, eps=self.eps
            )

        # 1. 提取几何特征 (共享 geom_ctx)
        h1, h2 = self.embedding(r, self.nuclear_coords, self.nuclear_charges, geom_ctx=geom_ctx)

        # 2. 等变前向
        if self.interaction_type == "psiformer":
            h = self.input_proj(h1)
            for block in self.layers:
                h = block(h, r_ee=geom_ctx.r_ee, d_ee=geom_ctx.d_ee)
            h1_final = h
        else:
            cur_h1, cur_h2 = h1, h2
            for layer in self.layers:
                cur_h1, cur_h2 = layer(cur_h1, cur_h2)
            h1_final = cur_h1

        # 3. 构造上下自旋轨道
        d_up = None
        if self.projector_up is not None:
            h1_up = h1_final[:, :self.n_up, :]
            dist_up = geom_ctx.d_en[:, :self.n_up, :, 0]
            d_up = self.projector_up(h1_up, dist_up)

        d_down = None
        if self.projector_down is not None:
            h1_down = h1_final[:, self.n_up:, :]
            dist_down = geom_ctx.d_en[:, self.n_up:, :, 0]
            d_down = self.projector_down(h1_down, dist_down)

        return d_up, d_down

    def _forward_log_psi_unprojected(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算未进行自旋投影的基础对数波函数 (轨道行列式 + 解析 Jastrow)。"""
        if geom_ctx is None or not geom_ctx.valid_for(r, self.eps):
            geom_ctx = GeometricContext.compute(
                r=r, nuclear_coords=self.nuclear_coords, eps=self.eps
            )

        d_up, d_down = self.forward_orbitals(r, geom_ctx=geom_ctx)
        log_abs_psi, sign_psi = BranchlessCofactorDeterminant.compute(
            d_up=d_up,
            d_down=d_down,
            det_weights=self.det_weights,
            n_up=self.n_up,
            n_down=self.n_down,
            num_determinants=self.num_determinants,
        )

        if self.jastrow is not None:
            log_j = self.jastrow(r, self.nuclear_coords, geom_ctx=geom_ctx)
            log_abs_psi = log_abs_psi + log_j

        return log_abs_psi, sign_psi

    def forward_log_psi(
        self,
        r: torch.Tensor,
        geom_ctx: Optional[GeometricContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """端到端计算总对数波函数 (log|Psi|, sign)，支持自旋空间投影。"""
        if self.spin_projector is not None:
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

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_log_psi(r)
