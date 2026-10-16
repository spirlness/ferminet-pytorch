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


class _CofactorDeterminant(torch.autograd.Function):
    """Determinant with inverse-free derivatives, including at singular matrices.

    Used only for singular branches. Recursing on minors in backward preserves
    higher derivatives at rank-deficient matrices without differentiating an SVD.
    """

    @staticmethod
    def forward(ctx, matrix: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(matrix, log_scale)
        sign, log_abs = torch.linalg.slogdet(matrix)
        return sign * torch.exp(log_abs + log_scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        matrix, log_scale = ctx.saved_tensors
        size = matrix.shape[-1]
        if size == 1:
            derivative = grad_output * torch.exp(log_scale)
            return derivative[..., None, None] + matrix * 0.0, None

        # 尺寸截断：根据 FINDING-PYT-04，小尺寸 (size <= 6，兼容单自旋通道 <= 3 构成的分块对角矩阵) 保持精确代数展开
        # size > 6 时自动切换至基于 Tikhonov 阻尼正则化的伴随矩阵 (O(N^3) 复杂度)，杜绝 (N!)^2 阶乘爆炸
        if size <= 6:
            row_idx = torch.arange(size, device=matrix.device).unsqueeze(1)
            col_idx = torch.arange(size - 1, device=matrix.device).unsqueeze(0)
            keep = torch.where(col_idx >= row_idx, col_idx + 1, col_idx)
            minors = matrix[..., keep[:, None, :, None], keep[None, :, None, :]]
            indices = torch.arange(size, device=matrix.device)
            parity = 1 - 2 * ((indices[:, None] + indices[None, :]) % 2)
            cofactors = parity * _CofactorDeterminant.apply(minors, log_scale[..., None, None])
            return grad_output[..., None, None] * cofactors, None
        else:
            # 尺寸截断：size > 4 切换至基于 Tikhonov 阻尼正则化的伴随矩阵 (O(N^3) 复杂度)
            eye = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
            damping = 1e-6
            matrix_damped = matrix + damping * eye
            sign, log_abs = torch.linalg.slogdet(matrix_damped)
            inv_mat = torch.linalg.inv(matrix_damped)
            det_val = sign * torch.exp(log_abs + log_scale)
            cofactors = (det_val[..., None, None] * inv_mat).transpose(-1, -2)
            return grad_output[..., None, None] * cofactors, None


def _determinant_terms(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return amplitude and detached log scale for a batch of square matrices.

    det(matrix) = amplitude * exp(log_scale). Ordinary branches retain slogdet
    scaling. Singular branches are placeholders, repaired jointly for both spins
    after choosing the scale of the total wavefunction.
    """
    with torch.no_grad():
        singular = torch.linalg.slogdet(matrix)[0] == 0

    # Mask BEFORE slogdet: masking its output cannot prevent 0 * inf in backward.
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    safe_matrix = torch.where(singular[..., None, None], identity, matrix)
    sign, log_abs = torch.linalg.slogdet(safe_matrix)
    log_scale = log_abs.detach()
    amplitude = torch.where(
        singular, torch.zeros_like(sign), sign * torch.exp(log_abs - log_scale)
    )
    return amplitude, log_scale


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
        eff_env_type = envelope_type if envelope_type is not None else ("gaussian" if has_jastrow else "exponential")
        eff_smooth = smooth_cuspless if smooth_cuspless is not None else True

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

        # 5. Resolve the target spin before choosing the two-electron cusp.
        if spin_projector is not None:
            self.spin_projector = spin_projector
        elif spin_adapted:
            spk = dict(spin_projector_kwargs or {})
            # 判定开壳层与自旋未配对体系:
            # 闭壳层单重态要求 n_up == n_down；若 n_up != n_down 则为自旋未配对体系，默认安全降级为 mode="none"
            is_open_shell = (n_up != n_down)
            if is_open_shell:
                spk.setdefault("mode", "none")
            else:
                spk.setdefault("mode", "singlet")
            self.spin_projector = SpinProjector(
                n_up=n_up,
                n_down=n_down,
                **spk,
            )
        else:
            self.spin_projector = None

        # 6. Kato 尖峰 Jastrow 因子
        pair_spin = None
        if self.total_electrons == 2 and isinstance(self.spin_projector, SpinProjector):
            pair_spin = {"singlet": 0, "triplet": 1}.get(self.spin_projector.mode)
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
        if pair_spin is not None and isinstance(self.jastrow, KatoCuspJastrow):
            expected_cusp = 0.25 if pair_spin == 1 else 0.5
            if self.jastrow.cusp_ee and self.jastrow.gamma_ee[0, 1].item() != expected_cusp:
                raise ValueError(
                    f"Jastrow cusp is incompatible with the projected spin; "
                    f"construct KatoCuspJastrow with two_electron_spin={pair_spin}"
                )

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

        # Keep singular determinant derivatives even when the total Psi is nonzero.
        if d_up is not None:
            amplitude_up, log_scale_up = _determinant_terms(d_up)
        else:
            amplitude_up = torch.ones(batch_size, self.num_determinants, device=r.device, dtype=r.dtype)
            log_scale_up = torch.zeros_like(amplitude_up)

        if d_down is not None:
            amplitude_down, log_scale_down = _determinant_terms(d_down)
        else:
            amplitude_down = torch.ones(batch_size, self.num_determinants, device=r.device, dtype=r.dtype)
            log_scale_down = torch.zeros_like(amplitude_down)

        # 合并上下自旋行列式
        amplitude_k = amplitude_up * amplitude_down  # (B, K)
        log_scale_k = log_scale_up + log_scale_down  # (B, K)

        # 多行列式加权求和: Psi = sum_k w_k * det_k
        w = self.det_weights.unsqueeze(0)  # (1, K)
        # LogSumExp 技巧保证数值稳定
        singular = amplitude_k.detach() == 0
        active_log_scale = log_scale_k.masked_fill(singular, float("-inf"))
        max_log = torch.amax(active_log_scale, dim=1, keepdim=True)  # (B, 1)
        max_log = torch.where(torch.isneginf(max_log), torch.zeros_like(max_log), max_log)

        relative_scale = torch.where(singular, torch.zeros_like(log_scale_k), log_scale_k - max_log)
        scaled_det = amplitude_k * torch.exp(relative_scale)
        # 无分支纯张量向量化代数余子式修正 (Branchless Cofactor Correction):
        # 消除 if singular.any() 与布尔切片，确保符号追踪 0 Graph Break
        if d_up is None:
            block = d_down
        elif d_down is None:
            block = d_up
        else:
            zeros = d_up.new_zeros(batch_size, self.num_determinants, self.n_up, self.n_down)
            block = torch.cat((torch.cat((d_up, zeros), dim=-1),
                               torch.cat((zeros.transpose(-1, -2), d_down), dim=-1)), dim=-2)
        row_scale = block.detach().abs().amax(dim=-1, keepdim=True)
        # Borrow the other branches' row scale for an exactly zero orbital.
        fallback = row_scale.amax(dim=1, keepdim=True)
        row_scale = torch.where(row_scale == 0, fallback, row_scale)
        row_scale = torch.where(row_scale == 0, torch.ones_like(row_scale), row_scale)
        normalized = block / row_scale
        correction_scale = row_scale.log().sum(dim=(-1, -2)) - max_log
        correction = _CofactorDeterminant.apply(normalized, correction_scale)
        # Preserve the exact LU zero even if normalization perturbs it.
        # correction - correction.detach() 在前向精确为 0，反向保留代数伴随矩阵导数
        scaled_det = torch.where(singular, correction - correction.detach(), scaled_det)
        summed = torch.sum(w * scaled_det, dim=1)  # (B,)

        abs_summed = torch.abs(summed)
        zero_mask = abs_summed == 0

        sign_psi = torch.sign(summed)
        sign_psi = torch.where(zero_mask, torch.zeros_like(sign_psi), sign_psi)

        safe_abs_summed = torch.clamp(abs_summed, min=1e-20)
        log_abs_psi = torch.where(
            zero_mask,
            torch.full_like(summed, float("-inf")),
            max_log.squeeze(1) + torch.log(safe_abs_summed),
        )

        return log_abs_psi, sign_psi

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


