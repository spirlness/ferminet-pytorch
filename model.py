import warnings
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from .features import ElectronFeatureEmbedding
    from .layers import FermiInteractionLayer
    from .envelope import NuclearEnvelope
except (ImportError, ValueError):
    from features import ElectronFeatureEmbedding
    from layers import FermiInteractionLayer
    from envelope import NuclearEnvelope


def _determinant_polynomial(matrix: torch.Tensor) -> torch.Tensor:
    """Compute a determinant with Berkowitz's division-free recurrence.

    The result is a polynomial in the matrix entries, so its higher
    derivatives are defined at singular matrices.  LU/slogdet derivatives use
    an inverse and therefore fail when a single FermiNet determinant branch is
    zero although the full determinant sum is nonzero.
    """
    n = matrix.shape[-1]
    if n == 1:
        return matrix[..., 0, 0]

    coefficients: Optional[torch.Tensor] = None
    for size in range(1, n + 1):
        block = matrix[..., n - size:n, n - size:n]
        if size == 1:
            coefficients = torch.stack(
                [torch.ones_like(block[..., 0, 0]), -block[..., 0, 0]], dim=-1
            )
            continue

        assert coefficients is not None
        a00, row, column = block[..., 0, 0], block[..., 0, 1:], block[..., 1:, 0]
        trailing = block[..., 1:, 1:]
        first_column = [torch.ones_like(a00), -a00]
        power_column = column
        for _ in range(size - 1):
            first_column.append(
                -(row.unsqueeze(-2) @ power_column.unsqueeze(-1)).squeeze((-1, -2))
            )
            power_column = (trailing @ power_column.unsqueeze(-1)).squeeze(-1)

        zero = torch.zeros_like(a00)
        toeplitz = torch.stack(
            [
                torch.stack(
                    [first_column[i - j] if i >= j else zero for j in range(size)], dim=-1
                )
                for i in range(size + 1)
            ],
            dim=-2,
        )
        coefficients = (toeplitz @ coefficients.unsqueeze(-1)).squeeze(-1)
    return ((-1) ** n) * coefficients[..., -1]


def _scaled_determinant(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return division-free determinant mantissas and per-branch log scales.

    Each row is divided by its own detached maximum magnitude before evaluating
    the determinant polynomial.  This keeps the polynomial in a safe numerical
    range even when different orbitals have very different magnitudes.  A zero
    row uses a scale of one: its determinant remains zero while its cofactor
    derivative remains available.  The identity

    ``det(A) = det(A / row_scale) * prod(row_scale)``

    is exact, and detaching the representation scale preserves all derivatives
    of the determinant polynomial, including at singular matrices.  Unlike an
    LU/SVD backward pass, this path never divides by a singular value and works
    with reverse-mode autograd, ``torch.func.jvp`` and ``vmap`` on CPU or CUDA.

    Non-finite matrix entries deliberately remain non-finite in the mantissa so
    callers cannot silently discard a corrupted determinant branch.
    """
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"determinant requires square matrices, got {tuple(matrix.shape)}")

    row_scale = torch.amax(torch.abs(matrix).detach(), dim=-1)
    usable_scale = torch.isfinite(row_scale) & (row_scale > 0)
    safe_row_scale = torch.where(usable_scale, row_scale, torch.ones_like(row_scale))
    scaled_matrix = matrix / safe_row_scale.unsqueeze(-1)
    row_log_scale = torch.sum(torch.log(safe_row_scale), dim=-1)

    # LU gives a stable magnitude for regular matrices, but its backward uses an
    # inverse and is undefined on a singular branch.  Rank classification is
    # deliberately outside autograd: no singular vectors or singular values are
    # differentiated.  This also catches repeated rows that an LU determinant can
    # report as a tiny nonzero number because of roundoff.  Replacing each inactive
    # branch by identity prevents a zero upstream gradient from still entering a
    # singular LU solve or an unnecessarily ill-conditioned polynomial.
    with torch.no_grad():
        finite_matrix = torch.isfinite(scaled_matrix).all(dim=(-2, -1))
        probe_sign, _ = torch.linalg.slogdet(scaled_matrix)
    eye = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    ).expand_as(scaled_matrix)
    with torch.no_grad():
        rank_probe = torch.where(finite_matrix[..., None, None], scaled_matrix, eye)
        exact_zero_row = (scaled_matrix == 0).all(dim=-1).any(dim=-1)
        exact_zero_column = (scaled_matrix == 0).all(dim=-2).any(dim=-1)
        exact_singular = (probe_sign == 0) | exact_zero_row | exact_zero_column
        # Rank tolerances are useful for float64 repeated-row matrices, but the
        # default float32 tolerance would incorrectly turn well-resolved small
        # determinants in large systems into nodes.  For small float32 blocks,
        # also recognize literal duplicate rows/columns without an SVD.
        if matrix.dtype == torch.float64 and matrix.shape[-1] <= 64:
            singular = finite_matrix & (
                torch.linalg.matrix_rank(rank_probe) < matrix.shape[-1]
            )
        elif matrix.shape[-1] <= 16:
            row_equal = (
                scaled_matrix.unsqueeze(-3) == scaled_matrix.unsqueeze(-2)
            ).all(dim=-1)
            column_equal = (
                scaled_matrix.transpose(-1, -2).unsqueeze(-3)
                == scaled_matrix.transpose(-1, -2).unsqueeze(-2)
            ).all(dim=-1)
            duplicate = row_equal.triu(diagonal=1).any(dim=(-1, -2)) | column_equal.triu(
                diagonal=1
            ).any(dim=(-1, -2))
            singular = finite_matrix & (exact_singular | duplicate)
        else:
            singular = finite_matrix & exact_singular
    singular_matrix = torch.where(singular[..., None, None], scaled_matrix, eye)
    regular_matrix = torch.where(singular[..., None, None], eye, scaled_matrix)

    singular_determinant = _determinant_polynomial(singular_matrix)
    # A numerically classified singular matrix represents an exact node in the
    # floating-point model.  Force the forward value to zero while retaining the
    # polynomial's cofactor and higher-order derivatives.
    singular_determinant = singular_determinant - singular_determinant.detach()
    regular_sign, regular_log_abs = torch.linalg.slogdet(regular_matrix)
    determinant = torch.where(singular, singular_determinant, regular_sign)
    log_scale = torch.where(
        singular, row_log_scale, row_log_scale + regular_log_abs
    )
    return determinant, log_scale


class FermiNet(nn.Module):
    """
    FermiNet (Fermi Neural Network) 深度波函数模型。

    参考论文:
        Pfau et al., "Ab-Initio Solution of the Many-Electron Schrödinger Equation
        with Deep Neural Networks", Phys. Rev. Research 2, 033429 (2020).

    核心架构：
    1. 几何特征提取：全向量化计算电子-核、电子-电子差向量及欧几里得距离。
    2. 自旋等变前馈层：多层 FermiInteractionLayer，自旋向上与向下电子分别进行均值池化交互。
       注意末层的双电子流输出不被消费 (分子轨道全部由单电子流 h1 构造)，
       因此末层不构造双电子流更新网络 (`linear_two` / `res_proj_two` 被移除)，
       仅保留单电子流更新路径；末层的 `forward` 返回 `(h1_next, None)`。
    3. 广义分子轨道构建：最高层单电子特征经线性投影为 Backflow 项，与原子核指数包络 (NuclearEnvelope) 相乘。
    4. 反对称波函数计算：根据泡利不相容原理，将自旋向上和向下电子分别构造成 Slater 行列式矩阵，
       计算对角块行列式乘积并在多行列式 (Multi-determinants) 间进行线性组合。
    5. 数值稳定性机制：提供基于 LogSumExp 技巧的稳定对数波函数 (log|Psi|, sign) 计算接口。

    关于核几何：
    - 本模型网络关于同自旋电子置换是严格等变的，但不具备旋转/平移波函数不变性以外的
      哈密顿量全对称群：平移不变性成立 (特征为相对坐标)，旋转不变性与核置换对称性不在
      本实现中强制 (核序置换对称性可通过 ``tie_nuclear_envelope=True`` 时向 NuclearEnvelope
      注入核电荷向量部分恢复——同电荷核共享包络参数)。
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
        use_nuclear_charges: bool = True,
        anisotropic_envelope: bool = False,
        tie_nuclear_envelope: bool = True,
        rotation_invariant: bool = False,
        compile: bool = False,
        compile_mode: str = "default",
        compile_backend: str = "inductor",
    ) -> None:
        """
        初始化 FermiNet 模型。

        Args:
            nuclear_coords (torch.Tensor): 原子核空间三维笛卡尔坐标。
                - Shape: `(M, 3)`，M 为原子核数目。
            nuclear_charges (torch.Tensor): 原子核电荷量 (原子序数 Z)。
                - Shape: `(M,)`。
            n_up (int): 自旋向上电子数 (N_up)。
            n_down (int): 自旋向下电子数 (N_down)。
            num_determinants (int): 行列式数量 (K)，增强多组态波函数表达能力。
            hidden_dim_one (int): 单电子流交互隐藏层特征维度。
            hidden_dim_two (int): 双电子流交互隐藏层特征维度。
            num_layers (int): 自旋等变相互作用层 (FermiInteractionLayer) 层数。
                末层不构造双电子流更新网络 (其 h2 输出不被消费)。
            eps (float): 数值微小量，防止除零或 log(0)。语义为距离平方的 clamp 下限
                (最小平方距离)，软化半径为 sqrt(eps)；默认 1e-24 对应 1e-12 Bohr。
            use_nuclear_charges (bool): 是否在几何特征中注入核电荷 Z_I (默认 True)。
                旋转不变模式下单核特征由 1 维 [d_{iI}] 提升为 2 维 [d_{iI}, Z_I]；
                笛卡尔模式下由 4 维提升为 5 维。
            anisotropic_envelope (bool): 是否启用各向异性可学习三维主轴度规核包络 (默认 False)。
            tie_nuclear_envelope (bool): 是否向 NuclearEnvelope 传入核电荷向量，使同电荷核
                共享包络参数 (核置换等变)；默认 True。置为 False 时包络按核索引独立参数化
                (旧行为)。
            rotation_invariant (bool): 是否使用旋转不变特征 (默认 False)。开启时特征只由
                标量不变量 (距离与核电荷) 构成，波函数对整体旋转**严格不变**；关闭时
                额外拼接笛卡尔位移向量，方向信息更丰富但不具备旋转不变性。笛卡尔模式
                是公共默认，用于表示一般空间态；它不自动保证
                特定角动量、宇称或纯总自旋态，也不构成基态收敛证明。
            compile (bool): 是否启用 PyTorch 2.0+ JIT 图编译优化 (默认为 False，按需显式启用)。
            compile_mode (str): 默认编译优化模式 (默认为 'default'，也可选 'reduce-overhead' 等)。
            compile_backend (str): 编译后端 (默认为 'inductor')。
        """
        super().__init__()
        assert n_up >= 0 and n_down >= 0, "电子数必须非负"
        assert n_up + n_down > 0, "体系必须包含至少一个电子"
        if rotation_invariant and anisotropic_envelope:
            raise ValueError(
                "rotation_invariant=True 与 anisotropic_envelope=True 不可同时使用："
                "实验室坐标轴各向异性包络会破坏整体旋转不变性。"
            )

        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.num_determinants = num_determinants
        self.hidden_dim_one = hidden_dim_one
        self.hidden_dim_two = hidden_dim_two
        self.num_layers = num_layers
        self.eps = eps
        self.use_nuclear_charges = use_nuclear_charges
        self.anisotropic_envelope = anisotropic_envelope
        self.tie_nuclear_envelope = tie_nuclear_envelope
        self.rotation_invariant = rotation_invariant
        self.use_compile = compile
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend

        # 精度告警状态标记 (防范单精度大体系病态数值下溢)
        self._warned_precision = False
        if self.total_electrons > 16 and nuclear_coords.dtype != torch.float64:
            warnings.warn(
                f"检测到多电子体系 (总电子数 N = {self.total_electrons} > 16) 且参数为单精度 ({nuclear_coords.dtype})。"
                "随着电子数增加，Slater 行列式乘积可能发生数值下溢，建议提升至双精度 (torch.float64) 以确保数值稳定性与计算精度。",
                UserWarning,
                stacklevel=2,
            )
            self._warned_precision = True

        # 编译运行时句柄与缓存
        self._compiled_forward = None
        self._compiled_forward_log_psi = None
        self._compiled_device = None

        # 注册非训练参数缓冲 (Buffer)
        self.register_buffer("nuclear_coords", nuclear_coords.clone().detach())
        self.register_buffer("nuclear_charges", nuclear_charges.clone().detach())

        num_nuclei = nuclear_coords.shape[0]
        self.num_nuclei = num_nuclei

        # 1. 几何特征提取模块
        self.embedding = ElectronFeatureEmbedding(
            eps=eps, rotation_invariant=rotation_invariant
        )
        # 输入维度由特征模块统一定义，避免两处重复推导
        in_dim_one, in_dim_two = ElectronFeatureEmbedding.feature_dims(
            num_nuclei=num_nuclei,
            use_nuclear_charges=use_nuclear_charges,
            rotation_invariant=rotation_invariant,
        )
        self.in_dim_one = in_dim_one
        self.in_dim_two = in_dim_two

        # 2. 自旋等变相互作用层序列
        # 末层的双电子流输出不被任何后续模块消费 (轨道仅由 h1 构造)，
        # 故末层不构造双电子流更新网络，避免产生恒收不到梯度的空转参数。
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

        # D5: tie_nuclear_envelope=True 时向包络传入核电荷向量 (同电荷核共享包络参数)。
        # detach().clone() 避免与模型 buffer 共享存储。
        envelope_nuclear_charges: Optional[torch.Tensor] = (
            self.nuclear_charges.detach().clone() if tie_nuclear_envelope else None
        )

        # 3. 轨道投影层 (Backflow 变换) 与原子核指数包络 (Envelope)
        if self.n_up > 0:
            # 映射为 K * n_up 维
            self.linear_orb_up = nn.Linear(hidden_dim_one, num_determinants * n_up)
            self.envelope_up = NuclearEnvelope(
                num_determinants=num_determinants,
                num_orbitals=n_up,
                num_nuclei=num_nuclei,
                eps=eps,
                anisotropic=anisotropic_envelope,
                nuclear_charges=envelope_nuclear_charges,
            )
        else:
            self.linear_orb_up = None
            self.envelope_up = None

        if self.n_down > 0:
            # 映射为 K * n_down 维
            self.linear_orb_down = nn.Linear(hidden_dim_one, num_determinants * n_down)
            self.envelope_down = NuclearEnvelope(
                num_determinants=num_determinants,
                num_orbitals=n_down,
                num_nuclei=num_nuclei,
                eps=eps,
                anisotropic=anisotropic_envelope,
                nuclear_charges=envelope_nuclear_charges,
            )
        else:
            self.linear_orb_down = None
            self.envelope_down = None

        # 4. 多行列式线性组合系数 (Determinant Weights)
        # 初始化为均等权重 1.0 / sqrt(K)
        self.det_weights = nn.Parameter(
            torch.ones(num_determinants) / (num_determinants ** 0.5)
        )

    def forward_orbitals(
        self, r: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向计算分子轨道方阵 D_up 与 D_down。

        Args:
            r (torch.Tensor): 电子三维空间坐标。
                - Shape: `(B, N, 3)`，其中 N = n_up + n_down。
                - Dtype: `torch.float32` 或 `torch.float64`。

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                - d_up (Optional[torch.Tensor]): 自旋向上电子的分子轨道矩阵。
                    - Shape: `(B, K, n_up, n_up)`。
                    - Dtype: 与 r 一致。若 n_up == 0 则为 None。
                - d_down (Optional[torch.Tensor]): 自旋向下电子的分子轨道矩阵。
                    - Shape: `(B, K, n_down, n_down)`。
                    - Dtype: 与 r 一致。若 n_down == 0 则为 None。
        """
        batch_size, num_electrons, _ = r.shape
        assert num_electrons == self.total_electrons, (
            f"电子数不匹配: 输入包含 {num_electrons} 电子，而模型要求 {self.total_electrons}"
        )

        if self.total_electrons > 16 and r.dtype != torch.float64 and not getattr(self, "_warned_precision", False):
            warnings.warn(
                f"当前多电子体系包含 {self.total_electrons} 个电子 (> 16)，在单精度 ({r.dtype}) 下可能面临行列式数值下溢或病态失效。"
                "强烈建议使用双精度 (torch.float64) 以保障波函数与能量计算的数值稳定性与精度。",
                UserWarning,
                stacklevel=2,
            )
            self._warned_precision = True

        # 1. 提取几何特征 (rotation_invariant=True 时只含标量不变量；use_nuclear_charges
        #    控制是否注入核电荷 Z_I)
        if self.use_nuclear_charges:
            h1, h2 = self.embedding(r, self.nuclear_coords, self.nuclear_charges)
        else:
            h1, h2 = self.embedding(r, self.nuclear_coords)

        # 2. 等变前馈交互传播
        # 末层 produce_two_electron_output=False，循环结束时 h2 为 None (不再被引用，
        # 后续轨道构造仅使用 h1)。
        for layer in self.layers:
            h1, h2 = layer(h1, h2, self.n_up, self.n_down)

        # 3. 构造自旋向上轨道
        d_up: Optional[torch.Tensor] = None
        if self.n_up > 0 and self.linear_orb_up is not None and self.envelope_up is not None:
            # 切片出前 n_up 个电子的单电子特征与空间坐标
            h1_up = h1[:, : self.n_up, :]  # (B, n_up, hidden_dim_one)
            r_up = r[:, : self.n_up, :]    # (B, n_up, 3)

            # Backflow 映射: (B, n_up, K * n_up) -> (B, n_up, K, n_up) -> (B, K, n_up, n_up)
            backflow_up = self.linear_orb_up(h1_up)
            backflow_up = backflow_up.view(batch_size, self.n_up, self.num_determinants, self.n_up)
            backflow_up = backflow_up.permute(0, 2, 1, 3)  # (B, K, n_up, n_up)

            # 指数衰减包络: (B, K, n_up, n_up)
            env_up = self.envelope_up(r_up, self.nuclear_coords)

            # 最终轨道矩阵: 逐元素哈达玛积
            d_up = backflow_up * env_up

        # 4. 构造自旋向下轨道
        d_down: Optional[torch.Tensor] = None
        if self.n_down > 0 and self.linear_orb_down is not None and self.envelope_down is not None:
            # 切片出后 n_down 个电子的特征与空间坐标
            h1_down = h1[:, self.n_up :, :]  # (B, n_down, hidden_dim_one)
            r_down = r[:, self.n_up :, :]    # (B, n_down, 3)

            # Backflow 映射: (B, n_down, K * n_down) -> (B, K, n_down, n_down)
            backflow_down = self.linear_orb_down(h1_down)
            backflow_down = backflow_down.view(
                batch_size, self.n_down, self.num_determinants, self.n_down
            )
            backflow_down = backflow_down.permute(0, 2, 1, 3)  # (B, K, n_down, n_down)

            # 指数衰减包络: (B, K, n_down, n_down)
            env_down = self.envelope_down(r_down, self.nuclear_coords)

            # 最终轨道矩阵
            d_down = backflow_down * env_down

        return d_up, d_down

    def _raw_forward_log_psi(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """原生 Eager 模式下的对数波函数前向计算逻辑."""
        batch_size = r.shape[0]
        d_up, d_down = self.forward_orbitals(r)

        # 1. Division-free determinants retain derivatives at singular branches.
        # Each determinant gets its own detached row-wise representation scale.
        if d_up is not None:
            det_up, log_scale_up = _scaled_determinant(d_up)
        else:
            det_up = torch.ones(batch_size, self.num_determinants, device=r.device, dtype=r.dtype)
            log_scale_up = torch.zeros_like(det_up)

        # 2. 计算自旋向下块行列式的对数绝对值与符号
        if d_down is not None:
            det_down, log_scale_down = _scaled_determinant(d_down)
        else:
            det_down = torch.ones(batch_size, self.num_determinants, device=r.device, dtype=r.dtype)
            log_scale_down = torch.zeros_like(det_down)

        # 3. Det_k = Det_up_k * Det_down_k.  Do not take a branch log here:
        # log(0) would discard the valid derivative of a zero determinant.
        det_k = det_up * det_down
        log_det_scale = log_scale_up + log_scale_down

        # 4. 多行列式带权重线性组合: Psi = sum_k w_k * Det_k
        #
        # 4. Detached common scales prevent under/overflow without changing the
        # represented nonzero wavefunction or blocking gradients at zero weights.
        w = self.det_weights.unsqueeze(0)  # (1, K)
        weight_scale = torch.amax(torch.abs(w).detach())
        weight_scale_safe = torch.where(weight_scale > 0, weight_scale, torch.ones_like(weight_scale))
        # Scale only after the up/down determinant product has been represented.
        # Per-spin normalization would erase reciprocal scales such as
        # (1e200 * 1e-200), incorrectly creating a wavefunction node.
        common_log_scale = torch.amax(log_det_scale.detach(), dim=1, keepdim=True)
        branch_scale = torch.exp(log_det_scale - common_log_scale)
        summed = torch.sum((w / weight_scale_safe) * det_k * branch_scale, dim=1)
        is_strictly_zero = (weight_scale == 0) | (summed == 0)

        sign_psi = torch.where(is_strictly_zero, torch.zeros_like(summed), torch.sign(summed))
        log_abs_psi = torch.where(
            is_strictly_zero,
            torch.full_like(summed, -float("inf")),
            common_log_scale.squeeze(1)
            + torch.log(weight_scale_safe)
            + torch.log(torch.abs(summed)),
        )

        return log_abs_psi, sign_psi

    def _raw_forward(self, r: torch.Tensor) -> torch.Tensor:
        """原生 Eager 模式下的波函数标量前向计算逻辑."""
        log_abs_psi, sign_psi = self._raw_forward_log_psi(r)
        psi = sign_psi * torch.exp(log_abs_psi)

        # 防范大体系标量数值下溢为 0.0 造成的静默计算失效
        if torch.any(psi == 0.0) and torch.any(torch.isfinite(log_abs_psi) & (sign_psi != 0)):
            if not getattr(self, "_warned_underflow", False):
                warnings.warn(
                    f"标量波函数幅值计算发生浮点数下溢 (Psi -> 0.0, 当前 log|Psi| 极小值约为 {log_abs_psi.min().item():.2f})。"
                    "在多电子大体系中，强烈建议使用 forward_log_psi 接口获取 (log|Psi|, sign) 以确保数值严谨性与稳定性。",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_underflow = True
        return psi

    def forward_log_psi(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        高数值稳定性前向计算波函数的对数绝对值与符号: (log|Psi(r)|, sign(Psi(r)))。
        在 CUDA 设备上默认自动集成并调用已编译的高性能图执行内核 (Compiled reduce-overhead)；
        在 CPU 设备或调试模式下自动平滑回退至原生 Eager 解释执行。

        Args:
            r (torch.Tensor): 电子空间三维坐标，Shape: `(B, N, 3)`。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (log|Psi(r)|, sign(Psi(r)))。
        """
        # CPU 设备或禁用编译时，直接走原生高效 Eager 路径，避免 Windows CPU C++ 编译器配置冲突
        if not self.use_compile or not r.is_cuda:
            return self._raw_forward_log_psi(r)

        # 若运行设备切换，重置编译句柄
        if self._compiled_device != r.device:
            self._compiled_forward_log_psi = None
            self._compiled_forward = None
            self._compiled_device = r.device

        if self._compiled_forward_log_psi is None:
            try:
                self._compiled_forward_log_psi = torch.compile(
                    self._raw_forward_log_psi,
                    backend=self.compile_backend,
                    mode=self.compile_mode,
                )
            except Exception:
                return self._raw_forward_log_psi(r)

        try:
            return self._compiled_forward_log_psi(r)
        except Exception:
            # 遭遇动态图或未知边界时安全回退至原生实现
            return self._raw_forward_log_psi(r)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        前向计算波函数标量幅值: Psi(r) = sign(Psi(r)) * exp(log|Psi(r)|)。
        在 CUDA 设备上默认自动集成并调用已编译的高性能图执行内核 (Compiled reduce-overhead)；
        在 CPU 设备或调试模式下自动平滑回退至原生 Eager 解释执行。

        Args:
            r (torch.Tensor): 电子空间三维坐标，Shape: `(B, N, 3)`。

        Returns:
            torch.Tensor: 体系波函数真实标量值 Psi(r)，Shape: `(B,)`。
        """
        if not self.use_compile or not r.is_cuda:
            return self._raw_forward(r)

        if self._compiled_device != r.device:
            self._compiled_forward_log_psi = None
            self._compiled_forward = None
            self._compiled_device = r.device

        if self._compiled_forward is None:
            try:
                self._compiled_forward = torch.compile(
                    self._raw_forward,
                    backend=self.compile_backend,
                    mode=self.compile_mode,
                )
            except Exception:
                return self._raw_forward(r)

        try:
            return self._compiled_forward(r)
        except Exception:
            return self._raw_forward(r)

    def enable_compile(
        self,
        mode: str = "default",
        backend: str = "inductor",
    ) -> "FermiNet":
        """
        启用 JIT 编译优化 (默认模式为 'default'，也可显式指定 'reduce-overhead')。

        Args:
            mode (str): 编译模式，默认为 'default'。
            backend (str): 编译后端，默认为 'inductor'。

        Returns:
            FermiNet: 自身引用，支持链式调用。
        """
        self.use_compile = True
        self.compile_mode = mode
        self.compile_backend = backend
        self._compiled_forward = None
        self._compiled_forward_log_psi = None
        self._compiled_device = None
        return self

    def disable_compile(self) -> "FermiNet":
        """
        关闭 JIT 编译优化，恢复为原生 Eager 解释执行模式 (用于代码调试或动态环境)。

        Returns:
            FermiNet: 自身引用，支持链式调用。
        """
        self.use_compile = False
        return self

    def compile_model(
        self,
        fullgraph: bool = False,
        dynamic: Optional[bool] = None,
        backend: str = "inductor",
        mode: str = "default",
        options: Optional[dict] = None,
    ) -> nn.Module:
        """
        利用 PyTorch 2.0+ 的 torch.compile 对 FermiNet 模型进行 JIT 图编译与算子融合优化。
        默认模式为 'default'，按需显式启用以防范 CUDAGraphs 静态显存覆写。
        显式禁用内部懒加载编译，彻底消除多重嵌套编译隐患。

        Args:
            fullgraph (bool): 是否强制要求全图编译无 Python 解释器 fallback。
            dynamic (Optional[bool]): 是否启用动态批次形状 (Dynamic shapes)。
            backend (str): 编译后端 (默认为 'inductor')。
            mode (str): 编译优化模式 (默认为 'default'；如需 CUDA Graphs 极致调度可显式传入 'reduce-overhead')。
            options (Optional[dict]): 传给编译后端的具体配置参数。

        Returns:
            nn.Module: 经 JIT 编译优化的高性能波函数计算模型。
        """
        self.disable_compile()  # 确保内部处于 Eager 模式，消除多重嵌套编译
        return torch.compile(
            self,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a checkpoint and migrate pre-R4 envelope group buffers.

        Earlier checkpoints did not serialize ``nucleus_group_index``.  Rebuild
        it from the checkpoint nuclear charges, not the receiving model, so a
        load cannot silently retain a stale nuclear ordering.
        """
        upgraded = state_dict.copy()
        checkpoint_charges = upgraded.get("nuclear_charges")
        for name in ("envelope_up", "envelope_down"):
            envelope = getattr(self, name, None)
            key = f"{name}.nucleus_group_index"
            if envelope is None:
                continue
            if checkpoint_charges is None:
                if key not in upgraded:
                    raise RuntimeError(
                        f"cannot migrate legacy checkpoint without nuclear_charges for {key}"
                    )
                continue
            charges = checkpoint_charges.detach().reshape(-1)
            if charges.numel() != envelope.num_nuclei:
                raise RuntimeError(
                    f"checkpoint has {charges.numel()} nuclear charges but {name} expects "
                    f"{envelope.num_nuclei}; construct a matching model before loading"
                )
            if self.tie_nuclear_envelope:
                _, expected_index = torch.unique(charges, sorted=True, return_inverse=True)
            else:
                expected_index = torch.arange(envelope.num_nuclei, device=charges.device)
            expected_index = expected_index.to(dtype=torch.long)
            if key not in upgraded:
                upgraded[key] = expected_index
            elif not torch.equal(
                upgraded[key].detach().reshape(-1).to(dtype=torch.long, device="cpu"),
                expected_index.detach().reshape(-1).to(dtype=torch.long, device="cpu"),
            ):
                raise RuntimeError(
                    f"checkpoint {key} conflicts with its nuclear_charges; refusing to load "
                    "a physically inconsistent envelope grouping"
                )
        return super().load_state_dict(upgraded, strict=strict, assign=assign)

    def check_system_consistency(self, system: "MolecularSystem") -> None:
        """
        校验模型几何缓冲与给定 MolecularSystem 是否一致。

        `FermiNet` 在构造时把核坐标与核电荷**拷贝**为自己的 buffer，而
        `MolecularSystem.update_geometry()` 只更新体系对象自身。因此若在构造模型
        之后更新了体系几何，波函数会使用旧几何、哈密顿量使用新几何，且**不会报错**
        (审查报告 D10)。本方法把这一隐患变成显式检查。

        Args:
            system (MolecularSystem): 待比对的分子体系。

        Raises:
            ValueError: 核数目、核坐标或核电荷不一致。
        """
        to_np64 = lambda t: t.detach().to(device="cpu", dtype=torch.float64)  # noqa: E731
        model_coords, system_coords = to_np64(self.nuclear_coords), to_np64(system.nuclear_coords)
        if model_coords.shape != system_coords.shape:
            raise ValueError(
                f"模型核坐标形状 {tuple(model_coords.shape)} 与体系 "
                f"{tuple(system_coords.shape)} 不一致"
            )
        if not torch.allclose(model_coords, system_coords, atol=1e-8, rtol=0.0):
            raise ValueError(
                "模型几何缓冲与 MolecularSystem 不一致：波函数会使用旧几何而哈密顿量"
                "使用新几何。更新体系几何后请重建模型。"
            )
        model_z, system_z = to_np64(self.nuclear_charges), to_np64(system.nuclear_charges)
        if model_z.shape != system_z.shape or not torch.allclose(
            model_z, system_z, atol=1e-12, rtol=0.0
        ):
            raise ValueError("模型核电荷缓冲与 MolecularSystem 不一致。")
        if self.n_up != system.n_up or self.n_down != system.n_down:
            raise ValueError(
                "模型自旋分区与 MolecularSystem 不一致："
                f"model=({self.n_up}, {self.n_down}), system=({system.n_up}, {system.n_down})"
            )

