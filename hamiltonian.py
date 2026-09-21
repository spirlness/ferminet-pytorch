from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings
import torch
import torch.nn as nn


@dataclass
class MolecularSystem:
    """
    量子化学分子体系定义。

    Attributes:
        nuclear_coords (torch.Tensor): 原子核空间三维笛卡尔坐标，Shape: `(M, 3)`。
        nuclear_charges (torch.Tensor): 原子核电荷量 (原子序数 Z)，Shape: `(M,)`。
        n_up (int): 自旋向上电子数 (N_up)。
        n_down (int): 自旋向下电子数 (N_down)。
    """
    nuclear_coords: torch.Tensor
    nuclear_charges: torch.Tensor
    n_up: int
    n_down: int
    _v_nn_const: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化后进行输入有效性校验并自动预计算核-核库仑排斥常数 V_nn."""
        assert self.nuclear_coords.ndim == 2 and self.nuclear_coords.shape[1] == 3, (
            f"原子核坐标必须是 2 维张量 (M, 3)，当前为 {self.nuclear_coords.shape}"
        )
        assert self.nuclear_charges.ndim == 1 and self.nuclear_charges.shape[0] == self.nuclear_coords.shape[0], (
            f"原子核电荷数 ({self.nuclear_charges.shape}) 必须与坐标原子数 ({self.nuclear_coords.shape[0]}) 匹配"
        )
        assert self.n_up >= 0 and self.n_down >= 0, "自旋电子数必须非负"
        assert self.n_up + self.n_down > 0, "体系必须包含至少一个电子"
        self._compute_v_nn_const()

    def _compute_v_nn_const(self) -> torch.Tensor:
        """
        预计算分子体系的核-核库仑排斥势能常数 V_nn。
        V_nn = sum_{1 <= I < J <= M} Z_I * Z_J / |R_I - R_J|
        采用严格上三角三维欧几里得距离索引，杜绝人造 eps 误差并节省显存。
        """
        num_nuclei = self.nuclear_coords.shape[0]
        if num_nuclei > 1:
            row_idx, col_idx = torch.triu_indices(
                num_nuclei, num_nuclei, offset=1, device=self.nuclear_coords.device
            )
            diff = self.nuclear_coords[row_idx] - self.nuclear_coords[col_idx]
            dist = torch.linalg.vector_norm(diff, dim=-1)
            charge_prod = self.nuclear_charges[row_idx] * self.nuclear_charges[col_idx]
            v_nn_const = torch.sum(charge_prod / dist)
        else:
            v_nn_const = torch.tensor(
                0.0, device=self.nuclear_coords.device, dtype=self.nuclear_coords.dtype
            )
        self._v_nn_const = v_nn_const
        return v_nn_const

    @property
    def v_nn(self) -> torch.Tensor:
        """获取预计算的核-核库仑排斥常数 V_nn (标量张量)."""
        if self._v_nn_const is None:
            return self._compute_v_nn_const()
        return self._v_nn_const

    def update_geometry(
        self,
        nuclear_coords: torch.Tensor,
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> "MolecularSystem":
        """
        更新分子几何构型并自动刷新预计算的核-核排斥常数 V_nn 缓存。

        Args:
            nuclear_coords (torch.Tensor): 新的原子核坐标，Shape: (M, 3)。
            nuclear_charges (Optional[torch.Tensor]): 可选的新原子核电荷，Shape: (M,)。

        Returns:
            MolecularSystem: 自身引用，支持链式调用。
        """
        self.nuclear_coords = nuclear_coords
        if nuclear_charges is not None:
            self.nuclear_charges = nuclear_charges
        self._compute_v_nn_const()
        return self

    def to(self, *args, **kwargs) -> "MolecularSystem":
        """
        将分子体系的核坐标与核电荷张量移动到指定设备或数据类型。
        完全兼容 PyTorch 原生 to(*args, **kwargs) 语法 (如 .to(torch.float64), .to('cuda'))，
        并同步迁移预计算并缓存的核-核排斥常数 V_nn。

        Returns:
            MolecularSystem: 自身引用，支持链式调用。
        """
        self.nuclear_coords = self.nuclear_coords.to(*args, **kwargs)
        self.nuclear_charges = self.nuclear_charges.to(*args, **kwargs)
        if self._v_nn_const is not None:
            self._v_nn_const = self._v_nn_const.to(*args, **kwargs)
        return self

    @property
    def total_electrons(self) -> int:
        return self.n_up + self.n_down

    @property
    def num_nuclei(self) -> int:
        return self.nuclear_coords.shape[0]

    def compute_potential_energy(self, r: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        全矢量化计算分子的总库仑势能: V(r) = V_en(r) + V_ee(r) + V_nn。
        严格禁止 Python 循环遍历电子坐标。
        自动确保核参数与输入电子坐标 r 处于相同设备 (Device) 与数据类型 (Dtype)。

        Args:
            r (torch.Tensor): 电子三维空间坐标。
                - Shape: `(B, N, 3)`。
                - Dtype: `torch.float32` 或 `torch.float64`。
            eps (float): 防止电子重合或电子撞核时的除零保护常数。

        Returns:
            torch.Tensor: 各采样构型的总势能 V(r) (单位: Hartree)。
                - Shape: `(B,)`。
        """
        batch_size, num_electrons, _ = r.shape
        device = r.device
        dtype = r.dtype

        # 动态对齐核张量到输入 r 的设备与数据类型
        nuclear_coords = self.nuclear_coords.to(device=device, dtype=dtype)
        nuclear_charges = self.nuclear_charges.to(device=device, dtype=dtype)

        # ---------------------------------------------------------------------
        # 1. 电子-原子核吸引势能: V_en = - sum_{i=1}^N sum_{I=1}^M Z_I / |r_i - R_I|
        # ---------------------------------------------------------------------
        # r: (B, N, 1, 3), R: (1, 1, M, 3)
        diff_en = r.unsqueeze(2) - nuclear_coords.unsqueeze(0).unsqueeze(0)
        dist_en = torch.sqrt(torch.sum(diff_en ** 2, dim=-1) + eps)  # (B, N, M)
        charges = nuclear_charges.unsqueeze(0).unsqueeze(0)          # (1, 1, M)

        v_en = -torch.sum(charges / dist_en, dim=(-1, -2))  # (B,)

        # ---------------------------------------------------------------------
        # 2. 电子-电子排斥势能: V_ee = sum_{1 <= i < j <= N} 1 / |r_i - r_j|
        # ---------------------------------------------------------------------
        if num_electrons > 1:
            diff_ee = r.unsqueeze(2) - r.unsqueeze(1)  # (B, N, N, 3)
            dist_ee = torch.sqrt(torch.sum(diff_ee ** 2, dim=-1) + eps)  # (B, N, N)

            # 构建严格上三角掩码矩阵 (i < j)
            mask_ee = torch.triu(
                torch.ones(num_electrons, num_electrons, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0)  # (1, N, N)

            v_ee = torch.sum(
                torch.where(mask_ee, 1.0 / dist_ee, torch.zeros_like(dist_ee)),
                dim=(-1, -2),
            )  # (B,)
        else:
            v_ee = torch.zeros(batch_size, device=device, dtype=dtype)

        # ---------------------------------------------------------------------
        # 3. 原子核-原子核排斥势能: 复用预计算并缓存的核-核排斥常数 V_nn，彻底消除重复计算
        # ---------------------------------------------------------------------
        v_nn_const = self.v_nn
        if v_nn_const.device != device or v_nn_const.dtype != dtype:
            v_nn_const = v_nn_const.to(device=device, dtype=dtype)
            self._v_nn_const = v_nn_const
        v_nn = v_nn_const.expand(batch_size)

        return v_en + v_ee + v_nn


def _compute_laplacian_exact(
    grad_flat: torch.Tensor,
    r: torch.Tensor,
    ones: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """
    经典反向模式逐维精确微分求解拉普拉斯算子:
        nabla^2 ln|Psi| = sum_{d=1}^{3N} d^2/dr_d^2 ln|Psi|
    """
    batch_size = r.shape[0]
    dim_total = grad_flat.shape[1]
    laplacian = torch.zeros(batch_size, device=r.device, dtype=r.dtype)
    for d in range(dim_total):
        g_d = grad_flat[:, d]
        sec_retain = True if create_graph else (d < dim_total - 1)
        sec_grad = torch.autograd.grad(
            outputs=g_d,
            inputs=r,
            grad_outputs=ones,
            create_graph=create_graph,
            retain_graph=sec_retain,
        )[0]
        sec_grad_flat = sec_grad.reshape(batch_size, -1)
        laplacian = laplacian + sec_grad_flat[:, d]
    return laplacian


def _compute_laplacian_hutchinson(
    grad_log_psi: torch.Tensor,
    r: torch.Tensor,
    ones: torch.Tensor,
    create_graph: bool,
    num_samples: int = 1,
    distribution: str = "rademacher",
) -> torch.Tensor:
    """
    基于 Rademacher 或 Gaussian 随机向量的 Hutchinson 随机迹估计器:
        Tr(nabla^2 ln|Psi|) = E_v [ v^T (nabla^2 ln|Psi|) v ]
    单次采样仅需 1 次方向导数反向传播，将大体系二次导耗时从 O(3N) 压缩至 O(1)。
    """
    batch_size = r.shape[0]
    laplacian_sum = torch.zeros(batch_size, device=r.device, dtype=r.dtype)

    for k in range(num_samples):
        # 1. 产生随机探测向量 v
        if distribution == "rademacher":
            # Rademacher 分布: 各分量独立以 0.5 概率取 +1 或 -1 (无偏且方差最小)
            v = torch.randint(0, 2, size=r.shape, device=r.device, dtype=r.dtype) * 2.0 - 1.0
        elif distribution == "gaussian":
            # 标准多元高斯分布: E[v v^T] = I
            v = torch.randn_like(r)
        else:
            raise ValueError(f"不支持的 Hutchinson 分布: {distribution}，可选 'rademacher', 'gaussian'")

        # 2. 计算方向内积标量: S = sum_{i, d} v_{i, d} * (nabla ln|Psi|)_{i, d}
        s = torch.sum(v * grad_log_psi, dim=(-1, -2))  # (B,)

        # 3. 反向传播计算 Hessian-Vector Product: Hv = nabla_r (v^T nabla_r ln|Psi|)
        sec_retain = True if create_graph else (k < num_samples - 1)
        hv = torch.autograd.grad(
            outputs=s,
            inputs=r,
            grad_outputs=ones,
            create_graph=create_graph,
            retain_graph=sec_retain,
        )[0]  # (B, N, 3)

        # 4. 累加单次样本随机迹估计: v^T (H v) = sum_{i, d} v_{i, d} * (Hv)_{i, d}
        tr_k = torch.sum(v * hv, dim=(-1, -2))  # (B,)
        laplacian_sum = laplacian_sum + tr_k

    return laplacian_sum / float(num_samples)


def _compute_laplacian_jvp(
    model: nn.Module,
    r: torch.Tensor,
    create_graph: bool = False,
    grad_log_psi: Optional[torch.Tensor] = None,
    ones: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    基于 torch.func.jvp 与 torch.func.vmap 前向模式微分的全向量化精确求解。
    批次维度与坐标维度均采用 vmap 向量化映射，严格无显式 Python 循环。
    若当前算子或环境限制无法支持，平滑回退至经典 exact 逆向微分模式。
    """
    batch_size, num_electrons, _ = r.shape
    dim_total = num_electrons * 3

    try:
        import torch.func as func

        eye = torch.eye(dim_total, device=r.device, dtype=r.dtype)

        def _compute_single_walker(x_single: torch.Tensor) -> torch.Tensor:
            def _single_log_psi(xf: torch.Tensor) -> torch.Tensor:
                x_3d = xf.view(1, num_electrons, 3)
                lp, _ = model.forward_log_psi(x_3d)
                return lp.squeeze(0)

            _grad_fn = func.grad(_single_log_psi)
            x_flat = x_single.reshape(-1)

            def _diag_elem(v: torch.Tensor) -> torch.Tensor:
                _, jvp_res = func.jvp(_grad_fn, (x_flat,), (v,))
                return torch.dot(v, jvp_res)

            return func.vmap(_diag_elem)(eye).sum()

        return func.vmap(_compute_single_walker)(r)
    except Exception as e:
        warnings.warn(
            f"torch.func.jvp 向量化求解遇到异常 ({e})，平滑回退至经典 exact 逆向模式。",
            UserWarning,
            stacklevel=2,
        )
        if ones is None:
            ones = torch.ones(batch_size, device=r.device, dtype=r.dtype)
        if grad_log_psi is None:
            log_abs_psi, _ = model.forward_log_psi(r)
            grad_log_psi = torch.autograd.grad(
                outputs=log_abs_psi,
                inputs=r,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
            )[0]
        return _compute_laplacian_exact(grad_log_psi.reshape(batch_size, -1), r, ones, create_graph)


def compute_local_energy(
    model: nn.Module,
    system: MolecularSystem,
    r: torch.Tensor,
    create_graph: bool = False,
    laplacian_method: str = "exact",
    num_hutchinson_samples: int = 1,
    hutchinson_distribution: str = "rademacher",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算给定电子构型 r 处的局部能量 (Local Energy):
        E_L(r) = T_L(r) + V(r)
    其中局部动能项利用对数波函数的一阶与二阶自动微分求取:
        T_L(r) = -0.5 * (nabla^2 log|Psi(r)| + |nabla log|Psi(r)|^2)

    Args:
        model (nn.Module): 实例化的 FermiNet 模型，具备 forward_log_psi 接口。
        system (MolecularSystem): 分子体系几何与物理定义。
        r (torch.Tensor): 电子三维坐标。
            - Shape: `(B, N, 3)`。
            - Dtype: `torch.float32` 或 `torch.float64`。必须设置 requires_grad=True。
        create_graph (bool): 是否在动能与势能计算中保留高阶计算图 (默认为 False)。
            - 若为 True，计算得到的 E_L 可直接对模型参数求导；
            - 若为 False，在完成二阶拉普拉斯后释放内部图，节省显存与算力。
        laplacian_method (str): 动能拉普拉斯算子求解策略:
            - 'exact' (默认): 经典反向模式逐维精确求导，结果具机器精度；
            - 'hutchinson': 基于 Rademacher/Gaussian 随机向量的高性能随机迹估计器，单步仅需 1 次反向传播；
            - 'exact_jvp': 基于 torch.func.jvp 前向模式微分的向量化精确求解。
        num_hutchinson_samples (int): Hutchinson 随机迹估计采样次数 (默认为 1)。
        hutchinson_distribution (str): Hutchinson 随机向量分布 ('rademacher' 或 'gaussian'，默认 'rademacher')。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - e_local (torch.Tensor): 局部总能量 E_L(r) = T_L(r) + V(r)，Shape: `(B,)`。
            - t_local (torch.Tensor): 局部动能 T_L(r)，Shape: `(B,)`。
            - v_local (torch.Tensor): 局部势能 V(r)，Shape: `(B,)`。
    """
    batch_size, num_electrons, _ = r.shape
    assert r.requires_grad, "输入坐标 r 必须开启梯度追踪 (requires_grad=True) 以便计算动能拉普拉斯项"

    valid_methods = ["exact", "exact_jvp", "jvp", "hutchinson"]
    if laplacian_method not in valid_methods:
        raise ValueError(f"不支持的拉普拉斯求解方法: {laplacian_method}，可选: {valid_methods}")
    if num_hutchinson_samples < 1:
        raise ValueError(f"Hutchinson 采样次数必须 >= 1，当前为: {num_hutchinson_samples}")

    # 当无需为模型参数保留高阶计算图 (create_graph=False) 时，临时禁用参数梯度追踪
    prev_param_states = []
    if not create_graph:
        for p in model.parameters():
            prev_param_states.append(p.requires_grad)
            p.requires_grad = False

    try:
        # 1. 计算对数波函数幅值 log|Psi(r)|
        log_abs_psi, _ = model.forward_log_psi(r)  # (B,)

        # 2. 计算一阶梯度: g = nabla_{r} log|Psi(r)|, shape (B, N, 3)
        ones = torch.ones_like(log_abs_psi)
        grad_log_psi = torch.autograd.grad(
            outputs=log_abs_psi,
            inputs=r,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N, 3)

        # 梯度模平方项: |nabla log|Psi||^2, shape (B,)
        grad_squared = torch.sum(grad_log_psi ** 2, dim=(-1, -2))

        # 3. 计算对数波函数的拉普拉斯项: nabla^2 log|Psi|
        if laplacian_method == "hutchinson":
            laplacian = _compute_laplacian_hutchinson(
                grad_log_psi=grad_log_psi,
                r=r,
                ones=ones,
                create_graph=create_graph,
                num_samples=num_hutchinson_samples,
                distribution=hutchinson_distribution,
            )
        elif laplacian_method in ["exact_jvp", "jvp"]:
            laplacian = _compute_laplacian_jvp(
                model=model,
                r=r,
                create_graph=create_graph,
                grad_log_psi=grad_log_psi,
                ones=ones,
            )
        else:
            # 默认 'exact'
            grad_flat = grad_log_psi.reshape(batch_size, -1)
            laplacian = _compute_laplacian_exact(
                grad_flat=grad_flat,
                r=r,
                ones=ones,
                create_graph=create_graph,
            )

        # 4. 局部动能计算: T_L = -0.5 * (nabla^2 log|Psi| + |nabla log|Psi||^2)
        t_local = -0.5 * (laplacian + grad_squared)

        # 5. 局部势能计算
        if create_graph:
            v_local = system.compute_potential_energy(r)
        else:
            with torch.no_grad():
                v_local = system.compute_potential_energy(r)

        e_local = t_local + v_local
    finally:
        if not create_graph:
            for p, state in zip(model.parameters(), prev_param_states):
                p.requires_grad = state

    return e_local, t_local, v_local
