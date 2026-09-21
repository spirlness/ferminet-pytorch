"""
变分量子蒙特卡洛 (VMC) 协方差梯度估计与局域能量截断
======================================================
实现满足变分原理、带有限批次协方差修正的参数梯度代理损失，
并集成基于中位数绝对偏差 (MAD) 的局部能量平滑截断 (Energy Clipping)。

理论背景:
1. 变分能量期望值:
       <E> = <Psi|H|Psi> / <Psi|Psi> = E_{r ~ |Psi|^2} [E_L(r)]
2. 变分无偏参数梯度定理:
       nabla_theta <E> = 2 * E_{r ~ |Psi|^2} [ (E_L(r) - <E_L>) * nabla_theta ln|Psi_theta(r)| ]
   该公式严格废除对局域能量 E_L 直接求三阶导数的错误逻辑，大幅降低梯度方差。
3. 局域能量截断 (Umrigar Tanh / Hard Clipping):
   在波函数节点或近核奇异区，E_L 可能出现离群奇异值。基于 MAD 进行对称平滑截断，
   有效抑制极端离群点引发的梯度发散，确保训练稳定性。
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from .hamiltonian import MolecularSystem, compute_local_energy
except (ImportError, ValueError):
    from hamiltonian import MolecularSystem, compute_local_energy


def clip_local_energy(
    e_local: torch.Tensor,
    clip_factor: float = 5.0,
    method: str = "tanh",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对局域能量序列 E_L(r) 进行基于中位数绝对偏差 (MAD) 的鲁棒截断。

    公式说明:
        E_median = median(E_L)
        MAD = median(|E_L - E_median|)
        Delta = max(clip_factor * MAD, eps)
        - Tanh 模式 (Umrigar):
          E_clipped = E_median + Delta * tanh((E_L - E_median) / Delta)
        - Hard 模式:
          E_clipped = clamp(E_L, E_median - Delta, E_median + Delta)

    Args:
        e_local (torch.Tensor): 各采样 Walker 处的原始局域能量，Shape: `(B,)`。
        clip_factor (float): 截断阈值因子 c (默认 5.0 倍 MAD)。
        method (str): 截断方式，可选 'tanh' (平滑连续) 或 'hard' (两端硬截断) 或 'none'。
        eps (float): 防止全样本能量相等时 MAD=0 的数值微小量。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - e_clipped (torch.Tensor): 截断后的局域能量，Shape: `(B,)`。
            - e_median (torch.Tensor): 标量，能量中位数。
            - mad (torch.Tensor): 标量，中位数绝对偏差。
    """
    finite_mask = torch.isfinite(e_local)
    if not finite_mask.any():
        # 极端退化：全部样本均为 NaN 或 Inf
        default_val = torch.tensor(0.0, device=e_local.device, dtype=e_local.dtype)
        return torch.zeros_like(e_local), default_val, default_val

    finite_energies = e_local[finite_mask]
    e_median = torch.median(finite_energies)
    mad = torch.median(torch.abs(finite_energies - e_median))

    if method == "none" or clip_factor <= 0:
        # 若非有限值存在，仍进行平滑填补以防梯度图崩溃
        e_safe = torch.nan_to_num(
            e_local,
            nan=e_median.item(),
            posinf=(e_median + 1e4).item(),
            neginf=(e_median - 1e4).item(),
        )
        return e_safe, e_median, mad

    # 截断半宽 Delta
    delta = torch.clamp(clip_factor * mad, min=eps)

    # 预清洗：NaN 映射为 e_median (使其偏离度为0，梯度贡献为0，实现稳健变分优化)；Inf 映射为远端饱和值
    clean_e = torch.nan_to_num(
        e_local,
        nan=e_median.item(),
        posinf=(e_median + 10.0 * delta).item(),
        neginf=(e_median - 10.0 * delta).item(),
    )

    if method == "tanh":
        # Umrigar 双曲正切平滑渐近截断
        diff = clean_e - e_median
        e_clipped = e_median + delta * torch.tanh(diff / delta)
    elif method == "hard":
        e_clipped = torch.clamp(clean_e, min=e_median - delta, max=e_median + delta)
    else:
        raise ValueError(f"不支持的截断方法: {method}，可选 'tanh', 'hard', 'none'")

    return e_clipped, e_median, mad


def vmc_surrogate_loss(
    log_abs_psi: torch.Tensor,
    e_local: torch.Tensor,
    clip_factor: float = 5.0,
    clip_method: str = "none",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算变分原理代理损失及能量统计指标。

    代理损失函数定义:
        L_VMC(theta) = 2/(B-1) * sum_i stop_gradient(E_i - mean(E)) * ln|Psi_theta(r_i)|
    在 ``clip_method='none'``、平衡分布和 IID 并行 walker 条件下，这是有限批次
    无偏协方差估计。显式启用 ``tanh`` 或 ``hard`` 时，优化目标是有偏但稳健的代理。

    Args:
        log_abs_psi (torch.Tensor): 具有参数梯度的对数波函数值 ln|Psi_theta(r)|，Shape: `(B,)`。
        e_local (torch.Tensor): 各 walker 处的局域能量数值 (若含梯度将自动 detach)，Shape: `(B,)`。
        clip_factor (float): 截断阈值倍数。
        clip_method (str): 截断方法 ('tanh', 'hard', 'none')。

    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]:
            - loss (torch.Tensor): 标量代理损失张量 (带模型参数梯度)。
            - stats (Dict[str, Any]): 包含物理能量统计量和估计器语义的字典。
    """
    batch_size = e_local.shape[0]
    assert log_abs_psi.shape[0] == batch_size, "log_abs_psi 与 e_local 的 Batch 大小必须一致"
    if batch_size < 2:
        raise ValueError(f"VMC covariance estimator requires B >= 2, got B={batch_size}")
    if clip_method not in ("none", "tanh", "hard"):
        raise ValueError(f"不支持的截断方法: {clip_method}，可选 'tanh', 'hard', 'none'")

    # 1. 局域能量截断与分离梯度
    e_local_detached = e_local.detach()
    finite_mask = torch.isfinite(e_local_detached)
    num_nonfinite = int((~finite_mask).sum().item())
    if clip_method == "none" and num_nonfinite:
        raise FloatingPointError(
            f"strict VMC gradient received {num_nonfinite} non-finite local energies "
            f"in batch B={batch_size}"
        )
    e_clipped, e_median, mad = clip_local_energy(
        e_local_detached, clip_factor=clip_factor, method=clip_method
    )

    # 2. B/(B-1) 等价于 leave-one-out baseline 的有限批次协方差修正。
    mean_clipped_energy = torch.mean(e_clipped)
    correction = batch_size / (batch_size - 1.0)
    delta_e = (correction * (e_clipped - mean_clipped_energy)).detach()

    # 3. 构造变分代理损失: L = 2.0 * mean( Delta E * ln|Psi| )
    loss = 2.0 * torch.mean(delta_e * log_abs_psi)

    # 4. 统计物理指标 (基于有限值安全统计)
    if finite_mask.any():
        finite_e = e_local_detached[finite_mask]
        raw_mean = torch.mean(finite_e).item()
        raw_var = torch.var(finite_e, unbiased=True).item() if finite_e.shape[0] > 1 else 0.0
    else:
        raw_mean = 0.0
        raw_var = 0.0
    raw_std = raw_var ** 0.5
    iid_sem_proxy = raw_std / (max(int(finite_mask.sum()), 1) ** 0.5)

    stats = {
        "energy_mean": raw_mean,
        "energy_std": raw_std,
        "energy_var": raw_var,
        "energy_iid_sem_proxy": iid_sem_proxy,
        "energy_clipped_mean": mean_clipped_energy.item(),
        "energy_median": e_median.item(),
        "energy_mad": mad.item(),
        "loss": loss.item(),
        "num_nonfinite": num_nonfinite,
        "gradient_estimator": "iid_covariance_corrected" if clip_method == "none" else "robust_clipped",
        "gradient_is_unbiased": clip_method == "none",
    }

    return loss, stats


class VMCLoss(nn.Module):
    """
    变分量子蒙特卡洛 (VMC) 损失模块封装。

    集成局域能量自动微分计算、可选稳健截断及协方差梯度代理损失反向传播。
    支持配置高精度 exact 或极速 hutchinson 动能求解器。
    """

    def __init__(
        self,
        clip_factor: float = 5.0,
        clip_method: str = "none",
        laplacian_method: str = "exact",
        num_hutchinson_samples: int = 1,
        hutchinson_distribution: str = "rademacher",
    ) -> None:
        """
        初始化 VMC 损失计算器。

        Args:
            clip_factor (float): 截断阈值倍数 (MAD 倍数，默认 5.0)。
            clip_method (str): 截断方案 ('tanh', 'hard', 'none')。
            laplacian_method (str): 局域能量拉普拉斯动能求解策略 ('exact', 'hutchinson', 'exact_jvp', 'jvp')。
            num_hutchinson_samples (int): Hutchinson 随机迹估计采样次数 (默认为 1)。
            hutchinson_distribution (str): Hutchinson 随机探测向量分布 ('rademacher' 或 'gaussian'，默认 'rademacher')。
        """
        super().__init__()
        self.clip_factor = clip_factor
        self.clip_method = clip_method
        self.laplacian_method = laplacian_method
        self.num_hutchinson_samples = num_hutchinson_samples
        self.hutchinson_distribution = hutchinson_distribution

    def forward(
        self,
        log_abs_psi: torch.Tensor,
        e_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        基于给定的对数波函数与局域能量计算协方差代理损失。

        Args:
            log_abs_psi (torch.Tensor): Shape: `(B,)`，要求保持模型参数的计算图。
            e_local (torch.Tensor): Shape: `(B,)`。

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: (loss, stats)
        """
        return vmc_surrogate_loss(
            log_abs_psi=log_abs_psi,
            e_local=e_local,
            clip_factor=self.clip_factor,
            clip_method=self.clip_method,
        )

    def compute_step_loss(
        self,
        model: nn.Module,
        system: MolecularSystem,
        walkers: torch.Tensor,
        laplacian_method: Optional[str] = None,
        num_hutchinson_samples: Optional[int] = None,
        hutchinson_distribution: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
        """
        一站式端到端计算给定 Walkers 下的局部能量与协方差代理损失。

        执行流程:
        1. 分离 walkers 坐标并开启坐标微分，调用 compute_local_energy 求解 E_L(r)。
        2. 剥离局部能量计算图 (仅保留物理数值)，消除三阶导冗余及内存泄漏。
        3. 对 walkers 前向模型获取具有参数梯度的 log|Psi_theta(r)|。
        4. 计算可选截断与有限批次协方差代理损失。

        Args:
            model (nn.Module): FermiNet 模型。
            system (MolecularSystem): 分子体系。
            walkers (torch.Tensor): 当前批次电子构型，Shape: `(B, N, 3)`。
            laplacian_method (Optional[str]): 若指定则覆盖默认拉普拉斯求解策略。
            num_hutchinson_samples (Optional[int]): 若指定则覆盖默认 Hutchinson 采样次数。
            hutchinson_distribution (Optional[str]): 若指定则覆盖默认 Hutchinson 随机分布。

        Returns:
            Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
                - loss (torch.Tensor): 可反向传播的协方差代理损失。
                - stats (Dict[str, Any]): 能量、方差与估计器语义。
                - e_local (torch.Tensor): 未截断的原始局域能量序列，Shape: `(B,)`。
        """
        lap_method = laplacian_method if laplacian_method is not None else self.laplacian_method
        hutch_samples = num_hutchinson_samples if num_hutchinson_samples is not None else self.num_hutchinson_samples
        hutch_dist = hutchinson_distribution if hutchinson_distribution is not None else self.hutchinson_distribution

        # 1. 计算局部能量 (使用独立坐标计算图)
        r_for_energy = walkers.clone().detach().requires_grad_(True)
        e_local, _, _ = compute_local_energy(
            model=model,
            system=system,
            r=r_for_energy,
            create_graph=False,
            laplacian_method=lap_method,
            num_hutchinson_samples=hutch_samples,
            hutchinson_distribution=hutch_dist,
        )
        e_local = e_local.detach()  # 严格阻断与能量高阶导图的关联

        # 2. 干净前向获取带有模型参数梯度的 log|Psi(r)|
        log_abs_psi, _ = model.forward_log_psi(walkers.detach())

        # 3. 计算代理损失
        loss, stats = self.forward(log_abs_psi, e_local)

        return loss, stats, e_local
