"""
变分量子蒙特卡洛 (VMC) 协方差梯度估计与局域能量截断
======================================================
实现基于变分协方差公式的参数梯度代理损失；能量截断会引入偏差，
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

from typing import Any, Dict, Optional, Tuple
import warnings
import torch
import torch.nn as nn

from .hamiltonian import MolecularSystem, compute_local_energy, _FreezeModelParameters


def clip_local_energy(
    e_local: torch.Tensor,
    clip_factor: float = 5.0,
    method: str = "tanh",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于中位数绝对偏差 (MAD) 的局域能量截断 (默认 Umrigar Tanh 模式)。
    mean_hard 使用标准中位数、平均绝对偏差和硬裁剪，无能量零点相关下限；
    该模式第三个返回值为平均绝对偏差，其余模式仍为 median MAD。

    公式:
        E_median = median(E_L), MAD = median(|E_L - E_median|), Delta = max(clip_factor * MAD, eps)
        E_clipped = E_median + Delta * tanh((E_L - E_median) / Delta)

    所有样本无效时抛出 FloatingPointError。部分无效输入保持返回形状，
    仅以有效样本估计中位数/MAD；VMC 损失会排除这些无效样本。
    """
    finite_mask = torch.isfinite(e_local)
    if not finite_mask.any():
        raise FloatingPointError(
            "No finite local energies; cannot compute energy statistics"
        )

    finite_e = e_local[finite_mask]
    if method == "mean_hard" and clip_factor > 0:
        # Standard median averages the two middle values for an even batch.
        center = torch.quantile(finite_e, 0.5)
        mean_deviation = (finite_e - center).abs().mean()
        width = clip_factor * mean_deviation
        clean = torch.where(finite_mask, e_local, center)
        return clean.clamp(center - width, center + width), center, mean_deviation
    e_median = torch.median(finite_e)
    mad = torch.median(torch.abs(finite_e - e_median))

    if method == "none" or clip_factor <= 0:
        return torch.where(finite_mask, e_local, e_median), e_median, mad

    min_physical_delta = max(float(eps), 1e-3, 0.01 * float(torch.abs(e_median).item()))
    delta = torch.clamp(clip_factor * mad, min=min_physical_delta)
    clean_e = torch.nan_to_num(
        e_local,
        nan=e_median.item(),
        posinf=(e_median + 10.0 * delta).item(),
        neginf=(e_median - 10.0 * delta).item(),
    )

    if method == "tanh":
        e_clipped = e_median + delta * torch.tanh((clean_e - e_median) / delta)
    elif method == "hard":
        e_clipped = torch.clamp(clean_e, min=e_median - delta, max=e_median + delta)
    else:
        raise ValueError(
            f"不支持的截断方法: {method}，可选 'tanh', 'hard', 'mean_hard', 'none'"
        )

    return e_clipped, e_median, mad


def vmc_surrogate_loss(
    log_abs_psi: torch.Tensor,
    e_local: torch.Tensor,
    clip_factor: float = 5.0,
    clip_method: str = "tanh",
    max_invalid_fraction: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算变分原理代理损失及能量统计指标。

    代理损失定义:
        L_VMC(theta) = 2.0 * mean( (E_clipped - mean(E_clipped)) * ln|Psi_theta| )
    未截断的总体协方差对应变分能量梯度；有限批次和截断估计不能宣称严格无偏:
        ∇_θ <E> = 2 E[ (E_L - <E>) ∇_θ ln|Psi| ]

    Args:
        log_abs_psi: 具有参数梯度的对数波函数值, Shape: (B,)
        e_local: 各 walker 处的局域能量数值, Shape: (B,)
        clip_factor: 截断阈值倍数 (MAD 倍数)
        clip_method: 截断方法 ('tanh', 'hard', 'mean_hard', 'none')
        max_invalid_fraction: 允许的最大非有限样本占比阈值 (默认 0.10, 即 10%)。若超标立即抛出 FloatingPointError。

    Returns:
        loss: 可反向传播的标量代理损失
        stats: 统计指标字典 (含 energy_mean, energy_std, loss, num_valid, num_invalid)

    非有限能量不参与损失或统计，并发出 RuntimeWarning。若非有限样本占比超过 max_invalid_fraction，
    或全部能量无效，或任意 log|Psi| 非有限时，抛出 FloatingPointError，禁止参数更新。
    """
    batch_size = e_local.shape[0]
    assert log_abs_psi.shape[0] == batch_size, (
        "log_abs_psi 与 e_local 的 Batch 大小必须一致"
    )

    # 1. 局域能量截断与分离梯度 (阻止对 E_L 进行高阶求导)
    e_local_detached = e_local.detach()
    valid_mask = torch.isfinite(e_local_detached) & torch.isfinite(log_abs_psi)
    if not valid_mask.any():
        raise FloatingPointError("No finite local energies; VMC update aborted")
    if not torch.isfinite(log_abs_psi).all():
        raise FloatingPointError("Nonfinite log|Psi|; VMC update aborted")

    num_valid = int(valid_mask.sum().item())
    num_invalid = batch_size - num_valid
    invalid_fraction = float(num_invalid) / float(batch_size)
    if invalid_fraction > max_invalid_fraction:
        raise FloatingPointError(
            f"Invalid local energy fraction {invalid_fraction:.2%} ({num_invalid}/{batch_size}) "
            f"exceeds maximum allowed {max_invalid_fraction:.2%}; VMC update aborted"
        )

    finite_e = e_local_detached[valid_mask]
    if num_invalid:
        warnings.warn(
            f"Excluded {num_invalid}/{batch_size} nonfinite local energies from the VMC estimate",
            RuntimeWarning,
            stacklevel=2,
        )

    e_clipped, e_median, mad = clip_local_energy(
        finite_e, clip_factor=clip_factor, method=clip_method
    )

    # 2. 计算中心化能量差
    mean_clipped_energy = torch.mean(e_clipped)
    delta_e = (e_clipped - mean_clipped_energy).detach()

    # 3. 构造变分代理损失: L = 2.0 * mean( Delta E * ln|Psi| )
    loss = 2.0 * torch.mean(delta_e * log_abs_psi[valid_mask])

    # 4. 统计物理指标
    raw_mean = torch.mean(finite_e).item()
    raw_var = torch.var(finite_e, unbiased=True).item() if num_valid > 1 else 0.0
    raw_std = raw_var**0.5

    stats = {
        "energy_mean": raw_mean,
        "energy_std": raw_std,
        "energy_var": raw_var,
        "energy_clipped_mean": mean_clipped_energy.item(),
        "energy_median": e_median.item(),
        "energy_mad": mad.item(),
        "energy_clip_scale_statistic": "mean_absolute_deviation"
        if clip_method == "mean_hard"
        else "median_absolute_deviation",
        "loss": loss.item(),
        "num_valid": num_valid,
        "num_invalid": num_invalid,
    }

    return loss, stats


class VMCLoss(nn.Module):
    """
    变分量子蒙特卡洛 (VMC) 损失模块封装。
    """

    def __init__(
        self,
        clip_factor: float = 5.0,
        clip_method: str = "tanh",
        max_invalid_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        self.clip_factor = clip_factor
        self.clip_method = clip_method
        self.max_invalid_fraction = float(max_invalid_fraction)

    def forward(
        self,
        log_abs_psi: torch.Tensor,
        e_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """计算代理损失。"""
        return vmc_surrogate_loss(
            log_abs_psi=log_abs_psi,
            e_local=e_local,
            clip_factor=self.clip_factor,
            clip_method=self.clip_method,
            max_invalid_fraction=self.max_invalid_fraction,
        )

    def compute_step_loss(
        self,
        model: nn.Module,
        system: MolecularSystem,
        walkers: torch.Tensor,
        geom_ctx: Optional[Any] = None,
        log_abs_psi: Optional[torch.Tensor] = None,
        kinetic_mode="reverse",
        num_probes: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
        """
        一站式计算局域能量与变分代理损失。

        Args:
            model: FermiNet 模型
            system: 分子体系
            walkers: 电子坐标构型, Shape: (B, N, 3)
            geom_ctx: 可选的 GeometricContext 共享几何上下文
            log_abs_psi: 可选的已计算带梯度对数波函数值 (若提供则复用参数梯度前向；坐标能量仍用独立前向)

        Returns:
            loss: 代理损失
            stats: 统计指标字典
            e_local: 局域能量, Shape: (B,)
        """
        # 1. 计算局部能量 (使用独立坐标计算图并隔离模型参数)
        r_for_energy = walkers.clone().detach().requires_grad_(True)
        with _FreezeModelParameters(model):
            e_local, _, _ = compute_local_energy(
                model=model,
                system=system,
                r=r_for_energy,
                geom_ctx=geom_ctx,
                kinetic_mode=kinetic_mode,
                num_probes=num_probes,
            )
        e_local = e_local.detach()

        # 2. 前向获取带模型参数梯度的 log|Psi(r)| (若传入则直接复用参数梯度图)
        if log_abs_psi is None:
            log_abs_psi, _ = model.forward_log_psi(walkers.detach())

        # 3. 计算代理损失
        loss, stats = self.forward(log_abs_psi, e_local)

        return loss, stats, e_local
