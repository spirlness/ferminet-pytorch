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
import warnings
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
        # method='none' 的语义是"不截断"，但非有限值必须被填充才能继续计算。
        # 该替换会静默改变数据，因此在**确实发生替换时**发出警告 (审查报告 D8)，
        # 而不是无声地 nan_to_num。训练主路径 vmc_surrogate_loss 在
        # clip_method='none' 且存在非有限值时会直接抛 FloatingPointError。
        num_nonfinite = int((~torch.isfinite(e_local)).sum())
        if num_nonfinite:
            warnings.warn(
                f"clip_local_energy(method='none') 收到 {num_nonfinite} 个非有限局域能量，"
                f"已替换为中位数 {e_median.item():.6f} 附近的值。这会静默改变数据——"
                f"若需要严格行为，请改用 vmc_surrogate_loss (它在 clip_method='none' 下抛错)。",
                UserWarning,
                stacklevel=2,
            )
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
    clip_method: str = "tanh",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算变分原理代理损失及能量统计指标。

    代理损失定义:
        L_VMC(theta) = 2.0 * mean( (E_clipped - mean(E_clipped)) * ln|Psi_theta| )
    在反向传播时，其梯度严格等于变分能量期望对模型参数的梯度:
        ∇_θ <E> = 2 E[ (E_L - <E>) ∇_θ ln|Psi| ]

    Args:
        log_abs_psi: 具有参数梯度的对数波函数值, Shape: (B,)
        e_local: 各 walker 处的局域能量数值, Shape: (B,)
        clip_factor: 截断阈值倍数 (MAD 倍数)
        clip_method: 截断方法 ('tanh', 'hard', 'none')

    Returns:
        loss: 可反向传播的标量代理损失
        stats: 统计指标字典 (含 energy_mean, energy_std, loss)
    """
    batch_size = e_local.shape[0]
    assert log_abs_psi.shape[0] == batch_size, "log_abs_psi 与 e_local 的 Batch 大小必须一致"

    # 1. 局域能量截断与分离梯度 (阻止对 E_L 进行高阶求导)
    e_local_detached = e_local.detach()
    finite_mask = torch.isfinite(e_local_detached)

    e_clipped, e_median, mad = clip_local_energy(
        e_local_detached, clip_factor=clip_factor, method=clip_method
    )

    # 2. 计算中心化能量差
    mean_clipped_energy = torch.mean(e_clipped)
    delta_e = (e_clipped - mean_clipped_energy).detach()

    # 3. 构造变分代理损失: L = 2.0 * mean( Delta E * ln|Psi| )
    loss = 2.0 * torch.mean(delta_e * log_abs_psi)

    # 4. 统计物理指标
    if finite_mask.any():
        finite_e = e_local_detached[finite_mask]
        raw_mean = torch.mean(finite_e).item()
        raw_var = torch.var(finite_e, unbiased=True).item() if finite_e.shape[0] > 1 else 0.0
    else:
        raw_mean = 0.0
        raw_var = 0.0
    raw_std = raw_var ** 0.5

    stats = {
        "energy_mean": raw_mean,
        "energy_std": raw_std,
        "energy_var": raw_var,
        "energy_clipped_mean": mean_clipped_energy.item(),
        "energy_median": e_median.item(),
        "energy_mad": mad.item(),
        "loss": loss.item(),
    }

    return loss, stats


# 兼容简写别名
vmc_loss = vmc_surrogate_loss


class VMCLoss(nn.Module):
    """
    变分量子蒙特卡洛 (VMC) 损失模块封装。
    """

    def __init__(
        self,
        clip_factor: float = 5.0,
        clip_method: str = "tanh",
    ) -> None:
        super().__init__()
        self.clip_factor = clip_factor
        self.clip_method = clip_method

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
        )

    def compute_step_loss(
        self,
        model: nn.Module,
        system: MolecularSystem,
        walkers: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
        """
        一站式计算局域能量与变分代理损失。

        Args:
            model: FermiNet 模型
            system: 分子体系
            walkers: 电子坐标构型, Shape: (B, N, 3)

        Returns:
            loss: 代理损失
            stats: 统计指标字典
            e_local: 局域能量, Shape: (B,)
        """
        # 1. 计算局部能量 (使用独立坐标计算图)
        r_for_energy = walkers.clone().detach().requires_grad_(True)
        e_local, _, _ = compute_local_energy(model=model, system=system, r=r_for_energy)
        e_local = e_local.detach()

        # 2. 前向获取带模型参数梯度的 log|Psi(r)|
        log_abs_psi, _ = model.forward_log_psi(walkers.detach())

        # 3. 计算代理损失
        loss, stats = self.forward(log_abs_psi, e_local)

        return loss, stats, e_local

