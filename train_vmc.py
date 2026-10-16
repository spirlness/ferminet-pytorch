"""
变分量子蒙特卡洛 (VMC) 端到端优化训练脚本
==========================================
实现基于有限批次协方差梯度与 MCMC 批量采样的完整基态能量求解闭环。
默认以氢分子 (H2, R = 1.4 a.u., 基态参考能约 -1.174 Hartree) 为基准体系演示。
"""

import argparse
import copy
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim

# 确保包能够被直接运行和识别
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ferminet_pytorch.model import FermiNet
    from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
    from ferminet_pytorch.mcmc import MCMCSampler
    from ferminet_pytorch.loss import VMCLoss
except ImportError:
    from model import FermiNet
    from hamiltonian import MolecularSystem, compute_local_energy
    from mcmc import MCMCSampler
    from loss import VMCLoss


def build_h2_system(
    bond_length: float = 1.4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> MolecularSystem:
    """
    构建氢分子 (H2) 物理体系。

    Args:
        bond_length (float): 两个氢原子核之间的平衡间距 (单位: Bohr / a.u.，默认 1.4)。
        device (Optional[torch.device]): 运行设备。
        dtype (torch.dtype): 数据精度。

    Returns:
        MolecularSystem: 初始化的氢分子体系。
    """
    half_r = bond_length / 2.0
    nuclear_coords = torch.tensor(
        [[0.0, 0.0, -half_r], [0.0, 0.0, half_r]],
        dtype=dtype,
        device=device,
    )
    nuclear_charges = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
    return MolecularSystem(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=1,
        n_down=1,
    )


def build_lih_system(
    bond_length: float = 3.015,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> MolecularSystem:
    """
    构建极性异核分子氢化锂 (LiH) 物理体系。
    锂核 Z_Li = 3.0，氢核 Z_H = 1.0；平衡键长约 3.015 a.u. (~ 1.595 Angstrom)。
    总电子数 N = 4 (闭壳层: N_up = 2, N_down = 2)。

    Args:
        bond_length (float): Li 与 H 核间距 (单位: Bohr / a.u.，默认 3.015)。
        device (Optional[torch.device]): 运算设备。
        dtype (torch.dtype): 数据精度。

    Returns:
        MolecularSystem: 初始化的氢化锂体系。
    """
    half_r = bond_length / 2.0
    # Li 在 -half_r, H 在 +half_r
    nuclear_coords = torch.tensor(
        [[0.0, 0.0, -half_r], [0.0, 0.0, half_r]],
        dtype=dtype,
        device=device,
    )
    # 异核核电荷量: Z_Li = 3.0, Z_H = 1.0
    nuclear_charges = torch.tensor([3.0, 1.0], dtype=dtype, device=device)
    return MolecularSystem(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=2,
        n_down=2,
    )


def build_cosine_scheduler(
    optimizer: optim.Optimizer, total_steps: int, min_ratio: float = 0.01
) -> optim.lr_scheduler.LambdaLR:
    """
    构建支持任意参数组独立初始学习率等比例退火的余弦退火调度器。

    克服 PyTorch 原生 CosineAnnealingLR 在单标量 eta_min 下
    对不同 base_lr 参数组导致的小学习率升温或卡死缺陷。
    """
    t_max = max(total_steps, 1)

    def lr_lambda(step: int) -> float:
        fraction = min(max(step, 0) / t_max, 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * fraction))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_autocorrelation_time(
    series: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    计算时间序列的滞后-1 自相关系数 rho_1、积分自相关时间 tau_int 和有效样本比例。

    对多 Walker 并行采样链，利用全局平稳均值汇聚各链自协方差，
    有效消除单链有限时间步均值的 Hurwicz-Kendall 负偏差，同时统一方差与协方差分母口径。

    Args:
        series: 形状为 (T,) 的标量序列或 (T, W) 的 walker 序列。

    Returns:
        rho_1: 滞后 1 的自相关系数，范围严格在 [-0.999, 0.999]。
        tau_int: 积分自相关时间 (用于校正有效样本量 N_eff = N_total / tau_int)。
        ess_fraction: 有效样本量占总样本量的比例 (N_eff / N_total)。
    """
    if series.ndim == 1:
        series = series.unsqueeze(1)  # (T, 1)

    T, W = series.shape
    if T < 2:
        return 0.0, 1.0, 1.0

    series_float = series.double()
    if not torch.isfinite(series_float).all():
        series_float = torch.nan_to_num(series_float, nan=0.0, posinf=0.0, neginf=0.0)

    # 跨时间与 Walker 的全局均值
    global_mean = torch.mean(series_float)
    diff = series_float - global_mean

    # 滞后 1 自协方差: sum_t sum_w d_t * d_{t+1} / ((T - 1) * W)
    cov_1 = torch.mean(diff[:-1, :] * diff[1:, :])

    # 对应子集的样本方差，保证 Cauchy-Schwarz 不等式成立从而 |rho_1| <= 1
    var_0 = torch.mean(diff[:-1, :] ** 2)
    var_1 = torch.mean(diff[1:, :] ** 2)
    denom = torch.sqrt(torch.clamp(var_0 * var_1, min=1e-12))

    rho_raw = (cov_1 / denom).item()
    if not math.isfinite(rho_raw):
        rho_1 = 0.0
    else:
        rho_1 = float(max(-0.999, min(0.999, rho_raw)))

    # 指数衰减近似下的自相关时间: tau_int = (1 + rho_1) / (1 - rho_1)
    if rho_1 > 0.0:
        tau_int = float((1.0 + rho_1) / max(1.0 - rho_1, 1e-4))
    else:
        tau_int = 1.0

    ess_fraction = float(1.0 / tau_int)
    return rho_1, tau_int, ess_fraction


def evaluate_frozen_wavefunction(
    model: nn.Module,
    system: MolecularSystem,
    num_walkers: int = 1024,
    num_steps: int = 100,
    burn_in_steps: int = 40,
    decorrelation_steps: int = 10,
    step_size: float = 0.5,
    target_acceptance: Optional[float] = None,
    move_mode: str = "all",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 2024,
    verbose: bool = True,
    exact_energy: Optional[float] = None,
    hf_energy: Optional[float] = None,
) -> Dict[str, Any]:
    """
    对训练好的冷冻波函数执行高统计精度 Monte Carlo 独立评估 (含去相关增强)。

    核心机制:
        在每次记录能量样本前，引入 MCMC 弛豫去相关步数 (decorrelation_steps >= 1)，
        使电子构型在相空间充分游走，显著压制样本间自相关系数 rho_1，大幅提升有效样本量 (ESS)，
        从而有效压制统计标准误 (SEM) 并杜绝自相关造成的统计假象。

    Args:
        model: FermiNet 深度波函数模型 (将置于 eval 模式并禁用梯度)。
        system: 目标分子体系。
        num_walkers: 并行 MCMC Walker 数量。
        num_steps: 能量采集记录步数。
        burn_in_steps: 正式采集前的热身预热步数。
        decorrelation_steps: 每次记录样本前执行的 MCMC 弛豫去相关步数 (>=1)。
        step_size: MCMC 移动提议步长。
        target_acceptance: 目标接受率。
        move_mode: MCMC 移动模式 ('all' 或 'one_by_one')。
        device: 计算设备。
        dtype: 浮点精度。
        seed: 随机数种子。
        verbose: 是否打印详细进度与报告。
        exact_energy: 可选的理论精确能量参考 (如 H2: -1.174476 Ha)。
        hf_energy: 可选的 Hartree-Fock 极限能量参考 (如 H2: -1.133630 Ha)。

    Returns:
        包含能量均值、方差、标准误 (SEM)、自相关系数 rho_1、自相关时间 tau_int、有效样本量等指标的字典。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = (
        torch.device(device)
        if device is not None
        else (system.nuclear_coords.device if hasattr(system, "nuclear_coords") else torch.device("cpu"))
    )

    was_training = model.training
    model.eval()
    decorrelation_steps = max(1, int(decorrelation_steps))

    try:
        sampler = MCMCSampler(
            model=model,
            system=system,
            num_walkers=num_walkers,
            step_size=step_size,
            target_acceptance=target_acceptance,
            adapt_step_size=True,
            move_mode=move_mode,
            device=target_device,
            dtype=dtype,
        )

        # 预热消除初始偏差
        start_burn = time.time()
        burn_acc = sampler.burn_in(num_steps=burn_in_steps)
        burn_time = time.time() - start_burn

        if verbose:
            print("=" * 72)
            print("          冷冻波函数独立统计评估 (去相关增强模式)")
            print("=" * 72)
            print(f"[*] 设备: {target_device} | 精度: {dtype}")
            print(f"[*] 配置: Walkers = {num_walkers}, 生产记录步数 = {num_steps}, 去相关步数 = {decorrelation_steps}")
            print(f"[*] 预热: {burn_in_steps} 步 (耗时: {burn_time:.2f}s, 接受率: {burn_acc * 100:.1f}%, 步长: {sampler.step_size:.4f})")
            print(f"[*] 开始高统计精度独立能量采样...", flush=True)

        all_energies: List[torch.Tensor] = []
        step_means: List[float] = []
        step_energies_matrix: List[torch.Tensor] = []

        start_eval = time.time()

        for step in range(1, num_steps + 1):
            # 评估去相关增强：每次采样前弛豫 decorrelation_steps 步
            walkers = sampler.sample(num_steps=decorrelation_steps, adapt=False)
            r = walkers.detach().clone().requires_grad_(True)
            # 计算无截断的真实局部能量
            e_loc, _, _ = compute_local_energy(model, system, r)
            valid_mask = torch.isfinite(e_loc)
            valid_e = e_loc[valid_mask].detach().cpu()
            if valid_e.numel() > 0:
                all_energies.append(valid_e)

            if valid_e.numel() == num_walkers:
                step_energies_matrix.append(valid_e)
            step_mean = float(valid_e.mean().item()) if valid_e.numel() > 0 else float("nan")
            step_means.append(step_mean)

            if verbose and (step == 1 or step % 20 == 0 or step == num_steps):
                running_mean = float(torch.cat(all_energies).mean().item()) if all_energies else float("nan")
                total_sampled = sum(len(e) for e in all_energies)
                print(
                    f"    - 进度: Step {step:^3d}/{num_steps} | "
                    f"累计样本: {total_sampled:^6d} | "
                    f"当前均值: {running_mean:.6f} Ha | "
                    f"最近步接受率: {sampler.last_acceptance_rate * 100:.1f}%",
                    flush=True,
                )

        eval_duration = time.time() - start_eval

        if not all_energies:
            # 异常防御：无任何有效采样
            return {
                "total_samples": 0,
                "energy_mean": float("nan"),
                "energy_var": float("nan"),
                "energy_std": float("nan"),
                "energy_sem": float("nan"),
                "energy_sem_naive": float("nan"),
                "energy_sem_ess": float("nan"),
                "energy_sem_block": float("nan"),
                "rho_1": 0.0,
                "rho_1_batch": 0.0,
                "tau_int": 1.0,
                "ess_fraction": 1.0,
                "effective_sample_size": 0.0,
                "decorrelation_steps": decorrelation_steps,
                "eval_time_sec": eval_duration,
            }

        concatenated = torch.cat(all_energies, dim=0).double()
        total_samples = concatenated.numel()

        mean_e = float(torch.mean(concatenated).item())
        var_e = float(torch.var(concatenated, unbiased=True).item()) if total_samples > 1 else 0.0
        std_e = math.sqrt(max(0.0, var_e))

        # 天真标准误 (忽略自相关)
        sem_naive = std_e / math.sqrt(max(1, total_samples))

        # 计算时间序列自相关与有效样本量
        if len(step_energies_matrix) == num_steps and num_steps >= 2:
            traj_tensor = torch.stack(step_energies_matrix, dim=0)  # (num_steps, num_walkers)
            rho_1, tau_int, ess_frac = compute_autocorrelation_time(traj_tensor)
        elif len(step_means) >= 2:
            valid_means = [m for m in step_means if math.isfinite(m)]
            if len(valid_means) >= 2:
                means_tensor = torch.tensor(valid_means, dtype=torch.float64)
                rho_1, tau_int, ess_frac = compute_autocorrelation_time(means_tensor)
            else:
                rho_1, tau_int, ess_frac = 0.0, 1.0, 1.0
        else:
            rho_1, tau_int, ess_frac = 0.0, 1.0, 1.0

        # 批次级自相关
        valid_step_means = [m for m in step_means if math.isfinite(m)]
        if len(valid_step_means) >= 2:
            means_tensor = torch.tensor(valid_step_means, dtype=torch.float64)
            rho_1_batch, _, _ = compute_autocorrelation_time(means_tensor)
        else:
            rho_1_batch = 0.0

        # 有效样本量与自相关校正后的标准误 (SEM_ESS)
        effective_sample_size = max(1.0, float(total_samples) / tau_int)
        sem_ess = std_e / math.sqrt(effective_sample_size)

        # 严密时间步分块平均法 (Flyvbjerg-Petersen Time-Step Block Averaging)
        # 严格在时间步 (production steps) 维度划分完整时间块，杜绝截断切断 Walker 批次
        n_steps_valid = len(valid_step_means)
        if n_steps_valid >= 4:
            num_blocks = min(20, max(2, n_steps_valid // 2))
            block_step_len = n_steps_valid // num_blocks
            usable_steps = num_blocks * block_step_len
            block_means = [
                sum(valid_step_means[b * block_step_len : (b + 1) * block_step_len]) / block_step_len
                for b in range(num_blocks)
            ]
            block_mean_all = sum(block_means) / num_blocks
            block_var = sum((bm - block_mean_all) ** 2 for bm in block_means) / max(1, num_blocks - 1)
            sem_block = math.sqrt(block_var / num_blocks)
        else:
            sem_block = sem_ess

        # 采用更保守严谨的 SEM 作为物理标准输出
        sem_e = max(sem_block, sem_ess)

        result: Dict[str, Any] = {
            "total_samples": total_samples,
            "energy_mean": mean_e,
            "energy_var": var_e,
            "energy_std": std_e,
            "energy_sem": sem_e,
            "energy_sem_naive": sem_naive,
            "energy_sem_ess": sem_ess,
            "energy_sem_block": sem_block,
            "rho_1": rho_1,
            "rho_1_batch": rho_1_batch,
            "tau_int": tau_int,
            "ess_fraction": ess_frac,
            "effective_sample_size": effective_sample_size,
            "decorrelation_steps": decorrelation_steps,
            "eval_time_sec": eval_duration,
        }
    finally:
        model.train(was_training)

    # 若提供了基准精确能或默认 H2 体系
    exact_ref = exact_energy
    if exact_ref is None and system.nuclear_charges.numel() == 2 and (system.nuclear_charges == 1.0).all():
        exact_ref = -1.17447593
        if hf_energy is None:
            hf_energy = -1.13362957

    if exact_ref is not None:
        abs_err_ha = abs(mean_e - exact_ref)
        abs_err_mha = abs_err_ha * 1000.0
        abs_err_kcal = abs_err_ha * 627.509474
        result["exact_energy"] = exact_ref
        result["abs_error_ha"] = abs_err_ha
        result["abs_error_mha"] = abs_err_mha
        result["abs_error_kcal"] = abs_err_kcal
        result["chemical_accuracy_met"] = (abs_err_mha <= 1.5936)
        if hf_energy is not None:
            corr_total = exact_ref - hf_energy
            corr_retrieved = ((mean_e - hf_energy) / corr_total) * 100.0 if corr_total != 0 else 0.0
            result["correlation_retrieved_pct"] = corr_retrieved

    if verbose:
        print("=" * 72)
        print("                独立评估报告 (去相关增强统计)")
        print("=" * 72)
        print(f"[*] 评估采样总数:   {total_samples:,} (有效样本量 ESS: {int(effective_sample_size):,}, 占比: {ess_frac*100:.1f}%)")
        print(f"[*] MCMC 去相关步数: {decorrelation_steps} 步 (自相关系数 rho_1: {rho_1:+.4f}, tau_int: {tau_int:.2f})")
        print(f"[*] 局部能量期望:   {mean_e:.6f} +/- {sem_e:.6f} Hartree (Naive SEM: {sem_naive:.6f})")
        print(f"[*] 局域能量方差:   {var_e:.6f} Hartree^2 (Std: {std_e:.5f})")
        if "abs_error_mha" in result:
            print(f"[*] 精确参考能量:   {exact_ref:.6f} Hartree")
            print(f"[*] 绝对误差:       {result['abs_error_ha']:.6f} Ha = {result['abs_error_mha']:.3f} mHa ({result['abs_error_kcal']:.3f} kcal/mol)")
            if "correlation_retrieved_pct" in result:
                print(f"[*] 相关能恢复率:   {result['correlation_retrieved_pct']:.2f}%")
            print(f"[*] 达到化学精度:   {'【是 (PASS)】' if result['chemical_accuracy_met'] else '【否 (FAIL)】'}")
        print("=" * 72)

    return result


def train_vmc(
    system: Optional[MolecularSystem] = None,
    model: Optional[FermiNet] = None,
    num_walkers: int = 512,
    num_epochs: int = 50,
    mcmc_steps_per_epoch: int = 10,
    burn_in_steps: int = 50,
    lr: float = 1e-3,
    step_size: float = 0.5,
    target_acceptance: Optional[float] = None,
    move_mode: str = "all",
    clip_factor: float = 5.0,
    clip_method: str = "tanh",
    grad_clip: float = 1.0,
    use_cusp_jastrow: bool = False,
    two_stage_jastrow: bool = False,
    jastrow_stage1_epochs: Optional[int] = None,
    jastrow_lr: Optional[float] = None,
    eval_after_train: bool = False,
    eval_steps: int = 50,
    eval_decorrelation_steps: int = 10,
    spin_adapted: bool = False,
    envelope_type: Optional[str] = None,
    smooth_cuspless: Optional[bool] = None,
    restore_best: bool = True,
    checkpoint_path: Optional[str] = None,
    compile: bool = False,
    compile_sampler: Optional[bool] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    执行简洁明了的 VMC 变分优化循环。

    Args:
        system: 目标分子体系 (默认自动创建 H2)
        model: FermiNet 模型 (若为 None 则自动创建轻量模型)
        num_walkers: MCMC 并行构型数
        num_epochs: 训练迭代轮数
        mcmc_steps_per_epoch: 每轮参数更新前执行的 MCMC 步数
        burn_in_steps: 训练开始前的 MCMC 预热步数
        lr: Adam 优化器学习率
        step_size: MCMC 初始移动步长
        target_acceptance: 目标接受率 (用于自适应步长调节，为 None 时由 MCMC 自动动态适配)
        move_mode: MCMC 移动模式 ('all' 或 'one_by_one')
        clip_factor: 局域能量 MAD 截断因子
        clip_method: 截断方式 ('tanh', 'hard', 'none')
        grad_clip: 梯度范数裁剪阈值
        use_cusp_jastrow: 是否启用显式 Kato 尖峰 Jastrow 因子
        spin_adapted: 是否启用自旋纯态适配网络 (如双电子单重态空间对称化)
        envelope_type: 核包络类型 ('gaussian' 或 'exponential', 默认根据 Jastrow 自动协同)
        smooth_cuspless: 是否使用平滑无尖峰特征 (默认根据 Jastrow 自动协同)
        restore_best: 训练结束后是否恢复为历史最低局域能量模型状态
        checkpoint_path: 模型检查点磁盘保存路径 (为 None 时不持久化)
        compile: 是否使用 torch.compile 对 MCMC 采样波函数进行内核融合编译加速
        compile_sampler: 是否显式指定对采样器使用编译加速 (等价于 compile)
        device: 运行设备 ('cpu' 或 'cuda')
        dtype: 浮点精度
        seed: 随机数种子
        verbose: 是否打印训练日志

    Returns:
        包含训练历史、模型、体系、采样器及历史最优状态的结果字典
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1. 体系与模型
    if system is None:
        system = build_h2_system(bond_length=1.4, device=target_device, dtype=dtype)
    else:
        system = system.to(device=target_device, dtype=dtype)

    if model is None:
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=8,
            hidden_dim_one=64,
            hidden_dim_two=16,
            num_layers=2,
            use_cusp_jastrow=use_cusp_jastrow,
            spin_adapted=spin_adapted,
            envelope_type=envelope_type,
            smooth_cuspless=smooth_cuspless,
        ).to(device=target_device, dtype=dtype)
    else:
        model = model.to(device=target_device, dtype=dtype)

    # 2. 损失函数与采样器
    vmc_loss = VMCLoss(clip_factor=clip_factor, clip_method=clip_method)

    use_compile = compile or bool(compile_sampler)
    sampler_forward_fn = None
    if use_compile:
        if verbose:
            print("[*] 编译优化: 正在为 MCMC 采样器启用 torch.compile 融合内核加速...")
        sampler_forward_fn = torch.compile(model.forward_log_psi)

    sampler = MCMCSampler(
        model=model,
        system=system,
        num_walkers=num_walkers,
        step_size=step_size,
        target_acceptance=target_acceptance,
        adapt_step_size=True,
        move_mode=move_mode,
        device=target_device,
        dtype=dtype,
        forward_log_psi_fn=sampler_forward_fn,
    )

    has_trainable_j = False
    if hasattr(model, "has_trainable_jastrow"):
        has_trainable_j = model.has_trainable_jastrow()
    elif getattr(model, "jastrow", None) is not None:
        has_trainable_j = any(isinstance(p, nn.Parameter) for p in model.jastrow.parameters())

    is_two_stage = two_stage_jastrow and has_trainable_j
    if two_stage_jastrow and not has_trainable_j and verbose:
        print("[*] 提示: 模型未启用可学习 Jastrow 因子，两阶段策略将以单阶段标准模式运行。")

    if is_two_stage:
        s1_epochs = jastrow_stage1_epochs if jastrow_stage1_epochs is not None else max(1, num_epochs // 2)
        s1_epochs = min(s1_epochs, num_epochs)
        eff_j_lr = jastrow_lr if jastrow_lr is not None else (lr * 0.1)

        # Stage 1: 冻结 Jastrow 参数
        if hasattr(model, "freeze_jastrow"):
            model.freeze_jastrow()
        elif getattr(model, "jastrow", None) is not None:
            for p in model.jastrow.parameters():
                p.requires_grad = False

        backbone_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(backbone_params, lr=lr)
        scheduler = build_cosine_scheduler(optimizer, max(s1_epochs, 1), min_ratio=0.01)
    else:
        s1_epochs = num_epochs
        eff_j_lr = lr
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = build_cosine_scheduler(optimizer, max(num_epochs, 1), min_ratio=0.01)

    # 3. 预热 (Burn-in)
    if verbose:
        print("=" * 72)
        print("         FermiNet 变分量子蒙特卡洛 (VMC) 教学训练")
        print("=" * 72)
        print(f"[*] 设备: {target_device} | 精度: {dtype}")
        print(f"[*] 体系: 核数 = {system.num_nuclei}, 电子数 = {system.total_electrons} (up={system.n_up}, down={system.n_down})")
        print(f"[*] 架构: 自旋纯态适配 = {spin_adapted} | Kato 尖峰 Jastrow = {use_cusp_jastrow}")
        if is_two_stage:
            print(f"[*] 策略: 两阶段 Jastrow 训练 (Stage 1: 1~{s1_epochs} 轮预训练轨道, Stage 2: {s1_epochs+1}~{num_epochs} 轮解冻微调 jastrow_lr={eff_j_lr:.2e})")
        print(f"[*] 采样: Walkers = {num_walkers}, 移动模式 = {move_mode}")
        print(f"[*] 编译: 采样器编译加速 = {use_compile}")
        print(f"[*] 预热: 正在执行 {burn_in_steps} 步 MCMC 预热...")

    start_burn_in = time.time()
    burn_in_acc = sampler.burn_in(num_steps=burn_in_steps)
    burn_in_time = time.time() - start_burn_in

    if verbose:
        print(f"    预热完成 (耗时: {burn_in_time:.2f}s, 平均接受率: {burn_in_acc * 100:.1f}%, 初始步长: {sampler.step_size:.4f})")
        print("-" * 72)
        print(f"{'Epoch':^6} | {'<E_raw> (Ha)':^14} | {'<E_clip> (Ha)':^14} | {'Var(E)':^10} | {'Acc (%)':^8} | {'Step':^7}")
        print("-" * 72)

    # 4. 训练主循环与最优模型状态跟踪 (分阶段与全局双重跟踪)
    best_energy = float("inf")
    best_epoch = 0
    best_model_state: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(model.state_dict())

    best_s1_energy = float("inf")
    best_s1_epoch = 0
    best_s1_state: Optional[Dict[str, torch.Tensor]] = None

    best_s2_energy = float("inf")
    best_s2_epoch = 0
    best_s2_state: Optional[Dict[str, torch.Tensor]] = None

    history: Dict[str, List[Any]] = {
        "energy_raw": [],
        "energy_clipped": [],
        "energy_var": [],
        "loss": [],
        "acceptance_rate": [],
        "step_size": [],
        "num_valid": [],
        "num_invalid": [],
        "stage": [],
    }

    start_train_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # 两阶段 Jastrow 切换点检测
        if is_two_stage and epoch == s1_epochs + 1:
            if verbose:
                print("=" * 72)
                print(f"[*] >>> 阶段切换: 进入 Stage 2 (Epoch {epoch} ~ {num_epochs})")
                print(f"[*] >>> 解冻 Jastrow 因子，激活独立微调学习率: jastrow_lr = {eff_j_lr:.2e}")
                print("=" * 72)

            # 1. 以 Stage 1 最优轨道状态作为 Stage 2 微调起点
            if restore_best and best_s1_state is not None:
                model.load_state_dict(best_s1_state)
                if verbose:
                    print(f"[*] 已成功将模型置于 Stage 1 最优轨道状态 (Epoch {best_s1_epoch}, E = {best_s1_energy:.5f} Ha)")

            # 2. 解冻 Jastrow 参数
            if hasattr(model, "unfreeze_jastrow"):
                model.unfreeze_jastrow()
            elif getattr(model, "jastrow", None) is not None:
                for p in model.jastrow.parameters():
                    p.requires_grad = True

            # 3. 构建多参数组优化器并重置专属调度器，杜绝学习率卡死
            if hasattr(model, "get_parameter_groups"):
                stage2_groups = model.get_parameter_groups(lr=lr, jastrow_lr=eff_j_lr)
            else:
                j_params = list(model.jastrow.parameters()) if getattr(model, "jastrow", None) is not None else []
                j_ids = {id(p) for p in j_params}
                b_params = [p for p in model.parameters() if id(p) not in j_ids and p.requires_grad]
                stage2_groups = [{"params": b_params, "lr": lr, "name": "backbone"}]
                if j_params:
                    stage2_groups.append({"params": [p for p in j_params if p.requires_grad], "lr": eff_j_lr, "name": "jastrow"})

            optimizer = optim.Adam(stage2_groups)
            remaining_epochs = max(num_epochs - s1_epochs, 1)
            scheduler = build_cosine_scheduler(optimizer, remaining_epochs, min_ratio=0.01)
            sampler.recompute_log_psi()

        # A. MCMC 抽样 (正式抽样冻结步长调整以保障细致平衡与时齐性)
        walkers = sampler.sample(num_steps=mcmc_steps_per_epoch, adapt=False)

        # B. 计算局域能量与代理损失
        loss, stats, _ = vmc_loss.compute_step_loss(model, system, walkers)

        # 紧随能量测定，在参数更新前保存产生当前最低能量期望的真实波函数状态 (theta_t)
        current_energy = float(stats["energy_mean"])
        if math.isfinite(current_energy):
            if current_energy < best_energy:
                best_energy = current_energy
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

            if is_two_stage:
                if epoch <= s1_epochs:
                    if current_energy < best_s1_energy:
                        best_s1_energy = current_energy
                        best_s1_epoch = epoch
                        best_s1_state = copy.deepcopy(model.state_dict())
                else:
                    if current_energy < best_s2_energy:
                        best_s2_energy = current_energy
                        best_s2_epoch = epoch
                        best_s2_state = copy.deepcopy(model.state_dict())

        # C. 反向传播更新网络参数
        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        scheduler.step()

        # D. 刷新当前 Walker 的对数波函数缓存
        sampler.recompute_log_psi()

        # E. 记录训练历史
        history["energy_raw"].append(stats["energy_mean"])
        history["energy_clipped"].append(stats["energy_clipped_mean"])
        history["energy_var"].append(stats["energy_var"])
        history["loss"].append(stats["loss"])
        history["acceptance_rate"].append(sampler.last_acceptance_rate)
        history["step_size"].append(sampler.step_size)
        history["num_valid"].append(stats["num_valid"])
        history["num_invalid"].append(stats["num_invalid"])
        history["stage"].append(1 if epoch <= s1_epochs else 2)

        # F. 磁盘 Checkpoint 持久化
        if checkpoint_path is not None:
            checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_energy": best_energy,
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
            }
            torch.save(checkpoint_data, checkpoint_path)

        # G. 周期性打印日志
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            stage_tag = f"S{1 if epoch <= s1_epochs else 2}"
            print(
                f"{epoch:^6d} | "
                f"{stats['energy_mean']:^14.5f} | "
                f"{stats['energy_clipped_mean']:^14.5f} | "
                f"{stats['energy_var']:^10.4f} | "
                f"{sampler.last_acceptance_rate * 100:^8.1f} | "
                f"{sampler.step_size:^7.4f} [{stage_tag}]"
            )

    total_train_time = time.time() - start_train_time
    final_energy = float(history["energy_raw"][-1]) if history["energy_raw"] else float("nan")
    final_var = float(history["energy_var"][-1]) if history["energy_var"] else float("nan")

    # 5. 恢复最优状态
    # 若为两阶段训练，目标状态为 Stage 2 联合微调产生的最优波函数，保留微调后的 Jastrow 参数
    target_best_state = None
    target_best_epoch = best_epoch
    target_best_energy = best_energy
    if is_two_stage and best_s2_state is not None:
        target_best_state = best_s2_state
        target_best_epoch = best_s2_epoch
        target_best_energy = best_s2_energy
    elif best_model_state is not None:
        target_best_state = best_model_state
        target_best_epoch = best_epoch
        target_best_energy = best_energy

    if restore_best and target_best_state is not None:
        model.load_state_dict(target_best_state)

    if verbose:
        print("-" * 72)
        print(f"[*] 训练完成! 总耗时: {total_train_time:.2f}s ({total_train_time / max(num_epochs, 1):.3f}s/epoch)")
        print(f"[*] 最优能量期望: {target_best_energy:.5f} Hartree (达成于 Epoch {target_best_epoch})")
        print(f"[*] 最终能量期望: {final_energy:.5f} Hartree (H2 真实基态参考: -1.1744 Hartree)")
        print(f"[*] 最终能量方差: {final_var:.5f}")
        if restore_best and target_best_state is not None:
            print(f"[*] 模型参数已成功恢复为 Epoch {target_best_epoch} 的最优权重。")
        if checkpoint_path is not None:
            print(f"[*] 训练 Checkpoint 已持久化保存至: {checkpoint_path}")
        print("=" * 72)

    # 6. 训练后独立评估 (可选)
    eval_result = None
    if eval_after_train:
        if verbose:
            print(f"[*] 正在执行训练后高统计精度冷冻评估 (去相关步数: {eval_decorrelation_steps})...")
        eval_result = evaluate_frozen_wavefunction(
            model=model,
            system=system,
            num_walkers=num_walkers,
            num_steps=eval_steps,
            decorrelation_steps=eval_decorrelation_steps,
            device=target_device,
            dtype=dtype,
            verbose=verbose,
        )

    return {
        "model": model,
        "system": system,
        "sampler": sampler,
        "history": history,
        "final_energy": final_energy,
        "final_variance": final_var,
        "best_model_state": target_best_state if target_best_state is not None else best_model_state,
        "best_energy": target_best_energy,
        "best_epoch": target_best_epoch,
        "stage1_best_energy": best_s1_energy if is_two_stage else None,
        "stage1_best_epoch": best_s1_epoch if is_two_stage else None,
        "stage2_best_energy": best_s2_energy if is_two_stage else None,
        "stage2_best_epoch": best_s2_epoch if is_two_stage else None,
        "is_two_stage": is_two_stage,
        "stage1_epochs": s1_epochs if is_two_stage else None,
        "eval_result": eval_result,
    }


def train_two_stage_jastrow(
    system: Optional[MolecularSystem] = None,
    model: Optional[FermiNet] = None,
    stage1_epochs: int = 25,
    stage2_epochs: int = 25,
    num_walkers: int = 512,
    mcmc_steps_per_epoch: int = 10,
    burn_in_steps: int = 50,
    lr: float = 1e-3,
    jastrow_lr: Optional[float] = None,
    eval_after_train: bool = True,
    eval_steps: int = 50,
    eval_decorrelation_steps: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    两阶段 Jastrow 变分量子蒙特卡洛训练高层工作流。

    第一阶段 (Stage 1): 冻结 Jastrow 阻尼参数 (b_ee, b_en)，使神经网络轨道平滑拟合电子波函数结构，
                       避免动能交叉项与 Jastrow 梯度剧烈震荡引发方差爆炸。
    第二阶段 (Stage 2): 解冻 Jastrow 参数，赋予独立温和学习率 (jastrow_lr) 开展端到端联合微调。
    评估阶段 (Eval):   训练后使用冷冻波函数与 MCMC 弛豫去相关步数进行高精度统计评估。
    """
    total_epochs = stage1_epochs + stage2_epochs
    return train_vmc(
        system=system,
        model=model,
        num_walkers=num_walkers,
        num_epochs=total_epochs,
        mcmc_steps_per_epoch=mcmc_steps_per_epoch,
        burn_in_steps=burn_in_steps,
        lr=lr,
        use_cusp_jastrow=True,
        two_stage_jastrow=True,
        jastrow_stage1_epochs=stage1_epochs,
        jastrow_lr=jastrow_lr,
        eval_after_train=eval_after_train,
        eval_steps=eval_steps,
        eval_decorrelation_steps=eval_decorrelation_steps,
        **kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FermiNet 变分量子蒙特卡洛 (VMC) 教学训练")
    parser.add_argument("--system", type=str, default="h2", choices=["h2", "lih"], help="目标体系 (h2 或 lih)")
    parser.add_argument("--epochs", type=int, default=50, help="训练迭代步数 (默认: 50)")
    parser.add_argument("--walkers", type=int, default=512, help="MCMC Walker 数量 (默认: 512)")
    parser.add_argument("--mcmc_steps", type=int, default=10, help="每步训练采样的 MCMC 步数 (默认: 10)")
    parser.add_argument("--burn_in", type=int, default=50, help="预热步数 (默认: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="优化器学习率 (默认: 1e-3)")
    parser.add_argument("--step_size", type=float, default=0.5, help="MCMC 初始移动步长 (默认: 0.5)")
    parser.add_argument("--move_mode", type=str, default="all", choices=["all", "one_by_one"], help="MCMC 移动模式")
    parser.add_argument("--clip_factor", type=float, default=5.0, help="局域能量 MAD 截断倍数 (默认: 5.0)")
    parser.add_argument("--clip_method", type=str, default="tanh", choices=["tanh", "hard", "none"], help="局域能量截断方法")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度范数裁剪阈值 (默认: 1.0)")
    parser.add_argument("--device", type=str, default=None, help="运行设备 (cuda 或 cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子 (默认: 42)")
    parser.add_argument("--cusp_jastrow", action="store_true", help="启用显式 Kato 尖峰 Jastrow 因子")
    parser.add_argument("--two_stage_jastrow", action="store_true", help="启用两阶段 Jastrow 训练策略 (先冻结预训练后解冻微调)")
    parser.add_argument("--jastrow_stage1_epochs", type=int, default=None, help="两阶段中 Stage 1 冻结 Jastrow 的迭代轮数")
    parser.add_argument("--jastrow_lr", type=float, default=None, help="Stage 2 中 Jastrow 因子的独立微调学习率")
    parser.add_argument("--eval_after_train", action="store_true", help="训练完成后立即执行冷冻波函数独立统计评估")
    parser.add_argument("--eval_steps", type=int, default=50, help="独立统计评估的采样记录步数 (默认: 50)")
    parser.add_argument("--eval_decorrelation_steps", type=int, default=10, help="评估采样前执行的 MCMC 弛豫去相关步数 (默认: 10)")
    parser.add_argument("--spin_adapted", action="store_true", help="启用自旋纯态投影网络")
    parser.add_argument("--envelope_type", type=str, default=None, choices=["gaussian", "exponential"], help="核包络类型")
    parser.add_argument("--smooth_cuspless", action="store_true", help="强制使用平滑无尖峰几何特征")
    parser.add_argument("--target_acceptance", type=float, default=None, help="MCMC 目标接受率 (全电子默认 0.234, 逐电子默认 0.50)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="模型检查点保存路径 (.pt)")
    parser.add_argument("--no_restore_best", action="store_true", help="禁用训练结束时自动恢复历史最优模型状态")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile 对 MCMC 采样波函数进行内核融合编译加速")
    parser.add_argument("--compile_sampler", action="store_true", help="显式指定对采样器使用编译加速 (等价于 --compile)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.system == "lih":
        system = build_lih_system()
    else:
        system = build_h2_system()

    train_vmc(
        system=system,
        num_epochs=args.epochs,
        num_walkers=args.walkers,
        mcmc_steps_per_epoch=args.mcmc_steps,
        burn_in_steps=args.burn_in,
        lr=args.lr,
        step_size=args.step_size,
        target_acceptance=args.target_acceptance,
        move_mode=args.move_mode,
        clip_factor=args.clip_factor,
        clip_method=args.clip_method,
        grad_clip=args.grad_clip,
        use_cusp_jastrow=args.cusp_jastrow or args.two_stage_jastrow,
        two_stage_jastrow=args.two_stage_jastrow,
        jastrow_stage1_epochs=args.jastrow_stage1_epochs,
        jastrow_lr=args.jastrow_lr,
        eval_after_train=args.eval_after_train,
        eval_steps=args.eval_steps,
        eval_decorrelation_steps=args.eval_decorrelation_steps,
        spin_adapted=args.spin_adapted,
        envelope_type=args.envelope_type,
        smooth_cuspless=args.smooth_cuspless if args.smooth_cuspless else None,
        restore_best=not args.no_restore_best,
        checkpoint_path=args.checkpoint_path,
        compile=args.compile,
        compile_sampler=args.compile_sampler,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()

