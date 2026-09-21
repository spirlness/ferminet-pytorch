"""
变分量子蒙特卡洛 (VMC) 端到端优化训练脚本
==========================================
实现基于无偏变分梯度与 MCMC 批量采样的完整基态能量求解闭环。
默认以氢分子 (H2, R = 1.4 a.u., 基态参考能约 -1.174 Hartree) 为基准体系演示。
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional
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
    from ferminet_pytorch.hamiltonian import MolecularSystem
    from ferminet_pytorch.mcmc import MCMCSampler
    from ferminet_pytorch.loss import VMCLoss
except ImportError:
    from model import FermiNet
    from hamiltonian import MolecularSystem
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


def train_vmc(
    system: Optional[MolecularSystem] = None,
    model: Optional[FermiNet] = None,
    num_walkers: int = 1024,
    num_epochs: int = 100,
    mcmc_steps_per_epoch: int = 10,
    burn_in_steps: int = 100,
    lr: float = 1e-3,
    step_size: float = 0.5,
    target_acceptance: float = 0.5,
    move_mode: str = "all",
    clip_factor: float = 5.0,
    clip_method: str = "tanh",
    laplacian_method: str = "exact",
    num_hutchinson_samples: int = 1,
    hutchinson_distribution: str = "rademacher",
    grad_clip: float = 1.0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    执行完整的 VMC 优化迭代闭环。

    Args:
        system (Optional[MolecularSystem]): 目标分子体系 (默认自动创建 H2)。
        model (Optional[FermiNet]): 预先初始化的 FermiNet (默认按系统尺寸自动创建)。
        num_walkers (int): 并行 MCMC Walker 数量。
        num_epochs (int): 训练迭代步数。
        mcmc_steps_per_epoch (int): 每一轮参数更新之间进行的 MCMC 抽样步数。
        burn_in_steps (int): 初始 MCMC 预热步数。
        lr (float): 优化器学习率。
        step_size (float): MCMC 提议高斯位移初值。
        target_acceptance (float): MCMC 目标接受率 (默认 0.5)。
        move_mode (str): MCMC 步进模式 ('all' 或 'one_by_one')。
        clip_factor (float): 局域能量平滑截断阈值因子 (MAD 倍数)。
        clip_method (str): 截断方式 ('tanh', 'hard', 'none')。
        laplacian_method (str): 局域动能拉普拉斯求解策略 ('exact', 'hutchinson', 'exact_jvp', 'jvp')。
        num_hutchinson_samples (int): Hutchinson 随机迹估计采样次数 (默认为 1)。
        hutchinson_distribution (str): Hutchinson 探测向量分布 ('rademacher' 或 'gaussian'，默认 'rademacher')。
        grad_clip (float): 梯度范数裁剪阈值。
        device (Optional[str]): 运行设备 ('cuda' 或 'cpu'，若为 None 自动检测)。
        dtype (torch.dtype): 浮点运算精度。
        seed (int): 随机数种子。
        verbose (bool): 是否打印日志与进度。

    Returns:
        Dict[str, Any]: 训练结果字典，包含模型、体系、历史能量与方差序列等。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设备检测与匹配
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 1. 体系与模型构建
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
            use_nuclear_charges=True,
        ).to(device=target_device, dtype=dtype)
    else:
        model = model.to(device=target_device, dtype=dtype)

    # 2. 损失函数与采样器构建
    vmc_loss = VMCLoss(
        clip_factor=clip_factor,
        clip_method=clip_method,
        laplacian_method=laplacian_method,
        num_hutchinson_samples=num_hutchinson_samples,
        hutchinson_distribution=hutchinson_distribution,
    )
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

    # 3. 预热 (Burn-in)
    if verbose:
        print("=" * 75)
        print("         FermiNet 变分量子蒙特卡洛 (VMC) 训练优化启动")
        print("=" * 75)
        print(f"[*] 运行设备: {target_device} | 浮点精度: {dtype}")
        print(f"[*] 体系信息: 原子核数 = {system.num_nuclei}, 总电子数 = {system.total_electrons}")
        print(f"[*] 采样配置: Walkers = {num_walkers}, 移动模式 = {move_mode}")
        print(f"[*] 预热阶段: 正在执行 {burn_in_steps} 步 MCMC 预热...")

    start_burn_in = time.time()
    burn_in_acc = sampler.burn_in(num_steps=burn_in_steps)
    burn_in_time = time.time() - start_burn_in

    if verbose:
        print(f"    预热完成 (耗时: {burn_in_time:.2f}s, 最终平均接受率: {burn_in_acc * 100:.1f}%, 步长: {sampler.step_size:.4f})")
        print("-" * 75)
        print(f"{'Epoch':^6} | {'<E_raw> (Ha)':^14} | {'<E_clip> (Ha)':^14} | {'Var(E)':^10} | {'SEM':^9} | {'Acc (%)':^8} | {'Step':^7}")
        print("-" * 75)

    # 4. 优化器配置
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. 训练主循环
    history: Dict[str, List[float]] = {
        "energy_raw": [],
        "energy_clipped": [],
        "energy_std": [],
        "energy_var": [],
        "energy_sem": [],
        "loss": [],
        "acceptance_rate": [],
        "step_size": [],
    }

    start_train_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # A. MCMC 步进更新电子构型
        walkers = sampler.sample(num_steps=mcmc_steps_per_epoch)

        # B. 计算局部能量与无偏代理损失
        loss, stats, _ = vmc_loss.compute_step_loss(model, system, walkers)

        # C. 优化器反向传播与更新
        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # D. 关键：模型参数更新后，刷新采样器波函数对数缓存
        sampler.recompute_log_psi()

        # E. 记录指标
        history["energy_raw"].append(stats["energy_mean"])
        history["energy_clipped"].append(stats["energy_clipped_mean"])
        history["energy_std"].append(stats["energy_std"])
        history["energy_var"].append(stats["energy_var"])
        history["energy_sem"].append(stats["energy_sem"])
        history["loss"].append(stats["loss"])
        history["acceptance_rate"].append(sampler.last_acceptance_rate)
        history["step_size"].append(sampler.step_size)

        # F. 打印日志
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            print(
                f"{epoch:^6d} | "
                f"{stats['energy_mean']:^14.5f} | "
                f"{stats['energy_clipped_mean']:^14.5f} | "
                f"{stats['energy_var']:^10.4f} | "
                f"{stats['energy_sem']:^9.4f} | "
                f"{sampler.last_acceptance_rate * 100:^8.1f} | "
                f"{sampler.step_size:^7.4f}"
            )

    total_train_time = time.time() - start_train_time
    if verbose:
        print("-" * 75)
        print(f"[*] 训练顺利完成! 总耗时: {total_train_time:.2f}s ({total_train_time / num_epochs:.3f}s/epoch)")
        final_energy = history["energy_raw"][-1]
        final_var = history["energy_var"][-1]
        ref_energy_str = ""
        if system.total_electrons == 2 and system.num_nuclei == 2:
            ref_energy_str = " (H2 参考: ~ -1.174 Hartree)"
        elif system.total_electrons == 4 and system.num_nuclei == 2:
            ref_energy_str = " (LiH 参考: ~ -8.070 Hartree)"
        print(f"[*] 最终能量期望值: {final_energy:.5f} Hartree{ref_energy_str}")
        print(f"[*] 最终能量方差: {final_var:.5f}")
        print("=" * 75)

    return {
        "model": model,
        "system": system,
        "sampler": sampler,
        "history": history,
        "final_energy": history["energy_raw"][-1],
        "final_variance": history["energy_var"][-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FermiNet 变分量子蒙特卡洛 (VMC) 训练")
    parser.add_argument("--system", type=str, default="h2", choices=["h2", "lih"], help="目标体系 (h2 或 lih，默认 h2)")
    parser.add_argument("--laplacian_method", type=str, default="exact", choices=["exact", "hutchinson", "exact_jvp", "jvp"], help="局域动能拉普拉斯求解策略")
    parser.add_argument("--hutch_samples", type=int, default=1, help="Hutchinson 采样次数 (默认 1)")
    parser.add_argument("--hutch_dist", type=str, default="rademacher", choices=["rademacher", "gaussian"], help="Hutchinson 随机向量分布")
    parser.add_argument("--epochs", type=int, default=50, help="训练迭代步数 (默认: 50)")
    parser.add_argument("--walkers", type=int, default=1024, help="MCMC Walker 数量 (默认: 1024)")
    parser.add_argument("--mcmc_steps", type=int, default=10, help="每步训练采样的 MCMC 步数 (默认: 10)")
    parser.add_argument("--burn_in", type=int, default=100, help="预热步数 (默认: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="优化器学习率 (默认: 1e-3)")
    parser.add_argument("--step_size", type=float, default=0.5, help="MCMC 初始移动步长 (默认: 0.5)")
    parser.add_argument("--move_mode", type=str, default="all", choices=["all", "one_by_one"], help="MCMC 移动模式")
    parser.add_argument("--clip_factor", type=float, default=5.0, help="局域能量 MAD 截断倍数 (默认: 5.0)")
    parser.add_argument("--clip_method", type=str, default="tanh", choices=["tanh", "hard", "none"], help="局域能量截断方法")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度范数裁剪阈值 (默认: 1.0)")
    parser.add_argument("--device", type=str, default=None, help="运行设备 (cuda 或 cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子 (默认: 42)")
    # 可视化参数
    parser.add_argument("--plot", action="store_true", default=False, help="训练完成后自动生成并展示全部可视化图表")
    parser.add_argument("--save_plots", action="store_true", default=False, help="将所有可视化图表保存为 PNG 文件")
    parser.add_argument("--plot_dir", type=str, default="assets/plots", help="图表保存目录 (默认: assets/plots)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.system == "lih":
        system = build_lih_system()
    else:
        system = build_h2_system()

    result = train_vmc(
        system=system,
        num_epochs=args.epochs,
        num_walkers=args.walkers,
        mcmc_steps_per_epoch=args.mcmc_steps,
        burn_in_steps=args.burn_in,
        lr=args.lr,
        step_size=args.step_size,
        move_mode=args.move_mode,
        clip_factor=args.clip_factor,
        clip_method=args.clip_method,
        laplacian_method=args.laplacian_method,
        num_hutchinson_samples=args.hutch_samples,
        hutchinson_distribution=args.hutch_dist,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )

    # 训练后可视化生成
    if args.plot or args.save_plots:
        try:
            from visualize import plot_all_visualizations
        except ImportError:
            from ferminet_pytorch.visualize import plot_all_visualizations

        # 从最终采样器获取当前 walkers 和局域能量
        final_walkers = result["sampler"].current_positions.detach()
        trained_model = result["model"]
        trained_system = result["system"]

        # 计算最终局域能量用于直方图
        with torch.no_grad():
            from hamiltonian import compute_local_energy
            e_local, _, _ = compute_local_energy(
                trained_model, trained_system, final_walkers
            )

        save_dir = args.plot_dir if args.save_plots else None
        saved = plot_all_visualizations(
            history=result["history"],
            system=trained_system,
            model=trained_model,
            walkers=final_walkers,
            local_energies=e_local,
            plot_dir=args.plot_dir if args.save_plots else "assets/plots",
            prefix=args.system,
        )

        if args.save_plots and saved:
            print(f"\n[*] 已保存 {len(saved)} 张可视化图表到 {os.path.abspath(args.plot_dir)}/:")
            for name, path in saved.items():
                print(f"    - {name}: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
