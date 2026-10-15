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
    spin_adapted: bool = False,
    envelope_type: Optional[str] = None,
    smooth_cuspless: Optional[bool] = None,
    restore_best: bool = True,
    checkpoint_path: Optional[str] = None,
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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs, 1), eta_min=lr * 0.01
    )

    # 3. 预热 (Burn-in)
    if verbose:
        print("=" * 72)
        print("         FermiNet 变分量子蒙特卡洛 (VMC) 教学训练")
        print("=" * 72)
        print(f"[*] 设备: {target_device} | 精度: {dtype}")
        print(f"[*] 体系: 核数 = {system.num_nuclei}, 电子数 = {system.total_electrons} (up={system.n_up}, down={system.n_down})")
        print(f"[*] 架构: 自旋纯态适配 = {spin_adapted} | Kato 尖峰 Jastrow = {use_cusp_jastrow}")
        print(f"[*] 采样: Walkers = {num_walkers}, 移动模式 = {move_mode}")
        print(f"[*] 预热: 正在执行 {burn_in_steps} 步 MCMC 预热...")

    start_burn_in = time.time()
    burn_in_acc = sampler.burn_in(num_steps=burn_in_steps)
    burn_in_time = time.time() - start_burn_in

    if verbose:
        print(f"    预热完成 (耗时: {burn_in_time:.2f}s, 平均接受率: {burn_in_acc * 100:.1f}%, 初始步长: {sampler.step_size:.4f})")
        print("-" * 72)
        print(f"{'Epoch':^6} | {'<E_raw> (Ha)':^14} | {'<E_clip> (Ha)':^14} | {'Var(E)':^10} | {'Acc (%)':^8} | {'Step':^7}")
        print("-" * 72)

    # 4. 训练主循环与最优模型状态跟踪
    best_energy = float("inf")
    best_epoch = 0
    best_model_state: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(model.state_dict())

    history: Dict[str, List[float]] = {
        "energy_raw": [],
        "energy_clipped": [],
        "energy_var": [],
        "loss": [],
        "acceptance_rate": [],
        "step_size": [],
        "num_valid": [],
        "num_invalid": [],
    }

    start_train_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # A. MCMC 抽样 (正式抽样冻结步长调整以保障细致平衡与时齐性)
        walkers = sampler.sample(num_steps=mcmc_steps_per_epoch, adapt=False)

        # B. 计算局域能量与代理损失
        loss, stats, _ = vmc_loss.compute_step_loss(model, system, walkers)

        # 紧随能量测定，在参数更新前保存产生当前最低能量期望的真实波函数状态 (theta_t)
        current_energy = float(stats["energy_mean"])
        if math.isfinite(current_energy) and current_energy < best_energy:
            best_energy = current_energy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

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
            print(
                f"{epoch:^6d} | "
                f"{stats['energy_mean']:^14.5f} | "
                f"{stats['energy_clipped_mean']:^14.5f} | "
                f"{stats['energy_var']:^10.4f} | "
                f"{sampler.last_acceptance_rate * 100:^8.1f} | "
                f"{sampler.step_size:^7.4f}"
            )

    total_train_time = time.time() - start_train_time
    final_energy = float(history["energy_raw"][-1]) if history["energy_raw"] else float("nan")
    final_var = float(history["energy_var"][-1]) if history["energy_var"] else float("nan")

    # 5. 恢复历史最优状态
    if restore_best and best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print("-" * 72)
        print(f"[*] 训练完成! 总耗时: {total_train_time:.2f}s ({total_train_time / max(num_epochs, 1):.3f}s/epoch)")
        print(f"[*] 最优能量期望: {best_energy:.5f} Hartree (达成于 Epoch {best_epoch})")
        print(f"[*] 最终能量期望: {final_energy:.5f} Hartree (H2 真实基态参考: -1.1744 Hartree)")
        print(f"[*] 最终能量方差: {final_var:.5f}")
        if restore_best and best_model_state is not None:
            print(f"[*] 模型参数已成功恢复为 Epoch {best_epoch} 的最优权重。")
        if checkpoint_path is not None:
            print(f"[*] 训练 Checkpoint 已持久化保存至: {checkpoint_path}")
        print("=" * 72)

    return {
        "model": model,
        "system": system,
        "sampler": sampler,
        "history": history,
        "final_energy": final_energy,
        "final_variance": final_var,
        "best_model_state": best_model_state,
        "best_energy": best_energy,
        "best_epoch": best_epoch,
    }


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
    parser.add_argument("--spin_adapted", action="store_true", help="启用自旋纯态投影网络")
    parser.add_argument("--envelope_type", type=str, default=None, choices=["gaussian", "exponential"], help="核包络类型")
    parser.add_argument("--smooth_cuspless", action="store_true", help="强制使用平滑无尖峰几何特征")
    parser.add_argument("--target_acceptance", type=float, default=None, help="MCMC 目标接受率 (全电子默认 0.234, 逐电子默认 0.50)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="模型检查点保存路径 (.pt)")
    parser.add_argument("--no_restore_best", action="store_true", help="禁用训练结束时自动恢复历史最优模型状态")
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
        use_cusp_jastrow=args.cusp_jastrow,
        spin_adapted=args.spin_adapted,
        envelope_type=args.envelope_type,
        smooth_cuspless=args.smooth_cuspless if args.smooth_cuspless else None,
        restore_best=not args.no_restore_best,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
