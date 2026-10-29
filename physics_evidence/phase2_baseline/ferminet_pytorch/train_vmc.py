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

from .model import FermiNet
from .hamiltonian import MolecularSystem, compute_local_energy
from .mcmc import MCMCSampler
from .loss import VMCLoss


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


from .vmc_statistics import autocorrelation_time as compute_autocorrelation_time
from .evaluation import evaluate_frozen_wavefunction


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
        sampler.recompute_log_psi()

    if verbose:
        print("-" * 72)
        print(f"[*] 训练完成! 总耗时: {total_train_time:.2f}s ({total_train_time / max(num_epochs, 1):.3f}s/epoch)")
        print(f"[*] 最优能量期望: {target_best_energy:.5f} Hartree (达成于 Epoch {target_best_epoch})")
        print(f"[*] 最终能量期望: {final_energy:.5f} Hartree (精度请参考冻结波函数独立评估)")
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

