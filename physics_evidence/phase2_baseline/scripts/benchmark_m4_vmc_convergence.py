"""
M4 端到端 VMC 训练收敛性与能量稳定性实测脚本
=============================================
全面实测 H2 分子基态下的 VMC 变分优化演化过程。
涵盖:
1. 逐 Epoch 跟踪局域能量期望 <E_L>、方差 Var(E_L)、标准差 Std(E_L) 及真实标准误 SEM。
2. 逐步监控 MCMC 接受率演化与自适应步长动态平衡过程。
3. 记录 Energy Clipping (MAD Tanh 截断) 触发频次、截断带宽与极端离群值抑制行为。
4. 对比原生 Baseline FermiNet、重构现代化 ComposableFermiNet 及 Cusp-Jastrow Spin-Adapted FermiNet。
5. 训练后冷冻波函数独立高精度统计评估 (Frozen Wavefunction Evaluation)。
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.composable_model import ComposableFermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.loss import VMCLoss, clip_local_energy
from ferminet_pytorch.train_vmc import build_h2_system, build_cosine_scheduler, compute_autocorrelation_time, evaluate_frozen_wavefunction

# 物理常数与基准参考
EXACT_ENERGY = -1.174475714220  # Pachucki (2010), Table IV, BO R=1.4 bohr (Hartree)
HF_ENERGY = -1.13362957     # Hartree-Fock 极限 (Hartree)
CORRELATION_ENERGY = EXACT_ENERGY - HF_ENERGY  # -40.846 mHa
HA_TO_KCAL = 627.509474
CHEMICAL_ACCURACY_MHA = 1.5936  # 1 kcal/mol


def detailed_vmc_train_tracker(
    experiment_name: str,
    model: nn.Module,
    system: MolecularSystem,
    num_walkers: int = 256,
    num_epochs: int = 15,
    mcmc_steps_per_epoch: int = 10,
    burn_in_steps: int = 40,
    lr: float = 1e-3,
    step_size: float = 0.5,
    clip_factor: float = 5.0,
    clip_method: str = "tanh",
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    带全量细节监控的端到端 VMC 训练执行器。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    system = system.to(device=target_device, dtype=dtype)
    model = model.to(device=target_device, dtype=dtype)

    sampler = MCMCSampler(
        model=model,
        system=system,
        num_walkers=num_walkers,
        step_size=step_size,
        adapt_step_size=True,
        move_mode="all",
        device=target_device,
        dtype=dtype,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = build_cosine_scheduler(optimizer, num_epochs, min_ratio=0.01)

    if verbose:
        print("\n" + "=" * 90)
        print(f"[*] 启动实测实验: 【{experiment_name}】")
        print(f"    - 模型类型: {model.__class__.__name__} (参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
        print(f"    - Walker 规模: {num_walkers} | 训练轮数: {num_epochs} | MCMC 步数/轮: {mcmc_steps_per_epoch}")
        print(f"    - 设备: {target_device} | 初始学习率: {lr} | 截断因子: {clip_factor} ({clip_method})")
        print(f"[*] 正在执行 {burn_in_steps} 步 MCMC 预热...")

    t0_burn = time.perf_counter()
    burn_acc = sampler.burn_in(num_steps=burn_in_steps)
    t_burn = time.perf_counter() - t0_burn

    if verbose:
        print(f"    预热完成! 耗时: {t_burn:.2f}s | 平均接受率: {burn_acc * 100:.1f}% | 预热后步长: {sampler.step_size:.4f}")
        print("-" * 90)
        print(f"{'Epoch':^5} | {'<E_raw> (Ha)':^12} | {'<E_clip> (Ha)':^12} | {'Var(E)':^9} | {'SEM':^9} | {'MAD':^7} | {'Clip#':^6} | {'Acc (%)':^7} | {'Step':^6} | {'Time':^6}")
        print("-" * 90)

    epoch_logs: List[Dict[str, Any]] = []
    best_energy = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    total_train_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        t_epoch_start = time.perf_counter()

        # 1. MCMC 抽样
        walkers = sampler.sample(num_steps=mcmc_steps_per_epoch, adapt=False)

        # 2. 局域能量计算
        r_eval = walkers.clone().detach().requires_grad_(True)
        e_local, t_local, v_local = compute_local_energy(
            model=model,
            system=system,
            r=r_eval,
            create_graph=False,
        )

        e_detached = e_local.detach()
        valid_mask = torch.isfinite(e_detached)
        num_valid = int(valid_mask.sum().item())
        num_invalid = num_walkers - num_valid

        if num_valid == 0:
            raise FloatingPointError(f"Epoch {epoch}: 全部局域能量均非有限值!")

        finite_e = e_detached[valid_mask]
        raw_mean = float(torch.mean(finite_e).item())
        raw_var = float(torch.var(finite_e, unbiased=True).item()) if num_valid > 1 else 0.0
        raw_std = math.sqrt(max(0.0, raw_var))
        sem_naive = raw_std / math.sqrt(max(1, num_valid))

        # 3. 局域能量截断与截断触发监控
        e_clipped, e_median, mad = clip_local_energy(
            finite_e, clip_factor=clip_factor, method=clip_method
        )
        clip_mean = float(torch.mean(e_clipped).item())
        delta = max(1e-3, clip_factor * float(mad.item()))

        # 统计受到截断压缩显著的离群点数 (|E_L - median| > delta)
        deviation = torch.abs(finite_e - e_median)
        clipped_samples = int((deviation > delta).sum().item())
        clip_ratio = clipped_samples / float(num_valid)

        # 4. 变分代理损失与梯度更新
        delta_e = (e_clipped - torch.mean(e_clipped)).detach()
        log_abs_psi, _ = model.forward_log_psi(walkers)
        loss = 2.0 * torch.mean(delta_e * log_abs_psi[valid_mask])

        # Snapshot theta_t, whose walkers produced raw_mean, before updating.
        # 跟踪历史最优
        if math.isfinite(raw_mean) and raw_mean < best_energy:
            best_energy = raw_mean
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        optimizer.zero_grad()
        loss.backward()

        # 监测梯度范数与数值健康度
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad_norm = math.sqrt(total_norm)

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        scheduler.step()

        # 刷新采样器对数波函数缓存
        sampler.recompute_log_psi()

        t_epoch = time.perf_counter() - t_epoch_start

        log_item = {
            "epoch": epoch,
            "energy_raw": raw_mean,
            "energy_clipped": clip_mean,
            "energy_var": raw_var,
            "energy_std": raw_std,
            "energy_sem": sem_naive,
            "energy_median": float(e_median.item()),
            "energy_mad": float(mad.item()),
            "clip_delta": delta,
            "clipped_samples": clipped_samples,
            "clip_ratio": clip_ratio,
            "loss": float(loss.item()),
            "grad_norm": total_grad_norm,
            "num_valid": num_valid,
            "num_invalid": num_invalid,
            "mcmc_acc": sampler.last_acceptance_rate,
            "step_size": sampler.step_size,
            "epoch_time": t_epoch,
            "kinetic_mean": float(t_local.mean().item()),
            "potential_mean": float(v_local.mean().item()),
        }
        epoch_logs.append(log_item)

        if verbose:
            print(
                f"{epoch:^5d} | "
                f"{raw_mean:^12.5f} | "
                f"{clip_mean:^12.5f} | "
                f"{raw_var:^9.4f} | "
                f"{sem_naive:^9.4f} | "
                f"{mad.item():^7.4f} | "
                f"{clipped_samples:^6d} | "
                f"{sampler.last_acceptance_rate * 100:^7.1f} | "
                f"{sampler.step_size:^6.4f} | "
                f"{t_epoch:^6.2f}s"
            )

    total_train_time = time.perf_counter() - total_train_start

    # 恢复最优权重
    model.load_state_dict(best_state)

    if verbose:
        print("-" * 90)
        print(f"[*] 训练完成! 总耗时: {total_train_time:.2f}s ({total_train_time / num_epochs:.3f}s/epoch)")
        print(f"[*] 最优能量: {best_energy:.5f} Hartree (达成于 Epoch {best_epoch})")
        print(f"[*] 最终能量: {epoch_logs[-1]['energy_raw']:.5f} Hartree (理论基准: {EXACT_ENERGY:.5f} Ha)")
        print(f"[*] 最终方差: {epoch_logs[-1]['energy_var']:.5f} | 最终 SEM: {epoch_logs[-1]['energy_sem']:.5f}")

    # 独立高精度冷冻评估
    if verbose:
        print("[*] 正在执行训练后 512-Walker 高统计精度冷冻评估 (去相关步数: 5)...")
    eval_res = evaluate_frozen_wavefunction(
        model=model,
        system=system,
        num_walkers=512,
        num_steps=30,
        burn_in_steps=30,
        decorrelation_steps=5,
        device=target_device,
        dtype=dtype,
        verbose=verbose,
        exact_energy=EXACT_ENERGY,
        hf_energy=HF_ENERGY,
    )

    return {
        "experiment_name": experiment_name,
        "model_class": model.__class__.__name__,
        "total_train_time": total_train_time,
        "best_energy": best_energy,
        "best_epoch": best_epoch,
        "final_energy": epoch_logs[-1]["energy_raw"],
        "final_var": epoch_logs[-1]["energy_var"],
        "final_sem": epoch_logs[-1]["energy_sem"],
        "epoch_logs": epoch_logs,
        "eval_result": eval_res,
    }


def run_all_benchmarks(device_str: Optional[str] = None, epochs: int = 15, walkers: int = 256) -> Dict[str, Any]:
    """
    运行全套实测基准并将结果返回。
    """
    target_device = (
        torch.device(device_str)
        if device_str is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float32

    system = build_h2_system(bond_length=1.4, device=target_device, dtype=dtype)

    all_results: Dict[str, Any] = {}

    # 1. Exp 1: Baseline FermiNet
    model_baseline = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=8,
        hidden_dim_one=64,
        hidden_dim_two=16,
        num_layers=2,
        use_cusp_jastrow=False,
        spin_adapted=False,
    )
    res_baseline = detailed_vmc_train_tracker(
        experiment_name="Exp1_Baseline_FermiNet",
        model=model_baseline,
        system=system,
        num_walkers=walkers,
        num_epochs=epochs,
        mcmc_steps_per_epoch=10,
        burn_in_steps=40,
        lr=1e-3,
        device=target_device,
        dtype=dtype,
        seed=42,
    )
    all_results["baseline"] = res_baseline

    # 2. Exp 2: ComposableFermiNet Modern (重构解耦现代化波函数)
    model_composable = ComposableFermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=8,
        hidden_dim_one=64,
        hidden_dim_two=16,
        num_layers=2,
        use_cusp_jastrow=False,
        spin_adapted=False,
    )
    res_composable = detailed_vmc_train_tracker(
        experiment_name="Exp2_Composable_FermiNet_Modern",
        model=model_composable,
        system=system,
        num_walkers=walkers,
        num_epochs=epochs,
        mcmc_steps_per_epoch=10,
        burn_in_steps=40,
        lr=1e-3,
        device=target_device,
        dtype=dtype,
        seed=42,
    )
    all_results["composable"] = res_composable

    # 3. Exp 3: Cusp-Jastrow + Spin-Adapted FermiNet (物理先验增强架构)
    model_physical = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=8,
        hidden_dim_one=64,
        hidden_dim_two=16,
        num_layers=2,
        use_cusp_jastrow=True,
        spin_adapted=True,
    )
    res_physical = detailed_vmc_train_tracker(
        experiment_name="Exp3_Cusp_Jastrow_Spin_Adapted",
        model=model_physical,
        system=system,
        num_walkers=walkers,
        num_epochs=epochs,
        mcmc_steps_per_epoch=10,
        burn_in_steps=40,
        lr=1e-3,
        device=target_device,
        dtype=dtype,
        seed=42,
    )
    all_results["physical_enhanced"] = res_physical

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4 端到端 VMC 训练收敛性与能量稳定性实测")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数 (默认: 15)")
    parser.add_argument("--walkers", type=int, default=256, help="Walker 数量 (默认: 256)")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda 或 cpu)")
    parser.add_argument("--output_json", type=str, default=None, help="导出 JSON 数据路径")
    args = parser.parse_args()

    results = run_all_benchmarks(device_str=args.device, epochs=args.epochs, walkers=args.walkers)

    if args.output_json:
        # 去除非可序列化对象
        serializable_results = copy.deepcopy(results)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n[*] 实测数据已成功保存至: {args.output_json}")
