"""
M4 端到端 VMC 训练收敛性与能量稳定性实测脚本
=============================================
全面实测 H2 分子基态下的 VMC 变分优化演化过程。
涵盖:
1. 逐 Epoch 跟踪局域能量期望 <E_L>、方差 Var(E_L)、标准差 Std(E_L) 及真实标准误 SEM。
2. 逐步监控 MCMC 接受率演化与自适应步长动态平衡过程。
3. 记录 Energy Clipping (MAD Tanh 截断) 触发频次、截断带宽与极端离群值抑制行为。
4. 对比原生 Baseline FermiNet、重构现代化 FermiNet 及 Cusp-Jastrow Spin-Adapted FermiNet。
5. 训练后冷冻波函数独立高精度统计评估 (Frozen Wavefunction Evaluation)。
"""

from ferminet_pytorch.vmc_driver import TrainingConfig, train_vmc
import argparse
import copy
import json
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem
from ferminet_pytorch.systems import build_h2_system
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction

# 物理常数与基准参考
EXACT_ENERGY = -1.174475714220  # Pachucki (2010), Table IV, BO R=1.4 bohr (Hartree)
HF_ENERGY = -1.13362957  # Hartree-Fock 极限 (Hartree)
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
    lr: float = 0.001,
    step_size: float = 0.5,
    clip_factor: float = 5.0,
    clip_method: str = "tanh",
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    verbose: bool = True,
):
    """Experiment setup and reporting; all updates use the shared VMCDriver."""
    config = TrainingConfig(
        num_walkers=num_walkers,
        num_epochs=num_epochs,
        mcmc_steps_per_epoch=mcmc_steps_per_epoch,
        burn_in_steps=burn_in_steps,
        lr=lr,
        step_size=step_size,
        clip_factor=clip_factor,
        clip_method=clip_method,
        grad_clip=grad_clip,
        device=device,
        dtype=dtype,
        seed=seed,
        verbose=verbose,
    )
    start = time.perf_counter()
    result = train_vmc(system=system, model=model, config=config)
    elapsed = time.perf_counter() - start
    history = result["history"]
    rows = [
        dict(
            epoch=i + 1,
            energy_raw=e,
            energy_clipped=history["energy_clipped"][i],
            energy_var=history["energy_var"][i],
            loss=history["loss"][i],
            mcmc_acc=history["acceptance_rate"][i],
        )
        for i, e in enumerate(history["energy_raw"])
    ]
    evaluation = evaluate_frozen_wavefunction(
        result["model"],
        result["system"],
        num_walkers=512,
        num_steps=30,
        burn_in_steps=30,
        decorrelation_steps=5,
        device=next(model.parameters()).device,
        dtype=dtype,
        verbose=verbose,
    )
    return dict(
        experiment_name=experiment_name,
        model_class=model.__class__.__name__,
        interaction_type=model.interaction_type,
        total_train_time=elapsed,
        best_energy=result["best_energy"],
        best_epoch=result["best_epoch"],
        final_energy=result["final_energy"],
        final_var=result["final_variance"],
        final_sem=evaluation["energy_sem"],
        epoch_logs=rows,
        eval_result=evaluation,
    )


def run_all_benchmarks(
    device_str: Optional[str] = None, epochs: int = 15, walkers: int = 256
) -> Dict[str, Any]:
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

    # 2. Exp 2: Psiformer (重构解耦现代化波函数)
    model_psiformer = FermiNet(
        interaction_type="psiformer",
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
    res_psiformer = detailed_vmc_train_tracker(
        experiment_name="Exp2_Psiformer",
        model=model_psiformer,
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
    all_results["composable"] = res_psiformer

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
    parser = argparse.ArgumentParser(
        description="M4 端到端 VMC 训练收敛性与能量稳定性实测"
    )
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数 (默认: 15)")
    parser.add_argument(
        "--walkers", type=int, default=256, help="Walker 数量 (默认: 256)"
    )
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda 或 cpu)")
    parser.add_argument(
        "--output_json", type=str, default=None, help="导出 JSON 数据路径"
    )
    args = parser.parse_args()

    results = run_all_benchmarks(
        device_str=args.device, epochs=args.epochs, walkers=args.walkers
    )

    if args.output_json:
        # 去除非可序列化对象
        serializable_results = copy.deepcopy(results)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n[*] 实测数据已成功保存至: {args.output_json}")
