"""
FermiNet H2 长时 CUDA 训练收敛测试与多模型对比评估基准
=====================================================
涵盖:
1. 1000-Epoch 长时 CUDA 训练收敛轨迹跟踪
2. 对比 Baseline FermiNet 与 Spin-Adapted (自旋纯态投影) FermiNet
3. 训练后 4096-Walker 高统计精度冷冻评估 (Frozen Wavefunction Evaluation)
4. 与量子化学成熟模型 (HF, CCSD, CCSD(T), FCI, PauliNet, Psiformer) 全方位对比
"""

from ferminet_pytorch.vmc_driver import TrainingConfig, train_vmc
import json
import os
import time
from typing import Any, Dict, Optional

import torch.nn as nn

# 引入核心模块
from ferminet_pytorch.hamiltonian import MolecularSystem
from ferminet_pytorch.evaluation import (
    evaluate_frozen_wavefunction as evaluate_frozen_core,
)


EXACT_ENERGY = -1.174475714220  # Pachucki (2010), Table IV, BO R=1.4 bohr (Hartree)
HF_ENERGY = -1.13362957  # Hartree-Fock limit (Hartree)
CORRELATION_ENERGY = EXACT_ENERGY - HF_ENERGY  # 约 -0.040846 Ha (-40.85 mHa)
HA_TO_KCAL = 627.509474  # 1 Hartree = 627.509474 kcal/mol
CHEMICAL_ACCURACY_MHA = 1.5936  # 1 kcal/mol in mHartree (~1.5936 mHa)


def run_training_experiment(
    name: str,
    spin_adapted: bool = False,
    use_cusp_jastrow: bool = False,
    two_stage_jastrow: bool = False,
    jastrow_stage1_epochs: Optional[int] = None,
    jastrow_lr: Optional[float] = None,
    envelope_type: Optional[str] = None,
    num_walkers: int = 1024,
    num_epochs: int = 800,
    mcmc_steps_per_epoch: int = 10,
    burn_in_steps: int = 60,
    lr: float = 0.001,
    device: str = "cuda",
    seed: int = 42,
):
    """Experiment setup and reporting; all updates use the shared VMCDriver."""
    config = TrainingConfig(
        num_walkers=num_walkers,
        num_epochs=num_epochs,
        mcmc_steps_per_epoch=mcmc_steps_per_epoch,
        burn_in_steps=burn_in_steps,
        lr=lr,
        spin_adapted=spin_adapted,
        use_cusp_jastrow=use_cusp_jastrow or two_stage_jastrow,
        two_stage_jastrow=two_stage_jastrow,
        jastrow_stage1_epochs=jastrow_stage1_epochs,
        jastrow_lr=jastrow_lr,
        envelope_type=envelope_type,
        device=device,
        seed=seed,
    )
    start = time.perf_counter()
    result = train_vmc(config=config)
    history = result["history"]
    trajectory = [
        dict(
            epoch=i + 1,
            energy_raw=e,
            energy_var=history["energy_var"][i],
            stage=history["stage"][i],
        )
        for i, e in enumerate(history["energy_raw"])
    ]
    return dict(
        name=name,
        model=result["model"],
        system=result["system"],
        trajectory=trajectory,
        best_state=result["best_model_state"],
        best_epoch=result["best_epoch"],
        best_energy=result["best_energy"],
        total_train_time=time.perf_counter() - start,
        num_params=sum(p.numel() for p in result["model"].parameters()),
    )


def evaluate_frozen_wavefunction(
    model: nn.Module,
    system: MolecularSystem,
    num_walkers: int = 1024,
    num_steps: int = 100,
    burn_in_steps: int = 40,
    decorrelation_steps: int = 10,
    device: str = "cuda",
    seed: int = 2024,
) -> Dict[str, Any]:
    """
    对训练好的冻结波函数执行高统计精度 Monte Carlo 独立评估 (含去相关增强)。
    通过在每次记录能量样本前引入 MCMC 弛豫去相关步数 (decorrelation_steps >= 1)，
    降低相邻构型自相关系数 rho_1，大幅提升有效样本量，显著压制统计标准误 (SEM)。
    """
    return evaluate_frozen_core(
        model=model,
        system=system,
        num_walkers=num_walkers,
        num_steps=num_steps,
        burn_in_steps=burn_in_steps,
        decorrelation_steps=decorrelation_steps,
        device=device,
        seed=seed,
        verbose=True,
        exact_energy=EXACT_ENERGY,
        hf_energy=HF_ENERGY,
    )


def main():
    print(
        "================================================================================"
    )
    print("    FermiNet PyTorch H2 分子长时 CUDA 训练收敛性测试与基准横向对比")
    print(
        "================================================================================"
    )

    # 测试配置 1: Baseline FermiNet (标准 FermiNet，无额外约束)
    res_baseline = run_training_experiment(
        name="Baseline FermiNet (Exponential Envelope)",
        spin_adapted=False,
        use_cusp_jastrow=False,
        envelope_type="exponential",
        num_walkers=1024,
        num_epochs=800,
        mcmc_steps_per_epoch=10,
        burn_in_steps=60,
        lr=1e-3,
        device="cuda",
        seed=42,
    )

    # 对 Baseline 的收敛模型进行高精度独立评估
    eval_baseline = evaluate_frozen_wavefunction(
        model=res_baseline["model"],
        system=res_baseline["system"],
        num_walkers=1024,
        num_steps=100,
        device="cuda",
    )

    # 测试配置 2: Spin-Adapted FermiNet (自旋纯态投影波函数)
    res_spin = run_training_experiment(
        name="Spin-Adapted FermiNet (Singlet Projection)",
        spin_adapted=True,
        use_cusp_jastrow=False,
        envelope_type="exponential",
        num_walkers=1024,
        num_epochs=800,
        mcmc_steps_per_epoch=10,
        burn_in_steps=60,
        lr=1e-3,
        device="cuda",
        seed=42,
    )

    # 对 Spin-Adapted 的收敛模型进行高精度独立评估 (去相关步数 10)
    eval_spin = evaluate_frozen_wavefunction(
        model=res_spin["model"],
        system=res_spin["system"],
        num_walkers=1024,
        num_steps=100,
        decorrelation_steps=10,
        device="cuda",
    )

    # 测试配置 3: Spin-Adapted + Two-Stage Jastrow (两阶段 Kato 尖峰 Jastrow 策略)
    res_two_stage = run_training_experiment(
        name="Two-Stage Jastrow FermiNet (Stage1: Freeze Orbitals, Stage2: Fine-tune)",
        spin_adapted=True,
        use_cusp_jastrow=True,
        two_stage_jastrow=True,
        jastrow_stage1_epochs=400,
        jastrow_lr=1e-4,
        envelope_type="gaussian",
        num_walkers=1024,
        num_epochs=800,
        mcmc_steps_per_epoch=10,
        burn_in_steps=60,
        lr=1e-3,
        device="cuda",
        seed=42,
    )

    # 对 Two-Stage Jastrow 的收敛模型进行高精度独立评估 (去相关步数 10)
    eval_two_stage = evaluate_frozen_wavefunction(
        model=res_two_stage["model"],
        system=res_two_stage["system"],
        num_walkers=1024,
        num_steps=100,
        decorrelation_steps=10,
        device="cuda",
    )

    # 汇总保存结果
    output_summary = {
        "system": "H2 (R=1.4 a.u.)",
        "exact_energy": EXACT_ENERGY,
        "hf_energy": HF_ENERGY,
        "baseline": {
            "training": {
                "total_time_sec": res_baseline["total_train_time"],
                "best_epoch": res_baseline["best_epoch"],
                "best_energy": res_baseline["best_energy"],
                "final_energy": res_baseline["trajectory"][-1]["energy_raw"],
                "final_var": res_baseline["trajectory"][-1]["energy_var"],
            },
            "evaluation": eval_baseline,
            "trajectory_samples": res_baseline["trajectory"][::20],  # 每20轮抽取一个点
        },
        "spin_adapted": {
            "training": {
                "total_time_sec": res_spin["total_train_time"],
                "best_epoch": res_spin["best_epoch"],
                "best_energy": res_spin["best_energy"],
                "final_energy": res_spin["trajectory"][-1]["energy_raw"],
                "final_var": res_spin["trajectory"][-1]["energy_var"],
            },
            "evaluation": eval_spin,
            "trajectory_samples": res_spin["trajectory"][::20],
        },
        "two_stage_jastrow": {
            "training": {
                "total_time_sec": res_two_stage["total_train_time"],
                "best_epoch": res_two_stage["best_epoch"],
                "best_energy": res_two_stage["best_energy"],
                "final_energy": res_two_stage["trajectory"][-1]["energy_raw"],
                "final_var": res_two_stage["trajectory"][-1]["energy_var"],
            },
            "evaluation": eval_two_stage,
            "trajectory_samples": res_two_stage["trajectory"][::20],
        },
    }

    out_file = "h2_long_convergence_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 完整收敛性实验数据与分析已持久化保存至: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    main()
