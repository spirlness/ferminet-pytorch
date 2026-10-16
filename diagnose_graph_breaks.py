"""
全面深度图断裂 (Graph Break) 探测与量化分析脚本
覆盖:
1. 两阶段 Jastrow 策略下的波函数前向 (forward_log_psi, forward, KatoCuspJastrow)
2. 局域能量计算 (compute_local_energy, compute_potential_energy, compute_kinetic_energy)
3. VMC 损失计算与截断 (clip_local_energy, vmc_surrogate_loss)
4. 评估去相关增强计算逻辑 (compute_autocorrelation_time 及相关张量操作)
5. 端到端流程整合与各模块编译可行性
"""

import sys
import os
import io
import math
import traceback
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch._dynamo

# 导入项目中相关核心模块
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.jastrow import KatoCuspJastrow
from ferminet_pytorch.hamiltonian import (
    MolecularSystem,
    compute_local_energy,
    compute_kinetic_energy,
)
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss
from ferminet_pytorch.train_vmc import (
    build_h2_system,
    build_lih_system,
    compute_autocorrelation_time,
)


def run_explain(fn, *args, **kwargs) -> Dict[str, Any]:
    """
    运行 torch._dynamo.explain 并捕获详细的图断裂原因和位置。
    """
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(fn)(*args, **kwargs)
        break_reasons = []
        for br in explanation.break_reasons:
            break_reasons.append({
                "reason": str(br.reason),
                "user_stack": [str(frame) for frame in getattr(br, "user_stack", [])],
            })
        return {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": break_reasons,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "graph_count": -1,
            "graph_break_count": -1,
            "break_reasons": [],
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def test_jastrow_and_forward(device: torch.device):
    print(f"\n=======================================================")
    print(f"[1] 测试两阶段 Jastrow 策略相关前向传播 (Device: {device})")
    print(f"=======================================================")
    results = {}
    system = build_h2_system(device=device, dtype=torch.float32)
    B = 16
    r = torch.randn(B, system.total_electrons, 3, device=device, dtype=torch.float32)

    # 1.1 独立 KatoCuspJastrow 模块
    print("\n--- 1.1 独立 KatoCuspJastrow 模块 ---")
    jastrow = KatoCuspJastrow(
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
    ).to(device)

    def jastrow_fwd(coords):
        return jastrow(coords, system.nuclear_coords)

    # (a) 解冻 (Stage 2)
    for p in jastrow.parameters():
        p.requires_grad = True
    res_jastrow_unfrozen = run_explain(jastrow_fwd, r)
    print(f"Jastrow (Stage 2 / 解冻) forward: breaks={res_jastrow_unfrozen['graph_break_count']}, graphs={res_jastrow_unfrozen['graph_count']}")
    results["jastrow_unfrozen"] = res_jastrow_unfrozen

    # (b) 冻结 (Stage 1)
    for p in jastrow.parameters():
        p.requires_grad = False
    res_jastrow_frozen = run_explain(jastrow_fwd, r)
    print(f"Jastrow (Stage 1 / 冻结) forward: breaks={res_jastrow_frozen['graph_break_count']}, graphs={res_jastrow_frozen['graph_count']}")
    results["jastrow_frozen"] = res_jastrow_frozen

    # 1.2 FermiNet 在不同 Jastrow 阶段下的 forward_log_psi
    print("\n--- 1.2 FermiNet forward_log_psi ---")
    # (a) 无 Jastrow (Baseline)
    model_no_jastrow = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=2,
        use_cusp_jastrow=False,
    ).to(device).eval()
    res_no_j = run_explain(model_no_jastrow.forward_log_psi, r)
    print(f"FermiNet (无 Jastrow) forward_log_psi: breaks={res_no_j['graph_break_count']}, graphs={res_no_j['graph_count']}")
    results["ferminet_no_jastrow_log_psi"] = res_no_j

    # (b) Stage 1: Jastrow 开启且冻结
    model_jastrow = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=2,
        use_cusp_jastrow=True,
    ).to(device).eval()
    model_jastrow.freeze_jastrow()
    res_stage1 = run_explain(model_jastrow.forward_log_psi, r)
    print(f"FermiNet (Stage 1 / 冻结 Jastrow) forward_log_psi: breaks={res_stage1['graph_break_count']}, graphs={res_stage1['graph_count']}")
    results["ferminet_stage1_log_psi"] = res_stage1

    # (c) Stage 2: Jastrow 开启且解冻
    model_jastrow.unfreeze_jastrow()
    res_stage2 = run_explain(model_jastrow.forward_log_psi, r)
    print(f"FermiNet (Stage 2 / 解冻 Jastrow) forward_log_psi: breaks={res_stage2['graph_break_count']}, graphs={res_stage2['graph_count']}")
    results["ferminet_stage2_log_psi"] = res_stage2

    # (d) 开启自旋纯态投影 + Jastrow (Stage 1 与 Stage 2)
    model_spin_jastrow = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=2,
        use_cusp_jastrow=True,
        spin_adapted=True,
    ).to(device).eval()
    model_spin_jastrow.freeze_jastrow()
    res_spin_s1 = run_explain(model_spin_jastrow.forward_log_psi, r)
    print(f"FermiNet (Spin + Stage 1 Jastrow) forward_log_psi: breaks={res_spin_s1['graph_break_count']}, graphs={res_spin_s1['graph_count']}")
    results["ferminet_spin_stage1_log_psi"] = res_spin_s1

    model_spin_jastrow.unfreeze_jastrow()
    res_spin_s2 = run_explain(model_spin_jastrow.forward_log_psi, r)
    print(f"FermiNet (Spin + Stage 2 Jastrow) forward_log_psi: breaks={res_spin_s2['graph_break_count']}, graphs={res_spin_s2['graph_count']}")
    results["ferminet_spin_stage2_log_psi"] = res_spin_s2

    # 1.3 FermiNet 在不同阶段下的 forward (直接输出 Psi)
    print("\n--- 1.3 FermiNet forward (直接返回 Psi) ---")
    res_fwd_no_j = run_explain(model_no_jastrow.forward, r)
    print(f"FermiNet (无 Jastrow) forward: breaks={res_fwd_no_j['graph_break_count']}, graphs={res_fwd_no_j['graph_count']}")
    results["ferminet_no_jastrow_fwd"] = res_fwd_no_j

    model_jastrow.freeze_jastrow()
    res_fwd_s1 = run_explain(model_jastrow.forward, r)
    print(f"FermiNet (Stage 1 Jastrow) forward: breaks={res_fwd_s1['graph_break_count']}, graphs={res_fwd_s1['graph_count']}")
    results["ferminet_stage1_fwd"] = res_fwd_s1

    model_jastrow.unfreeze_jastrow()
    res_fwd_s2 = run_explain(model_jastrow.forward, r)
    print(f"FermiNet (Stage 2 Jastrow) forward: breaks={res_fwd_s2['graph_break_count']}, graphs={res_fwd_s2['graph_count']}")
    results["ferminet_stage2_fwd"] = res_fwd_s2

    # 1.4 奇异碰撞构型测试 (det=0)
    print("\n--- 1.4 奇异碰撞构型测试 (det=0) ---")
    r_sing = r.clone()
    r_sing[:, 1, :] = r_sing[:, 0, :]
    res_sing_s1 = run_explain(model_jastrow.forward_log_psi, r_sing)
    print(f"FermiNet (Stage 1 Jastrow, 奇异坐标) forward_log_psi: breaks={res_sing_s1['graph_break_count']}, graphs={res_sing_s1['graph_count']}")
    results["ferminet_singular_stage1"] = res_sing_s1

    return results


def test_hamiltonian_and_local_energy(device: torch.device):
    print(f"\n=======================================================")
    print(f"[2] 测试哈密顿量与局域能量计算 (Device: {device})")
    print(f"=======================================================")
    results = {}
    system = build_h2_system(device=device, dtype=torch.float32)
    B = 8
    r = torch.randn(B, system.total_electrons, 3, device=device, dtype=torch.float32)

    # 2.1 势能计算 compute_potential_energy
    print("\n--- 2.1 势能计算 system.compute_potential_energy ---")
    res_pot = run_explain(system.compute_potential_energy, r)
    print(f"compute_potential_energy: breaks={res_pot['graph_break_count']}, graphs={res_pot['graph_count']}")
    if res_pot['graph_break_count'] > 0:
        for idx, br in enumerate(res_pot['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
    results["compute_potential_energy"] = res_pot

    # 2.2 动能与局域能量 compute_local_energy
    print("\n--- 2.2 局域能量计算 compute_local_energy ---")
    model_s1 = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=2,
        use_cusp_jastrow=True,
    ).to(device).eval()
    model_s1.freeze_jastrow()

    r_grad = r.clone().requires_grad_(True)

    # 探测 compute_local_energy (create_graph=False)
    def call_cle_s1(coords):
        return compute_local_energy(model_s1, system, coords, create_graph=False)

    res_cle_s1 = run_explain(call_cle_s1, r_grad)
    print(f"compute_local_energy (Stage 1 / Jastrow 冻结): success={res_cle_s1['success']}, breaks={res_cle_s1['graph_break_count']}, graphs={res_cle_s1['graph_count']}")
    if res_cle_s1['error']:
        print(f"  Error: {res_cle_s1['error']}")
    if res_cle_s1['graph_break_count'] > 0:
        for idx, br in enumerate(res_cle_s1['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
            for stack in br.get('user_stack', []):
                print(f"    Stack: {stack}")
    results["compute_local_energy_stage1"] = res_cle_s1

    # Stage 2 (解冻)
    model_s1.unfreeze_jastrow()
    def call_cle_s2(coords):
        return compute_local_energy(model_s1, system, coords, create_graph=False)
    res_cle_s2 = run_explain(call_cle_s2, r_grad)
    print(f"compute_local_energy (Stage 2 / Jastrow 解冻): success={res_cle_s2['success']}, breaks={res_cle_s2['graph_break_count']}, graphs={res_cle_s2['graph_count']}")
    if res_cle_s2['graph_break_count'] > 0:
        for idx, br in enumerate(res_cle_s2['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
    results["compute_local_energy_stage2"] = res_cle_s2

    # Baseline (无 Jastrow)
    model_no_j = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=2,
        use_cusp_jastrow=False,
    ).to(device).eval()
    def call_cle_no_j(coords):
        return compute_local_energy(model_no_j, system, coords, create_graph=False)
    res_cle_no_j = run_explain(call_cle_no_j, r_grad)
    print(f"compute_local_energy (无 Jastrow): success={res_cle_no_j['success']}, breaks={res_cle_no_j['graph_break_count']}, graphs={res_cle_no_j['graph_count']}")
    if res_cle_no_j['graph_break_count'] > 0:
        for idx, br in enumerate(res_cle_no_j['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
    results["compute_local_energy_no_jastrow"] = res_cle_no_j

    # 深入细分: 探测为什么 compute_local_energy 发生图断裂
    # 是因为 torch.autograd.grad 还是 _FreezeModelParameters 上下文管理器？
    print("\n--- 2.3 细分分解: _FreezeModelParameters 与 autograd.grad ---")
    from hamiltonian import _FreezeModelParameters

    def dummy_freeze_ctx(coords):
        with _FreezeModelParameters(model_s1):
            log_psi, _ = model_s1.forward_log_psi(coords)
            return log_psi
    res_freeze_ctx = run_explain(dummy_freeze_ctx, r)
    print(f"_FreezeModelParameters + forward_log_psi: breaks={res_freeze_ctx['graph_break_count']}, graphs={res_freeze_ctx['graph_count']}")
    if res_freeze_ctx['graph_break_count'] > 0:
        for idx, br in enumerate(res_freeze_ctx['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
    results["freeze_ctx_break"] = res_freeze_ctx

    def dummy_autograd_grad(coords):
        log_psi, _ = model_s1.forward_log_psi(coords)
        ones = torch.ones_like(log_psi)
        grad = torch.autograd.grad(log_psi, coords, grad_outputs=ones, create_graph=True)[0]
        return grad
    res_autograd_grad = run_explain(dummy_autograd_grad, r_grad)
    print(f"torch.autograd.grad (一阶导): breaks={res_autograd_grad['graph_break_count']}, graphs={res_autograd_grad['graph_count']}")
    if res_autograd_grad['graph_break_count'] > 0:
        for idx, br in enumerate(res_autograd_grad['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
    results["autograd_grad_break"] = res_autograd_grad

    return results


def test_loss_and_clipping(device: torch.device):
    print(f"\n=======================================================")
    print(f"[3] 测试 VMC 局域能量截断与代理损失 (Device: {device})")
    print(f"=======================================================")
    results = {}
    B = 64
    e_local = torch.randn(B, device=device, dtype=torch.float32) * 2.0 - 1.15
    log_abs_psi = torch.randn(B, device=device, dtype=torch.float32, requires_grad=True)

    # 3.1 clip_local_energy
    print("\n--- 3.1 clip_local_energy ---")
    # (a) method='tanh'
    def call_clip_tanh(e):
        return clip_local_energy(e, clip_factor=5.0, method="tanh")
    res_clip_tanh = run_explain(call_clip_tanh, e_local)
    print(f"clip_local_energy (tanh): breaks={res_clip_tanh['graph_break_count']}, graphs={res_clip_tanh['graph_count']}")
    if res_clip_tanh['graph_break_count'] > 0:
        for idx, br in enumerate(res_clip_tanh['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
            for stack in br.get('user_stack', []):
                print(f"    Stack: {stack}")
    results["clip_local_energy_tanh"] = res_clip_tanh

    # (b) method='hard'
    def call_clip_hard(e):
        return clip_local_energy(e, clip_factor=5.0, method="hard")
    res_clip_hard = run_explain(call_clip_hard, e_local)
    print(f"clip_local_energy (hard): breaks={res_clip_hard['graph_break_count']}, graphs={res_clip_hard['graph_count']}")
    results["clip_local_energy_hard"] = res_clip_hard

    # (c) method='none'
    def call_clip_none(e):
        return clip_local_energy(e, clip_factor=0.0, method="none")
    res_clip_none = run_explain(call_clip_none, e_local)
    print(f"clip_local_energy (none): breaks={res_clip_none['graph_break_count']}, graphs={res_clip_none['graph_count']}")
    results["clip_local_energy_none"] = res_clip_none

    # 3.2 vmc_surrogate_loss
    print("\n--- 3.2 vmc_surrogate_loss ---")
    def call_surrogate_loss(lp, el):
        return vmc_surrogate_loss(lp, el, clip_factor=5.0, clip_method="tanh")
    res_surr_loss = run_explain(call_surrogate_loss, log_abs_psi, e_local)
    print(f"vmc_surrogate_loss: breaks={res_surr_loss['graph_break_count']}, graphs={res_surr_loss['graph_count']}")
    if res_surr_loss['graph_break_count'] > 0:
        for idx, br in enumerate(res_surr_loss['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
            for stack in br.get('user_stack', []):
                print(f"    Stack: {stack}")
    results["vmc_surrogate_loss"] = res_surr_loss

    # 纯张量代理损失 (消除 stats 字典和 python control flow 后的核心公式)
    print("\n--- 3.3 纯张量代理损失计算公式 ---")
    def pure_tensor_surrogate_loss(lp, el_clipped):
        # L = 2.0 * mean( (el_clipped - mean(el_clipped)).detach() * lp )
        diff = (el_clipped - torch.mean(el_clipped)).detach()
        return 2.0 * torch.mean(diff * lp)
    res_pure_loss = run_explain(pure_tensor_surrogate_loss, log_abs_psi, e_local)
    print(f"pure_tensor_surrogate_loss: breaks={res_pure_loss['graph_break_count']}, graphs={res_pure_loss['graph_count']}")
    results["pure_tensor_surrogate_loss"] = res_pure_loss

    return results


def test_autocorrelation_and_decorrelation(device: torch.device):
    print(f"\n=======================================================")
    print(f"[4] 测试评估去相关增强核心张量计算 (Device: {device})")
    print(f"=======================================================")
    results = {}
    T, W = 50, 64
    series_2d = torch.randn(T, W, device=device, dtype=torch.float32)
    series_1d = torch.randn(T, device=device, dtype=torch.float32)

    # 4.1 compute_autocorrelation_time 完整函数
    print("\n--- 4.1 compute_autocorrelation_time 原生函数 ---")
    res_ac_2d = run_explain(compute_autocorrelation_time, series_2d)
    print(f"compute_autocorrelation_time (2D): breaks={res_ac_2d['graph_break_count']}, graphs={res_ac_2d['graph_count']}")
    if res_ac_2d['graph_break_count'] > 0:
        for idx, br in enumerate(res_ac_2d['break_reasons']):
            print(f"  Break {idx+1}: {br['reason']}")
            for stack in br.get('user_stack', []):
                print(f"    Stack: {stack}")
    results["autocorr_2d"] = res_ac_2d

    res_ac_1d = run_explain(compute_autocorrelation_time, series_1d)
    print(f"compute_autocorrelation_time (1D): breaks={res_ac_1d['graph_break_count']}, graphs={res_ac_1d['graph_count']}")
    results["autocorr_1d"] = res_ac_1d

    # 4.2 纯张量版本的自协方差核心计算 (排除 float/item/math.isfinite 等 Python 转换)
    print("\n--- 4.2 compute_autocorrelation_time 纯张量核心运算 ---")
    def pure_tensor_autocorr_core(series):
        if series.ndim == 1:
            series = series.unsqueeze(1)
        series_float = series.double()
        global_mean = torch.mean(series_float)
        diff = series_float - global_mean
        cov_1 = torch.mean(diff[:-1, :] * diff[1:, :])
        var_0 = torch.mean(diff[:-1, :] ** 2)
        var_1 = torch.mean(diff[1:, :] ** 2)
        denom = torch.sqrt(torch.clamp(var_0 * var_1, min=1e-12))
        rho_raw = cov_1 / denom
        rho_1 = torch.clamp(rho_raw, min=-0.999, max=0.999)
        tau_int = torch.where(
            rho_1 > 0.0,
            (1.0 + rho_1) / torch.clamp(1.0 - rho_1, min=1e-4),
            torch.tensor(1.0, dtype=torch.float64, device=series.device)
        )
        ess_fraction = 1.0 / tau_int
        return rho_1, tau_int, ess_fraction

    res_pure_ac = run_explain(pure_tensor_autocorr_core, series_2d)
    print(f"pure_tensor_autocorr_core: breaks={res_pure_ac['graph_break_count']}, graphs={res_pure_ac['graph_count']}")
    results["pure_autocorr_core"] = res_pure_ac

    return results


def test_mcmc_sampler_step(device: torch.device):
    print(f"\n=======================================================")
    print(f"[5] 测试 MCMC Sampler 在去相关步中的 step 计算图 (Device: {device})")
    print(f"=======================================================")
    results = {}
    system = build_h2_system(device=device, dtype=torch.float32)
    model = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=2,
        use_cusp_jastrow=True,
    ).to(device).eval()

    from ferminet_pytorch.mcmc import MCMCSampler
    sampler = MCMCSampler(
        model=model,
        system=system,
        num_walkers=64,
        step_size=0.5,
        device=device,
        dtype=torch.float32,
    )

    # MCMCSampler._step_all 核心张量转移接受计算
    # MCMCSampler.step 包含了采样随机数生成和接受率统计更新
    def sampler_step_kernel(walkers, current_log_prob, step_size):
        proposal = walkers + torch.randn_like(walkers) * step_size
        prop_log_abs, _ = model.forward_log_psi(proposal)
        prop_log_prob = 2.0 * prop_log_abs
        log_accept_ratio = prop_log_prob - current_log_prob
        accept_ratio = torch.exp(torch.clamp(log_accept_ratio, max=0.0))
        rand_uniform = torch.rand_like(accept_ratio)
        accept_mask = (rand_uniform < accept_ratio).unsqueeze(-1).unsqueeze(-1)
        new_walkers = torch.where(accept_mask, proposal, walkers)
        new_log_prob = torch.where(accept_mask.squeeze(-1).squeeze(-1), prop_log_prob, current_log_prob)
        return new_walkers, new_log_prob

    current_log_abs, _ = model.forward_log_psi(sampler.walkers)
    current_log_prob = 2.0 * current_log_abs

    res_mcmc_kernel = run_explain(sampler_step_kernel, sampler.walkers, current_log_prob, 0.5)
    print(f"MCMC Sampler 转移与接受判定核心计算图: breaks={res_mcmc_kernel['graph_break_count']}, graphs={res_mcmc_kernel['graph_count']}")
    results["mcmc_step_kernel"] = res_mcmc_kernel

    return results


def main():
    print("=" * 70)
    print("  FermiNet PyTorch 图断裂 (Graph Break) 全面深度诊断与量化测试")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    all_results = {}
    for dev in devices:
        all_results[str(dev)] = {
            "jastrow_and_forward": test_jastrow_and_forward(dev),
            "hamiltonian_and_local_energy": test_hamiltonian_and_local_energy(dev),
            "loss_and_clipping": test_loss_and_clipping(dev),
            "autocorrelation_and_decorrelation": test_autocorrelation_and_decorrelation(dev),
            "mcmc_sampler_step": test_mcmc_sampler_step(dev),
        }

    print("\n" + "=" * 70)
    print("  全模块诊断执行完毕!")
    print("=" * 70)

if __name__ == "__main__":
    main()
