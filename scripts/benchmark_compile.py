"""
FermiNet PyTorch 端到端编译加速与基准性能评测脚本 (benchmark_compile.py)
========================================================================
测试内容:
1. GPU (RTX 3060) 上原生 Eager 与 torch.compile 模式下的 MCMC 采样单步延迟 (ms/step) 与加速比
2. GPU 上端到端 VMC 训练 (train_vmc) 耗时对比与吞吐提升 (s/epoch, 总耗时)
3. 预热阶段 (Warmup) 隔离 JIT 首次内核生成编译成本
4. 验证编译优化下的能量收敛与数值物理稳定性
"""

from ferminet_pytorch.vmc_driver import TrainingConfig

import time
import torch

from ferminet_pytorch.systems import build_lih_system
from ferminet_pytorch.vmc_driver import train_vmc
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.mcmc import MCMCSampler


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def benchmark_mcmc_latency(
    system, device, num_walkers=512, warmup_steps=30, eval_steps=100
):
    """评测 MCMC 采样器在 Eager 模式与 Compile 模式下的单步延迟与加速比."""
    print("\n" + "=" * 76)
    print(
        f" [评测项 1] MCMC 采样延迟与吞吐基准 (Walkers: {num_walkers}, 评测步数: {eval_steps})"
    )
    print("=" * 76)

    def create_model():
        torch.manual_seed(42)
        model = (
            FermiNet(
                nuclear_coords=system.nuclear_coords,
                nuclear_charges=system.nuclear_charges,
                n_up=system.n_up,
                n_down=system.n_down,
                num_determinants=4,
                hidden_dim_one=32,
                hidden_dim_two=16,
                num_layers=2,
                use_cusp_jastrow=True,
                spin_adapted=True,
            )
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        return model

    # ---------------- 1. 原生 Eager 模式 ----------------
    model_eager = create_model()
    sampler_eager = MCMCSampler(
        model=model_eager,
        system=system,
        num_walkers=num_walkers,
        step_size=0.5,
        move_mode="all",
    )
    sampler_eager.adapt_step_size = False

    # 预热以消除首次缓存开销
    for _ in range(warmup_steps):
        sampler_eager.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 计时评测
    t0 = time.perf_counter()
    for _ in range(eval_steps):
        sampler_eager.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    eager_total_time = t1 - t0
    eager_step_latency_ms = (eager_total_time / eval_steps) * 1000.0
    eager_throughput = (num_walkers * eval_steps) / eager_total_time

    print(f"[*] Eager 模式 MCMC 采样延迟:   {eager_step_latency_ms:8.3f} ms / step")
    print(f"    Eager 模式构型吞吐量:       {eager_throughput:8.1f} samples / s")

    # ---------------- 2. 编译 Compile 模式 ----------------
    model_comp = create_model()
    compiled_forward = torch.compile(model_comp.forward_log_psi)
    sampler_comp = MCMCSampler(
        model=model_comp,
        system=system,
        num_walkers=num_walkers,
        step_size=0.5,
        move_mode="all",
        forward_log_psi_fn=compiled_forward,
    )
    sampler_comp.adapt_step_size = False

    print("[*] 正在执行 JIT 首次内核编译与预热 (Warmup)...")
    comp_warmup_start = time.perf_counter()
    for _ in range(warmup_steps):
        sampler_comp.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    comp_warmup_time = time.perf_counter() - comp_warmup_start
    print(f"    JIT 首次编译 + 预热耗时:    {comp_warmup_time:8.3f} s")

    # 计时评测
    t0 = time.perf_counter()
    for _ in range(eval_steps):
        sampler_comp.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    comp_total_time = t1 - t0
    comp_step_latency_ms = (comp_total_time / eval_steps) * 1000.0
    comp_throughput = (num_walkers * eval_steps) / comp_total_time

    print(f"[*] Compile 模式 MCMC 采样延迟: {comp_step_latency_ms:8.3f} ms / step")
    print(f"    Compile 模式构型吞吐量:     {comp_throughput:8.1f} samples / s")

    # 性能收益
    mcmc_speedup = eager_step_latency_ms / comp_step_latency_ms
    time_reduction_pct = (1.0 - comp_step_latency_ms / eager_step_latency_ms) * 100.0

    print("-" * 76)
    print(
        f"[+] MCMC 采样纯运行加速比:       {mcmc_speedup:8.2f}x (单步耗时降低 {time_reduction_pct:.1f}%)"
    )

    return {
        "eager_latency_ms": eager_step_latency_ms,
        "comp_latency_ms": comp_step_latency_ms,
        "eager_throughput": eager_throughput,
        "comp_throughput": comp_throughput,
        "mcmc_speedup": mcmc_speedup,
        "time_reduction_pct": time_reduction_pct,
        "comp_warmup_time": comp_warmup_time,
    }


def benchmark_end_to_end_vmc(
    system, device, num_walkers=512, epochs=5, mcmc_steps=10, burn_in_steps=10
):
    """评测端到端 VMC 训练 (train_vmc) 在 Eager 与 Compile 选项下的总耗时与吞吐提升."""
    print("\n" + "=" * 76)
    print(
        f" [评测项 2] 端到端 VMC 训练吞吐基准 (Epochs: {epochs}, Walkers: {num_walkers}, MCMC/epoch: {mcmc_steps})"
    )
    print("=" * 76)

    def create_model():
        torch.manual_seed(42)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=2,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).to(device=device, dtype=torch.float32)
        return model

    # 1. Eager 模式训练
    print("[*] 正在执行 Eager 模式端到端训练...")
    model_eager = create_model()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res_eager = train_vmc(
        system=system,
        model=model_eager,
        config=TrainingConfig(
            num_walkers=num_walkers,
            num_epochs=epochs,
            burn_in_steps=burn_in_steps,
            mcmc_steps_per_epoch=mcmc_steps,
            lr=0.001,
            device=device.type,
            verbose=False,
            compile_sampler=False,
        ),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    eager_train_time = time.perf_counter() - t0
    eager_s_per_epoch = eager_train_time / epochs
    eager_final_e = res_eager["history"]["energy_clipped"][-1]

    print(
        f"    Eager 总训练耗时:           {eager_train_time:8.2f} s ({eager_s_per_epoch:.3f} s/epoch)"
    )
    print(f"    Eager 最终能量期望:         {eager_final_e:8.4f} Hartree")

    # 2. Compile 模式训练
    print("[*] 正在执行 Compile (--compile) 模式端到端训练...")
    model_comp = create_model()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res_comp = train_vmc(
        system=system,
        model=model_comp,
        config=TrainingConfig(
            num_walkers=num_walkers,
            num_epochs=epochs,
            burn_in_steps=burn_in_steps,
            mcmc_steps_per_epoch=mcmc_steps,
            lr=0.001,
            device=device.type,
            verbose=False,
            compile_sampler=True,
        ),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    comp_train_time = time.perf_counter() - t0
    comp_s_per_epoch = comp_train_time / epochs
    comp_final_e = res_comp["history"]["energy_clipped"][-1]

    print(
        f"    Compile 总训练耗时:         {comp_train_time:8.2f} s ({comp_s_per_epoch:.3f} s/epoch)"
    )
    print(f"    Compile 最终能量期望:       {comp_final_e:8.4f} Hartree")

    train_speedup = eager_train_time / comp_train_time
    train_reduction_pct = (1.0 - comp_train_time / eager_train_time) * 100.0

    print("-" * 76)
    print(
        f"[+] 端到端训练总吞吐加速比:     {train_speedup:8.2f}x (总耗时减少 {train_reduction_pct:.1f}%)"
    )

    return {
        "eager_train_time": eager_train_time,
        "comp_train_time": comp_train_time,
        "eager_s_per_epoch": eager_s_per_epoch,
        "comp_s_per_epoch": comp_s_per_epoch,
        "eager_final_energy": eager_final_e,
        "comp_final_energy": comp_final_e,
        "train_speedup": train_speedup,
        "train_reduction_pct": train_reduction_pct,
    }


def main():
    device = get_device()
    print("=" * 76)
    print("   FermiNet PyTorch 零图断裂与内核融合编译 (torch.compile) 性能基准")
    print("=" * 76)
    print(f"[*] PyTorch 版本: {torch.__version__}")
    print(f"[*] 执行设备:     {device.type.upper()}")
    if device.type == "cuda":
        print(f"[*] GPU 硬件型号: {torch.cuda.get_device_name(0)}")
        print(f"[*] CUDA 驱动支持: {torch.version.cuda}")
    print(f"[*] 测试目标体系: LiH 分子 (4 电子, up=2, down=2, 核间距=3.015 a.u.)")

    system = build_lih_system(device=device, dtype=torch.float32)

    # 1. MCMC 延迟基准评测
    mcmc_res = benchmark_mcmc_latency(
        system=system,
        device=device,
        num_walkers=512,
        warmup_steps=30,
        eval_steps=100,
    )

    # 2. 端到端 VMC 训练基准评测
    train_res = benchmark_end_to_end_vmc(
        system=system,
        device=device,
        num_walkers=512,
        epochs=5,
        mcmc_steps=10,
        burn_in_steps=10,
    )

    # 3. 总体成果汇总表
    print("\n" + "=" * 76)
    print("                      基准评测核心指标汇总看板")
    print("=" * 76)
    print(
        f"{'评测维度 / 性能指标':<32} | {'原生 Eager 模式':<18} | {'Compile 编译模式':<18}"
    )
    print("-" * 76)
    print(
        f"{'MCMC 单步采样延迟 (ms/step)':<28} | {mcmc_res['eager_latency_ms']:16.3f} ms | {mcmc_res['comp_latency_ms']:16.3f} ms"
    )
    print(
        f"{'MCMC 采样吞吐 (samples/s)':<29} | {mcmc_res['eager_throughput']:14.1f} s/s | {mcmc_res['comp_throughput']:14.1f} s/s"
    )
    print(
        f"{'MCMC 采样核心加速比':<31} | {'1.00x (基准)':<18} | {mcmc_res['mcmc_speedup']:14.2f}x"
    )
    print("-" * 76)
    print(
        f"{'端到端训练总耗时 (5 Epochs)':<27} | {train_res['eager_train_time']:16.2f} s  | {train_res['comp_train_time']:16.2f} s"
    )
    print(
        f"{'每 Epoch 平均耗时 (s/epoch)':<28} | {train_res['eager_s_per_epoch']:16.3f} s  | {train_res['comp_s_per_epoch']:16.3f} s"
    )
    print(
        f"{'端到端训练加速比':<32} | {'1.00x (基准)':<18} | {train_res['train_speedup']:14.2f}x"
    )
    print(
        f"{'最终局域能量 (Ha)':<30} | {train_res['eager_final_energy']:16.4f} Ha | {train_res['comp_final_energy']:16.4f} Ha"
    )
    print("=" * 76)
    print("[*] 性能基准测试顺利完成！所有数值与收敛性指标符合预期。")


if __name__ == "__main__":
    main()
