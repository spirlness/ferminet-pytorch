"""
FermiNet PyTorch Compile 编译优化综合性能评测
==============================================
对比分析:
1. Eager 模式 (原生未编译)
2. Compiled 模式 (torch.compile 默认 inductor 后端)
3. Compiled 模式 (mode="reduce-overhead"，启用 CUDA Graphs 优化内核调度开销)

评测指标:
- 首次 JIT 编译耗时 (Compilation Latency)
- 稳定推断延迟 (Latency per step in ms)
- 批次吞吐量 (Throughput in Walkers / sec)
- 编译前后数值一致性 (Allclose Numerical Verification)
- 峰值显存对比 (Peak Memory Usage)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import torch
import torch.nn as nn

# 确保标准 sys.path 路径导入
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in [str(current_dir), str(parent_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ferminet_pytorch.model import FermiNet
    from ferminet_pytorch.hamiltonian import MolecularSystem
except ImportError:
    from model import FermiNet
    from hamiltonian import MolecularSystem


def setup_benchmark_system(num_atoms: int = 10, spacing: float = 1.8, device: Optional[str] = None):
    """构建用于编译评测的分子体系 (以 H10 链为例，10 核 + 10 电子)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    z_coords = torch.linspace(
        -(num_atoms - 1) * spacing / 2.0,
        (num_atoms - 1) * spacing / 2.0,
        num_atoms,
        dtype=torch.float32,
    )
    nuclear_coords = torch.zeros(num_atoms, 3, dtype=torch.float32)
    nuclear_coords[:, 2] = z_coords
    nuclear_charges = torch.ones(num_atoms, dtype=torch.float32)

    n_up = num_atoms // 2
    n_down = num_atoms - n_up

    system = MolecularSystem(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=n_up,
        n_down=n_down,
    ).to(device=device)

    model = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        rotation_invariant=True,
        num_determinants=8,
        hidden_dim_one=64,
        hidden_dim_two=16,
        num_layers=2,
    ).to(device)

    return system, model


def measure_inference(
    model_fn,
    r: torch.Tensor,
    warmup_steps: int = 5,
    benchmark_steps: int = 50,
    device_type: str = "cuda",
) -> Tuple[float, float, torch.Tensor]:
    """测量前向推断平均耗时 (ms) 与吞吐量 (walkers/s)."""
    batch_size = r.shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            out = model_fn(r)
            if device_type == "cuda":
                torch.cuda.synchronize()

    # Benchmark
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()

    with torch.no_grad():
        for _ in range(benchmark_steps):
            out = model_fn(r)

    if device_type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
    else:
        total_time_ms = (time.time() - t0) * 1000.0

    avg_latency_ms = total_time_ms / benchmark_steps
    throughput = (batch_size * benchmark_steps) / (total_time_ms / 1000.0)

    return avg_latency_ms, throughput, out


def run_compile_benchmark() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("           FermiNet PyTorch Compile 编译优化评测")
    print(f"           设备: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    # 1. 准备测试体系与模型 (以 H10 分子链为例: 10 核 10 电子，5 up + 5 down)
    system, model = setup_benchmark_system(num_atoms=10, spacing=1.8, device=device)
    model.eval()

    # MCMC 典型中等批次: 64 个 Walkers，每个 10 个电子
    batch_size = 64
    r = torch.randn(batch_size, system.total_electrons, 3, device=device)

    print(f"\n[1] 评测体系规模:")
    print(f"    - 原子数目: {system.num_nuclei} (10 个质子)")
    print(f"    - 电子数目: {system.total_electrons} (5 up + 5 down)")
    print(f"    - 采样 Walkers 批次 (Batch Size): {batch_size}")
    print(f"    - 行列式分支数 K: {model.num_determinants}")

    # 2. 评测 Eager 模式 (原生 PyTorch)
    print(f"\n[2] 测试 Eager 模式 (原生未编译)...")
    model.disable_compile()  # 显式关闭编译，确保纯净 Eager 运行，消除测试干扰
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    eager_lat, eager_thr, eager_out = measure_inference(
        model, r, warmup_steps=10, benchmark_steps=50, device_type=device
    )
    eager_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    print(f"    - Eager 平均延迟: {eager_lat:.3f} ms")
    print(f"    - Eager 吞吐量:   {eager_thr:.1f} Walkers/s")
    print(f"    - Eager 峰值显存: {eager_mem:.2f} MB")

    # 3. 评测 Compiled 模式 (模式: default)
    print(f"\n[3] 编译并测试 Compiled 模式 (backend='inductor', mode='default')...")
    comp_lat, comp_thr, comp_out, speedup_default = None, None, None, None
    try:
        t0_comp = time.time()
        compiled_model_default = torch.compile(model, backend="inductor", mode="default")
        # 触发首次 JIT 编译
        with torch.no_grad():
            _ = compiled_model_default(r)
            if device == "cuda":
                torch.cuda.synchronize()
        comp_time_default = time.time() - t0_comp
        print(f"    - 首次 JIT 编译耗时: {comp_time_default:.2f} s")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        comp_lat, comp_thr, comp_out = measure_inference(
            compiled_model_default, r, warmup_steps=5, benchmark_steps=50, device_type=device
        )
        comp_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
        speedup_default = eager_lat / comp_lat

        print(f"    - Compiled (default) 平均延迟: {comp_lat:.3f} ms")
        print(f"    - Compiled (default) 吞吐量:   {comp_thr:.1f} Walkers/s")
        print(f"    - Compiled (default) 峰值显存: {comp_mem:.2f} MB")
        print(f"    - 加速比 (Speedup):           {speedup_default:.2f}x")
    except Exception as e:
        print(f"    - Compiled (default) 模式未支持或跳过 (可能缺少 MSVC/C++ 编译器或 CUDA 环境): {e}")

    # 4. 评测 Compiled 模式 (mode="reduce-overhead", 包含 CUDA Graphs)
    print(f"\n[4] 编译并测试 Compiled 模式 (mode='reduce-overhead', 启用 CUDA Graphs)...")
    ro_lat, ro_thr, ro_out, speedup_ro = None, None, None, None
    try:
        t0_comp_ro = time.time()
        compiled_model_ro = torch.compile(model, backend="inductor", mode="reduce-overhead")
        with torch.no_grad():
            _ = compiled_model_ro(r)
            if device == "cuda":
                torch.cuda.synchronize()
        comp_time_ro = time.time() - t0_comp_ro
        print(f"    - 首次 JIT 编译耗时: {comp_time_ro:.2f} s")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ro_lat, ro_thr, ro_out = measure_inference(
            compiled_model_ro, r, warmup_steps=5, benchmark_steps=50, device_type=device
        )
        ro_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
        speedup_ro = eager_lat / ro_lat

        print(f"    - Compiled (reduce-overhead) 平均延迟: {ro_lat:.3f} ms")
        print(f"    - Compiled (reduce-overhead) 吞吐量:   {ro_thr:.1f} Walkers/s")
        print(f"    - Compiled (reduce-overhead) 峰值显存: {ro_mem:.2f} MB")
        print(f"    - 加速比 (Speedup):                   {speedup_ro:.2f}x")
    except Exception as e:
        print(f"    - reduce-overhead 模式未支持或跳过: {e}")

    # 5. 数值一致性严格验证 (Allclose Verification)
    print(f"\n[5] 数值一致性比对 (Eager vs Compiled):")
    if comp_out is not None:
        max_diff_default = torch.max(torch.abs(eager_out - comp_out)).item()
        rel_diff_default = torch.max(torch.abs(eager_out - comp_out) / (torch.abs(eager_out) + 1e-12)).item()
        is_close_default = torch.allclose(eager_out, comp_out, rtol=1e-4, atol=1e-5)

        print(f"    - Default 模式最大绝对误差: {max_diff_default:.2e}")
        print(f"    - Default 模式最大相对误差: {rel_diff_default:.2e}")
        print(f"    - 状态: {'[PASS] 严格数值一致' if is_close_default else '[FAIL]'}")
    else:
        print("    - Default 模式编译未执行，跳过数值一致性比对")

    if ro_out is not None:
        max_diff_ro = torch.max(torch.abs(eager_out - ro_out)).item()
        is_close_ro = torch.allclose(eager_out, ro_out, rtol=1e-4, atol=1e-5)
        print(f"    - Reduce-Overhead 模式最大绝对误差: {max_diff_ro:.2e}")
        print(f"    - 状态: {'[PASS] 严格数值一致' if is_close_ro else '[FAIL]'}")

    print("\n" + "=" * 80)
    print("                         编译优化评测总结")
    print("=" * 80)
    print(f"{'运行模式':<25} | {'平均延迟 (ms)':<15} | {'吞吐量 (Walkers/s)':<20} | {'相对加速比':<10}")
    print("-" * 80)
    print(f"{'Eager (原生)':<25} | {eager_lat:<15.3f} | {eager_thr:<20.1f} | {'1.00x':<10}")
    if comp_lat is not None:
        print(f"{'Compiled (default)':<25} | {comp_lat:<15.3f} | {comp_thr:<20.1f} | {f'{speedup_default:.2f}x':<10}")
    if ro_lat is not None:
        print(f"{'Compiled (reduce-overhead)':<25} | {ro_lat:<15.3f} | {ro_thr:<20.1f} | {f'{speedup_ro:.2f}x':<10}")
    print("=" * 80)


if __name__ == "__main__":
    run_compile_benchmark()
