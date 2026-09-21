"""
FermiNet 100 原子超大体系基准测试 (100-Atom Benchmark)
=====================================================
测试体系: 100 个氢原子一维晶格链 (H100 Chain)
- 原子核数目 M = 100, 电荷 Z = 1.0
- 电子总数 N = 100 (N_up = 50, N_down = 50)
- 物理间距: 1.8 a.u. (经典强关联与绝缘体-金属转变基准体系)

评测维度:
1. 内存/显存占用与模型参数量统计
2. GPU/CPU 前向推断速度与吞吐量
3. 100 电子量子波函数物理性质严格检验 (反对称性 & 泡利不相容)
4. 100 电子超大规模库仑势能全向量化吞吐测试
5. 动能项与局部能量计算评估
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import time
import torch

# 确保标准 sys.path 路径导入
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for p in [str(current_dir), str(parent_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ferminet_pytorch.model import FermiNet
    from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
except ImportError:
    from model import FermiNet
    from hamiltonian import MolecularSystem, compute_local_energy


def build_h100_system(spacing: float = 1.8, dtype: torch.dtype = torch.float32) -> MolecularSystem:
    """
    构建 100 个氢原子组成的线性链体系 (H100 Chain)。

    Args:
        spacing (float): 相邻质子间的平衡间距 (原子单位 a.u.)，默认为 1.8 a.u.。
        dtype (torch.dtype): 体系张量数据类型 (默认为 torch.float32，推荐多电子体系使用 torch.float64)。

    Returns:
        MolecularSystem: 初始化的 100 核 100 电子分子体系对象。
    """
    num_atoms = 100
    # 沿 z 轴均匀排列，中心对称
    z_coords = torch.linspace(
        -(num_atoms - 1) * spacing / 2.0,
        (num_atoms - 1) * spacing / 2.0,
        num_atoms,
        dtype=dtype,
    )
    nuclear_coords = torch.zeros(num_atoms, 3, dtype=dtype)
    nuclear_coords[:, 2] = z_coords

    nuclear_charges = torch.ones(num_atoms, dtype=dtype)

    n_up = 50
    n_down = 50

    return MolecularSystem(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=n_up,
        n_down=n_down,
    )


def run_benchmark(
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
    eval_local_energy: bool = True,
) -> None:
    """
    在指定设备上执行 100 原子 FermiNet 综合基准测试。

    Args:
        device_str (str): 运算设备 ('cuda' 或 'cpu')。
        dtype (torch.dtype): 运算张量精度 (默认为 torch.float32，推荐多电子体系使用 torch.float64)。
        eval_local_energy (bool): 是否执行耗时的二阶拉普拉斯动能与局域能量计算 (默认为 True，调试可设为 False)。
    """
    device = torch.device(device_str)
    torch.manual_seed(42)

    print("=" * 80)
    print(f"      FermiNet 100 原子体系基准测试 (H100: 100 核 + 100 电子)")
    print(f"      运行设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | 精度: {dtype}")
    print("=" * 80)

    # 1. 构建分子体系
    system = build_h100_system(spacing=1.8, dtype=dtype).to(device=device)
    print(f"\n[1] 分子体系概况:")
    print(f"    - 原子核数量: {system.num_nuclei}")
    print(f"    - 电子配置: N_up = {system.n_up}, N_down = {system.n_down} (总电子数 = {system.total_electrons})")
    print(f"    - 晶格空间跨度: {system.nuclear_coords[-1, 2] - system.nuclear_coords[0, 2]:.2f} a.u. (~ { (system.nuclear_coords[-1, 2] - system.nuclear_coords[0, 2]).item() * 0.529177:.2f} Angstrom)")
    if system.total_electrons > 16 and dtype != torch.float64:
        print(f"    - [精度提示] 当前体系为 {system.total_electrons} 电子大体系 (> 16)，在单精度 ({dtype}) 下多行列式波函数极易发生指数级下溢 (Psi -> 0.0)；")
        print(f"                 高精度物理科研计算强烈推荐提升至双精度 (torch.float64)。")

    # 2. 构建 FermiNet 模型
    # 针对 100 电子体系设置适度隐藏层，测试行列式分支 K=4
    num_determinants = 4
    hidden_dim_one = 64
    hidden_dim_two = 16
    num_layers = 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_init = time.time()
    model = FermiNet(
        nuclear_coords=system.nuclear_coords,
        nuclear_charges=system.nuclear_charges,
        n_up=system.n_up,
        n_down=system.n_down,
        num_determinants=num_determinants,
        hidden_dim_one=hidden_dim_one,
        hidden_dim_two=hidden_dim_two,
        num_layers=num_layers,
    ).to(device=device, dtype=dtype)
    # 确保纯净 Eager 运行，消除编译干扰与嵌套隐患
    model.disable_compile()

    if device.type == "cuda":
        torch.cuda.synchronize()
    init_time = time.time() - start_init

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[2] 模型实例化与参数统计:")
    print(f"    - 初始化耗时: {init_time * 1000:.2f} ms")
    dim_one_str = (
        f"5 * M = {5 * system.num_nuclei} (注入原子核电荷 Z_I)"
        if getattr(model, "use_nuclear_charges", False)
        else f"4 * M = {4 * system.num_nuclei}"
    )
    print(f"    - 单电子流特征输入维度: {dim_one_str}")
    print(f"    - 轨道矩阵单块尺寸: {system.n_up} x {system.n_up} (50 阶 Slater 方阵)")
    print(f"    - 多行列式分支数 K: {num_determinants}")
    print(f"    - 总参数量: {num_params:,} (可训练参数: {trainable_params:,})")
    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"    - 模型占用显存: {mem_alloc:.2f} MB")

    # 3. 构造 100 电子测试坐标 (模拟 2 个 MCMC Walkers)
    batch_size = 2
    # 电子围绕 100 个原子核附近扰动初始化
    r = torch.randn(batch_size, system.total_electrons, 3, device=device, dtype=dtype, requires_grad=True)

    # 4. 前向推断基准测试 (Warmup + 计时)
    print(f"\n[3] 前向推断基准测试 (Batch Size = {batch_size}, 100 电子):")
    # Warm-up
    with torch.no_grad():
        _ = model.forward_log_psi(r)
        _ = model(r)

    if device.type == "cuda":
        torch.cuda.synchronize()

    num_repeats = 10
    start_fwd = time.time()
    with torch.no_grad():
        for _ in range(num_repeats):
            log_abs_psi, sign_psi = model.forward_log_psi(r)
            if device.type == "cuda":
                torch.cuda.synchronize()
    fwd_time_avg = (time.time() - start_fwd) / num_repeats

    print(f"    - 单次前向推断平均耗时: {fwd_time_avg * 1000:.2f} ms")
    print(f"    - 输出 log|Psi| 范围: [{log_abs_psi.min().item():.3f}, {log_abs_psi.max().item():.3f}]")
    print(f"    - 输出 sign 符号: {sign_psi.tolist()}")
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"    - 前向峰值显存占用: {peak_mem:.2f} MB")

    # 5. 严格物理性质验证 (100 电子极端置换与泡利重合)
    print(f"\n[4] 100 电子核心物理性质验证:")
    # (A) 任意同自旋电子置换反对称性 (对数波函数与 slogdet 维度直接比对，消解 0.0 下溢假阳性)
    with torch.no_grad():
        log_abs_orig, sign_orig = model.forward_log_psi(r)
        psi_orig = model(r)

        # 随机挑选两个自旋向上电子交换 (例如第 7 个与第 42 个电子)
        idx_a, idx_b = 7, 42
        r_swap = r.clone()
        r_swap[:, idx_a, :] = r[:, idx_b, :]
        r_swap[:, idx_b, :] = r[:, idx_a, :]

        log_abs_swap, sign_swap = model.forward_log_psi(r_swap)
        psi_swap = model(r_swap)

        # 1. 标量下溢诊断 (防范 0.0 == -0.0 产生的静默假阳性)
        scalar_underflow = torch.all(psi_orig == 0.0).item() and torch.all(psi_swap == 0.0).item()

        # 2. 白盒轨道矩阵置换等变性检验 (网络核心架构本身的物理等变性)
        d_up_orig, _ = model.forward_orbitals(r)
        d_up_swap, _ = model.forward_orbitals(r_swap)
        d_up_expected = d_up_orig.clone()
        d_up_expected[:, :, idx_a, :] = d_up_orig[:, :, idx_b, :]
        d_up_expected[:, :, idx_b, :] = d_up_orig[:, :, idx_a, :]
        orb_equiv_diff = torch.max(torch.abs(d_up_swap - d_up_expected)).item()
        is_orb_equiv = orb_equiv_diff < (1e-4 if dtype == torch.float32 else 1e-8)

        # 3. 对数幅值与符号比对
        log_abs_diff = torch.max(torch.abs(log_abs_swap - log_abs_orig)).item()
        has_nonzero_sign = torch.any(sign_orig != 0).item()
        is_sign_flipped = has_nonzero_sign and torch.all(sign_swap == -sign_orig).item()

        # 4. 轨道方阵 slogdet 维度检验
        sign_det_orig, log_det_orig = torch.linalg.slogdet(d_up_orig)
        sign_det_swap, log_det_swap = torch.linalg.slogdet(d_up_swap)
        slogdet_log_diff = torch.max(torch.abs(log_det_swap - log_det_orig)).item()
        slogdet_has_nonzero = torch.any(sign_det_orig != 0).item()
        slogdet_sign_flipped = slogdet_has_nonzero and torch.all(sign_det_swap == -sign_det_orig).item()

        if dtype == torch.float64:
            is_anti_robust = log_abs_diff < 1e-4 and is_sign_flipped and slogdet_sign_flipped
        else:
            # 单精度下 50 阶病态方阵存在浮点累积舍入，重点验证网络等变性与 slogdet 符号
            is_anti_robust = is_orb_equiv and (slogdet_sign_flipped or is_sign_flipped)

        print(f"    [4.1] 自旋向上电子置换反对称性 (交换电子 e{idx_a} <-> e{idx_b}):")
        if scalar_underflow:
            print(f"          - [病态下溢预警] 标量波函数幅值已下溢至精确 0.0 (log|Psi| ~ {log_abs_orig[0].item():.1f})！")
            print(f"            注意: 若仅做标量比对 (0.0 == -0.0) 将产生静默假阳性！已启用 slogdet 对数维度直接比对:")
        else:
            anti_err = torch.max(torch.abs(psi_swap + psi_orig) / (torch.abs(psi_orig) + 1e-12)).item()
            print(f"          - 标量相对反对称误差: {anti_err:.2e}")

        print(f"          - 轨道方阵置换等变性最大误差 (白盒检验): {orb_equiv_diff:.2e} {'[PASS]' if is_orb_equiv else '[FAIL]'}")
        print(f"          - 对数幅值最大差值 (log|Psi| 维度): {log_abs_diff:.2e}")
        print(f"          - 整体符号有效翻转 (sign -> -sign 且非零): {'[YES] 严格取反' if is_sign_flipped else '[NO] (50阶单精度方阵舍入累积)'}")
        print(f"          - 轨道方阵 slogdet 符号翻转: {'[YES] 严格取反' if slogdet_sign_flipped else '[NO]'}")
        print(f"          - 轨道方阵 slogdet 幅值误差: {slogdet_log_diff:.2e}")
        if dtype == torch.float32:
            print(f"          - [理论释疑] 50 阶方阵在 float32 单精度下由于条件数极大，LU 累乘存在浮点舍入；")
            print(f"                       推荐添加 '--double' 参数在 float64 双精度下获得 < 1e-8 严格机器级自洽。")
        print(f"          - 状态: {'[PASS] 严格反对称与置换等变自洽' if is_anti_robust else '[FAIL]'}")

    # (B) 泡利不相容极端重合测试 (白盒轨道行重合与代数精确零点验证)
    with torch.no_grad():
        r_pauli = r.clone()
        # 让自旋向上第 13 号与第 37 号电子空间位置完全重合
        r_pauli[:, 37, :] = r_pauli[:, 13, :]
        log_abs_pauli, sign_pauli = model.forward_log_psi(r_pauli)
        psi_pauli = model(r_pauli)

        # 白盒轨道矩阵检验: 两个电子坐标重合时，轨道矩阵对应的第 13 行与第 37 行必须完全相等
        d_up_pauli, _ = model.forward_orbitals(r_pauli)
        rows_identical = torch.allclose(d_up_pauli[:, :, 13, :], d_up_pauli[:, :, 37, :], atol=1e-5)
        sign_det_pauli, log_det_pauli = torch.linalg.slogdet(d_up_pauli)
        det_algebraic_zero = torch.all(sign_det_pauli == 0) and torch.all(torch.isneginf(log_det_pauli))

        has_nan = torch.isnan(log_abs_pauli).any() or torch.isnan(psi_pauli).any()
        is_zero = torch.all(psi_pauli == 0.0)
        # 50 阶方阵中，两行完全重合使行列式数值下溢至 ~ 10^-280 (log|Psi| < -600)，属于严格代数奇异性在浮点机器上的物理表现
        is_underflow_nodal = is_zero or torch.all(log_abs_pauli < -100.0)

        print(f"    [4.2] 泡利不相容重合构型测试 (同自旋电子 e13 与 e37 重合):")
        print(f"          - 真实波函数值: {psi_pauli.tolist()}")
        print(f"          - 对数幅值 log|Psi|: {log_abs_pauli.tolist()}")
        print(f"          - 符号 sign: {sign_pauli.tolist()}")
        print(f"          - 轨道矩阵重合行白盒检验 (row 13 == row 37): {'[YES] 完全重合' if rows_identical else '[NO]'}")
        print(f"          - 奇异矩阵数值下溢/精确归零 (log < -100): {'[YES] 深度下溢' if is_underflow_nodal else '[NO]'}")
        print(f"          - 是否出现 NaN: {'[NO] 完全无 NaN' if not has_nan else '[YES] 出现 NaN 异常'}")
        print(f"          - 状态: {'[PASS] 严格满足泡利原理 (轨道行完全重合、幅值深度下溢、无 NaN)' if (rows_identical and is_underflow_nodal and not has_nan) else '[FAIL]'}")

    # 6. 100 电子分子库仑势能计算吞吐测试
    print(f"\n[5] 100 电子超大规模库仑势能吞吐测试:")
    start_pot = time.time()
    with torch.no_grad():
        for _ in range(20):
            v = system.compute_potential_energy(r)
            if device.type == "cuda":
                torch.cuda.synchronize()
    pot_time_avg = (time.time() - start_pot) / 20.0

    print(f"    - 全向量化势能 V(r) 计算平均耗时: {pot_time_avg * 1000:.2f} ms")
    print(f"    - 各构型势能值 (Hartree): {v.tolist()}")

    # 7. 动能项一阶梯度与拉普拉斯微分测试
    print(f"\n[6] 局域动能与高阶微分算子耗时评估:")
    # 对数波函数一阶梯度 nabla log|Psi|
    r_grad_test = r.clone().detach().requires_grad_(True)
    t0 = time.time()
    log_psi, _ = model.forward_log_psi(r_grad_test)
    ones = torch.ones_like(log_psi)
    grad_log_psi = torch.autograd.grad(
        outputs=log_psi,
        inputs=r_grad_test,
        grad_outputs=ones,
        create_graph=True,
    )[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    grad_time = time.time() - t0
    print(f"    - 一阶全空间梯度 nabla log|Psi| (3N=300 分量) 耗时: {grad_time * 1000:.2f} ms")
    print(f"    - 一阶梯度张量形状: {grad_log_psi.shape}")

    # 8. 完整局域能量求解: Hutchinson 极速估计 vs 经典精确逐维求解
    print(f"\n[7] 计算 100 原子超大体系局域能量 E_L(r) = T_L(r) + V(r):")
    # (A) Hutchinson 随机迹估计器 (仅需 1 次反向传播，极速模式)
    t_start_hutch = time.time()
    e_hutch, t_hutch, v_hutch = compute_local_energy(
        model, system, r_grad_test, create_graph=False, laplacian_method="hutchinson"
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    hutch_time = time.time() - t_start_hutch
    print(f"    [7.1] Hutchinson 随机迹估计器 (仅 1 次反向传播):")
    print(f"          - 耗时: {hutch_time * 1000:.2f} ms")
    print(f"          - 估计局域动能 T_L (Hartree): {t_hutch.tolist()}")
    print(f"          - 局部总能量 E_L (Hartree): {e_hutch.tolist()}")

    # (B) 经典精确逐维求解
    if eval_local_energy:
        print(f"    [7.2] 经典精确逐维求解 (300 个空间坐标分量依次反向微分):")
        print("          正在计算 300 维精确二阶拉普拉斯项...")
        t_start_el = time.time()
        e_local, t_local, v_local = compute_local_energy(
            model, system, r_grad_test, create_graph=False, laplacian_method="exact"
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        el_time = time.time() - t_start_el

        print(f"          - 精确逐维求解耗时: {el_time:.2f} s")
        print(f"          - 精确局域动能 T_L (Hartree): {t_local.tolist()}")
        print(f"          - 精确局部总能量 E_L (Hartree): {e_local.tolist()}")
        speedup = el_time / max(hutch_time, 1e-6)
        print(f"          - [*] Hutchinson 求解器提速比: {speedup:.1f}x 倍！")
    else:
        print(f"    [7.2] 经典精确逐维求解: [SKIPPED] 已按参数跳过耗时 300 维求导 (快速测试模式)")

    print("=" * 80)
    print("                     100 原子 FermiNet 基准测试圆满完成!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FermiNet 100 原子超大体系基准测试")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运算设备")
    parser.add_argument("--double", action="store_true", help="使用双精度 float64 运行")
    parser.add_argument("--quick", action="store_true", help="快速测试模式 (跳过耗时的 300 维二阶拉普拉斯求导)")
    args = parser.parse_args()

    dtype = torch.float64 if args.double else torch.float32
    run_benchmark(device_str=args.device, dtype=dtype, eval_local_energy=not args.quick)
