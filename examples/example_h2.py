"""
FermiNet 实例演示: 氢分子 (H2 Molecule) 波函数与局部能量计算
--------------------------------------------------------------
体系描述:
- 原子核: 2 个质子 (Z=1)，核间平衡距离 R = 1.4 a.u. (约 0.74 Å)
- 电子: 2 个电子 (1 个自旋向上, 1 个自旋向下)
- 物理目标: 计算体系的对数波函数 (log|Psi|, sign) 及局部能量 E_L(r) = T_L(r) + V(r)
"""

import os

# 确保包能够被直接运行和识别
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

import torch
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy


def run_h2_demo() -> None:
    # 保证演示可复现性
    torch.manual_seed(42)

    print("=" * 70)
    print("           FermiNet (PyTorch) 氢分子 (H2) 基础模型演示")
    print("=" * 70)

    # 1. 定义氢分子 (H2) 的物理几何结构
    # 原子核坐标沿 z 轴放置，间距 1.4 a.u.
    nuclear_coords = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32
    )
    nuclear_charges = torch.tensor([1.0, 1.0], dtype=torch.float32)
    n_up = 1
    n_down = 1

    system = MolecularSystem(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=n_up,
        n_down=n_down,
    )

    print(f"[1] 分子体系设定:")
    print(f"    - 原子核数: {system.num_nuclei} (电荷 Z: {nuclear_charges.tolist()})")
    print(f"    - 核坐标 (a.u.):\n{nuclear_coords.numpy()}")
    print(
        f"    - 电子配置: N_up = {n_up}, N_down = {n_down} (总电子数 = {system.total_electrons})"
    )

    # 2. 构建 FermiNet 模型
    num_determinants = 8
    hidden_dim_one = 64
    hidden_dim_two = 16
    num_layers = 2

    model = FermiNet(
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges,
        n_up=n_up,
        n_down=n_down,
        num_determinants=num_determinants,
        hidden_dim_one=hidden_dim_one,
        hidden_dim_two=hidden_dim_two,
        num_layers=num_layers,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] FermiNet 模型已构建:")
    print(f"    - 行列式分支数 K: {num_determinants}")
    print(
        f"    - 等变交互层数: {num_layers} (单电子维度: {hidden_dim_one}, 双电子维度: {hidden_dim_two})"
    )
    print(f"    - 可学习参数总量: {num_params}")

    # 3. 构造一批初始电子坐标 (4 个 MCMC Walkers 采样点)
    # 电子围绕两个核随机分布
    batch_size = 4
    # 电子坐标 shape: (B, N, 3)
    r = torch.randn(batch_size, system.total_electrons, 3, requires_grad=True)

    print(f"\n[3] 前向计算波函数:")
    log_abs_psi, sign_psi = model.forward_log_psi(r)
    with torch.no_grad():
        psi = model(r)

    for b in range(batch_size):
        print(f"    Walker {b + 1}:")
        print(
            f"      log|Psi(r)| = {log_abs_psi[b].item():.5f}, sign = {sign_psi[b].item():.0f}"
        )
        print(f"      真实波函数 Psi(r) = {psi[b].item():.5e}")

    # 4. 计算局部能量 (Local Energy)
    print(f"\n[4] 基于自动微分计算局部能量 E_L(r) = T_L(r) + V(r):")
    e_local, t_local, v_local = compute_local_energy(model, system, r)

    for b in range(batch_size):
        print(f"    Walker {b + 1}:")
        print(f"      动能项 T_L = {t_local[b].item():.5f} Hartree")
        print(f"      势能项 V   = {v_local[b].item():.5f} Hartree")
        print(f"      总局域能 E_L = {e_local[b].item():.5f} Hartree")

    print("\n[5] 提示:")
    print("    - 真实基态氢分子能量参考值为 ~ -1.174 Hartree。")
    print(
        "    - 初态未经变分训练时局域能会存在较大离散，经过 VMC 优化后 E_L 将收敛到方差极小的常数。"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_h2_demo()
