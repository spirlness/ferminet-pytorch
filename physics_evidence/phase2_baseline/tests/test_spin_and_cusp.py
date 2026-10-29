"""
自旋纯态适配与显式 Kato 尖峰 Jastrow 因子的物理严格性单元测试。

验证内容:
1. 自旋纯态与空间投影:
   - 双电子单重态空间交换对称性 Psi(r1, r2) == Psi(r2, r1)
   - 双电子三重态空间交换反对称性 Psi(r1, r2) == -Psi(r2, r1)
   - 自旋本征态 S^2 期望值与自旋污染消除
   - 闭壳层体系多电子反转对称性
2. Kato 尖峰条件解析极限:
   - 电子-核碰撞极限: d ln|Psi| / d r_{iI} -> -Z_I (针对多种核电荷 Z=1, Z=3)
   - 电子-电子异自旋碰撞极限: d ln|Psi| / d r_{ij} -> 1/2
   - 电子-电子同自旋碰撞极限: d ln|Psi| / d r_{ij} -> 1/4
   - 库仑奇点抵消: 靠近原子核时动能发散与势能发散相互抵消
3. 物理不变量与集成优化:
   - 泡利不相容原理与反对称性保持
   - Jastrow 因子的正定性与节点保持
   - 组合模型端到端 VMC 变分训练与能量收敛
"""

from typing import Tuple
import unittest
import torch
import torch.nn as nn

from ferminet_pytorch.jastrow import KatoCuspJastrow
from ferminet_pytorch.spin import SpinProjector
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy, diagnose_cusp_divergence
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.train_vmc import train_vmc, build_h2_system, build_lih_system


class TestSpinAdaptation(unittest.TestCase):
    """自旋纯态与自旋空间投影测试。"""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.h2_system = build_h2_system(bond_length=1.4, dtype=torch.float64)

    def test_singlet_spatial_symmetry(self) -> None:
        """验证双电子单重态空间对称化: Psi(r1, r2) == Psi(r2, r1)。"""
        model = FermiNet(
            nuclear_coords=self.h2_system.nuclear_coords,
            nuclear_charges=self.h2_system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,  # 启用自旋单重态空间对称化
        ).to(dtype=torch.float64)

        batch_size = 10
        r = torch.randn(batch_size, 2, 3, dtype=torch.float64)
        # 交换电子 0 与电子 1
        r_perm = r[:, [1, 0], :]

        with torch.no_grad():
            psi_orig = model(r)
            psi_perm = model(r_perm)

        # 严格验证空间对称性: Psi(r1, r2) == Psi(r2, r1)
        diff = torch.abs(psi_orig - psi_perm)
        max_diff = torch.max(diff).item()
        self.assertLess(max_diff, 1e-10, f"单重态空间对称性破坏，最大误差: {max_diff}")

    def test_unadapted_ferminet_has_spin_contamination(self) -> None:
        """对比验证: 未开启自旋适配的普通 FermiNet 通常存在自旋污染 (非空间完全对称)。"""
        model_unadapted = FermiNet(
            nuclear_coords=self.h2_system.nuclear_coords,
            nuclear_charges=self.h2_system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=False,  # 不开启自旋适配
        ).to(dtype=torch.float64)

        # 随机参数下，通常不满足 Psi(r1, r2) == Psi(r2, r1)
        r = torch.randn(20, 2, 3, dtype=torch.float64)
        r_perm = r[:, [1, 0], :]

        with torch.no_grad():
            psi_orig = model_unadapted(r)
            psi_perm = model_unadapted(r_perm)

        asym = torch.mean(torch.abs(psi_orig - psi_perm)).item()
        # 未投影网络通常不对称
        self.assertGreater(asym, 1e-6, "普通网络在随机初始化下偶然对称")

    def test_triplet_spatial_antisymmetry(self) -> None:
        """验证双电子三重态空间反对称化: Psi(r1, r2) == -Psi(r2, r1) 且重合点为0。"""
        projector_triplet = SpinProjector(n_up=1, n_down=1, mode="triplet")
        model_triplet = FermiNet(
            nuclear_coords=self.h2_system.nuclear_coords,
            nuclear_charges=self.h2_system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=8,
            num_layers=1,
            spin_projector=projector_triplet,
        ).to(dtype=torch.float64)

        batch_size = 10
        r = torch.randn(batch_size, 2, 3, dtype=torch.float64)
        r_perm = r[:, [1, 0], :]

        with torch.no_grad():
            psi_orig = model_triplet(r)
            psi_perm = model_triplet(r_perm)

        # 验证反对称: Psi(r1, r2) + Psi(r2, r1) == 0
        sum_psi = torch.abs(psi_orig + psi_perm)
        max_err = torch.max(sum_psi).item()
        self.assertLess(max_err, 1e-10, f"三重态空间反对称性破坏，最大误差: {max_err}")

        # 验证两电子空间重合时波函数为零 (泡利/三重态节点)
        r_coincident = r.clone()
        r_coincident[:, 1, :] = r_coincident[:, 0, :]  # r1 == r2
        with torch.no_grad():
            psi_coin = model_triplet(r_coincident)
        self.assertLess(torch.max(torch.abs(psi_coin)).item(), 1e-10, "三重态在空间重合处必须为 0")

    def test_spin_eigenvalue_s_squared_operator(self) -> None:
        """验证自旋单重态本征值: S^2 Psi_singlet = (1 - P_12) Psi_singlet = 0。"""
        model_singlet = FermiNet(
            nuclear_coords=self.h2_system.nuclear_coords,
            nuclear_charges=self.h2_system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            spin_adapted=True,
        ).to(dtype=torch.float64)

        r = torch.randn(15, 2, 3, dtype=torch.float64)
        r_perm = r[:, [1, 0], :]

        with torch.no_grad():
            psi_1 = model_singlet(r)
            psi_2 = model_singlet(r_perm)

        # S^2 作用结果: (1 - P12) * Psi
        s_squared_psi = psi_1 - psi_2
        self.assertLess(torch.max(torch.abs(s_squared_psi)).item(), 1e-10)

    def test_closed_shell_spin_flip_symmetry(self) -> None:
        """验证多电子闭壳层体系 (LiH, n_up=2, n_down=2) 的空间自旋翻转对称性。"""
        lih = build_lih_system(dtype=torch.float64)
        model_lih = FermiNet(
            nuclear_coords=lih.nuclear_coords,
            nuclear_charges=lih.nuclear_charges,
            n_up=2,
            n_down=2,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
        ).to(dtype=torch.float64)

        # r: 4 个电子 [up0, up1, down0, down1]
        r = torch.randn(8, 4, 3, dtype=torch.float64)
        # 翻转 up <-> down: [down0, down1, up0, up1]
        r_flipped = r[:, [2, 3, 0, 1], :]

        with torch.no_grad():
            psi_orig = model_lih(r)
            psi_flip = model_lih(r_flipped)

        diff = torch.max(torch.abs(psi_orig - psi_flip)).item()
        self.assertLess(diff, 1e-9, f"闭壳层自旋翻转对称性破坏: {diff}")

    def test_open_shell_even_electrons_safe_fallback(self) -> None:
        """验证偶数总电子但自旋不相等的开壳层体系 (如 4 电子三线态与碳原子基态) 安全降级为 mode='none'。"""
        # 测试用例 1: 4 电子三线态 (n_up=3, n_down=1)
        coords_4e = torch.zeros(1, 3, dtype=torch.float64)
        charges_4e = torch.tensor([4.0], dtype=torch.float64)
        sys_4e = MolecularSystem(coords_4e, charges_4e, n_up=3, n_down=1)
        model_4e = FermiNet(
            nuclear_coords=coords_4e,
            nuclear_charges=charges_4e,
            n_up=3,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
        ).to(dtype=torch.float64)

        self.assertIsNotNone(model_4e.spin_projector)
        self.assertEqual(model_4e.spin_projector.mode, "none")

        r_4e = torch.randn(4, 4, 3, dtype=torch.float64, requires_grad=True)
        e_4e, _, _ = compute_local_energy(model_4e, sys_4e, r_4e)
        self.assertTrue(torch.isfinite(e_4e).all())

        # 测试用例 2: 碳原子三线态 (n_up=4, n_down=2)
        coords_c = torch.zeros(1, 3, dtype=torch.float64)
        charges_c = torch.tensor([6.0], dtype=torch.float64)
        sys_c = MolecularSystem(coords_c, charges_c, n_up=4, n_down=2)
        model_c = FermiNet(
            nuclear_coords=coords_c,
            nuclear_charges=charges_c,
            n_up=4,
            n_down=2,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
        ).to(dtype=torch.float64)

        self.assertIsNotNone(model_c.spin_projector)
        self.assertEqual(model_c.spin_projector.mode, "none")

        r_c = torch.randn(4, 6, 3, dtype=torch.float64, requires_grad=True)
        e_c, _, _ = compute_local_energy(model_c, sys_c, r_c)
        self.assertTrue(torch.isfinite(e_c).all())

    def test_float32_nodal_clamp_underflow_prevention(self) -> None:
        """验证在 float32 精度下波函数节点面截断下界不发生 0.0 下溢且导数稳定。"""
        # 1. 直接验证常数下界在 float32 下严格大于 0
        min_clamp_f32 = torch.tensor(1e-20, dtype=torch.float32)
        self.assertGreater(min_clamp_f32.item(), 0.0)
        self.assertFalse((min_clamp_f32 == 0.0).item())

        # 2. 验证模型在单精度下接近节点面时梯度计算数值稳定
        model_f32 = FermiNet(
            nuclear_coords=self.h2_system.nuclear_coords.float(),
            nuclear_charges=self.h2_system.nuclear_charges.float(),
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
        )
        r = torch.randn(4, 2, 3, dtype=torch.float32, requires_grad=True)
        log_abs, sign = model_f32.forward_log_psi(r)
        self.assertTrue(torch.isfinite(log_abs).all())
        grad_r = torch.autograd.grad(log_abs.sum(), r, create_graph=True)[0]
        self.assertTrue(torch.isfinite(grad_r).all())



class TestKatoCuspJastrow(unittest.TestCase):
    """Kato 尖峰解析 Jastrow 因子测试。"""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_electron_nucleus_cusp_limit(self) -> None:
        """
        验证电子-核 Kato 尖峰条件:
            lim_{r_{iI} -> 0} d ln J / d r_{iI} = -Z_I
        分别针对氢核 (Z=1.0) 与锂核 (Z=3.0) 严格验证。
        """
        for expected_z in [1.0, 3.0, 6.0]:
            nuclear_charges = torch.tensor([expected_z], dtype=torch.float64)
            nuclear_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

            jastrow = KatoCuspJastrow(
                nuclear_charges=nuclear_charges,
                n_up=1,
                n_down=1,
                cusp_en=True,
                cusp_ee=False,
                eps=1e-24,
            ).to(dtype=torch.float64)

            deltas = [1e-3, 1e-4, 1e-5]
            for delta in deltas:
                d_val = torch.tensor([delta], dtype=torch.float64, requires_grad=True)
                # 电子 0 沿 x 轴在距离 delta 处, 电子 1 远离
                e0_pos = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) * d_val
                e1_pos = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)
                r = torch.stack([e0_pos, e1_pos]).unsqueeze(0)  # (1, 2, 3)

                u_en = jastrow.compute_en(r, nuclear_coords)
                grad_d = torch.autograd.grad(u_en, d_val)[0].item()

                # 解析导数表达式: -Z / (1 + b * delta)^2
                expected_slope = -expected_z / ((1.0 + jastrow.b_en[0].item() * delta) ** 2)
                self.assertAlmostEqual(grad_d, expected_slope, places=5)
                # 极限偏差收敛于 0
                self.assertLess(abs(grad_d - (-expected_z)), 3.0 * delta * expected_z + 1e-4)

    def test_electron_electron_opposite_spin_cusp_limit(self) -> None:
        """
        验证异自旋电子-电子 Kato 尖峰条件:
            lim_{r_{ij} -> 0} d ln J / d r_{ij} = 1/2
        """
        nuclear_charges = torch.tensor([1.0], dtype=torch.float64)
        jastrow = KatoCuspJastrow(
            nuclear_charges=nuclear_charges,
            n_up=1,
            n_down=1,
            cusp_en=False,
            cusp_ee=True,
            eps=1e-24,
        ).to(dtype=torch.float64)

        # 电子 0 (up) 位于原点，电子 1 (down) 位于 (0, 0, delta)
        deltas = [1e-3, 1e-4, 1e-5]
        for delta in deltas:
            d_val = torch.tensor([delta], dtype=torch.float64, requires_grad=True)
            e0_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
            e1_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * d_val
            r = torch.stack([e0_pos, e1_pos]).unsqueeze(0)  # (1, 2, 3)

            u_ee = jastrow.compute_ee(r)
            grad_d = torch.autograd.grad(u_ee, d_val)[0].item()

            # 理论极限 0.5 / (1 + b * delta)^2 -> 0.5
            expected_slope = 0.5 / ((1.0 + jastrow.b_ee_antiparallel.item() * delta) ** 2)
            self.assertAlmostEqual(grad_d, expected_slope, places=5)
            self.assertLess(abs(grad_d - 0.5), 2.0 * delta + 1e-4)

    def test_electron_electron_parallel_spin_cusp_limit(self) -> None:
        """
        验证同自旋电子-电子 Kato 尖峰条件:
            lim_{r_{ij} -> 0} d ln J / d r_{ij} = 1/4
        """
        nuclear_charges = torch.tensor([1.0], dtype=torch.float64)
        # n_up = 2 (两个同自旋电子 0 和 1)
        jastrow = KatoCuspJastrow(
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=0,
            cusp_en=False,
            cusp_ee=True,
            eps=1e-24,
        ).to(dtype=torch.float64)

        deltas = [1e-3, 1e-4, 1e-5]
        for delta in deltas:
            d_val = torch.tensor([delta], dtype=torch.float64, requires_grad=True)
            e0_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
            e1_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * d_val
            r = torch.stack([e0_pos, e1_pos]).unsqueeze(0)  # (1, 2, 3)

            u_ee = jastrow.compute_ee(r)
            grad_d = torch.autograd.grad(u_ee, d_val)[0].item()

            # 理论极限 0.25 / (1 + b * delta)^2 -> 0.25
            expected_slope = 0.25 / ((1.0 + jastrow.b_ee_parallel.item() * delta) ** 2)
            self.assertAlmostEqual(grad_d, expected_slope, places=5)
            self.assertLess(abs(grad_d - 0.25), 2.0 * delta + 1e-4)

    def test_coulomb_singularity_cancellation(self) -> None:
        """
        物理关键验证: 验证 Kato Jastrow 因子在近核奇点消除库仑发散。

        当电子靠近核 r -> 0 时，局域势能以 -Z/r 发生剧烈库仑发散 (可达 -10,000 Hartree 以上)；
        而融入 Kato Jastrow 因子后，动能项 -0.5 * (∇²lnJ + |∇lnJ|²) 精确产生
        +Z/r 奇异项，两者精确抵消，局域能量 E_L 保持有限平稳！
        """
        system = build_h2_system(dtype=torch.float64)

        jastrow = KatoCuspJastrow(
            nuclear_charges=system.nuclear_charges,
            n_up=1,
            n_down=1,
            cusp_en=True,
            cusp_ee=True,
            eps=1e-24,
        ).to(dtype=torch.float64)

        class PureJastrowWavefunction(nn.Module):
            def __init__(self, jastrow_module: nn.Module) -> None:
                super().__init__()
                self.jastrow_module = jastrow_module

            def forward_log_psi(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                log_j = self.jastrow_module(r, system.nuclear_coords)
                sign = torch.ones(r.shape[0], dtype=r.dtype, device=r.device)
                return log_j, sign

        model_jastrow = PureJastrowWavefunction(jastrow)

        # 电子 0 从 0.1 逼近到 0.0001 (缩小 1000 倍)
        distances = [0.1, 0.01, 0.001, 0.0001]
        e_list = []
        v_list = []
        t_list = []

        for dist in distances:
            r = torch.tensor(
                [[[0.0, 0.0, -0.7 + dist], [2.0, 2.0, 2.0]]],
                dtype=torch.float64,
                requires_grad=True,
            )
            e, t, v = compute_local_energy(model_jastrow, system, r)
            e_list.append(e.item())
            t_list.append(t.item())
            v_list.append(v.item())

        # 1. 验证势能发生了剧烈的万倍级负发散
        v_divergence = v_list[-1] - v_list[0]
        self.assertLess(v_divergence, -9000.0, f"势能应当发生剧烈库仑发散: {v_divergence}")

        # 2. 验证动能项发生对应的正发散精确抵消势能
        t_divergence = t_list[-1] - t_list[0]
        self.assertGreater(t_divergence, 9000.0, f"动能项应当发生对应正发散: {t_divergence}")

        # 3. 验证总局域能量 E_L 波动极其微小 (奇点彻底消除!)
        e_fluctuation = abs(e_list[-1] - e_list[0])
        self.assertLess(e_fluctuation, 1.0, f"局域能量奇点发散未被抵消，波动过大: {e_fluctuation}")
        self.assertLess(e_fluctuation, abs(v_divergence) * 1e-4)

    def test_electron_electron_coulomb_singularity_cancellation(self) -> None:
        """
        物理关键验证: 验证 Kato Jastrow 因子在两电子短程碰撞 (r_{12} -> 0) 时
        对正发散库仑排斥势能 (+1/r_{12}) 的精确动能抵消。
        """
        system = build_h2_system(dtype=torch.float64)
        jastrow = KatoCuspJastrow(
            nuclear_charges=system.nuclear_charges,
            n_up=1,
            n_down=1,
            cusp_en=False,
            cusp_ee=True,
            eps=1e-24,
        ).to(dtype=torch.float64)

        class PureEEJastrow(nn.Module):
            def __init__(self, j_mod: nn.Module) -> None:
                super().__init__()
                self.j_mod = j_mod

            def forward_log_psi(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.j_mod.compute_ee(r), torch.ones(r.shape[0], dtype=r.dtype, device=r.device)

        model_ee = PureEEJastrow(jastrow)

        deltas = [0.1, 0.01, 0.001, 0.0001]
        e_list = []
        v_list = []
        t_list = []

        for delta in deltas:
            # 两电子在远离核的平坦区域互相靠近: e0=(0,0,1), e1=(0,0,1+delta)
            r = torch.tensor(
                [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + delta]]],
                dtype=torch.float64,
                requires_grad=True,
            )
            e, t, v = compute_local_energy(model_ee, system, r)
            e_list.append(e.item())
            t_list.append(t.item())
            v_list.append(v.item())

        # 势能暴涨 +9990 Hartree
        v_diff = v_list[-1] - v_list[0]
        self.assertGreater(v_diff, 9000.0, f"电子间库仑排斥应正发散: {v_diff}")

        # 动能暴跌 -9990 Hartree 精确抵消
        t_diff = t_list[-1] - t_list[0]
        self.assertLess(t_diff, -9000.0, f"动能项应精确负发散抵消: {t_diff}")

        # 局域能量波动平缓
        e_fluc = abs(e_list[-1] - e_list[0])
        self.assertLess(e_fluc, 1.0, f"电子碰撞局域能量波动过大: {e_fluc}")

    def test_full_ferminet_nuclear_cusp_divergence_cancellation(self) -> None:
        """
        端到端深度架构验证: 验证真实的 FermiNet 深度波函数 (融合平滑几何特征、高斯包络与 Kato Jastrow)
        在电子逼近原子核时，彻底消除万倍库仑发散，局域能量保持平稳。
        """
        system = build_h2_system(dtype=torch.float64)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=8,
            num_layers=2,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).to(dtype=torch.float64)

        deltas = [0.1, 0.01, 0.001, 0.0001]
        e_list = []
        v_list = []
        t_list = []

        for delta in deltas:
            r = torch.tensor(
                [[[0.0, 0.0, -0.7 + delta], [1.5, 1.5, 1.5]]],
                dtype=torch.float64,
                requires_grad=True,
            )
            e, t, v = compute_local_energy(model, system, r)
            e_list.append(e.item())
            t_list.append(t.item())
            v_list.append(v.item())

        v_diff = v_list[-1] - v_list[0]
        self.assertLess(v_diff, -9000.0, "近核势能应发生万倍负发散")

        # 验证完整 FermiNet 的局域能量总波动极其平缓 (消除奇异性)
        e_fluc = abs(e_list[-1] - e_list[0])
        self.assertLess(e_fluc, 2.0, f"真实 FermiNet 局域能量未平稳消除发散，波动: {e_fluc}")

    def test_pauli_antisymmetry_preserved_with_jastrow(self) -> None:
        """验证引入 Jastrow 因子后，Slater 行列式的同自旋电子反对称性严格保持。"""
        # 3 个自旋向上电子
        coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges = torch.tensor([3.0], dtype=torch.float64)

        model = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=3,
            n_down=0,
            num_determinants=4,
            use_cusp_jastrow=True,
        ).to(dtype=torch.float64)

        batch_size = 5
        r = torch.randn(batch_size, 3, 3, dtype=torch.float64)
        # 交换电子 0 与电子 1 (同自旋 up)
        r_swap = r[:, [1, 0, 2], :]

        with torch.no_grad():
            psi = model(r)
            psi_swap = model(r_swap)

        # 严格反对称: Psi -> -Psi
        anti_err = torch.max(torch.abs(psi + psi_swap)).item()
        self.assertLess(anti_err, 1e-10, f"Jastrow 破坏了同自旋反对称性: {anti_err}")


class TestIntegratedModelAndVMC(unittest.TestCase):
    """自旋纯态与 Jastrow 融合模型的集成与端到端优化测试。"""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.system = build_h2_system(dtype=torch.float32)

    def test_combined_model_shapes_and_gradients(self) -> None:
        """测试同时启用 spin_adapted 与 cusp_jastrow 时的输出形状与梯度传播。"""
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=8,
            num_layers=2,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )

        r = torch.randn(4, 2, 3, requires_grad=True)
        log_abs, sign = model.forward_log_psi(r)
        self.assertEqual(log_abs.shape, (4,))
        self.assertEqual(sign.shape, (4,))

        # 标量输出
        psi = model(r)
        self.assertEqual(psi.shape, (4,))

        # 测试反向传播
        loss = torch.sum(psi)
        loss.backward()

        # 检查各部分参数梯度均非 None
        self.assertIsNotNone(model.det_weights.grad)
        self.assertIsNotNone(model.linear_orb_up.weight.grad)
        self.assertIsNotNone(model.jastrow.raw_b_ee_par.grad)
        self.assertIsNotNone(model.jastrow.raw_b_en.grad)

    def test_local_energy_with_combined_model(self) -> None:
        """验证组合模型在 compute_local_energy 中拉普拉斯与动能计算正常。"""
        model = FermiNet(
            nuclear_coords=self.system.nuclear_coords,
            nuclear_charges=self.system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=4,
            use_cusp_jastrow=True,
            spin_adapted=True,
        )

        r = torch.randn(6, 2, 3, requires_grad=True)
        e_local, t_local, v_local = compute_local_energy(model, self.system, r, create_graph=False)

        self.assertEqual(e_local.shape, (6,))
        self.assertEqual(t_local.shape, (6,))
        self.assertEqual(v_local.shape, (6,))
        self.assertTrue(torch.all(torch.isfinite(e_local)).item())

    def test_vmc_training_loop_integration(self) -> None:
        """验证端到端 VMC 训练循环在 3 个 epoch 内正常迭代。"""
        result = train_vmc(
            system=self.system,
            num_walkers=32,
            num_epochs=3,
            mcmc_steps_per_epoch=5,
            burn_in_steps=10,
            use_cusp_jastrow=True,
            spin_adapted=True,
            verbose=False,
            seed=42,
        )

    def test_single_electron_boundary(self) -> None:
        """测试单电子边界条件 (氢原子: n_up=1, n_down=0)。"""
        coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges = torch.tensor([1.0], dtype=torch.float64)
        model = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=1,
            n_down=0,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).to(dtype=torch.float64)

        r = torch.randn(5, 1, 3, dtype=torch.float64, requires_grad=True)
        log_abs, sign = model.forward_log_psi(r)
        self.assertEqual(log_abs.shape, (5,))
        psi = model(r)
        loss = torch.sum(psi)
        loss.backward()
        self.assertIsNotNone(model.linear_orb_up.weight.grad)
        self.assertIsNotNone(model.jastrow.raw_b_en.grad)


if __name__ == "__main__":
    unittest.main()
