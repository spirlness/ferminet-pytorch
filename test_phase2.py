"""
阶段 2 核心特性完备单元测试 (Phase 2 Unit Tests)
================================================
测试范围:
1. 高性能动能拉普拉斯求解器 (Exact, Hutchinson 随机迹估计, JVP 向量化前向模式)
2. 物理几何特征层注入原子核电荷 Z_I (5M 维特征与 4M 兼容模式，FermiNet 维度联动)
3. 各向异性可学习三维主轴度规核包络 (NuclearEnvelope anisotropic=True)
4. 极性异核分子 LiH (Z_Li=3, Z_H=1) 端到端波函数推断、反对称性与 VMC 训练演练
5. 100 原子超大规模体系下 Hutchinson 随机迹估计器极速动能求解与有限性校验
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ferminet_pytorch.features import ElectronFeatureEmbedding
from ferminet_pytorch.envelope import NuclearEnvelope
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import (
    MolecularSystem,
    compute_local_energy,
)
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.train_vmc import build_h2_system, build_lih_system, train_vmc
from ferminet_pytorch.benchmark_100_atoms import build_h100_system


class TestPhase2Features(unittest.TestCase):
    """阶段 2 核心物理增强与高性能动能求解器测试套件."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # 1. 物理几何特征层注入原子核电荷 Z_I
    # -------------------------------------------------------------------------
    def test_feature_embedding_charges_injection(self) -> None:
        """测试特征嵌入层在传入 nuclear_charges 时的 5M 维输出与未传入时的 4M 维输出."""
        b, n, m = 2, 4, 2
        r = torch.randn(b, n, 3)
        coords = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        charges = torch.tensor([3.0, 1.0])

        embedding = ElectronFeatureEmbedding()

        # A. 兼容模式: 未提供 charges -> 4 * M
        h1_4m, h2 = embedding(r, coords)
        self.assertEqual(h1_4m.shape, (b, n, 4 * m))
        self.assertEqual(h2.shape, (b, n, n, 4))

        # B. 阶段 2 注入模式: 提供 charges -> 5 * M
        h1_5m, h2_5m = embedding(r, coords, charges)
        self.assertEqual(h1_5m.shape, (b, n, 5 * m))
        self.assertEqual(h2_5m.shape, (b, n, n, 4))
        self.assertTrue(torch.allclose(h2, h2_5m))

        # 检验第 5 个特征分量是否与 nuclear_charges 严格一致
        h1_raw = h1_5m.view(b, n, m, 5)
        self.assertTrue(torch.allclose(h1_raw[:, :, 0, 4], torch.full((b, n), 3.0)))
        self.assertTrue(torch.allclose(h1_raw[:, :, 1, 4], torch.full((b, n), 1.0)))

    def test_nuclear_charges_flexible_shapes(self) -> None:
        """测试特征嵌入层对 (M,), (M, 1), (1, M) 等多种核电荷形状的鲁棒支持."""
        b, n, m = 2, 3, 2
        r = torch.randn(b, n, 3)
        coords = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        embedding = ElectronFeatureEmbedding()

        # (M,)
        h1_1d, _ = embedding(r, coords, torch.tensor([3.0, 1.0]))
        # (M, 1)
        h1_2d, _ = embedding(r, coords, torch.tensor([[3.0], [1.0]]))
        # (1, M)
        h1_row, _ = embedding(r, coords, torch.tensor([[3.0, 1.0]]))

        self.assertEqual(h1_1d.shape, (b, n, 5 * m))
        self.assertTrue(torch.allclose(h1_1d, h1_2d))
        self.assertTrue(torch.allclose(h1_1d, h1_row))

        # 错误形状断言测试
        with self.assertRaises(AssertionError):
            embedding(r, coords, torch.tensor([3.0, 1.0, 2.0]))  # 3 个核 != 2

    def test_ferminet_use_nuclear_charges_toggle(self) -> None:
        """测试 FermiNet 在 use_nuclear_charges=True/False 时的输入层维度与前向一致性."""
        coords = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        charges = torch.tensor([3.0, 1.0])
        n_up, n_down = 2, 2

        # 默认开启注入核电荷 (5M 维)
        model_with_z = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=n_up,
            n_down=n_down,
            use_nuclear_charges=True,
            num_determinants=2,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
        )
        self.assertEqual(model_with_z.in_dim_one, 5 * 2)
        self.assertEqual(model_with_z.layers[0].in_dim_one, 5 * 2)
        self.assertEqual(model_with_z.layers[0].linear_one.in_features, 3 * (5 * 2) + 2 * 4)

        # 显式关闭注入核电荷 (4M 维)
        model_no_z = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=n_up,
            n_down=n_down,
            use_nuclear_charges=False,
            num_determinants=2,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=1,
        )
        self.assertEqual(model_no_z.in_dim_one, 4 * 2)
        self.assertEqual(model_no_z.layers[0].in_dim_one, 4 * 2)
        self.assertEqual(model_no_z.layers[0].linear_one.in_features, 3 * (4 * 2) + 2 * 4)

        r = torch.randn(3, n_up + n_down, 3)
        with torch.no_grad():
            psi_z = model_with_z(r)
            psi_no_z = model_no_z(r)
        self.assertEqual(psi_z.shape, (3,))
        self.assertEqual(psi_no_z.shape, (3,))
        self.assertTrue(torch.isfinite(psi_z).all())
        self.assertTrue(torch.isfinite(psi_no_z).all())

    # -------------------------------------------------------------------------
    # 2. 各向异性可学习度规核包络扩展
    # -------------------------------------------------------------------------
    def test_nuclear_envelope_anisotropic_shapes_and_parameters(self) -> None:
        """测试 NuclearEnvelope 各向异性模式的参数形状与前向张量尺寸."""
        k, n_orb, m = 4, 3, 2
        env_aniso = NuclearEnvelope(
            num_determinants=k,
            num_orbitals=n_orb,
            num_nuclei=m,
            anisotropic=True,
        )
        self.assertTrue(env_aniso.anisotropic)
        # 衰减参数形状必须是 (K, N_orb, M, 3)
        self.assertEqual(env_aniso.decay_rates_raw.shape, (k, n_orb, m, 3))
        self.assertEqual(env_aniso.weights.shape, (k, n_orb, m))

        # 前向测试
        b = 2
        r = torch.randn(b, n_orb, 3)
        coords = torch.randn(m, 3)
        out = env_aniso(r, coords)
        self.assertEqual(out.shape, (b, k, n_orb, n_orb))
        self.assertTrue(torch.isfinite(out).all())

    def test_nuclear_envelope_anisotropic_degeneracy(self) -> None:
        """当三维主轴衰减参数完全各向同性相等时，各向异性包络必须精确退化为各向同性包络."""
        k, n_orb, m = 2, 2, 2
        env_iso = NuclearEnvelope(num_determinants=k, num_orbitals=n_orb, num_nuclei=m, anisotropic=False)
        env_aniso = NuclearEnvelope(num_determinants=k, num_orbitals=n_orb, num_nuclei=m, anisotropic=True)

        with torch.no_grad():
            # 共享相同的组合权重
            env_aniso.weights.copy_(env_iso.weights)
            # 将各向异性三维分量统一复制自各向同性衰减参数
            iso_raw = env_iso.decay_rates_raw.unsqueeze(-1).expand(k, n_orb, m, 3)
            env_aniso.decay_rates_raw.copy_(iso_raw)

            r = torch.randn(3, n_orb, 3)
            coords = torch.randn(m, 3)
            out_iso = env_iso(r, coords)
            out_aniso = env_aniso(r, coords)

        self.assertTrue(torch.allclose(out_iso, out_aniso, atol=1e-5, rtol=1e-5))

    def test_ferminet_with_anisotropic_envelope(self) -> None:
        """测试 FermiNet 集成各向异性包络时的端到端反对称性与前向正确性."""
        coords = torch.tensor([[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]])
        charges = torch.tensor([3.0, 1.0])
        n_up, n_down = 2, 2
        model = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=n_up,
            n_down=n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            anisotropic_envelope=True,
        )
        self.assertTrue(model.anisotropic_envelope)
        self.assertTrue(model.envelope_up.anisotropic)
        self.assertTrue(model.envelope_down.anisotropic)

        r = torch.randn(2, n_up + n_down, 3)
        # 交换两个自旋向上电子 (0 与 1)
        r_swap = r.clone()
        r_swap[:, 0, :] = r[:, 1, :]
        r_swap[:, 1, :] = r[:, 0, :]

        with torch.no_grad():
            psi = model(r)
            psi_swap = model(r_swap)

        self.assertEqual(psi.shape, (2,))
        self.assertTrue(torch.allclose(psi_swap, -psi, atol=1e-4, rtol=1e-4))

    def test_anisotropic_envelope_parameter_gradients(self) -> None:
        """测试各向异性包络的三维主轴衰减度规参数与组合权重在反向传播中获得有效梯度."""
        k, n_orb, m = 2, 2, 2
        env = NuclearEnvelope(num_determinants=k, num_orbitals=n_orb, num_nuclei=m, anisotropic=True)
        r = torch.randn(2, n_orb, 3, requires_grad=True)
        coords = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

        out = env(r, coords)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(env.decay_rates_raw.grad)
        self.assertEqual(env.decay_rates_raw.grad.shape, (k, n_orb, m, 3))
        self.assertTrue(torch.isfinite(env.decay_rates_raw.grad).all())
        self.assertTrue((env.decay_rates_raw.grad != 0.0).any())

        self.assertIsNotNone(env.weights.grad)
        self.assertEqual(env.weights.grad.shape, (k, n_orb, m))
        self.assertTrue(torch.isfinite(env.weights.grad).all())

    # -------------------------------------------------------------------------
    # 3. 高性能动能拉普拉斯求解器集成
    # -------------------------------------------------------------------------
    def test_hutchinson_kinetic_solver_basic_and_shapes(self) -> None:
        """测试 Hutchinson 动能求解器的计算形状、有限性与参数支持."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        b = 4
        r = torch.randn(b, system.total_electrons, 3, requires_grad=True)

        # 1. Rademacher 分布
        e_rad, t_rad, v_rad = compute_local_energy(
            model=model,
            system=system,
            r=r,
            laplacian_method="hutchinson",
            num_hutchinson_samples=1,
            hutchinson_distribution="rademacher",
        )
        self.assertEqual(e_rad.shape, (b,))
        self.assertEqual(t_rad.shape, (b,))
        self.assertEqual(v_rad.shape, (b,))
        self.assertTrue(torch.isfinite(e_rad).all())

        # 2. Gaussian 分布
        e_gauss, t_gauss, _ = compute_local_energy(
            model=model,
            system=system,
            r=r,
            laplacian_method="hutchinson",
            num_hutchinson_samples=2,
            hutchinson_distribution="gaussian",
        )
        self.assertEqual(e_gauss.shape, (b,))
        self.assertTrue(torch.isfinite(e_gauss).all())

    def test_hutchinson_unbiasedness_convergence(self) -> None:
        """数学验证: Hutchinson 随机迹估计在多次采样平均下无偏收敛于精确拉普拉斯值."""
        torch.manual_seed(123)
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        r = torch.randn(1, system.total_electrons, 3, requires_grad=True)

        # 精确值
        _, t_exact, _ = compute_local_energy(model, system, r, laplacian_method="exact")

        # Hutchinson 多样本估计 (K = 100)
        _, t_hutch, _ = compute_local_energy(
            model,
            system,
            r,
            laplacian_method="hutchinson",
            num_hutchinson_samples=100,
            hutchinson_distribution="rademacher",
        )

        # 检验相对误差在中心极限定理合理范围内 (< 15%)
        rel_err = torch.abs(t_hutch - t_exact) / (torch.abs(t_exact) + 1e-6)
        self.assertLess(rel_err.item(), 0.15)

    def test_exact_jvp_solver(self) -> None:
        """测试基于 torch.func.jvp 的前向模式微分精确向量化求解."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        r = torch.randn(2, system.total_electrons, 3, requires_grad=True)

        e_exact, t_exact, _ = compute_local_energy(model, system, r, laplacian_method="exact")
        e_jvp, t_jvp, _ = compute_local_energy(model, system, r, laplacian_method="exact_jvp")

        self.assertEqual(e_jvp.shape, (2,))
        self.assertTrue(torch.isfinite(e_jvp).all())
        # JVP 与经典 Exact 具有机器精度数学等价性
        self.assertTrue(torch.allclose(t_exact, t_jvp, atol=1e-4, rtol=1e-4))

    def test_laplacian_method_jvp_alias(self) -> None:
        """测试 laplacian_method='jvp' 作为 'exact_jvp' 别名时产生完全相同的数值."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        r = torch.randn(2, system.total_electrons, 3, requires_grad=True)

        e_jvp1, t_jvp1, _ = compute_local_energy(model, system, r, laplacian_method="exact_jvp")
        e_jvp2, t_jvp2, _ = compute_local_energy(model, system, r, laplacian_method="jvp")

        self.assertTrue(torch.allclose(e_jvp1, e_jvp2, atol=1e-6))
        self.assertTrue(torch.allclose(t_jvp1, t_jvp2, atol=1e-6))

    def test_laplacian_method_validation(self) -> None:
        """测试非法 laplacian_method 或非法采样数抛出预期异常."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=1,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        r = torch.randn(2, 2, 3, requires_grad=True)

        with self.assertRaises(ValueError):
            compute_local_energy(model, system, r, laplacian_method="invalid_solver")

        with self.assertRaises(ValueError):
            compute_local_energy(model, system, r, laplacian_method="hutchinson", num_hutchinson_samples=0)

    def test_hutchinson_create_graph(self) -> None:
        """测试 Hutchinson 求解在 create_graph=True 时参数梯度完整保留."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        r = torch.randn(2, 2, 3, requires_grad=True)
        e_hutch, _, _ = compute_local_energy(
            model, system, r, create_graph=True, laplacian_method="hutchinson"
        )
        loss = e_hutch.sum()
        loss.backward()
        self.assertIsNotNone(model.det_weights.grad)
        self.assertTrue(torch.isfinite(model.det_weights.grad).all())

    def test_hutchinson_multi_samples_create_graph(self) -> None:
        """测试多样本 Hutchinson 采样在 create_graph=True 下的高阶图正确性与参数反向梯度."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        r = torch.randn(2, 2, 3, requires_grad=True)
        e_hutch, _, _ = compute_local_energy(
            model,
            system,
            r,
            create_graph=True,
            laplacian_method="hutchinson",
            num_hutchinson_samples=3,
            hutchinson_distribution="rademacher",
        )
        loss = e_hutch.sum()
        loss.backward()
        self.assertIsNotNone(model.det_weights.grad)
        self.assertTrue(torch.isfinite(model.det_weights.grad).all())

    # -------------------------------------------------------------------------
    # 4. 极性异核分子 LiH 端到端全流程
    # -------------------------------------------------------------------------
    def test_lih_system_and_v_nn(self) -> None:
        """测试异核分子 LiH (Z_Li=3, Z_H=1) 的物理体系构建与核库仑势能."""
        lih = build_lih_system(bond_length=3.015)
        self.assertEqual(lih.num_nuclei, 2)
        self.assertEqual(lih.total_electrons, 4)
        self.assertEqual(lih.n_up, 2)
        self.assertEqual(lih.n_down, 2)
        self.assertTrue(torch.allclose(lih.nuclear_charges, torch.tensor([3.0, 1.0])))

        # V_nn 理论值: Z_Li * Z_H / R = 3.0 * 1.0 / 3.015 = 0.995025 Hartree
        expected_v_nn = 3.0 / 3.015
        self.assertAlmostEqual(lih.v_nn.item(), expected_v_nn, places=4)

    def test_lih_antisymmetry_and_forward(self) -> None:
        """验证极性异核体系 LiH 在核电荷注入与各向异性包络下的严格反对称性."""
        lih = build_lih_system()
        model = FermiNet(
            nuclear_coords=lih.nuclear_coords,
            nuclear_charges=lih.nuclear_charges,
            n_up=lih.n_up,
            n_down=lih.n_down,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=2,
            use_nuclear_charges=True,
            anisotropic_envelope=True,
        )

        b = 2
        r = torch.randn(b, lih.total_electrons, 3)

        # 交换两个自旋向下电子 (索引 2 与 3)
        r_swap = r.clone()
        r_swap[:, 2, :] = r[:, 3, :]
        r_swap[:, 3, :] = r[:, 2, :]

        with torch.no_grad():
            psi = model(r)
            psi_swap = model(r_swap)

        self.assertTrue(torch.allclose(psi_swap, -psi, atol=1e-4, rtol=1e-4))

    def test_lih_vmc_training_loop(self) -> None:
        """验证 LiH 体系端到端 VMC 闭环优化 (微型 2 步迭代，结合 Hutchinson 求解器)."""
        lih = build_lih_system()
        model = FermiNet(
            nuclear_coords=lih.nuclear_coords,
            nuclear_charges=lih.nuclear_charges,
            n_up=lih.n_up,
            n_down=lih.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_nuclear_charges=True,
        )

        res = train_vmc(
            system=lih,
            model=model,
            num_walkers=16,
            num_epochs=2,
            mcmc_steps_per_epoch=2,
            burn_in_steps=3,
            laplacian_method="hutchinson",
            evaluation_steps=4,
            evaluation_mcmc_steps=1,
            evaluation_burn_in_steps=2,
            evaluation_laplacian_method="hutchinson",
            verbose=False,
        )
        self.assertEqual(len(res["history"]["energy_raw"]), 2)
        self.assertTrue(torch.isfinite(torch.tensor(res["final_energy"])))

    # -------------------------------------------------------------------------
    # 5. 100 原子超大规模体系下 Hutchinson 极速求解
    # -------------------------------------------------------------------------
    def test_100_atoms_hutchinson_fast_solver(self) -> None:
        """
        核心极速特性验证:
        针对 100 原子、100 电子体系 (Hessian 维度 300x300)，
        Hutchinson 随机迹估计仅需 1 次反向传播即可瞬时求出局域动能与总能量，杜绝超时与显存爆炸。
        """
        sys100 = build_h100_system(spacing=1.8)
        model100 = FermiNet(
            nuclear_coords=sys100.nuclear_coords,
            nuclear_charges=sys100.nuclear_charges,
            n_up=sys100.n_up,
            n_down=sys100.n_down,
            num_determinants=1,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            use_nuclear_charges=True,
        )
        model100.disable_compile()

        # 模拟 1 个 Walker
        r = torch.randn(1, 100, 3, requires_grad=True)

        e_hutch, t_hutch, v_hutch = compute_local_energy(
            model=model100,
            system=sys100,
            r=r,
            create_graph=False,
            laplacian_method="hutchinson",
            num_hutchinson_samples=1,
        )

        self.assertEqual(e_hutch.shape, (1,))
        self.assertEqual(t_hutch.shape, (1,))
        self.assertEqual(v_hutch.shape, (1,))
        self.assertTrue(torch.isfinite(e_hutch).all())
        self.assertTrue(torch.isfinite(t_hutch).all())
        self.assertTrue(torch.isfinite(v_hutch).all())

    def test_vmc_loss_with_hutchinson_integration(self) -> None:
        """验证 VMCLoss 模块配置 laplacian_method='hutchinson' 时的端到端损失与反向传播."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        vmc_loss = VMCLoss(clip_factor=5.0, clip_method="tanh", laplacian_method="hutchinson")

        walkers = torch.randn(4, 2, 3)
        loss, stats, e_local = vmc_loss.compute_step_loss(model, system, walkers)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(e_local.shape, (4,))
        self.assertIn("energy_mean", stats)

        model.zero_grad()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
