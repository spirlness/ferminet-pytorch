from ferminet_pytorch.vmc_driver import TrainingConfig
import unittest

# 确保包能够被直接运行和识别

import torch

from ferminet_pytorch.features import ElectronFeatureEmbedding
from ferminet_pytorch.layers import FermiInteractionLayer
from ferminet_pytorch.envelope import NuclearEnvelope
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.loss import clip_local_energy, VMCLoss
from ferminet_pytorch.systems import build_h2_system
from ferminet_pytorch.vmc_driver import train_vmc


class TestFermiNet(unittest.TestCase):
    """
    FermiNet 核心模块与物理特性单元测试。
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        # 定义测试用的双原子体系 (类似 LiH 或 H2)
        # 2 个原子核, 4 个电子 (2 up, 2 down)
        self.nuclear_coords = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.nuclear_charges = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.n_up = 2
        self.n_down = 2
        self.total_electrons = self.n_up + self.n_down
        self.batch_size = 4
        self.num_determinants = 4
        self.hidden_dim_one = 32
        self.hidden_dim_two = 16
        self.num_layers = 2

        self.system = MolecularSystem(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
        )

        self.model = FermiNet(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
            num_determinants=self.num_determinants,
            hidden_dim_one=self.hidden_dim_one,
            hidden_dim_two=self.hidden_dim_two,
            num_layers=self.num_layers,
        )

    def test_feature_embedding_shapes(self) -> None:
        """测试特征嵌入层输出张量形状与数值有限性。"""
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        num_nuclei = self.nuclear_coords.shape[0]

        cartesian = ElectronFeatureEmbedding()
        h1, h2 = cartesian(r, self.nuclear_coords, self.nuclear_charges)
        self.assertEqual(
            h1.shape, (self.batch_size, self.total_electrons, 5 * num_nuclei)
        )
        self.assertEqual(
            h2.shape,
            (self.batch_size, self.total_electrons, self.total_electrons, 4),
        )
        self.assertTrue(torch.isfinite(h1).all())
        self.assertTrue(torch.isfinite(h2).all())

    def test_interaction_layer_shapes(self) -> None:
        """测试自旋等变相互作用层形状."""
        h1_in = torch.randn(self.batch_size, self.total_electrons, 8)
        h2_in = torch.randn(
            self.batch_size, self.total_electrons, self.total_electrons, 4
        )
        layer = FermiInteractionLayer(
            in_dim_one=8,
            in_dim_two=4,
            out_dim_one=self.hidden_dim_one,
            out_dim_two=self.hidden_dim_two,
            n_up=self.n_up,
            n_down=self.n_down,
        )
        h1_out, h2_out = layer(h1_in, h2_in)
        self.assertEqual(
            h1_out.shape,
            (self.batch_size, self.total_electrons, self.hidden_dim_one),
        )
        self.assertEqual(
            h2_out.shape,
            (
                self.batch_size,
                self.total_electrons,
                self.total_electrons,
                self.hidden_dim_two,
            ),
        )

    def test_envelope_shapes(self) -> None:
        """测试核包络衰减模块形状."""
        r_up = torch.randn(self.batch_size, self.n_up, 3)
        envelope = NuclearEnvelope(
            num_determinants=self.num_determinants,
            num_orbitals=self.n_up,
            num_nuclei=self.nuclear_coords.shape[0],
        )
        distances = torch.linalg.vector_norm(
            r_up[:, :, None] - self.nuclear_coords[None, None], dim=-1
        )
        env = envelope(distances)
        self.assertEqual(
            env.shape,
            (self.batch_size, self.num_determinants, self.n_up, self.n_up),
        )

    def test_model_forward_shapes(self) -> None:
        """测试 FermiNet 整体模型的前向推断形状."""
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        with torch.no_grad():
            psi = self.model(r)
            log_abs_psi, sign_psi = self.model.forward_log_psi(r)

        self.assertEqual(psi.shape, (self.batch_size,))
        self.assertEqual(log_abs_psi.shape, (self.batch_size,))
        self.assertEqual(sign_psi.shape, (self.batch_size,))
        # 验证数值一致性: Psi 与 sign * exp(log_abs_psi)
        reconstructed_psi = sign_psi * torch.exp(log_abs_psi)
        self.assertTrue(torch.allclose(psi, reconstructed_psi, rtol=1e-4, atol=1e-5))

    def test_antisymmetry_up_electrons(self) -> None:
        """
        核心物理测试: 交换任意两个自旋向上电子的坐标，波函数必须严格反号 Psi -> -Psi。
        """
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        # 交换第 0 个与第 1 个电子 (均为自旋向上电子)
        r_permuted = r.clone()
        r_permuted[:, 0, :] = r[:, 1, :]
        r_permuted[:, 1, :] = r[:, 0, :]

        with torch.no_grad():
            psi_orig = self.model(r)
            psi_perm = self.model(r_permuted)

        # 检验: psi_perm == -psi_orig
        self.assertTrue(
            torch.allclose(psi_perm, -psi_orig, rtol=1e-4, atol=1e-5),
            f"交换向上电子未反号! 原始波函数: {psi_orig}, 置换后波函数: {psi_perm}",
        )

    def test_antisymmetry_down_electrons(self) -> None:
        """
        核心物理测试: 交换任意两个自旋向下电子的坐标，波函数必须严格反号 Psi -> -Psi。
        """
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        # 交换第 2 个与第 3 个电子 (均为自旋向下电子: 索引 n_up + 0 与 n_up + 1)
        idx1 = self.n_up + 0
        idx2 = self.n_up + 1
        r_permuted = r.clone()
        r_permuted[:, idx1, :] = r[:, idx2, :]
        r_permuted[:, idx2, :] = r[:, idx1, :]

        with torch.no_grad():
            psi_orig = self.model(r)
            psi_perm = self.model(r_permuted)

        self.assertTrue(
            torch.allclose(psi_perm, -psi_orig, rtol=1e-4, atol=1e-5),
            f"交换向下电子未反号! 原始波函数: {psi_orig}, 置换后波函数: {psi_perm}",
        )

    def test_pauli_exclusion_zero(self) -> None:
        """
        泡利不相容原理: 两个同自旋电子处于同一空间位置时，波函数必须等于 0。
        """
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        # 让电子 0 与电子 1 (均为 up) 完全重合
        r[:, 1, :] = r[:, 0, :]

        with torch.no_grad():
            psi = self.model(r)

        # 验证接近 0
        self.assertTrue(
            torch.all(torch.abs(psi) < 1e-4),
            f"泡利不相容时波函数未趋于0: |Psi| = {torch.abs(psi)}",
        )

    def test_pauli_exclusion_exact_zero_and_no_nan(self) -> None:
        """
        泡利不相容原理严格数值测试: 两个同自旋电子完全重合时，波函数必须严格为 0，
        且对数波函数应精确输出 -inf 与 sign=0，绝不能产生 NaN。
        """
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        # 让电子 0 与电子 1 (均为 up) 完全重合
        r[:, 1, :] = r[:, 0, :]

        with torch.no_grad():
            log_abs_psi, sign_psi = self.model.forward_log_psi(r)
            psi = self.model(r)

        # 检验绝不产生 NaN
        self.assertFalse(torch.isnan(psi).any(), "泡利不相容构型波函数出现 NaN!")
        self.assertFalse(
            torch.isnan(log_abs_psi).any(), "泡利不相容对数波函数幅值出现 NaN!"
        )
        self.assertFalse(torch.isnan(sign_psi).any(), "泡利不相容波函数符号出现 NaN!")

        # 严格验证物理值: 真实波函数精确等于 0，符号为 0，对数幅值为 -inf
        self.assertTrue(torch.all(psi == 0.0), f"波函数未精确为 0: {psi}")
        self.assertTrue(torch.all(sign_psi == 0.0), f"符号未精确为 0: {sign_psi}")
        self.assertTrue(
            torch.all(torch.isneginf(log_abs_psi)),
            f"对数幅值未精确为 -inf: {log_abs_psi}",
        )

    def test_single_electron_spin_polarized(self) -> None:
        """
        极端边界条件测试: 单自旋极化单电子体系 (如孤立氢原子 H: N_up=1, N_down=0)。
        测试空自旋通道、单轨道行列式退化及单电子局域能计算。
        """
        nuclear_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        nuclear_charges = torch.tensor([1.0], dtype=torch.float32)
        system_h = MolecularSystem(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=1,
            n_down=0,
        )
        model_h = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=1,
            n_down=0,
            num_determinants=4,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )

        r_h = torch.randn(3, 1, 3, requires_grad=True)
        log_abs, sign = model_h.forward_log_psi(r_h)
        with torch.no_grad():
            psi_h = model_h(r_h)

        self.assertEqual(psi_h.shape, (3,))
        self.assertFalse(torch.isnan(psi_h).any(), "氢原子波函数计算出现 NaN")

        # 局域能量计算测试 (无电子-电子排斥，单核无核-核排斥)
        e_local, t_local, v_local = compute_local_energy(model_h, system_h, r_h)
        self.assertEqual(e_local.shape, (3,))
        self.assertFalse(torch.isnan(e_local).any(), "氢原子局域能出现 NaN")
        self.assertFalse(torch.isinf(e_local).any(), "氢原子局域能出现 Inf")

    def test_device_and_dtype_consistency(self) -> None:
        """
        测试体系与模型在 float64 双精度以及数据类型/设备对齐下的正确性。
        """
        system_f64 = MolecularSystem(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
        ).to(dtype=torch.float64)

        model_f64 = FermiNet(
            nuclear_coords=self.nuclear_coords.to(torch.float64),
            nuclear_charges=self.nuclear_charges.to(torch.float64),
            n_up=self.n_up,
            n_down=self.n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).to(dtype=torch.float64)

        r_f64 = torch.randn(
            2, self.total_electrons, 3, dtype=torch.float64, requires_grad=True
        )
        e_local, _, _ = compute_local_energy(model_f64, system_f64, r_f64)
        self.assertEqual(e_local.dtype, torch.float64)
        self.assertFalse(torch.isnan(e_local).any())

    def test_local_energy_computation(self) -> None:
        """
        测试局域能量计算 (局部动能拉普拉斯与库仑势能)。
        """
        r = torch.randn(2, self.total_electrons, 3, requires_grad=True)
        e_local, t_local, v_local = compute_local_energy(self.model, self.system, r)

        self.assertEqual(e_local.shape, (2,))
        self.assertEqual(t_local.shape, (2,))
        self.assertEqual(v_local.shape, (2,))
        # 检验无 NaN 和 Inf
        self.assertFalse(torch.isnan(e_local).any(), "局部总能量出现 NaN")
        self.assertFalse(torch.isinf(e_local).any(), "局部总能量出现 Inf")

    def test_compute_local_energy_create_graph(self) -> None:
        """
        测试 compute_local_energy 的 create_graph=True 选项，
        验证局域能计算图完整保留并可对模型参数求导。
        """
        r = torch.randn(2, self.total_electrons, 3, requires_grad=True)
        e_local, _, _ = compute_local_energy(
            self.model, self.system, r, create_graph=True
        )
        loss = torch.sum(e_local)
        loss.backward()

        # 检查模型关键参数梯度是否存在
        self.assertIsNotNone(self.model.det_weights.grad)
        self.assertFalse(torch.isnan(self.model.det_weights.grad).any())

    def test_v_nn_precomputed_and_migrated(self) -> None:
        """测试 MolecularSystem 初始化时预计算并缓存核-核排斥常数 V_nn，以及在 .to() 迁移时同步。"""
        system = MolecularSystem(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
        )
        self.assertTrue(hasattr(system, "v_nn"))
        self.assertIsInstance(system.v_nn, torch.Tensor)

        expected_v_nn = 1.0 * 1.0 / 2.0
        self.assertAlmostEqual(system.v_nn.item(), expected_v_nn, places=5)

        system.to(torch.float64)
        self.assertEqual(system.nuclear_coords.dtype, torch.float64)
        self.assertEqual(system.nuclear_charges.dtype, torch.float64)
        self.assertEqual(system.v_nn.dtype, torch.float64)
        self.assertAlmostEqual(system.v_nn.item(), expected_v_nn, places=5)

        r = torch.randn(self.batch_size, self.total_electrons, 3, dtype=torch.float64)
        v = system.compute_potential_energy(r)
        self.assertEqual(v.shape, (self.batch_size,))
        self.assertEqual(v.dtype, torch.float64)
        self.assertFalse(torch.isnan(v).any())

    def test_slogdet_antisymmetry_on_large_system(self) -> None:
        """
        验证阶段0防范: 在多电子体系下，采用对数波函数 slogdet 维度直接比对反对称性，
        严格验证非零符号反号与对数幅值相等，杜绝 0 == -0 假阳性。
        """
        n_atoms = 9
        coords = torch.zeros(n_atoms, 3, dtype=torch.float64)
        coords[:, 2] = torch.linspace(-4.0, 4.0, n_atoms, dtype=torch.float64)
        charges = torch.ones(n_atoms, dtype=torch.float64)
        n_up = 9
        n_down = 9

        model = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=n_up,
            n_down=n_down,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).to(dtype=torch.float64)

        batch_size = 2
        r = torch.randn(batch_size, n_up + n_down, 3, dtype=torch.float64)
        # 交换第 2 与第 7 个自旋向上电子
        r_swap = r.clone()
        r_swap[:, 2, :] = r[:, 7, :]
        r_swap[:, 7, :] = r[:, 2, :]

        with torch.no_grad():
            log_abs_orig, sign_orig = model.forward_log_psi(r)
            log_abs_swap, sign_swap = model.forward_log_psi(r_swap)

            d_up_orig, _ = model.forward_orbitals(r)
            d_up_swap, _ = model.forward_orbitals(r_swap)
            sign_det_orig, log_det_orig = torch.linalg.slogdet(d_up_orig)
            sign_det_swap, log_det_swap = torch.linalg.slogdet(d_up_swap)

        # 严格验证符号非零 (确保比对具备物理判决力，非 0 == -0)
        self.assertTrue(
            torch.all(sign_orig != 0), "波函数符号全零，无法作为有效反对称性检验判据"
        )
        self.assertTrue(torch.all(sign_det_orig != 0), "轨道行列式符号全零")

        # 验证 slogdet 对数幅值严格相等
        self.assertTrue(
            torch.allclose(log_abs_orig, log_abs_swap, rtol=1e-4, atol=1e-5),
            f"对数幅值不等: {log_abs_orig} vs {log_abs_swap}",
        )
        # 验证符号严格反号
        self.assertTrue(
            torch.all(sign_swap == -sign_orig),
            f"符号未严格反号: sign_orig={sign_orig}, sign_swap={sign_swap}",
        )
        self.assertTrue(torch.all(sign_det_swap == -sign_det_orig))
        self.assertTrue(
            torch.allclose(log_det_orig, log_det_swap, rtol=1e-4, atol=1e-5)
        )

    def test_mcmc_sampler_functionality(self) -> None:
        """验证批量 GPU/CPU MCMC 采样器的全电子与逐电子步进、自适应步长与预热."""
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=16,
            step_size=0.4,
            move_mode="all",
            adapt_step_size=True,
        )
        self.assertEqual(sampler.walkers.shape, (16, self.total_electrons, 3))
        acc_all = sampler.step()
        self.assertGreaterEqual(acc_all, 0.0)
        self.assertLessEqual(acc_all, 1.0)

        # 逐电子步进测试
        sampler.move_mode = "one_by_one"
        acc_one = sampler.step()
        self.assertGreaterEqual(acc_one, 0.0)
        self.assertLessEqual(acc_one, 1.0)

        # 抽样与克隆独立性测试
        sampled = sampler.sample(num_steps=2)
        self.assertEqual(sampled.shape, (16, self.total_electrons, 3))
        self.assertFalse(sampled.data_ptr() == sampler.walkers.data_ptr())

    def test_vmc_surrogate_loss_and_clipping(self) -> None:
        """验证基于 MAD 的稳健有偏截断与代理损失反向传播."""
        e_local = torch.tensor([-1.1, -1.2, 1000.0, -999.0], dtype=torch.float32)
        e_clip, med, mad = clip_local_energy(e_local, clip_factor=5.0, method="tanh")
        self.assertTrue(torch.isfinite(e_clip).all())
        self.assertLess(e_clip[2].item(), 100.0)
        self.assertGreater(e_clip[3].item(), -100.0)

        # 代理损失反向传播
        walkers = torch.randn(4, self.total_electrons, 3)
        vmc_loss = VMCLoss(clip_factor=5.0, clip_method="tanh")
        loss, stats, e_calc = vmc_loss.compute_step_loss(
            self.model, self.system, walkers
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("energy_mean", stats)
        self.assertIn("energy_clipped_mean", stats)

        self.model.zero_grad()
        loss.backward()
        has_grad = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_grad)

    def test_train_vmc_integration(self) -> None:
        """验证端到端 VMC 训练迭代闭环 (微型 2 步迭代)."""
        h2_sys = build_h2_system(bond_length=1.4)
        result = train_vmc(
            system=h2_sys,
            config=TrainingConfig(
                num_walkers=16,
                num_epochs=2,
                mcmc_steps_per_epoch=2,
                burn_in_steps=4,
                lr=0.001,
                verbose=False,
            ),
        )
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]["energy_raw"]), 2)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(e)) for e in result["history"]["energy_raw"]
            )
        )


if __name__ == "__main__":
    unittest.main()
