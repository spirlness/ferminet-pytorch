import os
import sys
import unittest
import warnings

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ferminet_pytorch.features import ElectronFeatureEmbedding
from ferminet_pytorch.layers import FermiInteractionLayer
from ferminet_pytorch.envelope import NuclearEnvelope
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss, VMCLoss
from ferminet_pytorch.train_vmc import build_h2_system, train_vmc


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
        """测试特征嵌入层输出张量形状 (两种特征模式)."""
        r = torch.randn(self.batch_size, self.total_electrons, 3)
        num_nuclei = self.nuclear_coords.shape[0]

        # 显式纯距离模式: 单电子流每核 1 维 [d_{iI}]，双电子流 1 维 [d_{ij}]
        invariant = ElectronFeatureEmbedding(rotation_invariant=True)
        h1, h2 = invariant(r, self.nuclear_coords)
        self.assertEqual(
            h1.shape, (self.batch_size, self.total_electrons, 1 * num_nuclei)
        )
        self.assertEqual(
            h2.shape,
            (self.batch_size, self.total_electrons, self.total_electrons, 1),
        )

        # 公共默认的笛卡尔模式: 每核 4 维 [r_{iI}, d_{iI}]，双电子流 4 维
        cartesian = ElectronFeatureEmbedding()
        h1_legacy, h2_legacy = cartesian(r, self.nuclear_coords)
        self.assertEqual(
            h1_legacy.shape, (self.batch_size, self.total_electrons, 4 * num_nuclei)
        )
        self.assertEqual(
            h2_legacy.shape,
            (self.batch_size, self.total_electrons, self.total_electrons, 4),
        )

    def test_interaction_layer_shapes(self) -> None:
        """测试自旋等变相互作用层形状."""
        h1_in = torch.randn(self.batch_size, self.total_electrons, 8)
        h2_in = torch.randn(self.batch_size, self.total_electrons, self.total_electrons, 4)
        layer = FermiInteractionLayer(
            in_dim_one=8,
            in_dim_two=4,
            out_dim_one=self.hidden_dim_one,
            out_dim_two=self.hidden_dim_two,
        )
        h1_out, h2_out = layer(h1_in, h2_in, self.n_up, self.n_down)
        self.assertEqual(
            h1_out.shape,
            (self.batch_size, self.total_electrons, self.hidden_dim_one),
        )
        self.assertEqual(
            h2_out.shape,
            (self.batch_size, self.total_electrons, self.total_electrons, self.hidden_dim_two),
        )

    def test_envelope_shapes(self) -> None:
        """测试核包络衰减模块形状."""
        r_up = torch.randn(self.batch_size, self.n_up, 3)
        envelope = NuclearEnvelope(
            num_determinants=self.num_determinants,
            num_orbitals=self.n_up,
            num_nuclei=self.nuclear_coords.shape[0],
        )
        env = envelope(r_up, self.nuclear_coords)
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
        self.assertFalse(torch.isnan(log_abs_psi).any(), "泡利不相容对数波函数幅值出现 NaN!")
        self.assertFalse(torch.isnan(sign_psi).any(), "泡利不相容波函数符号出现 NaN!")

        # 严格验证物理值: 真实波函数精确等于 0，符号为 0，对数幅值为 -inf
        self.assertTrue(torch.all(psi == 0.0), f"波函数未精确为 0: {psi}")
        self.assertTrue(torch.all(sign_psi == 0.0), f"符号未精确为 0: {sign_psi}")
        self.assertTrue(torch.all(torch.isneginf(log_abs_psi)), f"对数幅值未精确为 -inf: {log_abs_psi}")

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

        r_f64 = torch.randn(2, self.total_electrons, 3, dtype=torch.float64, requires_grad=True)
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

    def test_default_compile_flags(self) -> None:
        """
        验证阶段0纠偏: FermiNet 默认 compile 为 False，compile_mode 默认为 'default'，
        并支持显式链式 enable/disable_compile。

        本测试只检查标志位，不触发真正的编译后端，因此在任何环境下都应通过。
        """
        model = FermiNet(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
        )
        self.assertFalse(model.use_compile, "FermiNet 默认应禁用编译 (compile=False)")
        self.assertEqual(model.compile_mode, "default", "默认 compile_mode 应为 'default'")

        # 测试显式按需启用与关闭
        model.enable_compile(mode="reduce-overhead")
        self.assertTrue(model.use_compile)
        self.assertEqual(model.compile_mode, "reduce-overhead")

        model.disable_compile()
        self.assertFalse(model.use_compile)

    def test_compile_model_disables_internal_compile(self) -> None:
        """
        验证 `compile_model` 会先关闭内部懒加载编译以消除多重嵌套编译。

        `torch.compile` 需要可用的编译后端。若当前环境无法导入后端——例如 Windows
        的 GBK 默认编码下 torch inductor 读取模板文件时未指定编码，抛
        UnicodeDecodeError——则跳过本项：那是环境/区域设置问题，不是本仓库的缺陷
        (审查报告 D12)。标志位层面由 test_default_compile_flags 覆盖。
        """
        model = FermiNet(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
        )
        model.enable_compile()
        try:
            model.compile_model()  # 默认 mode 应为 default
        except (UnicodeDecodeError, ImportError) as exc:
            self.skipTest(
                f"torch.compile 在当前环境不可用 ({type(exc).__name__}: {exc})，"
                f"跳过编译路径检查。"
            )
        self.assertFalse(
            model.use_compile, "compile_model 应显式关闭内部编译以消除多重嵌套编译"
        )

    def test_v_nn_precomputed_and_migrated(self) -> None:
        """
        验证阶段0优化: MolecularSystem 初始化时预计算并缓存核-核排斥常数 V_nn，
        在 .to(device, dtype) 各种位置与关键字传参时同步迁移缓存，
        并在 update_geometry 几何构型变更时自动刷新缓存。
        """
        system = MolecularSystem(
            nuclear_coords=self.nuclear_coords,
            nuclear_charges=self.nuclear_charges,
            n_up=self.n_up,
            n_down=self.n_down,
        )
        # 验证缓存属性存在
        self.assertTrue(hasattr(system, "v_nn"))
        self.assertIsInstance(system.v_nn, torch.Tensor)

        # 理论计算 2 个核的库仑斥力: Z1 * Z2 / |R1 - R2|
        # R1 = (0, 0, -1), R2 = (0, 0, 1), dist = 2.0, Z1=1, Z2=1 => V_nn = 0.5 Hartree
        expected_v_nn = 1.0 * 1.0 / 2.0
        self.assertAlmostEqual(system.v_nn.item(), expected_v_nn, places=5)

        # 验证 .to(torch.float64) 位置参数传参 (防范因具名位置参数签名导致的 API 崩溃)
        system.to(torch.float64)
        self.assertEqual(system.nuclear_coords.dtype, torch.float64)
        self.assertEqual(system.nuclear_charges.dtype, torch.float64)
        self.assertEqual(system.v_nn.dtype, torch.float64)
        self.assertAlmostEqual(system.v_nn.item(), expected_v_nn, places=5)

        # 验证 .to('cpu') 位置参数传参
        system.to("cpu")
        self.assertEqual(system.nuclear_coords.device.type, "cpu")
        self.assertEqual(system.v_nn.device.type, "cpu")

        # 验证 update_geometry 几何构型变更时自动刷新 V_nn 缓存 (防范 stale cache)
        new_coords = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], dtype=torch.float64)
        system.update_geometry(new_coords)
        expected_new_v_nn = 1.0 * 1.0 / 1.0  # dist = 1.0 => V_nn = 1.0
        self.assertAlmostEqual(system.v_nn.item(), expected_new_v_nn, places=5)

        # 验证势能计算复用缓存正确性
        r = torch.randn(self.batch_size, self.total_electrons, 3, dtype=torch.float64)
        v = system.compute_potential_energy(r)
        self.assertEqual(v.shape, (self.batch_size,))
        self.assertEqual(v.dtype, torch.float64)
        self.assertFalse(torch.isnan(v).any())

    def test_precision_warning_for_large_systems(self) -> None:
        """
        验证阶段0防范: 多电子体系 (N > 16) 在单精度下自动发出精度提升建议告警。
        """
        n_atoms = 10
        coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
        charges = torch.ones(n_atoms, dtype=torch.float32)

        # 总电子数 18 (> 16)
        n_up = 9
        n_down = 9
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model_large_f32 = FermiNet(
                nuclear_coords=coords,
                nuclear_charges=charges,
                n_up=n_up,
                n_down=n_down,
                hidden_dim_one=16,
                hidden_dim_two=8,
                num_layers=1,
            )
            # 应该捕获到 UserWarning
            warning_msgs = [str(item.message) for item in w if issubclass(item.category, UserWarning)]
            self.assertTrue(
                any("总电子数 N = 18 > 16" in msg or "建议提升至双精度" in msg for msg in warning_msgs),
                "未检测到针对 N > 16 单精度大体系的精度提升告警！",
            )

        # 双精度下不应发出警告
        coords_f64 = coords.to(torch.float64)
        charges_f64 = charges.to(torch.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = FermiNet(
                nuclear_coords=coords_f64,
                nuclear_charges=charges_f64,
                n_up=n_up,
                n_down=n_down,
                hidden_dim_one=16,
                hidden_dim_two=8,
                num_layers=1,
            )
            warning_msgs = [str(item.message) for item in w if issubclass(item.category, UserWarning)]
            self.assertEqual(len(warning_msgs), 0, "双精度下不应发出下溢告警")

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
        self.assertTrue(torch.all(sign_orig != 0), "波函数符号全零，无法作为有效反对称性检验判据")
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
        self.assertTrue(torch.allclose(log_det_orig, log_det_swap, rtol=1e-4, atol=1e-5))

    def test_large_float32_underflow_keeps_finite_log_wavefunction(self) -> None:
        """R8: scalar underflow must not turn a nonzero state into a node."""
        # 针对 50 电子大体系，单精度下乘积波函数幅值约为 exp(-200) 发生真实物理浮点下溢
        coords = torch.zeros(50, 3, dtype=torch.float32)
        charges = torch.ones(50, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = FermiNet(
                nuclear_coords=coords,
                nuclear_charges=charges,
                n_up=25,
                n_down=25,
                num_determinants=1,
                hidden_dim_one=16,
                hidden_dim_two=8,
                num_layers=1,
            )

        r = torch.randn(2, 50, 3, dtype=torch.float32)
        log_abs, sign = model.forward_log_psi(r)
        self.assertTrue(torch.all(torch.isfinite(log_abs)))
        self.assertTrue(torch.all(sign != 0.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            psi = model(r)
        self.assertTrue(torch.all(psi == 0.0))
        self.assertTrue(any("下溢" in str(item.message) for item in caught))

    def test_benchmark_compile_helpers(self) -> None:
        """
        验证 benchmark_compile 模块中的辅助函数及 Eager 模式执行路径。
        """
        from ferminet_pytorch.benchmark_compile import setup_benchmark_system, measure_inference

        # 默认不传 device 测试自适应能力
        system, model = setup_benchmark_system(num_atoms=4, spacing=1.8)
        self.assertEqual(system.num_nuclei, 4)
        self.assertEqual(system.total_electrons, 4)
        self.assertFalse(model.use_compile)

        model.disable_compile()
        r = torch.randn(2, 4, 3, device=system.nuclear_coords.device)
        device_type = system.nuclear_coords.device.type
        lat, thr, out = measure_inference(model, r, warmup_steps=1, benchmark_steps=3, device_type=device_type)
        self.assertGreater(lat, 0.0)
        self.assertGreater(thr, 0.0)
        self.assertEqual(out.shape, (2,))

    def test_benchmark_100_atoms_helpers(self) -> None:
        """
        验证 benchmark_100_atoms 模块体系构建及精度支持。
        """
        from ferminet_pytorch.benchmark_100_atoms import build_h100_system

        # 验证默认 float32 构建
        sys_f32 = build_h100_system(spacing=1.8, dtype=torch.float32)
        self.assertEqual(sys_f32.num_nuclei, 100)
        self.assertEqual(sys_f32.total_electrons, 100)
        self.assertEqual(sys_f32.nuclear_coords.dtype, torch.float32)
        self.assertEqual(sys_f32.v_nn.dtype, torch.float32)

        # 验证 float64 双精度构建
        sys_f64 = build_h100_system(spacing=1.8, dtype=torch.float64)
        self.assertEqual(sys_f64.nuclear_coords.dtype, torch.float64)
        self.assertEqual(sys_f64.v_nn.dtype, torch.float64)

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
        loss, stats, e_calc = vmc_loss.compute_step_loss(self.model, self.system, walkers)
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
            num_walkers=16,
            num_epochs=2,
            mcmc_steps_per_epoch=2,
            burn_in_steps=4,
            lr=1e-3,
            evaluation_steps=4,
            evaluation_mcmc_steps=1,
            evaluation_burn_in_steps=2,
            verbose=False,
        )
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]["energy_raw"]), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(e)) for e in result["history"]["energy_raw"]))


if __name__ == "__main__":
    unittest.main()
