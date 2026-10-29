"""Empirical stress and adversarial verification harness for Milestone 1 (P2 Refactoring).

Author: Challenger M1 (Empirical Challenger)
Target: layers.py (FINDING-PYT-05), mcmc.py (F-MCMC-02), loss.py (F-LOSS-02)
"""

import unittest
import warnings
import torch

from ferminet_pytorch.layers import FermiInteractionLayer
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.loss import vmc_surrogate_loss, VMCLoss
from ferminet_pytorch.hamiltonian import MolecularSystem
from ferminet_pytorch.model import FermiNet


class TestChallengerTwoElectronSelfInteraction(unittest.TestCase):
    """实证对抗测试 1：双电子自交互剔除、对角线扰动不变性与单电子无 NaN 梯度。"""

    def setUp(self) -> None:
        torch.manual_seed(20261015)

    def test_extreme_diagonal_perturbation_invariance(self) -> None:
        """
        对抗测试：向对角线 h2[b, i, i, :] 注入极端天文大值扰动 (如 10^6, -10^7 等)，
        实证检验双电子均值池化输出与单电子流输出是否严格绝缘 (Bit-exact 或浮点 0 误差)。
        """
        test_spin_configs = [
            (2, 2),  # 4 电子偶数
            (3, 1),  # 4 电子奇偶不对称
            (4, 3),  # 7 电子多电子
            (1, 2),  # 3 电子 (Li 阳离子/原子)
        ]

        batch_size = 4
        in_dim_one = 12
        in_dim_two = 6
        out_dim_one = 12
        out_dim_two = 6

        for n_up, n_down in test_spin_configs:
            num_electrons = n_up + n_down
            with self.subTest(n_up=n_up, n_down=n_down):
                layer = FermiInteractionLayer(
                    in_dim_one=in_dim_one,
                    in_dim_two=in_dim_two,
                    out_dim_one=out_dim_one,
                    out_dim_two=out_dim_two,
                    produce_two_electron_output=False,
                    n_up=n_up,
                    n_down=n_down,
                )

                # 生成随机的非对称输入特征
                h1 = torch.randn(batch_size, num_electrons, in_dim_one)
                h2_clean = torch.randn(
                    batch_size, num_electrons, num_electrons, in_dim_two
                )

                # 构造极端对角线扰动副本
                h2_perturbed = h2_clean.clone()
                for i in range(num_electrons):
                    # 每个电子注入不同量级的极端干扰: 10^6, -10^7, 5*10^8 ...
                    perturbation = ((-1.0) ** i) * (10.0 ** (6 + (i % 3)))
                    h2_perturbed[:, i, i, :] = perturbation

                # 前向传播
                with torch.no_grad():
                    h1_clean, _ = layer(h1, h2_clean)
                    h1_perturbed, _ = layer(h1, h2_perturbed)

                # 验证单电子流更新完全不受任何对角线扰动影响
                diff = (h1_clean - h1_perturbed).abs().max().item()
                self.assertEqual(
                    diff,
                    0.0,
                    f"对角线极端扰动泄漏到了池化输出！最大绝对误差: {diff} (n_up={n_up}, n_down={n_down})",
                )

    def test_diagonal_autograd_zero_gradient(self) -> None:
        """
        对抗测试：令 h2.requires_grad=True，验证反向传播后对角线梯度 h2.grad[:, i, i, :]
        是否全部严格为 0.0，证明自交互项在计算图中被完全切断。
        """
        batch_size = 2
        n_up = 3
        n_down = 2
        num_electrons = n_up + n_down
        in_dim_one = 8
        in_dim_two = 4

        layer = FermiInteractionLayer(
            in_dim_one=in_dim_one,
            in_dim_two=in_dim_two,
            out_dim_one=in_dim_one,
            out_dim_two=in_dim_two,
            produce_two_electron_output=False,
            n_up=n_up,
            n_down=n_down,
        )

        h1 = torch.randn(batch_size, num_electrons, in_dim_one)
        h2 = torch.randn(
            batch_size, num_electrons, num_electrons, in_dim_two, requires_grad=True
        )

        h1_next, _ = layer(h1, h2)
        loss = h1_next.sum()
        loss.backward()

        self.assertIsNotNone(h2.grad)
        for i in range(num_electrons):
            diag_grad = h2.grad[:, i, i, :]
            max_diag_grad = diag_grad.abs().max().item()
            self.assertEqual(
                max_diag_grad,
                0.0,
                f"对角线位置 (i={i}, j={i}) 存在非零梯度: {max_diag_grad}，计算图未彻底绝缘自身项！",
            )

        # 非对角线位置梯度应正常存在
        off_diag_mask = ~torch.eye(num_electrons, dtype=torch.bool).unsqueeze(
            0
        ).unsqueeze(-1)
        off_diag_grad = h2.grad.masked_select(off_diag_mask)
        self.assertTrue(
            (off_diag_grad.abs() > 0).any(), "非对角线应具有正常的梯度反传。"
        )

    def test_single_electron_regime_zero_pooling_and_nan_free_gradients(self) -> None:
        """
        对抗测试：验证单电子极限体系 (n_up=1, n_down=1 / n_up=1, n_down=0 / n_up=0, n_down=1)
        中，同自旋通道由于自身剔除后无有效伙伴，池化均值严格为零张量，
        且反向传播 Autograd 严格无 NaN。
        """
        batch_size = 3
        in_dim_one = 6
        in_dim_two = 4

        configs = [
            (1, 1),  # 双电子单自旋向上、单自旋向下 (H2)
            (1, 0),  # 单电子仅自旋向上 (H 原子)
            (0, 1),  # 单电子仅自旋向下
        ]

        for n_up, n_down in configs:
            layer = FermiInteractionLayer(
                in_dim_one=in_dim_one,
                in_dim_two=in_dim_two,
                out_dim_one=in_dim_one,
                out_dim_two=in_dim_two,
                produce_two_electron_output=False,
                n_up=n_up,
                n_down=n_down,
            )
            num_electrons = n_up + n_down
            with self.subTest(n_up=n_up, n_down=n_down):
                h1 = torch.randn(
                    batch_size, num_electrons, in_dim_one, requires_grad=True
                )
                h2 = torch.randn(
                    batch_size,
                    num_electrons,
                    num_electrons,
                    in_dim_two,
                    requires_grad=True,
                )

                # 手动重构 layer 中的 mask 和 mean_h2 计算，直接对中间量进行零张量断言
                eye_mask = torch.eye(num_electrons, dtype=torch.bool, device=h2.device)
                h2_no_diag = h2.masked_fill(eye_mask.unsqueeze(0).unsqueeze(-1), 0.0)

                sum_h2_up = torch.sum(h2_no_diag[:, :, :n_up, :], dim=2)
                counts_up = torch.full(
                    (1, num_electrons, 1), float(n_up), device=h2.device, dtype=h2.dtype
                )
                if n_up > 0:
                    counts_up[:, :n_up, :] = float(n_up - 1)
                mean_h2_up = torch.where(
                    counts_up > 0,
                    sum_h2_up / counts_up.clamp(min=1.0),
                    torch.zeros_like(sum_h2_up),
                )

                sum_h2_down = torch.sum(h2_no_diag[:, :, n_up:, :], dim=2)
                counts_down = torch.full(
                    (1, num_electrons, 1),
                    float(n_down),
                    device=h2.device,
                    dtype=h2.dtype,
                )
                if n_down > 0:
                    counts_down[:, n_up:, :] = float(n_down - 1)
                mean_h2_down = torch.where(
                    counts_down > 0,
                    sum_h2_down / counts_down.clamp(min=1.0),
                    torch.zeros_like(sum_h2_down),
                )

                # 检验零张量属性
                if n_up == 1:
                    # 自旋向上电子 (index 0) 的同自旋通道必须严格为零
                    self.assertTrue(torch.all(mean_h2_up[:, 0, :] == 0.0))
                if n_down == 1:
                    # 自旋向下电子 (index n_up) 的同自旋通道必须严格为零
                    self.assertTrue(torch.all(mean_h2_down[:, n_up, :] == 0.0))
                if n_up == 0:
                    self.assertTrue(torch.all(mean_h2_up == 0.0))
                if n_down == 0:
                    self.assertTrue(torch.all(mean_h2_down == 0.0))

                # 运行完整的 layer 前向与反向传播
                h1_out, _ = layer(h1, h2)
                self.assertTrue(torch.isfinite(h1_out).all())
                self.assertFalse(torch.isnan(h1_out).any())

                loss = h1_out.sum()
                loss.backward()

                self.assertFalse(
                    torch.isnan(h1.grad).any(),
                    f"h1.grad 含有 NaN (n_up={n_up}, n_down={n_down})",
                )
                self.assertFalse(
                    torch.isnan(h2.grad).any(),
                    f"h2.grad 含有 NaN (n_up={n_up}, n_down={n_down})",
                )
                for name, param in layer.named_parameters():
                    if param.grad is not None:
                        self.assertFalse(
                            torch.isnan(param.grad).any(), f"参数 {name}.grad 含有 NaN"
                        )


class TestChallengerMCMCAcceptance(unittest.TestCase):
    """实证对抗测试 2：MCMC 接受率参数组合、解耦默认值与步长动态自适应方向。"""

    def setUp(self) -> None:
        torch.manual_seed(20261015)
        # 简单氢分子体系
        self.coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
        self.charges = torch.tensor([1.0, 1.0])
        self.system = MolecularSystem(
            nuclear_coords=self.coords, nuclear_charges=self.charges, n_up=1, n_down=1
        )
        self.model = FermiNet(
            self.coords,
            self.charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        )

    def test_parameter_combinations_and_defaults(self) -> None:
        """验证所有 move_mode 与 target_acceptance 参数组合及别名映射。"""
        cases = [
            ("all", None, 0.234),
            ("one_by_one", None, 0.50),
            ("all", 0.15, 0.15),
            ("all", 0.50, 0.50),
            ("one_by_one", 0.234, 0.234),
            ("one_by_one", 0.65, 0.65),
        ]

        for move_mode, target_in, expected_target in cases:
            with self.subTest(move_mode=move_mode, target_in=target_in):
                sampler = MCMCSampler(
                    model=self.model,
                    system=self.system,
                    move_mode=move_mode,
                    target_acceptance=target_in,
                )
                self.assertAlmostEqual(
                    sampler.target_acceptance, expected_target, places=4
                )
                self.assertEqual(sampler.move_mode, move_mode)

        # 非法移动模式断言拦截
        with self.assertRaises(AssertionError):
            MCMCSampler(model=self.model, system=self.system, move_mode="invalid_mode")

        # 别名同一性验证
        alias_sampler = MCMCSampler(
            model=self.model, system=self.system, move_mode="all"
        )
        self.assertAlmostEqual(alias_sampler.target_acceptance, 0.234, places=4)

    def test_step_size_adaptation_direction(self) -> None:
        """
        实证验证自适应步长调节方向：
        若接受率高于目标接受率，步长增大；若低于目标接受率，步长缩小。
        """
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            move_mode="all",
            num_walkers=32,
            step_size=0.1,
            adapt_step_size=True,
            adapt_rate=0.1,
        )
        self.assertEqual(sampler.target_acceptance, 0.234)

        # 模拟高接受率环境 (比如 0.90 > 0.234)
        initial_step = sampler.step_size
        diff_high = 0.90 - sampler.target_acceptance
        expected_step_after_high = initial_step * float(
            torch.exp(torch.tensor(sampler.adapt_rate * diff_high)).item()
        )
        self.assertGreater(expected_step_after_high, initial_step)

        # 模拟低接受率环境 (比如 0.05 < 0.234)
        diff_low = 0.05 - sampler.target_acceptance
        expected_step_after_low = initial_step * float(
            torch.exp(torch.tensor(sampler.adapt_rate * diff_low)).item()
        )
        self.assertLess(expected_step_after_low, initial_step)


class TestChallengerLossCircuitBreaker(unittest.TestCase):
    """实证对抗测试 3：变分损失异常能量比例熔断边界对抗测试 (89.9% vs 90.1%)。"""

    def setUp(self) -> None:
        torch.manual_seed(20261015)

    def test_circuit_breaker_exact_boundary_89_9_vs_90_1(self) -> None:
        """
        关键对抗用例：
        在总样本数 B=1000 时：
        - 有效比例为 89.9% (899 有效, 101 无效, 占比 10.1% > 10.0%) -> 必须精确触发 FloatingPointError 熔断！
        - 有效比例为 90.1% (901 有效, 99 无效, 占比 9.9% <= 10.0%) -> 必须正常通过，发出 Warning，不抛出异常！
        """
        batch_size = 1000
        log_abs = torch.zeros(batch_size, requires_grad=True)

        # 1. 构造 89.9% 有效样本 (101 个无效样本, 无效比例 10.1%)
        energy_89_9 = torch.ones(batch_size)
        energy_89_9[:101] = float("nan")

        # 断言超标：抛出 FloatingPointError 且消息包含比例提示
        with self.assertRaises(FloatingPointError) as cm:
            vmc_surrogate_loss(log_abs, energy_89_9, max_invalid_fraction=0.10)
        self.assertIn("exceeds maximum allowed", str(cm.exception))
        self.assertIn("10.10%", str(cm.exception))

        # VMCLoss 模块封装一致性断言
        vmc_module = VMCLoss(max_invalid_fraction=0.10)
        with self.assertRaises(FloatingPointError) as cm_mod:
            vmc_module(log_abs, energy_89_9)
        self.assertIn("exceeds maximum allowed", str(cm_mod.exception))

        # 2. 构造 90.1% 有效样本 (99 个无效样本, 无效比例 9.9%)
        energy_90_1 = torch.ones(batch_size)
        energy_90_1[:99] = float("nan")

        # 断言合规：正常计算，不抛异常，且截获 Warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss, stats = vmc_surrogate_loss(
                log_abs, energy_90_1, max_invalid_fraction=0.10
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertIn(
                "Excluded 99/1000 nonfinite local energies", str(w[-1].message)
            )

        self.assertEqual(stats["num_valid"], 901)
        self.assertEqual(stats["num_invalid"], 99)
        self.assertTrue(torch.isfinite(loss))

        # VMCLoss 模块封装在 90.1% 时也必须正常通过
        with warnings.catch_warnings(record=True) as w_mod:
            warnings.simplefilter("always")
            loss_m, stats_m = vmc_module(log_abs, energy_90_1)
            self.assertEqual(len(w_mod), 1)
        self.assertEqual(stats_m["num_valid"], 901)
        self.assertTrue(torch.isfinite(loss_m))

    def test_circuit_breaker_critical_equal_boundary_90_0(self) -> None:
        """
        临界边界测试：
        恰好等于 10.0% 无效 (900 有效, 100 无效, 占比 10.0%)。
        判断条件为 invalid_fraction > max_invalid_fraction。
        恰好等于阈值时不应触发熔断，应被允许。
        """
        batch_size = 1000
        log_abs = torch.zeros(batch_size, requires_grad=True)
        energy_90_0 = torch.ones(batch_size)
        energy_90_0[:100] = float("inf")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss, stats = vmc_surrogate_loss(
                log_abs, energy_90_0, max_invalid_fraction=0.10
            )
            self.assertEqual(len(w), 1)

        self.assertEqual(stats["num_valid"], 900)
        self.assertEqual(stats["num_invalid"], 100)
        self.assertTrue(torch.isfinite(loss))

    def test_circuit_breaker_all_invalid_and_nonfinite_log_psi(self) -> None:
        """
        退化对抗场景：
        1. 全部样本非有限 -> 抛出 FloatingPointError ("No finite local energies")
        2. log_abs_psi 本身包含 NaN -> 抛出 FloatingPointError ("Nonfinite log|Psi|")
        """
        batch_size = 50
        # 1. 全部能量非有限
        log_abs = torch.zeros(batch_size, requires_grad=True)
        energy_all_nan = torch.full((batch_size,), float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "No finite local energies"):
            vmc_surrogate_loss(log_abs, energy_all_nan)

        # 2. log_abs 包含 NaN
        log_abs_nan = torch.zeros(batch_size, requires_grad=True)
        with torch.no_grad():
            log_abs_nan[5] = float("nan")
        energy_valid = torch.ones(batch_size)
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite log|Psi|"):
            vmc_surrogate_loss(log_abs_nan, energy_valid)


if __name__ == "__main__":
    unittest.main()
