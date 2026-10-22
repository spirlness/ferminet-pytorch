"""
终极实证压力测试 (Final Empirical Stress Test)
==============================================
对 ferminet_pytorch 重构后的三大核心机制展开极限挑战：
1. 代数余子式大尺寸奇异矩阵求解 (size=8, size=10, det=0, 反向传播毫秒级无 OOM)
2. MCMC 生产采样细致平衡 (sample(adapt=False) 下非 50% 极端接受率步长严格冻结)
3. 退化方差下的 MAD 截断 (方差 -> 0 时代理损失梯度的非零有限性，无梯度窒息)
"""

import math
import time
import unittest
import torch
import torch.nn as nn

from ferminet_pytorch.model import _CofactorDeterminant, FermiNet
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss
from ferminet_pytorch.hamiltonian import MolecularSystem


class TestCofactorLargeSingular(unittest.TestCase):
    """
    挑战 1: 代数余子式大尺寸奇异矩阵求解。
    测试在 N=8, N=10 奇异矩阵 (det=0) 分支下，精确余子式导数是否有限；耗时仅作记录。
    """

    def test_singular_cofactor_size_8(self) -> None:
        """测试 size=8 奇异矩阵 (det=0) 的代数余子式反向传播与耗时。"""
        batch_size = 4
        size = 8
        torch.manual_seed(42)

        # 构造严格奇异矩阵：最后一行全为 0，确保行列式精确为 0
        mat = torch.randn(batch_size, size, size, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            mat[:, -1, :] = 0.0
            sign, _ = torch.linalg.slogdet(mat)
            self.assertTrue(torch.all(sign == 0), "构造的矩阵必须精确奇异 (det=0)")

        log_scale = torch.zeros(batch_size, dtype=torch.float32)

        # 预热一次
        det_warmup = _CofactorDeterminant.apply(mat, log_scale)
        loss_warmup = det_warmup.sum()
        loss_warmup.backward(retain_graph=True)
        mat.grad = None

        # 测量前向与反向传播耗时
        t0 = time.perf_counter()
        det = _CofactorDeterminant.apply(mat, log_scale)
        loss = det.sum()
        loss.backward()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0

        # 检验数值与耗时
        self.assertIsNotNone(mat.grad, "必须成功计算梯度")
        self.assertEqual(mat.grad.shape, (batch_size, size, size))
        self.assertTrue(torch.isfinite(mat.grad).all(), "梯度必须处处有限 (无 NaN/Inf)")
        # Timing is recorded, not a machine-dependent correctness condition.
        print(f"\n[Test 1.1 PASS] size=8 奇异矩阵 (det=0) 反向传播平稳执行，耗时: {elapsed_ms:.3f} ms，梯度全有限")

    def test_singular_cofactor_size_10(self) -> None:
        """测试 size=10 奇异矩阵 (det=0) 的代数余子式反向传播与耗时。"""
        batch_size = 4
        size = 10
        torch.manual_seed(100)

        # 构造奇异矩阵：最后一行全为 0，确保行列式严格为 0
        mat = torch.randn(batch_size, size, size, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            mat[:, -1, :] = 0.0
            sign, _ = torch.linalg.slogdet(mat)
            self.assertTrue(torch.all(sign == 0), "构造的矩阵必须精确奇异 (det=0)")

        log_scale = torch.zeros(batch_size, dtype=torch.float32)

        # 预热一次
        det_warmup = _CofactorDeterminant.apply(mat, log_scale)
        loss_warmup = det_warmup.sum()
        loss_warmup.backward(retain_graph=True)
        mat.grad = None

        # 测量耗时
        t0 = time.perf_counter()
        det = _CofactorDeterminant.apply(mat, log_scale)
        loss = det.sum()
        loss.backward()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0

        # 检验数值与耗时
        self.assertIsNotNone(mat.grad, "必须成功计算梯度")
        self.assertEqual(mat.grad.shape, (batch_size, size, size))
        self.assertTrue(torch.isfinite(mat.grad).all(), "梯度必须处处有限 (无 NaN/Inf)")
        # Exact derivatives are required regardless of the timing measurement.
        print(f"[Test 1.2 PASS] size=10 奇异矩阵 (det=0) 反向传播平稳执行，耗时: {elapsed_ms:.3f} ms，梯度全有限 (无阶乘爆炸/无 OOM)")

    def test_ferminet_large_system_nodal_surface_backward(self) -> None:
        """端到端检验：构建 8 电子体系 (n_up=8, n_down=0)，两电子重合于泡利节点面上时的反向传播。"""
        nuclear_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        nuclear_charges = torch.tensor([8.0], dtype=torch.float32)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=8,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )

        batch_size = 2
        walkers = torch.randn(batch_size, 8, 3, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            # 制造泡利节点碰撞：第 0 和第 1 个同自旋电子空间坐标完全重合
            walkers[:, 1, :] = walkers[:, 0, :]

        t0 = time.perf_counter()
        log_psi, sign = model.forward_log_psi(walkers)
        loss = log_psi.sum()
        loss.backward()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        self.assertIsNotNone(walkers.grad)
        self.assertTrue(torch.isfinite(walkers.grad).all(), "电子坐标导数处处有限")
        # Record wall time separately from correctness: CPU load and compiler
        # warmup make a fixed latency threshold unsuitable for this test.
        print(f"[Test 1.3 PASS] FermiNet 8-电子泡利节点全流程反向传播平稳，耗时: {elapsed_ms:.3f} ms")


class TestMCMCStepSizeFreezing(unittest.TestCase):
    """
    挑战 2: MCMC 生产采样细致平衡。
    验证在 sample(num_steps=20, adapt=False) 下，无论接受率偏离 50% 目标多大 (0% 或 100%)，
    步长 step_size 严格保持为初始步长不变。
    """

    def setUp(self) -> None:
        coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32)
        charges = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.system = MolecularSystem(coords, charges, n_up=1, n_down=1)
        self.model = FermiNet(coords, charges, n_up=1, n_down=1, num_determinants=2, hidden_dim_one=16, hidden_dim_two=8)

    def test_mcmc_frozen_step_size_under_zero_acceptance(self) -> None:
        """测试在极端低接受率 (接近 0%) 挑战下，sample(adapt=False) 步长严格不变。"""
        # 设置极大步长 (例如 20.0)，使得候选构型跳跃极大，接受率趋近 0%
        initial_step = 20.0
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=64,
            step_size=initial_step,
            target_acceptance=0.5,
            adapt_step_size=True,  # 即使实例本身开启了 adapt_step_size
        )

        # 生产采样阶段必须冻结
        samples = sampler.sample(num_steps=20, adapt=False)

        # 检验最新步长与初始步长
        final_step = sampler.step_size
        self.assertEqual(final_step, initial_step, "在 adapt=False 下，step_size 必须严格保持恒定不变")
        self.assertLess(sampler.last_acceptance_rate, 0.2, "应处于远离 50% 的低接受率状态")
        print(f"\n[Test 2.1 PASS] 极端低接受率 ({sampler.last_acceptance_rate*100:.1f}%) 挑战下，step_size 严格冻结: {initial_step} == {final_step}")

    def test_mcmc_frozen_step_size_under_high_acceptance(self) -> None:
        """测试在极端高接受率 (接近 100%) 挑战下，sample(adapt=False) 步长严格不变。"""
        # 设置极小步长 (例如 1e-4)，使得接受率接近 100%
        initial_step = 1e-4
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=64,
            step_size=initial_step,
            target_acceptance=0.5,
            adapt_step_size=True,
        )

        samples = sampler.sample(num_steps=20, adapt=False)
        final_step = sampler.step_size

        self.assertEqual(final_step, initial_step, "在 adapt=False 下，step_size 必须严格保持恒定不变")
        self.assertGreater(sampler.last_acceptance_rate, 0.8, "应处于远离 50% 的极高接受率状态")
        print(f"[Test 2.2 PASS] 极端高接受率 ({sampler.last_acceptance_rate*100:.1f}%) 挑战下，step_size 严格冻结: {initial_step} == {final_step}")

    def test_contrast_adaptive_vs_frozen(self) -> None:
        """对照测试：证明 adapt=True 会动态改变步长，而 adapt=False 能绝对冻结。"""
        step = 10.0
        sampler = MCMCSampler(
            model=self.model,
            system=self.system,
            num_walkers=64,
            step_size=step,
            target_acceptance=0.5,
            adapt_rate=0.1,
        )

        # 开启自适应：步长必定收缩
        sampler.sample(num_steps=5, adapt=True)
        step_after_adapt = sampler.step_size
        self.assertNotEqual(step_after_adapt, step, "adapt=True 时步长必须发生动态调整")

        # 冻结自适应：步长必须严格定格
        sampler.sample(num_steps=20, adapt=False)
        step_after_frozen = sampler.step_size
        self.assertEqual(step_after_frozen, step_after_adapt, "adapt=False 时步长必须完全冻结")
        print(f"[Test 2.3 PASS] 自适应 vs 冻结对比鲜明: 初始 {step} -> 自适应调整至 {step_after_adapt:.4f} -> 冻结保持 {step_after_frozen:.4f}")


class TestDegenerateVarianceMADClipping(unittest.TestCase):
    """
    挑战 3: 退化方差下的 MAD 截断与梯度有限性。
    验证在批次能量方差趋向 0 时，MAD 截断受物理下界保护，代理损失梯度非零有限，杜绝梯度窒息。
    """

    def test_mad_clipping_preserves_gradient_under_vanishing_variance(self) -> None:
        """测试在不同极小方差尺度 (1e-3, 1e-5, 1e-7) 下，代理损失梯度保持非零且有限。"""
        torch.manual_seed(42)
        batch_size = 64
        e_median_true = -1.174  # H2 基态能量尺度 (Hartree)

        # 生成零均值、方差为 1 的随机扰动向量
        raw_noise = torch.randn(batch_size, dtype=torch.float32)
        noise = (raw_noise - raw_noise.mean()) / raw_noise.std()

        # 虚拟波函数对数模参数化：log_abs_psi = weight * feature
        feature = torch.randn(batch_size, dtype=torch.float32)

        scales = [1e-3, 1e-5, 1e-7]
        for scale in scales:
            weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            log_abs_psi = weight * feature

            # 构造退化能量批次：方差为 scale^2，趋向 0
            e_local = e_median_true + scale * noise

            loss, stats = vmc_surrogate_loss(log_abs_psi, e_local, clip_factor=5.0, clip_method="tanh")
            loss.backward()

            # 1. 验证统计指标中 MAD 随 scale 趋于 0
            self.assertLess(stats["energy_mad"], scale * 2.0)

            # 2. 验证梯度存在、有限且非零 (绝对无梯度窒息)
            grad_val = weight.grad.item()
            self.assertTrue(math.isfinite(grad_val), f"尺度 {scale} 下梯度必须有限")
            self.assertNotEqual(grad_val, 0.0, f"尺度 {scale} 下梯度不得为零 (杜绝梯度窒息)")
            self.assertFalse(math.isnan(grad_val), f"尺度 {scale} 下梯度不得为 NaN")

            # 3. 验证与无截断理论协方差梯度的相对误差 (Taylor 展开一致性)
            # 无截断真实梯度: 2.0 * mean( (E_L - mean(E_L)) * feature )
            delta_e_exact = e_local - e_local.mean()
            theoretical_grad = (2.0 * torch.mean(delta_e_exact * feature)).item()
            rel_error = abs(grad_val - theoretical_grad) / (abs(theoretical_grad) + 1e-12)

            # 随 scale 减小，x = delta_E / Delta -> 0，tanh(x) -> x，相对误差按 O(scale^2) 极速衰减
            expected_tolerance = 0.01 if scale >= 1e-3 else 1e-4
            self.assertLess(
                rel_error,
                expected_tolerance,
                f"在退化方差尺度 {scale} 下，截断后梯度与真实协方差梯度的相对误差应低于 {expected_tolerance} (实测 {rel_error:.2e})",
            )
            print(f"[Test 3.1 PASS] 尺度 sigma={scale:.1e}: grad={grad_val:+.6e}, 理论一致性相对误差: {rel_error:.2e}")

    def test_mad_clipping_hard_method_under_vanishing_variance(self) -> None:
        """测试 hard 截断在方差趋近 0 时同样正常工作且保持梯度。"""
        batch_size = 32
        e_median = -0.5
        noise = torch.randn(batch_size) * 1e-6
        e_local = e_median + noise

        weight = nn.Parameter(torch.tensor(1.0))
        feature = torch.randn(batch_size)
        log_abs_psi = weight * feature

        loss, stats = vmc_surrogate_loss(log_abs_psi, e_local, clip_factor=5.0, clip_method="hard")
        loss.backward()

        self.assertTrue(math.isfinite(weight.grad.item()))
        self.assertNotEqual(weight.grad.item(), 0.0)
        print(f"[Test 3.2 PASS] hard 截断模式下退化方差梯度非零有限: grad={weight.grad.item():+.6e}")

    def test_exact_zero_variance_boundary_safety(self) -> None:
        """边界极限挑战：当所有 Walker 能量完全相等 (方差严格为 0) 时，系统平稳执行且 loss 为 0。"""
        batch_size = 32
        e_local = torch.full((batch_size,), -1.17, dtype=torch.float32)

        weight = nn.Parameter(torch.tensor(1.0))
        feature = torch.randn(batch_size)
        log_abs_psi = weight * feature

        loss, stats = vmc_surrogate_loss(log_abs_psi, e_local, clip_factor=5.0, clip_method="tanh")
        loss.backward()

        self.assertEqual(loss.item(), 0.0, "零方差下变分代理损失应精确为 0")
        self.assertEqual(weight.grad.item(), 0.0, "零方差下梯度应为 0")
        self.assertTrue(math.isfinite(weight.grad.item()), "零方差下梯度不得为 NaN/Inf")
        print(f"[Test 3.3 PASS] 精确零方差极限边界平稳执行: loss={loss.item():.1f}, grad={weight.grad.item():.1f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
