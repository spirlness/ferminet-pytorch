"""
几何不变量与数值稳定性回归测试 (单元 1 专属)
============================================
覆盖 `audit_review/independent_physics_audit.md` 中 D5 (核序置换不变性) 与
D6 (距离软化改 clamp 核) 的包络级回归测试。模型级 (log|Psi|) 断言见
test_physics_invariants.py；本文件聚焦 NuclearEnvelope / ElectronFeatureEmbedding
的参数形状、共享语义、距离核数值性质与重合点有限性。

独立成文件 (而非追加进 test_physics_invariants.py) 以避免与并行单元 2/3 的
测试改动冲突。
"""

import math
import os
import sys
import unittest

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ferminet_pytorch.envelope import NuclearEnvelope


class TestNuclearEnvelopeGroupSharing(unittest.TestCase):
    """D5 第一级：包络参数按电荷组共享的形状与置换不变性。"""

    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_homonuclear_ties_envelope_parameters(self) -> None:
        """正例：两个 Z=1 的核必须共享包络参数 (G=1)，且交换核序包络值不变。"""
        k, n_orb, m = 3, 2, 2
        env = NuclearEnvelope(
            num_determinants=k,
            num_orbitals=n_orb,
            num_nuclei=m,
            nuclear_charges=torch.tensor([1.0, 1.0]),
        ).double()
        self.assertEqual(env.num_groups, 1)
        self.assertTrue(env.tied)
        self.assertEqual(tuple(env.weights.shape), (k, n_orb, 1))
        self.assertEqual(tuple(env.decay_rates_raw.shape), (k, n_orb, 1))
        self.assertEqual(env.nucleus_group_index.dtype, torch.long)
        self.assertEqual(env.nucleus_group_index.tolist(), [0, 0])

        r = torch.randn(4, n_orb, 3, dtype=torch.float64)
        coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
        with torch.no_grad():
            out = env(r, coords)
            out_swapped = env(r, coords.flip(0))
        self.assertEqual(out.shape, (4, k, n_orb, n_orb))
        self.assertLess(float((out - out_swapped).abs().max()), 1e-12)

    def test_heteronuclear_does_not_tie_parameters(self) -> None:
        """反例：Z=[1,2] 的异电荷核不得共享参数，交换核序必须改变包络值。"""
        k, n_orb, m = 3, 2, 2
        env = NuclearEnvelope(
            num_determinants=k,
            num_orbitals=n_orb,
            num_nuclei=m,
            nuclear_charges=torch.tensor([1.0, 2.0]),
        ).double()
        self.assertEqual(env.num_groups, 2)
        self.assertFalse(env.tied)
        self.assertEqual(tuple(env.weights.shape), (k, n_orb, 2))
        self.assertEqual(tuple(env.decay_rates_raw.shape), (k, n_orb, 2))
        self.assertEqual(env.nucleus_group_index.tolist(), [0, 1])

        r = torch.randn(4, n_orb, 3, dtype=torch.float64)
        coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
        with torch.no_grad():
            out = env(r, coords)
            out_swapped = env(r, coords.flip(0))
        self.assertGreater(float((out - out_swapped).abs().max()), 0.1)

    def test_no_charges_backward_compatible_identity_grouping(self) -> None:
        """向后兼容：nuclear_charges=None 时 G=M、索引 arange(M)，形状与旧版一致。"""
        k, n_orb, m = 2, 2, 3
        env = NuclearEnvelope(
            num_determinants=k, num_orbitals=n_orb, num_nuclei=m
        ).double()
        self.assertEqual(env.num_groups, m)
        self.assertFalse(env.tied)
        self.assertEqual(tuple(env.weights.shape), (k, n_orb, m))
        self.assertEqual(tuple(env.decay_rates_raw.shape), (k, n_orb, m))
        self.assertEqual(env.nucleus_group_index.tolist(), [0, 1, 2])

    def test_group_index_is_saved_in_state_dict(self) -> None:
        """R4: the grouping is physical checkpoint state, not disposable metadata."""
        env = NuclearEnvelope(
            num_determinants=2,
            num_orbitals=1,
            num_nuclei=2,
            nuclear_charges=torch.tensor([1.0, 1.0]),
        )
        self.assertIn("nucleus_group_index", env.state_dict())

    def test_charges_numel_must_match_num_nuclei(self) -> None:
        """提供 nuclear_charges 时元素数必须与 num_nuclei 一致。"""
        with self.assertRaises(AssertionError):
            NuclearEnvelope(
                num_determinants=1,
                num_orbitals=1,
                num_nuclei=2,
                nuclear_charges=torch.tensor([1.0, 1.0, 1.0]),
            )

    def test_shared_parameters_receive_symmetric_gradient_sum(self) -> None:
        """同电荷核共享参数：反向传播时梯度应自动 scatter-add (对称和)。"""
        k, n_orb, m = 2, 1, 2
        env = NuclearEnvelope(
            num_determinants=k,
            num_orbitals=n_orb,
            num_nuclei=m,
            nuclear_charges=torch.tensor([1.0, 1.0]),
        ).double()
        coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
        r = torch.randn(1, n_orb, 3, dtype=torch.float64, requires_grad=True)
        env(r, coords).sum().backward()
        self.assertIsNotNone(env.weights.grad)
        self.assertTrue(torch.isfinite(env.weights.grad).all())
        self.assertGreater(float(env.weights.grad.abs().max()), 0.0)


class TestClampDistanceKernel(unittest.TestCase):
    """D6：sqrt(clamp(r^2, min=eps)) 距离核的数值性质。"""

    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_clamp_matches_unsoftened_distance_bitwise_above_eps(self) -> None:
        """在 r^2 >= eps 处 clamp 是恒等变换：距离与未软化值逐位相同。"""
        eps = 1e-24
        r = torch.randn(64, 3, dtype=torch.float64) * 10.0
        d_clamp = torch.sqrt(torch.clamp(torch.sum(r ** 2, dim=-1), min=eps))
        d_naive = torch.sqrt(torch.sum(r ** 2, dim=-1))
        self.assertTrue(torch.equal(d_clamp, d_naive))

    def test_bare_sqrt_produces_nan_gradient_at_zero_clamp_does_not(self) -> None:
        """记录为何不能直接删除 eps：裸 sqrt(r^2) 在 r=0 处梯度为 NaN (inf*0)，
        而 clamp 核在重合点梯度为零、数值有限。
        """
        origin = torch.zeros(1, 3, dtype=torch.float64)

        # 反例 (旧式裸 sqrt)：r=0 处 d/dr sqrt(r^2) = r/|r| 为 inf*0 -> NaN
        r_naive = origin.detach().clone().requires_grad_(True)
        torch.sqrt(torch.sum(r_naive ** 2, dim=-1)).sum().backward()
        self.assertTrue(torch.isnan(r_naive.grad).any())

        # 正例 (clamp 核)：r=0 处距离有限 (= sqrt(eps)) 且梯度为零，无 NaN
        r_clamp = origin.detach().clone().requires_grad_(True)
        eps = 1e-24
        d = torch.sqrt(torch.clamp(torch.sum(r_clamp ** 2, dim=-1), min=eps))
        self.assertTrue(torch.isfinite(d).all())
        d.sum().backward()
        self.assertFalse(torch.isnan(r_clamp.grad).any())
        self.assertTrue(torch.isfinite(r_clamp.grad).all())
        self.assertEqual(float(r_clamp.grad.abs().max()), 0.0)

    def test_envelope_finite_at_coincidence_point(self) -> None:
        """重合点有限性：电子与核完全重合 (r=0) 时包络值有限、梯度有限、无 NaN。"""
        for anisotropic in (False, True):
            with self.subTest(anisotropic=anisotropic):
                env = NuclearEnvelope(
                    num_determinants=2,
                    num_orbitals=1,
                    num_nuclei=2,
                    anisotropic=anisotropic,
                    nuclear_charges=torch.tensor([1.0, 1.0]),
                ).double()
                coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
                # 电子 0 与核 0 严格重合；电子 1 位于正常位置
                r = torch.tensor(
                    [[[-0.7, 0.0, 0.0]], [[0.3, 0.1, -0.2]]], dtype=torch.float64,
                    requires_grad=True,
                )
                out = env(r, coords)
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                self.assertIsNotNone(r.grad)
                self.assertTrue(torch.isfinite(r.grad).all())
                self.assertFalse(torch.isnan(r.grad).any())

    def test_envelope_distance_bias_is_exactly_zero_outside_softening(self) -> None:
        """clamp 核在软化区外零偏置：1e-6 Bohr 处包络值与解析 exp(-alpha*r) 一致。"""
        env = NuclearEnvelope(
            num_determinants=1,
            num_orbitals=1,
            num_nuclei=1,
            init_decay_rate=1.0,
            nuclear_charges=torch.tensor([1.0]),
        ).double()
        with torch.no_grad():
            env.weights.fill_(1.0)
            alpha = float(torch.nn.functional.softplus(env.decay_rates_raw.detach()).item())
        origin = torch.zeros(1, 3, dtype=torch.float64)
        r = torch.tensor([[[1e-6, 0.0, 0.0]]], dtype=torch.float64)
        out = env(r, origin)
        expected = math.exp(-alpha * 1e-6)
        self.assertAlmostEqual(float(out.item()), expected, delta=1e-15)


if __name__ == "__main__":
    unittest.main()
