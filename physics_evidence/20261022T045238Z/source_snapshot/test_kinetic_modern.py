"""
单元测试: test_kinetic_modern.py
验证 KineticEnergyCalculator 的多模式求解 (REVERSE, HUTCHINSON, FORWARD_JVP)。
"""

import unittest
import torch
from kinetic_modern import KineticEnergyCalculator, KineticMode


class TestKineticModern(unittest.TestCase):
    def test_modes_on_harmonic_wavefunction(self) -> None:
        """针对解析波函数 psi(r) = exp(-0.5 * alpha * sum(r^2)) 验证各模式动能数值吻合。"""
        # 对数波函数: log_abs_psi(r) = -0.5 * alpha * sum(r^2)
        # 梯度: grad_i = -alpha * r_i
        # 模平方: |grad|^2 = alpha^2 * sum(r^2)
        # 拉普拉斯: nabla^2 ln|psi| = -alpha * (3N)
        # 动能: T_L = -0.5 * (-3N*alpha + alpha^2 * sum(r^2))
        alpha = 2.0
        B, N = 8, 2
        r = torch.randn(B, N, 3, dtype=torch.float64, requires_grad=True)

        def log_abs_psi_fn(coords: torch.Tensor):
            val = -0.5 * alpha * torch.sum(coords ** 2, dim=(-1, -2))
            sign = torch.ones_like(val)
            return val, sign

        expected_lap = torch.full((B,), -alpha * (3 * N), dtype=torch.float64)
        expected_grad_sq = (alpha ** 2) * torch.sum(r ** 2, dim=(-1, -2))
        expected_t = -0.5 * (expected_lap + expected_grad_sq)

        # 1. REVERSE 模式
        t_rev, lap_rev, grad_rev = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi_fn=log_abs_psi_fn,
            r=r,
            mode=KineticMode.REVERSE,
        )
        self.assertTrue(torch.allclose(lap_rev, expected_lap, atol=1e-10))
        self.assertTrue(torch.allclose(grad_rev, expected_grad_sq, atol=1e-10))
        self.assertTrue(torch.allclose(t_rev, expected_t, atol=1e-10))

        # 2. FORWARD_JVP 模式
        t_jvp, lap_jvp, grad_jvp = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi_fn=log_abs_psi_fn,
            r=r,
            mode=KineticMode.FORWARD_JVP,
        )
        self.assertTrue(torch.allclose(lap_jvp, expected_lap, atol=1e-10))
        self.assertTrue(torch.allclose(t_jvp, expected_t, atol=1e-10))

        # 3. HUTCHINSON 模式 (无偏估计，大样本探针下平均误差小)
        torch.manual_seed(42)
        t_hutch, lap_hutch, _ = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi_fn=log_abs_psi_fn,
            r=r,
            mode=KineticMode.HUTCHINSON,
            num_probes=200,
        )
        # Hutchinson 估计值均值在预期值附近
        self.assertLess(torch.abs(lap_hutch.mean() - expected_lap.mean()).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
