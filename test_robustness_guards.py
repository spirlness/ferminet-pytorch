"""
边界行为与守卫回归测试 (D7-D10)
=================================
把 `audit_review/independent_physics_audit.md` 中 D7-D10 四项低影响缺陷的修复
固化为可执行断言。这些缺陷的共同点是"静默"——不产生错误答案，但会掩盖真实问题，
因此测试的重点是**行为是否可见、是否可恢复**。
"""

import os
import sys
import unittest
from unittest import mock

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss


def _h2_model(seed: int = 1, num_determinants: int = 4, num_layers: int = 2) -> FermiNet:
    torch.manual_seed(seed)
    coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return FermiNet(
        nuclear_coords=coords,
        nuclear_charges=charges,
        n_up=1,
        n_down=1,
        num_determinants=num_determinants,
        hidden_dim_one=32,
        hidden_dim_two=16,
        num_layers=num_layers,
    ).double()


class TestDeterminantWeightRecovery(unittest.TestCase):
    """D7：行列式权重归零后必须仍可恢复。"""

    def setUp(self) -> None:
        self.model = _h2_model()
        self.r = torch.randn(6, 2, 3, dtype=torch.float64)

    def test_combination_matches_direct_linear_combination(self) -> None:
        """多行列式组合必须与"直接算行列式再线性组合"的朴素实现一致。

        新的稳定化只对行列式提出最大值、权重保持线性因子，因此数值上应与
        未做任何近似的实现逐位接近。这条测试防止稳定化改写改变物理。
        """
        with torch.no_grad():
            d_up, d_down = self.model.forward_orbitals(self.r)
            det = torch.linalg.det(d_up) * torch.linalg.det(d_down)  # (B, K)
            psi_direct = det @ self.model.det_weights
            log_direct = torch.log(torch.abs(psi_direct))
            sign_direct = torch.sign(psi_direct)

            log_model, sign_model = self.model.forward_log_psi(self.r)

        self.assertLess(float((log_model - log_direct).abs().max()), 1e-10)
        self.assertTrue(torch.equal(sign_model, sign_direct))

    def test_zeroed_weight_keeps_nonzero_gradient(self) -> None:
        """把某个权重置零后，该分支的梯度必须非零（修复前精确为 0）。"""
        with torch.no_grad():
            self.model.det_weights[1] = 0.0
        self.model.zero_grad()
        log_psi, _ = self.model.forward_log_psi(self.r)
        loss, _ = vmc_surrogate_loss(
            log_psi, torch.randn(6, dtype=torch.float64), clip_method="none"
        )
        loss.backward()
        grads = self.model.det_weights.grad
        self.assertIsNotNone(grads)
        self.assertNotEqual(float(grads[1]), 0.0, "零权重分支的梯度仍为 0，无法恢复")

    def test_zeroed_weight_moves_away_from_zero_under_optimization(self) -> None:
        """从零权重出发做几步优化，该权重必须能离开零点。"""
        with torch.no_grad():
            self.model.det_weights[1] = 0.0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        for _ in range(5):
            optimizer.zero_grad()
            log_psi, _ = self.model.forward_log_psi(self.r)
            loss, _ = vmc_surrogate_loss(
                log_psi, torch.randn(6, dtype=torch.float64), clip_method="none"
            )
            loss.backward()
            optimizer.step()
        self.assertGreater(abs(float(self.model.det_weights[1])), 1e-8)

    def test_all_weights_zero_still_yields_pauli_zero(self) -> None:
        """所有权重为零时波函数必须恒为零（−inf / sign 0），而不是 NaN。"""
        with torch.no_grad():
            self.model.det_weights.zero_()
            log_psi, sign_psi = self.model.forward_log_psi(self.r)
        self.assertFalse(torch.isnan(log_psi).any())
        self.assertFalse(torch.isnan(sign_psi).any())
        self.assertTrue(torch.all(torch.isneginf(log_psi)))
        self.assertTrue(torch.all(sign_psi == 0.0))


class TestClipNoneDoesNotSilentlySubstitute(unittest.TestCase):
    """D8：`clip_local_energy(method='none')` 不得静默替换非有限值。"""

    def test_warns_when_nonfinite_values_are_substituted(self) -> None:
        e = torch.tensor([-1.1, float("nan"), -1.15], dtype=torch.float64)
        with self.assertWarns(UserWarning):
            out, _, _ = clip_local_energy(e, clip_factor=5.0, method="none")
        self.assertTrue(torch.isfinite(out).all(), "替换后应为有限值")

    def test_no_warning_when_all_values_are_finite(self) -> None:
        e = torch.tensor([-1.1, -1.2, -1.15], dtype=torch.float64)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            out, _, _ = clip_local_energy(e, clip_factor=5.0, method="none")
        self.assertTrue(torch.equal(out, e))

    def test_strict_training_path_raises_instead_of_substituting(self) -> None:
        """训练主路径在 clip_method='none' 下仍必须抛错（不得降级为警告）。"""
        log_psi = torch.randn(3, requires_grad=True, dtype=torch.float64)
        e = torch.tensor([-1.0, float("nan"), -2.0], dtype=torch.float64)
        with self.assertRaises(FloatingPointError):
            vmc_surrogate_loss(log_psi, e, clip_method="none")


class TestJVPFallbackVisibility(unittest.TestCase):
    """D9：JVP 求解器只应吞掉"能力不支持"类异常，真实 bug 必须抛出。"""

    def setUp(self) -> None:
        self.model = _h2_model()
        self.system = MolecularSystem(
            self.model.nuclear_coords.clone(), self.model.nuclear_charges.clone(), 1, 1
        )
        self.r = torch.randn(3, 2, 3, dtype=torch.float64, requires_grad=True)

    def _kinetic(self) -> torch.Tensor:
        _, t_local, _ = compute_local_energy(
            self.model, self.system, self.r, laplacian_method="exact_jvp"
        )
        return t_local

    def test_capability_failure_falls_back_with_warning(self) -> None:
        """RuntimeError（算子不支持）应触发带警告的回退，且结果与 exact 一致。"""
        import torch.func as func

        real_jvp = func.jvp

        def boom(*args, **kwargs):
            raise RuntimeError("simulated: operator not supported by vmap")

        with mock.patch.object(func, "jvp", boom):
            with self.assertWarns(UserWarning):
                t_fallback = self._kinetic()
        with mock.patch.object(func, "jvp", real_jvp):
            t_exact = self._kinetic()
        self.assertLess(float((t_fallback - t_exact).abs().max()), 1e-8)

    def test_unexpected_exception_propagates(self) -> None:
        """非能力类异常（如 ValueError）必须向上抛出，不得被静默吞掉。"""
        import torch.func as func

        def boom(*args, **kwargs):
            raise ValueError("simulated: genuine bug, must not be swallowed")

        with mock.patch.object(func, "jvp", boom):
            with self.assertRaises(ValueError):
                self._kinetic()


class TestGeometryConsistencyGuard(unittest.TestCase):
    """D10：模型几何缓冲与 MolecularSystem 失配必须被显式拦截。"""

    def _system(self) -> MolecularSystem:
        return MolecularSystem(
            torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            1,
            1,
        )

    def test_matching_geometry_passes(self) -> None:
        system = self._system()
        model = _h2_model()
        model.check_system_consistency(system)  # 不应抛出

    def test_geometry_updated_after_model_construction_is_detected(self) -> None:
        system = self._system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).double()
        # 构造模型之后更新体系几何 —— 修复前这会静默导致波函数与哈密顿量不一致
        system.update_geometry(
            torch.tensor([[-1.2, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        )
        with self.assertRaises(ValueError):
            model.check_system_consistency(system)

    def test_charge_mismatch_is_detected(self) -> None:
        system = self._system()
        model = _h2_model()
        system.update_geometry(
            system.nuclear_coords.clone(),
            torch.tensor([1.0, 3.0], dtype=torch.float64),
        )
        with self.assertRaises(ValueError):
            model.check_system_consistency(system)


if __name__ == "__main__":
    unittest.main()
