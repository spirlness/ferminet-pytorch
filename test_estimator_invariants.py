"""
能量估计器与势能数值不变量回归测试 (单元 3 专属)
================================================
固化 D2/D6 修复的可执行不变量，与 test_physics_invariants.py 互补:

1. D6 势能: clamp 核在所有 r^2 >= eps 处恒等，势能必须与朴素 sqrt 手算逐位相等；
   重合点有限、梯度为零；bare sqrt 核在 r=0 处梯度为 NaN (记录"为何不能删 eps")。
2. D2 Hutchinson 守卫: 请求 hutchinson 求解器时必须发出 UserWarning。
3. D2 solver provenance: stats 字典必须记录局域能量实际使用的拉普拉斯方法与探针数。
"""

import os
import sys
import unittest
import warnings

# 确保包能够被直接运行和识别
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.loss import VMCLoss


def _build_model(seed: int = 7, n_up: int = 1, n_down: int = 1) -> FermiNet:
    """构造 H2 几何下的双精度小模型，随机种子固定以保证可复现。"""
    torch.manual_seed(seed)
    coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return FermiNet(
        nuclear_coords=coords,
        nuclear_charges=charges,
        n_up=n_up,
        n_down=n_down,
        num_determinants=2,
        hidden_dim_one=16,
        hidden_dim_two=8,
        num_layers=1,
    ).double()


def _build_system() -> MolecularSystem:
    """标准 H2 体系: 两质子间距 1.4 Bohr，各带电荷 +1。"""
    coords = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return MolecularSystem(nuclear_coords=coords, nuclear_charges=charges, n_up=1, n_down=1)


class TestPotentialBitwiseExactness(unittest.TestCase):
    """D6: clamp 核势能在非重合点必须与朴素未软化计算逐位相等。"""

    def test_potential_bitwise_equal_to_naive_sqrt_formula(self) -> None:
        """随机构型上 compute_potential_energy(r) 必须与朴素 torch.sqrt 手算逐位相等。

        clamp 在所有 r^2 >= eps 处为恒等变换，故距离与未软化实现逐位相同，
        偏置精确为零 (加性 +eps 核则处处存在相对偏差 eps/(2 r^2))。
        所有电子-核与电子-电子距离均 >= sqrt(eps)，clamp 从不触发。
        """
        torch.manual_seed(11)
        system = _build_system()
        # 保证最小距离远大于 sqrt(eps)=1e-12: 把每个电子钉在远离核的位置
        r = 0.3 + 0.5 * torch.rand(8, 2, 3, dtype=torch.float64)
        v_impl = system.compute_potential_energy(r)

        # 朴素手算 (未软化): V_en + V_ee + V_nn
        coords = system.nuclear_coords
        charges = system.nuclear_charges
        diff_en = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)   # (B, N, M, 3)
        dist_en = torch.sqrt(torch.sum(diff_en ** 2, dim=-1))         # (B, N, M)
        v_en_naive = -torch.sum(charges / dist_en, dim=(-1, -2))
        diff_ee = r.unsqueeze(2) - r.unsqueeze(1)                     # (B, N, N, 3)
        dist_ee = torch.sqrt(torch.sum(diff_ee ** 2, dim=-1))         # (B, N, N)
        mask = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
        v_ee_naive = torch.sum(torch.where(mask, 1.0 / dist_ee, torch.zeros(1)), dim=(-1, -2))
        v_nn_naive = (charges[0] * charges[1] / torch.linalg.vector_norm(coords[0] - coords[1])).expand(v_en_naive.shape)
        v_naive = v_en_naive + v_ee_naive + v_nn_naive

        # 逐位相等 (bit-for-bit): 同一硬件上同一算子序列的确定性结果
        self.assertTrue(
            torch.equal(v_impl, v_naive),
            f"最大逐位差异: {(v_impl - v_naive).abs().max().item()}",
        )
        # clamp 从未触发的自证: 最小平方距离 (i<j 对) 远超 eps
        r2_ee_all = torch.sum(diff_ee ** 2, dim=-1)                   # (B, N, N)
        r2_min = float(torch.where(mask, r2_ee_all, torch.full_like(r2_ee_all, float("inf"))).min())
        self.assertGreater(r2_min, 1e-24)

    def test_bare_sqrt_kernel_nan_gradient_at_zero_clamp_is_safe(self) -> None:
        """记录"为何不能直接删 eps": 裸 sqrt(r^2) 在 r=0 处梯度为 NaN，clamp 核不会。

        以 diff 为叶子节点 (与实现一致): r^2 = sum(diff^2) 在 r=0 处梯度为 0，
        d(sqrt)/dr^2 = inf，链式法则给 inf * 0 = NaN。clamp 核给有限值与零梯度。
        """
        eps = 1e-24

        # 裸 sqrt 核: 重合点梯度 NaN
        diff_bare = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        r2_bare = torch.sum(diff_bare ** 2, dim=-1)
        torch.sqrt(r2_bare).sum().backward()
        self.assertTrue(torch.isnan(diff_bare.grad).all(), "裸 sqrt 在 r=0 处梯度应为 NaN")

        # 裸 1/sqrt 核 (V_en/V_ee 结构): 数值 inf，梯度 NaN
        diff_inv = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        (1.0 / torch.sqrt(torch.sum(diff_inv ** 2, dim=-1))).sum().backward()
        self.assertTrue(torch.isnan(diff_inv.grad).all(), "裸 1/sqrt 在 r=0 处梯度应为 NaN")

        # clamp 核: 势能有限 (-1/sqrt(eps)) 且梯度严格为零
        diff_clamp = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        dist_clamp = torch.sqrt(torch.clamp(torch.sum(diff_clamp ** 2, dim=-1), min=eps))
        v_clamp = dist_clamp.sum()
        v_clamp.backward()
        self.assertAlmostEqual(v_clamp.item(), eps ** 0.5, places=18)
        self.assertTrue(torch.isfinite(diff_clamp.grad).all())
        self.assertLess(float(diff_clamp.grad.abs().max()), 1e-6)

        # clamp 核 + 1/dist 结构 (V_en 精确形式): 有限且梯度为零
        diff_en = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        dist_en = torch.sqrt(torch.clamp(torch.sum(diff_en ** 2, dim=-1), min=eps))
        v_en = (1.0 / dist_en).sum()
        v_en.backward()
        self.assertAlmostEqual(v_en.item(), 1.0 / (eps ** 0.5), places=10)
        self.assertTrue(torch.isfinite(diff_en.grad).all())
        self.assertLess(float(diff_en.grad.abs().max()), 1e-3)


class TestHutchinsonGuardAndProvenance(unittest.TestCase):
    """D2: Hutchinson 守卫告警与 solver provenance 语义。"""

    def setUp(self) -> None:
        self.system = _build_system()
        self.model = _build_model(seed=21)
        self.r = torch.randn(1, 2, 3, dtype=torch.float64)

    def test_hutchinson_request_emits_user_warning(self) -> None:
        """请求 laplacian_method='hutchinson' 时必须发出 UserWarning (不可用于能量报告)。"""
        r_grad = self.r.clone().detach().requires_grad_(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_local_energy(
                self.model, self.system, r_grad, laplacian_method="hutchinson"
            )
        self.assertTrue(
            any(issubclass(w.category, UserWarning) for w in caught),
            f"未捕获 UserWarning，实际捕获: {[str(w.category) for w in caught]}",
        )
        # 告警应说明不可用于对外报告能量
        messages = " ".join(str(w.message) for w in caught if issubclass(w.category, UserWarning))
        self.assertIn("不可用于对外报告", messages)

    def test_exact_method_emits_no_hutchinson_warning(self) -> None:
        """默认 exact 路径不得发出 hutchinson 守卫告警 (避免训练日志噪音)。"""
        r_grad = self.r.clone().detach().requires_grad_(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_local_energy(self.model, self.system, r_grad, laplacian_method="exact")
        guard = [w for w in caught if issubclass(w.category, UserWarning) and "hutchinson" in str(w.message)]
        self.assertEqual(guard, [])

    def test_stats_record_laplacian_method_hutchinson(self) -> None:
        """compute_step_loss 的 stats 必须记录 hutchinson 方法与实际探针数。"""
        vmc_loss = VMCLoss(clip_method="tanh", laplacian_method="hutchinson", num_hutchinson_samples=4)
        walkers = torch.randn(4, 2, 3, dtype=torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss, stats, e_local = vmc_loss.compute_step_loss(self.model, self.system, walkers)
        self.assertEqual(stats["laplacian_method"], "hutchinson")
        self.assertEqual(stats["hutchinson_samples"], 4)
        self.assertTrue(torch.isfinite(loss))

    def test_stats_record_laplacian_method_exact(self) -> None:
        """默认 exact 路径: stats 记录 'exact'，hutchinson_samples 为 0。"""
        vmc_loss = VMCLoss(clip_method="tanh")
        walkers = torch.randn(4, 2, 3, dtype=torch.float64)
        _, stats, _ = vmc_loss.compute_step_loss(self.model, self.system, walkers)
        self.assertEqual(stats["laplacian_method"], "exact")
        self.assertEqual(stats["hutchinson_samples"], 0)

    def test_stats_provenance_reflects_per_call_override(self) -> None:
        """compute_step_loss 的 per-call 覆盖必须反映进 stats (而非仅实例默认)。"""
        vmc_loss = VMCLoss(clip_method="tanh", laplacian_method="exact")
        walkers = torch.randn(4, 2, 3, dtype=torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stats_h, _ = vmc_loss.compute_step_loss(
                self.model, self.system, walkers,
                laplacian_method="hutchinson", num_hutchinson_samples=2,
            )
        self.assertEqual(stats_h["laplacian_method"], "hutchinson")
        self.assertEqual(stats_h["hutchinson_samples"], 2)

    def test_vmc_surrogate_loss_provenance_defaults_backward_compatible(self) -> None:
        """公开 API vmc_surrogate_loss 不传 provenance 参数时默认 'exact' (向后兼容)。"""
        from ferminet_pytorch.loss import vmc_surrogate_loss

        log_psi = torch.randn(4, dtype=torch.float64, requires_grad=True)
        e_local = torch.randn(4, dtype=torch.float64)
        _, stats = vmc_surrogate_loss(log_psi, e_local)
        self.assertEqual(stats["laplacian_method"], "exact")
        self.assertEqual(stats["hutchinson_samples"], 0)


if __name__ == "__main__":
    unittest.main()
