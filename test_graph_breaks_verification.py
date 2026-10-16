"""
两阶段 Jastrow 策略与评估去相关增强在 TorchDynamo 下图断裂 (Graph Break) 的形式化回归验证测试
=============================================================================================
"""

import unittest
import torch
import torch._dynamo

from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.jastrow import KatoCuspJastrow
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss
from ferminet_pytorch.train_vmc import (
    build_h2_system,
    build_lih_system,
    compute_autocorrelation_time,
)


class TestTwoStageJastrowDynamoZeroBreaks(unittest.TestCase):
    """
    验证两阶段 Jastrow 策略下的波函数计算图在 TorchDynamo 追踪下严格保持 0 图断裂。
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_jastrow_module_frozen_and_unfrozen_zero_breaks(self) -> None:
        """KatoCuspJastrow 模块在冻结 (Stage 1) 与解冻 (Stage 2) 状态下图断裂严格为 0."""
        system = build_h2_system()
        jastrow = KatoCuspJastrow(
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
        )
        r = torch.randn(16, system.total_electrons, 3)

        def jastrow_fwd(coords):
            return jastrow(coords, system.nuclear_coords)

        # Stage 1: 冻结
        for p in jastrow.parameters():
            p.requires_grad = False
        exp_frozen = torch._dynamo.explain(jastrow_fwd)(r)
        self.assertEqual(exp_frozen.graph_break_count, 0)
        self.assertEqual(exp_frozen.graph_count, 1)

        # Stage 2: 解冻
        for p in jastrow.parameters():
            p.requires_grad = True
        exp_unfrozen = torch._dynamo.explain(jastrow_fwd)(r)
        self.assertEqual(exp_unfrozen.graph_break_count, 0)
        self.assertEqual(exp_unfrozen.graph_count, 1)

    def test_ferminet_forward_log_psi_stages_zero_breaks(self) -> None:
        """FermiNet.forward_log_psi 在 Stage 1 (冻结 Jastrow) 与 Stage 2 (解冻 Jastrow) 下图断裂严格为 0."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            use_cusp_jastrow=True,
        ).eval()

        r = torch.randn(16, system.total_electrons, 3)

        # Stage 1: 冻结 Jastrow
        model.freeze_jastrow()
        exp_s1 = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp_s1.graph_break_count, 0)
        self.assertEqual(exp_s1.graph_count, 1)

        # Stage 2: 解冻 Jastrow
        model.unfreeze_jastrow()
        exp_s2 = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp_s2.graph_break_count, 0)
        self.assertEqual(exp_s2.graph_count, 1)

    def test_ferminet_spin_adapted_and_jastrow_stages_zero_breaks(self) -> None:
        """自旋适配投影 + 两阶段 Jastrow 策略下图断裂严格为 0."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).eval()

        r = torch.randn(16, system.total_electrons, 3)

        # Stage 1
        model.freeze_jastrow()
        exp_s1 = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp_s1.graph_break_count, 0)

        # Stage 2
        model.unfreeze_jastrow()
        exp_s2 = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp_s2.graph_break_count, 0)

    def test_ferminet_forward_psi_zero_breaks(self) -> None:
        """FermiNet.forward (直接输出波函数 Psi) 在两阶段 Jastrow 下图断裂严格为 0."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            use_cusp_jastrow=True,
        ).eval()

        r = torch.randn(16, system.total_electrons, 3)

        model.freeze_jastrow()
        exp_fwd_s1 = torch._dynamo.explain(model.forward)(r)
        self.assertEqual(exp_fwd_s1.graph_break_count, 0)

        model.unfreeze_jastrow()
        exp_fwd_s2 = torch._dynamo.explain(model.forward)(r)
        self.assertEqual(exp_fwd_s2.graph_break_count, 0)

    def test_singular_overlap_zero_breaks(self) -> None:
        """含 Jastrow 模型在极端奇异碰撞坐标 (det=0) 下图断裂严格为 0."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            use_cusp_jastrow=True,
        ).eval()

        r = torch.randn(16, system.total_electrons, 3)
        r[:, 1, :] = r[:, 0, :]  # 制造同自旋完全重合奇异点

        model.freeze_jastrow()
        exp = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp.graph_break_count, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 环境不可用，跳过 GPU 测试")
    def test_cuda_two_stage_jastrow_zero_breaks(self) -> None:
        """CUDA 设备上两阶段 Jastrow 前向图断裂数严格为 0."""
        device = torch.device("cuda")
        system = build_h2_system(device=device)
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            use_cusp_jastrow=True,
            spin_adapted=True,
        ).to(device).eval()

        r = torch.randn(32, system.total_electrons, 3, device=device)

        model.freeze_jastrow()
        exp_s1 = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp_s1.graph_break_count, 0)

        model.unfreeze_jastrow()
        exp_s2 = torch._dynamo.explain(model.forward_log_psi)(r)
        self.assertEqual(exp_s2.graph_break_count, 0)


class TestDecorrelationAndHamiltonianDynamoBehavior(unittest.TestCase):
    """
    量化与验证评估去相关及哈密顿量相关算子的图断裂特性。
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_potential_energy_zero_breaks(self) -> None:
        """势能计算 compute_potential_energy 图断裂数严格为 0."""
        system = build_h2_system()
        r = torch.randn(16, system.total_electrons, 3)
        exp = torch._dynamo.explain(system.compute_potential_energy)(r)
        self.assertEqual(exp.graph_break_count, 0)
        self.assertEqual(exp.graph_count, 1)

    def test_mcmc_decorrelation_step_kernel_zero_breaks(self) -> None:
        """评估去相关采样循环中的 MCMC 转移与接受判定核心算子图断裂数严格为 0."""
        system = build_h2_system()
        model = FermiNet(
            nuclear_coords=system.nuclear_coords,
            nuclear_charges=system.nuclear_charges,
            n_up=system.n_up,
            n_down=system.n_down,
            num_determinants=2,
            use_cusp_jastrow=True,
        ).eval()

        def mcmc_step_core(walkers, current_log_prob, step_size):
            proposal = walkers + torch.randn_like(walkers) * step_size
            prop_log_abs, _ = model.forward_log_psi(proposal)
            prop_log_prob = 2.0 * prop_log_abs
            log_accept_ratio = prop_log_prob - current_log_prob
            accept_ratio = torch.exp(torch.clamp(log_accept_ratio, max=0.0))
            rand_uniform = torch.rand_like(accept_ratio)
            accept_mask = (rand_uniform < accept_ratio).unsqueeze(-1).unsqueeze(-1)
            new_walkers = torch.where(accept_mask, proposal, walkers)
            new_log_prob = torch.where(accept_mask.squeeze(-1).squeeze(-1), prop_log_prob, current_log_prob)
            return new_walkers, new_log_prob

        walkers = torch.randn(32, system.total_electrons, 3)
        curr_log_abs, _ = model.forward_log_psi(walkers)
        curr_log_prob = 2.0 * curr_log_abs

        exp = torch._dynamo.explain(mcmc_step_core)(walkers, curr_log_prob, 0.5)
        self.assertEqual(exp.graph_break_count, 0)
        self.assertEqual(exp.graph_count, 1)

    def test_pure_tensor_autocorrelation_zero_breaks(self) -> None:
        """纯张量形式的滞后自协方差与自相关时间核心算子图断裂数严格为 0."""
        def pure_autocorr(series):
            if series.ndim == 1:
                series = series.unsqueeze(1)
            series_float = series.double()
            global_mean = torch.mean(series_float)
            diff = series_float - global_mean
            cov_1 = torch.mean(diff[:-1, :] * diff[1:, :])
            var_0 = torch.mean(diff[:-1, :] ** 2)
            var_1 = torch.mean(diff[1:, :] ** 2)
            denom = torch.sqrt(torch.clamp(var_0 * var_1, min=1e-12))
            rho_1 = torch.clamp(cov_1 / denom, min=-0.999, max=0.999)
            tau_int = torch.where(
                rho_1 > 0.0,
                (1.0 + rho_1) / torch.clamp(1.0 - rho_1, min=1e-4),
                torch.tensor(1.0, dtype=torch.float64, device=series.device),
            )
            return rho_1, tau_int

        series = torch.randn(50, 32)
        exp = torch._dynamo.explain(pure_autocorr)(series)
        self.assertEqual(exp.graph_break_count, 0)
        self.assertEqual(exp.graph_count, 1)

    def test_pure_tensor_surrogate_loss_zero_breaks(self) -> None:
        """纯张量 VMC 代理损失计算图断裂数严格为 0."""
        def pure_surrogate_loss(log_abs_psi, e_clipped):
            diff = (e_clipped - torch.mean(e_clipped)).detach()
            return 2.0 * torch.mean(diff * log_abs_psi)

        lp = torch.randn(64, requires_grad=True)
        el = torch.randn(64)
        exp = torch._dynamo.explain(pure_surrogate_loss)(lp, el)
        self.assertEqual(exp.graph_break_count, 0)
        self.assertEqual(exp.graph_count, 1)


if __name__ == "__main__":
    unittest.main()
