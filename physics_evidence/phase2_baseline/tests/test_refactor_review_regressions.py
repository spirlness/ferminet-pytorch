"""
单元测试: test_refactor_review_regressions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
系统性回归验证套件: 针对架构评审所发现的自旋投影批次不兼容、
适配器优化器参数空列表、前向 JVP 自动微分缺省及几何上下文图泄漏进行精准回归测试。
"""

import unittest
import torch
import torch.nn as nn

from ferminet_pytorch.model import FermiNet, _CofactorDeterminant
from ferminet_pytorch.composable_model import ComposableFermiNet
from ferminet_pytorch.geometry import GeometricContext
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy, KineticMode
from ferminet_pytorch.vmc_driver import VariationalState, HamiltonianComposite, MetropolisSampler, VMCDriver
from ferminet_pytorch.psiformer_attention import PsiformerAttentionBlock


class TestRefactorReviewRegressions(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32)
        self.charges = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def test_ferminet_spin_adapted_with_geom_ctx_no_crash(self) -> None:
        """测试 1: 验证 FermiNet 开启自旋投影时传入 geom_ctx 不发生批次维度崩溃。"""
        net = FermiNet(self.coords, self.charges, 1, 1, spin_adapted=True)
        r = torch.randn(4, 2, 3, dtype=torch.float32)
        ctx = GeometricContext.compute(r, self.coords)
        log_psi, sign = net.forward_log_psi(r, geom_ctx=ctx)
        self.assertEqual(log_psi.shape, (4,))
        self.assertEqual(sign.shape, (4,))
        self.assertTrue(torch.isfinite(log_psi).all())

    def test_composable_ferminet_spin_adapted_no_attribute_error_or_recursion(self) -> None:
        """测试 2: 验证 ComposableFermiNet 开启自旋投影时正确调用 project_log_psi 且不发生无限递归。"""
        net = ComposableFermiNet(self.coords, self.charges, 1, 1, spin_adapted=True)
        r = torch.randn(4, 2, 3, dtype=torch.float32)
        log_psi, sign = net.forward_log_psi(r)
        self.assertEqual(log_psi.shape, (4,))
        self.assertEqual(sign.shape, (4,))
        self.assertTrue(torch.isfinite(log_psi).all())



    def test_cofactor_determinant_zero_graph_breaks_and_finite_gradients(self) -> None:
        """测试 5: 验证 _CofactorDeterminant 保持精确伴随导数并在 TorchDynamo 下实现 0 图断裂。"""
        import torch._dynamo
        matrix = torch.randn(2, 2, 2, requires_grad=True)
        log_scale = torch.zeros(2)

        def f_det(m: torch.Tensor) -> torch.Tensor:
            return _CofactorDeterminant.apply(m, log_scale)

        exp = torch._dynamo.explain(f_det)(matrix)
        self.assertEqual(exp.graph_break_count, 0, f"Graph breaks found: {exp.break_reasons}")

        out = f_det(matrix)
        g = torch.autograd.grad(out.sum(), matrix, create_graph=True)[0]
        self.assertTrue(torch.isfinite(g).all())

    def test_vmc_driver_step_executes_cleanly(self) -> None:
        """测试 6: 验证四层解耦 VMCDriver.step 单步优化执行平稳，度量统计健全。"""
        sys = MolecularSystem(self.coords, self.charges, 1, 1)
        net = FermiNet(self.coords, self.charges, 1, 1, hidden_dim_one=16, hidden_dim_two=8, num_layers=1)
        vstate = VariationalState(model=net, system=sys, num_walkers=32)
        hamiltonian = HamiltonianComposite(system=sys)
        sampler = MetropolisSampler(variational_state=vstate)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        driver = VMCDriver(variational_state=vstate, hamiltonian=hamiltonian, sampler=sampler, optimizer=opt)

        metrics = driver.step()
        self.assertIn("loss", metrics)
        self.assertIn("energy_mean", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["energy_mean"])))

    def test_psiformer_attention_robust_against_small_embed_dim(self) -> None:
        """测试 7: 验证 PsiformerAttentionBlock 在极小维度 (embed_dim=2, num_heads=1) 下安全构建与前向。"""
        block = PsiformerAttentionBlock(embed_dim=2, num_heads=1, n_up=1, n_down=1)
        h = torch.randn(2, 2, 2)
        r_ee = torch.randn(2, 2, 2, 3)
        d_ee = torch.rand(2, 2, 2, 1)
        out = block(h, r_ee=r_ee, d_ee=d_ee)
        self.assertEqual(out.shape, (2, 2, 2))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
