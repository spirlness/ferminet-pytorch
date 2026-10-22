"""
单元测试: test_unified_loss.py
验证单次前向协议下的 VMCLoss 与 VMCDriver 变分梯度的正确性。
"""

import unittest
import torch
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.vmc_driver import VariationalState, HamiltonianComposite, MetropolisSampler, VMCDriver


class TestUnifiedLoss(unittest.TestCase):
    def test_single_forward_gradient_match(self) -> None:
        """验证单次前向代理损失计算出的模型参数梯度与原二次前向方法严格相等。"""
        torch.manual_seed(42)
        coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float64)
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64)
        system = MolecularSystem(coords, charges, n_up=1, n_down=1)

        model = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        ).to(dtype=torch.float64)

        loss_module = VMCLoss()
        walkers = torch.randn(8, 2, 3, dtype=torch.float64)

        # 方式 1: 经典方式 (二次前向)
        model.zero_grad()
        loss_legacy, _, _ = loss_module.compute_step_loss(model, system, walkers)
        loss_legacy.backward()
        grads_legacy = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        # 方式 2: 单次前向方式 (外部提供一次前向计算)
        model.zero_grad()
        r_for_energy = walkers.clone().detach().requires_grad_(True)
        e_local, _, _ = compute_local_energy(model, system, r_for_energy)
        log_abs_psi, _ = model.forward_log_psi(walkers.detach())
        loss_single, _ = loss_module(log_abs_psi, e_local.detach())
        loss_single.backward()
        grads_single = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        self.assertAlmostEqual(loss_legacy.item(), loss_single.item(), places=8)
        for g_leg, g_sin in zip(grads_legacy, grads_single):
            self.assertTrue(torch.allclose(g_leg, g_sin, atol=1e-10))

    def test_vmc_driver_step(self) -> None:
        """验证 VMCDriver 四层解耦架构能够正常执行单步迭代并输出完整统计指标。"""
        coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        charges = torch.tensor([1.0], dtype=torch.float32)
        system = MolecularSystem(coords, charges, n_up=1, n_down=0)

        model = FermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=1,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=4,
            num_layers=1,
        )

        vstate = VariationalState(model=model, system=system, num_walkers=32)
        hamiltonian = HamiltonianComposite(system=system)
        sampler = MetropolisSampler(variational_state=vstate, step_size=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        driver = VMCDriver(
            variational_state=vstate,
            hamiltonian=hamiltonian,
            sampler=sampler,
            optimizer=optimizer,
            decorrelation_steps=1,
        )

        metrics = driver.step()
        self.assertIn("energy_mean", metrics)
        self.assertIn("mcmc_acceptance", metrics)
        self.assertIn("epoch_time", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["energy_mean"])))


if __name__ == "__main__":
    unittest.main()
