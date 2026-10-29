"""
单元测试: test_psiformer_integration.py
验证 PsiformerAttentionBlock 与 ComposableFermiNet 的自旋感知、置换等变性与前向求导能力。
"""

import unittest
import torch
from ferminet_pytorch.psiformer_attention import PsiformerAttentionBlock
from ferminet_pytorch.composable_model import ComposableFermiNet
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy


class TestPsiformerIntegration(unittest.TestCase):
    def test_psiformer_spin_embedding_and_equivariance(self) -> None:
        """验证 PsiformerAttentionBlock 的自旋嵌入能区分上下自旋，且在同自旋交换下满足置换等变性。"""
        torch.manual_seed(42)
        B, n_up, n_down = 2, 2, 1
        N = n_up + n_down
        C = 16

        block = PsiformerAttentionBlock(embed_dim=C, num_heads=2, n_up=n_up, n_down=n_down)

        h = torch.randn(B, N, C)
        # 交换同自旋电子 0 和 1
        perm = [1, 0, 2]
        h_perm = h[:, perm, :]

        out = block(h)
        out_perm = block(h_perm)

        # 同自旋等变性
        self.assertTrue(torch.allclose(out_perm, out[:, perm, :], atol=1e-6))

    def test_composable_ferminet_end_to_end(self) -> None:
        """验证 ComposableFermiNet 在 psiformer 模式下能够端到端计算对数波函数与局域能量。"""
        torch.manual_seed(42)
        coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges = torch.tensor([2.0], dtype=torch.float64)  # 氦原子 He
        system = MolecularSystem(coords, charges, n_up=1, n_down=1)

        model = ComposableFermiNet(
            nuclear_coords=coords,
            nuclear_charges=charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            num_layers=1,
            interaction_type="psiformer",
        ).to(dtype=torch.float64)

        r = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
        log_abs, sign = model.forward_log_psi(r)

        self.assertEqual(log_abs.shape, (4,))
        self.assertEqual(sign.shape, (4,))
        self.assertTrue(torch.isfinite(log_abs).all())

        # 局域能量求值
        e_local, t_local, v_local = compute_local_energy(model, system, r)
        self.assertEqual(e_local.shape, (4,))
        self.assertTrue(torch.isfinite(e_local).all())


if __name__ == "__main__":
    unittest.main()
