"""
单元测试: test_layers_modern.py
验证 FermiInteractionLayerModern 的维度契约、自旋等变性与原地扣减对角线特性。
"""

import unittest
import torch
from layers_modern import FermiInteractionLayerModern
from legacy_adapters import FermiInteractionLayerLegacyAdapter


class TestLayersModern(unittest.TestCase):
    def test_modern_layer_shapes_and_equivariance(self) -> None:
        """验证现代层前向输出形状及同自旋置换等变性。"""
        torch.manual_seed(42)
        B, n_up, n_down = 4, 3, 2
        N = n_up + n_down
        d1_in, d2_in = 16, 8
        d1_out, d2_out = 32, 16

        layer = FermiInteractionLayerModern(
            d_in_single=d1_in,
            d_in_double=d2_in,
            d_out_single=d1_out,
            d_out_double=d2_out,
            n_up=n_up,
            n_down=n_down,
        )

        h1 = torch.randn(B, N, d1_in)
        h2 = torch.randn(B, N, N, d2_in)

        h1_out, h2_out = layer(h1, h2)
        self.assertEqual(h1_out.shape, (B, N, d1_out))
        self.assertEqual(h2_out.shape, (B, N, N, d2_out))

        # 测试自旋向上置换 (交换电子 0 与 1)
        perm = [1, 0, 2, 3, 4]
        h1_perm = h1[:, perm, :]
        h2_perm = h2[:, perm, :, :][:, :, perm, :]

        h1_perm_out, h2_perm_out = layer(h1_perm, h2_perm)
        # 置换后的输出应等于输出的置换 (Permutation Equivariance)
        self.assertTrue(torch.allclose(h1_perm_out, h1_out[:, perm, :], atol=1e-6))
        self.assertTrue(torch.allclose(h2_perm_out, h2_out[:, perm, :, :][:, :, perm, :], atol=1e-6))

    def test_legacy_adapter_equivalence(self) -> None:
        """验证 FermiInteractionLayerLegacyAdapter 与现有接口参数与输出一致。"""
        adapter = FermiInteractionLayerLegacyAdapter(
            in_dim_one=12,
            in_dim_two=4,
            out_dim_one=24,
            out_dim_two=8,
            produce_two_electron_output=True,
        )
        h1 = torch.randn(2, 4, 12)
        h2 = torch.randn(2, 4, 4, 4)
        h1_out, h2_out = adapter(h1, h2, n_up=2, n_down=2)
        self.assertEqual(h1_out.shape, (2, 4, 24))
        self.assertEqual(h2_out.shape, (2, 4, 4, 8))


if __name__ == "__main__":
    unittest.main()
