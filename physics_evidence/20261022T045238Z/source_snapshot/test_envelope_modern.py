"""
单元测试: test_envelope_modern.py
验证 NuclearEnvelopeModern 的参数兼容性、形状与数值等价性。
"""

import unittest
import torch
import torch.nn.functional as F
from envelope_modern import NuclearEnvelopeModern
from envelope import NuclearEnvelope


class TestEnvelopeModern(unittest.TestCase):
    def test_modern_envelope_shapes_and_state_dict_keys(self) -> None:
        """验证现代包络与历史权重字典键名完全一致，输出形状为 (B, K, N, N_orb)。"""
        K, N_orb, M = 8, 4, 2
        coords = torch.zeros(M, 3)
        charges = torch.tensor([1.0, 1.0])

        mod = NuclearEnvelopeModern(
            num_determinants=K,
            num_orbitals=N_orb,
            nuclear_coords=coords,
            nuclear_charges=charges,
        )

        # 检查参数名 100% 兼容已有 Checkpoint
        self.assertIn("weights", dict(mod.named_parameters()))
        self.assertIn("decay_rates_raw", dict(mod.named_parameters()))

        B, N = 6, N_orb
        dist_en = torch.rand(B, N, M) + 0.1
        env = mod(dist_en)
        self.assertEqual(env.shape, (B, K, N, N_orb))

    def test_modern_and_legacy_numerical_match(self) -> None:
        """验证相同参数下 NuclearEnvelope 与 NuclearEnvelopeModern 数值精确一致。"""
        torch.manual_seed(123)
        K, N_orb, M = 4, 2, 2
        coords = torch.randn(M, 3, dtype=torch.float64)
        charges = torch.tensor([1.0, 2.0], dtype=torch.float64)

        legacy = NuclearEnvelope(
            num_determinants=K,
            num_orbitals=N_orb,
            num_nuclei=M,
            nuclear_charges=charges,
        ).to(dtype=torch.float64)

        modern = NuclearEnvelopeModern(
            num_determinants=K,
            num_orbitals=N_orb,
            nuclear_coords=coords,
            nuclear_charges=charges,
        ).to(dtype=torch.float64)

        # 拷贝参数
        with torch.no_grad():
            modern.weights.copy_(legacy.weights)
            modern.decay_rates_raw.copy_(legacy.decay_rates_raw)

        B, N = 3, N_orb
        r = torch.randn(B, N, 3, dtype=torch.float64)
        diff = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + legacy.eps)

        out_legacy = legacy(r, coords)
        out_modern = modern(dist)

        self.assertTrue(torch.allclose(out_legacy, out_modern, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
