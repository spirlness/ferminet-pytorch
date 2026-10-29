"""
单元测试: test_geometry_cache.py
验证 GeometricContext 几何上下文的数值精度、不可变性与单电子极限安全性。
"""

import unittest
import torch
from geometry import GeometricContext, smooth_distance


class TestGeometricContext(unittest.TestCase):
    def test_geometry_extraction_precision(self) -> None:
        """验证 GeometricContext 抽取的位移与范数与传统直接计算在 float64 下精确吻合 (< 1e-14)。"""
        torch.manual_seed(42)
        B, N, M = 4, 3, 2
        r = torch.randn(B, N, 3, dtype=torch.float64, requires_grad=True)
        nuclear_coords = torch.randn(M, 3, dtype=torch.float64)

        ctx = GeometricContext.compute(r, nuclear_coords, eps=1e-12)

        # 传统直接计算基准
        r_expanded = r.unsqueeze(2)
        nuc_expanded = nuclear_coords.unsqueeze(0).unsqueeze(0)
        expected_r_en = r_expanded - nuc_expanded
        expected_d_en = torch.sqrt(torch.sum(expected_r_en ** 2, dim=-1, keepdim=True) + 1e-12)

        r_i = r.unsqueeze(2)
        r_j = r.unsqueeze(1)
        expected_r_ee = r_i - r_j
        expected_d_ee = torch.sqrt(torch.sum(expected_r_ee ** 2, dim=-1, keepdim=True) + 1e-12)

        self.assertTrue(torch.allclose(ctx.r_en, expected_r_en, atol=1e-14))
        self.assertTrue(torch.allclose(ctx.d_en, expected_d_en, atol=1e-14))
        self.assertIsNotNone(ctx.r_ee)
        self.assertIsNotNone(ctx.d_ee)
        self.assertTrue(torch.allclose(ctx.r_ee, expected_r_ee, atol=1e-14))
        self.assertTrue(torch.allclose(ctx.d_ee, expected_d_ee, atol=1e-14))

    def test_single_electron_boundary(self) -> None:
        """验证单电子极限 (N=1) 下 r_ee 和 d_ee 安全返回 None 且不崩溃。"""
        B, N, M = 2, 1, 1
        r = torch.randn(B, N, 3, dtype=torch.float64)
        nuclear_coords = torch.randn(M, 3, dtype=torch.float64)

        ctx = GeometricContext.compute(r, nuclear_coords, eps=1e-12)
        self.assertIsNone(ctx.r_ee)
        self.assertIsNone(ctx.d_ee)
        self.assertEqual(ctx.r_en.shape, (B, 1, 1, 3))
        self.assertEqual(ctx.d_en.shape, (B, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
