"""Migrated prototype obligations, exercising imported production modules."""
import unittest

import torch

from ferminet_pytorch.layers_modern import FermiInteractionLayerModern
from ferminet_pytorch.psiformer_attention import PsiformerAttentionBlock


class TestProductionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def test_spin_pooling_against_explicit_electron_sums(self):
        for n_up, n_down in ((1, 1), (2, 1), (3, 3), (1, 0), (0, 2)):
            with self.subTest(spins=(n_up, n_down)):
                n = n_up + n_down
                layer = FermiInteractionLayerModern(3, 2, 4, 2, n_up, n_down).double()
                h1 = torch.randn(2, n, 3, dtype=torch.float64, requires_grad=True)
                h2 = torch.randn(2, n, n, 2, dtype=torch.float64, requires_grad=True)
                expected = []
                for i in range(n):
                    projected = layer.proj_h1_self(h1[:, i])
                    for start, end, single, pair in (
                        (0, n_up, layer.proj_h1_mean_up, layer.proj_h2_mean_up),
                        (n_up, n, layer.proj_h1_mean_down, layer.proj_h2_mean_down),
                    ):
                        if start == end:
                            continue
                        mean = sum(h1[:, j] for j in range(start, end)) / (end-start)
                        projected = projected + single(mean)
                        others = [j for j in range(start, end) if j != i]
                        if others:
                            projected = projected + pair(sum(h2[:, i, j] for j in others) / len(others))
                    expected.append(layer.act(projected) + layer.res_proj_single(h1[:, i]))
                expected = torch.stack(expected, dim=1)
                actual = layer(h1, h2)[0]
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                da = torch.autograd.grad(actual.square().sum(), (h1, h2), create_graph=True)
                de = torch.autograd.grad(expected.square().sum(), (h1, h2), create_graph=True,
                                         allow_unused=True)
                for a, e in zip(da, de):
                    torch.testing.assert_close(a, torch.zeros_like(a) if e is None else e,
                                               atol=1e-12, rtol=1e-12)

    def test_layer_second_derivatives_match_finite_differences(self):
        layer = FermiInteractionLayerModern(2, 1, 2, 1, 1, 1).double()
        h1 = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
        h2 = torch.randn(1, 2, 2, 1, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradgradcheck(lambda a, b: layer(a, b)[0], (h1, h2)))

    def test_attention_distinguishes_coincident_opposite_spins(self):
        block = PsiformerAttentionBlock(16, 2, n_up=1, n_down=1)
        h = torch.ones(1, 2, 16)
        out = block(h, torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 1))
        self.assertGreater(torch.linalg.vector_norm(out[:, 0] - out[:, 1]).item(), 1e-3)

    def test_attention_single_electron_is_finite_and_differentiable(self):
        block = PsiformerAttentionBlock(16, 2, n_up=1, n_down=0).double()
        h = torch.randn(2, 1, 16, dtype=torch.float64, requires_grad=True)
        out = block(h)
        self.assertEqual(out.shape, h.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(torch.autograd.grad(out.square().sum(), h)[0]).all())


if __name__ == "__main__":
    unittest.main()
