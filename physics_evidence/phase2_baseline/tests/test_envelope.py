"""Independent value and derivative checks for the sole envelope implementation."""
import unittest

import torch
from torch.nn import functional as F

from ferminet_pytorch.envelope import NuclearEnvelope


class TestNuclearEnvelope(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_values_coordinate_and_parameter_derivatives_against_scalar_sum(self):
        for kind in ("exponential", "gaussian"):
            with self.subTest(kind=kind):
                env = NuclearEnvelope(2, 2, 2, envelope_type=kind).double()
                r = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
                nuclei = torch.tensor([[0., 0., -.7], [0., 0., .7]], dtype=r.dtype)
                distance = torch.linalg.vector_norm(r[:, :, None] - nuclei, dim=-1)
                actual = env(distance)
                # Explicit scalar sums are a small mathematical oracle, not a
                # second trainable envelope class or production broadcast path.
                alpha = F.softplus(env.decay_rates_raw)
                expected = torch.stack([
                    sum(env.weights[k, o, m] * torch.exp(
                        -alpha[k, o, m] * distance[b, i, m] ** (2 if kind == "gaussian" else 1))
                        for m in range(2))
                    for b in range(2) for k in range(2)
                    for i in range(2) for o in range(2)
                ]).reshape(2, 2, 2, 2)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                inputs = (r, env.weights, env.decay_rates_raw)
                da = torch.autograd.grad(actual.sum(), inputs, create_graph=True)
                de = torch.autograd.grad(expected.sum(), inputs, create_graph=True)
                for a, e in zip(da, de):
                    torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)
                d2a = torch.autograd.grad(da[0].square().sum(), inputs, retain_graph=True)
                d2e = torch.autograd.grad(de[0].square().sum(), inputs)
                for a, e in zip(d2a, d2e):
                    torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)

    def test_radial_derivatives_have_analytic_limits(self):
        for kind in ("exponential", "gaussian"):
            env = NuclearEnvelope(1, 1, 1, init_decay_rate=2., envelope_type=kind).double()
            with torch.no_grad():
                env.weights.fill_(1.)
            d = torch.zeros(1, 1, 1, dtype=torch.float64, requires_grad=True)
            value = env(d).sum()
            first = torch.autograd.grad(value, d, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), d)[0]
            alpha = F.softplus(env.decay_rates_raw).item()
            self.assertAlmostEqual(value.item(), 1., places=12)
            self.assertAlmostEqual(first.item(), -alpha if kind == "exponential" else 0., places=12)
            self.assertAlmostEqual(second.item(), alpha**2 if kind == "exponential" else -2*alpha, places=12)

    def test_charge_initialization_and_parameter_roundtrip(self):
        charges = torch.tensor([1., 8., 100.], dtype=torch.float64)
        env = NuclearEnvelope(2, 3, 3, nuclear_charges=charges).double()
        torch.testing.assert_close(F.softplus(env.decay_rates_raw)[0, 0], charges, atol=1e-6, rtol=1e-6)
        other = NuclearEnvelope(2, 3, 3).double()
        other.load_state_dict(env.state_dict())
        d = torch.rand(2, 4, 3, dtype=torch.float64)
        torch.testing.assert_close(other(d), env(d), atol=0, rtol=0)

    def test_distance_and_configuration_contracts(self):
        env = NuclearEnvelope(2, 1, 2)
        self.assertEqual(env(torch.ones(3, 1, 2)).shape, (3, 2, 1, 1))
        for shape in ((3, 1, 2, 1), (3, 1, 3), (2,)):
            with self.assertRaises(ValueError):
                env(torch.ones(shape))
        for kwargs in ({"envelope_type": "unknown"}, {"init_decay_rate": 0.},
                       {"nuclear_charges": torch.ones(3)}):
            with self.assertRaises(ValueError):
                NuclearEnvelope(2, 1, 2, **kwargs)


if __name__ == "__main__":
    unittest.main()
