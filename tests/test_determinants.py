"""Regression tests for derivatives of sums containing zero determinants."""

import itertools
import unittest
from unittest.mock import patch

import torch

from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.model import FermiNet


def polynomial_det(matrix: torch.Tensor) -> torch.Tensor:
    """Independent Leibniz reference for the small matrices in these tests."""
    size = matrix.shape[-1]
    result = matrix[..., 0, 0] * 0.0
    for perm in itertools.permutations(range(size)):
        inversions = sum(
            perm[i] > perm[j] for i in range(size) for j in range(i + 1, size)
        )
        term = torch.ones_like(result)
        for i, j in enumerate(perm):
            term = term * matrix[..., i, j]
        result = result + ((-1) ** inversions) * term
    return result


class TestSingularDeterminants(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_zero_branches_match_polynomial_first_and_second_derivatives(self) -> None:
        """Include rank n-1, rank n-2, and both spin factors zero in one term."""
        for size, rank in ((1, 0), (2, 1), (2, 0), (3, 2), (3, 1)):
            for zero_down in (False, True):
                with self.subTest(size=size, rank=rank, zero_down=zero_down):
                    model = FermiNet(
                        torch.zeros(1, 3),
                        torch.ones(1),
                        size,
                        size,
                        num_determinants=2,
                        hidden_dim_one=8,
                        hidden_dim_two=4,
                        num_layers=1,
                    ).double()
                    matrices = torch.eye(size, dtype=torch.float64).repeat(
                        2, 1, 2, 1, 1
                    )
                    matrices[0, 0, 0, rank:, :] = 0.0
                    if zero_down:
                        matrices[1, 0, 0, -1, :] = 0.0
                    matrices.requires_grad_(True)
                    r = torch.zeros(1, 2 * size, 3, dtype=torch.float64)

                    def actual(x):
                        with patch.object(
                            model, "forward_orbitals", return_value=(x[0], x[1])
                        ):
                            return model.forward_log_psi(r)[0].sum()

                    def reference(x):
                        terms = polynomial_det(x[0]) * polynomial_det(x[1])
                        return (terms * model.det_weights).sum().abs().log()

                    torch.testing.assert_close(actual(matrices), reference(matrices))
                    for derivative in (
                        torch.autograd.functional.jacobian,
                        torch.autograd.functional.hessian,
                    ):
                        got = derivative(actual, matrices)
                        expected = derivative(reference, matrices)
                        self.assertTrue(torch.isfinite(got).all())
                        torch.testing.assert_close(got, expected, atol=1e-9, rtol=1e-9)

    def test_regular_branches_keep_logarithmic_dynamic_range(self) -> None:
        """Each spin determinant product underflows in float32, but log Psi does not."""
        model = FermiNet(
            torch.zeros(1, 3),
            torch.ones(1),
            2,
            2,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        )
        orbital = (torch.eye(2).repeat(1, 2, 1, 1) * 1e-20).requires_grad_()
        with patch.object(model, "forward_orbitals", return_value=(orbital, orbital)):
            log_abs, sign = model.forward_log_psi(torch.zeros(1, 4, 3))
        expected = (
            4.0 * torch.tensor(1e-20, dtype=torch.float64).log()
            + model.det_weights.double().sum().log()
        )
        torch.testing.assert_close(
            log_abs.double().squeeze(), expected, atol=1e-5, rtol=1e-6
        )
        self.assertEqual(sign.item(), 1.0)
        self.assertTrue(
            torch.isfinite(torch.autograd.grad(log_abs.sum(), orbital)[0]).all()
        )

    def test_zero_branch_preserves_tiny_total_wavefunction_and_derivatives(
        self,
    ) -> None:
        """Psi = scale**4 * (w0 * theta + w1), evaluated at theta = 0."""
        model = FermiNet(
            torch.zeros(1, 3),
            torch.ones(1),
            2,
            2,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        )
        theta = torch.tensor(0.0, requires_grad=True)
        scale = 1e-20
        regular = torch.eye(2) * scale
        singular = torch.diag(torch.stack((theta, torch.ones_like(theta)))) * scale
        up = torch.stack((singular, regular)).unsqueeze(0)
        down = regular.repeat(1, 2, 1, 1)
        with patch.object(model, "forward_orbitals", return_value=(up, down)):
            log_abs, sign = model.forward_log_psi(torch.zeros(1, 4, 3))
        expected = (
            4.0 * torch.tensor(scale, dtype=torch.float64).log()
            + model.det_weights[1].double().log()
        )
        torch.testing.assert_close(
            log_abs.double().squeeze(), expected, atol=1e-5, rtol=1e-6
        )
        self.assertEqual(sign.item(), 1.0)
        first = torch.autograd.grad(log_abs.sum(), theta, create_graph=True)[0]
        second = torch.autograd.grad(first, theta)[0]
        torch.testing.assert_close(first, torch.tensor(1.0))
        torch.testing.assert_close(second, torch.tensor(-1.0))

    def test_real_model_zero_orbital_has_finite_energy_and_parameter_gradients(
        self,
    ) -> None:
        for dtype in (torch.float32, torch.float64):
            for n_up, n_down in ((1, 0), (0, 1), (2, 2)):
                with self.subTest(dtype=dtype, n_up=n_up, n_down=n_down):
                    system = MolecularSystem(
                        torch.zeros(1, 3, dtype=dtype),
                        torch.ones(1, dtype=dtype),
                        n_up,
                        n_down,
                    )
                    model = FermiNet(
                        system.nuclear_coords,
                        system.nuclear_charges,
                        n_up,
                        n_down,
                        num_determinants=2,
                        hidden_dim_one=8,
                        hidden_dim_two=4,
                        num_layers=1,
                    ).to(dtype=dtype)
                    orbital_layer = (
                        model.linear_orb_up if n_up else model.linear_orb_down
                    )
                    with torch.no_grad():
                        orbital_layer.weight[0].zero_()
                        orbital_layer.bias[0].zero_()
                    r = (
                        torch.randn(2, n_up + n_down, 3, dtype=dtype) + 0.3
                    ).requires_grad_()
                    log_abs, sign = model.forward_log_psi(r)
                    self.assertTrue(torch.isfinite(log_abs).all())
                    self.assertTrue((sign != 0).all())
                    log_abs.sum().backward()
                    self.assertTrue(torch.isfinite(r.grad).all())
                    self.assertTrue(torch.isfinite(orbital_layer.bias.grad).all())
                    self.assertGreater(orbital_layer.bias.grad[0].abs().item(), 0.0)
                    model.zero_grad(set_to_none=True)
                    energy, _, _ = compute_local_energy(
                        model, system, r.detach().requires_grad_(), create_graph=True
                    )
                    self.assertTrue(torch.isfinite(energy).all())
                    energy.sum().backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
