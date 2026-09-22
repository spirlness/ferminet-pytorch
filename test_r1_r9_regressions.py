"""Focused counterexample regressions for the R1--R9 correctness fixes."""

import itertools
import math
import unittest
from collections import OrderedDict
from unittest import mock

import torch
import torch.nn as nn

from features import ElectronFeatureEmbedding
from hamiltonian import MolecularSystem, compute_local_energy
from mcmc import MCMCSampler
from model import FermiNet, _determinant_polynomial
from train_vmc import evaluate_trained_model


def _permutation_determinant(matrix: torch.Tensor) -> torch.Tensor:
    """Independent small-matrix polynomial reference, deliberately not LU."""
    size = matrix.shape[-1]
    value = torch.zeros_like(matrix[..., 0, 0])
    for permutation in itertools.permutations(range(size)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(size) for j in range(i + 1, size)
        )
        term = torch.ones_like(value)
        for row, column in enumerate(permutation):
            term = term * matrix[..., row, column]
        value = value - term if inversions % 2 else value + term
    return value


def _directional_derivatives(function, matrix: torch.Tensor, direction: torch.Tensor):
    """Return value and first through third directional derivatives."""
    value = function(matrix).sum()
    first = torch.autograd.grad(value, matrix, create_graph=True)[0]
    second_scalar = (first * direction).sum()
    if not second_scalar.requires_grad:
        zero = torch.zeros_like(matrix)
        return value.detach(), first.detach(), zero, zero
    second = torch.autograd.grad(second_scalar, matrix, create_graph=True)[0]
    third_scalar = (second * direction).sum()
    third = torch.autograd.grad(third_scalar, matrix)[0] if third_scalar.requires_grad else torch.zeros_like(matrix)
    return value.detach(), first.detach(), second.detach(), third.detach()


class _SingularBranchWavefunction(nn.Module):
    """Psi=x+1 written as a two-determinant sum with one singular branch at x=0."""

    def forward_log_psi(self, r: torch.Tensor):
        x = r[:, 0, 0]
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        singular_branch = torch.stack(
            (torch.stack((x, one), dim=-1), torch.stack((zero, one), dim=-1)), dim=-2
        )
        regular_branch = torch.stack(
            (torch.stack((one, zero), dim=-1), torch.stack((zero, one), dim=-1)), dim=-2
        )
        psi = _determinant_polynomial(torch.stack((singular_branch, regular_branch), dim=1)).sum(dim=1)
        return torch.log(torch.abs(psi)), torch.sign(psi)


class _ConstantLogWavefunction(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float64), requires_grad=False)

    def forward_log_psi(self, r: torch.Tensor):
        values = self.value.expand(r.shape[0])
        return values, torch.ones_like(values)


class TestR2SingularDeterminants(unittest.TestCase):
    def test_polynomial_matches_independent_reference_through_third_order(self) -> None:
        torch.manual_seed(231)
        for size in range(1, 5):
            for singular in (False, True):
                with self.subTest(size=size, singular=singular):
                    matrix = torch.randn(size, size, dtype=torch.float64)
                    if singular and size > 1:
                        matrix[-1] = matrix[0]
                    matrix.requires_grad_()
                    direction = torch.randn_like(matrix)
                    actual = _directional_derivatives(_determinant_polynomial, matrix, direction)
                    matrix_ref = matrix.detach().clone().requires_grad_()
                    expected = _directional_derivatives(
                        _permutation_determinant, matrix_ref, direction
                    )
                    for got, want in zip(actual, expected):
                        self.assertTrue(torch.isfinite(got).all())
                        self.assertTrue(torch.allclose(got, want, rtol=1e-9, atol=1e-10))

    def test_polynomial_supports_torch_func_jvp_and_vmap_at_repeated_rows(self) -> None:
        matrix = torch.tensor(
            [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 1.0]]],
            dtype=torch.float64,
        )
        tangent = torch.ones_like(matrix)
        value, jvp = torch.func.jvp(_determinant_polynomial, (matrix,), (tangent,))
        mapped = torch.vmap(_determinant_polynomial)(matrix)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.isfinite(jvp).all())
        self.assertTrue(torch.allclose(value, mapped))

    def test_local_energy_solvers_remain_finite_when_one_branch_is_singular(self) -> None:
        system = MolecularSystem(
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64), n_up=2, n_down=0,
        )
        r = torch.zeros(3, 2, 3, dtype=torch.float64, requires_grad=True)
        model = _SingularBranchWavefunction()
        for method in ("exact", "exact_jvp", "hutchinson"):
            with self.subTest(method=method):
                with mock.patch("hamiltonian.warnings.warn"):
                    energy, kinetic, potential = compute_local_energy(
                        model, system, r, create_graph=True, laplacian_method=method
                    )
                self.assertTrue(torch.isfinite(energy).all())
                self.assertTrue(torch.isfinite(kinetic).all())
                self.assertTrue(torch.isfinite(potential).all())

    def test_local_energy_rejects_a_true_total_wavefunction_node(self) -> None:
        system = MolecularSystem(
            torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            n_up=1, n_down=0,
        )
        r = torch.zeros(2, 1, 3, dtype=torch.float64, requires_grad=True)
        with self.assertRaisesRegex(FloatingPointError, "wavefunction node"):
            compute_local_energy(_ConstantLogWavefunction(float("-inf")), system, r)


class TestR4R5StateAndSystemGuards(unittest.TestCase):
    def _model(self, charges=(1.0, 3.0, 1.0), spins=(2, 1)) -> FermiNet:
        z = torch.tensor(charges, dtype=torch.float64)
        coords = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        return FermiNet(coords, z, *spins, num_determinants=2, hidden_dim_one=8,
                        hidden_dim_two=4, num_layers=1).double()

    def test_legacy_checkpoint_rebuilds_group_index_from_its_own_charges(self) -> None:
        source = self._model()
        checkpoint = OrderedDict(source.state_dict())
        checkpoint.pop("envelope_up.nucleus_group_index")
        checkpoint.pop("envelope_down.nucleus_group_index")
        receiver = self._model(charges=(3.0, 1.0, 1.0))
        receiver.load_state_dict(checkpoint)
        self.assertTrue(torch.equal(receiver.envelope_up.nucleus_group_index, torch.tensor([0, 1, 0])))

    def test_conflicting_saved_group_index_is_rejected(self) -> None:
        checkpoint = OrderedDict(self._model().state_dict())
        checkpoint["envelope_up.nucleus_group_index"] = torch.tensor([0, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            self._model().load_state_dict(checkpoint)

    def test_spin_mismatch_fails_before_sampling(self) -> None:
        model = self._model(spins=(2, 1))
        system = MolecularSystem(model.nuclear_coords, model.nuclear_charges, n_up=1, n_down=2)
        with self.assertRaisesRegex(ValueError, "自旋分区"):
            model.check_system_consistency(system)

    def test_evaluation_requires_current_sampler_model_system_and_walker_shape(self) -> None:
        model = self._model()
        system = MolecularSystem(model.nuclear_coords, model.nuclear_charges, n_up=2, n_down=1)
        sampler = MCMCSampler(model, system, num_walkers=3, dtype=torch.float64)
        other = self._model()
        with self.assertRaisesRegex(ValueError, "exact model"):
            evaluate_trained_model(other, system, sampler, evaluation_steps=2, evaluation_burn_in_steps=0)
        sampler.walkers = sampler.walkers[:2]
        with self.assertRaisesRegex(ValueError, "walkers must have shape"):
            evaluate_trained_model(model, system, sampler, evaluation_steps=2, evaluation_burn_in_steps=0)


class TestR6R9Behaviour(unittest.TestCase):
    def test_default_features_keep_direction_and_distance_mode_remains_invariant(self) -> None:
        nuclei = torch.zeros(1, 3, dtype=torch.float64)
        x = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
        y = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        default_x, _ = ElectronFeatureEmbedding()(x, nuclei)
        default_y, _ = ElectronFeatureEmbedding()(y, nuclei)
        scalar_x, _ = ElectronFeatureEmbedding(rotation_invariant=True)(x, nuclei)
        scalar_y, _ = ElectronFeatureEmbedding(rotation_invariant=True)(y, nuclei)
        self.assertFalse(torch.allclose(default_x, default_y))
        self.assertTrue(torch.allclose(scalar_x, scalar_y))

    def test_incompatible_rotation_and_anisotropic_envelope_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不可同时使用"):
            FermiNet(torch.zeros(1, 3), torch.ones(1), 1, 0,
                     rotation_invariant=True, anisotropic_envelope=True)

    def test_det_weight_scaling_preserves_log_shift_coordinate_derivative_and_node(self) -> None:
        model = FermiNet(torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
                         1, 0, num_determinants=2, hidden_dim_one=4, hidden_dim_two=2,
                         num_layers=1).double()
        def orbitals(r):
            x = r[:, 0, 0]
            one = torch.ones_like(x)
            return torch.stack((x, 2 * x), dim=1).unsqueeze(-1).unsqueeze(-1), None
        r = torch.tensor([[[0.75, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
        with mock.patch.object(model, "forward_orbitals", side_effect=orbitals):
            with torch.no_grad():
                model.det_weights.copy_(torch.tensor([1.0, 2.0], dtype=torch.float64))
            base_log, _ = model.forward_log_psi(r)
            base_grad = torch.autograd.grad(base_log.sum(), r, retain_graph=True)[0]
            system = MolecularSystem(model.nuclear_coords, model.nuclear_charges, n_up=1, n_down=0)
            base_energy, _, _ = compute_local_energy(model, system, r, laplacian_method="exact")
            with torch.no_grad():
                model.det_weights.mul_(1e-45)
            scaled_log, _ = model.forward_log_psi(r)
            scaled_grad = torch.autograd.grad(scaled_log.sum(), r)[0]
            scaled_energy, _, _ = compute_local_energy(model, system, r, laplacian_method="exact")
            self.assertAlmostEqual((scaled_log - base_log).item(), -45 * math.log(10), delta=1e-12)
            self.assertTrue(torch.allclose(base_grad, scaled_grad, rtol=1e-12, atol=1e-12))
            self.assertTrue(torch.allclose(base_energy, scaled_energy, rtol=1e-12, atol=1e-12))
            node_log, node_sign = model.forward_log_psi(torch.zeros_like(r))
            self.assertTrue(torch.isneginf(node_log).all())
            self.assertTrue((node_sign == 0).all())

    def test_nan_and_positive_infinity_never_replace_mcmc_cache_but_node_is_allowed(self) -> None:
        system = MolecularSystem(torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, dtype=torch.float64), 1, 0)
        for value in (float("nan"), float("inf")):
            with self.subTest(initial=value):
                with self.assertRaises(FloatingPointError):
                    MCMCSampler(_ConstantLogWavefunction(value), system, num_walkers=3, dtype=torch.float64)
        model = _ConstantLogWavefunction(-2.0)
        sampler = MCMCSampler(model, system, num_walkers=3, dtype=torch.float64)
        old_cache = sampler.current_log_abs_psi.clone()
        model.value.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            sampler.recompute_log_psi()
        self.assertTrue(torch.equal(old_cache, sampler.current_log_abs_psi))
        model.value.fill_(-float("inf"))
        self.assertTrue(torch.isneginf(sampler.recompute_log_psi()).all())

    def test_nonfinite_proposals_are_rejected_in_both_move_modes(self) -> None:
        class ProposalModel(nn.Module):
            def forward_log_psi(self, r: torch.Tensor):
                values = torch.where(
                    (r == 0).all(dim=(1, 2)), torch.zeros(r.shape[0], dtype=r.dtype),
                    torch.full((r.shape[0],), float("nan"), dtype=r.dtype),
                )
                return values, torch.ones_like(values)
        system = MolecularSystem(torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, dtype=torch.float64), 1, 0)
        initial = torch.zeros(3, 1, 3, dtype=torch.float64)
        for mode in ("all", "one_by_one"):
            with self.subTest(mode=mode):
                sampler = MCMCSampler(ProposalModel(), system, num_walkers=3, move_mode=mode,
                                      initial_walkers=initial, dtype=torch.float64, adapt_step_size=False)
                with mock.patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)), \
                     mock.patch("torch.randn", side_effect=lambda *shape, **kwargs: torch.ones(*shape, **kwargs)):
                    self.assertEqual(sampler.step(), 0.0)
                self.assertTrue(torch.equal(sampler.walkers, initial))
                self.assertTrue(torch.equal(sampler.current_log_abs_psi, torch.zeros(3, dtype=torch.float64)))

    def test_finite_proposal_can_leave_a_valid_node(self) -> None:
        class NodeExitModel(nn.Module):
            def forward_log_psi(self, r: torch.Tensor):
                is_node = (r == 0).all(dim=(1, 2))
                values = torch.where(
                    is_node, torch.full((r.shape[0],), float("-inf"), dtype=r.dtype),
                    torch.zeros(r.shape[0], dtype=r.dtype),
                )
                return values, torch.ones_like(values)
        system = MolecularSystem(torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, dtype=torch.float64), 1, 0)
        sampler = MCMCSampler(
            NodeExitModel(), system, num_walkers=2, move_mode="all",
            initial_walkers=torch.zeros(2, 1, 3, dtype=torch.float64),
            dtype=torch.float64, adapt_step_size=False,
        )
        with mock.patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)):
            self.assertEqual(sampler.step(), 1.0)
        self.assertTrue(torch.all(sampler.current_log_abs_psi == 0.0))


if __name__ == "__main__":
    unittest.main()
