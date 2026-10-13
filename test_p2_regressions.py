"""Physical invariants and failure handling from the multi-angle review."""

import unittest
from unittest.mock import patch

import torch

from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.jastrow import KatoCuspJastrow
from ferminet_pytorch.loss import clip_local_energy, vmc_surrogate_loss
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.spin import SpinProjector
from ferminet_pytorch.train_vmc import build_lih_system, train_vmc


class TestSpinProjectionRegressions(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_multielectron_s_squared_idempotence_and_antisymmetry(self) -> None:
        """Check actual S^2 eigenvalues, including the S=3 sector in six electrons."""
        for n_up, n_down, mode, spin in ((2, 2, "singlet", 0), (2, 2, "triplet", 1),
                                        (3, 3, "singlet", 0), (3, 3, "triplet", 1),
                                        (3, 1, "triplet", 1)):
            with self.subTest(n_up=n_up, n_down=n_down, mode=mode):
                model = FermiNet(torch.zeros(1, 3), torch.ones(1), n_up, n_down,
                                 num_determinants=2, hidden_dim_one=8, hidden_dim_two=4,
                                 num_layers=1, spin_adapted=True,
                                 spin_projector_kwargs={"mode": mode}).double()
                r = torch.randn(3, n_up + n_down, 3, dtype=torch.float64)
                with torch.no_grad():
                    psi = model(r)
                    self.assertGreater(psi.norm().item(), 1e-12)
                    s_squared = ((n_up + n_down) / 2 + (n_up - n_down) ** 2 / 4) * psi
                    for i in range(n_up):
                        for j in range(n_up, n_up + n_down):
                            permutation = list(range(n_up + n_down))
                            permutation[i], permutation[j] = permutation[j], permutation[i]
                            s_squared -= model(r[:, permutation])
                    residual = (s_squared - spin * (spin + 1) * psi).norm() / psi.norm()
                    self.assertLess(residual.item(), 1e-10)
                    log_twice, sign_twice = model.spin_projector.project_log_psi(model.forward_log_psi, r)
                    torch.testing.assert_close(sign_twice * log_twice.exp(), psi,
                                               atol=1e-13, rtol=1e-10)
                    swapped = r.clone()
                    swapped[:, [0, 1]] = r[:, [1, 0]]
                    torch.testing.assert_close(model(swapped), -psi, atol=1e-13, rtol=1e-10)

    def test_singlet_rejects_fully_antisymmetric_quintet(self) -> None:
        projector = SpinProjector(2, 2, "singlet")

        def vandermonde(r):
            x = r[:, :, 0]
            value = torch.stack([x[:, i] - x[:, j] for i in range(4)
                                 for j in range(i + 1, 4)], dim=-1).prod(dim=-1)
            return value.abs().log(), value.sign()

        r = torch.zeros(1, 4, 3, dtype=torch.float64)
        r[0, :, 0] = torch.arange(4, dtype=torch.float64)
        log_abs, sign = projector.project_log_psi(vandermonde, r)
        self.assertLess((sign * log_abs.exp()).abs().item(), 1e-12)

    def test_lih_with_jastrow_is_singlet_and_has_finite_energy_gradients(self) -> None:
        system = build_lih_system(dtype=torch.float64)
        model = FermiNet(system.nuclear_coords, system.nuclear_charges, 2, 2,
                         num_determinants=2, hidden_dim_one=8, hidden_dim_two=4,
                         num_layers=1, spin_adapted=True, use_cusp_jastrow=True).double()
        r = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        psi = model(r)
        s_squared = 2 * psi
        for i in range(2):
            for j in range(2, 4):
                permutation = list(range(4))
                permutation[i], permutation[j] = permutation[j], permutation[i]
                s_squared = s_squared - model(r[:, permutation])
        self.assertLess((s_squared.norm() / psi.norm()).item(), 1e-10)
        log_abs, _ = model.forward_log_psi(r)
        log_abs.sum().backward()
        self.assertTrue(torch.isfinite(r.grad).all())
        energy, _, _ = compute_local_energy(model, system, r.detach().requires_grad_())
        self.assertTrue(torch.isfinite(energy).all())

    def test_custom_single_permutation_is_applied_to_values_and_gradients(self) -> None:
        projector = SpinProjector(1, 1, "custom", [([1, 0], 1.0)])
        r = torch.tensor([[[1., 0., 0.], [2., 0., 0.]]], requires_grad=True)
        log_abs, _ = projector.project_log_psi(lambda x: (x[:, 0, 0], torch.ones(1)), r)
        self.assertEqual(log_abs.item(), 2.0)
        log_abs.sum().backward()
        torch.testing.assert_close(r.grad, torch.tensor([[[0., 0., 0.], [1., 0., 0.]]]))

    def test_incompatible_named_spin_is_rejected(self) -> None:
        for n_up, n_down, mode in ((1, 0, "singlet"), (3, 1, "singlet"),
                                   (2, 1, "triplet"), (4, 0, "triplet")):
            with self.subTest(n_up=n_up, n_down=n_down, mode=mode):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    SpinProjector(n_up, n_down, mode)


class TestTripletCuspRegressions(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.coords = torch.zeros(1, 3, dtype=torch.float64)
        self.charges = torch.ones(1, dtype=torch.float64)

    def test_triplet_cusp_derivative_is_one_quarter(self) -> None:
        jastrow = KatoCuspJastrow(self.charges, 1, 1, two_electron_spin=1).double()
        distance = torch.tensor(1e-6, dtype=torch.float64, requires_grad=True)
        direction = torch.tensor([1., 0., 0.], dtype=torch.float64)
        r = torch.stack((direction * 0, direction * distance)).unsqueeze(0)
        derivative = torch.autograd.grad(jastrow.compute_ee(r).sum(), distance)[0]
        self.assertAlmostEqual(derivative.item(), 0.25, places=5)

    def test_real_triplet_local_energy_remains_bounded_at_coalescence(self) -> None:
        for custom_projector in (False, True):
            with self.subTest(custom_projector=custom_projector):
                kwargs = ({"spin_projector": SpinProjector(1, 1, "triplet")}
                          if custom_projector else
                          {"spin_adapted": True, "spin_projector_kwargs": {"mode": "triplet"}})
                model = FermiNet(self.coords, self.charges, 1, 1,
                                 num_determinants=2, hidden_dim_one=16, hidden_dim_two=8,
                                 num_layers=1, use_cusp_jastrow=True, **kwargs).double()
                self.assertEqual(model.jastrow.gamma_ee[0, 1].item(), 0.25)
                system = MolecularSystem(self.coords, self.charges, 1, 1)
                energies = []
                for distance in (1e-3, 1e-4, 1e-5):
                    r = torch.tensor([[[.5, .2, .3], [.5 + distance, .2, .3]]],
                                     dtype=torch.float64, requires_grad=True)
                    energy, _, _ = compute_local_energy(model, system, r)
                    self.assertTrue(torch.isfinite(energy).all())
                    energies.append(energy.item())
                self.assertLess(abs(energies[-1] - energies[-2]), 0.1)
                self.assertLess(abs(energies[-1]), 100.0)

    def test_incompatible_explicit_jastrow_is_rejected(self) -> None:
        for kwargs in ({"jastrow": KatoCuspJastrow(self.charges, 1, 1)},
                       {"use_cusp_jastrow": True, "jastrow_kwargs": {"two_electron_spin": 0}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    FermiNet(self.coords, self.charges, 1, 1, spin_adapted=True,
                             spin_projector_kwargs={"mode": "triplet"}, **kwargs)


class TestInvalidEnergyRegressions(unittest.TestCase):
    def test_all_invalid_energy_raises_instead_of_reporting_zero(self) -> None:
        for values in ([float("nan")] * 3, [float("inf"), float("-inf"), float("nan")]):
            for method in ("tanh", "hard", "none"):
                with self.subTest(values=values, method=method):
                    energy = torch.tensor(values)
                    with self.assertRaises(FloatingPointError):
                        clip_local_energy(energy, method=method)
                    with self.assertRaises(FloatingPointError):
                        vmc_surrogate_loss(torch.zeros(3, requires_grad=True), energy,
                                           clip_method=method)

    def test_partial_invalid_energy_uses_only_valid_samples(self) -> None:
        for method in ("tanh", "hard", "none"):
            with self.subTest(method=method):
                log_abs = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
                energy = torch.tensor([1., float("nan"), 3., float("inf"), float("-inf")])
                with self.assertWarnsRegex(RuntimeWarning, "Excluded 3/5"):
                    loss, stats = vmc_surrogate_loss(log_abs, energy, clip_method=method)
                reference, _ = vmc_surrogate_loss(log_abs[[0, 2]], energy[[0, 2]], clip_method=method)
                torch.testing.assert_close(loss, reference)
                gradient = torch.autograd.grad(loss, log_abs)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(gradient[[1, 3, 4]].abs().sum().item(), 0.0)
                self.assertEqual(stats["num_valid"], 2)
                self.assertEqual(stats["num_invalid"], 3)
                self.assertEqual(stats["energy_mean"], 2.0)
                self.assertEqual(stats["energy_var"], 2.0)

    def test_nonfinite_log_wavefunction_aborts_update(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(FloatingPointError, "log"):
                    vmc_surrogate_loss(torch.tensor([0., value], requires_grad=True), torch.ones(2))

    def test_training_aborts_before_optimizer_step(self) -> None:
        torch.manual_seed(42)
        system = MolecularSystem(torch.zeros(1, 3), torch.ones(1), 1, 0)
        model = FermiNet(system.nuclear_coords, system.nuclear_charges, 1, 0,
                         num_determinants=2, hidden_dim_one=8, hidden_dim_two=4, num_layers=1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        invalid = torch.full((4,), float("nan"))
        with patch("ferminet_pytorch.loss.compute_local_energy", return_value=(invalid,) * 3):
            with patch("torch.optim.Adam.step") as step:
                with self.assertRaisesRegex(FloatingPointError, "No finite"):
                    train_vmc(system=system, model=model, num_walkers=4, num_epochs=1,
                              burn_in_steps=0, mcmc_steps_per_epoch=1, device="cpu", verbose=False)
                step.assert_not_called()
        for old, new in zip(before, model.parameters()):
            torch.testing.assert_close(old, new, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
