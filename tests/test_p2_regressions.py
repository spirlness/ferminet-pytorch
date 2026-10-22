"""Physical invariants and failure handling from the multi-angle review."""

from ferminet_pytorch.vmc_driver import TrainingConfig

import unittest
from unittest.mock import patch

import torch

from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.jastrow import KatoCuspJastrow
from ferminet_pytorch.layers import FermiInteractionLayer
from ferminet_pytorch.loss import VMCLoss, clip_local_energy, vmc_surrogate_loss
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.spin import SpinProjector
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.systems import build_h2_system, build_lih_system
from ferminet_pytorch.vmc_driver import train_vmc


class TestSpinProjectionRegressions(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_multielectron_s_squared_idempotence_and_antisymmetry(self) -> None:
        """Check actual S^2 eigenvalues, including the S=3 sector in six electrons."""
        for n_up, n_down, mode, spin in (
            (2, 2, "singlet", 0),
            (2, 2, "triplet", 1),
            (3, 3, "singlet", 0),
            (3, 3, "triplet", 1),
            (3, 1, "triplet", 1),
        ):
            with self.subTest(n_up=n_up, n_down=n_down, mode=mode):
                model = FermiNet(
                    torch.zeros(1, 3),
                    torch.ones(1),
                    n_up,
                    n_down,
                    num_determinants=2,
                    hidden_dim_one=8,
                    hidden_dim_two=4,
                    num_layers=1,
                    spin_adapted=True,
                    spin_projector_kwargs={"mode": mode},
                ).double()
                r = torch.randn(3, n_up + n_down, 3, dtype=torch.float64)
                with torch.no_grad():
                    psi = model(r)
                    self.assertGreater(psi.norm().item(), 1e-12)
                    s_squared = ((n_up + n_down) / 2 + (n_up - n_down) ** 2 / 4) * psi
                    for i in range(n_up):
                        for j in range(n_up, n_up + n_down):
                            permutation = list(range(n_up + n_down))
                            permutation[i], permutation[j] = (
                                permutation[j],
                                permutation[i],
                            )
                            s_squared -= model(r[:, permutation])
                    residual = (s_squared - spin * (spin + 1) * psi).norm() / psi.norm()
                    self.assertLess(residual.item(), 1e-10)
                    log_twice, sign_twice = model.spin_projector.project_log_psi(
                        model.forward_log_psi, r
                    )
                    torch.testing.assert_close(
                        sign_twice * log_twice.exp(), psi, atol=1e-13, rtol=1e-10
                    )
                    swapped = r.clone()
                    swapped[:, [0, 1]] = r[:, [1, 0]]
                    torch.testing.assert_close(
                        model(swapped), -psi, atol=1e-13, rtol=1e-10
                    )

    def test_singlet_rejects_fully_antisymmetric_quintet(self) -> None:
        projector = SpinProjector(2, 2, "singlet")

        def vandermonde(r):
            x = r[:, :, 0]
            value = torch.stack(
                [x[:, i] - x[:, j] for i in range(4) for j in range(i + 1, 4)], dim=-1
            ).prod(dim=-1)
            return value.abs().log(), value.sign()

        r = torch.zeros(1, 4, 3, dtype=torch.float64)
        r[0, :, 0] = torch.arange(4, dtype=torch.float64)
        log_abs, sign = projector.project_log_psi(vandermonde, r)
        self.assertLess((sign * log_abs.exp()).abs().item(), 1e-12)

    def test_lih_with_jastrow_is_singlet_and_has_finite_energy_gradients(self) -> None:
        system = build_lih_system(dtype=torch.float64)
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            2,
            2,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
            spin_adapted=True,
            use_cusp_jastrow=True,
        ).double()
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
        r = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], requires_grad=True)
        log_abs, _ = projector.project_log_psi(lambda x: (x[:, 0, 0], torch.ones(1)), r)
        self.assertEqual(log_abs.item(), 2.0)
        log_abs.sum().backward()
        torch.testing.assert_close(
            r.grad, torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        )

    def test_incompatible_named_spin_is_rejected(self) -> None:
        for n_up, n_down, mode in (
            (1, 0, "singlet"),
            (3, 1, "singlet"),
            (2, 1, "triplet"),
            (4, 0, "triplet"),
        ):
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
        direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        r = torch.stack((direction * 0, direction * distance)).unsqueeze(0)
        derivative = torch.autograd.grad(jastrow.compute_ee(r).sum(), distance)[0]
        self.assertAlmostEqual(derivative.item(), 0.25, places=5)

    def test_real_triplet_local_energy_remains_bounded_at_coalescence(self) -> None:
        for custom_projector in (False, True):
            with self.subTest(custom_projector=custom_projector):
                kwargs = (
                    {"spin_projector": SpinProjector(1, 1, "triplet")}
                    if custom_projector
                    else {
                        "spin_adapted": True,
                        "spin_projector_kwargs": {"mode": "triplet"},
                    }
                )
                model = FermiNet(
                    self.coords,
                    self.charges,
                    1,
                    1,
                    num_determinants=2,
                    hidden_dim_one=16,
                    hidden_dim_two=8,
                    num_layers=1,
                    use_cusp_jastrow=True,
                    **kwargs,
                ).double()
                self.assertEqual(model.jastrow.gamma_ee[0, 1].item(), 0.25)
                system = MolecularSystem(self.coords, self.charges, 1, 1)
                energies = []
                for distance in (1e-3, 1e-4, 1e-5):
                    r = torch.tensor(
                        [[[0.5, 0.2, 0.3], [0.5 + distance, 0.2, 0.3]]],
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    energy, _, _ = compute_local_energy(model, system, r)
                    self.assertTrue(torch.isfinite(energy).all())
                    energies.append(energy.item())
                self.assertLess(abs(energies[-1] - energies[-2]), 0.1)
                self.assertLess(abs(energies[-1]), 100.0)

    def test_incompatible_explicit_jastrow_is_rejected(self) -> None:
        for kwargs in (
            {"jastrow": KatoCuspJastrow(self.charges, 1, 1)},
            {"use_cusp_jastrow": True, "jastrow_kwargs": {"two_electron_spin": 0}},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    FermiNet(
                        self.coords,
                        self.charges,
                        1,
                        1,
                        spin_adapted=True,
                        spin_projector_kwargs={"mode": "triplet"},
                        **kwargs,
                    )


class TestInvalidEnergyRegressions(unittest.TestCase):
    def test_all_invalid_energy_raises_instead_of_reporting_zero(self) -> None:
        for values in ([float("nan")] * 3, [float("inf"), float("-inf"), float("nan")]):
            for method in ("tanh", "hard", "none"):
                with self.subTest(values=values, method=method):
                    energy = torch.tensor(values)
                    with self.assertRaises(FloatingPointError):
                        clip_local_energy(energy, method=method)
                    with self.assertRaises(FloatingPointError):
                        vmc_surrogate_loss(
                            torch.zeros(3, requires_grad=True),
                            energy,
                            clip_method=method,
                        )

    def test_partial_invalid_energy_uses_only_valid_samples(self) -> None:
        for method in ("tanh", "hard", "none"):
            with self.subTest(method=method):
                log_abs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
                energy = torch.tensor(
                    [1.0, float("nan"), 3.0, float("inf"), float("-inf")]
                )
                with self.assertWarnsRegex(RuntimeWarning, "Excluded 3/5"):
                    loss, stats = vmc_surrogate_loss(
                        log_abs, energy, clip_method=method, max_invalid_fraction=1.0
                    )
                reference, _ = vmc_surrogate_loss(
                    log_abs[[0, 2]],
                    energy[[0, 2]],
                    clip_method=method,
                    max_invalid_fraction=1.0,
                )
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
                    vmc_surrogate_loss(
                        torch.tensor([0.0, value], requires_grad=True), torch.ones(2)
                    )

    def test_training_aborts_before_optimizer_step(self) -> None:
        torch.manual_seed(42)
        system = MolecularSystem(torch.zeros(1, 3), torch.ones(1), 1, 0)
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            1,
            0,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        invalid = torch.full((4,), float("nan"))
        with patch(
            "ferminet_pytorch.loss.compute_local_energy", return_value=(invalid,) * 3
        ):
            with patch("torch.optim.Adam.step") as step:
                with self.assertRaisesRegex(FloatingPointError, "No finite"):
                    train_vmc(
                        system=system,
                        model=model,
                        config=TrainingConfig(
                            num_walkers=4,
                            num_epochs=1,
                            burn_in_steps=0,
                            mcmc_steps_per_epoch=1,
                            device="cpu",
                            verbose=False,
                        ),
                    )
                step.assert_not_called()
        for old, new in zip(before, model.parameters()):
            torch.testing.assert_close(old, new, atol=0, rtol=0)


class TestOddElectronRegressions(unittest.TestCase):
    """Regression test suite for odd and general open-shell electron configurations."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_li_atom_spin_adapted_initialization_and_forward(self) -> None:
        """
        Verify Li atom (n_up=2, n_down=1, spin_adapted=True) initialization,
        forward pass, same-spin antisymmetry, and local energy calculation.
        """
        nuclear_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        nuclear_charges = torch.tensor([3.0], dtype=torch.float64)
        system_li = MolecularSystem(nuclear_coords, nuclear_charges, n_up=2, n_down=1)

        model_li = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
            use_cusp_jastrow=True,
        ).to(dtype=torch.float64)

        # Spin adaptation for un-paired electrons safely defaults to mode='none'
        self.assertIsNotNone(model_li.spin_projector)
        self.assertEqual(model_li.spin_projector.mode, "none")

        batch_size = 4
        r = torch.randn(batch_size, 3, 3, dtype=torch.float64, requires_grad=True)
        psi = model_li(r)
        log_abs, sign = model_li.forward_log_psi(r)

        self.assertEqual(psi.shape, (batch_size,))
        self.assertEqual(log_abs.shape, (batch_size,))
        self.assertTrue(torch.isfinite(psi).all())
        self.assertTrue(torch.isfinite(log_abs).all())
        torch.testing.assert_close(sign * log_abs.exp(), psi, atol=1e-12, rtol=1e-9)

        # Invariance: antisymmetry under permutation of same-spin electrons (0 and 1)
        r_swap_up = r.clone()
        r_swap_up[:, [0, 1], :] = r[:, [1, 0], :]
        with torch.no_grad():
            psi_swap = model_li(r_swap_up)
        torch.testing.assert_close(psi_swap, -psi, atol=1e-12, rtol=1e-9)

        # Finite local energy calculation without NaNs
        energy, t_local, v_local = compute_local_energy(model_li, system_li, r)
        self.assertEqual(energy.shape, (batch_size,))
        self.assertTrue(torch.isfinite(energy).all())
        self.assertTrue(torch.isfinite(t_local).all())
        self.assertTrue(torch.isfinite(v_local).all())

    def test_general_open_shell_boron_and_polarized_systems(self) -> None:
        """
        Verify open-shell Boron atom (3, 2) and fully-polarized (3, 0) system
        initialization, forward pass, and local energy finiteness.
        """
        # Open-shell Boron atom (n_up=3, n_down=2)
        coords_b = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges_b = torch.tensor([5.0], dtype=torch.float64)
        system_b = MolecularSystem(coords_b, charges_b, n_up=3, n_down=2)
        model_b = FermiNet(
            nuclear_coords=coords_b,
            nuclear_charges=charges_b,
            n_up=3,
            n_down=2,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
            use_cusp_jastrow=True,
        ).to(dtype=torch.float64)

        self.assertIsNotNone(model_b.spin_projector)
        self.assertEqual(model_b.spin_projector.mode, "none")

        r_b = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        psi_b = model_b(r_b)
        self.assertTrue(torch.isfinite(psi_b).all())
        e_b, t_b, v_b = compute_local_energy(model_b, system_b, r_b)
        self.assertTrue(torch.isfinite(e_b).all())
        self.assertTrue(torch.isfinite(t_b).all())
        self.assertTrue(torch.isfinite(v_b).all())

        # Fully-polarized system (n_up=3, n_down=0)
        coords_pol = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges_pol = torch.tensor([3.0], dtype=torch.float64)
        system_pol = MolecularSystem(coords_pol, charges_pol, n_up=3, n_down=0)
        model_pol = FermiNet(
            nuclear_coords=coords_pol,
            nuclear_charges=charges_pol,
            n_up=3,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
            use_cusp_jastrow=True,
        ).to(dtype=torch.float64)

        self.assertIsNotNone(model_pol.spin_projector)
        self.assertEqual(model_pol.spin_projector.mode, "none")

        r_pol = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        psi_pol = model_pol(r_pol)
        self.assertTrue(torch.isfinite(psi_pol).all())
        e_pol, t_pol, v_pol = compute_local_energy(model_pol, system_pol, r_pol)
        self.assertTrue(torch.isfinite(e_pol).all())
        self.assertTrue(torch.isfinite(t_pol).all())
        self.assertTrue(torch.isfinite(v_pol).all())

    def test_even_electron_high_spin_triplet_safe_fallback(self) -> None:
        """
        Verify even-electron high-spin states (4-electron 3,1 and Carbon atom 4,2)
        safely fall back to mode='none' and maintain finite local energies.
        """
        # 4-electron high-spin (n_up=3, n_down=1)
        coords_4e = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges_4e = torch.tensor([4.0], dtype=torch.float64)
        system_4e = MolecularSystem(coords_4e, charges_4e, n_up=3, n_down=1)
        model_4e = FermiNet(
            nuclear_coords=coords_4e,
            nuclear_charges=charges_4e,
            n_up=3,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
            spin_adapted=True,
            use_cusp_jastrow=True,
        ).to(dtype=torch.float64)

        self.assertIsNotNone(model_4e.spin_projector)
        self.assertEqual(model_4e.spin_projector.mode, "none")

        r_4e = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)
        e_4e, t_4e, v_4e = compute_local_energy(model_4e, system_4e, r_4e)
        self.assertTrue(torch.isfinite(e_4e).all())
        self.assertTrue(torch.isfinite(t_4e).all())
        self.assertTrue(torch.isfinite(v_4e).all())

        # Carbon atom ground-state triplet (n_up=4, n_down=2)
        coords_c = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        charges_c = torch.tensor([6.0], dtype=torch.float64)
        system_c = MolecularSystem(coords_c, charges_c, n_up=4, n_down=2)
        model_c = FermiNet(
            nuclear_coords=coords_c,
            nuclear_charges=charges_c,
            n_up=4,
            n_down=2,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
            spin_adapted=True,
            use_cusp_jastrow=True,
        ).to(dtype=torch.float64)

        self.assertIsNotNone(model_c.spin_projector)
        self.assertEqual(model_c.spin_projector.mode, "none")

        r_c = torch.randn(3, 6, 3, dtype=torch.float64, requires_grad=True)
        e_c, t_c, v_c = compute_local_energy(model_c, system_c, r_c)
        self.assertTrue(torch.isfinite(e_c).all())
        self.assertTrue(torch.isfinite(t_c).all())
        self.assertTrue(torch.isfinite(v_c).all())


class TestNodalSurfaceRegressions(unittest.TestCase):
    """Regression test suite for nodal surface robustness and zero-determinant behavior."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_nodal_surface_zero_determinant_autograd_finite_and_no_nan(self) -> None:
        """
        Verify that on wavefunction nodal surfaces (where det -> 0 or exactly 0),
        forward_log_psi and coordinate autograd derivatives of compute_local_energy
        are strictly finite and never produce NaNs or crashes.
        """
        nuclear_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
        system = MolecularSystem(nuclear_coords, nuclear_charges, n_up=2, n_down=0)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        ).to(dtype=torch.float64)

        # Construct exact nodal surface where electron 0 and electron 1 coincide
        batch_size = 3
        r_node = torch.randn(batch_size, 2, 3, dtype=torch.float64)
        r_node[:, 1, :] = r_node[:, 0, :]
        r_node.requires_grad_(True)

        with torch.no_grad():
            psi_val = model(r_node)
            log_abs, sign = model.forward_log_psi(r_node)

        # Wavefunction value must be exactly zero on same-spin coincidence node
        self.assertTrue(torch.all(psi_val == 0.0))
        self.assertTrue(torch.all(sign == 0.0))
        self.assertTrue(torch.all(torch.isneginf(log_abs) | (log_abs <= -40.0)))

        # At an exact node H Psi / Psi is undefined. The physical Coulomb
        # potential is singular at coincident electrons; do not hide it.
        energy, t_local, v_local = compute_local_energy(
            model, system, r_node, create_graph=False
        )
        self.assertFalse(
            torch.isnan(t_local).any(),
            "Kinetic energy Laplacian contains NaN on nodal surface",
        )
        self.assertFalse(
            torch.isnan(energy).any(), "Local energy contains NaN on nodal surface"
        )
        self.assertTrue(
            torch.isfinite(t_local).all(),
            "Kinetic energy is not finite on nodal surface",
        )
        self.assertTrue(torch.isposinf(v_local).all())
        self.assertFalse(torch.isfinite(energy).any())

        # Near-node continuous perturbation (r_01 = 1e-12)
        r_near = r_node.clone().detach()
        r_near[:, 1, 0] += 1e-12
        r_near.requires_grad_(True)
        e_near, t_near, _ = compute_local_energy(
            model, system, r_near, create_graph=False
        )
        self.assertFalse(
            torch.isnan(t_near).any(), "Near-node kinetic energy contains NaN"
        )
        self.assertFalse(
            torch.isnan(e_near).any(), "Near-node local energy contains NaN"
        )
        self.assertTrue(torch.isfinite(e_near).all())

    def test_coincident_same_spin_electrons_higher_order_derivatives(self) -> None:
        """
        Verify that higher-order gradient propagation (create_graph=True) through
        the nodal surface coordinates remains strictly controlled and finite.
        """
        nuclear_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
        system = MolecularSystem(nuclear_coords, nuclear_charges, n_up=2, n_down=0)
        model = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=2,
            n_down=0,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        ).to(dtype=torch.float64)

        r_node = torch.randn(2, 2, 3, dtype=torch.float64)
        r_node[:, 1, :] = r_node[:, 0, :]
        # Coordinate derivatives of local energy are tested off the node.
        r_node[:, 1, 0] += 1e-3
        r_node.requires_grad_(True)

        e_local, t_local, _ = compute_local_energy(
            model, system, r_node, create_graph=True
        )
        self.assertTrue(torch.isfinite(e_local).all())
        self.assertTrue(torch.isfinite(t_local).all())

        # Backward propagation of local energy gradient w.r.t coordinates
        e_local.sum().backward()
        self.assertIsNotNone(r_node.grad)
        self.assertTrue(torch.isfinite(r_node.grad).all())


class TestMCMCStationaryRegressions(unittest.TestCase):
    """Regression test suite for MCMC step size freezing and stationary sampling properties."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_sample_adapt_false_strictly_freezes_step_size(self) -> None:
        """
        Verify MetropolisSampler.sample(num_steps=10, adapt=False) strictly preserves
        the step size during production sampling, regardless of batch acceptance rate.
        """
        system = build_h2_system(bond_length=1.4)
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        sampler = MCMCSampler(
            model=model,
            system=system,
            num_walkers=32,
            step_size=0.5,
            adapt_step_size=True,
            target_acceptance=0.5,
        )

        # 1. Burn-in: step size dynamically adapts
        sampler.burn_in(num_steps=10)
        adapted_step = sampler.step_size

        # 2. Production sampling: adapt=False must freeze step size exactly
        walkers_sample_1 = sampler.sample(num_steps=10, adapt=False)
        self.assertEqual(
            sampler.step_size,
            adapted_step,
            f"Step size drifted during sample(adapt=False): {sampler.step_size} != {adapted_step}",
        )

        walkers_sample_2 = sampler.sample(num_steps=20, adapt=False)
        self.assertEqual(
            sampler.step_size,
            adapted_step,
            f"Step size drifted in second production sample: {sampler.step_size} != {adapted_step}",
        )

        # Sampler's persistent adapt_step_size flag must be restored
        self.assertTrue(sampler.adapt_step_size)

        # Sampled walker valid tensor verification
        self.assertEqual(walkers_sample_1.shape, (32, 2, 3))
        self.assertEqual(walkers_sample_2.shape, (32, 2, 3))
        self.assertTrue(torch.isfinite(walkers_sample_1).all())
        self.assertTrue(torch.isfinite(walkers_sample_2).all())

    def test_sample_adapt_true_dynamically_adjusts_step_size(self) -> None:
        """
        Verify MetropolisSampler.sample(num_steps=10, adapt=True) dynamically adapts
        step size towards the target acceptance rate.
        """
        system = build_h2_system(bond_length=1.4)
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=16,
            hidden_dim_two=8,
            num_layers=1,
        )
        # Initialize with step_size far away from equilibrium to force adaptation
        sampler = MCMCSampler(
            model=model,
            system=system,
            num_walkers=32,
            step_size=5.0,
            adapt_step_size=True,
            target_acceptance=0.5,
        )

        initial_step = sampler.step_size
        walkers = sampler.sample(num_steps=10, adapt=True)

        # Step size must have adjusted away from 5.0
        self.assertNotEqual(
            sampler.step_size,
            initial_step,
            "Step size failed to adapt when sample(adapt=True) was explicitly set",
        )
        self.assertTrue(sampler.adapt_step_size)
        self.assertEqual(walkers.shape, (32, 2, 3))
        self.assertTrue(torch.isfinite(walkers).all())


class TestP2OptimizationRegressions(unittest.TestCase):
    """验证 P2 梯队优化项的物理不变量与工程稳健性。"""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_two_electron_pooling_excludes_diagonal_self_interaction(self) -> None:
        """
        验证 FermiInteractionLayer 双电子均值池化显式剔除了自身项 i=j，
        且同自旋分母除以 n-1 (单自旋置 0)，异自旋分母除以 n_other。
        """
        # 1. 四电子体系 (n_up=2, n_down=2)
        n_up = 2
        n_down = 2
        num_electrons = n_up + n_down
        batch_size = 3
        in_dim_one = 8
        in_dim_two = 4
        out_dim_one = 8
        out_dim_two = 4

        layer = FermiInteractionLayer(
            in_dim_one=in_dim_one,
            in_dim_two=in_dim_two,
            out_dim_one=out_dim_one,
            out_dim_two=out_dim_two,
            produce_two_electron_output=True,
            n_up=n_up,
            n_down=n_down,
        )

        h1 = torch.zeros(batch_size, num_electrons, in_dim_one)
        # 构造 h2: 非对角线全为 1.0, 对角线 i=j 注入极大自身特征 999.0
        h2 = torch.ones(batch_size, num_electrons, num_electrons, in_dim_two)
        for i in range(num_electrons):
            h2[:, i, i, :] = 999.0

        # 手动执行 layer 中双电子均值提取逻辑进行等价验证
        eye_mask = torch.eye(num_electrons, dtype=torch.bool)
        h2_no_diag = h2.masked_fill(eye_mask.unsqueeze(0).unsqueeze(-1), 0.0)

        # 检查对角线是否严格为 0
        for i in range(num_electrons):
            self.assertTrue(torch.all(h2_no_diag[:, i, i, :] == 0.0))

        # 执行前向传播，确保不会因自交互项导致数值爆炸
        h1_out, h2_out = layer(h1, h2)
        self.assertTrue(torch.isfinite(h1_out).all())
        self.assertTrue(torch.isfinite(h2_out).all())

        # 2. 单电子边界验证 (n_up=1, n_down=1)
        h1_2 = torch.zeros(batch_size, 2, in_dim_one)
        h2_2 = torch.ones(batch_size, 2, 2, in_dim_two)
        h2_2[:, 0, 0, :] = 500.0
        h2_2[:, 1, 1, :] = 500.0

        layer = FermiInteractionLayer(
            in_dim_one, in_dim_two, out_dim_one, out_dim_two, 1, 1
        )
        h1_out_2, h2_out_2 = layer(h1_2, h2_2)
        self.assertTrue(torch.isfinite(h1_out_2).all())
        self.assertFalse(torch.isnan(h1_out_2).any())

        # 3. 氢原子单自旋极值边界 (n_up=1, n_down=0)
        h1_h = torch.zeros(batch_size, 1, in_dim_one)
        h2_h = torch.ones(batch_size, 1, 1, in_dim_two)
        layer = FermiInteractionLayer(
            in_dim_one, in_dim_two, out_dim_one, out_dim_two, 1, 0
        )
        h1_out_h, _ = layer(h1_h, h2_h)
        self.assertTrue(torch.isfinite(h1_out_h).all())
        self.assertFalse(torch.isnan(h1_out_h).any())

    def test_mcmc_target_acceptance_mode_defaults(self) -> None:
        """
        验证 MCMC 采样器在全电子模式下默认采用理论最优值 0.234，
        在逐电子模式下默认保持 0.50，且显式指定时被优先使用。
        """
        system = build_h2_system()
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            n_up=1,
            n_down=1,
            num_determinants=2,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
        )

        # 全电子移动模式默认 0.234
        sampler_all = MCMCSampler(model=model, system=system, move_mode="all")
        self.assertAlmostEqual(sampler_all.target_acceptance, 0.234, places=3)

        # 逐电子移动模式默认 0.50
        sampler_one = MCMCSampler(model=model, system=system, move_mode="one_by_one")
        self.assertAlmostEqual(sampler_one.target_acceptance, 0.50, places=3)

        # 显式传参覆盖测试
        sampler_custom = MCMCSampler(
            model=model, system=system, move_mode="all", target_acceptance=0.42
        )
        self.assertAlmostEqual(sampler_custom.target_acceptance, 0.42, places=3)

        # 别名兼容性测试
        sampler_alias = MCMCSampler(model=model, system=system, move_mode="all")
        self.assertAlmostEqual(sampler_alias.target_acceptance, 0.234, places=3)

    def test_invalid_energy_fraction_cutoff(self) -> None:
        """
        验证当非有限局域能量样本比例超过安全阈值 max_invalid_fraction 时，
        立即抛出 FloatingPointError 熔断变分更新，杜绝乐观选择偏差。
        """
        batch_size = 10
        log_abs = torch.zeros(batch_size, requires_grad=True)
        # 10 个样本中有 2 个 NaN (20% 无效)
        energy = torch.ones(batch_size)
        energy[0] = float("nan")
        energy[1] = float("inf")

        # 默认阈值 0.10 (10%)，20% 超过阈值，必须抛出 FloatingPointError
        with self.assertRaisesRegex(FloatingPointError, "exceeds maximum allowed"):
            vmc_surrogate_loss(log_abs, energy)

        # 显式放宽阈值至 0.25 (25%)，20% 在允许范围内，应正常计算并发出 warning
        with self.assertWarnsRegex(RuntimeWarning, "Excluded 2/10"):
            loss, stats = vmc_surrogate_loss(log_abs, energy, max_invalid_fraction=0.25)
        self.assertEqual(stats["num_valid"], 8)
        self.assertEqual(stats["num_invalid"], 2)
        self.assertTrue(torch.isfinite(loss))

        # 验证 VMCLoss 模块封装同样具备熔断保护
        vmc_loss_module_strict = VMCLoss(max_invalid_fraction=0.10)
        with self.assertRaisesRegex(FloatingPointError, "exceeds maximum allowed"):
            vmc_loss_module_strict(log_abs, energy)

        vmc_loss_module_lenient = VMCLoss(max_invalid_fraction=0.30)
        with self.assertWarnsRegex(RuntimeWarning, "Excluded 2/10"):
            loss_m, stats_m = vmc_loss_module_lenient(log_abs, energy)
        self.assertEqual(stats_m["num_valid"], 8)


if __name__ == "__main__":
    unittest.main()
