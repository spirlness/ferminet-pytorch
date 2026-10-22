"""Statistical and reference-selection oracles independent of training."""

import unittest
from unittest.mock import patch

import torch

from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.hamiltonian import MolecularSystem
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.vmc_statistics import (
    autocorrelation_time,
    energy_statistics,
    apply_reference,
)


class TestVMCStatistics(unittest.TestCase):
    def test_white_noise_constant_and_short_series(self):
        torch.manual_seed(11)
        noise = torch.randn(4096, 16, dtype=torch.float64)
        _, tau, _ = autocorrelation_time(noise)
        self.assertAlmostEqual(tau, 1.0, delta=0.15)
        constant = energy_statistics(torch.ones(512, 16))
        self.assertEqual(constant["energy_sem"], 0.0)
        self.assertEqual(constant["split_rhat"], 1.0)
        self.assertEqual(constant["diagnostics_status"], "passed")
        self.assertEqual(
            energy_statistics(noise[:8])["diagnostics_status"], "insufficient_evidence"
        )
        with self.assertRaises(FloatingPointError):
            autocorrelation_time(torch.tensor([0.0, float("nan")]))

    def test_ar1_and_slow_mixture(self):
        torch.manual_seed(12)

        def ar(rho):
            e = torch.randn(16384, 32, dtype=torch.float64) * (1 - rho * rho) ** 0.5
            x = torch.empty_like(e)
            x[0] = torch.randn(32, dtype=x.dtype)
            for t in range(1, len(x)):
                x[t] = rho * x[t - 1] + e[t]
            return x

        slow = ar(0.98)
        self.assertAlmostEqual(autocorrelation_time(ar(0.8))[1], 9.0, delta=1.5)
        mixed = 0.7 * ar(0.1) + 0.3 * slow
        rho, tau, _ = autocorrelation_time(mixed)
        lag1_approx = (1 + rho) / (1 - rho)
        self.assertGreater(tau, 2 * lag1_approx)

    def test_chain_offsets_are_detected(self):
        torch.manual_seed(13)
        x = torch.randn(1024, 16, dtype=torch.float64) + torch.arange(16)
        result = energy_statistics(x)
        self.assertGreater(result["split_rhat"], 1.01)
        self.assertEqual(result["diagnostics_status"], "insufficient_evidence")

    def test_confidence_interval_controls_accuracy_claim(self):
        for status, sem, expected in [
            ("passed", 0.0001, True),
            ("passed", 0.002, False),
            ("insufficient_evidence", 0.0001, False),
        ]:
            result = {
                "energy_mean": -1.0,
                "energy_sem": sem,
                "diagnostics_status": status,
            }
            apply_reference(result, -1.0, label="manufactured", source="analytic")
            self.assertEqual(result["chemical_accuracy_met"], expected)

    def test_reference_requires_geometry_electrons_and_spin(self):
        for bond, n_up, n_down, spin, expected in [
            (1.4, 1, 1, True, True),
            (3.0, 1, 1, True, False),
            (1.4, 1, 0, False, False),
            (1.4, 2, 0, False, False),
        ]:
            coords = torch.tensor(
                [[0.0, 0.0, -bond / 2], [0.0, 0.0, bond / 2]], dtype=torch.float64
            )
            system = MolecularSystem(
                coords, torch.ones(2, dtype=torch.float64), n_up, n_down
            )
            net = FermiNet(
                coords,
                system.nuclear_charges,
                n_up,
                n_down,
                hidden_dim_one=8,
                hidden_dim_two=4,
                num_layers=1,
                num_determinants=2,
                spin_adapted=spin,
            ).double()
            with patch(
                "ferminet_pytorch.evaluation.compute_local_energy",
                return_value=(torch.ones(4), None, None),
            ):
                result = evaluate_frozen_wavefunction(
                    net,
                    system,
                    num_walkers=4,
                    num_steps=2,
                    burn_in_steps=0,
                    decorrelation_steps=1,
                    dtype=torch.float64,
                    verbose=False,
                )
            self.assertEqual("exact_energy" in result, expected)

    def test_frozen_nonfinite_energy_is_never_discarded(self):
        coords = torch.zeros(1, 3)
        system = MolecularSystem(coords, torch.ones(1), 1, 0)
        net = FermiNet(
            coords,
            system.nuclear_charges,
            1,
            0,
            hidden_dim_one=8,
            hidden_dim_two=4,
            num_layers=1,
            num_determinants=2,
        )
        with patch(
            "ferminet_pytorch.evaluation.compute_local_energy",
            return_value=(torch.tensor([0.0, float("nan")]), None, None),
        ):
            with self.assertRaises(FloatingPointError):
                evaluate_frozen_wavefunction(
                    net,
                    system,
                    num_walkers=2,
                    num_steps=1,
                    burn_in_steps=0,
                    verbose=False,
                )
        self.assertTrue(net.training)


if __name__ == "__main__":
    unittest.main()
